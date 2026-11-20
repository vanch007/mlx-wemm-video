"""Standalone MLX inference & streaming CLI for WeMM-Video (text/image/audio/video/video_audio)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .model import Model
from .processing_wemm import WeMMProcessor


def load_wemm_model(model_path: str | Path) -> Tuple[Model, WeMMProcessor]:
    """Load an MLX-converted WeMM-Video checkpoint (BF16 or mixed-precision quantized)."""
    model_dir = Path(model_path)
    cfg_dict = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(cfg_dict)
    model = Model(config)

    quant_cfg = cfg_dict.get("quantization") or cfg_dict.get("quantization_config")
    if isinstance(quant_cfg, dict) and quant_cfg:
        default_bits = int(quant_cfg.get("bits", 4))
        default_group = int(quant_cfg.get("group_size", 64))
        default_mode = quant_cfg.get("mode", "affine")

        def _class_predicate(path: str, module: nn.Module):
            if not hasattr(module, "to_quantized"):
                return False
            if path in quant_cfg:
                entry = quant_cfg[path]
                if entry is False:
                    return False
                if isinstance(entry, dict):
                    return {
                        "group_size": int(entry.get("group_size", default_group)),
                        "bits": int(entry.get("bits", default_bits)),
                        "mode": entry.get("mode", default_mode),
                    }
            if path.startswith("vision_tower") or path.startswith("audio_tower"):
                return False
            return True

        nn.quantize(
            model,
            group_size=default_group,
            bits=default_bits,
            mode=default_mode,
            class_predicate=_class_predicate,
        )

    index_file = model_dir / "model.safetensors.index.json"
    weights: Dict[str, mx.array] = {}
    if index_file.is_file():
        weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
        for shard in sorted(set(weight_map.values())):
            weights.update(mx.load(str(model_dir / shard)))
    else:
        weights = mx.load(str(model_dir / "model.safetensors"))

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()

    processor = WeMMProcessor.from_pretrained(model_dir)
    return model, processor


def _sample_next_token(logits: mx.array, temperature: float = 0.0, top_p: float = 0.95) -> mx.array:
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1)
    scaled = logits / temperature
    probs = mx.softmax(scaled, axis=-1)
    if 0.0 < top_p < 1.0:
        sorted_indices = mx.argsort(probs, axis=-1)[..., ::-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cum_probs = mx.cumsum(sorted_probs, axis=-1)
        mask = (cum_probs - sorted_probs) > top_p
        sorted_probs = mx.where(mask, 0.0, sorted_probs)
        sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
        chosen = mx.random.categorical(mx.log(mx.maximum(sorted_probs, 1e-12)))
        return mx.take_along_axis(sorted_indices, chosen[..., None], axis=-1).squeeze(-1)
    return mx.random.categorical(scaled)


def stream_generate(
    model: Model,
    processor: WeMMProcessor,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95,
    enable_thinking: bool = False,
    use_audio_in_video: bool = True,
    prefill_step_size: int = 2048,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    total_pixels: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    t_prep = time.time()
    inputs = processor.prepare_messages(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        use_audio_in_video=use_audio_in_video,
        fps=fps,
        max_frames=max_frames,
        total_pixels=total_pixels,
    )
    input_ids = inputs["input_ids"]
    prompt_tokens = int(input_ids.shape[1])

    emb_out = model.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=inputs.get("pixel_values"),
        pixel_values_videos=inputs.get("pixel_values_videos"),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        audio_feature=inputs.get("audio_feature"),
        attention_mask=inputs.get("attention_mask"),
    )
    inputs_embeds = emb_out.inputs_embeds
    position_ids = emb_out.position_ids
    mx.eval(inputs_embeds, position_ids)
    encode_sec = time.time() - t_prep

    cache = model.language_model.make_cache()
    t_prefill = time.time()
    total_len = int(inputs_embeds.shape[1])

    # Chunked prefill keeps peak Metal activation memory bounded on long videos
    pos = 0
    logits = None
    while pos < total_len:
        end = min(total_len, pos + prefill_step_size)
        chunk_ids = input_ids[:, pos:end]
        chunk_embeds = inputs_embeds[:, pos:end, :]
        chunk_pos = position_ids[:, :, pos:end] if position_ids is not None else None
        out = model.language_model(
            chunk_ids,
            inputs_embeds=chunk_embeds,
            cache=cache,
            position_ids=chunk_pos,
        )
        logits = out.logits[:, -1, :]
        mx.eval(logits, [c.state for c in cache])
        pos = end

    prefill_sec = time.time() - t_prefill
    eos_ids = set()
    if processor.tokenizer.eos_token_id is not None:
        eos_ids.add(int(processor.tokenizer.eos_token_id))
    cfg_eos = getattr(model.config, "eos_token_id", None)
    if isinstance(cfg_eos, int):
        eos_ids.add(cfg_eos)
    elif isinstance(cfg_eos, list):
        eos_ids.update(int(x) for x in cfg_eos)
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(im_end)

    next_tok = _sample_next_token(logits, temperature=temperature, top_p=top_p)
    mx.eval(next_tok)

    generated_ids: List[int] = []
    prev_text = ""
    t_gen = time.time()
    finish_reason = "length"

    for step in range(max_tokens):
        tok_id = int(next_tok.item())
        if tok_id in eos_ids:
            finish_reason = "stop"
            break
        generated_ids.append(tok_id)
        full_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Wait for complete multi-byte UTF-8 characters before flushing delta
        if full_text.endswith("\ufffd"):
            delta = ""
        else:
            delta = full_text[len(prev_text) :]
            prev_text = full_text
        yield {
            "text": delta,
            "token_id": tok_id,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated_ids),
            "encode_sec": encode_sec,
            "prefill_sec": prefill_sec,
            "decode_tps": len(generated_ids) / max(1e-6, time.time() - t_gen),
            "finish_reason": None,
        }
        out = model.language_model(next_tok[:, None], cache=cache)
        next_tok = _sample_next_token(out.logits[:, -1, :], temperature=temperature, top_p=top_p)
        mx.eval(next_tok)

    # Flush any buffered trailing text and report final finish_reason
    final_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    tail_delta = final_text[len(prev_text) :]
    yield {
        "text": tail_delta,
        "token_id": None,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated_ids),
        "encode_sec": encode_sec,
        "prefill_sec": prefill_sec,
        "decode_tps": len(generated_ids) / max(1e-6, time.time() - t_gen),
        "finish_reason": finish_reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Run WeMM-Video MLX inference on Apple Silicon.")
    parser.add_argument("--model", required=True, help="Path to MLX WeMM-Video checkpoint")
    parser.add_argument("--prompt", required=True, help="User question / instruction")
    parser.add_argument("--image", default=None, help="Path to image file")
    parser.add_argument("--audio", default=None, help="Path to audio file")
    parser.add_argument("--video", default=None, help="Path to video file")
    parser.add_argument("--no-video-audio", action="store_true", help="Ignore audio track in video")
    parser.add_argument("--thinking", action="store_true", help="Enable explicit <think> reasoning")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--total-pixels", type=int, default=None)
    args = parser.parse_args()

    content: List[Dict[str, Any]] = []
    if args.image:
        content.append({"type": "image", "image": args.image})
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    if args.video:
        kind = "video" if args.no_video_audio else "video_audio"
        entry: Dict[str, Any] = {"type": kind, "video": args.video}
        if args.fps is not None:
            entry["fps"] = args.fps
        if args.max_frames is not None:
            entry["max_frames"] = args.max_frames
        if args.total_pixels is not None:
            entry["total_pixels"] = args.total_pixels
        content.append(entry)
    content.append({"type": "text", "text": args.prompt})
    messages = [{"role": "user", "content": content}]

    model, processor = load_wemm_model(args.model)
    last_stats: Dict[str, Any] = {}
    for chunk in stream_generate(
        model,
        processor,
        messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=args.thinking,
        use_audio_in_video=not args.no_video_audio,
        fps=args.fps,
        max_frames=args.max_frames,
        total_pixels=args.total_pixels,
    ):
        print(chunk["text"], end="", flush=True)
        last_stats = chunk
    print()
    if last_stats:
        print(
            f"\n[stats] prompt={last_stats['prompt_tokens']} tok | "
            f"gen={last_stats['generated_tokens']} tok | "
            f"encode={last_stats['encode_sec']:.2f}s | "
            f"prefill={last_stats['prefill_sec']:.2f}s | "
            f"decode={last_stats['decode_tps']:.1f} tok/s"
        )


if __name__ == "__main__":
    main()
