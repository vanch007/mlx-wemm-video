"""Convert Tencent/WeMM-Video Hugging Face checkpoints to MLX & oMLX-compatible format."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

from .config import ModelConfig
from .model import Model, NORM_SHIFT_SUFFIXES, sanitize_wemm_key


STATIC_ASSETS_TO_COPY = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "LICENSE",
)


def _parse_dtype(dtype_name: str) -> mx.Dtype:
    mapping = {
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
        "float32": mx.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def _load_shard_subset(hf_dir: Path, shard_files: List[str], keys_needed: set[str]) -> Dict[str, mx.array]:
    loaded: Dict[str, mx.array] = {}
    for fname in shard_files:
        fpath = hf_dir / fname
        shard = mx.load(str(fpath))
        for k in list(shard.keys()):
            if k in keys_needed:
                loaded[k] = shard[k]
        del shard
    return loaded


def _sanitize_layer_experts(
    layer_idx: int,
    num_experts: int,
    layer_weights: Dict[str, mx.array],
) -> Dict[str, mx.array]:
    prefix = f"model.language_model.layers.{layer_idx}.mlp"
    out: Dict[str, mx.array] = {}

    if f"{prefix}.experts.0.gate_proj.weight" in layer_weights:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stacked = mx.stack(
                [layer_weights.pop(f"{prefix}.experts.{e}.{proj}.weight") for e in range(num_experts)],
                axis=0,
            )
            out[f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.{proj}.weight"] = stacked
    elif f"{prefix}.experts.gate_up_proj" in layer_weights:
        gate_up = layer_weights.pop(f"{prefix}.experts.gate_up_proj")
        gate_w, up_w = mx.split(gate_up, 2, axis=-2)
        out[f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.gate_proj.weight"] = gate_w
        out[f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.up_proj.weight"] = up_w
        out[f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.down_proj.weight"] = layer_weights.pop(
            f"{prefix}.experts.down_proj"
        )

    for orig_k, val in layer_weights.items():
        k = sanitize_wemm_key(orig_k)
        if k.endswith("linear_attn.conv1d.weight") and val.ndim == 3 and val.shape[-1] != 1:
            val = mx.moveaxis(val, 2, 1)
        if orig_k.startswith("model.language_model.") and any(k.endswith(sfx) for sfx in NORM_SHIFT_SUFFIXES) and val.ndim == 1:
            val = val + 1.0
        out[k] = val
    return out


def _sanitize_non_layer_weights(raw_weights: Dict[str, mx.array], in_channels: int = 3) -> Dict[str, mx.array]:
    out: Dict[str, mx.array] = {}
    for orig_k, val in raw_weights.items():
        if "mtp." in orig_k or orig_k.endswith("embeddings.position_ids"):
            continue
        k = sanitize_wemm_key(orig_k)
        if k in ("audio_tower.conv1.weight", "audio_tower.conv2.weight") and val.ndim == 3:
            if val.shape[-1] == 3 and val.shape[1] != 3:
                val = mx.moveaxis(val, 2, 1)
        elif k.endswith("patch_embed.proj.weight") and val.ndim == 5:
            if val.shape[-1] != in_channels and val.shape[1] == in_channels:
                val = val.transpose(0, 2, 3, 4, 1)
        elif orig_k.startswith("model.language_model.") and any(k.endswith(sfx) for sfx in NORM_SHIFT_SUFFIXES) and val.ndim == 1:
            val = val + 1.0
        out[k] = val
    return out


def _quantize_module_weights(
    key_prefix: str,
    weight: mx.array,
    *,
    bits: int,
    group_size: int,
    mode: str = "affine",
) -> Dict[str, mx.array]:
    q_w, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)
    return {
        f"{key_prefix}.weight": q_w,
        f"{key_prefix}.scales": scales,
        f"{key_prefix}.biases": biases,
    }


def _should_quantize_key(
    key: str,
    val: mx.array,
    *,
    q_bits: int,
    q_group_size: int,
    gate_bits: int = 8,
    quantize_embeddings: bool = False,
) -> Optional[Dict[str, Any]]:
    if not key.endswith(".weight"):
        return None
    if val.ndim not in (2, 3):
        return None
    if key.startswith("vision_tower.") or key.startswith("audio_tower."):
        return None
    if key.endswith("embed_tokens.weight") and not quantize_embeddings:
        return None
    if "conv1d" in key or "layernorm" in key or "norm" in key or "A_log" in key or "dt_bias" in key:
        return None
    if key.endswith("mlp.gate.weight") or key.endswith("shared_expert_gate.weight"):
        if val.shape[-1] % 64 == 0:
            return {"bits": gate_bits, "group_size": 64, "mode": "affine"}
        return None
    if val.shape[-1] % q_group_size != 0:
        return None
    return {"bits": q_bits, "group_size": q_group_size, "mode": "affine"}


class ShardWriter:
    """Writes MLX safetensors shards bounded by max_shard_bytes (~4.5 GB) and builds index."""

    def __init__(self, out_dir: Path, max_shard_bytes: int = int(4.5 * 1024**3)):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = max_shard_bytes
        self._buffer: Dict[str, mx.array] = {}
        self._buffer_bytes = 0
        self._shard_files: List[Tuple[str, List[str]]] = []
        self.total_size = 0

    def add_dict(self, tensors: Dict[str, mx.array]) -> None:
        for k, v in tensors.items():
            mx.eval(v)
            nbytes = int(v.nbytes)
            if self._buffer and (self._buffer_bytes + nbytes > self.max_shard_bytes):
                self.flush()
            self._buffer[k] = v
            self._buffer_bytes += nbytes
            self.total_size += nbytes

    def flush(self) -> None:
        if not self._buffer:
            return
        idx = len(self._shard_files) + 1
        tmp_name = f"_tmp_shard_{idx:05d}.safetensors"
        mx.save_safetensors(str(self.out_dir / tmp_name), self._buffer, metadata={"format": "mlx"})
        self._shard_files.append((tmp_name, list(self._buffer.keys())))
        self._buffer.clear()
        self._buffer_bytes = 0
        gc.collect()
        mx.clear_cache()

    def finalize(self) -> Dict[str, Any]:
        self.flush()
        total_shards = len(self._shard_files)
        weight_map: Dict[str, str] = {}
        for i, (tmp_name, keys) in enumerate(self._shard_files, start=1):
            final_name = (
                "model.safetensors"
                if total_shards == 1
                else f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            )
            (self.out_dir / tmp_name).rename(self.out_dir / final_name)
            for k in keys:
                weight_map[k] = final_name
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": weight_map,
        }
        if total_shards > 1:
            (self.out_dir / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return index


def convert_checkpoint(
    hf_path: str | Path,
    mlx_path: str | Path,
    *,
    dtype: str = "bfloat16",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    gate_bits: int = 8,
    quantize_embeddings: bool = False,
    omlx_model_type: str = "wemm_qwen3_5_moe",
    drop_audio: bool = False,
) -> Path:
    """Streaming layer-by-layer conversion of WeMM-Video-35BA3B to MLX & oMLX format."""
    t0 = time.time()
    hf_dir = Path(hf_path)
    out_dir = Path(mlx_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    target_dtype = _parse_dtype(dtype)

    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map: Dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        single = hf_dir / "model.safetensors"
        with safe_open(str(single), framework="numpy") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    num_layers = int(raw_cfg["text_config"]["num_hidden_layers"])
    num_experts = int(raw_cfg["text_config"].get("num_experts", 0) or 0)
    in_channels = int(raw_cfg.get("vision_config", {}).get("in_channels", 3))

    # Group keys by decoder layer vs non-layer (vision, audio, embeddings, norm, lm_head)
    layer_keys: Dict[int, set[str]] = defaultdict(set)
    layer_shards: Dict[int, set[str]] = defaultdict(set)
    other_keys: set[str] = set()
    other_shards: set[str] = set()

    for k, shard in weight_map.items():
        if k.startswith("mtp.") or k.endswith("embeddings.position_ids"):
            continue
        if drop_audio and k.startswith("model.audio."):
            continue
        if k.startswith("model.language_model.layers."):
            parts = k.split(".")
            l_idx = int(parts[3])
            layer_keys[l_idx].add(k)
            layer_shards[l_idx].add(shard)
        else:
            other_keys.add(k)
            other_shards.add(shard)

    writer = ShardWriter(out_dir)
    quantization_overrides: Dict[str, Any] = {}
    if quantize:
        quantization_overrides["group_size"] = q_group_size
        quantization_overrides["bits"] = q_bits
        quantization_overrides["mode"] = "affine"

    def _cast_and_maybe_quantize(sanitized: Dict[str, mx.array]) -> Dict[str, mx.array]:
        processed: Dict[str, mx.array] = {}
        for k, v in sanitized.items():
            # Keep A_log in float32 for Gated DeltaNet numerical stability
            if k.endswith("A_log"):
                v = v.astype(mx.float32)
            elif k.startswith("audio_tower.") and v.dtype in (mx.float32, mx.float16, mx.bfloat16):
                # Verified on M3 Max: audio_tower in fp16 is faster (350ms vs 397ms) and matches fp32 (cos=0.9998)
                v = v.astype(mx.float16)
            elif v.dtype in (mx.float32, mx.float16, mx.bfloat16):
                v = v.astype(target_dtype)

            if quantize:
                q_spec = _should_quantize_key(
                    k,
                    v,
                    q_bits=q_bits,
                    q_group_size=q_group_size,
                    gate_bits=gate_bits,
                    quantize_embeddings=quantize_embeddings,
                )
                if q_spec is not None:
                    mod_path = k[: -len(".weight")]
                    q_dict = _quantize_module_weights(
                        mod_path,
                        v,
                        bits=q_spec["bits"],
                        group_size=q_spec["group_size"],
                        mode=q_spec["mode"],
                    )
                    processed.update(q_dict)
                    if q_spec["bits"] != q_bits or q_spec["group_size"] != q_group_size:
                        quantization_overrides[mod_path] = {
                            "bits": q_spec["bits"],
                            "group_size": q_spec["group_size"],
                            "mode": q_spec["mode"],
                        }
                    continue
                elif k.endswith(".weight") and v.ndim in (2, 3):
                    # Explicitly mark unquantized linear/embedding layers for mlx-vlm/oMLX loader
                    mod_path = k[: -len(".weight")]
                    if (
                        mod_path.startswith("vision_tower.")
                        or mod_path.startswith("audio_tower.")
                        or mod_path.endswith("embed_tokens")
                        or mod_path.endswith("lm_head")
                    ):
                        quantization_overrides[mod_path] = False
            processed[k] = v
        return processed

    print(f"[1/{num_layers + 1}] Processing non-layer weights (vision/audio/embed/lm_head) from {len(other_shards)} shards...", flush=True)
    raw_other = _load_shard_subset(hf_dir, sorted(other_shards), other_keys)
    san_other = _sanitize_non_layer_weights(raw_other, in_channels=in_channels)
    del raw_other
    writer.add_dict(_cast_and_maybe_quantize(san_other))
    del san_other
    gc.collect()
    mx.clear_cache()

    for l_idx in range(num_layers):
        t_layer = time.time()
        raw_layer = _load_shard_subset(hf_dir, sorted(layer_shards[l_idx]), layer_keys[l_idx])
        san_layer = _sanitize_layer_experts(l_idx, num_experts, raw_layer)
        del raw_layer
        proc_layer = _cast_and_maybe_quantize(san_layer)
        del san_layer
        writer.add_dict(proc_layer)
        del proc_layer
        gc.collect()
        mx.clear_cache()
        print(
            f"[{l_idx + 2}/{num_layers + 1}] Layer {l_idx:02d} done in {time.time() - t_layer:.1f}s "
            f"(written {writer.total_size / 1e9:.2f} GB)",
            flush=True,
        )

    index = writer.finalize()

    # Build clean MLX & oMLX config.json
    out_cfg = json.loads(json.dumps(raw_cfg))
    out_cfg.pop("auto_map", None)
    out_cfg["model_type"] = "qwen3_5_moe" if drop_audio else omlx_model_type
    if drop_audio:
        out_cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        out_cfg.pop("audio_config", None)
        out_cfg.pop("audio_token_id", None)
    out_cfg["dtype"] = dtype
    out_cfg["text_config"]["dtype"] = dtype
    if "vision_config" in out_cfg:
        out_cfg["vision_config"]["dtype"] = dtype
    if "audio_config" in out_cfg:
        out_cfg["audio_config"]["dtype"] = dtype
    if "rope_parameters" in out_cfg["text_config"]:
        rp = out_cfg["text_config"]["rope_parameters"]
        if "type" not in rp and "rope_type" in rp:
            rp["type"] = rp["rope_type"]

    if quantize:
        out_cfg["quantization"] = quantization_overrides
        out_cfg["quantization_config"] = quantization_overrides

    (out_dir / "config.json").write_text(
        json.dumps(out_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for fname in STATIC_ASSETS_TO_COPY:
        src = hf_dir / fname
        if src.is_file():
            dst = out_dir / fname
            if fname == "processor_config.json":
                # Remove remote-code auto_map so oMLX loads torch-free processor cleanly
                pcfg = json.loads(src.read_text(encoding="utf-8"))
                pcfg.pop("auto_map", None)
                if drop_audio:
                    pcfg["processor_class"] = "Qwen3VLProcessor"
                    pcfg.pop("audio_config", None)
                dst.write_text(json.dumps(pcfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            else:
                shutil.copy2(src, dst)

    elapsed = time.time() - t0
    print(
        f"Conversion complete -> {out_dir}\n"
        f"  Total tensors: {len(index['weight_map'])}\n"
        f"  Total size:    {writer.total_size / 1e9:.2f} GB\n"
        f"  Elapsed:       {elapsed:.1f}s",
        flush=True,
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Convert WeMM-Video HF weights to MLX / oMLX format.")
    parser.add_argument("--hf-path", required=True, help="Local HF checkpoint directory")
    parser.add_argument("--mlx-path", required=True, help="Output MLX checkpoint directory")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--quantize", action="store_true", help="Enable mixed-precision MLX quantization")
    parser.add_argument("--q-bits", type=int, default=4, choices=[4, 6, 8], help="Bits for LLM/MoE weights")
    parser.add_argument("--q-group-size", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--gate-bits", type=int, default=8, help="Bits for MoE router & shared_expert_gate")
    parser.add_argument("--quantize-embeddings", action="store_true")
    parser.add_argument(
        "--drop-audio",
        action="store_true",
        help="Export vision+video+text only as standard qwen3_5_moe (zero-patch oMLX fallback)",
    )
    args = parser.parse_args()
    convert_checkpoint(
        args.hf_path,
        args.mlx_path,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        gate_bits=args.gate_bits,
        quantize_embeddings=args.quantize_embeddings,
        drop_audio=args.drop_audio,
    )


if __name__ == "__main__":
    main()
