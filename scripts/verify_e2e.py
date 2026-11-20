"""Run the 5 official Apollo-11 multimodal tests (text, image, audio, video, video_audio)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_wemm_video.generate import load_wemm_model, stream_generate

MEDIA_DIR = Path(__file__).resolve().parent.parent / "examples" / "media"


def run_case(model, processor, title: str, prompt: str, media_items: list, enable_thinking: bool = False):
    print(f"\n{'=' * 60}\n[{title}] thinking={enable_thinking}\n{'=' * 60}", flush=True)
    messages = [{"role": "user", "content": [*media_items, {"type": "text", "text": prompt}]}]
    out = []
    last = {}
    for chunk in stream_generate(
        model,
        processor,
        messages,
        max_tokens=256,
        temperature=0.0,
        enable_thinking=enable_thinking,
    ):
        print(chunk["text"], end="", flush=True)
        out.append(chunk["text"])
        last = chunk
    print()
    if last:
        print(
            f"--> prompt={last['prompt_tokens']} tok | gen={last['generated_tokens']} tok | "
            f"encode={last['encode_sec']:.2f}s | prefill={last['prefill_sec']:.2f}s | "
            f"decode={last['decode_tps']:.1f} tok/s",
            flush=True,
        )
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="Verify WeMM-Video MLX on the 5 Apollo-11 test cases.")
    parser.add_argument("--model", required=True, help="Path to converted MLX checkpoint")
    parser.add_argument("--thinking", action="store_true", help="Also run Thinking mode on video_audio")
    args = parser.parse_args()

    model, processor = load_wemm_model(args.model)
    run_case(
        model,
        processor,
        "1. Text",
        "A meeting starts at 09:30 and lasts 35 minutes. What time does it end?",
        [],
    )
    run_case(
        model,
        processor,
        "2. Image",
        "Which planet is shown? Describe what you see.",
        [{"type": "image", "image": str(MEDIA_DIR / "apollo-11-earth.jpg")}],
    )
    run_case(
        model,
        processor,
        "3. Audio",
        "Transcribe the English speech in this recording.",
        [{"type": "audio", "audio": str(MEDIA_DIR / "apollo-11-small-step.wav")}],
    )
    clip = {
        "video": str(MEDIA_DIR / "apollo-11-first-steps.mp4"),
        "fps": 1.0,
        "min_frames": 4,
        "max_frames": 16,
        "total_pixels": 262144,
    }
    run_case(
        model,
        processor,
        "4. Video (vision only)",
        "Describe what happens on screen.",
        [{"type": "video", **clip}],
    )
    run_case(
        model,
        processor,
        "5. Video + Audio (omni)",
        "Describe the scene and summarize the speech.",
        [{"type": "video_audio", **clip}],
    )
    if args.thinking:
        run_case(
            model,
            processor,
            "6. Video + Audio (Thinking)",
            "Describe the scene and summarize the speech.",
            [{"type": "video_audio", **clip}],
            enable_thinking=True,
        )


if __name__ == "__main__":
    main()
