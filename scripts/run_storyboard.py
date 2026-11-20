from __future__ import annotations

import argparse
import time
from pathlib import Path
from mlx_wemm_video.generate import load_wemm_model, stream_generate

DEFAULT_PROMPT = """你是一位资深的电影导演兼剪辑师。请对这段约95秒的电影片段进行专业级“电影拉片解读”，结合完整画面与原始对白/音效，按时间顺序覆盖从00:00到片尾（约01:35）的全部内容，不要中途省略。

请严格按以下三个部分输出：

## 一、全片剧情与戏剧结构概览
概括本场戏的场景氛围、核心人物关系、戏剧冲突升级节点（起、承、转、合）以及视听语言特点（光线、色调、雨夜音效、正反打节奏）。

## 二、逐镜头拉片与 Storyboard（分镜反推表）
按时间轴切分出每一个独立镜头（Shot 01, Shot 02... 直到片尾），使用 Markdown 表格输出完整 Storyboard，包含以下列：
| 镜号 | 起止时间 | 景别与机位视角 | 运镜方式 | 画面构图与光影调度 | 人物动作与微表情 | 原始台词（含说话人）与声音设计 | 导演意图/戏剧功能 |

## 三、剪辑节奏与视听技法复盘
总结本场戏在正反打轴线调度、台词节奏停顿、特写与反打的情绪压迫感、以及声音设计上的亮点。"""


def main():
    parser = argparse.ArgumentParser(description="Film breakdown and storyboard reverse-engineering via WeMM-Video MLX.")
    parser.add_argument("--model", required=True, help="Path to local MLX WeMM-Video checkpoint")
    parser.add_argument("--video", required=True, help="Path to local video file")
    parser.add_argument("--output", default="outputs/storyboard.md", help="Output Markdown path")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--total-pixels", type=int, default=4194304)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, processor = load_wemm_model(args.model)
    print(f"Model loaded in {time.time() - t0:.2f}s", flush=True)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video_audio",
                "video": args.video,
                "fps": args.fps,
                "min_frames": 4,
                "max_frames": args.max_frames,
                "total_pixels": args.total_pixels,
            },
            {"type": "text", "text": DEFAULT_PROMPT},
        ],
    }]

    chunks = []
    last = {}
    for ch in stream_generate(
        model,
        processor,
        messages,
        max_tokens=args.max_tokens,
        temperature=0.0,
        enable_thinking=args.thinking,
        use_audio_in_video=True,
        prefill_step_size=2048,
    ):
        chunks.append(ch["text"])
        last = ch
        if ch["generated_tokens"] % 250 == 0:
            print(
                f"[progress] prompt={ch['prompt_tokens']} | gen={ch['generated_tokens']} tok | "
                f"decode={ch['decode_tps']:.1f} tok/s",
                flush=True,
            )

    full_text = "".join(chunks)
    out_file.write_text(full_text, encoding="utf-8")
    print(
        f"DONE -> saved {out_file} ({len(full_text)} chars)\n"
        f"STATS: prompt={last.get('prompt_tokens')} tok | gen={last.get('generated_tokens')} tok | "
        f"encode={last.get('encode_sec', 0):.2f}s | prefill={last.get('prefill_sec', 0):.2f}s | "
        f"decode={last.get('decode_tps', 0):.1f} tok/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
