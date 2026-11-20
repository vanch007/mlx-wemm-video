from __future__ import annotations

import argparse
import time
from pathlib import Path
from mlx_wemm_video.generate import load_wemm_model, stream_generate
from mlx_wemm_video.media import detect_video_shots


def _fmt_ts(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:04.1f}"


def build_storyboard_prompt(duration: float, shots: list[tuple[float, float]] | None) -> str:
    dur_str = _fmt_ts(duration)
    anchor_block = ""
    if shots:
        lines = [
            f"- Shot {i + 1:02d}: {_fmt_ts(s)} - {_fmt_ts(e)} ({e - s:.1f}s)"
            for i, (s, e) in enumerate(shots)
        ]
        anchor_block = (
            f"\n【精确时间轴锚点（由视频帧切镜检测自动提取，总时长 {dur_str}，共 {len(shots)} 个真实镜头）】\n"
            + "\n".join(lines)
            + "\n请严格沿用以上每一个镜头的编号与精确起止时间填写分镜表，禁止自行均分或压缩时间轴，确保最后一镜精确对齐片尾 "
            + dur_str
            + "。\n"
        )
    else:
        anchor_block = (
            f"\n本视频总时长精确为 {dur_str}（{duration:.1f} 秒）。请严格对齐视频中给出的时间标记，禁止均分或压缩时间轴。\n"
        )

    return (
        f"你是一位资深的电影导演兼剪辑师。请对这段总时长 {dur_str} 的电影片段进行专业级“电影拉片解读”，"
        "结合完整画面与原始对白/音效，按时间顺序覆盖从 00:00.0 到片尾全部内容，语言精炼专业，不要中途省略。"
        + anchor_block
        + "\n请严格按以下三个部分完整输出：\n\n"
        "## 一、全片剧情与戏剧结构概览\n"
        "概括本场戏的场景氛围、核心人物关系、戏剧冲突升级节点（起、承、转、合，标明真实时间段）以及视听语言特点。\n\n"
        "## 二、逐镜头拉片与 Storyboard（分镜反推表）\n"
        "严格对应每个镜头，使用 Markdown 表格输出完整 Storyboard，包含以下列（每格表述精准凝练）：\n"
        "| 镜号 | 起止时间 | 景别与机位视角 | 运镜方式 | 画面构图与光影调度 | 人物动作与微表情 | 原始台词（含说话人）与声音设计 | 导演意图/戏剧功能 |\n\n"
        "## 三、剪辑节奏与视听技法复盘\n"
        "总结本场戏在正反打轴线调度、台词节奏停顿、特写与反打的情绪压迫感、以及声音设计上的亮点（完整收尾）。"
    )


def main():
    parser = argparse.ArgumentParser(description="Film breakdown and storyboard reverse-engineering via WeMM-Video MLX.")
    parser.add_argument("--model", required=True, help="Path to local MLX WeMM-Video checkpoint")
    parser.add_argument("--video", required=True, help="Path to local video file")
    parser.add_argument("--output", default="outputs/storyboard.md", help="Output Markdown path")
    parser.add_argument("--fps", type=float, default=1.5)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--total-pixels", type=int, default=4194304)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-shot-sec", type=float, default=1.2, help="Minimum shot duration for cut detection")
    parser.add_argument("--no-shot-anchors", action="store_true", help="Disable automatic visual shot-cut timecode anchors")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    shots, duration = detect_video_shots(args.video, min_shot_sec=args.min_shot_sec)
    print(f"Detected {len(shots)} shots across {duration:.2f}s (anchors={not args.no_shot_anchors})", flush=True)
    prompt = build_storyboard_prompt(duration, None if args.no_shot_anchors else shots)

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
            {"type": "text", "text": prompt},
        ],
    }]

    chunks = []
    last = {}
    next_report = 250
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
        if ch["generated_tokens"] >= next_report and ch["finish_reason"] is None:
            next_report += 250
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
        f"finish_reason={last.get('finish_reason')} | "
        f"encode={last.get('encode_sec', 0):.2f}s | prefill={last.get('prefill_sec', 0):.2f}s | "
        f"decode={last.get('decode_tps', 0):.1f} tok/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
