# mlx-wemm-video

Native Apple Silicon **MLX** port of [Tencent/WeMM-Video](https://github.com/Tencent/WeMM-Video) (starting with [tencent/WeMM-Video-35BA3B](https://huggingface.co/tencent/WeMM-Video-35BA3B)), built for omni-modal video + audio + image + text understanding on Apple Silicon Macs with zero-PyTorch runtime compatibility for **oMLX** / `mlx-vlm`.

## Pre-converted MLX Models on Hugging Face

| Variant | Quantization Policy | Size | Hugging Face Repo |
| --- | --- | --- | --- |
| **4-bit Mixed** | LLM & 256 MoE experts 4-bit (gs=64), MoE gates 8-bit, Vision BF16, Audio FP16, A_log FP32 | ~23.3 GB | [vanch007/WeMM-Video-35BA3B-mlx-4bit](https://huggingface.co/vanch007/WeMM-Video-35BA3B-mlx-4bit) |
| **6-bit Mixed** | LLM & 256 MoE experts 6-bit (gs=64), MoE gates 8-bit, Vision BF16, Audio FP16, A_log FP32 | ~31.4 GB | [vanch007/WeMM-Video-35BA3B-mlx-6bit](https://huggingface.co/vanch007/WeMM-Video-35BA3B-mlx-6bit) |
| **8-bit Mixed** | LLM & 256 MoE experts 8-bit (gs=64), MoE gates 8-bit, Vision BF16, Audio FP16, A_log FP32 | ~39.5 GB | [vanch007/WeMM-Video-35BA3B-mlx-8bit](https://huggingface.co/vanch007/WeMM-Video-35BA3B-mlx-8bit) |

## Architecture Highlights

- **Omni-modal MLX Model (`mlx_wemm_video/model.py`)**: Combines `Qwen3.5-MoE` (40 layers: 30 Metal-accelerated Gated DeltaNet + 10 GQA Full Attention with output gate; 256-choose-8 SwitchGLU routed experts + 1 shared expert), 27-layer `Qwen3-VL` 3D ViT, and native MLX `WeMMAudioTower`.
- **Native MLX Audio Tower (`mlx_wemm_video/audio.py`)**: 32-layer Whisper encoder (`d_model=1280`) -> 0.1s temporal windowing -> 2-layer BERT Q-Former (`hidden_size=4104`, `eps=1e-12`) -> `audio_proj` (`4104 -> 2048`). Verified against PyTorch FP32 reference (`cos=1.000000` in FP32; `cos=0.999826` in FP16 at 350 ms per 30s audio segment on M3 Max).
- **WeMM 3D M-RoPE (`mlx_wemm_video/model.py`)**: Expands `video_grid_thw` per temporal patch group so `<|video_pad|>` tokens receive 2D spatial coordinates while adjacent interleaved `<|audio_pad|>` tokens receive 1D sequential positions.
- **Torch-free macOS Media Pipeline (`mlx_wemm_video/media.py`, `processing_wemm.py`)**: Replaces Linux/CUDA-only `torchcodec` with native `ffmpeg`/`ffprobe`, OpenCV, SoundFile, and SciPy while matching `Qwen3VLVideoProcessor` pixel-for-pixel (`max abs diff 1.19e-7`).
- **Streaming Sharded Converter (`mlx_wemm_video/convert.py`)**: Layer-by-layer streaming conversion from 17-shard HF safetensors with bounded RAM.
- **Non-invasive oMLX Integration (`omlx_compat/mlx_vlm_wemm_compat.py`)**: Registers `wemm_qwen3_5_moe` onto `mlx_vlm.models`, `prompt_utils`, `AutoProcessor`, and `omlx.utils.video.NATIVE_VIDEO_MODEL_TYPES` via a removable `.pth` hook.

## M3 Max (128 GB) Benchmarks (`WeMM-Video-35BA3B-mlx-4bit`)

- **Model load time**: 1.20s - 2.78s
- **95s Chinese Movie Breakdown (96 frames + 4 audio segments, 3,413 prompt tokens)**: encode 2.97s, prefill 3.67s, decode **58.1 tok/s** avg (peak **75.0 tok/s**) over 3,106 tokens.
- **Apollo-11 5-modality verification (text, image, audio, video, video+audio, Thinking mode)**: all passing.

## Usage

### 1. Convert from Hugging Face

```bash
python -m mlx_wemm_video.convert \
  --hf-path /path/to/WeMM-Video-35BA3B \
  --mlx-path /path/to/WeMM-Video-35BA3B-mlx-4bit \
  --dtype bfloat16 --quantize --q-bits 4 --q-group-size 64 --gate-bits 8
```

Use `--q-bits 6` or `--q-bits 8` for 6-bit / 8-bit mixed precision.

### 2. Omni-modal Video Inference & Storyboard Extraction

```bash
python -m mlx_wemm_video.generate \
  --model /path/to/WeMM-Video-35BA3B-mlx-4bit \
  --video examples/media/apollo-11-first-steps.mp4 \
  --prompt "Describe the scene and summarize the speech."

python scripts/run_storyboard.py \
  --model /path/to/WeMM-Video-35BA3B-mlx-4bit \
  --video /path/to/movie_clip.mp4 \
  --output outputs/storyboard.md
```

### 3. Register with Local oMLX

```bash
/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 scripts/install_omlx_compat.py
```

## License

Source code is licensed under Apache License 2.0 (see [LICENSE](LICENSE)). Original model weights and architecture belong to Tencent WeChat Vision ([Tencent/WeMM-Video](https://github.com/Tencent/WeMM-Video)).
