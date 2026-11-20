"""Unit tests verifying WeMM-Video MLX architecture, audio tower, M-RoPE, sanitize & media."""

from __future__ import annotations

from pathlib import Path
import mlx.core as mx
import numpy as np

from mlx_wemm_video.audio import WeMMAudioTower
from mlx_wemm_video.config import AudioConfig, ModelConfig, QFormerConfig, TextConfig, VisionConfig
from mlx_wemm_video.media import (
    audio_timestamp_groups,
    decode_audio,
    decode_video,
    preprocess_frames_to_patches,
)
from mlx_wemm_video.model import Model, sanitize_wemm_key


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        model_type="wemm_qwen3_5_moe",
        image_token_id=248056,
        video_token_id=248057,
        audio_token_id=248076,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        text_config=TextConfig(
            model_type="qwen3_5_moe_text",
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=32,
            moe_intermediate_size=32,
            rms_norm_eps=1e-6,
            vocab_size=248320,
            max_position_embeddings=4096,
            full_attention_interval=4,
            rope_parameters={
                "type": "default",
                "mrope_section": [1, 1, 2],
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        ),
        vision_config=VisionConfig(
            model_type="qwen3_5_moe",
            depth=1,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=64,
            deepstack_visual_indexes=[],
        ),
        audio_config=AudioConfig(
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            num_mel_bins=128,
            max_source_positions=1500,
            second_per_window=0.1,
            second_stride=0.1,
            qformer_config=QFormerConfig(
                hidden_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=96,
                encoder_width=32,
                query_length=1,
                out_hidden_size=64,
            ),
        ),
    )


def test_audio_tower_shape():
    cfg = _tiny_config().audio_config
    tower = WeMMAudioTower(cfg)
    mel = mx.zeros((1, 128, 3000), dtype=mx.float32)
    out = tower(mel)
    mx.eval(out)
    assert out.shape == (1, 300, 64), f"Unexpected audio output shape: {out.shape}"


def test_mrope_video_audio_interleave():
    cfg = _tiny_config()
    model = Model(cfg)
    vs, ve, vp, ap = 248053, 248054, 248057, 248076
    tokens = (
        [10, 11]
        + [vs] + [vp] * 4 + [ap] * 3 + [ve]
        + [12, 13]
        + [vs] + [vp] * 4 + [ap] * 3 + [ve]
        + [14]
    )
    input_ids = mx.array([tokens], dtype=mx.int32)
    video_grid_thw = mx.array([[2, 4, 4]], dtype=mx.int32)
    pos_ids, deltas = model.language_model.get_rope_index(
        input_ids, video_grid_thw=video_grid_thw
    )
    mx.eval(pos_ids, deltas)
    assert pos_ids.shape == (3, 1, len(tokens))
    assert pos_ids[0, 0, 3:7].tolist() == [3, 3, 3, 3]
    assert pos_ids[1, 0, 3:7].tolist() == [3, 3, 4, 4]
    assert pos_ids[2, 0, 3:7].tolist() == [3, 4, 3, 4]
    assert pos_ids[0, 0, 7:10].tolist() == [5, 6, 7]
    assert pos_ids[1, 0, 7:10].tolist() == [5, 6, 7]
    assert pos_ids[2, 0, 7:10].tolist() == [5, 6, 7]


def test_sanitize_key_mapping():
    assert (
        sanitize_wemm_key("model.audio.qformer.bert.embeddings.LayerNorm.gamma")
        == "audio_tower.qformer_embeddings_ln.weight"
    )
    assert (
        sanitize_wemm_key(
            "model.audio.qformer.bert.encoder.layer.0.crossattention.output.LayerNorm.beta"
        )
        == "audio_tower.qformer_layers.0.crossattention.LayerNorm.bias"
    )
    assert (
        sanitize_wemm_key("model.language_model.layers.0.mlp.gate.weight")
        == "language_model.model.layers.0.mlp.gate.weight"
    )


def test_apollo11_media_decode():
    media_dir = Path(__file__).resolve().parent.parent / "examples" / "media"
    wav = decode_audio(media_dir / "apollo-11-small-step.wav")
    assert wav.ndim == 1 and len(wav) > 16000
    frames, meta = decode_video(media_dir / "apollo-11-first-steps.mp4", fps=1.0, min_frames=4, max_frames=8)
    assert frames.ndim == 4 and frames.shape[-1] == 3 and frames.shape[0] >= 4
    pix, grid = preprocess_frames_to_patches(frames, max_pixels=262144, is_video=True)
    assert grid.shape == (1, 3) and pix.shape[0] == int(np.prod(grid))
    sizes, times, eff = audio_timestamp_groups(300, 0.1, duration=12.4, timestamp_interval_sec=1.0)
    assert sum(sizes) == 300 and len(times) == 13 and eff == 124


if __name__ == "__main__":
    test_audio_tower_shape()
    test_mrope_video_audio_interleave()
    test_sanitize_key_mapping()
    test_apollo11_media_decode()
    print("ALL_SMOKE_TESTS_PASSED")
