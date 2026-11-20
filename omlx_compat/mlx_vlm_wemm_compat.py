"""oMLX / mlx-vlm compatibility bridge for Tencent/WeMM-Video (wemm_qwen3_5_moe).

Registers wemm_qwen3_5_moe cleanly onto mlx_vlm.models, mlx_vlm.prompt_utils,
AutoProcessor, and oMLX's native video model set without requiring PyTorch.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType

logger = logging.getLogger(__name__)

_MODEL_TYPES = ("wemm_qwen3_5_moe",)
_APPLIED = False


def is_applied() -> bool:
    return _APPLIED


def apply_mlx_vlm_wemm_compat_patch() -> bool:
    """Register wemm_qwen3_5_moe model + processor into mlx-vlm and oMLX."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm.models as vlm_models
        import mlx_vlm.prompt_utils as prompt_utils
        import mlx_vlm.utils as vlm_utils
        from mlx_vlm.models.base import install_auto_processor_patch

        import mlx_wemm_video.config as wemm_cfg
        import mlx_wemm_video.model as wemm_model
        import mlx_wemm_video.processing_wemm as wemm_proc

        # 1. Expose virtual module mlx_vlm.models.wemm_qwen3_5_moe
        mod_name = "mlx_vlm.models.wemm_qwen3_5_moe"
        if mod_name not in sys.modules:
            shim = ModuleType(mod_name)
            shim.Model = wemm_model.Model
            shim.LanguageModel = wemm_model.LanguageModel
            shim.VisionModel = wemm_model.VisionModel
            shim.ModelConfig = wemm_cfg.ModelConfig
            shim.TextConfig = wemm_cfg.TextConfig
            shim.VisionConfig = wemm_cfg.VisionConfig
            shim.AudioConfig = wemm_cfg.AudioConfig
            shim.WeMMProcessor = wemm_proc.WeMMProcessor
            sys.modules[mod_name] = shim
            setattr(vlm_models, "wemm_qwen3_5_moe", shim)

        # 2. Register prompt formatting rules (matches Qwen3.5 MoE + audio)
        model_cfg = getattr(prompt_utils, "MODEL_CONFIG", None)
        if isinstance(model_cfg, dict):
            for mt in _MODEL_TYPES:
                model_cfg[mt] = prompt_utils.MessageFormat.LIST_WITH_IMAGE_FIRST

        # 3. Register model remapping
        remapping = getattr(vlm_utils, "MODEL_REMAPPING", None)
        if isinstance(remapping, dict):
            remapping.setdefault("wemm_qwen3_5_moe", "wemm_qwen3_5_moe")

        # 4. Register torch-free WeMMProcessor on AutoProcessor
        install_auto_processor_patch(list(_MODEL_TYPES), wemm_proc.WeMMProcessor)

        # 5. If running inside oMLX, enable native video path for wemm_qwen3_5_moe
        try:
            import omlx.utils.video as omlx_video

            if hasattr(omlx_video, "NATIVE_VIDEO_MODEL_TYPES"):
                omlx_video.NATIVE_VIDEO_MODEL_TYPES = frozenset(
                    set(omlx_video.NATIVE_VIDEO_MODEL_TYPES) | set(_MODEL_TYPES)
                )
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("WeMM mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("WeMM-Video (wemm_qwen3_5_moe) mlx-vlm compatibility patch applied")
    return True
