from __future__ import annotations

from typing import Any

from .presets import MemoryPreset


def _comfy_model_management() -> Any | None:
    try:
        import comfy.model_management as model_management
    except Exception:
        return None
    return model_management


def empty_cache() -> None:
    import gc

    gc.collect()
    manager = _comfy_model_management()
    if manager is not None and hasattr(manager, "soft_empty_cache"):
        manager.soft_empty_cache()
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        return


def to_device(module: Any, device: str) -> None:
    module.to(device)


def _wrap_balanced_text_encoder(pipeline: Any, device: str) -> None:
    """Encode on CPU so the 7B Reason1 model never shares VRAM with the DiT."""
    encoder = pipeline.text_encoder
    if getattr(encoder, "_cmd_balanced_wrapped", False):
        return
    original = encoder.forward

    def forward(*args, **kwargs):
        to_device(encoder, "cpu")
        result = original(*args, **kwargs)
        if isinstance(result, dict):
            moved = {}
            for key, value in result.items():
                moved[key] = value.to(device) if hasattr(value, "to") else value
            result = moved
        empty_cache()
        return result

    encoder.forward = forward
    encoder._cmd_balanced_wrapped = True


def apply_preset(pipeline: Any, preset: MemoryPreset, device: str) -> None:
    """Keep the student on GPU. Offload helpers according to the preset."""
    if preset is MemoryPreset.FULL:
        to_device(pipeline.text_encoder, device)
        to_device(pipeline.generator, device)
        to_device(pipeline.vae, device)
        return

    to_device(pipeline.generator, device)
    if preset is MemoryPreset.LOW_VRAM:
        to_device(pipeline.text_encoder, "cpu")
        to_device(pipeline.vae, "cpu")
    else:
        to_device(pipeline.text_encoder, "cpu")
        to_device(pipeline.vae, device)
        _wrap_balanced_text_encoder(pipeline, device)
    empty_cache()


def offload_text_encoder(pipeline: Any) -> None:
    to_device(pipeline.text_encoder, "cpu")
    empty_cache()


def prepare_vae(pipeline: Any, device: str) -> None:
    to_device(pipeline.vae, device)
    empty_cache()
