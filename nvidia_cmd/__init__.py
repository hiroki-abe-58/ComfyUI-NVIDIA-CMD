"""NVIDIA CMD adapter for ComfyUI. Importing this package must not load models."""

from .presets import CHECKPOINT_PRESETS, MemoryPreset

__all__ = ["CHECKPOINT_PRESETS", "MemoryPreset"]
