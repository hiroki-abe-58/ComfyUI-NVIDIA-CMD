"""Scope process-wide patches to CMD pipeline construction only.

Official CMD import compiles FlexAttention and may place Reason1 on CUDA.
Those torch / transformers / huggingface_hub patches must not remain after
CausalInferencePipeline is built. CMD-module patches (SDPA, student load,
circular KV) stay applied; they are not restored here.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class LocalHFHubGuard:
    """Temporary local-file resolver for huggingface_hub.hf_hub_download."""

    def __init__(self, original_hf: Any, wrapped: Any) -> None:
        self._original_hf = original_hf
        self._wrapped = wrapped

    def bind_cosmos(self) -> None:
        module = sys.modules.get("cosmos.wrapper")
        if module is not None:
            module.hf_hub_download = self._wrapped

    def restore(self) -> None:
        import huggingface_hub

        huggingface_hub.hf_hub_download = self._original_hf
        module = sys.modules.get("cosmos.wrapper")
        if module is not None:
            module.hf_hub_download = self._original_hf


@contextmanager
def scoped_torch_compile_identity() -> Iterator[None]:
    """Replace torch.compile with identity while official CMD imports/builds."""
    dynamo_prev = os.environ.get("TORCHDYNAMO_DISABLE")
    compile_prev = os.environ.get("TORCH_COMPILE_DISABLE")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import torch

    original = torch.compile

    def _identity(fn=None, *args, **kwargs):
        if fn is None:
            return lambda inner: inner
        return fn

    torch.compile = _identity  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.compile = original
        if dynamo_prev is None:
            os.environ.pop("TORCHDYNAMO_DISABLE", None)
        else:
            os.environ["TORCHDYNAMO_DISABLE"] = dynamo_prev
        if compile_prev is None:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
        else:
            os.environ["TORCH_COMPILE_DISABLE"] = compile_prev


@contextmanager
def scoped_text_encoder_cpu() -> Iterator[None]:
    """Force Reason1 onto CPU only while the pipeline constructor runs."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    original = Qwen2_5_VLForConditionalGeneration.from_pretrained

    @classmethod
    def wrapped(cls, *args, **kwargs):
        kwargs["device_map"] = "cpu"
        kwargs["low_cpu_mem_usage"] = True
        raw = original.__func__ if hasattr(original, "__func__") else original
        return raw(cls, *args, **kwargs)

    Qwen2_5_VLForConditionalGeneration.from_pretrained = wrapped
    try:
        yield
    finally:
        Qwen2_5_VLForConditionalGeneration.from_pretrained = original


@contextmanager
def scoped_local_hf_hub(model_root: Path) -> Iterator[LocalHFHubGuard]:
    """Resolve VAE / base files from models/nvidia_cmd before Hugging Face."""
    import huggingface_hub

    original = huggingface_hub.hf_hub_download
    root = Path(model_root)

    def wrapped(repo_id: str, filename: str, *args, **kwargs):
        for candidate in (
            Path(repo_id) / filename,
            root / "vae" / filename,
            root / filename,
            Path(filename),
        ):
            if candidate.is_file():
                return str(candidate.resolve())
        return original(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = wrapped
    guard = LocalHFHubGuard(original, wrapped)
    guard.bind_cosmos()
    try:
        yield guard
    finally:
        guard.restore()


@contextmanager
def scoped_cmd_construction(model_root: Path) -> Iterator[LocalHFHubGuard]:
    """Apply compile / Reason1 / HF patches only around official construct."""
    with scoped_torch_compile_identity():
        with scoped_text_encoder_cpu():
            with scoped_local_hf_hub(model_root) as hf_guard:
                yield hf_guard
