from __future__ import annotations

import os
from typing import Literal

AttentionBackend = Literal["auto", "sdpa", "flash_attention"]


def resolve_backend(requested: str) -> AttentionBackend:
    value = requested.lower().strip()
    if value not in {"auto", "sdpa", "flash_attention"}:
        raise ValueError(
            f"Unsupported attention backend: {requested}. Use auto, sdpa, or flash_attention."
        )
    if value == "auto":
        return "sdpa"
    if value == "flash_attention":
        try:
            import flash_attn  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "flash_attention was requested but flash-attn is not importable. "
                "Use sdpa on Windows Native / RTX 50."
            ) from exc
    return value  # type: ignore[return-value]


def disable_torch_compile() -> None:
    """Skip torch.compile. Official causal_model compiles FlexAttention at import."""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import torch

    if getattr(torch.compile, "_cmd_disabled", False):
        return

    def _identity(fn=None, *args, **kwargs):
        if fn is None:
            return lambda inner: inner
        return fn

    _identity._cmd_disabled = True  # type: ignore[attr-defined]
    torch.compile = _identity  # type: ignore[assignment]


def force_sdpa() -> None:
    """Force official cosmos.runtime to skip Transformer Engine and use SDPA."""
    os.environ["CMD_ATTENTION_BACKEND"] = "sdpa"
    try:
        import cosmos.runtime as runtime
    except Exception:
        return

    runtime._TransformerEngineAttention = None
    runtime._transformer_engine_rope = None

    original = runtime.attention

    def sdpa_only(query, key, value, *, is_causal: bool = False, **kwargs):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        del kwargs
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if not getattr(sdpa_only, "_shape_logged", False):
            print(f"CMD: SDPA q={tuple(q.shape)} k={tuple(k.shape)}", flush=True)
            sdpa_only._shape_logged = True  # type: ignore[attr-defined]
        backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
        try:
            with sdpa_kernel(backends):
                output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=is_causal
                )
        except Exception:
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )
        return output.transpose(1, 2)

    if getattr(original, "_cmd_sdpa_patched", False):
        return
    sdpa_only._cmd_sdpa_patched = True  # type: ignore[attr-defined]
    runtime.attention = sdpa_only
