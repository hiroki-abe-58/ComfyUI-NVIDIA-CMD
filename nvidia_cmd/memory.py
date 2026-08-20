from __future__ import annotations

from typing import Any

from .presets import PIXEL_HEIGHT, PIXEL_WIDTH, MemoryPreset

_DIT_BLOCKS = 28
_HEADS = 16
_HEAD_DIM = 128
_PATCH = 2


def _host_memory() -> dict[str, float]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_available_gb": round(vm.available / (1024**3), 2),
            "ram_percent": vm.percent,
        }
    except Exception:
        return {"ram_total_gb": -1, "ram_available_gb": -1, "ram_percent": -1}


def _cuda_memory() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"vram_total_gb": 0, "vram_alloc_gb": 0, "vram_reserved_gb": 0, "vram_free_gb": 0}
        props = torch.cuda.get_device_properties(0)
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
        return {
            "vram_total_gb": round(props.total_memory / (1024**3), 2),
            "vram_alloc_gb": round(allocated / (1024**3), 2),
            "vram_reserved_gb": round(reserved / (1024**3), 2),
            "vram_free_gb": round((props.total_memory - reserved) / (1024**3), 2),
        }
    except Exception:
        return {"vram_total_gb": -1, "vram_alloc_gb": -1, "vram_reserved_gb": -1, "vram_free_gb": -1}


def estimate_generation_footprint(
    latent_frames: int,
    *,
    local_attn_size: int,
    height: int = PIXEL_HEIGHT,
    width: int = PIXEL_WIDTH,
    latent_h: int = 60,
    latent_w: int = 104,
) -> dict[str, Any]:
    tokens = (latent_h // _PATCH) * (latent_w // _PATCH)
    per_frame_kv = tokens * _HEADS * _HEAD_DIM * 2 * 2 * _DIT_BLOCKS
    kv_uncapped = per_frame_kv * latent_frames
    kv_capped = per_frame_kv * max(1, local_attn_size)
    pixel_frames = 1 + (latent_frames - 1) * 4
    pixel_bytes = pixel_frames * 3 * height * width * 4
    return {
        "latent_frames": latent_frames,
        "pixel_frames": pixel_frames,
        "tokens_per_frame": tokens,
        "local_attn_size": local_attn_size,
        "kv_uncapped_gb": round(kv_uncapped / (1024**3), 2),
        "kv_capped_gb": round(kv_capped / (1024**3), 2),
        "decode_fp32_gb": round(pixel_bytes / (1024**3), 2),
        "comfy_copies_gb": round((pixel_bytes * 3) / (1024**3), 2),
    }


def preflight_or_raise(latent_frames: int, local_attn_size: int, height: int, width: int, latent_shape: Any) -> dict[str, Any]:
    latent_h = 60
    latent_w = 104
    if latent_shape is not None and len(latent_shape) >= 5:
        latent_h = int(latent_shape[3])
        latent_w = int(latent_shape[4])
    footprint = estimate_generation_footprint(
        latent_frames,
        local_attn_size=local_attn_size,
        height=height,
        width=width,
        latent_h=latent_h,
        latent_w=latent_w,
    )
    host = _host_memory()
    cuda = _cuda_memory()
    payload = {**footprint, **host, **cuda}
    vram_total = cuda["vram_total_gb"]
    ram_avail = host["ram_available_gb"]
    unsafe_kv = vram_total > 0 and footprint["kv_capped_gb"] > vram_total * 0.55
    unsafe_ram = (
        ram_avail >= 0
        and (footprint["kv_capped_gb"] + footprint["comfy_copies_gb"]) > ram_avail * 0.85
    )
    if unsafe_kv or unsafe_ram:
        raise MemoryError(
            "CMD refused generation: even the local-attn KV window "
            f"({footprint['kv_capped_gb']}GiB) exceeds the safe budget on "
            f"{vram_total}GiB VRAM / {ram_avail}GiB free RAM."
        )
    return payload


def circular_history_slots(current_idx: int, capacity: int) -> list[int]:
    if current_idx <= 0 or capacity <= 0:
        return []
    start = max(0, current_idx - capacity)
    return [index % capacity for index in range(start, current_idx)]


def install_capped_kv_cache(pipeline: Any, local_attn_size: int) -> None:
    """Keep only local_attn_size latent frames in GPU KV. Official cache stores every frame."""
    import torch

    cap = max(1, int(local_attn_size))
    generator = pipeline.generator
    original_init = generator.initialize_kv_cache
    if not getattr(original_init, "_cmd_capped", False):

        def capped_init(max_frames, batch_size=1, dtype=None, device=None):
            used = min(int(max_frames), cap)
            print(f"CMD: KV cache frames {max_frames} -> {used} (local_attn_size)", flush=True)
            return original_init(max_frames=used, batch_size=batch_size, dtype=dtype, device=device)

        capped_init._cmd_capped = True  # type: ignore[attr-defined]
        generator.initialize_kv_cache = capped_init

    from cosmos.kv_cache import AttentionOpWithKVCache

    if getattr(AttentionOpWithKVCache.forward, "_cmd_circular", False):
        return

    def circular_forward(self, q, k, v, *, kv_cache_cfg, **kwargs):
        capacity = self.max_cache_size
        current_idx = int(kv_cache_cfg.current_idx)
        if kv_cache_cfg.run_with_kv and current_idx > 0 and capacity:
            history_k = []
            history_v = []
            for slot in circular_history_slots(current_idx, capacity):
                hist_k = self.k_cache[slot]
                hist_v = self.v_cache[slot]
                if hist_k is None or hist_v is None:
                    raise RuntimeError(f"KV cache slot {slot} empty at frame {current_idx}")
                history_k.append(hist_k)
                history_v.append(hist_v)
            k_out = torch.cat(history_k + [k], dim=1)
            v_out = torch.cat(history_v + [v], dim=1)
        else:
            k_out = k
            v_out = v
        if kv_cache_cfg.store_kv and capacity:
            index = current_idx % capacity
            self.k_cache[index] = k.detach()
            self.v_cache[index] = v.detach()
        if kv_cache_cfg.run_with_kv and capacity is not None:
            self.start_idx = max(0, current_idx - capacity)
        return self.attn_op(q, k_out, v_out, **kwargs)

    circular_forward._cmd_circular = True  # type: ignore[attr-defined]
    AttentionOpWithKVCache.forward = circular_forward


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
