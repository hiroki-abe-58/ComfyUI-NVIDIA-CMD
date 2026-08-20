from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attention import AttentionBackend, disable_torch_compile, force_sdpa, resolve_backend
from .paths import default_model_root, resolve_file
from .presets import CHECKPOINT_PRESETS, CheckpointPreset, MemoryPreset
from .upstream import ensure_official_cmd_on_path, find_official_cmd_root

_COSMOS_CHECKPOINT_OPTIONAL_BUFFERS = {
    "model.accum_video_sample_counter",
    "model.accum_image_sample_counter",
    "model.accum_iteration",
    "model.accum_train_in_hours",
}


@dataclass
class LoadedCMD:
    pipeline: Any
    config: Any
    preset: CheckpointPreset
    checkpoint_path: Path
    attention_backend: AttentionBackend
    memory_preset: MemoryPreset
    device: str
    dtype_name: str


def load_config(preset: CheckpointPreset, upstream_root: Path) -> Any:
    from omegaconf import OmegaConf

    family_dir = upstream_root / "configs" / "cosmos"
    default_path = family_dir / "default_config.yaml"
    config_path = family_dir / preset.config_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Official config missing: {config_path}")
    config = OmegaConf.load(config_path)
    if default_path.is_file():
        config = OmegaConf.merge(OmegaConf.load(default_path), config)
    config.num_frame_per_block = preset.num_frame_per_block
    config.model_kwargs.local_attn_size = preset.local_attn_size
    if preset.camera:
        config.camera_conditioning = True
        config.camera_frame_stride = 4
        config.camera_patch_size = 16
        config.model_kwargs.camera_conditioning = True
        config.model_kwargs.camera_patch_size = 16
        config.model_kwargs.camera_init_seed = 0
    return config


def _resolve_local_component(search_dirs: list[Path], names: list[str]) -> Path | None:
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()
        if directory.exists() and any(directory.iterdir()):
            marker = directory / "config.json"
            if marker.is_file():
                return directory.resolve()
    return None


def force_text_encoder_cpu() -> None:
    """Keep Reason1 on CPU. Transformers may otherwise dispatch 7B onto CUDA."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    original = Qwen2_5_VLForConditionalGeneration.from_pretrained

    @classmethod
    def wrapped(cls, *args, **kwargs):
        kwargs["device_map"] = "cpu"
        kwargs["low_cpu_mem_usage"] = True
        raw = original.__func__ if hasattr(original, "__func__") else original
        return raw(cls, *args, **kwargs)

    Qwen2_5_VLForConditionalGeneration.from_pretrained = wrapped


def install_local_hf_hub(model_root: Path) -> None:
    """Resolve VAE / base files from models/nvidia_cmd before Hugging Face."""
    import huggingface_hub

    original = huggingface_hub.hf_hub_download

    def wrapped(repo_id: str, filename: str, *args, **kwargs):
        for candidate in (
            Path(repo_id) / filename,
            model_root / "vae" / filename,
            model_root / filename,
            Path(filename),
        ):
            if candidate.is_file():
                return str(candidate.resolve())
        return original(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = wrapped
    try:
        import cosmos.wrapper as cosmos_wrapper

        cosmos_wrapper.hf_hub_download = wrapped
    except Exception:
        pass


def apply_local_asset_overrides(config: Any, model_root: Path) -> None:
    text_dir = _resolve_local_component(
        [model_root / "text_encoder", model_root],
        ["nvidia/Cosmos-Reason1-7B", "Cosmos-Reason1-7B"],
    )
    if text_dir is not None and text_dir.is_dir():
        config.text_encoder_name = str(text_dir)

    vae_file = None
    vae_names = ("tokenizer.pth", "Wan2.1_VAE.pth", "wan_2.1_vae.pth")
    for directory in (model_root / "vae", model_root):
        for name in vae_names:
            candidate = directory / name
            if candidate.is_file():
                vae_file = candidate
                break
        if vae_file is not None:
            break
    if vae_file is not None:
        config.vae_model_name = str(vae_file.parent)
        config.vae_checkpoint_filename = vae_file.name


def require_local_inference_assets(config: Any, model_root: Path) -> None:
    text_name = str(config.text_encoder_name)
    if not Path(text_name).is_dir():
        raise FileNotFoundError(
            "Cosmos-Reason1-7B is not on disk. Place it under "
            f"{model_root / 'text_encoder'} and do not rely on automatic download."
        )
    vae_name = Path(str(config.vae_model_name)) / str(
        getattr(config, "vae_checkpoint_filename", "tokenizer.pth")
    )
    fallbacks = [
        model_root / "vae" / "tokenizer.pth",
        model_root / "vae" / "Wan2.1_VAE.pth",
    ]
    if not vae_name.is_file() and not any(path.is_file() for path in fallbacks):
        raise FileNotFoundError(
            "Wan VAE is not on disk. Place tokenizer.pth from "
            "nvidia/Cosmos-Predict2.5-2B (gated) or Wan2.1_VAE.pth under "
            f"{model_root / 'vae'}."
        )


def load_generator_checkpoint(pipeline: Any, checkpoint_path: Path) -> None:
    """Map released bare DiT keys onto pipeline.generator (official contract)."""
    from safetensors.torch import load_file

    state_dict = load_file(str(checkpoint_path), device="cpu")
    normalized = {}
    for name, value in state_dict.items():
        name = name.removeprefix("generator.")
        normalized_name = name if name.startswith("model.") else f"model.{name}"
        if normalized_name in normalized:
            raise RuntimeError(f"Duplicate checkpoint key after normalization: {normalized_name}")
        normalized[normalized_name] = value

    incompatible = pipeline.generator.load_state_dict(normalized, strict=False)
    invalid_missing = set(incompatible.missing_keys) - _COSMOS_CHECKPOINT_OPTIONAL_BUFFERS
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible Cosmos generator checkpoint: "
            f"missing={sorted(invalid_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )


def install_student_direct_load(checkpoint_path: Path) -> None:
    """Skip Predict2.5 base .pt download; initialize the causal DiT from the student."""
    from cosmos.wrapper import CosmosDiffusionWrapper

    student_path = Path(checkpoint_path)

    @classmethod
    def _load_model(
        cls,
        model_name,
        checkpoint_filename,
        is_causal,
        local_attn_size,
        sink_size,
    ):
        del model_name, checkpoint_filename
        import torch
        from safetensors.torch import load_file

        if is_causal:
            from cosmos.causal_model import CausalCosmosModel as Model
        else:
            from cosmos.minimal_v1_lvg_dit import MinimalV1LVGDiT as Model

        with torch.device("meta"):
            model_kwargs = cls._model_kwargs(is_causal)
            if is_causal:
                model_kwargs.update(
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                )
            model = Model(**model_kwargs)

        state_dict = load_file(str(student_path), device="cpu")
        remapped = {}
        for name, value in state_dict.items():
            name = name.removeprefix("generator.").removeprefix("model.")
            remapped[name] = value
        model.load_state_dict(remapped, strict=False, assign=True)
        optional = {
            name.removeprefix("model.")
            for name in _COSMOS_CHECKPOINT_OPTIONAL_BUFFERS
        }
        for name, tensor in list(model.named_buffers()):
            if tensor.is_meta and name in optional:
                model.register_buffer(name, torch.zeros(tensor.shape, dtype=tensor.dtype), persistent=True)
        unloaded = [
            name
            for name, tensor in (list(model.named_parameters()) + list(model.named_buffers()))
            if tensor.is_meta
        ]
        if unloaded:
            raise RuntimeError(
                "Student checkpoint did not initialize parameters: " + ", ".join(unloaded[:10])
            )
        return model

    CosmosDiffusionWrapper._load_model = _load_model


def load_cmd_pipeline(
    *,
    checkpoint: str = "chunk1_short",
    checkpoint_path: str | None = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    attention_backend: str = "sdpa",
    memory_preset: str = "BALANCED",
    model_root: str | None = None,
) -> LoadedCMD:
    if dtype not in {"bfloat16", "bf16"}:
        raise ValueError("MVP supports only bfloat16.")

    backend = resolve_backend(attention_backend)
    preset_name = checkpoint
    if preset_name not in CHECKPOINT_PRESETS:
        raise ValueError(
            f"Unknown checkpoint preset: {checkpoint}. "
            f"Known: {', '.join(CHECKPOINT_PRESETS)}"
        )
    preset = CHECKPOINT_PRESETS[preset_name]
    memory = MemoryPreset(memory_preset)

    upstream_root = find_official_cmd_root()
    ensure_official_cmd_on_path(upstream_root)
    search_root = Path(model_root) if model_root else default_model_root()
    install_local_hf_hub(search_root)

    import torch

    torch.set_grad_enabled(False)
    disable_torch_compile()
    force_text_encoder_cpu()
    print("CMD: official repo ready, loading CausalInferencePipeline", flush=True)

    from pipeline.causal_inference import CausalInferencePipeline

    from .memory import apply_preset, empty_cache, install_capped_kv_cache

    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt is None:
        ckpt = resolve_file(
            preset.filename,
            [search_root / "transformer", search_root, Path.cwd() / "checkpoints"],
        )
    elif not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    config = load_config(preset, upstream_root)
    apply_local_asset_overrides(config, search_root)
    require_local_inference_assets(config, search_root)
    if backend == "sdpa":
        force_sdpa()
        print("CMD: attention backend=sdpa", flush=True)
    install_local_hf_hub(search_root)
    install_student_direct_load(ckpt)
    print(f"CMD: constructing pipeline from {ckpt.name}", flush=True)
    pipeline = CausalInferencePipeline(config, device=device)
    install_capped_kv_cache(pipeline, preset.local_attn_size)
    pipeline.text_encoder.to("cpu")
    te_devices = {str(param.device) for param in pipeline.text_encoder.parameters()}
    print(f"CMD: text encoder devices={sorted(te_devices)}", flush=True)
    empty_cache()
    print("CMD: loading student weights", flush=True)
    load_generator_checkpoint(pipeline, ckpt)
    pipeline.generator.to(dtype=torch.bfloat16)
    pipeline.vae.to(dtype=torch.bfloat16)
    apply_preset(pipeline, memory, device)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"CMD: ready memory={memory.value} allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB", flush=True)
    else:
        print(f"CMD: ready memory={memory.value} device={device}", flush=True)

    return LoadedCMD(
        pipeline=pipeline,
        config=config,
        preset=preset,
        checkpoint_path=ckpt,
        attention_backend=backend,
        memory_preset=memory,
        device=device,
        dtype_name="bfloat16",
    )
