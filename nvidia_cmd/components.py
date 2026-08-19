from __future__ import annotations

from pathlib import Path

from .paths import default_model_root, resolve_file
from .presets import TEXT_ENCODER_ID, VAE_FILENAME, VAE_REPO_ID


def describe_required_assets(model_root: Path | None = None) -> str:
    root = model_root or default_model_root()
    return (
        "Place official files manually:\n"
        f"  {root / 'transformer'}  <- nvidia/cmd student *.safetensors\n"
        f"  {root / 'text_encoder'} <- {TEXT_ENCODER_ID}\n"
        f"  {root / 'vae'}          <- {VAE_REPO_ID}/{VAE_FILENAME}\n"
        "This project never downloads those weights automatically."
    )


def text_encoder_search_dirs(model_root: Path | None = None) -> list[Path]:
    root = model_root or default_model_root()
    return [root / "text_encoder", root]


def vae_search_dirs(model_root: Path | None = None) -> list[Path]:
    root = model_root or default_model_root()
    return [root / "vae", root]


def transformer_search_dirs(model_root: Path | None = None) -> list[Path]:
    root = model_root or default_model_root()
    return [root / "transformer", root]


def resolve_optional_local_dir(name: str, search_dirs: list[Path]) -> str | None:
    try:
        path = resolve_file(name, search_dirs)
    except FileNotFoundError:
        directory_hits = [item for item in search_dirs if (item / name).is_dir() or item.name == name]
        if directory_hits:
            return str(directory_hits[0])
        return None
    return str(path.parent if path.is_file() else path)
