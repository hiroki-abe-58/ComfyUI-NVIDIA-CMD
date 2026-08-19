from __future__ import annotations

import os
from pathlib import Path


def _existing_dir(path: Path) -> Path | None:
    if path.is_dir():
        return path.resolve()
    return None


def find_comfy_root() -> Path | None:
    env = os.environ.get("COMFYUI_ROOT")
    if env:
        found = _existing_dir(Path(env))
        if found is not None:
            return found

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "main.py").is_file() and (parent / "comfy").is_dir():
            return parent
    return None


def default_model_root() -> Path:
    comfy = find_comfy_root()
    if comfy is not None:
        return comfy / "models" / "nvidia_cmd"
    env = os.environ.get("CMD_MODEL_ROOT")
    if env:
        return Path(env)
    return Path.cwd() / "models" / "nvidia_cmd"


def resolve_file(path_or_name: str, search_dirs: list[Path]) -> Path:
    candidate = Path(path_or_name)
    if candidate.is_file():
        return candidate.resolve()
    for directory in search_dirs:
        nested = directory / path_or_name
        if nested.is_file():
            return nested.resolve()
    searched = ", ".join(str(item) for item in search_dirs)
    raise FileNotFoundError(
        f"CMD file not found: {path_or_name}. Looked in: {searched}. "
        "Download checkpoints manually. This node never auto-downloads large models."
    )
