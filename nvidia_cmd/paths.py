from __future__ import annotations

import os
from pathlib import Path


def _existing_dir(path: Path) -> Path | None:
    if path.is_dir():
        return path.resolve()
    return None


def _is_comfy_root(path: Path) -> bool:
    return (path / "main.py").is_file() and (path / "comfy").is_dir()


def find_comfy_root() -> Path | None:
    env = os.environ.get("COMFYUI_ROOT")
    if env:
        found = _existing_dir(Path(env))
        if found is not None:
            return found

    try:
        import folder_paths

        base = getattr(folder_paths, "base_path", None)
        if base:
            found = _existing_dir(Path(base))
            if found is not None and _is_comfy_root(found):
                return found
    except ImportError:
        pass

    # Junction 経由の custom_node では resolve() すると ComfyUI ルートを見失う。
    for here in (Path(__file__), Path(__file__).resolve()):
        for parent in here.parents:
            if _is_comfy_root(parent):
                return parent.resolve()
    return None


def default_model_root() -> Path:
    comfy = find_comfy_root()
    if comfy is not None:
        return comfy / "models" / "nvidia_cmd"

    env = os.environ.get("CMD_MODEL_ROOT")
    if env:
        return Path(env)

    package_models = Path(__file__).resolve().parent.parent / "models" / "nvidia_cmd"
    if package_models.is_dir():
        return package_models

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
