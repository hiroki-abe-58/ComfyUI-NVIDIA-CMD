from __future__ import annotations

import os
import sys
from pathlib import Path


REQUIRED_MARKERS = (
    "inference.py",
    "pipeline/causal_inference.py",
    "cosmos/runtime.py",
    "utils/model_factory.py",
)


def _looks_like_official_cmd(root: Path) -> bool:
    return all((root / marker).exists() for marker in REQUIRED_MARKERS)


def find_official_cmd_root() -> Path:
    env = os.environ.get("CMD_UPSTREAM")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))

    here = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            here / "third_party" / "cmd",
            here.parent / "cmd",
            Path.cwd() / "third_party" / "cmd",
            Path.cwd() / "cmd-upstream",
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if _looks_like_official_cmd(resolved):
            return resolved

    raise FileNotFoundError(
        "Official NVIDIA CMD repository was not found. Clone "
        "https://github.com/nv-tlabs/cmd and set CMD_UPSTREAM to that directory. "
        "This adapter does not vendor official sources."
    )


_OFFICIAL_TOP_LEVEL = ("utils", "cosmos", "pipeline", "wan", "inference")


def _ensure_package_markers(root: Path) -> None:
    """Official utils/ has no __init__.py; make it a regular package so it wins."""
    for name in _OFFICIAL_TOP_LEVEL:
        directory = root / name
        marker = directory / "__init__.py"
        if directory.is_dir() and not marker.exists():
            marker.write_text("# package marker added by ComfyUI-NVIDIA-CMD adapter\n")


def ensure_official_cmd_on_path(root: Path | None = None) -> Path:
    resolved = root or find_official_cmd_root()
    _ensure_package_markers(resolved)
    path = str(resolved)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

    normalized = path.replace("\\", "/").lower()
    for name in list(sys.modules):
        top = name.split(".", 1)[0]
        if top not in _OFFICIAL_TOP_LEVEL:
            continue
        module = sys.modules[name]
        module_file = (getattr(module, "__file__", "") or "").replace("\\", "/").lower()
        if not module_file or normalized not in module_file:
            del sys.modules[name]
    return resolved
