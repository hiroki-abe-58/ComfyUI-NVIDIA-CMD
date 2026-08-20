from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def resolve_camera_path(camera_path: str | Path) -> Path:
    candidate = Path(camera_path)
    if candidate.is_file():
        return candidate.resolve()

    search: list[Path] = []
    package_root = Path(__file__).resolve().parent.parent
    if not candidate.is_absolute():
        search.extend(
            [
                Path.cwd() / candidate,
                package_root / candidate,
                package_root / "examples" / candidate.name,
            ]
        )
    search.append(package_root / "examples" / candidate.name)
    try:
        import folder_paths

        input_dir = Path(folder_paths.get_input_directory())
        search.append(input_dir / candidate.name)
        search.append(input_dir / candidate)
    except Exception:
        pass

    for item in search:
        if item.is_file():
            return item.resolve()
    looked = ", ".join(str(item) for item in search)
    raise FileNotFoundError(f"Camera NPZ not found: {camera_path}. Looked in: {looked}")


def load_camera_npz(camera_path: str | Path) -> tuple[Any, Any]:
    path = resolve_camera_path(camera_path)
    with np.load(path, allow_pickle=False) as camera:
        if "target_w2c" not in camera or "target_intrinsics" not in camera:
            raise ValueError(
                "Camera NPZ must contain target_w2c and target_intrinsics, "
                "matching official nv-tlabs/cmd inference.py."
            )
        world_to_camera = np.asarray(camera["target_w2c"], dtype=np.float32)
        intrinsics = np.asarray(camera["target_intrinsics"], dtype=np.float32)
    return world_to_camera, intrinsics


def poses_from_w2c(
    world_to_camera: Any,
    intrinsics: Any,
    latent_frames: int,
    frame_stride: int = 4,
    batch_size: int = 1,
) -> tuple[Any, Any]:
    import torch

    pixel_frames = 1 + (latent_frames - 1) * frame_stride
    if len(world_to_camera) < pixel_frames or len(intrinsics) < pixel_frames:
        raise ValueError(
            f"Camera input needs {pixel_frames} pixel frames for {latent_frames} latents."
        )
    camera_to_world = np.linalg.inv(world_to_camera[:pixel_frames]).astype(np.float32)
    poses = torch.from_numpy(camera_to_world).unsqueeze(0)
    calibration = torch.from_numpy(intrinsics[:pixel_frames]).unsqueeze(0)
    return poses.repeat(batch_size, 1, 1, 1), calibration.repeat(batch_size, 1, 1, 1)
