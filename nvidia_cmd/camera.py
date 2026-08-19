from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_camera_npz(camera_path: str | Path) -> tuple[Any, Any]:
    path = Path(camera_path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera NPZ not found: {path}")
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
