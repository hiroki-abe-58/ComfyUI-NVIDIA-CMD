from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvidia_cmd.camera import load_camera_npz


def test_load_camera_npz(tmp_path: Path):
    path = tmp_path / "cam.npz"
    frames = 5
    np.savez(
        path,
        target_w2c=np.repeat(np.eye(4, dtype=np.float32)[None, ...], frames, axis=0),
        target_intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None, ...], frames, axis=0),
    )
    world_to_camera, intrinsics = load_camera_npz(path)
    assert world_to_camera.shape[0] == frames
    assert intrinsics.shape[0] == frames
