from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def identity_camera(latent_frames: int, frame_stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    pixel_frames = 1 + (latent_frames - 1) * frame_stride
    world_to_camera = np.repeat(np.eye(4, dtype=np.float32)[None, ...], pixel_frames, axis=0)
    # fx, fy, cx, cy as a 3x3-style intrinsic packed to 3x3
    intrinsics = np.repeat(
        np.array([[832.0, 0.0, 416.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)[
            None, ...
        ],
        pixel_frames,
        axis=0,
    )
    return world_to_camera, intrinsics


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a static identity camera NPZ.")
    parser.add_argument("--latent-frames", type=int, default=32)
    parser.add_argument("--output", default="examples/identity_camera.npz")
    args = parser.parse_args()
    world_to_camera, intrinsics = identity_camera(args.latent_frames)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, target_w2c=world_to_camera, target_intrinsics=intrinsics)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
