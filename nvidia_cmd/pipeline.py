from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .camera import load_camera_npz, poses_from_w2c
from .loader import LoadedCMD
from .memory import empty_cache, offload_text_encoder, preflight_or_raise, prepare_vae
from .presets import FPS, PIXEL_HEIGHT, PIXEL_WIDTH, MemoryPreset


def _prepare_image(image, height: int, width: int) -> Any:
    import torch
    from PIL import Image
    from torchvision import transforms

    if isinstance(image, np.ndarray):
        if image.ndim == 4:
            image = image[0]
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    image = image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    tensor = transform(image)
    return tensor.unsqueeze(0).unsqueeze(2)


def generate_video(
    loaded: LoadedCMD,
    *,
    image,
    prompt: str,
    seed: int = 0,
    camera_path: str | Path | None = None,
    num_output_frames: int | None = None,
) -> np.ndarray:
    """Return uint8 frames as [T, H, W, C]."""
    import torch

    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    with torch.inference_mode():
        return _generate_video_inner(
            loaded,
            image=image,
            prompt=prompt,
            seed=seed,
            camera_path=camera_path,
            num_output_frames=num_output_frames,
        )


def _generate_video_inner(
    loaded: LoadedCMD,
    *,
    image,
    prompt: str,
    seed: int,
    camera_path: str | Path | None,
    num_output_frames: int | None,
) -> np.ndarray:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pipeline = loaded.pipeline
    config = loaded.config
    device = loaded.device
    frames = num_output_frames or loaded.preset.num_output_frames
    height = int(getattr(config, "height", PIXEL_HEIGHT))
    width = int(getattr(config, "width", PIXEL_WIDTH))
    preflight_or_raise(
        frames,
        int(loaded.preset.local_attn_size),
        height,
        width,
        getattr(config, "image_or_video_shape", None),
    )

    image_tensor = _prepare_image(image, height, width).to(device=device, dtype=torch.bfloat16)
    prepare_vae(pipeline, device)
    initial_latent = pipeline.vae.encode_to_latent(image_tensor).to(
        device=device, dtype=torch.bfloat16
    )
    if loaded.memory_preset is MemoryPreset.LOW_VRAM:
        from .memory import to_device

        to_device(pipeline.vae, "cpu")
        empty_cache()

    camera_poses = None
    camera_intrinsics = None
    if loaded.preset.camera or camera_path is not None:
        if camera_path is None:
            raise ValueError("Camera checkpoint requires a camera NPZ.")
        world_to_camera, intrinsics = load_camera_npz(camera_path)
        camera_poses, camera_intrinsics = poses_from_w2c(
            world_to_camera,
            intrinsics,
            frames,
            frame_stride=int(getattr(config, "camera_frame_stride", 4)),
        )
        camera_poses = camera_poses.to(device)
        camera_intrinsics = camera_intrinsics.to(device)

    noise = torch.randn(
        [1, frames - 1, *config.image_or_video_shape[2:]],
        device=device,
        dtype=torch.bfloat16,
    )
    if torch.cuda.is_available():
        print(
            "CMD: before inference "
            f"allocated={torch.cuda.memory_allocated() / (1024**3):.2f}GiB",
            flush=True,
        )

    prepare_vae(pipeline, device)
    video, _latents = pipeline.inference(
        noise=noise,
        text_prompts=[prompt],
        return_latents=True,
        initial_latent=initial_latent,
        camera_poses=camera_poses,
        camera_intrinsics=camera_intrinsics,
    )

    offload_text_encoder(pipeline)
    empty_cache()

    frames_nchw = video[0].detach().float().cpu().clamp(0, 1)
    frames_nhwc = (frames_nchw.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"CMD: peak allocated={peak:.0f}MiB", flush=True)
    if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
        pipeline.vae.model.clear_cache()
    empty_cache()
    return frames_nhwc


def frames_to_comfy_image(frames: np.ndarray) -> Any:
    import torch

    tensor = torch.from_numpy(frames.astype(np.float32) / 255.0)
    return tensor


def write_mp4(frames: np.ndarray, output_path: str | Path, fps: int = FPS) -> Path:
    import torch
    from torchvision.io import write_video

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(path), torch.from_numpy(frames), fps=fps)
    return path
