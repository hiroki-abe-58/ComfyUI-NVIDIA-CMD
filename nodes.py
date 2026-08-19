from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .nvidia_cmd.paths import default_model_root
    from .nvidia_cmd.presets import CHECKPOINT_PRESETS, PIXEL_HEIGHT, PIXEL_WIDTH
except ImportError:
    from nvidia_cmd.paths import default_model_root
    from nvidia_cmd.presets import CHECKPOINT_PRESETS, PIXEL_HEIGHT, PIXEL_WIDTH


class NVIDIACMDModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (list(CHECKPOINT_PRESETS.keys()), {"default": "chunk1_short"}),
                "dtype": (["bfloat16"], {"default": "bfloat16"}),
                "device": (["cuda"], {"default": "cuda"}),
                "attention": (["auto", "sdpa"], {"default": "sdpa"}),
                "memory_preset": (["BALANCED", "FULL", "LOW_VRAM"], {"default": "BALANCED"}),
            },
            "optional": {
                "checkpoint_path": ("STRING", {"default": ""}),
                "model_root": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("CMD_MODEL",)
    RETURN_NAMES = ("cmd_model",)
    FUNCTION = "load"
    CATEGORY = "NVIDIA/CMD"

    def load(
        self,
        checkpoint: str,
        dtype: str,
        device: str,
        attention: str,
        memory_preset: str,
        checkpoint_path: str = "",
        model_root: str = "",
    ):
        try:
            from .nvidia_cmd.loader import load_cmd_pipeline
        except ImportError:
            from nvidia_cmd.loader import load_cmd_pipeline

        loaded = load_cmd_pipeline(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path or None,
            dtype=dtype,
            device=device,
            attention_backend=attention,
            memory_preset=memory_preset,
            model_root=model_root or None,
        )
        return (loaded,)


class NVIDIACMDImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cmd_model": ("CMD_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "camera": ("CMD_CAMERA",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "NVIDIA/CMD"

    def generate(self, cmd_model: Any, image, prompt: str, seed: int, camera=None):
        try:
            from .nvidia_cmd.pipeline import frames_to_comfy_image, generate_video
        except ImportError:
            from nvidia_cmd.pipeline import frames_to_comfy_image, generate_video

        frames = generate_video(
            cmd_model,
            image=image.detach().cpu().numpy(),
            prompt=prompt,
            seed=int(seed),
            camera_path=None if camera is None else camera.get("path"),
        )
        return (frames_to_comfy_image(frames),)


class NVIDIACMDCameraControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_npz": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CMD_CAMERA",)
    RETURN_NAMES = ("camera",)
    FUNCTION = "load"
    CATEGORY = "NVIDIA/CMD"

    def load(self, camera_npz: str):
        try:
            from .nvidia_cmd.camera import load_camera_npz
        except ImportError:
            from nvidia_cmd.camera import load_camera_npz

        path = Path(camera_npz)
        load_camera_npz(path)
        return ({"path": str(path.resolve())},)


class NVIDIACMDLongVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return NVIDIACMDImageToVideo.INPUT_TYPES()

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "NVIDIA/CMD"

    def generate(self, cmd_model: Any, image, prompt: str, seed: int, camera=None):
        if not cmd_model.preset.long_video:
            raise ValueError(
                "NVIDIACMDLongVideo requires a long checkpoint such as chunk1_long or chunk4_long."
            )
        return NVIDIACMDImageToVideo().generate(cmd_model, image, prompt, seed, camera)


NODE_CLASS_MAPPINGS = {
    "NVIDIACMDModelLoader": NVIDIACMDModelLoader,
    "NVIDIACMDImageToVideo": NVIDIACMDImageToVideo,
    "NVIDIACMDCameraControl": NVIDIACMDCameraControl,
    "NVIDIACMDLongVideo": NVIDIACMDLongVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NVIDIACMDModelLoader": "NVIDIA CMD Model Loader",
    "NVIDIACMDImageToVideo": "NVIDIA CMD Image to Video",
    "NVIDIACMDCameraControl": "NVIDIA CMD Camera Control",
    "NVIDIACMDLongVideo": "NVIDIA CMD Long Video",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PIXEL_HEIGHT",
    "PIXEL_WIDTH",
    "default_model_root",
]
