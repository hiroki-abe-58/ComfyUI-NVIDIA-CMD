from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryPreset(str, Enum):
    FULL = "FULL"
    BALANCED = "BALANCED"
    LOW_VRAM = "LOW_VRAM"


@dataclass(frozen=True)
class CheckpointPreset:
    name: str
    filename: str
    config_name: str
    num_output_frames: int
    num_frame_per_block: int
    local_attn_size: int
    camera: bool = False
    long_video: bool = False


CHECKPOINT_PRESETS: dict[str, CheckpointPreset] = {
    "chunk1_short": CheckpointPreset(
        name="chunk1_short",
        filename="chunk1_short_t24_l21.safetensors",
        config_name="t24_l21_student_context_distillation.yaml",
        num_output_frames=24,
        num_frame_per_block=1,
        local_attn_size=21,
    ),
    "chunk4_short": CheckpointPreset(
        name="chunk4_short",
        filename="chunk4_short_t21_l16.safetensors",
        config_name="self_forcing_dmd.yaml",
        num_output_frames=21,
        num_frame_per_block=4,
        local_attn_size=16,
    ),
    "chunk1_long": CheckpointPreset(
        name="chunk1_long",
        filename="chunk1_long_t126_l21.safetensors",
        config_name="t24_l21_rollout_context_distillation.yaml",
        num_output_frames=126,
        num_frame_per_block=1,
        local_attn_size=21,
        long_video=True,
    ),
    "chunk4_long": CheckpointPreset(
        name="chunk4_long",
        filename="chunk4_long_t121_l16.safetensors",
        config_name="self_forcing_dmd.yaml",
        num_output_frames=121,
        num_frame_per_block=4,
        local_attn_size=16,
        long_video=True,
    ),
    "chunk1_camera": CheckpointPreset(
        name="chunk1_camera",
        filename="chunk1_camera_control_t32_l21.safetensors",
        config_name="t32_l21_camera_student_distillation.yaml",
        num_output_frames=32,
        num_frame_per_block=1,
        local_attn_size=21,
        camera=True,
    ),
    "chunk4_camera": CheckpointPreset(
        name="chunk4_camera",
        filename="chunk4_camera_control_t29_l24.safetensors",
        config_name="self_forcing_dmd.yaml",
        num_output_frames=29,
        num_frame_per_block=4,
        local_attn_size=24,
        camera=True,
    ),
}

TEXT_ENCODER_ID = "nvidia/Cosmos-Reason1-7B"
VAE_REPO_ID = "nvidia/Cosmos-Predict2.5-2B"
VAE_FILENAME = "tokenizer.pth"
STUDENT_REPO_ID = "nvidia/cmd"
PIXEL_HEIGHT = 480
PIXEL_WIDTH = 832
FPS = 16
