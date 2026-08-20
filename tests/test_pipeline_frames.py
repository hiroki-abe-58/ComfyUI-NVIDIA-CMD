from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from nvidia_cmd.pipeline import frames_to_comfy_image


def test_frames_to_comfy_image_converts_uint8_nhwc():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[1, 0, 0] = 255
    tensor = frames_to_comfy_image(frames)
    assert tuple(tensor.shape) == (2, 8, 8, 3)
    assert tensor.dtype.itemsize == 4
    assert float(tensor[1, 0, 0, 0]) == 1.0
