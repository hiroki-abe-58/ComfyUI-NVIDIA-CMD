from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

import torch

from nvidia_cmd.runtime_guard import scoped_local_hf_hub, scoped_torch_compile_identity


def test_compile_identity_restores_original():
    original = torch.compile
    with scoped_torch_compile_identity():
        assert torch.compile is not original
        fn = torch.compile(lambda value: value + 1)
        assert fn(2) == 3
    assert torch.compile is original


def test_compile_guard_restores_env_when_unset():
    os.environ.pop("TORCHDYNAMO_DISABLE", None)
    os.environ.pop("TORCH_COMPILE_DISABLE", None)
    with scoped_torch_compile_identity():
        assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
        assert os.environ["TORCH_COMPILE_DISABLE"] == "1"
    assert "TORCHDYNAMO_DISABLE" not in os.environ
    assert "TORCH_COMPILE_DISABLE" not in os.environ


def test_compile_guard_restores_preexisting_env():
    os.environ["TORCHDYNAMO_DISABLE"] = "0"
    os.environ["TORCH_COMPILE_DISABLE"] = "0"
    try:
        with scoped_torch_compile_identity():
            assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
        assert os.environ["TORCHDYNAMO_DISABLE"] == "0"
        assert os.environ["TORCH_COMPILE_DISABLE"] == "0"
    finally:
        os.environ.pop("TORCHDYNAMO_DISABLE", None)
        os.environ.pop("TORCH_COMPILE_DISABLE", None)


def test_inference_mode_does_not_leave_grad_disabled():
    assert torch.is_grad_enabled()
    with torch.inference_mode():
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


def test_local_hf_hub_restores_download():
    huggingface_hub = pytest.importorskip("huggingface_hub")
    original = huggingface_hub.hf_hub_download
    with scoped_local_hf_hub(ROOT):
        assert huggingface_hub.hf_hub_download is not original
    assert huggingface_hub.hf_hub_download is original
