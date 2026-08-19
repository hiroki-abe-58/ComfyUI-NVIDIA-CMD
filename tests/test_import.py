from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_package_import_does_not_require_upstream():
    module = importlib.import_module("nvidia_cmd")
    assert module.CHECKPOINT_PRESETS["chunk1_short"].filename.endswith(".safetensors")


def test_comfy_node_mappings_exist():
    nodes = importlib.import_module("nodes")
    assert "NVIDIACMDModelLoader" in nodes.NODE_CLASS_MAPPINGS
    assert "NVIDIACMDImageToVideo" in nodes.NODE_CLASS_MAPPINGS
    assert nodes.NVIDIACMDModelLoader.FUNCTION == "load"


def test_root_init_exports_mappings():
    spec = importlib.util.spec_from_file_location(
        "comfyui_nvidia_cmd_root",
        ROOT / "__init__.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "NVIDIACMDModelLoader" in module.NODE_CLASS_MAPPINGS
