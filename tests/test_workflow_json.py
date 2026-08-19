from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_basic_workflow_has_loader_and_i2v():
    data = json.loads((ROOT / "workflows" / "cmd_i2v_basic.json").read_text(encoding="utf-8"))
    types = {node["type"] for node in data["nodes"]}
    assert "NVIDIACMDModelLoader" in types
    assert "NVIDIACMDImageToVideo" in types


def test_camera_workflow_has_camera_node():
    data = json.loads((ROOT / "workflows" / "cmd_camera_control.json").read_text(encoding="utf-8"))
    types = {node["type"] for node in data["nodes"]}
    assert "NVIDIACMDCameraControl" in types
    assert data["nodes"][0]["widgets_values"][0] == "chunk1_camera"


def test_long_workflow_uses_long_checkpoint():
    data = json.loads((ROOT / "workflows" / "cmd_long_basic.json").read_text(encoding="utf-8"))
    types = {node["type"] for node in data["nodes"]}
    assert "NVIDIACMDLongVideo" in types
    assert data["nodes"][0]["widgets_values"][0] == "chunk1_long"
