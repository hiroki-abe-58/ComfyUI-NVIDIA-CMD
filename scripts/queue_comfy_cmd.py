from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT_TEXT = (
    "A cinematic tracking shot of a red sports car driving along a wet coastal "
    "road at dusk, reflections on the asphalt, detailed motion, natural lighting."
)


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    last_error: Exception | None = None
    for attempt in range(8):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 15))
    raise last_error if last_error is not None else RuntimeError(f"request failed: {url}")


def wait_until_up(base: str, attempts: int = 60) -> None:
    last_error = None
    for _ in range(attempts):
        try:
            request_json(f"{base}/system_stats")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"ComfyUI is not reachable at {base}: {last_error}")


def build_prompt(kind: str, camera_npz: str) -> dict:
    loader = {
        "class_type": "NVIDIACMDModelLoader",
        "inputs": {
            "checkpoint": "chunk1_long" if kind == "long" else "chunk1_camera",
            "dtype": "bfloat16",
            "device": "cuda",
            "attention": "sdpa",
            "memory_preset": "BALANCED",
            "checkpoint_path": "",
            "model_root": "",
        },
    }
    image = {"class_type": "LoadImage", "inputs": {"image": "cmd_i2v_input.png"}}
    generate_type = "NVIDIACMDLongVideo" if kind == "long" else "NVIDIACMDImageToVideo"
    generate = {
        "class_type": generate_type,
        "inputs": {
            "cmd_model": ["1", 0],
            "image": ["2", 0],
            "prompt": PROMPT_TEXT,
            "seed": 0,
        },
    }
    preview = {"class_type": "PreviewImage", "inputs": {"images": ["4" if kind == "camera" else "3", 0]}}
    graph = {"1": loader, "2": image}
    if kind == "camera":
        graph["3"] = {
            "class_type": "NVIDIACMDCameraControl",
            "inputs": {"camera_npz": camera_npz},
        }
        generate["inputs"]["camera"] = ["3", 0]
        graph["4"] = generate
        graph["5"] = preview
    else:
        graph["3"] = generate
        graph["4"] = preview
    return graph


def queue_and_wait(base: str, prompt: dict, timeout_s: int) -> dict:
    queued = request_json(f"{base}/prompt", {"prompt": prompt})
    prompt_id = queued["prompt_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        history = request_json(f"{base}/history/{prompt_id}", timeout=60)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(5)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {timeout_s}s")


def assert_success(kind: str, result: dict) -> None:
    status = result.get("status") or {}
    messages = status.get("messages") or []
    errors = [item for item in messages if item and item[0] == "execution_error"]
    if errors:
        raise RuntimeError(f"{kind} failed: {errors[0]}")
    if status.get("status_str") == "error" or status.get("completed") is False:
        raise RuntimeError(f"{kind} failed: {json.dumps(status, ensure_ascii=False)}")
    outputs = result.get("outputs") or {}
    if not outputs:
        raise RuntimeError(f"{kind} produced no outputs: {json.dumps(result, ensure_ascii=False)[:800]}")
    print(f"{kind} ok outputs={list(outputs)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8188")
    parser.add_argument("--kind", choices=["long", "camera", "both"], default="both")
    parser.add_argument("--camera-npz", default="")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    camera_npz = args.camera_npz or str(
        Path(__file__).resolve().parents[1] / "examples" / "identity_camera.npz"
    )
    wait_until_up(args.base)
    kinds = ["long", "camera"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        print(f"queue {kind}", flush=True)
        result = queue_and_wait(args.base, build_prompt(kind, camera_npz), args.timeout)
        assert_success(kind, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
