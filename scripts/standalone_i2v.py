from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from nvidia_cmd.attention import resolve_backend
from nvidia_cmd.loader import load_cmd_pipeline
from nvidia_cmd.pipeline import generate_video, write_mp4
from nvidia_cmd.presets import FPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows Native standalone I2V for NVIDIA CMD (SDPA, no WSL)."
    )
    parser.add_argument("--image", required=True, help="RGB conditioning image")
    parser.add_argument("--prompt", default="", help="Text prompt")
    parser.add_argument("--prompt-file", default="", help="Optional prompt text file")
    parser.add_argument("--checkpoint", default="chunk1_short")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--camera-path", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention", default="sdpa", choices=["auto", "sdpa", "flash_attention"])
    parser.add_argument("--memory-preset", default="BALANCED")
    parser.add_argument("--output", default="outputs/cmd_i2v.mp4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-output-frames", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = resolve_backend(args.attention)
    import torch

    torch.set_grad_enabled(False)
    print(f"CMD attention backend={backend}")

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    prompt = prompt.strip()
    if not prompt:
        raise SystemExit("Prompt is empty. Pass --prompt or --prompt-file.")

    image = Image.open(args.image).convert("RGB")
    loaded = load_cmd_pipeline(
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path or None,
        dtype="bfloat16",
        device=args.device,
        attention_backend=args.attention,
        memory_preset=args.memory_preset,
        model_root=args.model_root or None,
    )
    print(f"Loaded {loaded.checkpoint_path} attention={loaded.attention_backend}", flush=True)
    print("CMD: starting generation", flush=True)
    frames = generate_video(
        loaded,
        image=image,
        prompt=prompt,
        seed=args.seed,
        camera_path=args.camera_path or None,
        num_output_frames=args.num_output_frames,
    )
    output = write_mp4(frames, args.output, fps=FPS)
    print(f"Wrote {output} frames={len(frames)} backend={loaded.attention_backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
