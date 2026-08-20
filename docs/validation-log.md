# Validation log

測定日: 2026-08-19  
環境: Windows 11 Native, RTX 5090 32GB, ComfyUI portable PyTorch 2.9.1+cu130, WSL なし, flash-attn なし

## Phase 1

`scripts/standalone_i2v.py` が `outputs/cmd_i2v.mp4` を出力した。  
checkpoint `chunk1_short_t24_l21.safetensors`, 24 latent frames → 93 pixel frames, 801960 bytes, backend=sdpa。

## Phase 2

`flash_attn` は未インストール。ログは `CMD attention backend=sdpa` と `backend=sdpa`。

## Phase 3–5

`NVIDIACMDModelLoader` / `NVIDIACMDImageToVideo` を実装。import 時に重みを読まない。  
`workflows/cmd_i2v_basic.json` を同梱。pytest は import / mapping / workflow JSON を担保。

## Phase 6

BALANCED: Reason1 は CPU、DiT と VAE は GPU。勾配は process 全体の `torch.set_grad_enabled(False)` ではなく、構築と生成を `torch.inference_mode()` に限定する。  
実測は `docs/vram-measurements.jsonl`。

## Phase 7–8

`NVIDIACMDCameraControl` と `NVIDIACMDLongVideo`、workflow を追加。

## Phase 9

v0.1.0。NOTICE に Built on NVIDIA Cosmos を記載。  
当時は OneWay Noncommercial を前面に出せていなかったため private のままにしていた。

## ComfyUI portable (2026-08-20)

ComfyUI Windows Portable で 3 workflow が Queue 成功。  
入力画像は `outputs/comfyUI/sampleimage.png`（UI 上は `car-red.png`）。  
画面は `screenshots/`、mp4 は `outputs/comfyUI/`。

- `cmd_i2v_basic.json` / `chunk1_short` → `outputs/comfyUI/cmd_i2v_basic.mp4`
- `cmd_long_basic.json` / `chunk1_long` → `outputs/comfyUI/cmd_long_basic.mp4`。peak allocated=23852MiB、約 267s。KV は `local_attn_size=21` に制限
- `cmd_camera_control.json` / `chunk1_camera` + `identity_camera.npz` → `outputs/comfyUI/cmd_camera_control.mp4`

`NVIDIACMDSaveVideo` が ComfyUI `output/` に 16fps mp4 を書く。

## Public release hardening (2026-08-20)

`nvidia_cmd/runtime_guard.py` で `torch.compile` / Reason1 `from_pretrained` / `hf_hub_download` を pipeline 構築中だけに限定し、終了後に restore。  
`generate_video` と standalone は `torch.inference_mode()`。CMD モジュール限定パッチ（SDPA / student load / circular KV）は復元しない。  
README.md を英語化し、日本語は README_ja.md へ移した。
