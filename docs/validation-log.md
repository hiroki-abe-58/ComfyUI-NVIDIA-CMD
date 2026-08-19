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
`workflows/cmd_i2v_basic.json` を同梱。pytest 10 passed。

## Phase 6

BALANCED: Reason1 は CPU、DiT と VAE は GPU、`torch.set_grad_enabled(False)`。  
実測は `docs/vram-measurements.jsonl`。

## Phase 7–8

`NVIDIACMDCameraControl` と `NVIDIACMDLongVideo`、workflow を追加。  
camera / long の実生成は追加 checkpoint 未配置のため未実施。

## Phase 9

v0.1.0。NOTICE に Built on NVIDIA Cosmos を記載。  
公式 CMD は NVIDIA OneWay Noncommercial のため、このリポジトリは public にしない。
