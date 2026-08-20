# Public readiness (Phase 9)

調査日: 2026-08-19

## Gate

| Check | Result |
| --- | --- |
| Import without model/CUDA build | pytest 10 passed |
| Loader / I2V nodes exist | implemented |
| Workflow `cmd_i2v_basic.json` | ComfyUI portable で mp4 成功 |
| Workflow `cmd_long_basic.json` | ComfyUI portable で mp4 成功（KV cap、peak 23852MiB） |
| Workflow `cmd_camera_control.json` | ComfyUI portable で mp4 成功 |
| NOTICE / LICENSE / Built on NVIDIA Cosmos | present |
| Official CMD license | NVIDIA OneWay Noncommercial (research/education only) |
| RTX 5090 standalone mp4 | `outputs/cmd_i2v.mp4` 93 frames, backend=sdpa |
| RTX 5090 ComfyUI mp4 | `outputs/comfyUI/` に 3 本 |
| flash-attn not required | confirmed (`flash_attn` is None) |
| README VRAM table | measured values only |

## Public judgment

**private のままにする。** 公式 CMD コードと学生重みは NVIDIA OneWay Noncommercial License で、研究・教育以外の利用ができない。public 化はこの制約を README 全面に出したうえで別判断する。

v0.1.0 は adapter 実装 + 5090 での standalone I2V 成功をもって private tag 可能。
