# Public readiness

調査日: 2026-08-20

v0.2.0 adapter。ライセンスと unofficial disclaimer を README 前面に出せば **Public にしてよい状態**。visibility の変更はリポジトリ所有者が行う。この文書は判断基準であり、Public ボタンを押す操作ではない。

## Gate

- Import without model/CUDA build: pytest passed（runtime guard を含む）
- Loader / I2V / Long / Camera / Save Video nodes: implemented
- Workflow `cmd_i2v_basic.json`: ComfyUI Portable で mp4 成功
- Workflow `cmd_long_basic.json`: ComfyUI Portable で mp4 成功（KV cap、peak 23852MiB）
- Workflow `cmd_camera_control.json`: ComfyUI Portable で mp4 成功
- NOTICE / LICENSE / Built on NVIDIA Cosmos: present
- Official CMD license: NVIDIA OneWay Noncommercial (research/education only)
- README 先頭付近に unofficial disclaimer と OneWay Noncommercial がある
- README.md 英語、README_ja.md 日本語、相互リンク
- RTX 5090 standalone mp4: `outputs/cmd_i2v.mp4` 93 frames, backend=sdpa
- RTX 5090 ComfyUI mp4: `outputs/comfyUI/` に 3 本
- flash-attn not required: confirmed
- README VRAM numbers: measured values only
- Tested environment: Windows 11 Native, RTX 5090 32GB sm_120, PyTorch 2.9.1+cu130, ComfyUI Portable
- ComfyUI-Win-Blackwell: tested with, not required
- Process-wide `torch.compile` / HF / transformers / grad patches: scoped to construct / inference_mode
- Known Issues: README に記載
- version: `0.2.0` in `pyproject.toml` and README

## Public judgment

**license を前面に出せば Public 可。** 公式 CMD コードと学生重みは NVIDIA OneWay Noncommercial License で、研究・教育以外の利用はできない。この制約は README / NOTICE に出した。GitHub visibility と Release 作成はユーザーが行う。

残リスク（Public を止めないが、README Known Issues に書いたもの）:

- 公式 checkout への `sys.path.insert(0)` と generic `sys.modules` 削除
- CMD ロード後、同一プロセス内の cosmos / CMD モジュールは改変されたまま
- Linux / RTX 40 / 他 ComfyUI 構成は未検証
