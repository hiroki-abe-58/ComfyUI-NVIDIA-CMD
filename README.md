# ComfyUI-NVIDIA-CMD

Windows 11 Native 向けの ComfyUI custom node です。NVIDIA CMD（Context-Matched Distillation）の因果 few-step I2V を、公式 Linux スタックや FlashAttention なしで呼び出します。

Built on NVIDIA Cosmos

公式推論コードは vendor しません。`CMD_UPSTREAM` または `third_party/cmd` に [nv-tlabs/cmd](https://github.com/nv-tlabs/cmd) を置いて使います。adapter 本体は `nvidia_cmd/` です（標準ライブラリ `cmd` との衝突を避けるため）。

## 1. Status

v0.2.0 adapter。BF16 + SDPA + 832x480。  
Windows 11 + RTX 5090 + ComfyUI portable で 3 workflow の mp4 生成を確認済み。

- standalone: `outputs/cmd_i2v.mp4`（93 frames, backend=sdpa）
- ComfyUI: `outputs/comfyUI/cmd_i2v_basic.mp4` / `cmd_long_basic.mp4` / `cmd_camera_control.mp4`

## 2. What this is

CMD は新しい単体アーキテクチャではなく、Cosmos-Predict2.5-2B を因果 few-step 学生へ蒸留した重みです。論文: [arXiv:2608.13391](https://arxiv.org/abs/2608.13391)。

## 3. Requirements

- Windows 11 Native（WSL 不要）
- NVIDIA GPU。検証対象は RTX 5090 32GB / sm_120
- 既存 ComfyUI の Blackwell 対応 PyTorch。この node は `torch` を入れ替えません
- 公式リポジトリ [nv-tlabs/cmd](https://github.com/nv-tlabs/cmd)
- 手動配置した重み（自動ダウンロードしません）

## 4. Install

ComfyUI の `custom_nodes` に clone し、ComfyUI と同じ Python で adapter 依存だけ入れます。

```powershell
cd <ComfyUI>\custom_nodes
git clone <this-repo> ComfyUI-NVIDIA-CMD
cd ComfyUI-NVIDIA-CMD
.\<ComfyUI-python> -m pip install -r requirements.txt
git clone https://github.com/nv-tlabs/cmd.git third_party\cmd
```

`flash-attn` / Transformer Engine / `natten` / 公式 Triton は入れません。

## 5. Model layout

巨大重みは手動で置きます。

```text
<ComfyUI>/models/nvidia_cmd/
  transformer/
    chunk1_short_t24_l21.safetensors
    chunk1_long_t126_l21.safetensors          # long workflow
    chunk1_camera_control_t32_l21.safetensors # camera workflow
  text_encoder/          # nvidia/Cosmos-Reason1-7B
  vae/tokenizer.pth      # または Wan2.1_VAE.pth
```

```powershell
hf download nvidia/cmd chunk1_short_t24_l21.safetensors --local-dir <ComfyUI>\models\nvidia_cmd\transformer
hf download nvidia/Cosmos-Reason1-7B --local-dir <ComfyUI>\models\nvidia_cmd\text_encoder
hf download nvidia/Cosmos-Predict2.5-2B tokenizer.pth --local-dir <ComfyUI>\models\nvidia_cmd\vae
# Predict2.5 is gated. Public fallback (same Wan2.1 VAE mean/std):
hf download ali-vilab/VACE-Wan2.1-1.3B-Preview Wan2.1_VAE.pth --local-dir <ComfyUI>\models\nvidia_cmd\vae
```

環境変数:

- `CMD_UPSTREAM`: 公式 `nv-tlabs/cmd` のルート
- `CMD_MODEL_ROOT`: 上記 `nvidia_cmd` ディレクトリ
- `COMFYUI_ROOT`: ComfyUI ルート（任意）

## 6. Nodes

- `NVIDIACMDModelLoader`
- `NVIDIACMDImageToVideo`
- `NVIDIACMDCameraControl`（camera checkpoint + NPZ。同梱は `examples/identity_camera.npz`）
- `NVIDIACMDLongVideo`（`chunk1_long` / `chunk4_long`）
- `NVIDIACMDSaveVideo`（16fps mp4。ComfyUI の `output/` へ保存）

KSampler 互換にはしません。negative prompt / CFG は学生推論では使いません。

long は公式 KV cache が全 latent を保持すると VRAM を溢して Windows が落ちます。adapter は `local_attn_size`（chunk1_long では 21）の環状バッファに制限します。

## 7. Workflows

clone 後はモデル配置と公式 repo の場所だけが追加作業です。  
入力画像の実体は `outputs/comfyUI/sampleimage.png`（ComfyUI 上では `car-red.png` として読み込み）。

入力画像:

![CMD input image](outputs/comfyUI/sampleimage.png)

### cmd_i2v_basic.json

`chunk1_short` + `NVIDIACMDImageToVideo` + `NVIDIACMDSaveVideo`。  
結果: [outputs/comfyUI/cmd_i2v_basic.mp4](outputs/comfyUI/cmd_i2v_basic.mp4)

![cmd_i2v_basic ComfyUI](screenshots/cmd_i2v_basic.png)

### cmd_long_basic.json

`chunk1_long` + `NVIDIACMDLongVideo` + `NVIDIACMDSaveVideo`。  
126 latent / 約 501 pixel frames / 16fps。RTX 5090 で peak allocated=23852MiB。  
結果: [outputs/comfyUI/cmd_long_basic.mp4](outputs/comfyUI/cmd_long_basic.mp4)

![cmd_long_basic ComfyUI](screenshots/cmd_long_basic.png)

### cmd_camera_control.json

`chunk1_camera` + `identity_camera.npz` + `NVIDIACMDImageToVideo`。  
結果: [outputs/comfyUI/cmd_camera_control.mp4](outputs/comfyUI/cmd_camera_control.mp4)

![cmd_camera_control ComfyUI](screenshots/cmd_camera_control.png)

## 8. Standalone (no ComfyUI UI)

```powershell
$env:CMD_UPSTREAM = "E:\ComfyUI-NVIDIA-CMD\third_party\cmd"
$env:CMD_MODEL_ROOT = "E:\ComfyUI-NVIDIA-CMD\models\nvidia_cmd"
python scripts\standalone_i2v.py --image examples\cmd_i2v_input.png --prompt-file examples\prompt.txt --attention sdpa --output outputs\cmd_i2v.mp4
```

成功条件: PowerShell から mp4 が 1 本出ること。backend ログは `sdpa`。

## 9. VRAM

BALANCED は text encode 後に Reason1 を外し、DiT で chunk 生成し、VAE decode します。README の数値は実測のみです。

| Stage | used MiB | total MiB | GPU | Notes |
| --- | --- | --- | --- | --- |
| idle | 584 | 32607 | RTX 5090 | nvidia-smi |
| loaded | 4188 | 32607 | RTX 5090 | torch allocated, BALANCED |
| peak | 22709 | 32607 | RTX 5090 | max_memory_allocated, 8-frame probe |
| long peak | 23852 | 32607 | RTX 5090 | ComfyUI `cmd_long_basic`, KV capped to 21 |
| vae_decode | 22709 | 32607 | RTX 5090 | BALANCED では VAE を GPU に残す |
| after | 610 | 32607 | RTX 5090 | プロセス終了後 |

記録コマンド: `python scripts/record_vram.py --label idle`

## 10. License

- このリポジトリの adapter コード: Apache-2.0
- 公式 CMD コードと学生重み: NVIDIA OneWay Noncommercial License（研究・教育のみ）
- Cosmos-Predict2.5 / Cosmos-Reason1 重み: NVIDIA Open Model License
- 詳細は `NOTICE` と `LICENSE`

public 化は、import / load / 5090 生成 / workflow / ライセンス確認が実測で通ってから判断します。

## 11. Attention

既定は PyTorch SDPA。`flash_attention` は検出できたときだけ選べます。Windows Native / RTX 50 では SDPA を使ってください。

## 12. Validation

- `python -m pytest tests`
- Phase 1: `scripts/standalone_i2v.py` が Traceback なしで動画を出す
- Phase 2: `flash-attn` なし、ログが `backend=sdpa`
- ComfyUI portable で 3 workflow が Queue 成功し、`outputs/comfyUI/` に mp4 が出ること

一次情報の整理は `docs/upstream-inventory.md` にあります。
