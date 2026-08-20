# ComfyUI-NVIDIA-CMD

Windows Native 向けの ComfyUI custom node です。NVIDIA CMD（Context-Matched Distillation）の因果 few-step I2V を、公式 Linux スタックや FlashAttention パッケージなしで呼び出します。

WSL 不要。`flash-attn` パッケージ不要。Attention は PyTorch SDPA です。

[English README](README.md)

**このプロジェクトは非公式です。** NVIDIA の公開物でも、推奨でも、サポート対象でもありません。[nv-tlabs/cmd](https://github.com/nv-tlabs/cmd) の checkout を呼ぶ adapter です。公式 CMD コードと学生重みは **NVIDIA OneWay Noncommercial License**（研究・教育のみ）です。[ライセンス](#ライセンス) を読んでください。

Built on NVIDIA Cosmos。バージョン **0.2.0**。

## Demo

入力画像（リポジトリ内は `outputs/comfyUI/sampleimage.png`。ComfyUI 上の名前は `car-red.png`）:

![CMD input image](outputs/comfyUI/sampleimage.png)

- Short I2V: [cmd_i2v_basic.mp4](outputs/comfyUI/cmd_i2v_basic.mp4) / [screenshot](screenshots/cmd_i2v_basic.png)
- Long video: [cmd_long_basic.mp4](outputs/comfyUI/cmd_long_basic.mp4) / [screenshot](screenshots/cmd_long_basic.png)
- Camera control: [cmd_camera_control.mp4](outputs/comfyUI/cmd_camera_control.mp4) / [screenshot](screenshots/cmd_camera_control.png)

standalone 成功例: [outputs/cmd_i2v.mp4](outputs/cmd_i2v.mp4)（93 frames, backend=`sdpa`）。

## このプロジェクトの位置づけ

CMD は新しい単体アーキテクチャではありません。Cosmos-Predict2.5-2B を因果 few-step 学生へ蒸留した重みです。論文: [arXiv:2608.13391](https://arxiv.org/abs/2608.13391)。この adapter は公式の Linux / FlashAttention 前提を必須にせず、Windows Native と Blackwell 向け PyTorch SDPA で学生推論を回します。

- Windows Native。WSL は不要
- RTX 50 / Blackwell: `flash-attn` wheel ではなく PyTorch SDPA
- short I2V / long / camera を ComfyUI node として提供
- 公式ソースは vendor しない。`CMD_UPSTREAM` または `third_party/cmd` を使う

公式推論コードは vendor しません。adapter 本体は `nvidia_cmd/` です（標準ライブラリ `cmd` との衝突を避けるため）。

## 機能

- Nodes: `NVIDIACMDModelLoader` / `NVIDIACMDImageToVideo` / `NVIDIACMDCameraControl` / `NVIDIACMDLongVideo` / `NVIDIACMDSaveVideo`
- BF16、832x480、16fps。`NVIDIACMDSaveVideo` が ComfyUI の `output/` へ mp4 を書く
- long は公式 KV が全 latent を持つと 32GB でも溢れるため、`local_attn_size` の環状バッファに制限する
- KSampler 互換にはしない。negative prompt / CFG は学生推論では使わない
- 重みの自動ダウンロードはしない

### 状態

検証済み（[検証環境](#検証環境)）:

- `chunk1_short` / `cmd_i2v_basic.json`
- `chunk1_long` / `cmd_long_basic.json`（126 latent / 約 501 pixel frames / KV=21）
- `chunk1_camera` / `cmd_camera_control.json` + `examples/identity_camera.npz`
- pytest（import / node mapping / workflow JSON / camera / KV cap / runtime guard の restore）
- `flash-attn` なし、backend=`sdpa`

未検証:

- Linux、RTX 40 系、下記以外の ComfyUI 構成
- `chunk4_*`（preset はあるが実測 workflow なし）
- FP8 / SageAttention / FlashAttention を必須パスにすること
- 公式 `utils` / `pipeline` import の完全 isolation

## 検証環境

実測のみ:

- Windows 11 Native
- RTX 5090 32GB / sm_120
- PyTorch 2.9.1+cu130（ComfyUI 側の Blackwell 対応 PyTorch。この node は `torch` を入れ替えない）
- ComfyUI Portable

[ComfyUI-Win-Blackwell](https://github.com/hiroki-abe-58/ComfyUI-Win-Blackwell) でも動作確認済み。**tested with** であり、required ではない。Blackwell 向け PyTorch が既に動いている ComfyUI なら使える。

`transformers` は ComfyUI 環境のものを使う。公式 CMD の `requirements.txt` は入れない。

## インストール

ComfyUI の `custom_nodes` に clone し、ComfyUI と同じ Python で adapter 依存だけ入れます。

```powershell
cd <ComfyUI>\custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-NVIDIA-CMD ComfyUI-NVIDIA-CMD
cd ComfyUI-NVIDIA-CMD
.\<ComfyUI-python> -m pip install -r requirements.txt
git clone https://github.com/nv-tlabs/cmd.git third_party\cmd
```

`flash-attn` / Transformer Engine / `natten` / 公式 Triton は入れません。

環境変数:

- `CMD_UPSTREAM`: 公式 `nv-tlabs/cmd` のルート
- `CMD_MODEL_ROOT`: 上記 `nvidia_cmd` ディレクトリ
- `COMFYUI_ROOT`: ComfyUI ルート（任意）

## モデル配置

巨大重みは手動で置きます。自動ダウンロードしません。

```text
<ComfyUI>/models/nvidia_cmd/
  transformer/
    chunk1_short_t24_l21.safetensors
    chunk1_long_t126_l21.safetensors
    chunk1_camera_control_t32_l21.safetensors
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

`Cosmos-Predict2.5-2B` は gated です。`tokenizer.pth` の前に Hugging Face で NVIDIA Open Model License に同意してください。

## Workflows

clone 後の追加作業は、モデル配置と公式 repo の場所だけです。`workflows/` の JSON を読み込みます。

### cmd_i2v_basic.json

`chunk1_short` + `NVIDIACMDImageToVideo` + `NVIDIACMDSaveVideo`。

### cmd_long_basic.json

`chunk1_long` + `NVIDIACMDLongVideo` + `NVIDIACMDSaveVideo`。公式 KV cache が全 latent を保持すると VRAM を溢して Windows が落ちます。adapter は `local_attn_size`（chunk1_long では 21）の環状バッファに制限します。

### cmd_camera_control.json

`chunk1_camera` + `identity_camera.npz` + `NVIDIACMDImageToVideo`。同梱は `examples/identity_camera.npz`。

## メモリ / ベンチマーク

以下は RTX 5090 32GB、BALANCED、ComfyUI Portable の実測です。推計ではありません。

BALANCED は text encode 後に Reason1 を外し、DiT で chunk 生成し、VAE は GPU に残して decode します。

- idle: 584 MiB used / 32607 MiB total（`nvidia-smi`）
- loaded: 4188 MiB torch allocated, BALANCED
- peak（8-frame probe）: 22709 MiB `max_memory_allocated`
- long peak（`cmd_long_basic`）: **23852 MiB**、約 **267 秒**、KV は **21**
- vae_decode: 22709 MiB（BALANCED では VAE を GPU に残す）
- プロセス終了後: 610 MiB

記録コマンド: `python scripts/record_vram.py --label idle`

## 構成メモ

- パッケージ名は `nvidia_cmd`（標準ライブラリ `cmd` を隠さない）
- 公式 CMD は tree にコピーしない。`CMD_UPSTREAM` または `third_party/cmd` を [nv-tlabs/cmd](https://github.com/nv-tlabs/cmd) に向ける
- 構築中だけのパッチ（`torch.compile` identity、Reason1 の `device_map=cpu`、ローカル `hf_hub_download`）は `CausalInferencePipeline` 構築後に戻す
- 推論は process 全体の `torch.set_grad_enabled(False)` ではなく `torch.inference_mode()`
- CMD モジュール限定のパッチ（`cosmos.runtime.attention` の SDPA、student `_load_model`、環状 KV）は同一プロセス内では残る

## 既知の問題

- `ensure_official_cmd_on_path` はまだ `sys.path.insert(0)`、汎用名（`utils` / `pipeline` / `wan` / `inference`）の `sys.modules` 削除、公式 checkout への `__init__.py` 書き込みを行う。0.2.0 では完全 isolation しない
- CMD ロード後、同一プロセス内の cosmos / 公式 CMD クラスは改変されたまま
- `TORCHDYNAMO_DISABLE` / `TORCH_COMPILE_DISABLE` は構築中だけ立てて戻す。同一プロセスで公式モジュールを後から再 import すると、loader を通さない限り compile が再び走る
- Linux / RTX 40 系 / Portable 以外の ComfyUI は未検証
- `chunk4_*` / FP8 / SageAttention はこのリリースの対象外

## ライセンス

- このリポジトリの adapter コード: Apache-2.0（`LICENSE`）
- 公式 CMD コードと学生重み: NVIDIA OneWay Noncommercial License（研究・教育のみ）
- Cosmos-Predict2.5 / Cosmos-Reason1 重み: NVIDIA Open Model License

詳細は [`NOTICE`](NOTICE) と [`LICENSE`](LICENSE)。

この adapter を使うこと自体が、公式 CMD や Cosmos 重みの商用許諾にはなりません。研究・教育以外で使う前に upstream のライセンスを読んでください。

## Roadmap

公開可能な 0.2.0 の次:

- 公式 import の isolation を狭める（汎用 `sys.modules` 削除をやめる）
- 機材があれば Linux / RTX 40 系の実測
- ComfyUI Registry 掲載は任意（Publisher ID はここでは作らない）
- chunk4 / FP8 / SageAttention は実測パスができてから

## Credits

Built on NVIDIA Cosmos.

- [NVIDIA CMD](https://github.com/nv-tlabs/cmd) / [arXiv:2608.13391](https://arxiv.org/abs/2608.13391)
- [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [Cosmos-Reason1](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)

## Standalone（ComfyUI UI なし）

```powershell
$env:CMD_UPSTREAM = "<repo>\third_party\cmd"
$env:CMD_MODEL_ROOT = "<ComfyUI>\models\nvidia_cmd"
python scripts\standalone_i2v.py --image examples\cmd_i2v_input.png --prompt-file examples\prompt.txt --attention sdpa --output outputs\cmd_i2v.mp4
```

成功条件: PowerShell から mp4 が 1 本出ること。ログは `backend=sdpa`。

## Validation

```powershell
python -m pytest tests
```

- Phase 1: `scripts/standalone_i2v.py` が Traceback なしで動画を出す
- Phase 2: `flash-attn` なし、ログが `backend=sdpa`
- ComfyUI Portable で 3 workflow が Queue 成功し、`outputs/comfyUI/` に mp4 が出ること

一次情報の整理は [`docs/upstream-inventory.md`](docs/upstream-inventory.md) にあります。
