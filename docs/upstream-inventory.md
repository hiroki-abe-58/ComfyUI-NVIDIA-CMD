# Upstream inventory (Phase 0)

調査日: 2026-08-19  
一次情報のみ。仮説は明示する。

## Official artifacts

### Code

- Repository: https://github.com/nv-tlabs/cmd
- Default branch: `main`
- Entry point: `inference.py`
- Few-step pipeline: `pipeline/causal_inference.py` (`CausalInferencePipeline`)
- Model factory: `utils/model_factory.py`
- Cosmos adapter: `cosmos/wrapper.py`
- Causal DiT: `cosmos/causal_model.py`
- Attention runtime: `cosmos/runtime.py`
- Camera: `cosmos/camera_conditioning.py`
- Preferred OS in HF card: Linux
- Install README requires `flash-attn`, but `requirements.txt` does **not** pin it.

### Weights

- Hugging Face: https://huggingface.co/nvidia/cmd
- Gated: false
- License metadata: `nvidia-oneway-noncommercial`
- License link: https://github.com/nv-tlabs/cmd/blob/main/LICENSE
- Last modified (API): 2026-08-18T14:14:56Z
- Used storage (API): 75,187,921,676 bytes

| File | Role |
| --- | --- |
| `chunk1_short_t24_l21.safetensors` | MVP student |
| `chunk4_short_t21_l16.safetensors` | chunk-4 short |
| `chunk1_long_t126_l21.safetensors` | chunk-1 long |
| `chunk4_long_t121_l16.safetensors` | chunk-4 long |
| `chunk1_camera_control_t32_l21.safetensors` | chunk-1 camera |
| `chunk4_camera_control_t29_l24.safetensors` | chunk-4 camera |
| `chunk1_teacher_t24_l21.safetensors` | teacher (not for MVP inference) |
| `chunk4_teacher_t21_l16.safetensors` | teacher |
| `chunk1_teacher_t32_l21_camera.safetensors` | camera teacher |
| `chunk4_teacher_t29_l24_camera.safetensors` | camera teacher |

### License (confirmed)

- Official code + weights: **NVIDIA OneWay Noncommercial License**
- Use limitation: non-commercial research or educational purposes only
- NOTICE also attributes Apache-2.0 code from Self-Forcing, Wan2.1, Cosmos-Predict2.5
- Predict2.5 / Reason1 model weights remain **NVIDIA Open Model License**
- This repository's original adapter code is Apache-2.0 and must remain separable from official CMD sources

## Confirmed inference contract

### Components

| Component | Hugging Face ID | Notes |
| --- | --- | --- |
| Student DiT | `nvidia/cmd` safetensors | keys are bare `blocks.0...`; loader prefixes `model.` |
| Text encoder | `nvidia/Cosmos-Reason1-7B` | `Qwen2_5_VLForConditionalGeneration`; concatenates normalized hidden states of all 28 language layers → 28 * 3584 = 100352 channels |
| VAE | `nvidia/Cosmos-Predict2.5-2B` file `tokenizer.pth` | Wan2.1 VAE, z_dim=16, packaged mean/std in `CosmosVAEWrapper` |
| Base architecture | Cosmos-Predict2.5-2B | 28 blocks, 16 heads, 2048 channels, patch 2x1, in/out 16 |

### I2V geometry (official comments)

- Pixel: **480 x 832**, 16 fps
- Latent: **16 x 60 x 104** (spatial 8x, temporal 4x)
- 93 pixel frames ↔ 24 latent frames
- Independent first frame: clean I2V latent at t=0, then generated frames
- `chunk1_short`: 24 latent frames, block size 1, local attention 21
- `chunk4_short`: 21 latent frames, block size 4, local attention 16
- `chunk1_long`: 126 latent frames → 501 pixel frames
- Camera: pixel cameras every 4 frames (`camera_frame_stride=4`), patch 16, NPZ keys `target_w2c` and `target_intrinsics`

### Sampler

- Student uses `denoising_step_list: [1000, 750, 500, 250]` (4 steps)
- `warp_denoising_step: true`
- `timestep_shift: 5.0`
- `context_noise: 128` when committing KV
- `guidance_scale` is used only when `num_inference_steps > 0` (multi-step teacher path)
- **Student few-step path does not apply CFG.** Negative prompt is unused at student inference.

### Attention (confirmed in source)

- Causal student `atten_backend` is `"i4"`
- Full-sequence causal mask uses **PyTorch FlexAttention**
- Streaming KV path calls `i4_attention_op` → `cosmos.runtime.attention`
- `cosmos/runtime.py` uses Transformer Engine if imported and CUDA; otherwise **`torch.nn.functional.scaled_dot_product_attention`**
- `torch_attention_op` is an explicit SDPA helper
- `flash-attn` is README-only; not a Python import hard-fail in `runtime.py`
- `natten` / Megatron / context-parallel paths exist but are unused for the 2B causal I2V config

### Checkpoint key mapping (from `inference.py`)

Released safetensors contain bare DiT keys such as `blocks.0...`.  
Loader remaps to `model.<key>` on `pipeline.generator`.  
Optional missing buffers allowed:

- `model.accum_video_sample_counter`
- `model.accum_image_sample_counter`
- `model.accum_iteration`
- `model.accum_train_in_hours`

**Hypothesis (not verified by opening a local `.safetensors` in this session):** keys match CausalCosmosModel / MinimalV1LVGDiT, not Diffusers `CosmosTransformer3DModel` 1:1. Phase 1 must load through official `CausalInferencePipeline`, not Diffusers.

## Official inference found?

**Yes.** Phase 1 must wrap `nv-tlabs/cmd` `inference.py` / `CausalInferencePipeline` with an SDPA-first attention patch. Do not vendor the whole repo. Do not depend on `cosmos-predict2.5` as a Python package.

## Windows / RTX 5090 notes

- Official stack prefers Linux and optionally flash-attn / Transformer Engine
- SDPA fallback is already in official `runtime.py`
- This project must force SDPA (`CMD_ATTENTION_BACKEND=sdpa`) so TE/flash-attn are never required
- Do not pin `torch` in this custom node's requirements
- FlexAttention + `torch.compile` may be fragile on Windows; Phase 1 should prefer the streaming KV path used by official I2V inference
