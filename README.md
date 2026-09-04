# V-JEPA 2.1 — Grand Tour (ANYmal-D) fine-tuning notebook

Self-contained pipeline for fine-tuning [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2)
on the [ANYmal-D / Grand Tour dataset](https://grand-tour.leggedrobotics.com/)
(HF: `leggedrobotics/grand_tour_dataset`) and decoding predicted latents back to
pixels. Runs as a single marimo notebook cell or as a plain script.

> Research plan, surveyed-papers digest, and experimental code (goal-conditioned
> world model, action conditioning, etc.) live on the
> **[`experiments` branch](../../tree/experiments)** — this branch intentionally
> contains only the notebook and its dependencies.

## 1. How to run

### Prerequisites
- Python 3.12+, PyTorch 2.1+ with a GPU (CUDA **or** ROCm — the code is pure
  PyTorch + `timm` + `einops`, bf16 autocast is auto-selected when supported),
  plus `numpy`, `Pillow`, `matplotlib`, `huggingface_hub`, `timm`, `einops`.
- **HuggingFace account with Grand Tour access** (gated — request
  [here](https://grand-tour.leggedrobotics.com/)): `huggingface-cli login`.
- Internet on first run: the pipeline downloads the official V-JEPA 2.1
  ViT-B/384 checkpoint (~2 GB) into `~/.cache/torch/hub/checkpoints`.

### Clone
```bash
git clone --recursive <repo-url>   # fetches the pinned vjepa2 submodule
# if already cloned:  git submodule update --init --recursive
```
`vjepa2/` is the only pinned submodule and is imported at runtime
(`app.vjepa_2_1` / `src`); the pipeline auto-clones it if missing.

### Run the pipeline
`grandtour_vjepab_pipeline.py` is a straight-line, resumable script. **Paste its
entire contents into a single marimo Python cell and run it**, or run headless:

```bash
marimo edit          # notebook workflow
python grandtour_vjepab_pipeline.py   # or as a plain script
```

What it does (re-running skips completed stages):

| Stage | What | Outputs |
|---|---|---|
| Setup | download official ViT-B/384 checkpoint | `~/.cache/torch/hub/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt` |
| Data | pull missing missions from HF, build 64-frame clips (stride 1, step 32) | in-memory clip records |
| Stage 1 | fine-tune encoder + predictor with the V-JEPA masked-latent objective (10 epochs, `lr 1e-5`, EMA target `0.996`) | `vjepa21_grandtour_stage1_{epoch,best}.pth` |
| Stage 2 | predict the latent of frame `t+8` from frame `t`, decode to 384×384 RGB | `stage2_*.pth`, `vjepa21_grandtour_final_sharp_decoder.pth` |

Configuration: constants at the top of the file and env vars
(`VJEPA_ROOT`, `GRANDTOUR_DATA_ROOT`, `CHECKPOINT_DIR`). Key knobs:
`MISSIONS[:k]` for a smoke subset, `MAX_CLIPS_PER_MISSION=300`,
`BATCH_SIZE=4` (96 GB GPU at 64×384², bf16), `PRECOMPUTE_STAGE2_LATENTS=True`
(one-time latent cache, ~21 GB disk, ~2–4× faster Stage 2),
`STAGE1_BEST_DRIVE_PATH` / `STAGE1_BEST_GDRIVE_ID` / `CONTINUE_STAGE1` /
`RESUME_STAGE2` for resuming from existing checkpoints.

GPU notes: NVIDIA molab (RTX Pro 6000, 96 GB) and AMD MI300X (ROCm) both work
unchanged; bf16 is used when available, the fp16 `GradScaler` path otherwise.
On molab's 12 h session cap, every stage is checkpoint-resumable.

## 2. Repository layout

```
grandtour_vjepab_pipeline.py   the entire pipeline (marimo cell / plain script)
vjepa2/                        submodule: facebookresearch/vjepa2 (model code)
```

Gotchas: the released checkpoint key is `ema_encoder` (predictor head is
re-initialised to the encoder embed dim, 768 for ViT-B);
`torch.hub.load('facebookresearch/vjepa2', ...)` is currently broken — loading
is handled inside the pipeline; Grand Tour images are 1920×1280 HDR (resize →
crop to 384, no horizontal flip); only missions with `hdr_front` footage and
enough consecutive frames are usable (see `MISSIONS` in the file).
