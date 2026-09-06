# amd/ — run the V-JEPA quadruped world model on the AMD MI300X (ROCm) server

Single source of truth for reproducing + continuing the molab runs on the
private AMD cluster. Everything is plain PyTorch (SDPA) — no CUDA-only deps.

Reference results (molab): feature cosine 0.890/0.854 held-out (ICE-1/SPX-2),
control-signal +58%/+79%, ~40-mission corpus. Checkpoints:
`huggingface.co/rayaanoidpr/vjepa-wm-bigrun` (big_run_step4000.pt, ...8000.pt).

## 1. Environment (once)

```bash
conda create -n vjepa-amd python=3.12 -y && conda activate vjepa-amd
# ROCm PyTorch (adjust rocm version to your driver; MI300X uses gfx942)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e .            # repo deps (timm, einops, zarr<3, huggingface_hub, ...)
huggingface-cli login       # your HF token (needed only for checkpoint download)
git submodule update --init --recursive   # fetches vjepa2
```

Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
→ expect `True AMD Instinct MI300X ...`.

## 2. Checkpoints (once)

```bash
huggingface-cli download rayaanoidpr/vjepa-wm-bigrun \
  big_run_step8000.pt --local-dir ckpts
# optional earlier milestone: big_run_step4000.pt
```
`big_run.py` auto-resumes from the latest `ckpts/big_run_step*.pt` it finds.

## 3. Data (per fresh box)

```bash
python amd/pull_missions.py core      # ETH-1..3 + SPX-1 (train) + ICE-1/SPX-2 (held-out), full topics
python amd/pull_missions.py all       # + remaining 42 missions (background-able)
```
Public dataset `leggedrobotics/grand_tour_dataset`; idempotent (skips existing).

## 4. Training / eval order

```bash
# (a) main training — full recipe, resumable; target total steps
python amd/big_run.py 8000            # resumes from ckpts/big_run_step*.pt if present

# (b) probe suite on the final ckpt (transient/steady far displacement,
#     velocity control-signal, contact/bob hidden-state readouts)
python amd/probe.py --model ckpts/big_run_step8000.pt

# (c) decoder → pixels: command-conditioned decode + held-out PSNR + robot-mask
#     metrics (masks exist only for ETH-1) -> eval_decode.py
python amd/eval_decode.py

# (d) controllability ablations: multi-stream cmd+proprio model, per-stream
#     dropout, per-stream shuffle/novel-command tests -> eval_control.py
python amd/eval_control.py

# (e) causal-vs-random on the motion metric
python amd/train_random.py 1200       # matched random-mask predictor
python amd/probe.py --model ckpts/big_run_step8000.pt --vs-random ckpts/rand_pred1200.pt
```

Logs: `big_run_log.txt` etc. in the run dir; every stage prints progress —
on molab-style shells keep prints frequent (silent >5 min can drop streams).

## 5. Recipe (what the run does)

Frozen official V-JEPA 2.1 ViT-B/384 encoder + causal tubelet masking
(boundary randomized over {6,8,10}) + motion-weighted LN-L1 + per-window
command tokens (indices ≥ grid, frame coord 8+) + velocity-consistency aux
head (λ=0.5) + command dropout 25%. fp32 (bf16 RoPE has dtype bugs in this
stack), batch 4, stride 4 (400 ms @10 fps), 16-frame clips @384. Details and
all historical numbers: `molab/big_run_results.md`; paper-technique mapping:
repo README §2.7 + `SURVEYED_PAPERS.md`.

Notes:
- `ok()` gates missions on having images + command + state-estimator topics;
  `big_run.py` rebuilds its clip index every 500 steps, so background data
  pulls are picked up automatically.
- Save ckpts to HF periodically (`huggingface-cli upload`); sandbox sessions
  die — never keep the only copy on a node.
