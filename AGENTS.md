# AGENTS.md — onboarding + continuation guide for AI agents

Goal (one line): **build the best world model for quadruped locomotion** — a
latent world-action model over ANYmal-D/Grand Tour data that predicts future
latents given the past + commanded motion, tracks hidden state, and is
decodable/steerable for planning.

Read first: `README.md` (§1 how to run, §2 plan, §3 surveyed papers) and
`SURVEYED_PAPERS.md`. Historical results/logs: `molab/`. Runnable package for
GPU servers: `amd/`. This file is the operational cheat-sheet for agents.

## Repository layout (branch `experiments` = active work; `main` = notebook only)

- `grandtour_vjepab_pipeline.py` — legacy single-cell V-JEPA 2.1 fine-tune (marimo).
- `vjepa_nav/` — packaged goal-conditioned WM + DiT decoder (older design; reference).
- `amd/` — **current production code** (portable, ROCm/CUDA):
  - `big_run.py` — main trainer: frozen V-JEPA 2.1 ViT-B/384 encoder, causal
    tubelet masking (boundary {6,8,10}), motion-weighted LN-L1, per-window
    command tokens (indices ≥ grid), velocity-consistency aux head (λ=0.5),
    command dropout 25%. fp32, batch 4, stride 4, 16×384² clips. Resumable
    (auto-loads latest `ckpts/big_run_step*.pt`), clip index rebuilds every 500
    steps, retry-on-truncated-jpeg sampler. Eval held-out ICE-1/SPX-2 every 200
    steps (cos + velRMSE true-vs-shuffled commands).
  - `pull_missions.py` — idempotent HF data pull (`core` / `all` / timestamps).
  - `probe.py` — readout probes on final ckpt: transient/steady far-horizon
    displacement vs copy-last/teacher; bob; `--vs-random` / `--train-random`.
  - `eval_decode.py` — command-conditioned decoder (teacher/causal/true-vs-shuf
    commands), PSNR/SSIM, robot-mask + robot-motion metrics (ETH-1 only).
  - `eval_control.py` — multi-stream (cmd+proprio) controllability model:
    per-stream dropout, per-stream shuffle / zero / reversed / halved commands.
  - `setup.sh` — uv-based env setup (auto NVIDIA/ROCm).
- `configs/missions_split.csv` — LiMo 40/2/6 split (train/val/test env names).
- `vjepa2/` — submodule (facebookresearch); imported as `app.vjepa_2_1`.

## Key numbers (baseline for comparisons — always report against these)

Frozen-encoder causal run, ~40 missions, step 8000:
- held-out cosine: **ICE-1 0.890 | SPX-2 0.854**
- velocity-readout control-signal (velRMSE true→shuffled cmds): 0.061→0.097
  (+58%), 0.052→0.092 (+79%); re-sampled ICE +105%.
- far-horizon (3 s) displacement readout (m) beats copy-last everywhere:
  ICE steady 0.344 vs 0.608; SPX steady 0.493 vs 0.709; transient ~0.72-0.91.
- Metrics: `cos` = LN-cosine vs frozen official teacher on causal query b=8;
  velRMSE = velocity readout of predicted latents under true vs shuffled cmds;
  displacement readouts use linear heads fit on ETH-1 teacher features.

Checkpoints: HF `rayaanoidpr/vjepa-wm-bigrun` (`big_run_step4000.pt`,
`big_run_step8000.pt`, logs, README). Full series: `molab/big_run_results.md`.

## Data facts (Grand Tour / ANYmal-D)

- HF `leggedrobotics/grand_tour_dataset`, downloads are PUBLIC (no token needed).
- 10 Hz `hdr_front` (1920×1280) + camera ts zarr; `anymal_command_twist`
  (~9 Hz) = action; `anymal_state_state_estimator` = proprio (twist_lin/ang,
  foot contacts, joint pos/vel); `dlio_map_odometry` = pose GT; `hdr_front_mask`
  exists only for ETH-1 (robot pixels = BLACK/low value in those PNGs).
- Mission dirs: `<root>/<timestamp>/{images/hdr_front/*.jpeg,
  data/{hdr_front,anymal_command_twist,anymal_state_state_estimator,
  dlio_map_odometry}}`.
- fps ≈ 10 ⇒ stride 4 ≈ 400 ms gaps; tubelet j covers sampled frames (2j,2j+1).

## Environment / gotchas (learned the hard way)

- **fp32 only in standalone scripts** — bf16 (autocast or model casts) breaks
  the vjepa2 RoPE attention path (mixed dtypes) when run outside the marimo
  kernel. Keep `.cuda()` fp32.
- The published V-JEPA ckpt key is `ema_encoder`; predictor head re-init to
  embed dim 768. Encoder is FROZEN (fine-tuning empirically loses — plan §2.7).
- Decoder/EVAL paths: scripts auto-fallback from `/marimo/*` to `data/`,
  `ckpts/`, `logs` when `/marimo` does not exist (works on molab + AMD box).
- marimo notebooks (if used): imports are single-owner per name across cells;
  long cells need progress prints < ~3-5 min apart or the stream drops; kernels
  restart + sandbox sessions die — keep the only copies on HF/git/disk, never
  only on a node. `marimo._code_mode` (`cm`) is the only safe way to edit cells.
- When both a training run and a puller run concurrently, half-written JPEGs
  can appear — the trainer already retries; don't "fix" by deleting missions.
- Adding new missions: append timestamps to `CORE`/`ADD` lists in
  `amd/big_run.py` + `amd/pull_missions.py` (held-out eval pair must stay out:
  ICE-1 `2024-11-18-13-48-19`, SPX-2 `2024-11-02-17-18-32`).

## Current open items / next work (as of last session)

1. **Robot-mask pixel eval** — decoder + robot-motion metrics coded
   (`amd/eval_decode.py`); needs a run for final mask numbers (masks ETH-1 only;
   val envs lack masks — state this limitation in any write-up).
2. **Controllability ablations + proprio stream** — coded (`amd/eval_control.py`);
   needs a run (per-stream shuffle/zero/novel-command table on held-out ICE-1).
3. **Causal-vs-random + hidden-state (contact/bob) on step-8000** — coded
   (`amd/probe.py --train-random N` / `--vs-random`); needs a run.
Then: proper train/test split from `configs/missions_split.csv`, decoder v2 /
per-frame or VAE-latent decode + LPIPS/FVD, value/Q-over-latent planning
(QWM/Beyond-Imitation), MPC-in-latent (VIP-Loco), extrapolation head (LDR),
sim-scale data (FetchMan / Active Flow), port to AMD MI300X (done: amd/).

## Paper→technique map (why the recipe looks like this)

Frozen single teacher (no EMA drift) ≈ LeVJEPA; causal tubelet masks ≈
block-causal (LeVJEPA); hidden-state aux heads = the fix demanded by
"Can Video World Models Track Unobserved World States?"; command tokens in the
predictor + per-stream dropout ≈ Flex-π multi-stream; modality-aligned
prediction ≈ Humanoid Next-Token; robot-motion-first eval ≈ Hydra-0 (whole-frame
PSNR/feature cosine cannot adjudicate dynamics); decoder teacher→predicted
distill ≈ GenFirst ordering; planning/value later per QWM / Beyond-Imitation /
VIP-Loco. Details: README §2.7 + SURVEYED_PAPERS.md.

## Agent conventions

- Keep every experimental claim paired with a baseline (copy-last-context,
  random-mask) and on held-out envs (ICE-1/SPX-2) with transient & steady
  segments at ≥3 s horizons.
- Prefer adding self-contained scripts under `amd/` over notebook cells; test
  with `python -m py_compile` before running; commit results into
  `molab/big_run_results.md` and push ckpts to HF.
- GPU runs: launch detached with logs to files; poll with short streams.
