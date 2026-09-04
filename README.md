# V-JEPA 2.1 world models for quadruped locomotion (ANYmal-D / Grand Tour)

Experiments toward **a latent video world model for quadruped locomotion**:
fine-tuning [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) on the
[ANYmal-D / Grand Tour dataset](https://grand-tour.leggedrobotics.com/)
(HF: `leggedrobotics/grand_tour_dataset`), predicting future latents, decoding
them back to pixels, and (next) conditioning on actions/proprioception so the
latent video is controllable.

**Current pipeline:** a single self-contained notebook script
(`grandtour_vjepab_pipeline.py`) used through **marimo** — V-JEPA 2.1 ViT-B/384
fine-tune with masked-latent (JEPA) objectives, then latent→pixel decoding.
**Compute:** NVIDIA molab (RTX Pro 6000, 96 GB) today; a **private AMD
Instinct MI300X (ROCm) cluster** is the next target — see §1.4.

- **[§1 How to run](#1-how-to-run)** — first
- **[§2 Plan](#2-plan-causal-v-jepa-world-model-for-quadruped-locomotion)** — second
- **[§3 Surveyed papers](#3-surveyed-papers)** → full digest in [`SURVEYED_PAPERS.md`](SURVEYED_PAPERS.md)

---

## 1. How to run

### 1.1 Prerequisites

- Python **3.12+** (repo pins `zarr<3`; the package metadata requires ≥3.12).
- PyTorch **2.1+** with a GPU (CUDA or ROCm — see §1.4), `timm`, `einops`.
- **HuggingFace account with Grand Tour dataset access** (gated —
  [request here](https://grand-tour.leggedrobotics.com/)); log in with a token:
  `huggingface-cli login`.
- Internet access on first run: the pipeline downloads the official
  **V-JEPA 2.1 ViT-B/384 checkpoint** (~2 GB) from `dl.fbaipublicfiles.com` into
  `~/.cache/torch/hub/checkpoints`.

### 1.2 Clone + install (includes submodules)

```bash
git clone --recursive <repo-url> && cd vjepa
# or, if already cloned:  git submodule update --init --recursive

conda create -n vjepa python=3.12 -y && conda activate vjepa
pip install -e .          # torch, torchvision, zarr<3, timm, einops, hf_hub, ...
pip install marimo        # notebook runner used for the pipeline
```

One pinned submodule provides the model code:
- `vjepa2/` — facebookresearch/vjepa2 (V-JEPA 2.1 encoder/predictor, imported
  at runtime as `app.vjepa_2_1` / `src`).

(The LiMo repo, `leggedrobotics/less-is-more`, was used only as a reference for
its data recipe and is **not** a dependency — the parts used here are
re-implemented in `scripts/` + `configs/missions_split.csv`.)

### 1.3 Get the data

The Grand Tour missions used here are the **teleop / navigation recordings with
`hdr_front` camera frames** listed in `configs/missions_split.csv`
(40 train / 2 val / 6 test environments). Download a mission subset with:

```bash
python scripts/pull_data.py --dataset-folder data/grandtour \
    --missions-csv configs/missions_split.csv          # all 48 missions
# or a quick smoke subset:
python scripts/pull_data.py --dataset-folder data/grandtour --missions \
    2024-10-01-11-29-55 2024-10-01-11-47-44
```

This fetches `<mission>/data/hdr_front` + `dlio_map_odometry` zarrs and
`<mission>/images/hdr_front/*.jpeg` (JPEGs are read lazily at train time).
Extending to the action/proprioception/mask topics
(`anymal_command_twist`, `anymal_state_actuator`,
`anymal_state_state_estimator`, `hdr_front_mask`) is planned work — edit
`DATA_TOPICS` / `MissionSource` in `vjepa_nav/data/grandtour.py`.

### 1.4 Run the current pipeline (marimo notebook)

**The pipeline is `grandtour_vjepab_pipeline.py` — paste its entire contents
into a single marimo Python cell and run it** (it is a straight-line,
resumable script, not a sequence of dependent cells):

```bash
marimo edit            # create/reopen a notebook, paste the file into one cell
# or run it headless as a plain script:
python grandtour_vjepab_pipeline.py
```

What it does (all resumable — re-running skips completed stages):

| Stage | What | Outputs |
|---|---|---|
| Setup | auto-clone V-JEPA repo if missing; download official ViT-B/384 ckpt | `~/.cache/torch/hub/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt` |
| Data | pull missing missions, build 64-frame clips (stride 1, step 32) | in-memory records |
| Stage 1 | fine-tune encoder+predictor with the V-JEPA masked-latent objective (10 epochs, `lr 1e-5`, EMA target `0.996`) | `vjepa21_grandtour_stage1_{epoch,best}.pth` |
| Stage 2 | predict the latent of frame `t+8` from frame `t` and decode to 384×384 RGB (`FutureSpatialPredictor` + `SharpRGBDecoder`) | `stage2_*.pth`, `vjepa21_grandtour_final_sharp_decoder.pth` |

Config happens at the top of the file (constants) and via env vars:

| Env var | Default | Meaning |
|---|---|---|
| `VJEPA_ROOT` | `<base>/vjepa2` | V-JEPA 2.1 repo (auto-cloned if missing) |
| `GRANDTOUR_DATA_ROOT` | `<base>/grandtour` | mission data |
| `CHECKPOINT_DIR` | `<base>/vjepa2_grandtour_checkpoints` | all checkpoints / latent caches |

Key knobs: `MISSIONS[:k]` for a smoke subset, `MAX_CLIPS_PER_MISSION=300`,
`BATCH_SIZE=4` (fits the 96 GB GPU at 64×384² with bf16),
`PRECOMPUTE_STAGE2_LATENTS=True` (one-time latent cache, ~21 GB disk, makes
Stage 2 ~2–4× faster), `STAGE1_BEST_DRIVE_PATH` / `STAGE1_BEST_GDRIVE_ID`
(resume Stage 1 from an existing best checkpoint), `CONTINUE_STAGE1` /
`RESUME_STAGE2`.

GPU notes (NVIDIA **molab**, and the planned **AMD MI300X cluster**):

- The code is plain PyTorch + `timm` + `einops` (SDPA attention) — **no CUDA-only
  deps**, so it runs on ROCm as-is.
- bf16 autocast is auto-selected when supported (`torch.cuda.bf16_supported()`
  is true on MI300X under ROCm); the fp16 GradScaler path is only used when bf16
  is unavailable. No changes needed for ROCm installs of PyTorch.
- On molab (12 h session cap): every stage is checkpoint-resumable; mirror
  checkpoints off-box to HF (`vjepa_nav/utils/checkpoint.py` has
  `upload_to_hf`; the blob has Drive-download hooks).

### 1.5 Reproducibility conventions

- Seed 42 for data splits / masks / torch.
- Train/val/test splits are **by mission** (`configs/missions_split.csv`) —
  never sample clips across environments at random without keeping mission
  boundaries (leaks otherwise).
- Large artifacts (`*.pth`, `data/`, `logs/`, `ckpt/`) are gitignored; model
  weights are meant to live on HF or local disk, not in this repo.

---

## 2. Plan — causal V-JEPA world model for quadruped locomotion

Big picture: **the best world model for quadruped locomotion.** This plan takes
the project from where it is today (a random-masked V-JEPA 2.1 Grand-Tour
fine-tune + a decode-to-pixels experiment) to an **action-controllable latent
world model that provably tracks the quadruped's state, extrapolates past its
training horizon, and is decodable to pixels as an evaluation/planning
interface**.

Full paper-by-paper motivation: [`SURVEYED_PAPERS.md`](SURVEYED_PAPERS.md).

### 2.0 Where things stand (verified against this repo)

**Dataset reality check (matters more than the model):**
- "ANYmal-D" = the robot (ANYbotics ANYmal D + Boxi payload); the dataset is the
  **Grand Tour dataset** (49 environments, ~50k steps, 150k images; gated).
- Per-mission HF tree exposes everything needed beyond images:
  - `images/hdr_front/*.jpeg` + `data/hdr_front` (camera ts) — currently pulled,
  - **`images/hdr_front_mask`** — binary robot-pixel mask (robot-only metrics),
  - **`data/anymal_command_twist`** — commanded twist = the action stream,
  - **`data/anymal_state_actuator`** (12 joints: pos/vel/torque),
    **`anymal_state_state_estimator`** (base state), `anymal_imu` — proprio,
  - `dlio_map_odometry` (world-pose GT), `cpt7_ie_*` (RTK-GPS), depth cameras.
- Two code paths exist today: the **notebook pipeline**
  (`grandtour_vjepab_pipeline.py`, ViT-B/384, 64-frame clips, tubelet 2,
  *random* 75% block masks, stage-1 fine-tune + stage-2 latent→pixel decoder) —
  the substrate for the causal experiment — and the packaged `vjepa_nav/`
  (GoalVJEPA ViT-L, per-frame image-path latents, GOAL-token conditioning,
  ConditionalDiT decoder + eval).

**Experiment restated precisely** (what this plan is about):
> Load the released V-JEPA 2.1 checkpoint, continue training on ANYmal-D
> `hdr_front` clips; context = frames **0…t**, targets = frames **t+1…N**
> (temporal/causal masking instead of random masking); decode predicted future
> latents **to pixels** to see if the model predicts the quadruped's path;
> later condition the predictor on **proprio/actions** so the latent video is
> controllable.

That is a **causal latent world model** — the right core. The plan below makes
it rigorous and adds the eval that actually answers "can it predict the path".

### 2.1 Target architecture (one paragraph + knobs)

- **Encoder:** V-JEPA 2.1 ViT-B/16, 384², tubelet 2 (substrate: notebook
  pipeline), *fine-tuned* (never trained from scratch at this data size). Deploy
  with **block-causal attention** so "encode up to t" is a property of the
  model, rollout is cheap and never re-encodes full clips (LeVJEPA).
- **Task/mask:** context tubelets = frames ≤ t, target tubelets = frames > t
  (even t; tubelet-major boundary — see gotcha §2.5.2). Loss = V-JEPA 2.1 dense
  predictive loss (LN L1) vs the frozen EMA target encoder. Keep a small
  fraction of *original random-mask* batches in the mix to avoid forgetting the
  representation.
- **Making the task non-trivial** (high-rate video makes "next frame" ~free):
  temporal stride (context ≈ 0.5–1 s, targets 1–4 s ahead), randomized boundary
  t per sample, and predict a *subsampled* set of future tubelets.
- **Action/proprio stream (controllability):** per-frame embedding of (command
  twist + actuator state + base state) → action tokens inserted into the
  predictor context in time order (GOAL-token pattern in `goal_vjepa.py`).
  Regime A (asked for): given actions for t+1…N, predict the future. Regime B
  (later): infer actions from predicted future.
- **Decoder ("project latents to pixels"):** per-tubelet token → pixels (what
  V-JEPA decoders do); existing decoders (`SharpRGBDecoder` / `ConditionalDiT`)
  are per-frame image-path latents — pick per phase budget in §2.3 Phase 2.
- **Eval discipline (the #1 paper lesson):** pixel quality is NOT the answer —
  the quadruped's hidden state (contact phase, foot placement, executed vs
  commanded twist) is not supervised by any pixel loss. Two eval tiers: (a)
  video/path prediction (pixels + robot-mask metrics), (b) hidden-state tracking
  (proprio/contact/odom readouts + extrapolation tests). Report both.

### 2.2 Evaluation protocol ("can the model predict the path?")

Split = **missions, not clips** (LiMo 40/2/6; test environments never seen in
training). OOD = environments outside the train split (e.g. train on
ETH+Jungfraujoch, test on ICE/station/urban).

**Tier 1 — video & path prediction**
- Latent: masked-target LN-L1 vs teacher; cosine(ŝ_t, s_t) per tubelet over the
  horizon; rank/collapse monitor.
- Decode: PSNR / SSIM / LPIPS / FVD, twice — teacher-cond decode (decoder
  ceiling) and predicted-cond decode (the question).
- **Robot-centric (Hydra-0 lesson):** restrict all pixel metrics to
  `hdr_front_mask` pixels; report **robot-motion error** (predicted vs actual
  body motion via mask centroid/bbox/flow). Basically free given the mask topic.
- **Path probe:** linear readout from each frame's predicted latent → SE(2)
  displacement vs DLIO odom GT; report endpoint error vs two baselines —
  *copy-last-frame* and *teacher-latent readout* (upper bound).

**Tier 2 — hidden-state tracking** (papers: Track-Unobserved, LDR, QWM)
- Aux readouts from future latents: base twist, joint pos/vel, contact forces,
  odom (vs `anymal_state_*`).
- **Extrapolation tests:** (i) double the prediction horizon at eval; (ii)
  novel terrain environments; (iii) novel commanded twists. Everyone degrades;
  the question is *how fast* and whether action/proprio conditioning flattens
  the curve (should — the controllable-WM justification).

**Control-readiness (later):** open-loop command rollouts (predicted vs real
video under the same command sequence); QWM-style imagination-time selection
once a value net exists.

### 2.3 Phases (hypothesis → change → deliverable → go/no-go)

> New package `vjepa_wm/` beside `vjepa_nav/`; port the notebook's logic
> module-by-module (it is the most complete substrate for the video-path
> design).

**Phase 0 — Rebuild the pipe on ANYmal-D with all streams (1–2 molab sessions)**
- Dataset access + HF token verified.
- Extend `scripts/pull_data.py` / `vjepa_nav/data/grandtour.py`: add
  `hdr_front_mask` (images) + `anymal_command_twist`, `anymal_state_actuator`,
  `anymal_state_state_estimator` (data); keep `dlio_map_odometry`.
- Pull 3–6 missions (ETH-1/2/3, ICE-1, SPX-2); measure hdr_front fps, body
  speed, joint/twist rates, timestamp alignment (nearest-neighbour sync as in
  `MissionSource`). **Write the numbers down** — they set stride/horizon.
- Smoke-run the existing notebook pipeline stage 1 (~200 steps); fix
  resume/ckpt→HF.
- *Deliverable:* data manifest + fps table + smoke checkpoint.

**Phase 1 — Causal masking on the video path (1 session, small scale)**
- `TemporalCausalMasker` (pure, unit-tested): given N frames, tubelet k=2,
  boundary t (even) → `masks_x` = tubelets with frames ≤ t, `masks_y` = rest;
  assert no tubelet straddles t, context ⊂ past.
- Regression check: same seed/config with *random* masks must reproduce the
  saved `vjepa21_grandtour_stage1_best.pth` val loss (port ≠ behaviour change).
- Short causal-mask run (1 mission, few epochs): val latent loss + cosine vs
  random-mask baseline at matched steps.
- *Go/no-go:* causal val loss ≤ random-mask loss at matched tokens seen (easier
  task). Much higher → check mask bug / context dilution.
- *Deliverable:* `vjepa_wm/masks.py` (+tests), baseline comparison table.

**Phase 2 — Full pretrain + decode-to-pixels (3–5 sessions, the real run)**
- Pretrain: train-split missions (~40), clips with stride / boundary / t
  randomization per §2.1; 20–30% random-mask batches kept; bf16 + grad-accum
  (batch 4–8 / 96 GB); EMA target frozen; ckpt every N steps + HF mirror. Track
  Tier-1 latent metrics on held-out missions.
- Decoder: (a) per-frame decode reusing the ConditionalDiT (image path, cheaper)
  **or** (b) V-JEPA-style tubelet→2-frames video decoder at 256²/384² (honest
  to the experiment description). Train on **teacher latents first**, then a
  short **distill pass on predicted latents** (V-JEPA decoder protocol, as in
  the S1/S2 recipe).
- *Deliverable:* causal WM ckpt + decoder ckpt + Tier-1 report (incl.
  robot-mask metrics) + montages `start | gt | teacher-dec | pred-dec`.

**Phase 3 — Action/proprio conditioning → controllable latent video
(2–3 sessions)**
- `ActionVJEPA`: dataset returns synchronized per-frame (twist, actuator,
  base-state); predictor gets action tokens (past actions in context; **future
  actions = the conditioning tested**, Regime A). Encoder stays block-causal on
  frames ≤ t.
- Ablations (matched budget): none vs proprio-only vs twist-only vs
  proprio+twist vs SE(2)-goal (GoalVJEPA reference).
- Decision metrics: (i) latent loss under *wrong* future actions must be worse
  than under *true* ones (control-signal check — flat ⇒ model ignores actions);
  (ii) commanded-twist ↔ decoded body-motion correlation (does the video obey
  the command?); (iii) Tier-2 readouts; (iv) novel-command extrapolation.
- *Deliverable:* ActionVJEPA module + ablation table. **Core paper result if
  the numbers hold.**

**Phase 4+ — Downstream (stretch; gated on Phase 3)**
- Value/planning on top (QWM + Beyond-Imitation pattern): small Q over
  (latent state, command) trained on real clips; pick commands by imagination.
- VIP-Loco-style kinodynamic readout + MPC-in-latent if Phase-3 latents are
  SE(2)/kinematic-interpretable.
- LDR-style kinematic-integration dynamics head if extrapolation is the
  bottleneck.
- Context: sim-generated data (FetchMan) + verifier-expanded training
  (Active Flow Expansion).

### 2.4 Implementation checklist (mapped to files)

| Task | Files |
|---|---|
| Pull mask / proprio / twist topics | `vjepa_nav/data/grandtour.py` (`DATA_TOPICS`, `MissionSource`), `scripts/pull_data.py` |
| Temporal causal masker + tests | new `vjepa_wm/masks.py` |
| Port notebook stage-1/2 into a package | `vjepa_wm/{data,models,train}.py` from `grandtour_vjepab_pipeline.py` |
| Action tokens in the predictor | GOAL-token pattern in `vjepa_nav/models/goal_vjepa.py` → `ActionVJEPA` |
| Video decoder | extend `vjepa_nav/models/diffusion_decoder.py` or new tubelet decoder |
| Eval with robot mask | new `vjepa_wm/evaluate.py` (extend `vjepa_nav/evaluate.py`) |
| Resumability / HF mirror | reuse `vjepa_nav/utils/checkpoint.py` |

### 2.5 Risks & mitigations

1. **"Next frame is trivial"** (high-rate HDR): stride + randomized t + target
   subsampling (§2.1); ablate a *copy-last-token* baseline.
2. **Mask leakage at the t boundary** (tubelet spans frames t/t+1): even t,
   tubelet-major masks, unit tests. The most common silent bug in this exact
   experiment.
3. **Pixel metrics mislead** ("Track Unobserved States"): always pair decode
   metrics with robot-mask + hidden-state readouts + extrapolation tests.
4. **Long-horizon / extrapolation collapse**: expected for vanilla AR
   transformers; measure it (§2.2 Tier 2) and keep LDR/stateful-rollout as the
   known upgrade path.
5. **Forgetting the pretrained representation** during fine-tune: keep 20–30%
   original random-mask batches; freeze encoder except LayerNorm first
   (`GoalVJEPA.freeze_encoder_except_ln` pattern), unfreeze later if needed.
6. **Data access / session caps**: register now; everything resumable; ckpts →
   HF.
7. **Memory**: ViT-B/64fr/384² @ batch 4 fits 96 GB; for ViT-L / longer context
   drop to 256² or enable activation checkpointing (already supported by the
   encoder).

### 2.6 Immediate next actions (this week)

1. Register for Grand Tour access; confirm the HF token on molab.
2. Extend the topic puller (mask + twist + actuator + state-estimator); pull 3
   missions; write the fps/timing table.
3. Write `vjepa_wm/masks.py` (causal masker + tests) and port the notebook blob
   into `vjepa_wm/` (the notebook already has the video-path plumbing wanted).
4. Read `SURVEYED_PAPERS.md` — "Track Unobserved States" (shapes the eval) and
   "LeVJEPA" (shapes the encoder-attention choice) first.

---

## 3. Surveyed papers

Surveyed 2026-09 while designing this project. Full annotations (claim →
mechanism → what to steal, with priorities per phase) are in
**[`SURVEYED_PAPERS.md`](SURVEYED_PAPERS.md)**; quick index:

| Paper | arXiv | Why it matters here |
|---|---|---|
| V-JEPA 2 (concept) | [2404.08471](https://arxiv.org/abs/2404.08471) | the backbone + dense latent loss |
| [LeVJEPA](https://arxiv.org/abs/2608.27395) | [2608.27395](https://arxiv.org/abs/2608.27395) | block-causal encoder; token dropping; no-EMA pretraining |
| [Flex-π](https://arxiv.org/abs/2608.10860) | [2608.10860](https://arxiv.org/abs/2608.10860) | multi-stream WAM; per-stream dropout → compute flexibility |
| [Zero-WAM](https://arxiv.org/abs/2608.26103) | [2608.26103](https://arxiv.org/abs/2608.26103) | in-context human-video task spec |
| [Hydra-0](https://arxiv.org/abs/2608.18077) | [2608.18077](https://arxiv.org/abs/2608.18077) | robot-motion eval metrics; action-as-motion |
| [Humanoid Locomotion as Next Token Prediction](https://arxiv.org/abs/2402.19469) | [2402.19469](https://arxiv.org/abs/2402.19469) | modality-aligned sensorimotor next-token prediction |
| [VIP-Loco](https://arxiv.org/abs/2603.14345) | [2603.14345](https://arxiv.org/abs/2603.14345) | kinodynamic latent + planning for legged robots |
| [QWM](https://arxiv.org/abs/2608.17163) | [2608.17163](https://arxiv.org/abs/2608.17163) | world models for test-time search on top of Q-learning |
| [Beyond Imitation](https://arxiv.org/abs/2608.21204) | [2608.21204](https://arxiv.org/abs/2608.21204) | small Q on top of BC; self-improvement from rollouts |
| [Can Video World Models Track Unobserved World States?](https://arxiv.org/abs/2608.30692) | [2608.30692](https://arxiv.org/abs/2608.30692) | video fidelity ≠ hidden-state tracking (drives our eval) |
| [LDR (world evolves)](https://arxiv.org/abs/2608.09926) | [2608.09926](https://arxiv.org/abs/2608.09926) | kinematic-integration latent dynamics for extrapolation |
| [GenFirst](https://arxiv.org/abs/2608.29335) | [2608.29335](https://arxiv.org/abs/2608.29335) | generation-before-reconstruction end-to-end schedule |
| [Active Flow Expansion](https://arxiv.org/abs/2606.08802) | [2606.08802](https://arxiv.org/abs/2606.08802) | verifier-driven expansion of the generable set |
| [FetchMan](https://arxiv.org/abs/2608.17027) | [2608.17027](https://arxiv.org/abs/2608.17027) | sim-scene scale + RL-after-BC lessons |

---

## 4. Repository layout & gotchas

```
configs/missions_split.csv      LiMo train/val/test mission split (40/2/6)
grandtour_vjepab_pipeline.py    ★ current pipeline: V-JEPA 2.1 ViT-B Grand-Tour
                                  fine-tune (stage 1) + latent→pixel decode (stage 2).
                                  Single marimo cell / plain script.
scripts/pull_data.py            download missions (hdr_front + DLIO odom + JPEGs)
vjepa_nav/                      packaged alternative path: GoalVJEPA (ViT-L,
                                image-path, GOAL-conditioned) + ConditionalDiT
                                decoder + eval (S1/S2/S2-eval CLIs, resumable)
vjepa2/                         submodule: facebookresearch/vjepa2 (model code)
SURVEYED_PAPERS.md              paper digest (see §3)
```

Gotchas learned the hard way:

- Released 2.1 checkpoint key is `ema_encoder`; the predictor head (trained
  against a ViT-G teacher, 1664-dim) is re-initialised to the encoder embed dim
  (1024 for ViT-L / 768 for ViT-B).
- `torch.hub.load('facebookresearch/vjepa2', ...)` is currently broken
  (`localhost:8300`); all loading goes through
  `vjepa_nav/utils/checkpoint.py` or the notebook's own loader.
- Grand Tour images are 1920×1280 HDR; resize→crop to model resolution
  (no horizontal flip, to keep the goal's lateral axis sign intact).
- zarr v2 is pinned (`zarr<3`, LiMo format); builder falls back to zarr v3
  writes if v2 isn't installed.
- Only teleop / navigation missions with `hdr_front` and real future frames are
  usable for future-frame prediction tasks (see `MISSIONS` in the notebook).
