# Validation experiment (exp-v2 baseline run)

Everything below was run on an RTX PRO 6000 (96 GB) container, kernel
python 3.13 for tooling + a uv python 3.10 venv for jepa-wms (torch 2.14
+ cu130). Training data: 4 ETH daytime missions
(`2024-10-01-11-29-55`, `2024-10-01-11-47-44`, `2024-10-01-12-00-49`,
`2024-11-02-17-10-25`), hdr_front camera.

## Phase 1 - world model (`gt_v0_..._AdaLN_d6_2roll_1n`)

- Frozen DINOv2-vits14 frame encoder (torch.hub, cached under $TORCH_HOME).
- AdaLN predictor depth 6, embed 384; 12-frame windows @ 5 fps (raw 24-frame
  windows at 10 fps); 2-step rollout; L2 latent loss; batch = #episodes (4);
  flat lr 5e-4; 1000 steps (4 epochs x 250); bf16.
- Launch: `bash grandtour/scripts/run_training.sh wm`
- Loss (epoch mean of L2, per-step prints vary ~0.9-2.5 due to tiny batches):

  | epoch | avg loss |
  |---|---|
  | 1 | 2.75 |
  | 2 | 1.87 |
  | 3 | 1.63 |
  | 4 | 1.53 |

## Phase 1 eval - copy-baseline / rollout (guide section 10)

`python grandtour/scripts/eval_copy_baseline.py` (24 windows, ctx = 4):

  | metric | model mse | copy-last-frame mse | ratio |
  |---|---|---|---|
  | teacher 1-step | 1.573 | 1.553 | 1.01 |
  | rollout h=1 | 1.526 | 1.460 | 1.05 |
  | rollout h=2 | 1.945 | 2.177 | 0.89 |
  | rollout h=3 | 2.185 | 2.454 | 0.89 |
  | rollout h=4 | 2.405 | 2.749 | 0.88 |
  | rollout h=5 | 2.552 | 2.922 | 0.87 |
  | rollout h=6 | 2.740 | 3.127 | 0.88 |

Interpretation: teacher-forcing numbers are not informative here (global
attention + short training allow near-copy shortcuts), but the
autoregressive rollout - which never sees future frames - beats the
no-dynamics baseline by ~12% from h >= 2, i.e. the model learned real
action-conditioned dynamics on the trained domain. Same-domain only;
generalization to unseen weather/terrain was not tested in this run.

## Phase 2 - decoder head + future-frame visualization

- VM2M-style ViT decoder (patch 8, depth 8, embed 384, LPIPS + pixel
  loss, pixelloss_weight 10 / perceptual 1), trained 3 epochs x 250 steps
  as an autoencoder on the frozen encoder features:
  `bash grandtour/scripts/run_training.sh step2`  (loss 1.02 -> 0.49)
- Checkpoints (auto-saved by jepa-wms):
  `$JEPAWM_LOGS/grandtour_sweep/<run>/jepa-latest.pth.tar`
  and `..._image_head.pth.tar` (separate file per head).
- `python grandtour/scripts/viz_future_frames.py` renders contact sheets
  (GT-recon row + context/rolled-out-prediction row) to `$GT_VIZ`.

## Reproducibility caveats

- Trained weights from the container session are NOT committed (too large);
  the commands above regenerate them in ~5-6 min on the same GPU class.
- The predictor config uses global attention (`attn.local_window_time: -1`).
  For a stronger model, enable the block-causal mask (set it to e.g. 3 as in
  the upstream DROID configs), use DINOv2 vitb/vitl, depth 12 and 2000+ steps.
- Held-out/val split: with only 4 missions, validation currently samples the
  same missions. Add more missions and point `data.validation.val_datasets` at
  a separate directory when scaling.
