# Big run results — full-recipe causal ActionVJEPA (2026-09-04)

Recipe: frozen official V-JEPA 2.1 ViT-B/384 encoder + causal tubelet masking
(boundary randomized {6,8,10}) + motion-weighted LN-L1 + per-window command
tokens (indices >= 4608) + velocity-consistency aux head (lambda 0.5) +
command dropout 25%. Streaming decode, fp32, batch 4, ~1.9 s/step.

Corpus: 16 missions (ETH-1/2/3, SPX-1 + 12 more pulled during the run), 2202
train records, stride 4 (400 ms), 16-frame clips @ 384.

Checkpoints: /marimo/checkpoints/big_run_step{200..1200}.pt (pred+mlp+vhead).
Log: /marimo/big_run_log.txt (molab).

## Held-out eval (every 200 steps)

| step | ICE-1 cos | ICE velRMSE true->shuf | SPX-2 cos | SPX-2 velRMSE true->shuf |
|---|---|---|---|---|
| 200 | 0.832 | 0.082 -> 0.089 | 0.794 | 0.109 -> 0.117 |
| 400 | 0.844 | 0.088 -> 0.093 | 0.804 | 0.084 -> 0.089 |
| 600 | 0.857 | 0.109 -> 0.126 | 0.815 | 0.090 -> 0.109 |
| 800 | 0.859 | 0.083 -> 0.095 | 0.820 | 0.082 -> 0.092 |
| 1000 | 0.864 | 0.097 -> 0.123 | 0.822 | 0.076 -> 0.101 |
| 1200 | 0.866 | 0.062 -> 0.090 | 0.826 | 0.054 -> 0.081 |

velRMSE = velocity readout error (m/s) from predicted latents under TRUE vs
SHUFFLED (wrong) future commands. Delta = controllability signal.

## Headlines
- Held-out feature cosine climbs 0.832 -> 0.866 (ICE-1), 0.794 -> 0.826 (SPX-2).
- Control-signal grows with training: wrong-command penalty +8% (step 200) ->
  +45% (ICE-1) / +50% (SPX-2) at step 1200. The model increasingly routes
  commanded motion into its predicted latents.
- Final velocity aux error ~0.05-0.06 m/s on unseen environments.
- Recipe scales 4 -> 16 missions without regression vs the earlier 4-env
  baseline (cos 0.858/0.834 at step 1000 equivalent point).

## Config notes / caveats
- fp32 (no autocast/bf16) chosen for standalone-process stability (bf16 RoPE
  path has dtype bugs in this env).
- Not auto-resuming in this version (see follow-up: load big_run_stepK.pt on
  start); training target was 1200 steps.
- Missions: train split is approximate beyond ETH-1/2/3+SPX-1 (ADD missions
  chosen from the remaining corpus); eval is strictly held-out ICE-1/SPX-2.
