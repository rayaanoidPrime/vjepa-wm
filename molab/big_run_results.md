# Big run results — full-recipe causal ActionVJEPA (2026-09-04/05)

Recipe: frozen official V-JEPA 2.1 ViT-B/384 encoder + causal tubelet masking
(boundary randomized {6,8,10}) + motion-weighted LN-L1 + per-window command
tokens (indices >= 4608) + velocity-consistency aux head (lambda 0.5) +
command dropout 25%. Streaming decode, fp32, batch 4, ~2.0 s/step.

Training: Phase A fresh 200→1200 steps (16 missions, 2202 records); Phase B
resumed 1200→4000 steps while the puller added missions (train records grew
2762 → 5552 by step 3500, i.e. ~40 missions at ~140 clips each). Stride 4
(400 ms), 16-frame clips @ 384.

Artifacts: molab/big_run_log.txt (full log), molab/big_run_console.txt
(clean, no errors), big_run_step4000.pt (pred+mlp+vh; 92 MB, kept out of git).
Earlier checkpoints big_run_step{200..3800}.pt were on the molab sandbox
(not downloaded) — step4000 is the final/best.

## Held-out eval series (cosine | velRMSE true -> shuffled, m/s)

Phase A (fresh, corpus fixed ~16 missions):
| step | ICE-1 cos | ICE vel | SPX-2 cos | SPX-2 vel |
|---|---|---|---|---|
| 200 | 0.832 | 0.082 -> 0.089 | 0.794 | 0.109 -> 0.117 |
| 400 | 0.844 | 0.088 -> 0.093 | 0.804 | 0.084 -> 0.089 |
| 600 | 0.857 | 0.109 -> 0.126 | 0.815 | 0.090 -> 0.109 |
| 800 | 0.859 | 0.083 -> 0.095 | 0.820 | 0.082 -> 0.092 |
| 1000 | 0.864 | 0.097 -> 0.123 | 0.822 | 0.076 -> 0.101 |
| 1200 | 0.866 | 0.062 -> 0.090 | 0.826 | 0.054 -> 0.081 |

Phase B (resumed; corpus growing as missions pulled):
| step | ICE-1 cos | ICE vel | SPX-2 cos | SPX-2 vel |
|---|---|---|---|---|
| 1400 | 0.871 | 0.075 -> 0.111 | 0.829 | 0.085 -> 0.106 |
| 1600 | 0.872 | 0.067 -> 0.093 | 0.832 | 0.067 -> 0.087 |
| 1800 | 0.874 | 0.081 -> 0.097 | 0.833 | 0.071 -> 0.093 |
| 2000 | 0.875 | 0.075 -> 0.106 | 0.835 | 0.075 -> 0.095 |
| 2200 | 0.875 | 0.074 -> 0.100 | 0.835 | 0.067 -> 0.091 |
| 2400 | 0.877 | 0.066 -> 0.098 | 0.836 | 0.089 -> 0.114 |
| 2600 | 0.877 | 0.062 -> 0.097 | 0.837 | 0.058 -> 0.089 |
| 2800 | 0.878 | 0.096 -> 0.120 | 0.840 | 0.113 -> 0.138 |
| 3000 | 0.880 | 0.086 -> 0.113 | 0.840 | 0.120 -> 0.142 |
| 3200 | 0.879 | 0.070 -> 0.097 | 0.841 | 0.080 -> 0.100 |
| 3400 | 0.881 | 0.111 -> 0.126 | 0.842 | 0.108 -> 0.131 |
| 3600 | 0.884 | 0.086 -> 0.114 | 0.842 | 0.088 -> 0.116 |
| 3800 | 0.884 | 0.059 -> 0.095 | 0.845 | 0.057 -> 0.086 |
| 4000 | 0.885 | 0.057 -> 0.092 | 0.845 | 0.056 -> 0.087 |

## Headlines
- Held-out feature cosine: 0.832 -> 0.885 (ICE-1), 0.794 -> 0.845 (SPX-2)
  across ~40-mission training; monotone-ish gains through the whole run.
- Controllability signal persists and strengthens: wrong-command penalty on the
  velocity readout +8% (step 200) -> +60% (ICE-1) / +56% (SPX-2) at step 4000.
- Final velocity aux error ~0.056-0.057 m/s on unseen environments
  (velocity scale ~0.3-0.4 m/s).
- Recipe scaled 4 -> ~40 missions with continued improvement; no regression.

## Notes
- fp32 (no autocast/bf16: bf16 RoPE path has dtype bugs in standalone runs).
- Resume: loads latest big_run_step*.pt; sampler retries truncated JPEGs
  (pullers run concurrently).
- Mission split beyond core 4 is approximate (remaining corpus treated as
  train); eval is strictly held-out ICE-1/SPX-2. Full per-mission split from
  configs/missions_split.csv should be applied for the production run.

## Addendum 9: resumed run to step 8000 (2026-09-05, ~40 missions)
Resumed from step4000 in a fresh sandbox; corpus grew 2762 -> 5552 records
(~40 missions). Final step-8000 val:
| env | cos | velRMSE true -> shuffled |
|---|---|---|
| ICE-1 | 0.8903 | 0.061 -> 0.097 (+58%) |
| SPX-2 | 0.8538 | 0.052 -> 0.092 (+79%) |

## Addendum 10: probe suite on step-8000 (3 s displacement, m)
| cell | teacher | causal | copy |
|---|---|---|---|
| ICE-1 transient | 0.866 | 0.905 | 1.136 |
| ICE-1 steady | 0.700 | 0.344 | 0.608 |
| SPX-2 transient | 0.792 | 0.722 | 0.997 |
| SPX-2 steady | 0.777 | 0.493 | 0.709 |
Control-signal (ICE-1, re-sampled): velRMSE true 0.120 vs shuffled 0.245 (+105%).
Causal beats copy in all cells; steady-far gains large.

Checkpoints mirrored to HF: https://huggingface.co/rayaanoidpr/vjepa-wm-bigrun
(big_run_step4000.pt, big_run_step8000.pt, log, README).
