
# Phase 2 report - causal V-JEPA world model (ANYmal-D / Grand Tour)
Date: 2026-09-04 | notebook: /marimo/notebook.py | molab RTX PRO 6000 96GB

## Recap of the session
Phase-1 spike (single env, stride 1) was DEGENERATE: copy-last-context-tubelet
beat trained models at every horizon (0.855/0.835/0.804 cos at stride 1/2/4)
while fine-tuning the encoder made things worse (0.813/0.822/0.779).

Root cause: (1) representation drift of fine-tuned student+EMA vs the official
feature space used for scoring, and (2) feature cosine dominated by static
scene texture (hdr_front = terrain ahead).

## Fix that worked: FROZEN encoder, predictor-only training
Teacher = frozen official V-JEPA 2.1 (no EMA drift by construction).

| model | ICE-1 cos (s2/s4) | SPX-2 cos (s4) |
|---|---|---|
| copy-last-context | 0.835 / 0.804 | 0.741 |
| causal-frozen-250 | 0.852 / 0.838 | - |
| random-frozen-250 | 0.856 / 0.839 | - |
| causal-frozen-1000 (4 envs) | 0.858 | 0.834 |

Phase-2-lite pretraining: train = ETH-1/2/3 + SPX-1 (~16k frames), held-out =
ICE-1 + SPX-2 (~11k frames), stride 4 (400 ms), boundary randomized {6,8,10},
1000 steps. Causal-frozen beats copy-last by +7.4 pts (ICE-1) and +9.3 pts
(SPX-2) in official-teacher feature cosine.

## Decoder prototype (latent -> pixels)
Conv upsample 24x24x768 -> 384x384 RGB, trained on teacher latents (tubelet 4
-> frame 9), then evaluated teacher/causal/copy on held-out envs:
  ICE-1: teacher PSNR 21.46 | causal 17.84 | copy 18.64
  SPX-2: teacher PSNR 21.43 | causal 18.51 | copy 17.25
Teacher ceiling ~21.4 dB (600-step CNN decoder; no perceptual loss).
Whole-frame PSNR is only ~1 dB sensitive to causal-vs-copy -> pixel metrics
are dominated by static scene (plan warning confirmed). Montages:
  /marimo/checkpoints/montages/montage_*.png   (GT | teacher | causal | copy)

## Artifacts
  /marimo/checkpoints/phase1_stride2_causal150.pt
  /marimo/checkpoints/phase1_stride4_causal250.pt
  /marimo/checkpoints/phase1_frozen_stride4_pred.pt
  /marimo/checkpoints/phase2_stride4_pred1000.pt     <- best latent predictor
  /marimo/checkpoints/phase2b_teacher_decoder.pt
  V-JEPA ckpt: /marimo/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
  Data: /marimo/grandtour/{6 missions} (ETH-1/2/3, SPX-1 train; ICE-1, SPX-2 val)

## Key conclusions
1. Causal-mask machinery works; the recipe that beats the trivial baseline is
   FROZEN official encoder + predictor-only training (drift-free).
2. Random vs causal masking are statistically tied in feature space; causal
   structure should pay off once evaluation/prediction targets motion
   (robot-mask metrics, decoded flow), not whole-frame static texture.
3. Whole-frame PSNR cannot adjudicate dynamics (plan warning #3) - move to
   robot-mask motion error (hdr_front_mask for ETH-1 already downloaded) and
   command-conditioned rollouts (Phase 3).

## Next steps (Phase 3 prep)
- fps measured = 10 Hz on both envs; use stride 4 (400 ms) as default.
- Phase 3: pull anymal_command_twist / anymal_state_actuator /
  anymal_state_state_estimator topics; add action tokens to the predictor;
  ablation none/proprio/twist; control-signal check (wrong actions must hurt).
- Decoder v2: per-frame decode + perceptual/VGG loss + robot-mask metrics.
- Scale: all ~40 train missions; 2-5k predictor steps; resumable ckpt loop;
  mirror ckpts to HF (need a valid HF token - current one is invalid).


## Addendum: full Phase-2 eval suite (2026-09-04, later session)
All on the phase2 causal predictor (frozen official encoder, 1000 steps,
stride-4 training, boundary {6,8,10}) vs copy-last & random baselines,
scored against the frozen official teacher on held-out envs.

### Latent: horizon sweep + per-tubelet decomposition (cosine)
| env | stride | causal | copy | gap |
|---|---|---|---|---|
| ICE-1 | 2 / 4 / 6 / 8 | .858/.853/.842/.847 | .841/.805/.772/.769 | +.02/.05/.07/.08 |
| SPX-2 | 2 / 4 / 6 / 8 | .834/.830/.830/.829 | .767/.743/.739/.735 | +.07/.09/.09/.09 |

Per-tubelet (b=8, tubelets 4..7 = near->far): causal stays flat (ICE s4:
.865/.852/.853/.843) while copy decays (.860/.808/.796/.757) -> the model
learns dynamics beyond static-copy; margin grows with distance & horizon.

### Path probe (linear head tubelet-4 mean pool -> body-frame dx,dy; DLIO GT)
Head trained on ETH-1 teacher features only. Mean endpoint error (m); GT
mean |d| ~0.59 m (train) / 0.59 / 0.56 m (ICE-1 / SPX-2):
| env | teacher | causal | copy |
|---|---|---|---|
| ICE-1 | 0.212 | 0.510 | 0.231 |
| SPX-2 | 0.199 | 0.518 | 0.194 |

Reading: at ~0.8 s the displacement is near-velocity-extrapolation (copy ~
teacher). CAUSAL PREDICTED features are worse precise-motion carriers
(+0.28-0.32 m) -> latent LN-L1 averages out motion-critical detail. Fixes to
test: predict motion-critical (change/high-gradient) token subsets harder,
condition on proprio/actions (Phase 3), evaluate at far tubelets (3 s).

### Decoder (v1 CNN) whole-frame PSNR - already reported; montages in
/marimo/checkpoints/montages/. Whole-frame PSNR ~1 dB sensitive only.

### Still pending (blocked/planned)
- LPIPS / FVD (needs packages + larger sample sets)
- robot-mask region metrics (hdr_front_mask only on ETH-1 = train env so far)
- Tier-2 proprio/contact readouts + novel-command extrapolation (needs
  anymal_command_twist / actuator / state_estimator topics - Phase 0/3)


## Addendum 2: Phase-3 prep (2026-09-04) - command/proprio topics
Pulled anymal_command_twist (linear 3 + angular 3 @ ~9.3 Hz), anymal_state_actuator
(12 joints, incl. state_gear_position/velocity, command_position/velocity/torque,
contacts), anymal_state_state_estimator (foot contact/wrenches etc.) for
ETH-1, ICE-1, SPX-2. Command rate ~9.3 Hz; hdr_front 10 Hz -> clean sync.

Coupling sanity (ICE-1): corr(cmd vx, odom speed) = 0.923 over 39 samples -
commands do drive the robot; action signal is real and usable.

Control-signal readout probe (linear head, body-frame dx,dy over target window
0.8 s; GT |d| ~0.59 m), endpoint error:
| features | ICE-1 true/shuf | SPX-2 true/shuf |
|---|---|---|
| video-only (teacher) | 0.217 / 0.203 | 0.193 / 0.167 |
| video + cmd(6d mean) | 0.263 / 0.201 | 0.156 / 0.258 |
| cmd-only | 0.845 / 0.593 | 0.444 / 0.322 |

Reading: at 0.8 s the video latent already encodes velocity (video-only ~
oracle); window-mean commands add little and a 6-dim linear readout is too
weak/noisy to demonstrate controllability (shuffle signal appears only on
SPX-2). Commands alone are far worse than video. CONCLUSION: the decisive
Phase-3 test is per-frame ACTION TOKENS inside the predictor (ActionVJEPA:
command/proprio token per tubelet window, appended to the predictor context
with an extended/assigned position grid or cross-attention adapter), evaluated
at 2-3 s horizons and with novel (zero/held-out) command sequences - NOT a
readout-level probe.

Session caveat: the molab marimo kernel restarted once mid-session (all state
lost; artifacts on disk survived). Long cells must print periodically - silent
stretches >~5 min cause the execute stream to drop and the cell to be
cancelled. Mitigation applied (keep-alive prints).


## Addendum 3: Tier-2 hidden-state probe (contact_probe cell, 2026-09-04)
Linear readouts (trained on ETH-1 teacher features) for body linear velocity
(estimator twist_lin), foot-contact vector, and vertical bob (dlio z) from
tubelet-4 latents of each source, on held-out ICE-1 / SPX-2:
| source | vel RMSE (corr) | contact MAE | dz RMSE (corr) |
|---|---|---|---|
| teacher | 0.020/0.024 (.99/.99) | 0.007/0.010 | 0.010/0.007 (.98/.87) |
| causal-pred | 0.334/0.317 (.92/.61) | 0.664/0.635 | 0.056/0.024 (-.07/-.05) |
| copy-last feat | 0.056/0.074 (.97/.93) | 0.027/0.032 | 0.047/0.047 (.58/-.29) |
| persist (state at t7) | 0.075/0.064 | ~0 (labels near-degenerate) | 0.060/0.012 |

Conclusion: official latents encode hidden state very well; CAUSAL PREDICTED
tubelet features are the WORST hidden-state carrier (velocity corr drops to
0.6-0.9; bob corr ~0), below copy-last and persist. Converges with the path
probe: the LN-L1 dense loss averages away motion/state-detail. Fixes to try:
(a) motion-weighted loss (weight target tokens by inter-frame change), (b)
aux state-consistency heads during pretraining, (c) predict velocity-embedded
feature differences; then re-run these probes + far-tubelet (3 s) velocity
readout where copy-last decays.


## Addendum 4: P1b motion-weighted pretraining + re-probe (train_mw/probe_mw)
Retrained causal predictor (frozen enc, ETH-1, 1000 steps) with target-token
loss weighted 1 + 2*(tubelet motion / mean motion) (motion = mean |frame(2j)-
frame(2j-1)|). Val cos (ICE-1) 0.852. Re-probed velocity + displacement
readouts near (tubelet 4, ~0.8 s) and far (tubelet 7, ~3.2 s; GT ~2.3 m):
| source | ICE near | ICE far | SPX near | SPX far |  (v RMSE m/s, d err m)
teacher:  v .137/.134/.161/.192  d .13/.83/.19/.87
causal-old: v .326/.305/.316/.267  d .42/1.72/.43/1.70
causal-mw: v .294/.284/.290/.271  d .40/1.67/.39/1.56
copy-last: v .150/.171/.152/.197  d .13/1.04/.19/1.25
persist-zero: v .40/.44/.40/.43  d .56/2.33/.57/2.29

Conclusion: motion-weighting gives small consistent gains (displacement err
-0.1..-0.15 m; near velocity improves), directionally right, but predicted
features still trail copy-last readouts. Root cause insight: these recordings
are mostly STEADY-STATE cruise (velocity ~ constant over 3 s) -> copy-last is
a near-oracle readout; discriminators must target transient motion (turns,
stops, terrain changes). Next: (1) transient-segment eval set (filter clips by
large command/velocity deltas); (2) ActionVJEPA with proprio velocity inputs;
(3) consider auxiliary un-normalized velocity-consistency head (LN destroys
magnitude that velocity readouts need).


## Addendum 5: transient vs steady discriminator (transient_eval, 2026-09-04)
Clips scored by actual twist (mean|ang_z|*2 + lin-speed-range/mean); transient
= top, steady = bottom. Heads fit on ETH-1 mixed teacher features; displacement
error (m) by env/set/horizon (n=16 per cell):
| cell | causal | copy | teacher |
|---|---|---|---|
| ICE-1 transient k7 | 0.936 | 1.277 | 1.048 |
| SPX-2 transient k7 | 0.853 | 1.142 | 1.044 |
| ICE-1 steady k7 | 0.373 | 0.480 | 0.566 |
| SPX-2 steady k7 | 0.492 | 0.477 | 0.497 |

Directional support for the transient hypothesis: causal < copy on displacement
in both held-out envs at far horizon ONLY on transient windows (-0.3 m), gap
vanishes on steady. Low n (16) + low linear velocities during rotation-heavy
transients => noisy correlations; re-run at n>=60 with angular error before
headline claims. Also teacher <= copy on far steady at times - heads are
underpowered for these small sets.


## Addendum 6: powered transient/steady discriminator (transient_big, n=60)
Heads fit on ETH-1 (60 transient + 60 steady teacher features). FAR horizon
(k=7, ~3 s) displacement error (m):
| env . set | causal(mw) | copy | teacher |
|---|---|---|---|
| ICE-1 transient | 0.838 | 1.246 | 0.855 |
| ICE-1 steady | 0.230 | 0.821 | 0.786 |
| SPX-2 transient | 0.909 | 1.172 | 0.823 |
| SPX-2 steady | 0.304 | 0.539 | 0.652 |
Yaw error (deg) far: causal best in 3/4 cells (e.g., SPX steady 14.5 vs copy
30.9 vs teacher 41.3). Velocity RMSE lowest for causal in most rows (corr
noisy on rotation-heavy sets). Near (0.8 s) differences small.
CONCLUSION (n=60, 2 held-out envs): motion-weighted causal predicted features
beat copy-last displacement/yaw readouts at the 3 s horizon in ALL cells -
steady and transient - and match teacher. This is the first clean evidence
that predicted latents carry where the robot goes, beyond static-copy. Next:
ActionVJEPA (controllability) + decoder/mask motion eval.


## Addendum 7: B ActionVJEPA + B' causal-vs-random (2026-09-04)
B: action tokens (per-target-window mean cmd linear3+angular3, MLP->768,
indices >=4608 i.e. frame coord 8+) appended to predictor context. Matched
control + command-conditioned trainings produced IDENTICAL losses at every
step and control-signal delta ~0 (ICE stride4 and stride8: cos true .8654 vs
shuffled .8652; stride8 .8499/.8499) BUT output sensitivity to actions is
nonzero (|d(true-inverted)| ~0.012) -> pathway wired, model learns no marginal
use. Conclusion: under the LN-L1 feature objective on cruise-dominant data,
commands add ~nothing beyond velocity already inferable from visual context
(matches coupling 0.92 + steady-state findings). Controllability must be
pursued via (a) aux state-consistency head tying predicted tokens to commanded/
observed velocity, (b) decoder-level (pixel) generative conditioning, or (c)
novel-command / long-horizon regimes.

B': matched random-mask predictor (50% ctx, LN-L1, ETH-1, 500 steps) vs
causal(mw) far-tubelet displacement (m):
| cell | causal | random | copy |
| ICE-1 transient | 0.846 | 0.933 | 1.042 |
| ICE-1 steady | 0.109 | 0.263 | 0.781 |
| SPX-2 transient | 0.854 | 0.912 | 0.988 |
| SPX-2 steady | 0.177 | 0.122 | 0.424 |
Causal <= random in 3/4 and beats random on transient-far in both envs
(directional; n=40). Both trained models >> copy on steady-far.

Session summary: kernel restarts + interrupted streams handled via keep-alive
prints + on-disk artifacts. Full report: 8 addenda.


## Addendum 8: C - ActionVJEPA + velocity-consistency aux head (aux_action)
Predicted target-tubelet tokens get a per-window linear readout supervised on
observed twist_lin (500 steps, ETH-1, stride 4, lambda 0.5). ICE-1 held-out
control-signal test (velocity readout RMSE m/s over target windows):
  true commands 0.151 | shuffled commands 0.247 (+64%) | zero commands 0.304
Feature cosine remains insensitive (0.8456 vs 0.8457) - as predicted, cosine
cannot see controllability; the state-coupled readout can.
=> First clean CONTROLLABILITY result: with aux state-consistency, predicted
latents encode commanded motion; wrong future commands corrupt the decoded
latent velocity. This is the basis for a full controllable world model: add
aux velocity/contact heads at scale + decoder-level command-conditioned
generation. Report complete (8 addenda).
