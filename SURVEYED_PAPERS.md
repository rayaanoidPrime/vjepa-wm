# Surveyed papers — what to steal for the quadruped world model

Context: these papers were surveyed while building **a latent video world model
for quadruped locomotion** (V-JEPA 2.1 on the ANYmal-D / Grand Tour dataset).
Each entry: **claim → mechanism → what to steal**, with a priority tag
(`Now` / `P1…P6` = phase in `README §2 Plan`, `later`/`context` = stretch).
Retrieved 2026-09 from arXiv.

Big-picture takeaway that shapes the whole project:
> The best world model for quadruped locomotion is a **world-action model
> (WAM)** that (1) predicts the future *given* the command/proprio stream, not
> just pixels; (2) actually **tracks hidden physical state** (contact, terrain,
> body twist) — pixel loss alone never supervises this; (3) is decodable to
> pixels only as an *evaluation/planning interface*, not as the objective; and
> (4) is designed so a **planner / value function** can be bolted on top
> (imagination-time search) — that is where world models pay off for control.

---

## Backbone / representation

### [V-JEPA 2](https://arxiv.org/abs/2404.08471) — the concept behind your backbone
- **Claim:** predicting *masked* latent features of future/occluded regions from
  an unmasked context (EMA target encoder, L1/L2 in LayerNorm space, no pixel
  loss) yields transferable video representations.
- **Mechanism:** student encoder + predictor vs a frozen EMA target; block
  masks; dense predictive loss on target tokens. V-JEPA 2.1 = released
  [checkpoints + code](https://github.com/facebookresearch/vjepa2) (ViT-B/L/G).
- **What to steal:** everything — it *is* the base. **Priority: Now.**

### [LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics](https://arxiv.org/abs/2608.27395)
- **Claim:** the V-JEPA trick needs neither EMA asymmetry nor a capacity-limited
  predictor. A single encoder + projector trained with an invariance loss on
  global/local views + **SIGReg** is provably collapse-free, matches/surpasses
  V-JEPA 2 at **5.6–20.8× less pretraining compute** (ViT-S/B/L), and supports
  **block-causal attention** at no accuracy cost.
- **What to steal (Now):** the **block-causal encoder** design for the causal
  task — context = frames ≤ t, so the encoder only attends within the past →
  cheap autoregressive rollout, no re-encoding of full clips.
  **(Later):** if a backbone is ever pretrained from scratch on robot video,
  swap the random-mask objective for LeVJEPA's (SIGReg); uniform token dropping
  also gives inference-time compute flexibility (same idea as Flex-π).

## The world-action / next-token family (controllable futures)

### [Flex-π: A Multi-Stream World-Action Model with Compute Flexibility](https://arxiv.org/abs/2608.10860)
- **Claim:** a frozen video VAE that encodes RGB *also encodes 3D pointmaps
  almost losslessly* (free lunch, no extra training); supervising a 6B WAM on
  RGB + geometry + DINO semantics in a **shared latent space**, denoised jointly
  with actions inside a Mixture-of-Transformers, beats baselines 2–7× on precise
  bimanual tasks at π0.5 speed.
- **Mechanism:** all visual signals in one latent + joint denoising with
  actions; **per-stream dropout + cross-modality forcing** → one checkpoint runs
  on any subset of streams, from fast action-only to full joint generation.
- **What to steal (Now, cheap):** the **multi-stream mindset for ANYmal-D**
  (RGB + depth + IMUs + proprio + masks + RTK-GPS all exist per mission):
  one dynamics latent space; every auxiliary stream an optional, droppable
  conditioning signal.
  **(Later):** if you go generative, predict inside a frozen VAE-latent shared
  across modalities rather than raw JEPA latents.

### [Zero-WAM: In-Context World-Action Modeling from Human Videos](https://arxiv.org/abs/2608.26103)
- **Claim:** a **human video in context** is a natural task specification;
  a causal video-action model with an *in-context future chunk prediction (IFP)*
  objective executes unseen tasks (+29.5 pts vs the best video-action baseline).
- **What to steal (Later):** in-context route/terrain video → "walk where the
  video walks"; reuse IFP to stop the model copying the prompt's visual identity
  instead of its *motion*.

### [Hydra-0: Action Flow for Generalist World Modeling and Control](https://arxiv.org/abs/2608.18077)
- **Claim:** representing actions as **pixel motion (action flow)** is a shared
  *visual* interface across embodiments; cuts robot-motion error 90.4% /
  object-motion error 60.2% vs action-conditioned baselines; has an **emergent
  inverse mode** (object flow → compatible robot motion → executable actions).
- **What to steal (Now, cheap):** the **evaluation metric** — *robot-motion
  error* / *object-motion error* on motion fields, not raw pixels. ANYmal-D
  ships `hdr_front_mask` (robot pixel masks): measure prediction quality on the
  robot foreground and its motion, not just whole-frame PSNR.
  **(Later):** action-flow is overkill for legged robots (your action stream is
  commands/proprio); keep as a benchmark-only idea.

### [Humanoid Locomotion as Next Token Prediction](https://arxiv.org/abs/2402.19469)
- **Claim:** real-world humanoid control = **autoregressive next-token
  prediction over modality-aligned sensorimotor trajectories**; per-input-token,
  predict the next token of the *same modality*; handles missing modalities
  (video without actions); trains on sim policies + MoCap + YouTube; walks
  zero-shot from 27 h of walking data; generalizes to unseen commands.
- **What to steal (Now, conceptually):** (1) **modality-aligned prediction** —
  predict proprio-with-proprio, video-latent-with-video-latent,
  command-with-command; cross-modality only through attention; (2) the
  **missing-modality trick** — train jointly with and without action labels so
  non-ANYmal quadruped video can be absorbed later.

## Planning / self-improvement on top of the world model

### [VIP-Loco: A Visually Guided Infinite-Horizon Planning Framework for Legged Locomotion](https://arxiv.org/abs/2603.14345)
- **Claim:** MPC is interpretable but blind to high-dim perception; RL adapts to
  visuals but doesn't plan. VIP-Loco trains an **internal model mapping
  proprio+depth into compact kinodynamic features** used by RL *and* runs
  **infinite-horizon MPC in that learned feature space** at deployment (slopes,
  stairs, gaps; Go1, Cassie, TronA1-W).
- **What to steal (Later, big idea):** the world model can output a
  **planning-friendly kinodynamic latent** (body + feet pose/velocity), not only
  video tokens; if the latent is interpretable you can plan over it with an
  MPC-style optimizer — evidence for *interpretable* latents for legged robots.

### [Q-Learning With World Models (QWM)](https://arxiv.org/abs/2608.17163)
- **Claim:** training policy/value on imagined rollouts → compounding bias and
  poor scaling to visual robotics. QWM keeps policy + Q-function trained
  **only on real transitions** and uses the world model for **test-time search
  over imagined trajectories** to pick high-value actions; beats SOTA on
  Robomimic/LIBERO in sample efficiency and performance.
- **What to steal (Later, design rule):** separate the *predictive* world model
  from *value/policy* machinery; do **imagination-time action selection**
  (search over command rollouts scored by a value net trained on real data),
  not policy gradients through the model.

### [Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning](https://arxiv.org/abs/2608.21204)
- **Claim:** BC can't self-improve; RL fine-tuning of huge models doesn't scale.
  Give a big BC policy a **small off-policy Q-function**: (a) single-step
  **Q-weighted averaging over BC draws** at inference; (b) online, fine-tune
  *only the Q*, absorbing failures. 10 iterations lift LIBERO-10 93→99% and a
  real bimanual stack-cups 40→90% from its own rollouts, BC frozen.
- **What to steal (Later):** the cheapest self-improvement loop for locomotion —
  freeze the world model/action head, learn a small Q over (latent-state,
  command), let deployment rollouts improve the *selector*, not the generator.

## What can actually go wrong (science of the world model)

### [Can Video World Models Track Unobserved World States?](https://arxiv.org/abs/2608.30692)
- **Claim (the most important paper for this project):** **video fidelity ≠
  hidden-state tracking.** In an action-conditioned *video Shell Game*, AR
  Transformers, Mamba, and linear attention (nonnegative eigenvalues) fit 5
  swaps then **fall toward chance when extrapolating while still rendering
  plausible video** — the pixel target never supervises the unseen state, so the
  state lives in the *architecture* (an append-only KV cache), re-derived from
  history each chunk. Only mechanisms that **carry state across chunks and
  revise it in place** (linear attention with negative eigenvalues; TTT with a
  nonlinear fast weight) extrapolate.
- **What to steal (Now — defines the evaluation and the architecture ceiling):**
  1. The pixel-decoding experiment will *not* tell you whether the model tracks
     the quadruped's hidden state (contact phase, foot placement, terrain class,
     commanded-vs-executed twist). Add **hidden-state probes**: readouts from
     latents → proprio/contact/twist/odom at future times, plus extrapolation
     tests (double horizon, novel terrain/commands).
  2. If hidden-state tracking is the goal, **supervise it directly** (aux
     losses on `anymal_state_*`) and/or give the model a **recurrent /
     stateful mechanism** for rollout; a plain next-tubelet transformer is
     expected to degrade on long horizons.

### [Learning How the World Evolves: Extrapolative Video World Models via Latent Dynamics Reasoning (LDR)](https://arxiv.org/abs/2608.09926)
- **Claim:** video diffusion fits pixels but not *transitions*. LDR casts the
  latent transition as **explicit kinematic integration** (numerically integrate
  low-order dynamics; regress only the high-order residual) on a **structured
  latent**; OOD error gap vs diffusion is >20× smaller with 26× fewer params,
  143× faster; extrapolates under severe shifts (e.g. red ball → blue square,
  reversed direction).
- **What to steal (Later, high-value):** give the latent dynamics head an
  inductive bias matching rigid-body kinematics — model **body/foot positions &
  velocities** as low-order integrated states, let the transformer predict only
  residuals/forces. This complements VIP-Loco's kinodynamic planning latent and
  is exactly where pure-JEPA video models are weak (extrapolation).

## Generative-model plumbing (only if/when going end-to-end generative)

### [GenFirst: Generation Before Reconstruction for End-to-End Latent Generative Modeling](https://arxiv.org/abs/2608.29335)
- **Claim:** joint VAE+generator training collapses because reconstruction is
  fast/strongly supervised while generation is slow; the KL **entropy term**
  prevents collapse. **GenFirst**: generative objective first under weak
  reconstruction pressure, then strengthen reconstruction → first
  collapse-free end-to-end latent generative training (SiT gFID 0.97 @256).
- **What to steal (Later):** if co-training the decoder with the predictor (the
  current stage-2 combined loss), use a GenFirst schedule instead of a fixed
  `0.5·latent + 1.0·RGB` mix and keep an entropy/regularization term so the
  latent doesn't collapse into decode-only information.

### [Active Flow Expansion for Out-of-Distribution Discovery](https://arxiv.org/abs/2606.08802)
- **Claim:** distribution matching can't reach OOD designs; instead expand the
  model's **generable set** by continued pretraining on *synthetic* samples
  validated by a **verifier**, with statistical guarantees.
- **What to steal (Later, moonshot):** the world model will never see enough
  cliffs/sand/ice in real data — but you have a simulator. Use a **dynamics
  verifier (sim + tracking controller)** to generate novel trajectories and
  continue training the world model on what the verifier accepts: the
  world-model analogue of FetchMan's sim-experience scaling.

## Data-scale / policy lessons from adjacent papers

### [FetchMan: Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences](https://arxiv.org/abs/2608.17027)
- **Claim:** sim→real for visual whole-body control at scale: BC on synthetic
  demos hits a ceiling regardless of data; **RL (Flow-GRPO) breaks it**; 150k
  simulated scenes → 73.3% real zero-shot reach-and-pick on Unitree G1.
- **What to steal (Context):** your pipeline is real-data-only today; ANYmal-D
  gives ~50k steps / 150k images. Long-term: real ANYmal-D (rich, realistic) +
  sim (arbitrary terrains/commands) with the world model as the **common latent
  space** (cf. Flex-π). Also expect to need an RL/GRPO pass after BC for the
  action head (BC-ceiling lesson).

---

## Quick assignment table

| Paper | arXiv | Steal now | Steal later | Priority |
|---|---|---|---|---|
| V-JEPA 2 (concept) | [2404.08471](https://arxiv.org/abs/2404.08471) | backbone + dense latent loss | — | 🔴 base |
| LeVJEPA | [2608.27395](https://arxiv.org/abs/2608.27395) | block-causal encoder, token dropping | SIGReg pretraining | 🟠 P2 |
| Hydra-0 | [2608.18077](https://arxiv.org/abs/2608.18077) | robot-motion / foreground eval metrics | action-flow interface | 🟠 P1 eval |
| Flex-π | [2608.10860](https://arxiv.org/abs/2608.10860) | multi-stream, per-stream dropout | shared frozen-VAE latent | 🟠 P3 |
| Zero-WAM | [2608.26103](https://arxiv.org/abs/2608.26103) | — | in-context video task spec | ⚪ later |
| Next-Token Locomotion | [2402.19469](https://arxiv.org/abs/2402.19469) | modality-aligned prediction, missing-modality training | — | 🟢 design |
| VIP-Loco | [2603.14345](https://arxiv.org/abs/2603.14345) | — | kinodynamic latent + MPC | ⚪ later |
| QWM | [2608.17163](https://arxiv.org/abs/2608.17163) | — | test-time search, real-only value | ⚪ P5 |
| Beyond Imitation | [2608.21204](https://arxiv.org/abs/2608.21204) | — | small Q on top, Q-weighted selection | ⚪ P5 |
| Track Unobserved States | [2608.30692](https://arxiv.org/abs/2608.30692) | hidden-state probes + aux proprio losses + extrapolation tests | recurrent/stateful rollout | 🔴 P1 eval |
| LDR (world evolves) | [2608.09926](https://arxiv.org/abs/2608.09926) | — | kinematic-integration latent head | ⚪ P6 |
| GenFirst | [2608.29335](https://arxiv.org/abs/2608.29335) | — | end-to-end decoder schedule | ⚪ P4 |
| Active Flow Expansion | [2606.08802](https://arxiv.org/abs/2606.08802) | — | verifier-driven synthetic expansion | ⚪ moonshot |
| FetchMan | [2608.17027](https://arxiv.org/abs/2608.17027) | — | sim-scene scale + RL-after-BC | ⚪ context |

## Data / code references
- ANYmal-D robot + Grand Tour dataset: <https://grand-tour.leggedrobotics.com/> ·
  HF: `leggedrobotics/grand_tour_dataset` (gated — request access).
- V-JEPA 2.1 checkpoints & code: <https://github.com/facebookresearch/vjepa2>
- LiMo (data recipe reference): <https://github.com/leggedrobotics/less-is-more>
