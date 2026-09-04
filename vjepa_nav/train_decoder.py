"""S2: train the conditional diffusion decoder (latent h_t -> pixel frame).

Conditioning modes:
  teacher  : condition on frozen-encoder (teacher) latents of the GT frame
  pred     : condition on the goal-conditioned *predicted* latents
  mix      : 50/50 per sample

The V-JEPA decoder protocol: train on teacher latents first, then a short
distill pass on predicted latents.

Usage:
    python vjepa_nav/train_decoder.py \
      --dataset-folder data/grandtour --clips-folder data/clips \
      --world-model ckpt/s1/step_50000.pt --ckpt-dir ckpt/s2 \
      --cond-mode teacher
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vjepa2"))

from vjepa_nav.data.clip_dataset import ClipDataset, collate_clips
from vjepa_nav.models.diffusion_decoder import ConditionalDiT, GaussianDiffusion
from vjepa_nav.models.goal_vjepa import GoalVJEPA
from vjepa_nav.utils.checkpoint import find_latest, save_checkpoint, upload_to_hf
from vjepa_nav.utils.images import denormalize, save_tile


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-folder", type=Path, default="data/grandtour")
    ap.add_argument("--clips-folder", type=Path, default="data/clips")
    ap.add_argument("--world-model", type=Path, required=True, help="S1 GoalVJEPA checkpoint")
    ap.add_argument("--ckpt-dir", type=Path, default="ckpt/s2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cond-mode", choices=["teacher", "pred", "mix"], default="teacher")
    ap.add_argument("--num-future", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--diffusion-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--total-steps", type=int, default=200000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--sample-freq", type=int, default=5000)
    ap.add_argument("--ckpt-freq", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--hf-token", default="")
    args = ap.parse_args()
    seed_all(args.seed)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- world model (frozen) ---------------------------------------------
    wm = GoalVJEPA.build(args.num_future, args.img_size, 16, device)
    sd = torch.load(args.world_model, map_location=device, weights_only=False)
    wm.load_state_dict(sd["model"] if "model" in sd else sd)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    print(f"[world-model] loaded {args.world_model}")

    # ---- decoder -----------------------------------------------------------
    dit = ConditionalDiT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        hidden=args.hidden,
        depth=args.depth,
        num_heads=args.heads,
        cond_dim=wm.encoder.embed_dim,
        cond_len=(args.img_size // 16) ** 2,
    ).to(device)
    diffusion = GaussianDiffusion(args.diffusion_steps, device=device)
    opt = torch.optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=args.wd)

    step = 0
    if args.resume:
        ckpt = find_latest(args.ckpt_dir, "step")
        if ckpt is not None:
            sd = torch.load(ckpt, map_location=device, weights_only=False)
            dit.load_state_dict(sd["dit"])
            opt.load_state_dict(sd["opt"])
            step = sd.get("step", 0)
            print(f"[resume] {ckpt} at step {step}")

    train_ds = ClipDataset(
        args.clips_folder, args.dataset_folder, "train",
        resolution=args.img_size, train=True, num_future=args.num_future,
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_clips, drop_last=True, pin_memory=True,
    )
    loader_iter = iter(loader)

    print(f"decoder params: {sum(p.numel() for p in dit.parameters()):,}")
    print(f"train samples: {len(train_ds):,} | cond-mode: {args.cond_mode}")

    t_start = time.time()
    while step < args.total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        start = batch["start"].to(device)
        future = batch["future"].to(device)
        goal = batch["goal"].to(device)
        B, K = future.shape[0], future.shape[1]
        frames = future.reshape(B * K, 3, args.img_size, args.img_size)  # (BK,3,H,W)

        # ---- conditioning latents ------------------------------------------
        with torch.no_grad():
            mode = args.cond_mode
            if mode == "mix":
                mode = random.choice(["teacher", "pred"])
            if mode == "teacher":
                cond = wm.encode_targets(future)  # (B, K*P, D)
            else:  # pred
                z_pred = wm.predict_latents(start, goal)  # (B, K*P, D)
                cond = z_pred
            cond = cond.reshape(B * K, -1, cond.shape[-1])

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            t = torch.randint(0, args.diffusion_steps, (B * K,), device=device)
            noise = torch.randn_like(frames)
            x_t = diffusion.q_sample(frames, t, noise)
            eps_pred = dit(x_t, t, cond)
            loss = nn.functional.mse_loss(eps_pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.log_freq == 0:
            el = time.time() - t_start
            print(f"[{step:6d}/{args.total_steps}] loss {loss.item():.5f} {el/args.log_freq*1e3:.0f} ms/step", flush=True)
            t_start = time.time()

        if step % args.sample_freq == 0:
            _sample(dit, diffusion, wm, loader, device, args, save=True)

        if step % args.ckpt_freq == 0:
            p = args.ckpt_dir / f"step_{step}.pt"
            save_checkpoint({"dit": dit.state_dict(), "opt": opt.state_dict(), "step": step}, p)
            if args.hf_repo:
                try:
                    upload_to_hf(p, args.hf_repo, args.hf_token)
                except Exception as e:  # noqa: BLE001
                    print(f"[hf] {e}", flush=True)
            keep = sorted(args.ckpt_dir.glob("step_*.pt"))
            for old in keep[:-5]:
                old.unlink(missing_ok=True)

    with open(args.ckpt_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print("done.")


@torch.no_grad()
def _sample(dit, diffusion, wm, loader, device, args, save: bool = False) -> None:
    """Generate a few future frames with DDIM (teacher + pred conditioning)."""
    dit.eval()
    batch = next(iter(loader))
    start = batch["start"][:2].to(device)
    future = batch["future"][:2].to(device)
    goal = batch["goal"][:2].to(device)
    B, K = future.shape[0], future.shape[1]
    frames_gt = future.reshape(B * K, 3, args.img_size, args.img_size)

    teacher_cond = wm.encode_targets(future).reshape(B * K, -1, wm.encoder.embed_dim)
    pred_cond = wm.predict_latents(start, goal).reshape(B * K, -1, wm.encoder.embed_dim)

    gen_t = diffusion.ddim_sample(dit, teacher_cond, frames_gt.shape, steps=50)
    gen_p = diffusion.ddim_sample(dit, pred_cond, frames_gt.shape, steps=50)

    rows = []
    for b in range(B):
        row = [denormalize(start[b, :, 0])]
        row += [denormalize(frames_gt[b * K + k]) for k in range(K)]
        row += [denormalize(gen_t[b * K + k]) for k in range(K)]
        row += [denormalize(gen_p[b * K + k]) for k in range(K)]
        rows.append(row)
    save_tile(rows, "start | gt | teacher | pred", args.ckpt_dir / "samples", f"sample_{getattr(args, '_sample_n', 0)}.png")
    args._sample_n = getattr(args, "_sample_n", 0) + 1
    dit.train()
