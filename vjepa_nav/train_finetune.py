"""S1: fine-tune the goal-conditioned V-JEPA 2.1 latent world model.

Predicts the latent features of K future frames from (start frame, SE(2) goal),
regressing against the frozen target encoder's features of the actual future
frames (V-JEPA 2.1 dense predictive loss, L1).

molab-safe: checkpoints every --ckpt-freq steps (default 30 min of compute),
plus resume-from-latest so a 12h session can be restarted manually.

Usage:
    python vjepa_nav/train_finetune.py \
      --dataset-folder data/grandtour --clips-folder data/clips \
      --ckpt-dir ckpt/s1 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vjepa2"))  # for app.vjepa_2_1 / src imports

from vjepa_nav.data.clip_dataset import ClipDataset, collate_clips
from vjepa_nav.models.goal_vjepa import GoalVJEPA, vjepa_loss
from vjepa_nav.utils.checkpoint import (
    VJEPA2_1_VITL_URL,
    download_file,
    find_latest,
    load_pretrained_vjepa2_1,
    save_checkpoint,
    upload_to_hf,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WarmupCosine:
    def __init__(self, lr, warmup_steps, total_steps, final_lr=0.0):
        self.lr = lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = total_steps
        self.final_lr = final_lr

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        t = min((step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps), 1.0)
        return self.final_lr + 0.5 * (self.lr - self.final_lr) * (1 + math.cos(math.pi * t))


def make_optimizer(model: GoalVJEPA, lr: float, ln_lr: float, wd: float):
    groups = [
        {"params": [], "lr": lr, "base_lr": lr, "weight_decay": wd},
        {"params": [], "lr": ln_lr, "base_lr": ln_lr, "weight_decay": 0.0},
    ]
    ln_names = set()
    for m in model.encoder.modules():
        if isinstance(m, nn.LayerNorm):
            ln_names.update(p for p in m.parameters())
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p in ln_names:
            groups[1]["params"].append(p)
        else:
            groups[0]["params"].append(p)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def log_val(model, loader, device, steps: int = 200) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= steps:
                break
            start = batch["start"].to(device)
            future = batch["future"].to(device)
            goal = batch["goal"].to(device)
            z_pred, h = model(start, future, goal)
            losses.append(vjepa_loss(z_pred, h).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-folder", type=Path, default="data/grandtour")
    ap.add_argument("--clips-folder", type=Path, default="data/clips")
    ap.add_argument("--ckpt-dir", type=Path, default="ckpt/s1")
    ap.add_argument("--ckpt-url", default=VJEPA2_1_VITL_URL, help="pretrained ViT-L checkpoint URL")
    ap.add_argument("--ckpt-path", type=Path, default="data/vjepa2_1_vitl_dist_vitG_384.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num-future", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ln-lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--total-steps", type=int, default=50000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--val-freq", type=int, default=2000)
    ap.add_argument("--ckpt-freq", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=239)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-repo", default="", help="push checkpoints to this HF model repo")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()
    seed_all(args.seed)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- data ---------------------------------------------------------------
    train_ds = ClipDataset(args.clips_folder, args.dataset_folder, "train",
                           resolution=args.img_size, train=True, num_future=args.num_future)
    val_ds = ClipDataset(args.clips_folder, args.dataset_folder, "val",
                         resolution=args.img_size, train=False, num_future=args.num_future)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_clips, drop_last=True, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_clips, drop_last=False, pin_memory=True,
    )

    # ---- model ---------------------------------------------------------------
    model = GoalVJEPA.build(args.num_future, args.img_size, args.patch_size, device)
    model.freeze_encoder_except_ln()
    model.train()

    step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = find_latest(args.ckpt_dir)
        if ckpt is not None:
            sd = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(sd["model"])
            step = sd.get("step", 0)
            best_val = sd.get("best_val", float("inf"))
            print(f"[resume] loaded {ckpt} at step {step}")
    else:
        if not args.ckpt_path.exists():
            print(f"[download] {args.ckpt_url}")
            download_file(args.ckpt_url, args.ckpt_path)
        report = load_pretrained_vjepa2_1(model, args.ckpt_path)
        print("[pretrained] encoder_missing:", report["encoder_missing"])
        print("[pretrained] predictor_missing:", report["predictor_missing"])
        print("[pretrained] predictor_unexpected:", report["predictor_unexpected"])

    optimizer = make_optimizer(model, args.lr, args.ln_lr, args.wd)
    if args.resume and step > 0:
        sd = torch.load(find_latest(args.ckpt_dir), map_location=device, weights_only=False)
        if "optimizer" in sd:
            optimizer.load_state_dict(sd["optimizer"])
            print("[resume] optimizer restored")

    total_steps = args.total_steps
    lr_sched = WarmupCosine(args.lr, args.warmup_steps, total_steps)
    use_amp = device.type == "cuda"
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"train samples: {len(train_ds):,} | val samples: {len(val_ds):,}")

    loader_iter = iter(train_loader)
    t_start = time.time()
    while step < total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        start = batch["start"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        goal = batch["goal"].to(device, non_blocking=True)

        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * (lr_sched(step) / args.lr)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            z_pred, h = model(start, future, goal)
            loss = vjepa_loss(z_pred, h)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        step += 1
        if step % args.log_freq == 0:
            el = time.time() - t_start
            print(
                f"[{step:6d}/{total_steps}] loss {loss.item():.4f} "
                f"lr {lr_sched(step):.2e} {el/args.log_freq*1e3:.0f} ms/step",
                flush=True,
            )
            t_start = time.time()

        if step % args.val_freq == 0:
            v = log_val(model, val_loader, device)
            if v < best_val:
                best_val = v
                save_checkpoint({"model": model.state_dict(), "step": step, "best_val": best_val},
                                args.ckpt_dir / "best.pt")
                print(f"[val] step {step} val_loss {v:.4f} (best, saved)", flush=True)
            else:
                print(f"[val] step {step} val_loss {v:.4f} (best {best_val:.4f})", flush=True)

        if step % args.ckpt_freq == 0:
            ckpt_path = args.ckpt_dir / f"step_{step}.pt"
            save_checkpoint({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_val": best_val,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}", flush=True)
            if args.hf_repo:
                try:
                    upload_to_hf(ckpt_path, args.hf_repo, args.hf_token)
                    print("[hf] uploaded", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[hf] upload failed: {e}", flush=True)
            keep = sorted(args.ckpt_dir.glob("step_*.pt"))
            for p in keep[:-5]:
                p.unlink(missing_ok=True)

    print("done.")
    with open(args.ckpt_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


if __name__ == "__main__":
    main()
