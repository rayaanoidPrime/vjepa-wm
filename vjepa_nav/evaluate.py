"""S2 evaluation: decoded future-frame quality vs ground truth.

Metrics: PSNR, SSIM, LPIPS (optional) on held-out val/test splits, measured on
both teacher-latent and predicted-latent conditioning. Also writes montages.

Usage:
    python vjepa_nav/evaluate.py \
      --dataset-folder data/grandtour --clips-folder data/clips \
      --world-model ckpt/s1/step_50000.pt --decoder ckpt/s2/step_200000.pt \
      --split val --num-samples 64 --out-dir logs/s2_eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vjepa2"))

from vjepa_nav.data.clip_dataset import ClipDataset, collate_clips
from vjepa_nav.models.diffusion_decoder import ConditionalDiT, GaussianDiffusion
from vjepa_nav.models.goal_vjepa import GoalVJEPA
from vjepa_nav.utils.images import denormalize, psnr, save_tile, ssim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-folder", type=Path, default="data/grandtour")
    ap.add_argument("--clips-folder", type=Path, default="data/clips")
    ap.add_argument("--world-model", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-future", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default="logs/s2_eval")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wm = GoalVJEPA.build(args.num_future, args.img_size, 16, device)
    sd = torch.load(args.world_model, map_location=device, weights_only=False)
    wm.load_state_dict(sd["model"] if "model" in sd else sd)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    dit = ConditionalDiT(
        img_size=args.img_size,
        patch_size=8,
        hidden=args.hidden,
        depth=args.depth,
        num_heads=args.heads,
        cond_dim=wm.encoder.embed_dim,
        cond_len=(args.img_size // 16) ** 2,
    ).to(device)
    sd = torch.load(args.decoder, map_location=device, weights_only=False)
    dit.load_state_dict(sd["dit"] if "dit" in sd else sd)
    dit.eval()
    diffusion = GaussianDiffusion(1000, device=device)

    ds = ClipDataset(args.clips_folder, args.dataset_folder, args.split,
                     resolution=args.img_size, train=False, num_future=args.num_future)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_clips
    )

    try:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    except Exception:  # noqa: BLE001
        lpips_fn = None

    agg = {"psnr_t": [], "ssim_t": [], "psnr_p": [], "ssim_p": [], "lpips_t": [], "lpips_p": []}
    n = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if n >= args.num_samples:
                break
            start = batch["start"].to(device)
            future = batch["future"].to(device)
            goal = batch["goal"].to(device)
            B, K = future.shape[0], future.shape[1]
            gt = denormalize(future.reshape(B * K, 3, args.img_size, args.img_size))

            teacher_cond = wm.encode_targets(future).reshape(B * K, -1, wm.encoder.embed_dim)
            pred_cond = wm.predict_latents(start, goal).reshape(B * K, -1, wm.encoder.embed_dim)

            gen_t = denormalize(diffusion.ddim_sample(dit, teacher_cond, gt.shape, steps=args.ddim_steps))
            gen_p = denormalize(diffusion.ddim_sample(dit, pred_cond, gt.shape, steps=args.ddim_steps))

            for k in range(B * K):
                a_t, a_p, g = gen_t[k], gen_p[k], gt[k]
                agg["psnr_t"].append(psnr(a_t, g))
                agg["ssim_t"].append(ssim(a_t, g))
                agg["psnr_p"].append(psnr(a_p, g))
                agg["ssim_p"].append(ssim(a_p, g))
                if lpips_fn is not None:
                    agg["lpips_t"].append(lpips_fn(a_t.unsqueeze(0), g.unsqueeze(0)).item())
                    agg["lpips_p"].append(lpips_fn(a_p.unsqueeze(0), g.unsqueeze(0)).item())
            n += B

            if bi < 4:
                rows = []
                for b in range(B):
                    rows.append([denormalize(start[b, :, 0])] +
                                [gt[b * K + k] for k in range(K)] +
                                [gen_t[b * K + k] for k in range(K)] +
                                [gen_p[b * K + k] for k in range(K)])
                save_tile(rows, "start | gt | teacher-dec | pred-dec", args.out_dir, f"montage_{bi}.png")

    def _m(k):
        return (float(np.mean(agg[k])), float(np.std(agg[k])))

    print(f"[{args.split}] teacher-cond: PSNR {_m('psnr_t')}  SSIM {_m('ssim_t')}  LPIPS {_m('lpips_t')}")
    print(f"[{args.split}] pred-cond   : PSNR {_m('psnr_p')}  SSIM {_m('ssim_p')}  LPIPS {_m('lpips_p')}")


if __name__ == "__main__":
    main()
