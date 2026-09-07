#!/usr/bin/env python3
"""Decoder fine-tune on world-model latents (step-2b "predfeat" style).

Warm-starts a decoder head (default: the published vm2m dv3vitl decoder for the
Path A dv3 latent space) and fine-tunes it on the GrandTour domain, supervising
the head on (a) GT-encoder reconstruction and (b) the *frozen* predictor's
shift-1 rollout latents. The result decodes predicted futures far more honestly
than the out-of-domain pretrained decoder.

Requires an already-trained stage-1 world model (see GT_WM_TAG) and a decoder
checkpoint to warm-start from (see GT_DEC_CKPT, defaults to the published
vm2m_lpips_dv3vitl_256_INet decoder).

Environment (same roots as viz_action_rollout.py):
  JEPAWM_HOME / JEPAWM_DSET / JEPAWM_LOGS / TORCH_HOME / JEPAWM_OSSCKPT
  GT_WM_TAG     world-model run (config + ckpt) under $JEPAWM_LOGS/grandtour_sweep
  GT_DEC_CKPT   warm-start head checkpoint
  GT_DEC_YAML   head architecture config yaml
  GT_HEAD_TAG   output run name (default gt_v3_step2b_predfeat)
  GT_HEAD_LR / GT_HEAD_EPOCHS / GT_HEAD_IPE   lr, epochs, iters per epoch

Run from grandtour/scripts:
  python finetune_decoder.py
Output: $JEPAWM_LOGS/grandtour_sweep/<GT_HEAD_TAG>/jepa-latest_image_head.pth.tar
(same format viz_action_rollout.py loads via GT_DEC_CKPT).
"""
import os
import sys
import time
import warnings
from pathlib import Path

import _paths

warnings.filterwarnings("ignore")
os.chdir(str(_paths.JEPAWM_HOME))
sys.path.insert(0, str(_paths.JEPAWM_HOME))
sys.path.insert(0, str(_paths.Path(__file__).resolve().parent))
os.environ.setdefault("TORCH_HOME", str(_paths.TORCH_HOME))

import numpy as np
import torch
import yaml

from app.plan_common.datasets.grandtour_dset import GrandTourVideoDataset
from app.plan_common.datasets.transforms import make_transforms
from viz_action_rollout import build, DEVICE

LR = float(os.environ.get("GT_HEAD_LR", "5e-4"))
EPOCHS = int(os.environ.get("GT_HEAD_EPOCHS", "3"))
IPE = int(os.environ.get("GT_HEAD_IPE", "150"))
OUTTAG = os.environ.get("GT_HEAD_TAG", "gt_v3_step2b_predfeat")


def main():
    torch.set_grad_enabled(False)
    tag = os.environ.get("GT_WM_TAG")
    assert tag, "set GT_WM_TAG (stage-1 run under $JEPAWM_LOGS/grandtour_sweep)"
    assert "GT_DEC_CKPT" in os.environ, "set GT_DEC_CKPT (warm-start head checkpoint)"
    ds, tr, inv, wm, head = build()          # frozen WM + warm-started head
    head.model.train()
    for p in head.model.parameters():
        p.requires_grad_(True)

    cfg = yaml.safe_load(open(_paths.JEPAWM_LOGS / "grandtour_sweep" / tag / f"{tag}.yaml"))
    da = cfg["data_aug"]
    trA = make_transforms(img_size=cfg["data"]["img_size"], random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0), random_resize_scale=(1.0, 1.0),
        auto_augment=False, motion_shift=False, reprob=0.0, normalize=da["normalize"])
    dsA = GrandTourVideoDataset(
        data_path=_paths.JEPAWM_DSET / "grand_tour",
        frames_per_clip=cfg["data"]["droid"]["dataset_fpcs"][0],
        fps=cfg["data"]["droid"]["fps"], transform=trA, seed=0)

    opt = torch.optim.AdamW(head.model.parameters(), lr=LR, weight_decay=0.05)
    outdir = _paths.JEPAWM_LOGS / "grandtour_sweep" / OUTTAG
    outdir.mkdir(parents=True, exist_ok=True)
    gstep = 0
    for ep in range(EPOCHS):
        acc = {"recon": 0.0, "predfeat": 0.0}
        t0 = time.time()
        for _it in range(IPE):
            with torch.no_grad():
                obs, A, _, _ = dsA[0]                    # random 12-frame window (normalized)
                vis = obs["visual"].unsqueeze(0).to(DEVICE)
                A = A.unsqueeze(0).to(DEVICE)
                target_rgb = head.preprocess_rgb(vis)
                vf, _, _ = wm.encode({"visual": vis}, A)  # (1,T,V,16,16,1024)
                pf, _, _ = wm.forward_pred(vf, A, None)   # shift-1 preds over the window
            torch.set_grad_enabled(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_recon = head.compute_loss(vf.float(), target_rgb, global_step=gstep)["loss"]
                l_pred = head.compute_loss(pf[:, :-1].float().detach(),
                                           target_rgb[:, 1:], global_step=gstep)["loss"]
                loss = l_recon + l_pred
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.set_grad_enabled(False)
            acc["recon"] += float(l_recon.detach())
            acc["predfeat"] += float(l_pred.detach())
            gstep += 1
            if _it % 25 == 0:
                print(f"ep {ep + 1} it {_it} recon {l_recon.item():.4f} "
                      f"predfeat {l_pred.item():.4f}", flush=True)
        print(f"== epoch {ep + 1}: recon-avg {acc['recon'] / IPE:.4f} "
              f"predfeat-avg {acc['predfeat'] / IPE:.4f} ({(time.time() - t0) / IPE:.2f}s/it)",
              flush=True)
        sd = {("module." + k): v for k, v in head.model.state_dict().items()}
        torch.save({"model": sd, "epoch": ep + 1, "opt": None, "scaler": None},
                   outdir / "jepa-latest_image_head.pth.tar")
    print("FT_DONE", flush=True)


if __name__ == "__main__":
    main()
