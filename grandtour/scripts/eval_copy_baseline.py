#!/usr/bin/env python3
"""Copy-baseline + rollout evaluation for a trained GrandTour world model.

Measures, over windows sampled from the trained dataset:
  1-step teacher-forcing error, its 'copy-last-frame' baseline, and
  autoregressive rollout error vs the copy baseline at horizons h = 1..H.
Rollout error beating the no-dynamics baseline at h >= 2 is the sign the
model learned action-conditioned dynamics (guide section 10).

Usage (after training gt_v0):
  python eval_copy_baseline.py [--ckpt PATH] [--nwin 24]
Env: JEPAWM_HOME / JEPAWM_DSET / JEPAWM_LOGS (see grandtour/env.sh)
"""
import argparse
import os
import sys
import warnings

import _paths

warnings.filterwarnings("ignore")

os.chdir(str(_paths.JEPAWM_HOME))
sys.path.insert(0, str(_paths.JEPAWM_HOME))
os.environ.setdefault("TORCH_HOME", str(_paths.TORCH_HOME))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.plan_common.datasets.transforms import make_transforms
from app.plan_common.datasets.grandtour_dset import GrandTourVideoDataset
from app.vjepa_wm.utils import init_video_model
from app.vjepa_wm.video_wm import VideoWM

DEVICE = "cuda:0"
CFG_REL = os.environ.get("GT_CFG_REL", "configs/vjepa_wm/grandtour_sweep/gt_v0_12f_fps5_r224_dv2vits_AdaLN_d6_2roll_1n.yaml")
RUN_NAME = os.environ.get("GT_RUN", "gt_v0_12f_fps5_r224_dv2vits_AdaLN_d6_2roll_1n")
# e.g. GT_RUN=gt_v1_12f_fps5_r224_dv2vitl_AdaLN_d12c_2roll_1n to evaluate the scaled model
BATCH = 4
CTX = 4        # context frames for the rollout
HROLL = 6      # rollout steps


def build(ckpt_path):
    cfg = yaml.safe_load(open(os.path.join(_paths.JEPAWM_HOME, CFG_REL)))
    da = cfg["data_aug"]
    d = cfg["data"]
    m = cfg["model"]
    transform = make_transforms(img_size=d["img_size"],
        random_horizontal_flip=da["random_horizontal_flip"],
        random_resize_aspect_ratio=tuple(da["random_resize_aspect_ratio"]),
        random_resize_scale=tuple(da["random_resize_scale"]),
        auto_augment=da["auto_augment"], motion_shift=da["motion_shift"],
        reprob=da["reprob"], normalize=da["normalize"])
    ds = GrandTourVideoDataset(data_path=str(_paths.JEPAWM_DSET / "grand_tour"),
        frames_per_clip=d["droid"]["dataset_fpcs"][0], fps=d["droid"]["fps"],
        transform=transform, seed=7)
    model_kwargs = {k: v for k, v in m.items() if k not in
        ("rollout_cfg", "heads_cfg", "pretrained_path", "visual_encoder", "action_encoder",
         "proprio_encoder", "predictor", "wm_encoding", "attn")}
    model_kwargs.update(m["visual_encoder"])
    model_kwargs.update(m["action_encoder"])
    model_kwargs.update(m["proprio_encoder"])
    model_kwargs.update(m["predictor"])
    model_kwargs.update(dict(device=DEVICE, img_size=d["img_size"],
        action_dim=ds.action_dim, proprio_dim=ds.proprio_dim,
        cfgs_attn_pattern=m.get("attn"), use_proprio=False, use_action=True))
    predictor, encoder, ae, pe = init_video_model(**model_kwargs)
    wm_kwargs = dict(device=DEVICE, encoder=encoder, predictor=predictor, action_encoder=ae,
        proprio_encoder=pe, action_dim=ds.action_dim, proprio_dim=ds.proprio_dim,
        use_proprio=False, use_action=True, action_tokens=1, proprio_tokens=0,
        grid_size=m.get("grid_size", 16), tubelet_size_enc=1,
        action_conditioning=m.get("action_conditioning", "token"),
        proprio_encoding=m.get("proprio_encoding", "feature"),
        enc_type=m["visual_encoder"]["enc_type"], pred_type=m["predictor"]["pred_type"],
        action_encoder_inpred=m["action_encoder"]["action_encoder_inpred"],
        proprio_encoder_inpred=False, action_skip=1, frameskip=1,
        img_size=d["img_size"], heads={}, scaler=None, optimizer=None,
        clip_grad=1.0, mixed_precision=False, use_radamw=False,
        cfgs_loss=cfg["loss"], dup_image=False, batchify_video=True, normalize_reps=False)
    wm = VideoWM(**wm_kwargs).to(DEVICE)
    for p in wm.parameters():
        p.requires_grad_(False)
    ck = torch.load(ckpt_path, map_location="cpu")
    wm.predictor.load_state_dict({k.replace("module.", ""): v for k, v in ck["predictor"].items()})
    wm.eval()
    print(f"loaded predictor from {ckpt_path}")
    return ds, wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwin", type=int, default=24)
    ap.add_argument("--ckpt", default=str(_paths.JEPAWM_LOGS / "grandtour_sweep" / RUN_NAME / "jepa-latest.pth.tar"))
    args = ap.parse_args()
    ds, wm = build(args.ckpt)
    s1m = s1c = 0.0
    roll = {h: [0.0, 0.0] for h in range(1, HROLL + 1)}
    ns = 0
    with torch.no_grad():
        for start in range(0, args.nwin, BATCH):
            items = [ds[(start + i) % len(ds)] for i in range(BATCH)]
            B = len(items)
            vis = torch.stack([it[0]["visual"] for it in items]).to(DEVICE)
            act = torch.stack([it[1] for it in items]).to(DEVICE)
            video_features, _, action_features = wm.encode({"visual": vis}, act)
            B_, T, V, H, W, D = video_features.shape
            pred_vis, _, _ = wm.forward_pred(video_features, action_features, None)
            s1m += F.mse_loss(pred_vis[:, :-1], video_features[:, 1:]).item() * B
            s1c += F.mse_loss(video_features[:, :-1], video_features[:, 1:]).item() * B
            feats = video_features[:, :CTX].clone()
            acts = act[:, :CTX].clone()
            for h in range(1, HROLL + 1):
                out, _, _ = wm.predictor(feats, acts, None)
                nxt6 = out[:, -1].view(B, 1, H, W, D).unsqueeze(2)
                tgt_idx = CTX - 1 + h
                if tgt_idx < T:
                    tgt = video_features[:, tgt_idx:tgt_idx + 1]
                    roll[h][0] += F.mse_loss(nxt6, tgt).item() * B
                    copyf = video_features[:, CTX - 1:CTX].expand_as(tgt)
                    roll[h][1] += F.mse_loss(copyf, tgt).item() * B
                if tgt_idx < T - 1:
                    feats = torch.cat([feats, nxt6], dim=1)
                    acts = torch.cat([acts, act[:, tgt_idx:tgt_idx + 1]], dim=1)
            ns += B
    print(f"\n== eval on {ns} windows (T={T}, ctx={CTX}) ==")
    print(f"teacher 1-step model={s1m/ns:.4f}  copy-1={s1c/ns:.4f}  ratio={s1m/max(s1c, 1e-9):.3f}")
    print("horizon | model mse | copy(last-ctx) mse | ratio")
    for h in range(1, HROLL + 1):
        mm, cc = roll[h]
        if mm == 0:
            continue
        print(f"  h={h:2d}   | {mm/ns:.4f}   | {cc/ns:.4f}           | {mm/max(cc, 1e-9):.3f}")


if __name__ == "__main__":
    main()
