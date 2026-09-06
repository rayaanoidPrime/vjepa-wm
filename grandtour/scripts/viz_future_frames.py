#!/usr/bin/env python3
"""Visualize future-frame predictions: decode GT + rollout latents with a decoder head.

Requires a trained step-2 (decoder head) checkpoint next to the world-model
checkpoint, e.g.:
  $JEPAWM_LOGS/grandtour_sweep/gt_v0_step2_vm2m/jepa-latest_image_head.pth.tar
Outputs contact-sheet PNGs (top = GT recon from encoder, bottom = context +
rolled-out predictions with red borders) into $GT_VIZ.

Usage:
  python viz_future_frames.py [--gt-run RUN_NAME] [--s2-run RUN_NAME] [--nwin 6]
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
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.plan_common.datasets.transforms import make_transforms, make_inverse_transforms
from app.plan_common.datasets.grandtour_dset import GrandTourVideoDataset
from app.plan_common.models.wm_heads import WorldModelViTImageHead
from app.vjepa_wm.utils import init_video_model
from app.vjepa_wm.video_wm import VideoWM

DEVICE = "cuda:0"
CFG_REL = "configs/vjepa_wm/grandtour_sweep/gt_v0_12f_fps5_r224_dv2vits_AdaLN_d6_2roll_1n.yaml"
S2_CFG_REL = "configs/vjepa_wm/grandtour_sweep/gt_v0_step2_vm2m.yaml"
CTX = 4


def load_model(gt_run, s2_run, cfg_rel, s2_cfg_rel):
    cfg = yaml.safe_load(open(os.path.join(_paths.JEPAWM_HOME, cfg_rel)))
    cfg2 = yaml.safe_load(open(os.path.join(_paths.JEPAWM_HOME, s2_cfg_rel)))
    da = cfg["data_aug"]
    d = cfg["data"]
    m = cfg["model"]
    transform = make_transforms(img_size=d["img_size"],
        random_horizontal_flip=da["random_horizontal_flip"],
        random_resize_aspect_ratio=tuple(da["random_resize_aspect_ratio"]),
        random_resize_scale=tuple(da["random_resize_scale"]),
        auto_augment=da["auto_augment"], motion_shift=da["motion_shift"],
        reprob=da["reprob"], normalize=da["normalize"])
    inv = make_inverse_transforms(img_size=d["img_size"], **cfg["data_aug"])
    ds = GrandTourVideoDataset(data_path=str(_paths.JEPAWM_DSET / "grand_tour"),
        frames_per_clip=d["droid"]["dataset_fpcs"][0], fps=d["droid"]["fps"],
        transform=transform, seed=11)
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

    base = _paths.JEPAWM_LOGS / "grandtour_sweep"
    gt_ckpt = base / gt_run / "jepa-latest.pth.tar"
    ck = torch.load(str(gt_ckpt), map_location="cpu")
    wm.predictor.load_state_dict({k.replace("module.", ""): v for k, v in ck["predictor"].items()})
    head_cfg = cfg2["model"]["heads_cfg"]["architectures"]["image_head"]["config"]
    head = WorldModelViTImageHead(head_config=head_cfg, inverse_transform=inv, device=DEVICE)
    head.load_checkpoint(str(base / s2_run / "jepa-latest_image_head.pth.tar"))
    head.model.to(DEVICE)
    head.eval()
    head.model.eval()
    for p in head.model.parameters():
        p.requires_grad_(False)
    wm.eval()
    print("model + head loaded")
    return ds, wm, head


def rollout_next(wm, feats, acts):
    out, _, _ = wm.predictor(feats, acts, None)
    return out[:, -1].view(feats.shape[0], 1, feats.shape[3], feats.shape[4], feats.shape[5]).unsqueeze(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-run", default="gt_v0_12f_fps5_r224_dv2vits_AdaLN_d6_2roll_1n")
    ap.add_argument("--s2-run", default="gt_v0_step2_vm2m")
    ap.add_argument("--nwin", type=int, default=6)
    args = ap.parse_args()
    ds, wm, head = load_model(args.gt_run, args.s2_run, CFG_REL, S2_CFG_REL)
    out_dir = _paths.GT_VIZ
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for wi in range(args.nwin):
            ei = wi % len(ds)
            obs, act, _, _ = ds[ei]
            vis = obs["visual"].unsqueeze(0).to(DEVICE)
            actb = act.unsqueeze(0).to(DEVICE)
            vf, _, _ = wm.encode({"visual": vis}, actb)
            B, T, V, H, W, D = vf.shape
            gt_img = head.decode(vf.float())
            feats = vf[:, :CTX].clone()
            acts = actb[:, :CTX].clone()
            preds = []
            for h in range(1, T - CTX + 1):
                nxt = rollout_next(wm, feats, acts)
                preds.append(nxt)
                ti = CTX - 1 + h
                if ti < T - 1:
                    feats = torch.cat([feats, nxt], dim=1)
                    acts = torch.cat([acts, actb[:, ti:ti + 1]], dim=1)
            pred_feats = torch.cat([vf[:, :CTX], torch.cat(preds, dim=1)], dim=1)
            pred_img = head.decode(pred_feats.float())
            gtA = gt_img[0, :, 0]
            prA = pred_img[0, :, 0]
            fig, axes = plt.subplots(2, T, figsize=(T * 1.15, 2.3))
            for t in range(T):
                for r, arr in ((0, gtA[t]), (1, prA[t])):
                    axes[r, t].imshow(arr)
                    axes[r, t].axis("off")
                    if r == 0:
                        axes[r, t].set_title(f"t={t}", fontsize=8)
            for t in range(CTX, T):
                for sp in axes[1, t].spines.values():
                    sp.set_color("red")
                    sp.set_linewidth(3)
            fig.suptitle(f"clip {wi} (ep {ei}): top=GT(recon)  bottom=ctx+PREDICTED (red)", fontsize=10)
            plt.tight_layout()
            pth = str(out_dir / f"clip{wi:02d}_ep{ei}.png")
            fig.savefig(pth, dpi=72)
            plt.close(fig)
            print("saved", pth)


if __name__ == "__main__":
    main()
