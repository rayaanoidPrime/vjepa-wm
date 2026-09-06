# Counterfactual action-conditioning probe (guide item 9).
# Same context rolled out under different commanded twists: REC/STOP/LEFT/RIGHT/FWD.
# Reports (a) latent error vs GT future per mode+horizon, (b) divergence from REC,
#          (c) decoded contact sheets per mode saved as PNG.
import warnings, os, sys
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ.setdefault("JEPAWM_DSET", "/marimo/datasets")
os.environ.setdefault("TORCH_HOME", "/marimo/torchhub")
from app.plan_common.datasets.transforms import make_transforms, make_inverse_transforms
from app.plan_common.datasets.grandtour_dset import GrandTourVideoDataset
from app.plan_common.models.wm_heads import WorldModelViTImageHead
from app.vjepa_wm.utils import init_video_model
from app.vjepa_wm.video_wm import VideoWM

DEVICE = "cuda:0"
GT_CFG = os.environ.get("GT_CFG_YAML", "/marimo/jepa-wms/configs/vjepa_wm/grandtour_sweep/gt_v0_12f_fps5_r224_dv2vits_AdaLN_d6_2roll_1n.yaml")
S2_CFG = os.environ.get("GT_S2_YAML", "/marimo/jepa-wms/configs/vjepa_wm/grandtour_sweep/gt_v0_step2_vm2m.yaml")
GT_CKPT = os.environ.get("GT_CKPT", "/marimo/logs/grandtour_sweep/gt_v0_12f_fps5_r224_dv2vits_AdaLN_d6_2roll_1n/jepa-latest.pth.tar")
HEAD_CKPT = os.environ.get("GT_HEAD_CKPT", "/marimo/logs/grandtour_sweep/gt_v0_step2_vm2m/jepa-latest_image_head.pth.tar")
OUT = os.environ.get("GT_CF_OUT", "/marimo/logs/grandtour_sweep/gt_v0_step2_vm2m/viz_cf")
CTX = 4
NWIN = 6
MODES = {"REC": None, "STOP": (0., 0., 0.), "LEFT": (0., 0., 0.6),
         "RIGHT": (0., 0., -0.6), "FWD": (0.6, 0., 0.)}

def build():
    cfg = yaml.safe_load(open(GT_CFG)); cfg2 = yaml.safe_load(open(S2_CFG))
    da = cfg["data_aug"]; d = cfg["data"]; m = cfg["model"]
    tr = make_transforms(img_size=d["img_size"], random_horizontal_flip=da["random_horizontal_flip"],
        random_resize_aspect_ratio=tuple(da["random_resize_aspect_ratio"]),
        random_resize_scale=tuple(da["random_resize_scale"]), auto_augment=da["auto_augment"],
        motion_shift=da["motion_shift"], reprob=da["reprob"], normalize=da["normalize"])
    inv = make_inverse_transforms(img_size=d["img_size"], **cfg["data_aug"])
    ds = GrandTourVideoDataset(data_path="/marimo/datasets/grand_tour",
        frames_per_clip=d["droid"]["dataset_fpcs"][0], fps=d["droid"]["fps"],
        transform=tr, seed=3)
    mkw = {k: v for k, v in m.items() if k not in ("rollout_cfg", "heads_cfg", "pretrained_path",
        "visual_encoder", "action_encoder", "proprio_encoder", "predictor", "wm_encoding", "attn")}
    mkw.update(m["visual_encoder"]); mkw.update(m["action_encoder"]); mkw.update(m["proprio_encoder"])
    mkw.update(m["predictor"])
    mkw.update(dict(device=DEVICE, img_size=d["img_size"], action_dim=ds.action_dim,
        proprio_dim=ds.proprio_dim, cfgs_attn_pattern=m.get("attn"), use_proprio=False, use_action=True))
    predictor, encoder, ae, pe = init_video_model(**mkw)
    wmk = dict(device=DEVICE, encoder=encoder, predictor=predictor, action_encoder=ae, proprio_encoder=pe,
        action_dim=ds.action_dim, proprio_dim=ds.proprio_dim, use_proprio=False, use_action=True,
        action_tokens=1, proprio_tokens=0, grid_size=m.get("grid_size", 16), tubelet_size_enc=1,
        action_conditioning=m.get("action_conditioning", "token"), proprio_encoding=m.get("proprio_encoding", "feature"),
        enc_type=m["visual_encoder"]["enc_type"], pred_type=m["predictor"]["pred_type"],
        action_encoder_inpred=m["action_encoder"]["action_encoder_inpred"], proprio_encoder_inpred=False,
        action_skip=1, frameskip=1, img_size=d["img_size"], heads={}, scaler=None, optimizer=None,
        clip_grad=1.0, mixed_precision=False, use_radamw=False, cfgs_loss=cfg["loss"],
        dup_image=False, batchify_video=True, normalize_reps=False)
    wm = VideoWM(**wmk).to(DEVICE)
    for p in wm.parameters(): p.requires_grad_(False)
    ck = torch.load(GT_CKPT, map_location="cpu")
    wm.predictor.load_state_dict({k.replace("module.", ""): v for k, v in ck["predictor"].items()})
    hcfg = cfg2["model"]["heads_cfg"]["architectures"]["image_head"]["config"]
    head = WorldModelViTImageHead(head_config=hcfg, inverse_transform=inv, device=DEVICE)
    head.load_checkpoint(HEAD_CKPT); head.model.to(DEVICE); head.eval(); head.model.eval()
    for p in head.model.parameters(): p.requires_grad_(False)
    wm.eval()
    print("loaded", flush=True)
    return ds, wm, head

def rollout(wm, feats, acts_seq):
    T = acts_seq.shape[0]; preds = []
    for idx in range(CTX, T):
        out, _, _ = wm.predictor(feats, acts_seq[:idx].unsqueeze(0).to(feats.device), None)
        nxt = out[:, -1].view(1, 1, feats.shape[3], feats.shape[4], feats.shape[5]).unsqueeze(2)
        preds.append(nxt)
        feats = torch.cat([feats, nxt], 1)
    return preds

def main():
    ds, wm, head = build()
    from pathlib import Path as P
    P(OUT).mkdir(parents=True, exist_ok=True)
    hist = {name: {h: [0.0, 0.0] for h in range(1, 9)} for name in MODES}
    ns = 0
    with torch.no_grad():
        for wi in range(NWIN):
            ei = wi % len(ds)
            obs, act, _, _ = ds[ei]
            vis = obs["visual"].unsqueeze(0).to(DEVICE)
            actb = act.unsqueeze(0).to(DEVICE)
            vf, _, _ = wm.encode({"visual": vis}, actb)
            B, T, V, H, W, D = vf.shape
            gt_future = vf[:, CTX:]
            acts = {}
            for name, cmd in MODES.items():
                a = act.clone()
                if cmd is not None:
                    a[CTX:, 0], a[CTX:, 1], a[CTX:, 2] = cmd
                acts[name] = a
            outs = {name: rollout(wm, vf[:, :CTX].clone(), acts[name]) for name in MODES}
            for name, preds in outs.items():
                for h in range(1, T - CTX + 1):
                    hist[name][h][0] += F.mse_loss(preds[h-1], gt_future[:, h-1:h]).item()
            rec = outs["REC"]
            for name, preds in outs.items():
                if name == "REC":
                    continue
                for h in range(1, T - CTX + 1):
                    hist[name][h][1] += F.mse_loss(preds[h-1], rec[h-1]).item()
            ns += 1
            if wi < 6:
                ncol = T - CTX
                gt_dec = head.decode(gt_future.float())[0, :, 0].numpy()
                dec = {}
                for name, preds in outs.items():
                    arr = np.stack([head.decode(p.float())[0, 0, 0].numpy() for p in preds])
                    dec[name] = arr
                nrow = 1 + len(MODES)
                fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.2, nrow * 1.35))
                for j in range(ncol):
                    axes[0, j].imshow(gt_dec[j]); axes[0, j].axis("off")
                    axes[0, j].set_title(f"f{j+1}", fontsize=8)
                axes[0, 0].set_ylabel("GT", fontsize=8)
                for r, name in enumerate(MODES, start=1):
                    for j in range(ncol):
                        axes[r, j].imshow(dec[name][j]); axes[r, j].axis("off")
                    axes[r, 0].set_ylabel(name, fontsize=8)
                fig.suptitle(f"counterfactual actions, same ctx (clip {wi}, ep {ei})", fontsize=10)
                plt.tight_layout()
                pth = f"{OUT}/cf_clip{wi:02d}.png"
                fig.savefig(pth, dpi=72); plt.close(fig)
                print("saved", pth, flush=True)
    print("\n== latent error vs GT future (per horizon, mean over %d windows) ==" % ns)
    hdrs = "".join(("  h%d " % h).rjust(7) for h in range(1, 9))
    print("mode  |" + hdrs)
    for name in MODES:
        row = "".join(("%6.3f" % (hist[name][h][0]/ns)).rjust(7) for h in range(1, 9))
        print("%-6s|%s" % (name, row))
    print("\n== divergence from REC rollout (latent mse) ==")
    for name in MODES:
        if name == "REC":
            continue
        row = "".join(("%6.3f" % (hist[name][h][1]/ns)).rjust(7) for h in range(1, 9))
        print("%-6s|%s" % (name, row))
    print("CF DONE")

if __name__ == "__main__":
    main()
