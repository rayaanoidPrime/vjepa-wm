# 20-frame future-prediction GIFs under counterfactual action modes.
# ctx=4 (recon) then 16 autoregressive predictions @5fps = 4.0 s total.
# Top row: GT encoder-recon frames 0..11 (dark beyond). Bottom row: predictions.
import warnings, os
warnings.filterwarnings("ignore")
import numpy as np, torch, yaml
from PIL import Image, ImageDraw
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
OUT = os.environ.get("GT_GIF_OUT", "/marimo/logs/grandtour_sweep/gt_v0_step2_vm2m/gifs")
CTX = 4
NF = 20          # total frames in the gif (ctx + 16 predicted)
EPS = [0, 2]     # episodes (windows) to animate
MODES = [("FWD", (0.6, 0.0, 0.0)), ("LEFT", (0.0, 0.0, 0.6)),
         ("RIGHT", (0.0, 0.0, -0.6)), ("STOP", (0.0, 0.0, 0.0))]

def build():
    cfg = yaml.safe_load(open(GT_CFG)); cfg2 = yaml.safe_load(open(S2_CFG))
    da = cfg["data_aug"]; d = cfg["data"]; m = cfg["model"]
    tr = make_transforms(img_size=d["img_size"], random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0), random_resize_scale=(1.0, 1.0),
        auto_augment=False, motion_shift=False, reprob=0.0, normalize=da["normalize"])
    inv = make_inverse_transforms(img_size=d["img_size"], **cfg["data_aug"])
    ds = GrandTourVideoDataset(data_path="/marimo/datasets/grand_tour",
        frames_per_clip=d["droid"]["dataset_fpcs"][0], fps=d["droid"]["fps"], transform=tr, seed=21)
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

def make_gif(frames, path, duration_ms=200):
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=duration_ms, loop=0)

def compose(frames_top, frames_bot, mode, clip, scale=2):
    H, W = frames_top[0].shape[:2]
    HW, WW = H * scale, W * scale
    bar = 26
    out = []
    for i, (t, b) in enumerate(zip(frames_top, frames_bot)):
        img = Image.new("RGB", (WW, HW * 2 + bar + 4), (18, 18, 18))
        d = ImageDraw.Draw(img)
        d.text((6, 4), "clip %d | %s | t=%d  (top=GT recon, bottom=pred @5fps)" % (clip, mode, i), fill=(230, 230, 230))
        if t is not None:
            img.paste(Image.fromarray(t).resize((WW, HW), Image.NEAREST), (0, bar))
        if b is not None:
            img.paste(Image.fromarray(b).resize((WW, HW), Image.NEAREST), (0, bar + HW + 4))
        out.append(img)
    return out

def main():
    ds, wm, head = build()
    from pathlib import Path as P
    P(OUT).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for ei in EPS:
            obs, act, _, _ = ds[ei]
            vis = obs["visual"].unsqueeze(0).to(DEVICE)
            actb = act.unsqueeze(0).to(DEVICE)
            vf, _, _ = wm.encode({"visual": vis}, actb)     # (1,T,1,H,W,D)
            B, T, V, H, W, D = vf.shape
            T = min(T, NF)
            vf = vf[:, :T]
            gt_imgs = head.decode(vf.float())[0, :, 0].numpy()   # T,224,224,3
            gt_top = [gt_imgs[i] if i < T else None for i in range(NF)]
            # action template rows
            act_all = torch.zeros(NF, 39)
            act_all[:T] = act[:T]
            for i in range(T, NF):                              # hold last recorded joints+cmd
                act_all[i] = act[T - 1]
            for name, cmd in MODES:
                a = act_all.clone()
                a[CTX:, 0], a[CTX:, 1], a[CTX:, 2] = cmd
                WB = 4  # bounded context window (causal predictor mask sized for 12 frames)
                win = vf[:, :CTX].clone()
                acts = a.clone()
                preds_imgs = []
                for i in range(0, CTX):
                    preds_imgs.append(gt_imgs[i])
                for idx in range(CTX, NF):
                    cw = win[:, -WB:]
                    aw = acts[idx - cw.shape[1]:idx]
                    out, _, _ = wm.predictor(cw, aw.unsqueeze(0).to(DEVICE), None)
                    nxt = out[:, -1].view(1, 1, H, W, D).unsqueeze(2)
                    win = torch.cat([win, nxt], 1)
                    dec = head.decode(nxt.float())[0, 0, 0].numpy()
                    preds_imgs.append(dec)
                frames = compose(gt_top, preds_imgs, name, ei)
                path = "%s/gif_ep%d_%s.gif" % (OUT, ei, name.lower())
                make_gif([np.asarray(f) for f in frames], path)
                print("saved", path, flush=True)
    print("GIF DONE")

if __name__ == "__main__":
    main()
