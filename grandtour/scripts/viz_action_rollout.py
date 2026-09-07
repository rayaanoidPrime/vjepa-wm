#!/usr/bin/env python3
"""Action-conditioned future rollouts for the GrandTour JEPA world model (final demo).

Loads a trained stage-1 world model (frozen DINOv2 encoder + action-conditioned
AdaLN predictor) from a jepa-wms checkpoint and decodes the rollouts with a
pretrained image head. The default head is the autoencoder decoder that
facebookresearch/jepa-wms publishes for the DINOv2 ViT-S/14 encoder
(``vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar``), i.e. the exact-fit decoder
for our ``gt_v0`` world model. Any head config + checkpoint can be pointed to
instead (e.g. our in-domain step-2 heads).

Row semantics (matches stage-1 training, ``compute_loss shift=1``): predictor
slot t outputs the features of frame t+1, conditioned by the action row at
frame t. So the future frame with index p is generated from the action row
p-1, and a rollout only reaches your command once it occupies that slot.

Outputs: MP4 + GIF, default 14 frames @ 5 fps = 4 GT context + 10 future
(layout per frame: LEFT = model output, RIGHT = actual GT frame; context
frames show the head's recon of the GT frame on the left).

gt mode  replay the video's OWN recorded actions (cmd twist + joints) for the
         next N future frames and compare prediction vs actual GT side-by-side
         (prints per-frame latent cosine similarity to GT features).
plan mode feed a scripted twist schedule (FWD/LEFT/RIGHT/STOP) while holding
         the joint state from the context boundary.

Environment:
  JEPAWM_HOME / JEPAWM_DSET / JEPAWM_LOGS / TORCH_HOME   repo + data roots (see _paths)
  GT_WM_TAG     world-model run folder under $JEPAWM_LOGS/grandtour_sweep
                (must contain <tag>.yaml and jepa-latest.pth.tar)
  GT_DEC_YAML   image-head config yaml (default: upstream vm2m dv2vits 224 dec)
  GT_DEC_CKPT   image-head checkpoint (their dv2vits decoder, or our step-2 head)
  GT_MISSION    mission timestamp to use (default: auto-pick, see GT_MODE)
  GT_START      10 Hz start frame (default: auto-pick)
  GT_MODE       'gt' (default) | 'plan'
  GT_PLAN       command schedule for plan mode, e.g. "FWD 10, LEFT 5, FWD 5"
  GT_NCTX       context frames (default 4)
  GT_NFUT       future frames (default 10)
  GT_OUT        output directory

Examples:
  python viz_action_rollout.py                              # gt, auto forward clip
  GT_MODE=plan GT_PLAN="FWD 10,LEFT 5,FWD 5" python viz_action_rollout.py
"""
import os
import sys
import warnings
from pathlib import Path

import _paths

warnings.filterwarnings("ignore")
os.chdir(str(_paths.JEPAWM_HOME))
sys.path.insert(0, str(_paths.JEPAWM_HOME))
os.environ.setdefault("TORCH_HOME", str(_paths.TORCH_HOME))

import h5py
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from app.plan_common.datasets.grandtour_dset import GrandTourVideoDataset
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.models.wm_heads import WorldModelViTImageHead
from app.vjepa_wm.utils import init_video_model
from app.vjepa_wm.video_wm import VideoWM

DEVICE = "cuda:0"
DEC_IMG = 224  # decoded image side; set from the decoder config at build()
LOGS = _paths.JEPAWM_LOGS / "grandtour_sweep"
DEC_YAML_DEF = _paths.JEPAWM_HOME / "configs/vjepa_wm/vm2m/open_source_decs/step2_lpips_vm2m_vits_vitldec_224_vjtrans.yaml"
TAG = os.environ.get("GT_WM_TAG", "gt_v0_12f_fps5_r224_dv2vits_AdaLN_d6_2roll_1n")
MODE = os.environ.get("GT_MODE", "gt")
NCTX = int(os.environ.get("GT_NCTX", "4"))
NFUT = int(os.environ.get("GT_NFUT", "10"))
WB = 4
OUT = os.environ.get("GT_OUT", str(_paths.JEPAWM_LOGS / "grandtour_sweep" / "action_video"))
TWISTS = {"FWD": (0.6, 0.0, 0.0), "LEFT": (0.0, 0.0, 0.6), "RIGHT": (0.0, 0.0, -0.6), "STOP": (0.0, 0.0, 0.0)}


def build():
    """World model (encoder+predictor, frozen) + image head (frozen)."""
    cfg = yaml.safe_load(open(LOGS / TAG / f"{TAG}.yaml"))
    da = cfg["data_aug"]
    d = cfg["data"]
    m = cfg["model"]
    tr = make_transforms(
        img_size=d["img_size"], random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0), random_resize_scale=(1.0, 1.0),
        auto_augment=False, motion_shift=False, reprob=0.0, normalize=da["normalize"])
    inv = make_inverse_transforms(img_size=d["img_size"], **da)
    ds = GrandTourVideoDataset(
        data_path=_paths.JEPAWM_DSET / "grand_tour",
        frames_per_clip=d["droid"]["dataset_fpcs"][0], fps=d["droid"]["fps"],
        transform=tr, seed=9)
    mkw = {k: v for k, v in m.items() if k not in (
        "rollout_cfg", "heads_cfg", "pretrained_path", "visual_encoder",
        "action_encoder", "proprio_encoder", "predictor", "wm_encoding", "attn")}
    for sec in ("visual_encoder", "action_encoder", "proprio_encoder", "predictor"):
        mkw.update(m[sec])
    mkw.update(device=DEVICE, img_size=d["img_size"], action_dim=ds.action_dim,
               proprio_dim=ds.proprio_dim, cfgs_attn_pattern=m.get("attn"),
               use_proprio=False, use_action=True)
    predictor, encoder, ae, pe = init_video_model(**mkw)
    wm = VideoWM(device=DEVICE, encoder=encoder, predictor=predictor,
                 action_encoder=ae, proprio_encoder=pe, action_dim=ds.action_dim,
                 proprio_dim=ds.proprio_dim, use_proprio=False, use_action=True,
                 action_tokens=1, proprio_tokens=0, grid_size=m.get("grid_size", 16),
                 tubelet_size_enc=1, action_conditioning="token", proprio_encoding="none",
                 enc_type=m["visual_encoder"]["enc_type"], pred_type=m["predictor"]["pred_type"],
                 action_encoder_inpred=True, proprio_encoder_inpred=False, action_skip=1,
                 frameskip=1, img_size=d["img_size"], heads={}, scaler=None, optimizer=None,
                 clip_grad=1.0, mixed_precision=False, use_radamw=False,
                 cfgs_loss=cfg["loss"], dup_image=False, batchify_video=True,
                 normalize_reps=bool((m.get("wm_encoding") or {}).get("normalize_reps", False))).to(DEVICE)
    for p in wm.parameters():
        p.requires_grad_(False)
    ck = torch.load(LOGS / TAG / "jepa-latest.pth.tar", map_location="cpu")
    wm.predictor.load_state_dict({k.replace("module.", ""): v for k, v in ck["predictor"].items()})
    dec = yaml.safe_load(open(os.environ.get("GT_DEC_YAML", str(DEC_YAML_DEF))))
    hcfg = dec["model"]["heads_cfg"]["architectures"]["image_head"]["config"]
    imgsz = hcfg.get("img_size", 224)
    globals()["DEC_IMG"] = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
    head = WorldModelViTImageHead(head_config=hcfg, inverse_transform=inv, device=DEVICE)
    head.load_checkpoint(os.environ["GT_DEC_CKPT"])
    head.model.to(DEVICE).eval()
    for p in head.model.parameters():
        p.requires_grad_(False)
    wm.eval()
    print(f"world model: {TAG} | head: {os.path.basename(os.environ['GT_DEC_CKPT'])}", flush=True)
    return ds, tr, inv, wm, head


def choose_window(ds, want):
    """Episode + 10 Hz start for a window of `want` fps5 frames (env override or auto 'forward')."""
    if os.environ.get("GT_MISSION"):
        ep = next(e for e in ds.episodes if _paths.Path(e.path).parent.name == os.environ["GT_MISSION"])
        fstp = max(1, int(round((1.0 / ds.fps) / ep.dt)))
        s0 = int(os.environ.get("GT_START", "0"))
        return ep, fstp, s0
    # auto: strongest forward motion in the future window, with real content
    cands = []
    for epi in ds.episodes:
        fstp = max(1, int(round((1.0 / ds.fps) / epi.dt)))
        need = want * fstp
        if need > epi.n_frames:
            continue
        vx = np.clip(epi.cmd[:, 0], 0, None)
        for s in range(0, epi.n_frames - need + 1, fstp):
            r = s + fstp * np.arange(want)
            cands.append((vx[r[NCTX:want]].mean(), epi, fstp, s))
    cands.sort(reverse=True, key=lambda c: c[0])
    for _, epi, fstp, s in cands[:60]:
        r = s + fstp * np.arange(want)
        stds = []
        with h5py.File(epi.path, "r") as h5:
            for k in (0, want // 2, want - 1):
                stds.append(np.asarray(epi.read_frame(h5, int(r[k])), np.uint8).std())
        if min(stds) > 22:
            return epi, fstp, s
    return cands[0][1], cands[0][2], cands[0][3]


def load_window(ds, tr, inv, ep, fstp, s0, length):
    """Frames, normalized actions and GT images for `length` fps5 frames from s0."""
    r = s0 + fstp * np.arange(length)
    with h5py.File(ep.path, "r") as h5:
        ims = [np.asarray(ep.read_frame(h5, int(i)), np.uint8) for i in r]
    arr = torch.as_tensor(np.stack(ims), dtype=torch.float32) / 255.0
    buf = arr.permute(0, 3, 1, 2).contiguous()
    T = tr(buf)                                  # (L,C,H,W) normalized
    invT = inv(T)
    gt = ([u8(invT[t]) for t in range(invT.shape[0])] if invT.dim() == 4 else [u8(invT)])
    joints = (torch.as_tensor(ep.joints[r], dtype=torch.float32) - ds.joint_mean) / ds.joint_std
    cmd = torch.as_tensor(ep.cmd[r], dtype=torch.float32)
    A = torch.cat([cmd, joints], -1)             # (L,39) recorded actions
    return T, A, gt


def u8(t):
    """Tensor/np to uint8 RGB with channels last (accepts C,H,W or H,W,C)."""
    t = t.detach().cpu().float()
    if t.dim() == 3 and t.shape[2] != 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    t = (t * 255.0 if t.max() <= 1.0 else t).clamp(0, 255)
    return t.byte().numpy()


def panel(imL, imR, header):
    s = 2
    w = DEC_IMG * s * 2 + 4
    c = Image.new("RGB", (w, DEC_IMG * s + 28), (16, 16, 16))
    ImageDraw.Draw(c).text((8, 7), header, fill=(235, 235, 235))
    c.paste(Image.fromarray(imL).resize((DEC_IMG * s, DEC_IMG * s), Image.NEAREST), (0, 28))
    c.paste(Image.fromarray(imR).resize((DEC_IMG * s, DEC_IMG * s), Image.NEAREST), (DEC_IMG * s + 4, 28))
    return np.asarray(c)


def save(frames, name):
    import imageio.v2 as iio
    Path(OUT).mkdir(parents=True, exist_ok=True)
    iio.mimsave(f"{OUT}/{name}.mp4", frames, fps=5, codec="libx264", quality=7)
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(f"{OUT}/{name}.gif", save_all=True, append_images=pil[1:], duration=200, loop=0)
    print(f"saved {OUT}/{name}.mp4/.gif ({len(frames)} frames)", flush=True)


def rollout(ds, tr, inv, wm, head):
    """Encode NCTX context frames, then generate NFUT frames one at a time from
    the action rows that follow the context (row t drives frame t+1)."""
    ep, fstp, s0 = choose_window(ds, NCTX + NFUT)
    T, A, gt = load_window(ds, tr, inv, ep, fstp, s0, NCTX + NFUT)
    name = f"{_paths.Path(ep.path).parent.name}_{s0}"
    print(f"window: mission={name.rsplit('_', 1)[0]} start10hz={s0} "
          f"(cmd vx/yaw @ctx-last: {ep.cmd[s0 + fstp * (NCTX - 1)][0]:.2f}/"
          f"{ep.cmd[s0 + fstp * (NCTX - 1)][2]:.2f})", flush=True)
    with torch.no_grad():
        vf, _, _ = wm.encode({"visual": T.unsqueeze(0).to(DEVICE)}, A.unsqueeze(0).to(DEVICE))
        B, L, V, H, W, D = vf.shape
        win = vf[:, :NCTX].clone()
        recon = head.decode(win.float())[0, :, 0]
        frames = [panel(u8(recon[i]), gt[i], f"CTX t=+{i} (recon | GT)") for i in range(NCTX)]
        sims = []
        p = NCTX
        for k in range(NFUT):
            cw = win[:, -WB:]
            aw = A.unsqueeze(0).to(DEVICE)[:, p - WB:p]
            out, _, _ = wm.predictor(cw, aw, None)
            nxt = out[:, -1].view(1, 1, H, W, D).unsqueeze(2)
            win = torch.cat([win, nxt], 1)
            dec = u8(head.decode(nxt.float())[0, 0, 0])
            sim = torch.nn.functional.cosine_similarity(
                nxt.reshape(1, H * W, D).float(), vf[:, p].reshape(1, H * W, D).float(), dim=-1).mean().item()
            sims.append(sim)
            frames.append(panel(dec, gt[p], f"pred t=+{p} | GT actual (recorded a_{{{p - 1}}})"))
            p += 1
    print("latent cosine sim vs GT future frames (h1..h10): "
          + " ".join(f"{s:.3f}" for s in sims) + f" | mean {np.mean(sims):.3f}", flush=True)
    save(frames, name + "_gtcond")
    return name


def rollout_plan(ds, tr, inv, wm, head):
    """Scripted twists with the joint state held from the context boundary."""
    plan = [tok.split() for tok in os.environ.get("GT_PLAN", "FWD 10").split(",")]
    segs = [(cmd, int(n)) for cmd, n in plan]
    assert all(cmd in TWISTS for cmd, _ in segs), "plan commands: FWD/LEFT/RIGHT/STOP"
    seq = [cmd for cmd, n in segs for _ in range(n)]
    N = len(seq)
    total = NCTX + N
    ep, fstp, s0 = choose_window(ds, total)
    T, A, gt = load_window(ds, tr, inv, ep, fstp, s0, total)
    name = f"{_paths.Path(ep.path).parent.name}_{s0}"
    A = A.clone()
    for k in range(N):                                   # rows NCTX-1 .. NCTX-1+N-1
        A[NCTX - 1 + k, :3] = torch.as_tensor(TWISTS[seq[k]])
        A[NCTX - 1 + k, 3:] = A[NCTX - 1, 3:]            # hold joints from context boundary
    with torch.no_grad():
        vf, _, _ = wm.encode({"visual": T.unsqueeze(0).to(DEVICE)}, A.unsqueeze(0).to(DEVICE))
        B, L, V, H, W, D = vf.shape
        win = vf[:, :NCTX].clone()
        frames = [panel(u8(head.decode(vf[:, i:i + 1].float())[0, 0, 0]), gt[i], f"CTX t=+{i} (recon | GT)")
                  for i in range(NCTX)]
        p = NCTX
        for k in range(N):
            cw = win[:, -WB:]
            aw = A.unsqueeze(0).to(DEVICE)[:, p - WB:p]
            out, _, _ = wm.predictor(cw, aw, None)
            nxt = out[:, -1].view(1, 1, H, W, D).unsqueeze(2)
            win = torch.cat([win, nxt], 1)
            frames.append(panel(u8(head.decode(nxt.float())[0, 0, 0]), gt[p],
                                f"pred t=+{p} | GT actual (cmd {seq[k]})"))
            p += 1
    save(frames, name + "_plan")
    return name


def main():
    torch.set_grad_enabled(False)
    assert "GT_DEC_CKPT" in os.environ, "set GT_DEC_CKPT to the decoder checkpoint"
    ds, tr, inv, wm, head = build()
    if MODE == "gt":
        rollout(ds, tr, inv, wm, head)
    else:
        rollout_plan(ds, tr, inv, wm, head)


if __name__ == "__main__":
    main()
