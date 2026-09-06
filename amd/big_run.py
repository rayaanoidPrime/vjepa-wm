"""Big-run trainer: frozen V-JEPA 2.1 + causal tubelet masking + motion-weighted
LN-L1 + per-window command tokens + velocity-consistency aux head + cmd dropout.

Resumable (loads latest ckpts/big_run_step*.pt), fp32, streaming decode with
retry-on-truncated-jpeg, clip-index rebuild every 500 steps so background data
pulls are picked up. Eval on held-out ICE-1 / SPX-2 every 200 steps.

Usage:  python amd/big_run.py [total_steps=8000]
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vjepa2"))
from app.vjepa_2_1.models import vision_transformer as vjepa_vit
from app.vjepa_2_1.models.predictor import vit_predictor

HERE = Path(__file__).resolve().parent
ROOT = (Path("/marimo/grandtour") if Path("/marimo").exists()
        else HERE.parent / "data" / "grandtour")
CK = (Path("/marimo/checkpoints") if Path("/marimo").exists()
      else HERE.parent / "ckpts")
LOG = Path("/marimo/big_run_log.txt") if Path("/marimo").exists() else HERE / "big_run_log.txt"
CK.mkdir(parents=True, exist_ok=True)
LOG.touch()

CORE = ["2024-10-01-11-29-55", "2024-10-01-11-47-44",
        "2024-10-01-12-00-49", "2024-11-02-17-10-25"]
ADD = [  # all remaining missions except the held-out pair below
"2024-11-02-17-43-10", "2024-11-02-21-12-51", "2024-11-03-07-52-45",
"2024-11-03-07-57-34", "2024-11-03-08-17-23", "2024-11-03-13-51-43",
"2024-11-03-13-59-54", "2024-11-04-10-57-34", "2024-11-04-12-55-59",
"2024-11-04-13-07-13", "2024-11-04-16-05-00", "2024-11-11-12-07-40",
"2024-11-11-12-42-47", "2024-11-11-14-29-44", "2024-11-11-16-14-23",
"2024-11-14-11-17-02", "2024-11-14-12-01-26", "2024-11-14-13-45-37",
"2024-11-14-14-36-02", "2024-11-14-15-22-43", "2024-11-14-16-04-09",
"2024-11-15-10-16-35", "2024-11-15-11-18-14", "2024-11-15-11-37-15",
"2024-11-15-12-06-03", "2024-11-15-14-14-12", "2024-11-15-14-43-52",
"2024-11-15-16-41-14", "2024-11-18-12-05-01", "2024-11-18-13-22-14",
"2024-11-18-15-46-05", "2024-11-18-16-59-23", "2024-11-18-17-13-09",
"2024-11-18-17-31-36", "2024-11-25-14-57-08", "2024-11-25-16-36-19",
"2024-12-03-13-15-38", "2024-12-03-13-26-40", "2024-12-09-09-34-43",
"2024-12-09-09-41-46", "2024-12-09-11-28-28", "2024-12-09-11-53-11",
]
VAL = {"2024-11-18-13-48-19": "ICE", "2024-11-02-17-18-32": "SPX2"}

IMG, NF, PATCH, TUB = 384, 16, 16, 2
P = (IMG // PATCH) ** 2
T = NF // TUB
B = 4
STRIDE = 4
LAM = 0.5
ACT_P = 0.25
BOUND = (6, 8, 10)
ACT_BASE = T * P
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
VJEPA_CKPT = CK / "vjepa2_1_vitb_dist_vitG_384.pt"


def log(*a):
    s = " ".join(map(str, a))
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def enc_kw():
    return dict(patch_size=PATCH, img_size=(IMG, IMG), num_frames=NF,
                tubelet_size=TUB, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
                uniform_power=False, use_rope=True, img_temporal_dim_size=1,
                interpolate_rope=True)


def pred_kw(d):
    return dict(img_size=(IMG, IMG), patch_size=PATCH, use_mask_tokens=True,
                embed_dim=d, predictor_embed_dim=384, out_embed_dim=d,
                num_frames=NF, tubelet_size=TUB, depth=12, num_heads=12,
                num_mask_tokens=8, use_rope=True, uniform_power=False,
                use_sdpa=True, use_silu=False, wide_silu=True,
                n_output_distillation=1, return_all_tokens=False,
                img_temporal_dim_size=1, interpolate_rope=True)


def build_enc():
    if not VJEPA_CKPT.exists():
        log("downloading official V-JEPA 2.1 ckpt ...")
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
            str(VJEPA_CKPT))
    e = vjepa_vit.vit_base(**enc_kw())
    sd = torch.load(VJEPA_CKPT, map_location="cpu", weights_only=True)["ema_encoder"]
    e.load_state_dict({k.replace("module.", "").replace("backbone.", ""): v
                       for k, v in sd.items()}, strict=False)
    e = e.cuda().eval()
    for p in e.parameters():
        p.requires_grad_(False)
    return e


class MLP(nn.Module):
    def __init__(s, din, h, dout):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, dout))

    def forward(s, a):
        return s.net(a)


enc = build_enc()
pred = vit_predictor(**pred_kw(768)).cuda()
mlp = MLP(6, 1024, 768).cuda()
vh = MLP(768, 512, 3).cuda()
opt = torch.optim.AdamW(list(pred.parameters()) + list(mlp.parameters()) +
                        list(vh.parameters()), lr=1e-4, weight_decay=0.05)
tr = [p for p in list(pred.parameters()) + list(mlp.parameters()) +
      list(vh.parameters()) if p.requires_grad]

# ---- data ---------------------------------------------------------------
_topic_cache, _frames_cache = {}, {}


def ok(m):
    d = ROOT / m
    return ((d / "images" / "hdr_front").exists()
            and (d / "data" / "anymal_command_twist").exists()
            and (d / "data" / "anymal_state_state_estimator").exists())


def frames(m):
    if m not in _frames_cache:
        d = ROOT / m / "images" / "hdr_front"
        fs = sorted([p for p in d.iterdir() if p.suffix.lower() in
                     {".jpg", ".jpeg", ".png"}], key=lambda p: p.name) if d.exists() else []
        _frames_cache[m] = fs
    return _frames_cache[m]


def topics(m):
    import zarr
    if m not in _topic_cache:
        d = ROOT / m
        g = zarr.open_group(str(d / "data" / "anymal_command_twist"), mode="r")
        ze = zarr.open_group(str(d / "data" / "anymal_state_state_estimator"), mode="r")
        zc = zarr.open_group(str(d / "data" / "hdr_front"), mode="r")
        _topic_cache[m] = dict(tsc=np.asarray(g["timestamp"][:]),
                               lin=np.asarray(g["linear"][:]),
                               ang=np.asarray(g["angular"][:]),
                               tse=np.asarray(ze["timestamp"][:]),
                               tw=np.asarray(ze["twist_lin"][:]),
                               cam=np.asarray(zc["timestamp"][:]))
    return _topic_cache[m]


def winmean(ts, arr, cam, f0, f1, pad=1):
    t0, t1 = cam[f0], cam[f1]
    i0 = max(0, int(np.searchsorted(ts, t0)) - pad)
    i1 = min(len(ts), int(np.searchsorted(ts, t1)) + pad)
    return arr[i0:i1].mean(0) if i1 > i0 else np.zeros(arr.shape[1], np.float32)


def make_recs(m, maxc=140):
    fs = frames(m)
    if len(fs) < NF * STRIDE + 2:
        return []
    out = []
    for s in range(0, len(fs) - 1 - (NF - 1) * STRIDE, NF):
        out.append(tuple(s + i * STRIDE for i in range(NF)))
        if len(out) >= maxc:
            break
    return out


def build_train_recs():
    out = []
    for m in CORE + ADD:
        if ok(m) and len(frames(m)) >= 400:
            out.extend([(m, r) for r in make_recs(m)])
    return out


def decode(m, rec):
    fs = frames(m)
    side = int(IMG / 224 * 256)
    im = [TF.center_crop(TF.resize(Image.open(fs[i]).convert("RGB"), side), IMG)
          for i in rec]
    v = torch.stack([TF.to_tensor(x) for x in im], 1)
    mn = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    sd = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    return (v - mn) / sd


def clip_meta(m, rec):
    t = topics(m)
    acts = np.stack([np.concatenate([winmean(t["tsc"], t["lin"], t["cam"],
                                             rec[2 * j], rec[2 * j + 1]),
                                     winmean(t["tsc"], t["ang"], t["cam"],
                                             rec[2 * j], rec[2 * j + 1])])
                     for j in range(1, T)], 0).astype(np.float32)
    vels = np.stack([winmean(t["tse"], t["tw"], t["cam"], rec[2 * j],
                             rec[2 * j + 1]) for j in range(1, T)], 0).astype(np.float32)
    return acts, vels


def vis_masks(b_tok, b=B):
    tidx = torch.arange(T * P, device="cuda")
    tt = tidx // P
    return (tidx[tt < b_tok].repeat(b, 1), tidx[tt >= b_tok].repeat(b, 1))


def fwd(vids, mx, my, mx_all, acts, b_tok):
    ctx = enc(vids, masks=[mx])
    atok = mlp(acts[:, b_tok - 1:])
    out, _ = pred(torch.cat([ctx, atok], 1), [mx_all], [my], mod="video")
    nw = T - b_tok
    return out, vh(out.view(vids.shape[0], nw, P, 768).mean(2))


def teacher(vids, my):
    with torch.no_grad():
        full = enc(vids)
    return torch.gather(full, 1, my.unsqueeze(-1).expand(-1, -1, 768))


# ---- validation pools ----------------------------------------------------
_val = {}
for _m, _tag in VAL.items():
    _recs = [(_m, r) for r in make_recs(_m, maxc=24)]
    _cl, _ac, _vl = [], [], []
    for _x, _r in _recs:
        _cl.append(decode(_m, _r))
        _a, _v = clip_meta(_m, _r)
        _ac.append(_a)
        _vl.append(_v)
    _val[_tag] = (torch.stack(_cl), torch.from_numpy(np.stack(_ac)),
                  torch.from_numpy(np.stack(_vl)))


def evaluate(tag):
    clips, acts, vels = _val[tag]
    enc.eval(); pred.eval(); mlp.eval(); vh.eval()
    b_tok = 4
    mx, my = vis_masks(b_tok, b=6)
    ai = torch.arange(ACT_BASE, ACT_BASE + (T - b_tok), device="cuda").repeat(6, 1)
    mx_all = torch.cat([mx, ai], 1)
    cs, e, sh = [], [], []
    with torch.no_grad():
        for i in range(0, len(clips), 6):
            v = clips[i:i + 6].cuda()
            a = acts[i:i + 6].cuda()
            ve = vels[i:i + 6].cuda()
            out, vel = fwd(v, mx, my, mx_all, a, b_tok)
            tg = teacher(v, my)
            out2, vel2 = fwd(v, mx, my, mx_all, torch.roll(a, 1, 0), b_tok)
            p = F.layer_norm(out.float(), (768,))
            t = F.layer_norm(tg.float(), (768,))
            cs.append(F.cosine_similarity(p, t, dim=-1).mean().item())
            e.append(float(F.mse_loss(vel.float(), ve[:, 3:]).sqrt()))
            sh.append(float(F.mse_loss(vel2.float(), ve[:, 3:]).sqrt()))
    return float(np.mean(cs)), float(np.mean(e)), float(np.mean(sh))


# ---- resume --------------------------------------------------------------
def _latest_ckpt():
    _fs = sorted(CK.glob("big_run_step*.pt"),
                 key=lambda p: int(p.stem.rsplit("step", 1)[1]))
    return _fs[-1] if _fs else None


_start = 0
_lc = _latest_ckpt()
if _lc is not None:
    _sd = torch.load(_lc, map_location="cuda", weights_only=True)
    pred.load_state_dict(_sd["pred"])
    mlp.load_state_dict(_sd["mlp"])
    vh.load_state_dict(_sd["vh"])
    _start = int(_sd.get("step", 0))
    log("resume from", _lc.name, "step", _start)

# ---- main loop -----------------------------------------------------------
_rng = np.random.default_rng(0)
_recs_train = build_train_recs()
log("big_run start | missions:", sum(1 for m in CORE + ADD if ok(m)),
    "| train recs:", len(_recs_train), "| steps:", STEPS, "| batch:", B)
t0 = time.time()
for s in range(_start, STEPS):
    if s % 500 == 0 and s > _start:
        _recs_train = build_train_recs()
        log("step", s, "rebuilt train recs:", len(_recs_train))
    b = BOUND[int(_rng.integers(0, len(BOUND)))]
    b_tok = b // 2
    mx, my = vis_masks(b_tok)
    ai = torch.arange(ACT_BASE, ACT_BASE + (T - b_tok), device="cuda").repeat(B, 1)
    mx_all = torch.cat([mx, ai], 1)
    _got, _trs = [], 0
    while len(_got) < B and _trs < 60:
        _trs += 1
        _m, _r = _recs_train[int(_rng.integers(0, len(_recs_train)))]
        try:
            _vid = decode(_m, _r)
            _a, _v = clip_meta(_m, _r)
        except OSError:
            continue
        _got.append((_vid, _a, _v))
    vids = torch.stack([g[0] for g in _got]).cuda()
    acts = torch.from_numpy(np.stack([g[1] for g in _got])).cuda()
    vels = torch.from_numpy(np.stack([g[2] for g in _got])).cuda()
    if _rng.random() < ACT_P:
        acts = torch.zeros_like(acts)
    opt.zero_grad(set_to_none=True)
    out, vel = fwd(vids, mx, my, mx_all, acts, b_tok)
    d = torch.stack([(vids[:, :, 2 * j] - vids[:, :, 2 * j - 1]).abs()
                     .mean(dim=(1, 2, 3)) for j in range(b_tok, T)], 1)
    w = (1.0 + 2.0 * (d / (d.mean(1, keepdim=True) + 1e-6) - 1.0)).clamp(0.25, 4.0)
    p = F.layer_norm(out.float(), (768,))
    t = F.layer_norm(teacher(vids, my).float(), (768,))
    diff = (p - t).abs()
    rows = torch.arange(diff.shape[1], device="cuda") // P
    wt = w[:, rows].unsqueeze(-1)
    f_loss = (diff * wt).mean()
    v_loss = F.mse_loss(vel.float(), vels[:, b_tok - 1:])
    loss = f_loss + LAM * v_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tr, 1.0)
    opt.step()
    if (s + 1) % 50 == 0:
        log(f"step {s + 1:5d} | feat {float(f_loss.detach()):.4f} | "
            f"vel {float(v_loss.detach()):.4f} | "
            f"{time.time()-t0:5.0f}s")
    if (s + 1) % 200 == 0 or (s + 1) == STEPS:
        mets = {}
        for tag in _val:
            c, e, sh = evaluate(tag)
            mets[tag] = dict(cos=c, vel_rmse=e, vel_shuf=sh)
            log(f"val {tag} | cos {c:.4f} | velRMSE true {e:.4f} | shuf {sh:.4f}")
        torch.save(dict(pred=pred.state_dict(), mlp=mlp.state_dict(),
                        vh=vh.state_dict(), step=s + 1, mets=mets),
                   CK / f"big_run_step{s + 1}.pt")
        log("saved big_run_step", s + 1)
with open(str(CK) + "/big_run_DONE", "w") as f:
    f.write(json.dumps({"steps": STEPS}))
log("big_run complete")
