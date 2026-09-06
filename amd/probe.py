"""Final-model probe suite + causal-vs-random + hidden-state readouts.

python amd/probe.py --model ckpts/big_run_step8000.pt
python amd/probe.py --model ckpts/big_run_step8000.pt --vs-random ckpts/rand_pred.pt
python amd/probe.py --model ... --train-random 1200     # train matched random first

Outputs: far-horizon (3 s) displacement teacher/causal/copy(+random) on
transient & steady windows of held-out ICE-1/SPX-2; velocity control-signal;
contact + bob hidden-state readouts.
"""
import argparse
import sys
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
IMG, NF, PATCH, TUB = 384, 16, 16, 2
P = (IMG // PATCH) ** 2
T = NF // TUB
B = 6
STRIDE = 4
CORE = ["2024-10-01-11-29-55", "2024-10-01-11-47-44",
        "2024-10-01-12-00-49", "2024-11-02-17-10-25"]


def ekw():
    return dict(patch_size=PATCH, img_size=(IMG, IMG), num_frames=NF,
                tubelet_size=TUB, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
                uniform_power=False, use_rope=True, img_temporal_dim_size=1,
                interpolate_rope=True)


def pkw(d):
    return dict(img_size=(IMG, IMG), patch_size=PATCH, use_mask_tokens=True,
                embed_dim=d, predictor_embed_dim=384, out_embed_dim=d,
                num_frames=NF, tubelet_size=TUB, depth=12, num_heads=12,
                num_mask_tokens=8, use_rope=True, uniform_power=False,
                use_sdpa=True, use_silu=False, wide_silu=True,
                n_output_distillation=1, return_all_tokens=False,
                img_temporal_dim_size=1, interpolate_rope=True)


def build_enc():
    e = vjepa_vit.vit_base(**ekw())
    sd = torch.load(CK / "vjepa2_1_vitb_dist_vitG_384.pt", map_location="cpu",
                    weights_only=True)["ema_encoder"]
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
pred = vit_predictor(**pkw(768)).cuda().eval()
mlp = MLP(6, 1024, 768).cuda().eval()
vh = MLP(768, 512, 3).cuda().eval()
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="big_run_step8000.pt")
ap.add_argument("--vs-random", default=None)
ap.add_argument("--train-random", type=int, default=0)
args = ap.parse_args()
fs = torch.load(CK / args.model, map_location="cuda", weights_only=True)
pred.load_state_dict(fs["pred"])
mlp.load_state_dict(fs["mlp"])
vh.load_state_dict(fs["vh"])
print("loaded", args.model, flush=True)

_frames_cache = {}
_top_cache = {}


def frames(m):
    if m not in _frames_cache:
        d = ROOT / m / "images" / "hdr_front"
        fs2 = sorted([p for p in d.iterdir() if p.suffix.lower() in
                      {".jpg", ".jpeg"}], key=lambda p: p.name) if d.exists() else []
        _frames_cache[m] = fs2
    return _frames_cache[m]


def decode(m, rec):
    fl = frames(m)
    side = int(IMG / 224 * 256)
    im = [TF.center_crop(TF.resize(Image.open(fl[i]).convert("RGB"), side), IMG)
          for i in rec]
    v = torch.stack([TF.to_tensor(x) for x in im], 1)
    mn = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    sd = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    return (v - mn) / sd


def data(m):
    import zarr
    if m not in _top_cache:
        d = ROOT / m
        zd = zarr.open_group(str(d / "data" / "dlio_map_odometry"), mode="r")
        zc = zarr.open_group(str(d / "data" / "hdr_front"), mode="r")
        ze = zarr.open_group(str(d / "data" / "anymal_state_state_estimator"), mode="r")
        _top_cache[m] = dict(tsd=np.asarray(zd["timestamp"][:]),
                             pos=np.asarray(zd["pose_pos"][:]),
                             ori=np.asarray(zd["pose_orien"][:]),
                             cam=np.asarray(zc["timestamp"][:]),
                             tse=np.asarray(ze["timestamp"][:]),
                             tw=np.asarray(ze["twist_lin"][:]),
                             ang=np.asarray(ze["twist_ang"][:]))
    return _top_cache[m]


class M:
    def __init__(s, mission):
        d = data(mission)
        s.__dict__.update(d)

    def _n(s, ts, arr):
        i = int(np.searchsorted(arr, ts))
        if i >= len(arr):
            i = len(arr) - 1
        if i > 0 and abs(arr[i - 1] - ts) < abs(arr[i] - ts):
            i -= 1
        return i

    def profile(s, rec):
        ie = [s._n(s.cam[f], s.tse) for f in rec[::2]]
        v = np.array([np.linalg.norm(s.tw[i][:2]) for i in ie])
        a = np.abs(s.ang[ie][:, 2])
        return float(a.mean() * 2.0 + (v.max() - v.min()) / (v.mean() + 0.05))

    def dydyaw(s, fr, fe):
        def R(q):
            x, y, z, w = q
            return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                             [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                             [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]],
                            dtype=float)
        ir, ie2 = s._n(s.cam[fr], s.tsd), s._n(s.cam[fe], s.tsd)
        Rr, Re = R(s.ori[ir]), R(s.ori[ie2])
        d = (Rr.T @ (s.pos[ie2] - s.pos[ir]))[:2]
        ya = np.arctan2(Re[1, 0], Re[0, 0]) - np.arctan2(Rr[1, 0], Rr[0, 0])
        return np.array([d[0], d[1], ya], np.float32)

    def dz(s, f0, f1):
        return float(s.pos[s._n(s.cam[f1], s.tsd)][2]
                     - s.pos[s._n(s.cam[f0], s.tsd)][2])


def sel(m, nt, ns):
    mm = M(m)
    cands = []
    for st in range(0, len(mm.cam) - 1 - 15 * STRIDE, 6):
        rec = tuple(st + i * STRIDE for i in range(16))
        cands.append((mm.profile(rec), rec))
    cands.sort(key=lambda t: t[0], reverse=True)
    return [c[1] for c in cands[:nt]], [c[1] for c in cands[-ns:]], mm


def build(m, nt, ns):
    tr, st, mm = sel(m, nt, ns)
    out = {}
    for tag, recs in (("trans", tr), ("steady", st)):
        cl, y7, zz = [], [], []
        for rec in recs:
            cl.append(decode(m, rec))
            y7.append(mm.dydyaw(rec[7], rec[15]))
            zz.append(mm.dz(rec[13], rec[15]))
        out[tag] = (cl, np.array(y7, np.float32), np.array(zz, np.float32))
        print(" built", m[:8], tag, len(cl), flush=True)
    return out


def causal_tokens(v, j, model_pred=pred):
    b = v.shape[0]
    btok = 4
    tidx = torch.arange(T * P, device="cuda")
    tt = tidx // P
    mx = tidx[tt < btok].repeat(b, 1)
    my = tidx[tt >= btok].repeat(b, 1)
    nw = T - btok
    ai = torch.arange(T * P, T * P + nw, device="cuda").repeat(b, 1)
    mx_all = torch.cat([mx, ai], 1)
    ctx = enc(v, masks=[mx])
    atok = mlp(torch.zeros(b, nw, 6, device="cuda"))
    out, _ = model_pred(torch.cat([ctx, atok], 1), [mx_all], [my], mod="video")
    return out[:, (j - 4) * P:(j - 3) * P].float()


def feats(clips, source, j=7, rand_pred=None):
    outs = []
    with torch.no_grad():
        for i in range(0, len(clips), B):
            sl = clips[i:i + B]
            v = torch.stack(sl).cuda()
            if source == "teacher":
                tok = enc(v).view(v.shape[0], T, P, 768)[:, j]
            elif source == "causal":
                tok = causal_tokens(v, j)
            elif source == "random":
                tok = causal_tokens(v, j, model_pred=rand_pred)
            else:  # copy: last context tubelet
                tok = enc(v).view(v.shape[0], T, P, 768)[:, 3]
            outs.append(tok.float().mean(1))
    return torch.cat(outs)


def fit(X, y, dim, iters=250):
    h = nn.Linear(768, dim).cuda()
    o = torch.optim.AdamW(h.parameters(), lr=1e-3)
    yt = torch.as_tensor(np.asarray(y, np.float32)).cuda()
    for _ in range(iters):
        idx = torch.randperm(len(X), device="cuda")[:24]
        loss = F.mse_loss(h(X[idx]), yt[idx])
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()
    return h.eval()


ETH = "2024-10-01-11-29-55"
VAL = {"2024-11-18-13-48-19": "ICE-1", "2024-11-02-17-18-32": "SPX-2"}

# optional: train matched random-mask model on available corpus
rand_pred = None
if args.train_random or args.vs_random:
    if args.vs_random:
        rand_pred = vit_predictor(**pkw(768)).cuda().eval()
        rand_pred.load_state_dict(torch.load(CK / args.vs_random,
                                             map_location="cuda",
                                             weights_only=True)["pred"])
    else:
        from random import shuffle
        rand_pred = vit_predictor(**pkw(768)).cuda()
        ropt = torch.optim.AdamW(rand_pred.parameters(), lr=1e-4,
                                 weight_decay=0.05)
        mlist = [m for m in CORE
                 if (ROOT / m / "images" / "hdr_front").exists()]
        recs = []
        for m in mlist:
            fl = frames(m)
            for st in range(0, len(fl) - 61, 16):
                recs.append((m, tuple(st + i * STRIDE for i in range(16))))
                if len(recs) >= 3000:
                    break
        rng = np.random.default_rng(0)
        print("training random-mask predictor", args.train_random, "steps ...",
              flush=True)
        for s in range(args.train_random):
            idx = rng.integers(0, len(recs), size=4)
            v = torch.stack([decode(*recs[i]) for i in idx]).cuda()
            mx, my = (torch.randperm(T * P, generator=torch.Generator().manual_seed(
                5000 + s), device="cuda")[: int(T * P * 0.5)].repeat(4, 1),
                      None)
            n_ctx = int(T * P * 0.5)
            perm = torch.randperm(T * P, device="cuda")
            mx = perm[:n_ctx].repeat(4, 1)
            my = perm[n_ctx:].repeat(4, 1)
            ropt.zero_grad(set_to_none=True)
            ctx = enc(v, masks=[mx])
            out, _ = rand_pred(ctx, [mx], [my], mod="video")
            with torch.no_grad():
                full = enc(v)
            tg = torch.gather(full, 1, my.unsqueeze(-1).expand(-1, -1, 768))
            loss = F.smooth_l1_loss(F.layer_norm(out.float(), (768,)),
                                    F.layer_norm(tg.float(), (768,)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rand_pred.parameters(), 1.0)
            ropt.step()
            if (s + 1) % 200 == 0:
                print(" rand step", s + 1, "loss", float(loss.detach()),
                      flush=True)
        torch.save({"pred": rand_pred.state_dict()}, CK / "rand_pred.pt")
        rand_pred.eval()
        print("saved rand_pred.pt", flush=True)

print("building probe sets ...", flush=True)
tr = build(ETH, 40, 40)
val = {t: build(m, 24, 24) for m, t in VAL.items()}
_Xf = torch.cat([feats(tr[k][0], "teacher") for k in ("trans", "steady")])
_yf = np.concatenate([tr[k][1] for k in ("trans", "steady")])
_zf = np.concatenate([tr[k][2] for k in ("trans", "steady")])
hd = fit(_Xf, _yf, 3)
hz = fit(_Xf, _zf, 1)
print("heads fit", flush=True)
srcs = ["teacher", "causal"] + (["random"] if rand_pred is not None else []) + ["copy"]
for m, t in VAL.items():
    for k in ("trans", "steady"):
        clips, y7, zz = val[m][k]
        for src in srcs:
            X = feats(clips, src)
            with torch.no_grad():
                ed = hd(X).cpu().numpy()
                ez = hz(X).cpu().numpy()[:, 0]
            d2 = float(np.sqrt(np.mean(np.sum((ed - y7)[:, :2] ** 2, axis=1))))
            dzr = float(np.sqrt(np.mean((ez - zz) ** 2)))
            print(f"[{t} {k}] {src:7s} d {d2:.3f}m | bob {dzr:.4f}", flush=True)
print("probe done", flush=True)
