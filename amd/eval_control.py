import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, "/marimo/vjepa2")
from app.vjepa_2_1.models import vision_transformer as vjepa_vit
from app.vjepa_2_1.models.predictor import vit_predictor

HERE = Path(__file__).resolve().parent
BASE = Path("/marimo") if Path("/marimo").exists() else HERE.parent
ROOT = (Path("/marimo/grandtour") if Path("/marimo").exists()
        else BASE / "data" / "grandtour")
CK = (Path("/marimo/checkpoints") if Path("/marimo").exists()
     else BASE / "ckpts")

IMG, NF, PATCH, TUB = 384, 16, 16, 2
P = (IMG // PATCH) ** 2; T = NF // TUB; B = 4; STRIDE = 4
STEPS = 500
CMD_DIM, PROP_DIM = 6, 7            # cmd: lin3+ang3 ; proprio: twist_lin3 + per-leg joint-vel rms4


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


enc = vjepa_vit.vit_base(**ekw())
sd = torch.load(CK / "vjepa2_1_vitb_dist_vitG_384.pt", map_location="cpu",
                weights_only=True)["ema_encoder"]
enc.load_state_dict({k.replace("module.", "").replace("backbone.", ""): v
                     for k, v in sd.items()}, strict=False)
enc = enc.cuda().eval()
for p in enc.parameters():
    p.requires_grad_(False)

pred = vit_predictor(**pkw(768)).cuda()
mlp = nn.Sequential(nn.Linear(CMD_DIM + PROP_DIM, 1024), nn.GELU(),
                    nn.Linear(1024, 768)).cuda()
vh = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 3)).cuda()
opt = torch.optim.AdamW(list(pred.parameters()) + list(mlp.parameters()) +
                        list(vh.parameters()), lr=1e-4, weight_decay=0.05)
trp = [p for p in list(pred.parameters()) + list(mlp.parameters()) +
       list(vh.parameters()) if p.requires_grad]


def frames(m):
    d = ROOT / m / "images" / "hdr_front"
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}],
                  key=lambda p: p.name)


def decode(m, rec):
    fl = frames(m)
    side = int(IMG / 224 * 256)
    im = [TF.center_crop(TF.resize(Image.open(fl[i]).convert("RGB"), side), IMG)
          for i in rec]
    v = torch.stack([TF.to_tensor(x) for x in im], 1)
    mn = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    sd_ = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    return (v - mn) / sd_


def dat(m):
    import zarr
    d = ROOT / m
    gc = zarr.open_group(str(d / "data" / "anymal_command_twist"), mode="r")
    ze = zarr.open_group(str(d / "data" / "anymal_state_state_estimator"), mode="r")
    zc = zarr.open_group(str(d / "data" / "hdr_front"), mode="r")
    return dict(tsc=np.asarray(gc["timestamp"][:]),
                cmd=np.concatenate([np.asarray(gc["linear"][:]),
                                    np.asarray(gc["angular"][:])], 1),
                tse=np.asarray(ze["timestamp"][:]),
                tw=np.asarray(ze["twist_lin"][:]),
                ang=np.asarray(ze["twist_ang"][:]),
                jv=np.asarray(ze["joint_velocities"][:]),
                cam=np.asarray(zc["timestamp"][:]))


_dc = {}


def D(m):
    if m not in _dc:
        _dc[m] = dat(m)
    return _dc[m]


def _win_mean(ts, arr, cam, f0, f1):
    t0, t1 = cam[f0], cam[f1]
    i0 = max(0, int(np.searchsorted(ts, t0)) - 1)
    i1 = min(len(ts), int(np.searchsorted(ts, t1)) + 1)
    return arr[i0:i1].mean(0) if i1 > i0 else np.zeros(arr.shape[1], np.float32)


def feats(m, rec, j):
    d = D(m)
    cmd = _win_mean(d["tsc"], d["cmd"], d["cam"], rec[2 * j], rec[2 * j + 1])
    tv = _win_mean(d["tse"], d["tw"], d["cam"], rec[2 * j], rec[2 * j + 1])
    jv = _win_mean(d["tse"], d["jv"], d["cam"], rec[2 * j], rec[2 * j + 1])
    leg = np.sqrt((jv.reshape(4, 3) ** 2).mean(1))          # per-leg rms
    prop = np.concatenate([tv, leg])
    return cmd.astype(np.float32), prop.astype(np.float32)


def pool(m, n):
    fl = frames(m)
    recs = []
    for st in range(0, len(fl) - 1 - 15 * STRIDE, NF):
        recs.append(tuple(st + i * STRIDE for i in range(NF)))
        if len(recs) >= n:
            break
    out = []
    for r in recs:
        c = np.stack([feats(m, r, j)[0] for j in range(4, T)])
        pr = np.stack([feats(m, r, j)[1] for j in range(4, T)])
        vv = np.stack([_win_mean(D(m)["tse"], D(m)["tw"], D(m)["cam"],
                                 r[2 * j], r[2 * j + 1]) for j in range(4, T)])
        out.append((decode(m, r), c, pr, vv.astype(np.float32)))
        if len(out) % 40 == 0:
            print("  pool", m[:8], len(out), flush=True)
    return out, recs


def vis(b_tok, b=B):
    tidx = torch.arange(T * P, device="cuda")
    tt = tidx // P
    return (tidx[tt < b_tok].repeat(b, 1), tidx[tt >= b_tok].repeat(b, 1))


def fwd(v, c, p, mx, my, mx_all):
    a = torch.cat([c, p], 2)                       # (B, nw, 13)
    atok = mlp(a)
    ctx = enc(v, masks=[mx])
    out, _ = pred(torch.cat([ctx, atok], 1), [mx_all], [my], mod="video")
    nw = T - 4
    per = out.view(v.shape[0], nw, P, 768).mean(2)
    return out, vh(per)


def teacher(v, my):
    with torch.no_grad():
        full = enc(v)
    return torch.gather(full, 1, my.unsqueeze(-1).expand(-1, -1, 768))


print("building pools ...", flush=True)
tr, _ = pool("2024-10-01-11-29-55", 140)
va, _ = pool("2024-11-18-13-48-19", 24)
rng = np.random.default_rng(0)
print("training multi-stream model (cmd+proprio, per-stream dropout) ...",
      flush=True)
for s in range(STEPS):
    b = 8
    btok = 4
    mx, my = vis(btok)
    nw = T - btok
    ai = torch.arange(T * P, T * P + nw, device="cuda").repeat(B, 1)
    mx_all = torch.cat([mx, ai], 1)
    idx = rng.integers(0, len(tr), size=B)
    v = torch.stack([tr[i][0] for i in idx]).cuda()
    c = torch.from_numpy(np.stack([tr[i][1][btok - 4:] for i in idx])).cuda().float()
    p = torch.from_numpy(np.stack([tr[i][2][btok - 4:] for i in idx])).cuda().float()
    vv = torch.from_numpy(np.stack([tr[i][3][btok - 4:] for i in idx])).cuda().float()
    if rng.random() < 0.25:
        c = torch.zeros_like(c)
    if rng.random() < 0.25:
        p = torch.zeros_like(p)
    opt.zero_grad(set_to_none=True)
    out, vel = fwd(v, c, p, mx, my, mx_all)
    fL = F.smooth_l1_loss(F.layer_norm(out.float(), (768,)),
                          F.layer_norm(teacher(v, my).float(), (768,)))
    vL = F.mse_loss(vel.float(), vv)
    loss = fL + 0.5 * vL
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trp, 1.0)
    opt.step()
    if (s + 1) % 100 == 0:
        print(f"step {s+1} feat {float(fL.detach()):.4f} vel {float(vL.detach()):.4f}",
              flush=True)
torch.save({"pred": pred.state_dict(), "mlp": mlp.state_dict(),
            "vh": vh.state_dict()}, CK / "ms_pred500.pt")
print("saved ms_pred500.pt", flush=True)

# eval: ICE val, per-variant velocity RMSE
pred.eval(); mlp.eval(); vh.eval()
btok = 4
mx, my = vis(btok)
nw = T - btok
ai = torch.arange(T * P, T * P + nw, device="cuda").repeat(B, 1)
mx_all = torch.cat([mx, ai], 1)


def vel_rmse(variant, seed=9):
    rng = np.random.default_rng(seed)
    errs = []
    with torch.no_grad():
        for _ in range(3):
            idx = rng.integers(0, len(va), size=B)
            v = torch.stack([va[i][0] for i in idx]).cuda()
            c = torch.from_numpy(np.stack([va[i][1] for i in idx])).cuda().float()
            p = torch.from_numpy(np.stack([va[i][2] for i in idx])).cuda().float()
            vv = torch.from_numpy(np.stack([va[i][3] for i in idx])).cuda().float()
            if variant == "shuf_cmd":
                c = torch.roll(c, 1, 0)
            elif variant == "shuf_prop":
                p = torch.roll(p, 1, 0)
            elif variant == "shuf_both":
                c = torch.roll(c, 1, 0); p = torch.roll(p, 1, 0)
            elif variant == "zero":
                c = torch.zeros_like(c); p = torch.zeros_like(p)
            elif variant == "rev_cmd":          # novel: reversed command
                c = -c
            elif variant == "half_cmd":          # novel: half-magnitude command
                c = c * 0.5
            _, vel = fwd(v, c, p, mx, my, mx_all)
            errs.append(float(F.mse_loss(vel.float(), vv).sqrt()))
    return float(np.mean(errs))


res = {}
for name in ("true", "shuf_cmd", "shuf_prop", "shuf_both", "zero", "rev_cmd",
             "half_cmd"):
    res[name] = vel_rmse(name)
    print(f"[ICE-1 velRMSE] {name}: {res[name]:.4f}", flush=True)
json.dump(res, open(str(CK) + "/eval2_results.json", "w"), indent=1)
print("eval2 done", flush=True)
