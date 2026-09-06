import sys, json, time
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

LOG = Path(str(CK.parent) + "/eval1_log.txt")
IMG, NF, PATCH, TUB = 384, 16, 16, 2
P = (IMG // PATCH) ** 2; T = NF // TUB; B = 4; STRIDE = 4; GRID = 24


def log(*a):
    s = " ".join(map(str, a))
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


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


class MLP(nn.Module):
    def __init__(s, din, h, dout):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, dout))
    def forward(s, a):
        return s.net(a)


enc = vjepa_vit.vit_base(**ekw())
sd = torch.load(CK / "vjepa2_1_vitb_dist_vitG_384.pt", map_location="cpu",
                weights_only=True)["ema_encoder"]
enc.load_state_dict({k.replace("module.", "").replace("backbone.", ""): v
                     for k, v in sd.items()}, strict=False)
enc = enc.cuda().eval()
for p in enc.parameters():
    p.requires_grad_(False)
pred = vit_predictor(**pkw(768)).cuda().eval()
mlp = MLP(6, 1024, 768).cuda().eval()
fs = torch.load(CK / "big_run_step8000.pt", map_location="cuda", weights_only=True)
pred.load_state_dict(fs["pred"]); mlp.load_state_dict(fs["mlp"])


class Dec(nn.Module):
    def __init__(s):
        super().__init__()
        s.cmd_mlp = nn.Linear(6, 768)
        def up(ci, co):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear",
                                 align_corners=False), nn.Conv2d(ci, co, 3, padding=1),
                                 nn.GroupNorm(32, co), nn.GELU(),
                                 nn.Conv2d(co, co, 3, padding=1), nn.GELU())
        s.in_proj = nn.Sequential(nn.Conv2d(768, 512, 1), nn.GroupNorm(32, 512), nn.GELU())
        s.u1 = up(512, 256); s.u2 = up(256, 128); s.u3 = up(128, 64); s.u4 = up(64, 32)
        s.ref = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                              nn.Conv2d(32, 16, 3, padding=1), nn.GELU(),
                              nn.Conv2d(16, 3, 3, padding=1))
    def forward(s, toks, cmd):
        x = (toks + s.cmd_mlp(cmd).unsqueeze(1)).permute(0, 2, 1).view(-1, 768, GRID, GRID)
        x = s.in_proj(x); x = s.u1(x); x = s.u2(x); x = s.u3(x); x = s.u4(x)
        return torch.sigmoid(s.ref(x))


dec = Dec().cuda()
if (CK / "decoder_cmd.pt").exists():
    dec.load_state_dict(torch.load(CK / "decoder_cmd.pt", map_location="cuda",
                                   weights_only=True)["dec"])
dec.train()


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


def den(fr):
    mn = torch.tensor([0.485, 0.456, 0.406], device=fr.device).view(3, 1, 1)
    sd_ = torch.tensor([0.229, 0.224, 0.225], device=fr.device).view(3, 1, 1)
    return (fr * sd_ + mn).clamp(0, 1)


def _topic_data(m):
    import zarr
    d = ROOT / m
    gc = zarr.open_group(str(d / "data" / "anymal_command_twist"), mode="r")
    zc = zarr.open_group(str(d / "data" / "hdr_front"), mode="r")
    return (np.asarray(gc["timestamp"][:]), np.concatenate(
        [np.asarray(gc["linear"][:]), np.asarray(gc["angular"][:])], 1),
        np.asarray(zc["timestamp"][:]))


_tc = {}


def _top(m):
    if m not in _tc:
        _tc[m] = _topic_data(m)
    return _tc[m]


def cmd_win(m, rec, j):
    ts, cmd, cam = _top(m)
    t0, t1 = cam[rec[2 * j]], cam[rec[2 * j + 1]]
    i0 = max(0, int(np.searchsorted(ts, t0)) - 1)
    i1 = min(len(ts), int(np.searchsorted(ts, t1)) + 1)
    return cmd[i0:i1].mean(0).astype(np.float32) if i1 > i0 else np.zeros(6, np.float32)


def pool(m, n):
    fl = frames(m)
    recs = []
    for st in range(0, len(fl) - 1 - 15 * STRIDE, NF):
        recs.append(tuple(st + i * STRIDE for i in range(NF)))
        if len(recs) >= n:
            break
    out = []
    for r in recs:
        out.append((decode(m, r), np.stack([cmd_win(m, r, j) for j in range(4, T)])))
        if len(out) % 30 == 0:
            log(f"  pool {m[:8]} {len(out)}")
    return out, recs


# continue decoder training w/ sharper losses on ETH teacher cond (j random 4..7)
print("refining decoder (perceptual-ish losses) ...", flush=True)
eth, _ = pool("2024-10-01-11-29-55", 160)
opt = torch.optim.AdamW(dec.parameters(), lr=1e-4, weight_decay=0.01)
rng = np.random.default_rng(1)


def ssim_map(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(a, 11, stride=1, padding=5)
    mu2 = F.avg_pool2d(b, 11, stride=1, padding=5)
    s1 = F.avg_pool2d(a * a, 11, 1, 5) - mu1 * mu1
    s2 = F.avg_pool2d(b * b, 11, 1, 5) - mu2 * mu2
    s12 = F.avg_pool2d(a * b, 11, 1, 5) - mu1 * mu2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    dnm = (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)
    return (num / (dnm + 1e-8)).clamp(0, 1)


for st in range(300):
    idx = rng.integers(0, len(eth), size=B)
    clips = [eth[i][0] for i in idx]
    cmds = np.stack([eth[i][1] for i in idx])
    j = 4 + int(rng.integers(0, 4))
    v = torch.stack(clips).cuda()
    with torch.no_grad():
        full = enc(v)
    tok = full.view(B, T, P, 768)[:, j].float()
    gt = den(v[:, :, 2 * j].contiguous())
    opt.zero_grad(set_to_none=True)
    rec = dec(tok, torch.from_numpy(cmds[:, j - 4]).cuda().float())
    l1 = F.l1_loss(rec, gt)
    gx = F.l1_loss(rec[:, :, :, 1:] - rec[:, :, :, :-1],
                   gt[:, :, :, 1:] - gt[:, :, :, :-1])
    loss = l1 + 0.4 * gx + 0.4 * (1 - ssim_map(rec, gt).mean())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
    opt.step()
    if (st + 1) % 100 == 0:
        log(f"dec refine {st + 1} loss {float(loss.detach()):.4f}")
torch.save({"dec": dec.state_dict()}, CK / "decoder_cmd_v2.pt")
dec.eval()


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse < 1e-12 else 10 * np.log10(1 / mse)


def dec_sources(clips, cmds_b, j=7):
    B2 = clips.shape[0]
    out = {}
    with torch.no_grad():
        full = enc(clips)
        tok_t = full.view(B2, T, P, 768)[:, j].float()
        btok = 4
        tidx = torch.arange(T * P, device="cuda")
        tt = tidx // P
        mx = tidx[tt < btok].repeat(B2, 1)
        my = tidx[tt >= btok].repeat(B2, 1)
        nw = T - btok
        ai = torch.arange(T * P, T * P + nw, device="cuda").repeat(B2, 1)
        mx_all = torch.cat([mx, ai], 1)
        ctx = enc(clips, masks=[mx])
        for name, sh in (("true", 0), ("shuf", 1)):
            acts = torch.roll(cmds_b, sh, dims=0)
            atok = mlp(acts)
            pr, _ = pred(torch.cat([ctx, atok], 1), [mx_all], [my], mod="video")
            out[name] = dec(pr[:, (j - 4) * P:(j - 3) * P].float(),
                            acts[:, j - 4])
        out["teacher"] = dec(tok_t, cmds_b[:, j - 4])
    return out


def evald(m, tag, n=16):
    pl, _ = pool(m, n)
    rng = np.random.default_rng(3)
    acc = {k: [] for k in ("teacher", "true", "shuf")}
    for _ in range(n // B):
        idx = rng.integers(0, len(pl), size=B)
        v = torch.stack([pl[i][0] for i in idx]).cuda()
        cm = torch.from_numpy(np.stack([pl[i][1] for i in idx])).cuda().float()
        gt = den(v[:, :, 14].contiguous())
        res = dec_sources(v, cm, j=7)
        for k in acc:
            r = res[k]
            acc[k].append((float(psnr(r, gt)),
                           float(1 - ssim_map(r, gt).mean()),
                           float(F.mse_loss(r, gt).sqrt())))
    res = {k: tuple(float(np.mean([x[i] for x in acc[k]])) for i in range(3))
           for k in acc}
    log(f"[{tag}] " + " | ".join(f"{k}: PSNR {res[k][0]:.2f} SSIM {res[k][1]:.3f} "
                                  f"RMSE {res[k][2]:.4f}" for k in res))
    return res


r1 = evald("2024-11-18-13-48-19", "ICE-1")
r2 = evald("2024-11-02-17-18-32", "SPX-2")

# robot-mask region metrics on ETH-1
def mask_at(idx):
    p = ROOT / "2024-10-01-11-29-55" / "images" / "hdr_front_mask" / f"{int(idx):06d}.png"
    if not p.exists():
        return None
    mk = TF.center_crop(TF.resize(Image.open(p).convert("L"), int(IMG / 224 * 256)), IMG)
    mk = TF.to_tensor(mk).cuda()
    return (mk < 0.5).float()  # robot = black


def eth_mask_metrics(n=12):
    pl, recs = pool("2024-10-01-11-29-55", n)
    rng = np.random.default_rng(5)
    inacc = {k: [] for k in ("teacher", "true", "shuf")}
    mot = []
    for it in range(n // B):
        idx = rng.integers(0, len(pl), size=B)
        v = torch.stack([pl[i][0] for i in idx]).cuda()
        cm = torch.from_numpy(np.stack([pl[i][1] for i in idx])).cuda().float()
        gt = den(v[:, :, 14].contiguous())
        res = dec_sources(v, cm, j=7)
        masks = []
        for i in idx:
            mk = mask_at(recs[i][14])
            masks.append(mk if mk is not None else torch.ones(1, IMG, IMG, device="cuda"))
        M = torch.stack(masks)
        for k in inacc:
            err = (res[k] - gt).abs()
            inacc[k].append(float((err * M).sum() / (M.sum() + 1e-6)))
        mot.append(float(((res["shuf"] - res["true"]).abs() * M).mean()
                         - ((res["shuf"] - res["true"]).abs() * (1 - M)).mean()))
    log("ETH-1 mask-region MAE: " + " | ".join(
        f"{k} {float(np.mean(inacc[k])):.4f}" for k in inacc) +
        f" | cmd-effect in-mask minus out-mask: {float(np.mean(mot)):.4f}")
    return {k: float(np.mean(inacc[k])) for k in inacc}, float(np.mean(mot))


mk_res, mot = eth_mask_metrics(12)
json.dump({"ice": r1, "spx": r2, "mask": mk_res, "motion": mot},
          open(str(CK) + "/eval1_results.json", "w"), indent=1)
log("eval1 done")
