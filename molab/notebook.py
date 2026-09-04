
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    return


@app.cell
def config():
    from pathlib import Path

    # ---- Phase-1 causal temporal masking on V-JEPA 2.1 (ANYmal-D / Grand Tour) ----
    # Real-data paths are filled in by the pull cell once missions are downloaded.

    CONFIG = {
        "base": Path("/marimo"),
        "vjepa_root": Path("/marimo/vjepa2"),
        "data_root": Path("/marimo/grandtour"),
        "ckpt_dir": Path("/marimo/checkpoints"),
        "vjepa_ckpt_url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        "vjepa_ckpt_name": "vjepa2_1_vitb_dist_vitG_384.pt",

        # video geometry (native checkpoint resolution)
        "num_frames": 16,
        "temporal_stride": 1,        # sample every k-th frame (tune after fps measurement)
        "clip_step": 8,
        "img_size": 384,             # official 2.1 ViT-B/384 resolution
        "patch_size": 16,
        "tubelet_size": 2,
        "boundary_frame": 8,         # context = frames [0, t); MUST be even for tubelet k=2

        # training
        "batch_size": 4,
        "lr": 1e-4,
        "ema_momentum": 0.996,
        "grad_clip": 1.0,
        "seed": 42,

        # smoke missions
        "train_missions": ["2024-10-01-11-29-55"],   # ETH-1 Polyterasse
        "val_missions": ["2024-11-18-13-48-19"],     # ICE-1 Ice Palace (held out)
    }
    CONFIG["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    return CONFIG, Path


@app.cell
def imports():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    return F, nn, np, torch


app._unparsable_cell(
    r"""
    import os
    import subprocess
    import sys

    _cfg = CONFIG
    if not (_cfg["vjepa_root"] / "app").exists():
        print("cloning facebookresearch/vjepa2 ->", _cfg["vjepa_root"])N
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/facebookresearch/vjepa2", str(_cfg["vjepa_root"])],
            check=True,
        )
    else:
        print("vjepa2 already present")

    for _pkg in ("timm", "einops"):
        try:
            __import__(_pkg)
        except ImportError:
            print("installing", _pkg)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)

    sys.path.insert(0, str(_cfg["vjepa_root"]))
    vjepa_ready = True
    print("bootstrap ok at", _cfg["vjepa_root"])

    """,
    name="bootstrap"
)


@app.cell
def masks(CONFIG, torch):
    # Causal temporal masker for V-JEPA 2.1 tubelet grids.
    #
    # Token grid: flattened over (temporal-tubelet, spatial-patch) ids,
    #   idx = t_tok * P + s   where P = (H/patch)^2 patches per frame-window.
    # Tubelet t_tok covers frames [t_tok*k, (t_tok+1)*k - 1], k = tubelet_size.
    # A causal split "context = frames < boundary, targets = the rest" is only
    # leak-free when boundary is a multiple of k (no tubelet straddles the cut).

    def temporal_causal_masks(
        boundary_frame, num_temporal_tokens, spatial_tokens,
        tubelet_size=2, batch_size=1, device="cpu",
    ):
        assert boundary_frame % tubelet_size == 0, (
            "boundary_frame must be a multiple of tubelet_size (else a tubelet "
            "straddles the cut and leaks future frames into context)"
        )
        n_ctx_t = boundary_frame // tubelet_size
        assert 0 < n_ctx_t < num_temporal_tokens, "boundary outside the temporal grid"
        total = num_temporal_tokens * spatial_tokens
        idx = torch.arange(total, device=device)
        tt = idx // spatial_tokens                      # temporal id of each token
        ctx = idx[tt < n_ctx_t]
        tgt = idx[tt >= n_ctx_t]
        masks_x = ctx.repeat(batch_size, 1).contiguous()
        masks_y = tgt.repeat(batch_size, 1).contiguous()
        return masks_x, masks_y


    def random_mask(boundary_frame, num_temporal_tokens, spatial_tokens,
                    tubelet_size=2, batch_size=1, mask_ratio=0.75, seed=0, device="cpu"):
        # Random-mask baseline (V-JEPA 2.1 style): keep (1-ratio) random tokens as
        # context. Deterministic per seed so runs are comparable.
        total = num_temporal_tokens * spatial_tokens
        n_target = max(1, min(total - 1, int(round(total * mask_ratio))))
        n_ctx = total - n_target
        gen = torch.Generator(device=device).manual_seed(seed)
        perm = torch.randperm(total, generator=gen, device=device)
        masks_x = perm[:n_ctx].repeat(batch_size, 1).contiguous()
        masks_y = perm[n_ctx:].repeat(batch_size, 1).contiguous()
        return masks_x, masks_y


    # ---- unit tests on the spike geometry (all demo vars private) ---------------
    _P = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
    _T_TOK = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = 2

    _mx, _my = temporal_causal_masks(
        CONFIG["boundary_frame"], _T_TOK, _P,
        tubelet_size=CONFIG["tubelet_size"], batch_size=_B,
    )
    assert _mx.shape == (_B, _P * (CONFIG["boundary_frame"] // CONFIG["tubelet_size"]))
    assert set(_mx[0].tolist()).isdisjoint(set(_my[0].tolist())), "context/target overlap"
    assert _mx.shape[1] + _my.shape[1] == _T_TOK * _P, "masks do not tile the grid"

    _ctx_t = _mx[0] // _P
    _tgt_t = _my[0] // _P
    assert _ctx_t.max().item() < CONFIG["boundary_frame"] // CONFIG["tubelet_size"]
    assert _tgt_t.min().item() >= CONFIG["boundary_frame"] // CONFIG["tubelet_size"]
    assert _ctx_t.max().item() < _tgt_t.min().item(), "future tubelet in context"

    # leakage must be rejected, not silently accepted
    try:
        temporal_causal_masks(CONFIG["boundary_frame"] + 1, _T_TOK, _P,
                              tubelet_size=CONFIG["tubelet_size"], batch_size=1)
        raise AssertionError("odd boundary did not raise")
    except AssertionError as _e:
        assert "multiple of tubelet_size" in str(_e)

    # random-mask baseline sanity
    _rmx, _rmy = random_mask(0, _T_TOK, _P, mask_ratio=0.75, seed=0, batch_size=_B)
    assert set(_rmx[0].tolist()).isdisjoint(set(_rmy[0].tolist()))

    mask_tests_ok = True
    print(f"P={_P} temporal_tokens={_T_TOK} -> context {tuple(_mx.shape)} target {tuple(_my.shape)}")
    print("causal mask tests passed; boundary leakage rejected at odd boundaries")

    return random_mask, temporal_causal_masks


@app.cell
def dataset(CONFIG, Path, torch):
    # Grand Tour (ANYmal-D) hdr_front clip helpers.
    # Functions only - safe to define with no data present. Real JPEG IO is lazy.

    def discover_frames(mission_dir):
        """Sorted image paths of one mission dir ([] if absent)."""
        d = Path(mission_dir) / "images" / "hdr_front"
        if not d.exists():
            return []
        files = [p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        files.sort(key=lambda p: p.name)
        return files


    def build_clip_records(mission_dir, num_frames, stride=1, step=None, max_clips=None):
        """List of index-tuples of length num_frames over a mission's frames."""
        files = discover_frames(mission_dir)
        if len(files) < num_frames:
            return []
        step = step or num_frames // 2
        out = []
        for start in range(0, len(files) - (num_frames - 1) * stride, step):
            idx = tuple(start + i * stride for i in range(num_frames))
            out.append(idx)
            if max_clips is not None and len(out) >= max_clips:
                break
        return out


    def make_clip_records(data_root, missions, num_frames, stride=1, step=None, max_clips=None):
        """Aggregate clip records over missions; each record = (mission, idx_tuple)."""
        records = []
        for m in missions:
            mission_dir = Path(data_root) / m
            for idx in build_clip_records(mission_dir, num_frames, stride, step, max_clips):
                records.append((m, idx))
        return records


    def load_clip(mission_dir, idx, img_size, train=False):
        """(mission_dir, idx_tuple) -> normalized [C, T, H, W] float tensor."""
        from PIL import Image
        import torchvision.transforms.functional as TF

        side = int(img_size / 224 * 256)
        imgs = []
        for i in idx:
            p = mission_dir / "images" / "hdr_front" / f"{int(i):06d}.jpeg"
            img = Image.open(p).convert("RGB")
            img = TF.resize(img, side)
            img = TF.center_crop(img, img_size)
            imgs.append(TF.to_tensor(img))            # [C, H, W] in [0, 1]
        video = torch.stack(imgs, dim=1)              # [C, T, H, W]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return (video - mean) / std


    def dummy_clip_batch(b, c, t, h, w, device="cuda"):
        """Random normalized-looking clip batch for plumbing tests (no data needed)."""
        return torch.randn(b, c, t, h, w, device=device)


    # ---- sanity (no data required) --------------------------------------------
    _dummy = dummy_clip_batch(CONFIG["batch_size"], 3, CONFIG["num_frames"],
                              CONFIG["img_size"], CONFIG["img_size"])
    print("dummy clip batch", tuple(_dummy.shape))
    _missions = CONFIG["train_missions"] + CONFIG["val_missions"]
    _existing = [m for m in _missions
                 if discover_frames(Path(CONFIG["data_root"]) / m)]
    print("missions with frames on molab right now:", _existing or "none (run the pull cell)")
    dataset_ok = True

    return build_clip_records, discover_frames, dummy_clip_batch, load_clip


@app.cell
def model(CONFIG, dummy_clip_batch, temporal_causal_masks, torch, vjepa_ready):
    assert vjepa_ready, "run the bootstrap cell first"

    # Single-owner imports of the vjepa2 submodule code.
    from app.vjepa_2_1.models import vision_transformer as vjepa_vit
    from app.vjepa_2_1.models.predictor import vit_predictor


    def _enc_kwargs(cfg):
        return dict(
            patch_size=cfg["patch_size"],
            img_size=(cfg["img_size"], cfg["img_size"]),
            num_frames=cfg["num_frames"],
            tubelet_size=cfg["tubelet_size"],
            use_sdpa=True, use_SiLU=False, wide_SiLU=True,
            uniform_power=False, use_rope=True,
            img_temporal_dim_size=1, interpolate_rope=True,
        )


    def _pred_kwargs(cfg, embed_dim):
        return dict(
            img_size=(cfg["img_size"], cfg["img_size"]),
            patch_size=cfg["patch_size"],
            use_mask_tokens=True,
            embed_dim=embed_dim,
            predictor_embed_dim=384,
            out_embed_dim=embed_dim,
            num_frames=cfg["num_frames"],
            tubelet_size=cfg["tubelet_size"],
            depth=12, num_heads=12, num_mask_tokens=8,
            use_rope=True, uniform_power=False, use_sdpa=True,
            use_silu=False, wide_silu=True, n_output_distillation=1,
            return_all_tokens=False,
            img_temporal_dim_size=1, interpolate_rope=True,
        )


    def build_spike_models(device="cuda"):
        """Random-init ViT-B encoder + frozen EMA-style target + predictor."""
        torch.manual_seed(CONFIG["seed"])
        enc = vjepa_vit.vit_base(**_enc_kwargs(CONFIG))
        tgt = vjepa_vit.vit_base(**_enc_kwargs(CONFIG))
        pred = vit_predictor(**_pred_kwargs(CONFIG, enc.embed_dim))
        enc.to(device); tgt.to(device); pred.to(device)
        tgt.eval()
        for p in tgt.parameters():
            p.requires_grad_(False)
        return enc, tgt, pred


    def wm_forward(video, masks_x, masks_y, encoder, target_encoder, predictor):
        """(video, [B,nctx], [B,ntgt]) -> (pred [B,ntgt,D], tgt [B,ntgt,D])."""
        ctx = encoder(video, masks=[masks_x])                # [B, nctx, D]
        pred, _ = predictor(ctx, [masks_x], [masks_y], mod="video")  # [B, ntgt, D]
        with torch.no_grad():
            all_t = target_encoder(video)                    # [B, total, D]
        tgt = torch.gather(
            all_t, 1, masks_y.unsqueeze(-1).expand(-1, -1, all_t.shape[-1])
        )
        return pred, tgt


    # ---- instantiate + plumbing check (all demo vars private) ------------------
    encoder, target_encoder, predictor = build_spike_models("cuda")
    _edim = encoder.embed_dim
    print("encoder embed_dim", _edim,
          "| enc params", sum(p.numel() for p in encoder.parameters()) // 1_000_000, "M",
          "| pred params", sum(p.numel() for p in predictor.parameters()) // 1_000_000, "M")

    _P = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
    _T_TOK = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _video = dummy_clip_batch(_B, 3, CONFIG["num_frames"], CONFIG["img_size"], CONFIG["img_size"])
    _mx, _my = temporal_causal_masks(CONFIG["boundary_frame"], _T_TOK, _P,
                                     CONFIG["tubelet_size"], batch_size=_B, device=_video.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _pred, _tgt = wm_forward(_video, _mx, _my, encoder, target_encoder, predictor)
    print("pred", tuple(_pred.shape), "target", tuple(_tgt.shape))
    assert _pred.shape == _tgt.shape, "shape mismatch"
    _loss = (_pred.float() - _tgt.float()).abs().mean()
    assert torch.isfinite(_loss), "non-finite loss"
    print("plumbing loss (L1, no LN):", float(_loss))
    wm_plumbing_ok = True

    return build_spike_models, encoder, predictor, target_encoder, wm_forward


@app.cell
def train(
    CONFIG,
    F,
    dummy_clip_batch,
    encoder,
    predictor,
    target_encoder,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # JEPA loss, EMA, optimizer + smoke train loop on a FIXED dummy clip.
    # Purpose: prove the causal-masked training mechanics (fwd/bwd/step/EMA)
    # actually lower the loss end-to-end before any real data is pulled.

    def jepa_loss(pred, target, reduction="mean"):
        p = F.layer_norm(pred.float(), (pred.shape[-1],))
        t = F.layer_norm(target.float(), (target.shape[-1],))
        return F.smooth_l1_loss(p, t, reduction=reduction)


    @torch.no_grad()
    def update_ema(online, target, momentum=0.996):
        for po, pt in zip(online.parameters(), target.parameters()):
            pt.mul_(momentum).add_(po, alpha=1.0 - momentum)


    def token_cosine(pred, target):
        p = F.layer_norm(pred.float(), (pred.shape[-1],))
        t = F.layer_norm(target.float(), (target.shape[-1],))
        return F.cosine_similarity(p, t, dim=-1).mean().item()


    _trainable = [p for p in list(encoder.parameters()) + list(predictor.parameters())
                  if p.requires_grad]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=CONFIG["lr"], weight_decay=0.05,
    )


    def smoke_step(video, masks_x, masks_y):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, tgt = wm_forward(video, masks_x, masks_y,
                                   encoder, target_encoder, predictor)
            loss = jepa_loss(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_trainable, CONFIG["grad_clip"])
        optimizer.step()
        update_ema(encoder, target_encoder, CONFIG["ema_momentum"])
        return float(loss.detach()), token_cosine(pred.detach(), tgt.detach())


    # ---- smoke run: fixed dummy clip so the loss MUST drop if mechanics work ----
    _P = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
    _T_TOK = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    torch.manual_seed(7)
    _fixed_video = dummy_clip_batch(_B, 3, CONFIG["num_frames"],
                                    CONFIG["img_size"], CONFIG["img_size"])
    _mx, _my = temporal_causal_masks(CONFIG["boundary_frame"], _T_TOK, _P,
                                     CONFIG["tubelet_size"], batch_size=_B,
                                     device=_fixed_video.device)

    smoke_history = []
    for _step in range(12):
        _l, _c = smoke_step(_fixed_video, _mx, _my)
        smoke_history.append((_l, _c))
        if _step in (0, 3, 7, 11):
            print(f"smoke step {_step:2d} | loss {_l:.4f} | token cosine {_c:.4f}")

    _l0 = smoke_history[0][0]
    _lf = smoke_history[-1][0]
    print(f"loss {_l0:.4f} -> {_lf:.4f}  (drop {(_l0 - _lf) / _l0 * 100:.1f}%)")
    assert _lf < _l0, "loss did not decrease on a fixed batch - training broken"
    smoke_train_ok = True

    return jepa_loss, smoke_step, token_cosine, update_ema


@app.cell
def checkpoint(CONFIG, encoder, target_encoder, torch):
    # Download + load the official V-JEPA 2.1 ViT-B/384 checkpoint (ema_encoder).
    # The predictor stays freshly initialised (head dims differ from the ViT-G
    # teacher it was trained against) - same policy as the notebook pipeline.

    _ckpt_path = CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"]
    if not (_ckpt_path.exists() and _ckpt_path.stat().st_size > 500_000_000):
        print("downloading", CONFIG["vjepa_ckpt_url"])
        torch.hub.download_url_to_file(CONFIG["vjepa_ckpt_url"], str(_ckpt_path))
    print("checkpoint size GB", round(_ckpt_path.stat().st_size / 1e9, 2))

    _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=True)
    _sd = _ckpt["ema_encoder"]
    _clean = {k.replace("module.", "").replace("backbone.", ""): v
              for k, v in _sd.items()}

    _miss_e, _unexp_e = encoder.load_state_dict(_clean, strict=False)
    _miss_t, _unexp_t = target_encoder.load_state_dict(_clean, strict=False)
    print("encoder missing", len(_miss_e), "unexpected", len(_unexp_e))
    print("target  missing", len(_miss_t), "unexpected", len(_unexp_t))
    del _ckpt, _sd, _clean

    encoder.train()
    target_encoder.eval()
    for _p in target_encoder.parameters():
        _p.requires_grad_(False)

    pretrained_loaded = True
    print("V-JEPA 2.1 ViT-B/384 loaded into student + target")

    return


@app.cell
def pull(CONFIG, Path, discover_frames):
    # Pull two ANYmal-D smoke missions (hdr_front images + camera timestamps).
    # Downloads are PUBLIC on HF (no token needed). Layout mirrors the notebook:
    #   <data_root>/<mission>/images/hdr_front/<frame_id:06d>.jpeg
    #   <data_root>/<mission>/data/hdr_front/       (zarr with "timestamp")

    def pull_mission_tars(mission, data_root):
        from huggingface_hub import snapshot_download
        import tarfile

        cache = Path(snapshot_download(
            "leggedrobotics/grand_tour_dataset",
            allow_patterns=[f"{mission}/images/hdr_front*", f"{mission}/data/hdr_front*"],
            repo_type="dataset",
        ))
        for tar in cache.rglob("*.tar"):
            rel = tar.relative_to(cache)
            dst_parent = Path(data_root) / rel.parent
            dst_parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar) as tf:
                tf.extractall(path=dst_parent)
            print("extracted", rel)
        return Path(data_root) / mission


    def mission_fps(mission_dir):
        """Approx camera fps from the hdr_front timestamp zarr (best effort)."""
        try:
            import zarr
        except ImportError:
            return None
        z = Path(mission_dir) / "data" / "hdr_front"
        try:
            g = zarr.open_group(str(z), mode="r")
            ts = g["timestamp"][:]
            n = len(ts)
            return n / max(1e-9, float(ts[-1] - ts[0])) if n > 1 else None
        except Exception as e:  # noqa: BLE001
            print("fps measure skipped:", repr(e))
            return None


    _mission_dirs = []
    for _m in CONFIG["train_missions"] + CONFIG["val_missions"]:
        _dir = pull_mission_tars(_m, CONFIG["data_root"])
        _mission_dirs.append(_dir)
        _n = len(discover_frames(_dir))
        _fps = mission_fps(_dir)
        print(_m, "frames:", _n, "| approx fps:", round(_fps, 2) if _fps else "n/a")
    missions_pulled = True

    return (pull_mission_tars,)


@app.cell
def real_run(
    CONFIG,
    Path,
    build_clip_records,
    encoder,
    jepa_loss,
    load_clip,
    np,
    predictor,
    smoke_step,
    target_encoder,
    temporal_causal_masks,
    token_cosine,
    torch,
    wm_forward,
):
    # Phase-1 REAL run: causal temporal masking, fine-tuned from the official
    # V-JEPA 2.1 ViT-B/384 checkpoint.
    #   train: ETH-1 (2024-10-01-11-29-55)    held-out val: ICE-1 (2024-11-18-13-48-19)
    #   context = frames [0, 8),  targets = frames [8, 16)  (tubelet-major, t even)

    _real_img = CONFIG["img_size"]
    _real_n = CONFIG["num_frames"]
    _real_stride = CONFIG["temporal_stride"]


    def preload_pool(mission, max_clips):
        mission_dir = Path(CONFIG["data_root"]) / mission
        records = build_clip_records(mission_dir, _real_n, stride=_real_stride,
                                     step=CONFIG["clip_step"], max_clips=max_clips)
        pool = []
        for _ri, (rec) in enumerate(records):
            pool.append(load_clip(mission_dir, rec, _real_img))  # [C,T,H,W] cpu
            if (_ri + 1) % 50 == 0:
                print(f"  preloaded {mission} {_ri + 1}/{len(records)}")
        return pool


    print("preloading train pool (ETH-1) ...")
    train_pool = preload_pool(CONFIG["train_missions"][0], max_clips=240)
    print("preloading val pool (ICE-1) ...")
    val_pool = preload_pool(CONFIG["val_missions"][0], max_clips=48)
    print(f"train clips {len(train_pool)} | val clips {len(val_pool)}")

    _P = (_real_img // CONFIG["patch_size"]) ** 2
    _T_TOK = _real_n // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _mx, _my = temporal_causal_masks(CONFIG["boundary_frame"], _T_TOK, _P,
                                     CONFIG["tubelet_size"], batch_size=_B, device="cuda")

    _rng = np.random.default_rng(0)


    def _train_batch():
        idx = _rng.integers(0, len(train_pool), size=_B)
        vids = torch.stack([train_pool[i] for i in idx]).to("cuda")
        return vids


    def _eval_metrics(samples=8):
        encoder.eval()
        predictor.eval()
        losses, coses = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                idx = _rng.integers(0, len(val_pool), size=_B)
                vids = torch.stack([val_pool[i] for i in idx]).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, tgt = wm_forward(vids, _mx, _my, encoder, target_encoder, predictor)
                    loss = jepa_loss(pred, tgt)
                losses.append(float(loss))
                coses.append(token_cosine(pred.float(), tgt.float()))
        encoder.train()
        predictor.train()
        return float(np.mean(losses)), float(np.mean(coses))


    real_history = []
    _total = 150
    print("training causal fine-tune ...")
    for _s in range(_total):
        vids = _train_batch()
        _l, _c = smoke_step(vids, _mx, _my)
        if (_s + 1) % 25 == 0:
            _vl, _vc = _eval_metrics()
            real_history.append({"step": _s + 1, "train_loss": _l,
                                 "val_loss": _vl, "val_cosine": _vc})
            print(f"step {_s + 1:4d} | train {_l:.4f} | val loss {_vl:.4f} | "
                  f"val cosine {_vc:.4f}", flush=True)

    _vl0 = real_history[0]["val_loss"] if real_history else float("nan")
    print("FINAL causal run: steps", _total,
          "| first-val", round(_vl0, 4),
          "| last val_loss", round(real_history[-1]["val_loss"], 4),
          "| last val_cosine", round(real_history[-1]["val_cosine"], 4))

    _save = CONFIG["ckpt_dir"] / f"phase1_causal_step{_total}.pt"
    torch.save({"step": _total,
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "history": real_history}, _save)
    print("saved", _save)
    real_run_ok = True

    return train_pool, val_pool


@app.cell
def baselines(
    CONFIG,
    build_spike_models,
    encoder,
    jepa_loss,
    np,
    predictor,
    random_mask,
    temporal_causal_masks,
    token_cosine,
    torch,
    train_pool,
    update_ema,
    val_pool,
    wm_forward,
):
    # Baselines for the causal Phase-1 run (matched budget).
    # 1) random-mask fine-tune: 150 steps, 50% context (same token budget as the
    #    causal run: context == target == 2304 tokens), same data/batches/optimizer.
    # 2) copy-last-context-tubelet predictor (no training).
    # All scored vs ONE frozen official V-JEPA 2.1 teacher (never EMA-updated) on
    # the held-out ICE-1 val clips so numbers are directly comparable.

    def _fresh_pretrained_models(device="cuda"):
        enc, tgt, pred = build_spike_models(device)     # fresh random architectures
        _p = CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"]
        _ck = torch.load(_p, map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        tgt.eval()
        for _pp in tgt.parameters():
            _pp.requires_grad_(False)
        enc.train()
        return enc, tgt, pred


    _P = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
    _T_TOK = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _N_CTX_T = CONFIG["boundary_frame"] // CONFIG["tubelet_size"]
    _MX, _MY = temporal_causal_masks(CONFIG["boundary_frame"], _T_TOK, _P,
                                     CONFIG["tubelet_size"], batch_size=_B, device="cuda")


    def _sample_videos(pool, rng):
        idx = rng.integers(0, len(pool), size=_B)
        return torch.stack([pool[i] for i in idx]).to("cuda")


    def eval_vs_official(enc, pred, official_tgt, samples=8, seed=5):
        """Score predictions (causal masks) against the frozen official teacher."""
        enc.eval(); pred.eval()
        rng = np.random.default_rng(seed)
        losses, coses = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                vids = _sample_videos(val_pool, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr, tg = wm_forward(vids, _MX, _MY, enc, official_tgt, pred)
                losses.append(jepa_loss(pr, tg).item())
                coses.append(token_cosine(pr.float(), tg.float()))
        enc.train(); pred.train()
        return float(np.mean(losses)), float(np.mean(coses))


    # ---- official (frozen) teacher + copy-last baseline -------------------------
    _enc_o, _tgt_o, _pred_o = _fresh_pretrained_models()

    def copy_last_metrics(official_tgt, samples=8, seed=5):
        rng = np.random.default_rng(seed)
        losses, coses = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                vids = _sample_videos(val_pool, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    full = official_tgt(vids)                       # [B,total,D]
                f = full.view(-1, _T_TOK, _P, full.shape[-1])
                last_ctx = f[:, _N_CTX_T - 1:_N_CTX_T]              # last context tubelet
                _nt = _T_TOK - _N_CTX_T
                pred_copy = last_ctx.expand(-1, _nt, _P, -1).reshape(-1, _nt * _P, full.shape[-1])
                tg = torch.gather(full, 1, _MY.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
                losses.append(jepa_loss(pred_copy, tg).item())
                coses.append(token_cosine(pred_copy.float(), tg.float()))
        return float(np.mean(losses)), float(np.mean(coses))


    copy_last = copy_last_metrics(_tgt_o)
    print(f"copy-last (no training)       | val loss {copy_last[0]:.4f} | cos {copy_last[1]:.4f}")

    # ---- causal-trained model (globals from real_run) vs official teacher -------
    _causal = eval_vs_official(encoder, predictor, _tgt_o)
    print(f"causal-150 (trained)          | val loss {_causal[0]:.4f} | cos {_causal[1]:.4f}")

    # ---- random-mask baseline training (fresh pretrained init, EMA target) ------
    _enc_r, _tgt_r, _pred_r = _fresh_pretrained_models()
    _opt_r = torch.optim.AdamW(list(_enc_r.parameters()) + list(_pred_r.parameters()),
                               lr=CONFIG["lr"], weight_decay=0.05)
    _trainable_r = [p for p in list(_enc_r.parameters()) + list(_pred_r.parameters())
                    if p.requires_grad]
    _rng_r = np.random.default_rng(0)

    random_history = []
    for _s in range(150):
        _mxr, _myr = random_mask(0, _T_TOK, _P, tubelet_size=CONFIG["tubelet_size"],
                                 mask_ratio=0.5, seed=1000 + _s, batch_size=_B,
                                 device="cuda")
        _vids = _sample_videos(train_pool, _rng_r)
        _opt_r.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _pr, _tg = wm_forward(_vids, _mxr, _myr, _enc_r, _tgt_r, _pred_r)
            _loss = jepa_loss(_pr, _tg)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_r, CONFIG["grad_clip"])
        _opt_r.step()
        update_ema(_enc_r, _tgt_r, CONFIG["ema_momentum"])
        if (_s + 1) % 25 == 0:
            _vl, _vc = eval_vs_official(_enc_r, _pred_r, _tgt_r, seed=7)   # its own teacher space
            random_history.append({"step": _s + 1, "train_loss": float(_loss.detach()),
                                   "val_loss_own": _vl, "val_cosine_own": _vc})
            print(f"random step {_s + 1:4d} | train {float(_loss.detach()):.4f} | "
                  f"own-te
acher val cos {_vc:.4f}", flush=True)

    _random = eval_vs_official(_enc_r, _pred_r, _tgt_o)
    print(f"random-150 (trained)          | val loss {_random[0]:.4f} | cos {_random[1]:.4f}")

    print("--- held-out ICE-1, causal-mask query, official frozen teacher ---")
    print(f"{'model':24s} {'val LN-L1':>9s} {'cosine':>8s}")
    print(f"{'copy-last-context':24s} {copy_last[0]:9.4f} {copy_last[1]:8.4f}")
    print(f"{'causal-150':24s} {_causal[0]:9.4f} {_causal[1]:8.4f}")
    print(f"{'random-150':24s} {_random[0]:9.4f} {_random[1]:8.4f}")

    baseline_results = {"copy_last": copy_last, "causal": _causal,
                        "random": _random, "random_history": random_history}
    baselines_ok = True

    return


@app.cell
def horizon(
    CONFIG,
    Path,
    build_clip_records,
    build_spike_models,
    jepa_loss,
    load_clip,
    np,
    random_mask,
    temporal_causal_masks,
    token_cosine,
    torch,
    update_ema,
    wm_forward,
):
    # Horizon-fix spike: stride-1 run was degenerate (copy-last won). Measure fps,
    # retrain with a real temporal stride, re-run copy-last / causal / random vs the
    # frozen official teacher on held-out ICE-1.

    def _ensure_zarr():
        try:
            import zarr
            return True
        except ImportError:
            import subprocess, sys
            print("installing zarr ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=True)
            try:
                import zarr
                return True
            except ImportError:
                return False


    def _mission_fps(mission_dir):
        try:
            import zarr
            g = zarr.open_group(str(Path(mission_dir) / "data" / "hdr_front"), mode="r")
            ts = g["timestamp"][:]
            n = len(ts)
            return (n / max(1e-9, float(ts[-1] - ts[0])), n) if n > 1 else (None, n)
        except Exception as e:
            print("fps read failed:", repr(e))
            return None, None


    _zarr_ok = _ensure_zarr()
    _fps_train, _n_train = _mission_fps(Path(CONFIG["data_root"]) / CONFIG["train_missions"][0])
    _fps_val, _n_val = _mission_fps(Path(CONFIG["data_root"]) / CONFIG["val_missions"][0])
    print(f"zarr ok {_zarr_ok} | ETH-1 fps {_fps_train} ({_n_train} ts) | ICE-1 fps {_fps_val} ({_n_val} ts)")

    _fps = _fps_train or _fps_val or 20.0
    STRIDE = max(2, int(round(0.15 * _fps)))
    print(f"stride {STRIDE} | gap {STRIDE/_fps*1000:.0f} ms | context ~{7*STRIDE/_fps:.2f} s | "
          f"last-ctx->last-target ~{8*STRIDE/_fps:.2f} s")


    def _preload(mission, max_clips):
        mdir = Path(CONFIG["data_root"]) / mission
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=STRIDE,
                                  step=CONFIG["num_frames"], max_clips=max_clips)
        pool = [load_clip(mdir, r, CONFIG["img_size"]) for r in recs]
        print(f"  preloaded {mission}: {len(pool)} clips (stride {STRIDE})")
        return pool


    print("preloading stride pools ...")
    train_pool_h = _preload(CONFIG["train_missions"][0], max_clips=200)
    val_pool_h = _preload(CONFIG["val_missions"][0], max_clips=40)

    _P = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
    _T_TOK = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _N_CTX_T = CONFIG["boundary_frame"] // CONFIG["tubelet_size"]
    _MX, _MY = temporal_causal_masks(CONFIG["boundary_frame"], _T_TOK, _P,
                                     CONFIG["tubelet_size"], batch_size=_B, device="cuda")


    def _fresh_pretrained_models(device="cuda"):
        enc, tgt, pred = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        tgt.eval()
        for _pp in tgt.parameters():
            _pp.requires_grad_(False)
        enc.train()
        return enc, tgt, pred


    def _sample(pool, rng):
        idx = rng.integers(0, len(pool), size=_B)
        return torch.stack([pool[i] for i in idx]).to("cuda")


    def _official_score(enc, pred, official_tgt, pool, samples=8, seed=5):
        enc.eval(); pred.eval()
        rng = np.random.default_rng(seed)
        losses, coses = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                vids = _sample(pool, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr, tg = wm_forward(vids, _MX, _MY, enc, official_tgt, pred)
                losses.append(jepa_loss(pr, tg).item())
                coses.append(token_cosine(pr.float(), tg.float()))
        enc.train(); pred.train()
        return float(np.mean(losses)), float(np.mean(coses))


    def _copy_last(official_tgt, pool, samples=8, seed=5):
        rng = np.random.default_rng(seed)
        losses, coses = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                vids = _sample(pool, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    full = official_tgt(vids)
                f = full.view(-1, _T_TOK, _P, full.shape[-1])
                last_ctx = f[:, _N_CTX_T - 1:_N_CTX_T]
                _nt = _T_TOK - _N_CTX_T
                pred_copy = last_ctx.expand(-1, _nt, _P, -1).reshape(-1, _nt * _P, full.shape[-1])
                tg = torch.gather(full, 1, _MY.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
                losses.append(jepa_loss(pred_copy, tg).item())
                coses.append(token_cosine(pred_copy.float(), tg.float()))
        return float(np.mean(losses)), float(np.mean(coses))


    def _train_masked(models, mode, pool, steps=150, seed_rng=0):
        enc, tgt, pred = models
        opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                                lr=CONFIG["lr"], weight_decay=0.05)
        tr = [p for p in list(enc.parameters()) + list(pred.parameters()) if p.requires_grad]
        rng = np.random.default_rng(seed_rng)
        for _s in range(steps):
            if mode == "causal":
                mx, my = _MX, _MY
            else:
                mx, my = random_mask(0, _T_TOK, _P, tubelet_size=CONFIG["tubelet_size"],
                                     mask_ratio=0.5, seed=1000 + _s, batch_size=_B, device="cuda")
            vids = _sample(pool, rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr, tg = wm_forward(vids, mx, my, enc, tgt, pred)
                loss = jepa_loss(pr, tg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tr, CONFIG["grad_clip"])
            opt.step()
            update_ema(enc, tgt, CONFIG["ema_momentum"])
        return enc, tgt, pred


    _enc_o, _tgt_o, _pred_o = _fresh_pretrained_models()
    _copy_last = _copy_last(_tgt_o, val_pool_h)
    print(f"copy-last (stride {STRIDE}, no training) | loss {_copy_last[0]:.4f} | cos {_copy_last[1]:.4f}")

    print("training causal (stride) ...")
    _enc_c, _tgt_c, _pred_c = _fresh_pretrained_models()
    _enc_c, _tgt_c, _pred_c = _train_masked((_enc_c, _tgt_c, _pred_c), "causal", train_pool_h, 150)
    _causal_h = _official_score(_enc_c, _pred_c, _tgt_o, val_pool_h)
    print(f"causal-150 (stride) | loss {_causal_h[0]:.4f} | cos {_causal_h[1]:.4f}")

    print("training random (stride) ...")
    _enc_r, _tgt_r, _pred_r = _fresh_pretrained_models()
    _enc_r, _tgt_r, _pred_r = _train_masked((_enc_r, _tgt_r, _pred_r), "random", train_pool_h, 150)
    _random_h = _official_score(_enc_r, _pred_r, _tgt_o, val_pool_h)
    print(f"random-150 (stride) | loss {_random_h[0]:.4f} | cos {_random_h[1]:.4f}")

    print("--- held-out ICE-1 (official frozen teacher, causal query) ---")
    print(f"{'model':22s} stride={str(STRIDE):<2s} {'val LN-L1':>9s} {'cosine':>8s}")
    print(f"{'copy-last-context':22s} {STRIDE:<6d} {_copy_last[0]:9.4f} {_copy_last[1]:8.4f}")
    print(f"{'causal-150':22s} {STRIDE:<6d} {_causal_h[0]:9.4f} {_causal_h[1]:8.4f}")
    print(f"{'random-150':22s} {STRIDE:<6d} {_random_h[0]:9.4f} {_random_h[1]:8.4f}")

    horizon_results = {"stride": STRIDE, "fps": _fps,
                       "copy_last": _copy_last, "causal": _causal_h, "random": _random_h}
    _save = CONFIG["ckpt_dir"] / f"phase1_stride{STRIDE}_causal150.pt"
    torch.save({"stride": STRIDE, "encoder": _enc_c.state_dict(),
                "predictor": _pred_c.state_dict()}, _save)
    print("saved", _save)
    horizon_ok = True

    return train_pool_h, val_pool_h


app._unparsable_cell(
    r"""
    # Horizon run #2: longer horizon + more steps. fps=10 -> gap 0.4 s (stride 4),
    # context ~2.8 s, last-context -> last-target ~3.2 s. 250 steps.

    _GAP_S = 0.40
    _STEPS = 250


    def _preload(mission, max_clips):
        mdir = Path(CONFIG["data_root"]) / mission
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=STRIDE2,
                                  step=CONFIG["num_frames"], max_clips=max_clips)
        pool = [load_clip(mdir, r, CONFIG["img_size"]) for r in recs]
        print(f"  preloaded {mission}: {len(pool)} clips (stride {STRIDE2})")
        return pool


    STRIDE2 = max(2, int(round(_GAP_S * 10.0)))
    print(f"stride {STRIDE2} | gap {STRIDE2/10.0*1000:.0f} ms | ctx ~{7*STRIDE2/10:.2f} s | "
          f"last-ctx->last-target ~{8*STRIDE2/10:.2f} s")

    print("preloading stride-4 pools ...")
    train_pool_h2 = _preload(CONFIG["train_missions"][0], max_clips=200)
    val_pool_h2 = _preload(CONFIG["val_missions"][0], max_clips=40)

    _P2 = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
    _T2 = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B2 = CONFIG["batch_size"]
    _NCTX2 = CONFIG["boundary_frame"] // CONFIG["tubelet_size"]
    _MX2, _MY2 = temporal_causal_masks(CONFIG["boundary_frame"], _T2, _P2,
                                       CONFIG["tubelet_size"], batch_size=_B2, device="cuda")


    def _fresh2(device="cuda"):
        enc, tgt, pred = build_spike_moim goindels(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        tgt.eval()
        for _p in tgt.parameters():
            _p.requires_grad_(False)
        enc.train()
        return enc, tgt, pred


    def _samp(pool, rng):
        return torch.stack([pool[i] for i in rng.integers(0, len(pool), size=_B2)]).to("cuda")


    def _score(enc, pred, otgt, pool, samples=8, seed=5):
        enc.eval(); pred.eval()
        rng = np.random.default_rng(seed)
        ls, cs = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B2)):
                vids = _samp(pool, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr, tg = wm_forward(vids, _MX2, _MY2, enc, otgt, pred)
                ls.append(jepa_loss(pr, tg).item())
                cs.append(token_cosine(pr.float(), tg.float()))
        enc.train(); pred.train()
        return float(np.mean(ls)), float(np.mean(cs))


    def _copy2(otgt, pool, samples=8, seed=5):
        rng = np.random.default_rng(seed)
        ls, cs = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B2)):
                vids = _samp(pool, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    full = otgt(vids)
                f = full.view(-1, _T2, _P2, full.shape[-1])
                lc = f[:, _NCTX2 - 1:_NCTX2]
                _nt = _T2 - _NCTX2
                pc = lc.expand(-1, _nt, _P2, -1).reshape(-1, _nt * _P2, full.shape[-1])
                tg = torch.gather(full, 1, _MY2.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
                ls.append(jepa_loss(pc, tg).item())
                cs.append(token_cosine(pc.float(), tg.float()))
        return float(np.mean(ls)), float(np.mean(cs))


    def _train2(models, mode, pool, steps=_STEPS, seed_rng=0):
        enc, tgt, pred = models
        opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                                lr=CONFIG["lr"], weight_decay=0.05)
        tr = [p for p in list(enc.parameters()) + list(pred.parameters()) if p.requires_grad]
        rng = np.random.default_rng(seed_rng)
        for _s in range(steps):
            if mode == "causal":
                mx, my = _MX2, _MY2
            else:
                mx, my = random_mask(0, _T2, _P2, tubelet_size=CONFIG["tubelet_size"],
                                     mask_ratio=0.5, seed=2000 + _s, batch_size=_B2, device="cuda")
            vids = _samp(train_pool_h2 if pool is None else pool, rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr, tg = wm_forward(vids, mx, my, enc, tgt, pred)
                loss = jepa_loss(pr, tg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tr, CONFIG["grad_clip"])
            opt.step()
            update_ema(enc, tgt, CONFIG["ema_momentum"])
        return enc, tgt, pred


    _oe, _ot, _op = _fresh2()
    _copy_h2 = _copy2(_ot, val_pool_h2)
    print(f"copy-last (stride {STRIDE2}) | loss {_copy_h2[0]:.4f} | cos {_copy_h2[1]:.4f}")

    print(f"training causal (stride {STRIDE2}, {_STEPS} steps) ...")
    _ce, _ct, _cp = _fresh2()
    _ce, _ct, _cp = _train2((_ce, _ct, _cp), "causal", None)
    _causal_h2 = _score(_ce, _cp, _ot, val_pool_h2)
    print(f"causal-{_STEPS} | loss {_causal_h2[0]:.4f} | cos {_causal_h2[1]:.4f}")

    print(f"training random (stride {STRIDE2}, {_STEPS} steps) ...")
    _re, _rt, _rp = _fresh2()
    _re, _rt, _rp = _train2((_re, _rt, _rp), "random", None)
    _random_h2 = _score(_re, _rp, _ot, val_pool_h2)
    print(f"random-{_STEPS} | loss {_random_h2[0]:.4f} | cos {_random_h2[1]:.4f}")

    print("--- held-out ICE-1, official frozen teacher, causal query (run #2) ---")
    print(f"{'model':22s} stride={str(STRIDE2):<2s} {'val LN-L1':>9s} {'cosine':>8s}")
    print(f"{'copy-last-context':22s} {STRIDE2:<6d} {_copy_h2[0]:9.4f} {_copy_h2[1]:8.4f}")
    print(f"{'causal-'+str(_STEPS):22s} {STRIDE2:<6d} {_causal_h2[0]:9.4f} {_causal_h2[1]:8.4f}")
    print(f"{'random-'+str(_STEPS):22s} {STRIDE2:<6d} {_random_h2[0]:9.4f} {_random_h2[1]:8.4f}")

    horizon2_results = {"stride": STRIDE2, "copy_last": _copy_h2,
                        "causal": _causal_h2, "random": _random_h2}
    torch.save({"stride": STRIDE2, "encoder": _ce.state_dict(),
                "predictor": _cp.state_dict()},
               CONFIG["ckpt_dir"] / f"phase1_stride{STRIDE2}_causal{_STEPS}.pt")
    print("saved ckpt stride", STRIDE2)
    horizon2_ok = True

    """,
    name="horizon2"
)


@app.cell
def frozen(
    CONFIG,
    build_spike_models,
    jepa_loss,
    np,
    random_mask,
    temporal_causal_masks,
    token_cosine,
    torch,
    train_pool_h,
    train_pool_h2,
    val_pool_h,
    val_pool_h2,
    wm_forward,
):
    # Frozen-encoder (predictor-only) causal fine-tune. Drift hypothesis test:
    # fine-tuned students lose to copy-last partly because the student+EMA drift
    # away from the official feature space. Freezing the encoder keeps prediction
    # in that space by construction. Runs at stride 2 and stride 4, scored vs the
    # frozen official teacher on held-out ICE-1.

    _STEPS = 250


    def _frozen_pair(device="cuda"):
        """Frozen official encoder (student-encoder == teacher) + fresh predictor."""
        enc, tgt, pred = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _m in (enc, tgt):
            for _p in _m.parameters():
                _p.requires_grad_(False)
        pred.train()
        return enc, tgt, pred


    def _samp(pool, rng, b):
        return torch.stack([pool[i] for i in rng.integers(0, len(pool), size=b)]).to("cuda")


    def _score(enc, pred, tgt, pool, mx, my, samples=8, seed=5):
        enc.eval(); pred.eval()
        rng = np.random.default_rng(seed)
        ls, cs = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                vids = _samp(pool, rng, _B)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr, tg = wm_forward(vids, mx, my, enc, tgt, pred)
                ls.append(jepa_loss(pr, tg).item())
                cs.append(token_cosine(pr.float(), tg.float()))
        pred.train()
        return float(np.mean(ls)), float(np.mean(cs))


    def _copy(enc, pool, mx, my, samples=8, seed=5):
        rng = np.random.default_rng(seed)
        ls, cs = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                vids = _samp(pool, rng, _B)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    full = enc(vids)
                f = full.view(-1, _T, _P, full.shape[-1])
                lc = f[:, _NCTX - 1:_NCTX]
                _nt = _T - _NCTX
                pc = lc.expand(-1, _nt, _P, -1).reshape(-1, _nt * _P, full.shape[-1])
                tg = torch.gather(full, 1, my.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
                ls.append(jepa_loss(pc, tg).item())
                cs.append(token_cosine(pc.float(), tg.float()))
        return float(np.mean(ls)), float(np.mean(cs))


    def _train_frozen(enc, pred, tgt, pool, mode, mx, my, steps=_STEPS, seed_rng=0):
        opt = torch.optim.AdamW(pred.parameters(), lr=CONFIG["lr"], weight_decay=0.05)
        tr = [p for p in pred.parameters() if p.requires_grad]
        rng = np.random.default_rng(seed_rng)
        for _s in range(steps):
            if mode == "causal":
                mx_s, my_s = mx, my
            else:
                mx_s, my_s = random_mask(0, _T, _P, tubelet_size=CONFIG["tubelet_size"],
                                         mask_ratio=0.5, seed=3000 + _s, batch_size=_B,
                                         device="cuda")
            vids = _samp(pool, rng, _B)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr, tg = wm_forward(vids, mx_s, my_s, enc, tgt, pred)
                loss = jepa_loss(pr, tg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tr, CONFIG["grad_clip"])
            opt.step()
        return pred


    _P = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _NCTX = CONFIG["boundary_frame"] // CONFIG["tubelet_size"]
    _MX, _MY = temporal_causal_masks(CONFIG["boundary_frame"], _T, _P,
                                     CONFIG["tubelet_size"], batch_size=_B, device="cuda")


    def _run_frozen(pool_train, pool_val, tag):
        enc, tgt, pred = _frozen_pair()
        _copy_now = _copy(enc, pool_val, _MX, _MY)
        print(f"[{tag}] copy-last      | loss {_copy_now[0]:.4f} | cos {_copy_now[1]:.4f}")

        _ce, _ct, _cp = _frozen_pair()
        _cp = _train_frozen(_ce, _cp, _ct, pool_train, "causal", _MX, _MY)
        _causal = _score(_ce, _cp, _ct, pool_val, _MX, _MY)
        print(f"[{tag}] causal-frozen-{_STEPS} | loss {_causal[0]:.4f} | cos {_causal[1]:.4f}")

        _re, _rt, _rp = _frozen_pair()
        _rp = _train_frozen(_re, _rp, _rt, pool_train, "random", _MX, _MY)
        _random = _score(_re, _rp, _rt, pool_val, _MX, _MY)
        print(f"[{tag}] random-frozen-{_STEPS} | loss {_random[0]:.4f} | cos {_random[1]:.4f}")
        return {"copy_last": _copy_now, "causal": _causal, "random": _random, "pred": pred}


    _frozen_s2 = _run_frozen(train_pool_h, val_pool_h, "stride2")
    _frozen_s4 = _run_frozen(train_pool_h2, val_pool_h2, "stride4")

    print("--- FROZEN-ENCODER summary (official teacher, causal query, ICE-1) ---")
    print(f"{'model':24s} stride2 cos   stride4 cos")
    for _name, _k in (("copy-last", "copy_last"), ("causal-frozen", "causal"),
                      ("random-frozen", "random")):
        print(f"{_name:24s} {_frozen_s2[_k][1]:9.4f}   {_frozen_s4[_k][1]:9.4f}")
    print("(finetuned causal, earlier runs: stride2 .8222 | stride4 .7791)")

    frozen_results = {"stride2": _frozen_s2, "stride4": _frozen_s4}
    torch.save({"pred": _frozen_s4["pred"].state_dict(), "results": frozen_results},
               CONFIG["ckpt_dir"] / "phase1_frozen_stride4_pred.pt")
    print("saved ckpt phase1_frozen_stride4_pred.pt")
    frozen_ok = True

    return


@app.cell
def data2(CONFIG, discover_frames, pull_mission_tars):
    # Pull 4 more missions for the Phase-2-lite corpus (public downloads).
    # train envs: ETH-2 (indoor), ETH-3 (outdoor), SPX-1 (Jungfraujoch Sphinx)
    # held-out:   SPX-2 (Jungfraujoch Sphinx 2 - test split)

    _P2_MISSIONS = {
        "train": ["2024-10-01-11-47-44", "2024-10-01-12-00-49", "2024-11-02-17-10-25"],
        "test":  ["2024-11-02-17-18-32"],
    }

    phase2_mission_counts = {}
    for _split, _ms in _P2_MISSIONS.items():
        for _m in _ms:
            _dir = pull_mission_tars(_m, CONFIG["data_root"])
            _n = len(discover_frames(_dir))
            phase2_mission_counts[_m] = {"split": _split, "frames": _n}
            print(f"{_split:5s} {_m} frames: {_n}")

    phase2_pull_ok = True

    return


@app.cell
def phase2(
    CONFIG,
    Path,
    build_clip_records,
    build_spike_models,
    jepa_loss,
    load_clip,
    np,
    temporal_causal_masks,
    token_cosine,
    torch,
    wm_forward,
):
    # Phase-2-lite: multi-environment causal pretraining (predictor-only, frozen
    # official encoder - the recipe that beat copy-last in Phase 1).
    #   train: ETH-1/2/3 + SPX-1   held-out: ICE-1 + SPX-2
    #   stride 4 (~400 ms gaps), boundary randomized per step over {6,8,10}.
    #   teacher = frozen official V-JEPA 2.1 (no EMA drift by construction).

    _P2_TRAIN = ["2024-10-01-11-29-55", "2024-10-01-11-47-44",
                 "2024-10-01-12-00-49", "2024-11-02-17-10-25"]
    _P2_VAL = ["2024-11-18-13-48-19", "2024-11-02-17-18-32"]   # ICE-1, SPX-2
    _STRIDE = 4
    _STEPS = 1000
    _BOUNDARIES = (6, 8, 10)


    def _preload2(mission, max_clips):
        mdir = Path(CONFIG["data_root"]) / mission
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=_STRIDE,
                                  step=CONFIG["num_frames"], max_clips=max_clips)
        return [load_clip(mdir, r, CONFIG["img_size"]) for r in recs]


    print("preloading phase-2 pools ...")
    _p2_train_pool = []
    for _m in _P2_TRAIN:
        _pool = _preload2(_m, max_clips=110)
        _p2_train_pool += _pool
        print(f"  {_m}: {len(_pool)} clips")
    _p2_val_pools = {}
    for _m in _P2_VAL:
        _p2_val_pools[_m] = _preload2(_m, max_clips=32)
        print(f"  val {_m}: {len(_p2_val_pools[_m])} clips")

    _P = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _NCTX = 8 // CONFIG["tubelet_size"]
    _MX8, _MY8 = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                       batch_size=_B, device="cuda")


    def _samp(pool, rng):
        return torch.stack([pool[i] for i in rng.integers(0, len(pool), size=_B)]).to("cuda")


    def _score(enc, pred, pool, mx, my, samples=8, seed=11):
        enc.eval(); pred.eval()
        rng = np.random.default_rng(seed)
        ls, cs = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                vids = _samp(pool, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr, tg = wm_forward(vids, mx, my, enc, enc, pred)
                ls.append(jepa_loss(pr, tg).item())
                cs.append(token_cosine(pr.float(), tg.float()))
        enc.eval(); pred.train()
        return float(np.mean(ls)), float(np.mean(cs))


    def _copy_score(enc, pool, mx, my, samples=8, seed=13):
        rng = np.random.default_rng(seed)
        ls, cs = [], []
        with torch.no_grad():
            for _ in range(max(1, samples // _B)):
                vids = _samp(pool, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    full = enc(vids)
                f = full.view(-1, _T, _P, full.shape[-1])
                lc = f[:, _NCTX - 1:_NCTX]
                _nt = _T - _NCTX
                pc = lc.expand(-1, _nt, _P, -1).reshape(-1, _nt * _P, full.shape[-1])
                tg = torch.gather(full, 1, my.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
                ls.append(jepa_loss(pc, tg).item())
                cs.append(token_cosine(pc.float(), tg.float()))
        return float(np.mean(ls)), float(np.mean(cs))


    # frozen official encoder (student == teacher) + fresh trainable predictor
    _enc, _tgt, _pred = build_spike_models("cuda")
    _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                     map_location="cpu", weights_only=True)
    _sd = {k.replace("module.", "").replace("backbone.", ""): v
           for k, v in _ck["ema_encoder"].items()}
    _enc.load_state_dict(_sd, strict=False)
    _tgt.load_state_dict(_sd, strict=False)
    _enc.eval(); _tgt.eval()
    for _m in (_enc, _tgt):
        for _p in _m.parameters():
            _p.requires_grad_(False)
    _pred.train()

    _opt = torch.optim.AdamW(_pred.parameters(), lr=CONFIG["lr"], weight_decay=0.05)
    _tr = [p for p in _pred.parameters() if p.requires_grad]
    _rng = np.random.default_rng(0)

    print("copy-last references (official space):")
    _copy_ref = {}
    for _vm in _P2_VAL:
        _c = _copy_score(_enc, _p2_val_pools[_vm], _MX8, _MY8)
        _copy_ref[_vm] = _c
        print(f"  {_vm}: loss {_c[0]:.4f} | cos {_c[1]:.4f}")

    phase2_history = []
    print(f"training causal predictor (frozen enc, stride {_STRIDE}, {_STEPS} steps) ...")
    for _s in range(_STEPS):
        _b = _BOUNDARIES[int(_rng.integers(0, len(_BOUNDARIES)))]
        _mx, _my = temporal_causal_masks(_b, _T, _P, CONFIG["tubelet_size"],
                                         batch_size=_B, device="cuda")
        _vids = _samp(_p2_train_pool, _rng)
        _opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _pr, _tg = wm_forward(_vids, _mx, _my, _enc, _tgt, _pred)
            _loss = jepa_loss(_pr, _tg)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(_tr, CONFIG["grad_clip"])
        _opt.step()
        if (_s + 1) % 100 == 0:
            _row = {"step": _s + 1, "train_loss": float(_loss.detach())}
            for _vm in _P2_VAL:
                _vl, _vc = _score(_enc, _pred, _p2_val_pools[_vm], _MX8, _MY8)
                _row[_vm + "_loss"], _row[_vm + "_cos"] = _vl, _vc
            phase2_history.append(_row)
            print(f"step {_s + 1:5d} | train {_row['train_loss']:.4f} | "
                  + " | ".join(f"{_vm[:4]} cos {_row[_vm+'_cos']:.4f}" for _vm in _P2_VAL),
                  flush=True)
        if (_s + 1) % 400 == 0:
            torch.save({"step": _s + 1, "pred": _pred.state_dict()},
                       CONFIG["ckpt_dir"] / f"phase2_stride{_STRIDE}_pred{_s+1}.pt")

    torch.save({"step": _STEPS, "pred": _pred.state_dict()},
               CONFIG["ckpt_dir"] / f"phase2_stride{_STRIDE}_pred{_STEPS}.pt")

    print("--- PHASE-2-lite summary (official frozen teacher, causal query b=8) ---")
    print(f"{'model':26s} {'ICE-1 cos':>10s} {'SPX-2 cos':>10s}")
    _ice = _score(_enc, _pred, _p2_val_pools[_P2_VAL[0]], _MX8, _MY8, samples=16, seed=21)
    _spx = _score(_enc, _pred, _p2_val_pools[_P2_VAL[1]], _MX8, _MY8, samples=16, seed=22)
    print(f"{'copy-last':26s} {_copy_ref[_P2_VAL[0]][1]:>10.4f} {_copy_ref[_P2_VAL[1]][1]:>10.4f}")
    print(f"{'causal-frozen-'+str(_STEPS):26s} {_ice[1]:>10.4f} {_spx[1]:>10.4f}")

    phase2_results = {"history": phase2_history, "copy_ref": _copy_ref,
                      "final": {"ICE-1": _ice, "SPX-2": _spx},
                      "stride": _STRIDE, "steps": _STEPS}
    phase2_ok = True

    return


@app.cell
def decode(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    temporal_causal_masks,
    torch,
    train_pool_h2,
    val_pool_h2,
    wm_forward,
):
    # Phase-2b: latent -> pixels decoder spike (self-contained).
    # Conditioning sources compared on held-out clips:
    #   teacher : official-encoder latent of the target tubelet (decoder ceiling)
    #   causal  : phase2 causal predictor output (loaded from ckpt) for that tubelet
    #   copy    : last-context tubelet latent (static-scene baseline)
    # GT = second frame of the tubelet window (frame 9 for tubelet 4 at boundary 8).

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _TUB = 4
    _GTF = 9
    _MX8, _MY8 = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                       batch_size=_B, device="cuda")


    def _charbonnier(a, b, eps=1e-3):
        return torch.s
qrt((a - b).pow(2) + eps ** 2).mean()


    def _denorm(fr):
        mean = torch.tensor([0.485, 0.456, 0.406], device=fr.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=fr.device).view(3, 1, 1)
        return (fr * std + mean).clamp(0, 1)


    def _psnr(a, b):
        mse = torch.mean((a - b) ** 2).item()
        return float("inf") if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


    def _ssim1(a, b):  # returns 1-SSIM (0 = identical)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu1 = F.avg_pool2d(a, 11, stride=1, padding=5)
        mu2 = F.avg_pool2d(b, 11, stride=1, padding=5)
        s1 = F.avg_pool2d(a * a, 11, 1, 5) - mu1 * mu1
        s2 = F.avg_pool2d(b * b, 11, 1, 5) - mu2 * mu2
        s12 = F.avg_pool2d(a * b, 11, 1, 5) - mu1 * mu2
        num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
        den = (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)
        return (1.0 - num.clamp(0, 1).mean()).item()


    class ConvFrameDecoder(nn.Module):
        """24x24x768 latent grid -> 384x384 RGB frame."""

        def __init__(self, dim=768):
            super().__init__()
            self.in_proj = nn.Sequential(nn.Conv2d(dim, 512, 1), nn.GroupNorm(32, 512), nn.GELU())
            self.up1 = self._up(512, 256)
            self.up2 = self._up(256, 128)
            self.up3 = self._up(128, 64)
            self.up4 = self._up(64, 32)
            self.refine = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 3, 3, padding=1),
            )

        @staticmethod
        def _up(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(32, cout), nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.GELU(),
            )

        def forward(self, x):  # [B, 768, 24, 24]
            x = self.in_proj(x)
            x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
            return torch.sigmoid(self.refine(x))


    def _tok_grid(tokens):
        return tokens.permute(0, 2, 1).view(-1, 768, _GRID, _GRID).contiguous()


    def _teacher_tokens(enc, vids, t_idx):
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                full = enc(vids)
        return full.view(vids.shape[0], -1, _P, full.shape[-1])[:, t_idx].float()


    def _gt_frame(vids, t):
        return _denorm(vids[:, :, t].contiguous())


    def _load_frozen_enc():
        enc, tgt, _ = build_spike_models("cuda")
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _m in (enc, tgt):
            for _p in _m.parameters():
                _p.requires_grad_(False)
        return enc, tgt


    _enc, _tgt = _load_frozen_enc()
    _, _, _pred = build_spike_models("cuda")          # arch only
    _ckp = torch.load(CONFIG["ckpt_dir"] / "phase2_stride4_pred1000.pt",
                      map_location="cuda", weights_only=True)
    _pred.load_state_dict(_ckp["pred"])
    _pred.eval()
    print("loaded frozen official encoder + phase2 causal predictor (step", _ckp.get("step"), ")")

    # training pool: reuse public stride-4 ETH-1 pool if present, else preload
    if "train_pool_h2" in dir() and len(train_pool_h2) >= 100:
        _train_pool = train_pool_h2
        print("decoder train pool: existing stride-4 ETH-1 pool", len(_train_pool))
    else:
        _mdir = Path(CONFIG["data_root"]) / CONFIG["train_missions"][0]
        _recs = build_clip_records(_mdir, CONFIG["num_frames"], stride=4,
                                   step=CONFIG["num_frames"], max_clips=120)
        _train_pool = [load_clip(_mdir, r, CONFIG["img_size"]) for r in _recs]
        print("decoder train pool: preloaded", len(_train_pool))

    _val_pools = {}
    if "val_pool_h2" in dir() and len(val_pool_h2) >= 24:
        _val_pools["2024-11-18-13-48-19"] = val_pool_h2
    else:
        _mdir = Path(CONFIG["data_root"]) / "2024-11-18-13-48-19"
        _recs = build_clip_records(_mdir, CONFIG["num_frames"], stride=4,
                                   step=CONFIG["num_frames"], max_clips=24)
        _val_pools["2024-11-18-13-48-19"] = [load_clip(_mdir, r, CONFIG["img_size"]) for r in _recs]
    _mdir2 = Path(CONFIG["data_root"]) / "2024-11-02-17-18-32"
    _recs2 = build_clip_records(_mdir2, CONFIG["num_frames"], stride=4,
                                step=CONFIG["num_frames"], max_clips=24)
    _val_pools["2024-11-02-17-18-32"] = [load_clip(_mdir2, r, CONFIG["img_size"]) for r in _recs2]
    print("val pools:", {k: len(v) for k, v in _val_pools.items()})

    decoder = ConvFrameDecoder(768).to("cuda")
    _dec_ckpt = CONFIG["ckpt_dir"] / "phase2b_teacher_decoder.pt"
    if _dec_ckpt.exists():
        decoder.load_state_dict(torch.load(_dec_ckpt, map_location="cuda",
                                           weights_only=True)["decoder"])
        print("loaded existing decoder ckpt (skipping training)")
    else:
        _opt_d = torch.optim.AdamW(decoder.parameters(), lr=2e-4, weight_decay=0.01)
        _rng_d = np.random.default_rng(0)
        print(f"training teacher-latent decoder (tubelet {_TUB} -> frame {_GTF}) ...")
        for _s in range(600):
            _idx = _rng_d.integers(0, len(_train_pool), size=_B)
            _vids = torch.stack([_train_pool[i] for i in _idx]).to("cuda")
            _tok = _teacher_tokens(_enc, _vids, _TUB)
            _gt = _gt_frame(_vids, _GTF)
            _opt_d.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _rec = decoder(_tok_grid(_tok).float())
                _loss = _charbonnier(_rec.float(), _gt.float())
            _loss.backward()
            _opt_d.step()
            if (_s + 1) % 150 == 0:
                print(f"decode step {_s + 1} | loss {float(_loss.detach()):.4f}", flush=True)

    torch.save({"decoder": decoder.state_dict()},
               CONFIG["ckpt_dir"] / "phase2b_teacher_decoder.pt")


    def _decode_metrics(pool, tag, n=8, seed=31):
        rng = np.random.default_rng(seed)
        agg = {"teacher": ([], []), "causal": ([], []), "copy": ([], [])}
        with torch.no_grad():
            for _ in range(max(1, n // _B)):
                _vids = torch.stack([pool[i] for i in rng.integers(0, len(pool), size=_B)]).to("cuda")
                _gt = _gt_frame(_vids, _GTF)
                _tok = _teacher_tokens(_enc, _vids, _TUB)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _pr, _ = wm_forward(_vids, _MX8, _MY8, _enc, _enc, _pred)
                _tf = torch.full((_B, _T * _P, 768), float("nan"), device="cuda")
                _tf[:, _MY8[0]] = _pr.float()
                _cau = _tf.view(_B, _T, _P, 768)[:, _TUB].float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _fe = _enc(_vids)
                _cpy = _fe.view(_B, _T, _P, 768)[:, 3].float()   # last context tubelet
                for _k, _tk in (("teacher", _tok), ("causal", _cau), ("copy", _cpy)):
                    _d = decoder(_tok_grid(_tk).float()).float()
                    agg[_k][0].append(_psnr(_d, _gt))
                    agg[_k][1].append(_ssim1(_d, _gt))
        out = {k: (float(np.mean(v[0])), float(np.mean(v[1]))) for k, v in agg.items()}
        print(f"[{tag}] " + " | ".join(f"{k}: PSNR {v[0]:.2f} (1-SSIM {v[1]:.3f})"
                                       for k, v in out.items()))
        return out


    dec_res = {}
    for _vm, _pool in _val_pools.items():
        dec_res[_vm] = _decode_metrics(_pool, _vm)

    # montage GT | teacher | causal | copy for first 2 clips of each val env
    from PIL import Image


    def _pil(t):
        a = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(a)


    _sdir = CONFIG["ckpt_dir"] / "montages"
    _sdir.mkdir(parents=True, exist_ok=True)
    for _vm, _pool in _val_pools.items():
        for _ci in range(2):
            _v = _pool[_ci].unsqueeze(0).to("cuda")
            _gt = _gt_frame(_v, _GTF)
            _tok = _teacher_tokens(_enc, _v, _TUB)
            _mx1, _my1 = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                               batch_size=1, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _pr, _ = wm_forward(_v, _mx1, _my1, _enc, _enc, _pred)
                _fe = _enc(_v)
            _tf = torch.full((1, _T * _P, 768), float("nan"), device="cuda")
            _tf[:, _my1[0]] = _pr.float()
            _cau = _tf.view(1, _T, _P, 768)[:, _TUB].float()
            _cpy = _fe.view(1, _T, _P, 768)[:, 3].float()
            with torch.no_grad():
                _d_tea = decoder(_tok_grid(_tok).float())
                _d_cau = decoder(_tok_grid(_cau).float())
                _d_cpy = decoder(_tok_grid(_cpy).float())
            _canvas = Image.new("RGB", (4 * 384 + 6, 388), (255, 255, 255))
            for _i, _im in enumerate([_pil(_gt[0]), _pil(_d_tea[0]),
                                      _pil(_d_cau[0]), _pil(_d_cpy[0])]):
                _canvas.paste(_im, (_i * (384 + 2), 2))
            _p = _sdir / f"montage_{_vm[:8]}_{_ci}.png"
            _canvas.save(_p)
            print("saved", _p)
    phase2b_ok = True

    return


@app.cell
def evals_latent(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    np,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # Phase-2 evals (latent): per-tubelet horizon decomposition (b=8, tubelets 4..7)
    # and stride extrapolation sweep 2/4/6/8 on held-out ICE-1 & SPX-2.
    # causal = phase2 frozen-encoder predictor; copy = last-context-tubelet repeat.

    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _m in (enc, tgt):
            for _p in _m.parameters():
                _p.requires_grad_(False)
        return enc, tgt


    def _load_pred(device="cuda"):
        _, _, pred = build_spike_models(device)
        pred.load_state_dict(torch.load(CONFIG["ckpt_dir"] / "phase2_stride4_pred1000.pt",
                                        map_location="cuda", weights_only=True)["pred"])
        pred.eval()
        return pred


    _enc, _tgt = _load_frozen()
    _pred = _load_pred()
    print("loaded official encoder + phase2 causal predictor")

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _NCTX = 8 // CONFIG["tubelet_size"]          # context tubelets at b=8


    def _pool(mission, stride, n):
        mdir = Path(CONFIG["data_root"]) / mission
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=stride,
                                  step=CONFIG["num_frames"], max_clips=n)
        return [load_clip(mdir, r, CONFIG["img_size"]) for r in recs]


    def _predict(vids, mx, my):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr, tg = wm_forward(vids, mx, my, _enc, _enc, _pred)
        return pr.float(), tg.float()


    def _copy_tokens(vids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            full = _enc(vids)
        return full.view(vids.shape[0], _T, _P, 768)[:, _NCTX - 1].float()  # tubelet 3


    def _tubelet_slice(tokens, k):
        return tokens[:, (k - _NCTX) * _P:(k - _NCTX + 1) * _P]


    def _l1_cos(a, b):
        p = F.layer_norm(a, (a.shape[-1],))
        t = F.layer_norm(b, (b.shape[-1],))
        return F.cosine_similarity(p, t, dim=-1).mean().item()


    def _eval(mission, stride, n=8, seed=17, n_pool=None):
        pool = _pool(mission, stride, n_pool or n)
        rng = np.random.default_rng(seed)
        mx, my = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                       batch_size=_B, device="cuda")
        agg = {"causal": [], "copy": [], "per_tubelet_causal": [0.0] * 4,
               "per_tubelet_copy": [0.0] * 4}
        cnt = 0
        with torch.no_grad():
            for _ in range(max(1, n // _B)):
                _vids = torch.stack([pool[i] for i in rng.integers(0, len(pool), size=_B)]).to("cuda")
                pr, tg = _predict(_vids, mx, my)
                cp = _copy_tokens(_vids)
                agg["causal"].append(_l1_cos(pr, tg))
                cp_full = cp.unsqueeze(1).expand(-1, 4, -1, -1).reshape(-1, 4 * _P, 768).contiguous()
                agg["copy"].append(_l1_cos(cp_full, tg))
                for _j in range(4):
                    k = _NCTX + _j
                    agg["per_tubelet_causal"][_j] += _l1_cos(_tubelet_slice(pr, k), _tubelet_slice(tg, k))
                    cpk = cp  # [B, P, D] last-context tubelet tokens
                    agg["per_tubelet_copy"][_j] += _l1_cos(cpk, _tubelet_slice(tg, k))
                cnt += 1
        res = {"causal": float(np.mean(agg["causal"])),
               "copy": float(np.mean(agg["copy"])),
               "per_causal": [v / cnt for v in agg["per_tubelet_causal"]],
               "per_copy": [v / cnt for v in agg["per_tubelet_copy"]]}
        print(f"[{mission[:8]} s{stride}] causal {res['causal']:.4f} | copy {res['copy']:.4f} "
              f"| per-tubelet causal {[round(x, 3) for x in res['per_causal']]} "
              f"copy {[round(x, 3) for x in res['per_copy']]}")
        return res


    _envs = {"2024-11-18-13-48-19": "ICE-1", "2024-11-02-17-18-32": "SPX-2"}
    evals_latent_results = {}
    for _m in _envs:
        evals_latent_results[_envs[_m]] = {}
        for _s in (2, 4, 6, 8):
            evals_latent_results[_envs[_m]][_s] = _eval(_m, _s, n=8, seed=17)

    print("--- horizon sweep summary (cosine vs official teacher, causal query b=8) ---")
    for _env in evals_latent_results:
        print(f"{_env}: " + " | ".join(
            f"s{_s} caus {evals_latent_results[_env][_s]['causal']:.3f}/"
            f"copy {evals_latent_results[_env][_s]['copy']:.3f}" for _s in (2, 4, 6, 8)))
    evals_latent_ok = True

    return


@app.cell
def path_probe(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # Path probe: can the latent carry the quadruped's path?
    # Linear head (mean-pooled tubelet-4 tokens) -> (dx, dy) body-frame displacement
    # between sampled frames i=7 and i=9 (~0.8 s at stride 4), GT from DLIO odom.
    # Feature sources compared on held-out envs:
    #   teacher : official encoder tubelet-4 latent (oracle)
    #   causal  : phase2 causal predictor output for tubelet 4
    #   copy    : last-context tubelet (index 3) latent (predicts ~zero motion)

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4
    _NCTX = 8 // CONFIG["tubelet_size"]


    def _pull_dlio(mission, data_root):
        from huggingface_hub import snapshot_download
        import tarfile
        cache = Path(snapshot_download(
            "leggedrobotics/grand_tour_dataset",
            allow_patterns=[f"{mission}/data/dlio_map_odometry*"],
            repo_type="dataset"))
        for tar in cache.rglob("*.tar"):
            rel = tar.relative_to(cache)
            dst_parent = Path(data_root) / rel.parent
            dst_parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar) as tf:
                tf.extractall(path=dst_parent)
            print("extracted", rel)


    class _Sync:
        def __init__(self, mission_dir):
            import zarr
            self.zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
            self.zd = zarr.open_group(str(mission_dir / "data" / "dlio_map_odometry"), mode="r")
            self.ts_cam = self.zc["timestamp"][:]
            self.ts_d = self.zd["timestamp"][:]
            self.pos = self.zd["pose_pos"][:]
            self.quat = self.zd["pose_orien"][:]   # xyzw

        def _near(self, ts):
            i = int(np.searchsorted(self.ts_d, ts))
            if i >= len(self.ts_d):
                i = len(self.ts_d) - 1
            if i > 0 and abs(self.ts_d[i - 1] - ts) < abs(self.ts_d[i] - ts):
                i -= 1
            return i

        def pose(self, frame):
            i = self._near(self.ts_cam[frame])
            return self.pos[i], self.quat[i]


    def _qmat(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)


    def _body_delta(sync, f7, f9):
        p7, q7 = sync.pose(f7)
        p9, q9 = sync.pose(f9)
        R7 = _qmat(q7)
        d = np.asarray(p9) - np.asarray(p7)
        return R7.T @ d            # body-frame translation at t7


    # ---- 1) pull DLIO for train + val missions ---------------------------------
    _path_missions = {"train": "2024-10-01-11-29-55",
                      "val1": "2024-11-18-13-48-19",
                      "val2": "2024-11-02-17-18-32"}
    for _k, _m in _path_missions.items():
        _dir = Path(CONFIG["data_root"]) / _m
        if not (_dir / "data" / "dlio_map_odometry").exists():
            _pull_dlio(_m, CONFIG["data_root"])
        print(_k, "dlio ok")

    # ---- 2) records + clips (idx-aligned) + labels -----------------------------
    def _probe_data(mission, n):
        mdir = Path(CONFIG["data_root"]) / mission
        sync = _Sync(mdir)
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=_STRIDE,
                                  step=CONFIG["num_frames"], max_clips=n)
        clips, labels = [], []
        for rec in recs:
            try:
                d = _body_delta(sync, rec[7], rec[9])
            except Exception:
                continue
            clips.append(load_clip(mdir, rec, CONFIG["img_size"]))
            labels.append(d[:2].astype(np.float32))    # dx, dy (m)
        return clips, np.array(labels), recs


    print("building train probe data (ETH-1) ...")
    train_clips, train_y, _ = _probe_data(_path_missions["train"], 140)
    print("building val probe data ...")
    val1_clips, val1_y, _ = _probe_data(_path_missions["val1"], 30)
    val2_clips, val2_y, _ = _probe_data(_path_missions["val2"], 30)
    print(f"train {len(train_y)} val1 {len(val1_y)} val2 {len(val2_y)} "
          f"| GT mean |d| train {np.linalg.norm(train_y, axis=1).mean():.3f} m")

    # ---- 3) official encoder + causal predictor (loaded like other eval cells) --
    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    def _load_pred(device="cuda"):
        _, _, pred = build_spike_models(device)
        pred.load_state_dict(torch.load(CONFIG["ckpt_dir"] / "phase2_stride4_pred1000.pt",
                                        map_location="cuda", weights_only=True)["pred"])
        pred.eval()
        return pred


    _enc, _tgt = _load_frozen()
    _pred = _load_pred()
    _MX, _MY = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                     batch_size=_B, device="cuda")


    def _feats(clips, source):
        """mean-pooled tubelet features [n, 768] per clip for teacher/causal/copy."""
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), _B):
                _sl = clips[i:i + _B]
                _b = len(_sl)
                vids = torch.stack(_sl).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if source == "teacher":
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, _NCTX]
                    elif source == "causal":
                        _mx, _my = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                                         batch_size=_b, device="cuda")
                        pr, _ = wm_forward(vids, _mx, _my, _enc, _enc, _pred)
                        tok = pr[:, :_P]  # first target tubelet (idx 4) = first rows of masks_y order
                    else:  # copy: last context tubelet (index 3)
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, 3]
                outs.append(tok.float().mean(dim=1))
        return torch.cat(outs, dim=0)


    print("extracting features ...")
    _X_tr = _feats(train_clips, "teacher")
    _X1 = {s: _feats(val1_clips, s) for s in ("teacher", "causal", "copy")}
    _X2 = {s: _feats(val2_clips, s) for s in ("teacher", "causal", "copy")}

    # ---- 4) fit linear head on teacher features (ETH-1) -------------------------
    _head = nn.Linear(768, 2).to("cuda")
    _opt = torch.optim.AdamW(_head.parameters(), lr=1e-3)
    _ytr = torch.from_numpy(train_y).to("cuda")
    for _e in range(80):
        _perm = torch.randperm(len(_X_tr), device="cuda")[:_B * 8]
        _loss = F.mse_loss(_head(_X_tr[_perm]), _ytr[_perm])
        _opt.zero_grad(set_to_none=True)
        _loss.backward()
        _opt.step()
    _head.eval()


    def _endpoint_err(X, y):
        errs = torch.norm(_head(X) - torch.from_numpy(y).to("cuda"), dim=1)
        return float(errs.mean())


    print("--- path probe: mean endpoint error (m), linear head trained on ETH-1 teacher ---")
    for _nm, _X, _y in (("ICE-1", _X1, val1_y), ("SPX-2", _X2, val2_y)):
        _gt = float(np.linalg.norm(_y, axis=1).mean())
        _row = {"teacher": _endpoint_err(_X["teacher"], _y),
                "causal": _endpoint_err(_X["causal"], _y),
                "copy": _endpoint_err(_X["copy"], _y),
                "gt_dist_m": _gt}
        path_probe_results = path_probe_results | {_nm: _row} if "path_probe_results" in dir() else {_nm: _row}
        print(f"{_nm}: GT |d| {_gt:.3f} m | teacher {_row['teacher']:.3f} m | "
              f"causal {_row['causal']:.3f} m | copy {_row['copy']:.3f} m")
    path_probe_ok = True

    return (path_probe_results,)


@app.cell
def topics(CONFIG, Path):
    # Phase-3 prep: pull command/proprio topics for ETH-1 (train) + ICE-1 + SPX-2
    # (held-out) and inspect schemas / sync vs hdr_front camera timestamps.

    _TOPIC_MISSIONS = ["2024-10-01-11-29-55", "2024-11-18-13-48-19", "2024-11-02-17-18-32"]
    _TOPICS = ["anymal_command_twist", "anymal_state_actuator",
               "anymal_state_state_estimator"]


    def _pull_topic(mission, topic, data_root):
        from huggingface_hub import snapshot_download
        import tarfile
        cache = Path(snapshot_download(
            "leggedrobotics/grand_tour_dataset",
            allow_patterns=[f"{mission}/data/{topic}*"],
            repo_type="dataset"))
        for tar in cache.rglob("*.tar"):
            rel = tar.relative_to(cache)
            dst_parent = Path(data_root) / rel.parent
            dst_parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar) as tf:
                tf.extractall(path=dst_parent)


    def _describe(mission, topic):
        import zarr
        g = zarr.open_group(str(Path(CONFIG["data_root"]) / mission / "data" / topic),
                            mode="r")
        out = {"keys": list(g.keys())}
        for k in g.keys():
            a = g[k]
            try:
                n = len(a)
                import numpy as _np
                if n:
                    s = _np.asarray(a[0])
                    out[k] = {"len": n, "dtype": str(a.dtype), "first": str(s)[:80]}
            except Exception as e:  # noqa: BLE001
                out[k] = {"err": repr(e)}
        return out


    topic_schemas = {}
    for _m in _TOPIC_MISSIONS:
        for _t in _TOPICS:
            _p = Path(CONFIG["data_root"]) / _m / "data" / _t
            if not _p.exists():
                _pull_topic(_m, _t, CONFIG["data_root"])
            try:
                topic_schemas[f"{_m}/{_t}"] = _describe(_m, _t)
            except Exception as _e:  # noqa: BLE001
                topic_schemas[f"{_m}/{_t}"] = {"pull_or_open_err": repr(_e)}
            print(_m[:8], _t, "->", str(topic_schemas[f"{_m}/{_t}"])[:300])

    # camera ts span for reference
    import zarr as _z
    for _m in _TOPIC_MISSIONS:
        _g = _z.open_group(str(Path(CONFIG["data_root"]) / _m / "data" / "hdr_front"), mode="r")
        _ts = _g["timestamp"]
        print(_m[:8], "hdr_front ts: n", len(_ts), "span_s",
              round(float(_ts[-1] - _ts[0]), 1))
    topics_pulled = True

    return


@app.cell
def cmd_probe(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    temporal_causal_masks,
    torch,
):
    # Phase-3 signal experiment: controllability at readout level.
    # (a) coupling sanity: commanded linear vx vs odom-derived speed (ICE-1)
    # (b) control-signal probe: linear head -> (dx,dy) displacement over the target
    #     window, conditioned on (video tubelet-4 features | +command | command-only)
    #     with TRUE vs SHUFFLED commands. If +true command < video-only and
    #     shuffled > true, the command stream carries control signal.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4
    _NCTX = 8 // CONFIG["tubelet_size"]
    _MISSIONS = {"train": "2024-10-01-11-29-55",
                 "val1": "2024-11-18-13-48-19",
                 "val2": "2024-11-02-17-18-32"}


    class _Cmd:
        def __init__(self, mission_dir):
            import zarr
            g = zarr.open_group(str(mission_dir / "data" / "anymal_command_twist"), mode="r")
            self.ts = np.asarray(g["timestamp"][:])
            self.linear = np.asarray(g["linear"][:])     # (N,3)
            self.angular = np.asarray(g["angular"][:])   # (N,3)

        def window_mean(self, t0, t1):
            i0 = int(np.searchsorted(self.ts, t0))
            i1 = int(np.searchsorted(self.ts, t1))
            i0 = max(0, i0 - 1)
            i1 = min(len(self.ts), i1 + 1)
            if i1 <= i0:
                return None
            return np.concatenate([self.linear[i0:i1].mean(0),
                                   self.angular[i0:i1].mean(0)])  # 6-dim


    class _Odo:
        def __init__(self, mission_dir):
            import zarr
            zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
            zd = zarr.open_group(str(mission_dir / "data" / "dlio_map_odometry"), mode="r")
            self.ts_cam = np.asarray(zc["timestamp"][:])
            self.ts_d = np.asarray(zd["timestamp"][:])
            self.pos = np.asarray(zd["pose_pos"][:])
            self.orien = np.asarray(zd["pose_orien"][:])   # xyzw

        def _near(self, ts):
            i = int(np.searchsorted(self.ts_d, ts))
            if i >= len(self.ts_d):
                i = len(self.ts_d) - 1
            if i > 0 and abs(self.ts_d[i - 1] - ts) < abs(self.ts_d[i] - ts):
                i -= 1
            return i

        def speed(self, f0, f1):
            p0 = self.pos[self._near(self.ts_cam[f0])]
            p1 = self.pos[self._near(self.ts_cam[f1])]
            dt = float(self.ts_cam[f1] - self.ts_cam[f0])
            return (float(np.linalg.norm(p1 - p0) / dt) if dt > 0 else 0.0), self.ts_cam[f0]

        def body_delta(self, f7, f9):
            x, y, z, w = self.orien[self._near(self.ts_cam[f7])]
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]], dtype=float)
            d = self.pos[self._near(self.ts_cam[f9])] - self.pos[self._near(self.ts_cam[f7])]
            return R.T @ d


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    def _load_pred
(device="cuda"):
        _, _, pred = build_spike_models(device)
        pred.load_state_dict(torch.load(CONFIG["ckpt_dir"] / "phase2_stride4_pred1000.pt",
                                        map_location="cuda", weights_only=True)["pred"])
        pred.eval()
        return pred


    _enc, _tgt = _load_frozen()
    _pred = _load_pred()
    _MX, _MY = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                     batch_size=_B, device="cuda")

    # ---- (a) coupling sanity on ICE-1: cmd vx vs odom speed --------------------
    _cmd = _Cmd(Path(CONFIG["data_root"]) / _MISSIONS["val1"])
    _odo = _Odo(Path(CONFIG["data_root"]) / _MISSIONS["val1"])
    _vx, _sp, _tt = [], [], []
    for _f0 in range(0, min(len(_odo.ts_cam) - 20, 4000), 100):
        _f1 = _f0 + 10
        _s, _t0 = _odo.speed(_f0, _f1)
        _i = int(np.searchsorted(_cmd.ts, _t0))
        if 0 < _i < len(_cmd.ts):
            _vx.append(float(_cmd.linear[_i][0]))
            _sp.append(_s)
            _tt.append(_t0)
    _corr = float(np.corrcoef(_vx, _sp)[0, 1]) if len(_vx) > 3 else float("nan")
    print(f"[ICE-1] coupling corr(cmd.vx, odom speed) = {_corr:.3f} over {len(_vx)} samples "
          f"| cmd rate ~{len(_cmd.ts) / max(1e-9, float(_cmd.ts[-1] - _cmd.ts[0])):.1f} Hz")

    # ---- (b) control-signal probe ----------------------------------------------
    def _probe_data(mission, n):
        mdir = Path(CONFIG["data_root"]) / mission
        cmd = _Cmd(mdir)
        sync = _Odo(mdir)
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=_STRIDE,
                                  step=CONFIG["num_frames"], max_clips=n)
        clips, y, cmd_feat = [], [], []
        for rec in recs:
            f7, f9 = rec[7], rec[9]
            t7, t9 = sync.ts_cam[f7], sync.ts_cam[f9]
            cf = cmd.window_mean(t7, t9)
            clips.append(load_clip(mdir, rec, CONFIG["img_size"]))
            y.append(sync.body_delta(f7, f9)[:2].astype(np.float32))
            cmd_feat.append(cf if cf is not None else np.zeros(6, dtype=np.float32))
            if len(clips) % 20 == 0:
                print(f"  probe {mission[:8]} {len(clips)} clips", flush=True)
        return clips, np.array(y), np.array(cmd_feat, dtype=np.float32)


    def _teacher_feats(clips):
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), _B):
                vids = torch.stack(clips[i:i + _B]).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    full = _enc(vids)
                outs.append(full.view(vids.shape[0], _T, _P, 768)[:, _NCTX].float().mean(1))
                if (i // _B) % 8 == 0:
                    print(f"  feats {i + len(vids)}/{len(clips)}", flush=True)
        return torch.cat(outs)


    print("building probe data ...")
    _tr_clips, _tr_y, _tr_cmd = _probe_data(_MISSIONS["train"], 140)
    _v1_clips, _v1_y, _v1_cmd = _probe_data(_MISSIONS["val1"], 30)
    _v2_clips, _v2_y, _v2_cmd = _probe_data(_MISSIONS["val2"], 30)
    _Xtr = _teacher_feats(_tr_clips)
    _X1 = _teacher_feats(_v1_clips)
    _X2 = _teacher_feats(_v2_clips)
    _Ytr = torch.from_numpy(_tr_y).to("cuda")
    _Ctr = torch.from_numpy(_tr_cmd).to("cuda")
    _C1 = torch.from_numpy(_v1_cmd).to("cuda")
    _C2 = torch.from_numpy(_v2_cmd).to("cuda")


    def _fit_and_eval(make_x, tag):
        """make_x(train_X, train_C, val_C_shift) -> (tr, v1, v2) feature tensors."""
        tr = make_x(_Xtr, _Ctr, None)
        # shuffled commands: roll along sample axis by a fixed offset
        _k = 17
        v1_true = make_x(_X1, _C1, None)
        v2_true = make_x(_X2, _C2, None)
        v1_shuf = make_x(_X1, torch.roll(_C1, _k, dims=0), _k)
        v2_shuf = make_x(_X2, torch.roll(_C2, _k, dims=0), _k)
        out = {}
        for _nm, _X, _y, _C in (("v1_true", v1_true, _v1_y, _C1),
                                ("v2_true", v2_true, _v2_y, _C2),
                                ("v1_shuf", v1_shuf, _v1_y, torch.roll(_C1, _k, dims=0)),
                                ("v2_shuf", v2_shuf, _v2_y, torch.roll(_C2, _k, dims=0))):
            head = nn.Linear(tr.shape[1], 2).to("cuda")
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
            yt = torch.from_numpy(_y).to("cuda")
            for _e in range(120):
                _perm = torch.randperm(len(tr), device="cuda")[:_B * 8]
                loss = F.mse_loss(head(tr[_perm]), _Ytr[_perm])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                err = float(torch.norm(head(_X) - yt, dim=1).mean())
            out[_nm] = err
        print(f"[{tag}] v1_true {out['v1_true']:.3f} | v1_shuf {out['v1_shuf']:.3f} | "
              f"v2_true {out['v2_true']:.3f} | v2_shuf {out['v2_shuf']:.3f}")
        return out


    results_cmd = {}
    results_cmd["video_only"] = _fit_and_eval(lambda X, C, s: X, "video-only")
    results_cmd["video+cmd"] = _fit_and_eval(
        lambda X, C, s: torch.cat([X, C], dim=1), "video+cmd")
    results_cmd["cmd_only"] = _fit_and_eval(
        lambda X, C, s: C, "cmd-only")
    print("GT mean |d| (val):", round(float(np.linalg.norm(_v1_y, axis=1).mean()), 3),
          round(float(np.linalg.norm(_v2_y, axis=1).mean()), 3))
    phase3_cmd_ok = True

    return


@app.cell
def contact_probe(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # Tier-2 hidden-state probe: do future latents carry the quadruped's state?
    # Readouts (linear heads trained on ETH-1 TEACHER features, applied to all
    # feature sources) for three hidden states over the target window (tubelet 4):
    #   R1 body linear velocity  (estimator twist_lin, window mean)  -> RMSE/corr
    #   R2 foot-contact vector   (estimator *_FOOT_contact, 4 bits)  -> MAE/acc@.5
    #   R3 vertical bob dz (dlio pose z over window)                 -> RMSE/corr
    # Baselines: persist-at-context (B1/B2: state at t7; B3: dz=0) and the
    # copy-last-tubelet feature readout.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4
    _NCTX = 8 // CONFIG["tubelet_size"]
    _FEET = ["LF", "LH", "RF", "RH"]


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    def _load_pred(device="cuda"):
        _, _, pred = build_spike_models(device)
        pred.load_state_dict(torch.load(CONFIG["ckpt_dir"] / "phase2_stride4_pred1000.pt",
                                        map_location="cuda", weights_only=True)["pred"])
        pred.eval()
        return pred


    class _Est:
        def __init__(self, mission_dir):
            import zarr
            g = zarr.open_group(str(mission_dir / "data" / "anymal_state_state_estimator"),
                                mode="r")
            self.ts = np.asarray(g["timestamp"][:])
            self.twist = np.asarray(g["twist_lin"][:])          # (N,3) body velocity
            self.contact = np.stack([np.asarray(g[f"{f}_FOOT_contact"][:])
                                     for f in _FEET], axis=1)   # (N,4) uint8
            zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
            self.ts_cam = np.asarray(zc["timestamp"][:])
            zd = zarr.open_group(str(mission_dir / "data" / "dlio_map_odometry"), mode="r")
            self.ts_d = np.asarray(zd["timestamp"][:])
            self.pos = np.asarray(zd["pose_pos"][:])

        def _win(self, t0, t1):
            i0 = max(0, int(np.searchsorted(self.ts, t0)) - 1)
            i1 = min(len(self.ts), int(np.searchsorted(self.ts, t1)) + 1)
            return i0, max(i0 + 1, i1)

        def labels(self, f7, f9):
            t7, t9 = self.ts_cam[f7], self.ts_cam[f9]
            i0, i1 = self._win(t7, t9)
            return (self.twist[i0:i1].mean(0),
                    self.contact[i0:i1].mean(0).astype(np.float32))

        def persist(self, f7):
            i = max(0, int(np.searchsorted(self.ts, self.ts_cam[f7])) - 1)
            return self.twist[i], self.contact[i].astype(np.float32)

        def dz(self, f7, f9):
            j7 = int(np.searchsorted(self.ts_d, self.ts_cam[f7]))
            j9 = int(np.searchsorted(self.ts_d, self.ts_cam[f9]))
            j7 = max(0, min(len(self.ts_d) - 1, j7))
            j9 = max(0, min(len(self.ts_d) - 1, j9))
            return float(self.pos[j9][2] - self.pos[j7][2])


    def _probe_data(mission, n):
        mdir = Path(CONFIG["data_root"]) / mission
        est = _Est(mdir)
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=_STRIDE,
                                  step=CONFIG["num_frames"], max_clips=n)
        clips, yv, yc, yz, pv, pc = [], [], [], [], [], []
        for _j, rec in enumerate(recs):
            v, c = est.labels(rec[7], rec[9])
            z = est.dz(rec[7], rec[9])
            pv_, pc_ = est.persist(rec[7])
            clips.append(load_clip(mdir, rec, CONFIG["img_size"]))
            yv.append(v.astype(np.float32)); yc.append(c); yz.append(z)
            pv.append(pv_.astype(np.float32)); pc.append(pc_)
            if (_j + 1) % 20 == 0:
                print(f"  probe {mission[:8]} {_j + 1}/{len(recs)}", flush=True)
        return (clips, np.array(yv), np.array(yc), np.array(yz),
                np.array(pv), np.array(pc))


    _enc, _tgt = _load_frozen()
    _pred = _load_pred()
    _MX, _MY = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                     batch_size=_B, device="cuda")

    print("building probe data (train ETH-1, val ICE-1/SPX-2) ...")
    _tr = _probe_data("2024-10-01-11-29-55", 120)
    _v1 = _probe_data("2024-11-18-13-48-19", 24)
    _v2 = _probe_data("2024-11-02-17-18-32", 24)


    def _feats(clips, source):
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), _B):
                _sl = clips[i:i + _B]
                _b = len(_sl)
                vids = torch.stack(_sl).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if source == "teacher":
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, _NCTX]
                    elif source == "causal":
                        _mx, _my = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                                         batch_size=_b, device="cuda")
                        pr, _ = wm_forward(vids, _mx, _my, _enc, _enc, _pred)
                        tok = pr[:, :_P]
                    else:
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, 3]
                outs.append(tok.float().mean(dim=1))
                if (i // _B) % 10 == 0:
                    print(f"  feats {source} {min(i + _B, len(clips))}/{len(clips)}", flush=True)
        return torch.cat(outs)


    def _fit(X, y, dim, iters=150):
        h = nn.Linear(768, dim).to("cuda")
        o = torch.optim.AdamW(h.parameters(), lr=1e-3)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32)).to("cuda")
        for _ in range(iters):
            _p = torch.randperm(len(X), device="cuda")[:_B * 8]
            loss = F.mse_loss(h(X[_p]), yt[_p])
            o.zero_grad(set_to_none=True)
            loss.backward(); o.step()
        h.eval()
        return h, yt


    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / den) if den > 0 else 0.0


    def _eval_set(name, data):
        clips, yv, yc, yz, pv, pc = data
        print(f"[{name}] feats ...", flush=True)
        ft = {s: _feats(clips, s) for s in ("teacher", "causal", "copy")}
        out = {}
        hv, _ = _fit(ft["teacher"], yv, 3)
        hc, _ = _fit(ft["teacher"], yc, 4)
        hz, _ = _fit(ft["teacher"], yz[:, None], 1)
        with torch.no_grad():
            for s, X in ft.items():
                ev = hv(X).cpu().numpy()
                ec = hc(X).cpu().numpy()
                ez = hz(X).cpu().numpy()[:, 0]
                out[s] = {
                    "vel_rmse": float(np.sqrt(np.mean((ev - yv) ** 2))),
                    "vel_corr": _corr(ev[:, 0], yv[:, 0]),
                    "cont_mae": float(np.mean(np.abs(ec - yc))),
                    "cont_acc": float(np.mean((ec > 0.5).astype(float) == (yc > 0.5).astype(float))),
                    "dz_rmse": float(np.sqrt(np.mean((ez - yz) ** 2))),
                    "dz_corr": _corr(ez, yz),
                }
            # persist baselines
            b_vel = float(np.sqrt(np.mean((pv - yv) ** 2)))
            b_cont = float(np.mean(np.abs(pc - yc)))
            b_dz = float(np.sqrt(np.mean(yz ** 2)))
            out["persist"] = {"vel_rmse": b_vel, "cont_mae": b_cont, "dz_rmse": b_dz}
        for s in ("teacher", "causal", "copy", "persist"):
            r = out[s]
            print(f"[{name}] {s:8s} velRMSE {r.get('vel_rmse', float('nan')):.4f} "
                  f"(corr {r.get('vel_corr', float('nan')):.2f}) | "
                  f"contact MAE {r.get('cont_mae', float('nan')):.4f} "
                  f"acc {r.get('cont_acc', float('nan')):.3f} | "
                  f"dzRMSE {r.get('dz_rmse', float('nan')):.4f} "
                  f"(corr {r.get('dz_corr', float('nan')):.2f})", flush=True)
        return out


    hidden_state_results = {"train": _eval_set("ETH-1", _tr),
                            "ICE-1": _eval_set("ICE-1", _v1),
                            "SPX-2": _eval_set("SPX-2", _v2)}
    contact_probe_ok = True

    return


@app.cell
def train_mw(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    np,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # P1b: motion-weighted causal pretraining (frozen official encoder).
    # Target-token loss weight ~ 1 + ALPHA*(tubelet motion / mean motion), where
    # motion = mean |frame(2j) - frame(2j-1)| per target tubelet. Goal: predicted
    # latents must preserve motion/velocity detail that plain LN-L1 averages away.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4
    _ALPHA = 2.0
    _STEPS = 1000
    _TRAIN_M = "2024-10-01-11-29-55"
    _VAL_M = "2024-11-18-13-48-19"


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    _enc, _tgt = _load_frozen()
    _, _, _pred = build_spike_models("cuda")
    _pred.train()

    print("preloading pools ...", flush=True)


    def _pool(mission, n):
        mdir = Path(CONFIG["data_root"]) / mission
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=_STRIDE,
                                  step=CONFIG["num_frames"], max_clips=n)
        out = []
        for _k, r in enumerate(recs):
            out.append(load_clip(mdir, r, CONFIG["img_size"]))
            if (_k + 1) % 40 == 0:
                print(f"  pool {mission[:8]} {_k + 1}/{len(recs)}", flush=True)
        return out


    train_pool_mw = _pool(_TRAIN_M, 160)
    val_pool_mw = _pool(_VAL_M, 20)
    print("pools", len(train_pool_mw), len(val_pool_mw), flush=True)


    def _tubelet_motion_w(vids, b_tok):
        """[B, nb] weights for target tubelets j in [b_tok, T)."""
        d = torch.stack([
            (vids[:, :, 2 * j] - vids[:, :, 2 * j - 1]).abs().mean(dim=(1, 2, 3))
            for j in range(b_tok, _T)], dim=1)                       # [B, nb]
        w = 1.0 + _ALPHA * (d / (d.mean(dim=1, keepdim=True) + 1e-6) - 1.0)
        return w.clamp(0.25, 4.0)


    def weighted_loss(pred, tgt, vids, b_tok):
        p = F.layer_norm(pred.float(), (pred.shape[-1],))
        t = F.layer_norm(tgt.float(), (tgt.shape[-1],))
        diff = (p - t).abs()
        ntgt = diff.shape[1]
        nb = _T - b_tok
        w = _tubelet_motion_w(vids, b_tok)                            # [B, nb]
        row_j = (torch.arange(ntgt, device=diff.device) // _P)        # offset idx
        wt = w[:, row_j].unsqueeze(-1)                                # [B, ntgt, 1]
        return (diff * wt).mean()


    _opt = torch.optim.AdamW(_pred.parameters(), lr=CONFIG["lr"], weight_decay=0.05)
    _tr = [p for p in _pred.parameters() if p.requires_grad]
    _rng = np.random.default_rng(0)
    _BOUNDARIES = (6, 8, 10)


    def _samp(pool):
        idx = _rng.integers(0, len(pool), size=_B)
        return torch.stack([pool[i] for i in idx]).to("cuda")


    print("training motion-weighted causal predictor ...", flush=True)
    mw_history = []
    for _s in range(_STEPS):
        _b = _BOUNDARIES[int(_rng.integers(0, len(_BOUNDARIES)))]
        _b_tok = _b // CONFIG["tubelet_size"]
        _mx, _my = temporal_causal_masks(_b, _T, _P, CONFIG["tubelet_size"],
                                         batch_size=_B, device="cuda")
        _vids = _samp(train_pool_mw)
        _opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _pr, _tg = wm_forward(_vids, _mx, _my, _enc, _tgt, _pred)
            _loss = weighted_loss(_pr, _tg, _vids, _b_tok)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(_tr, CONFIG["grad_clip"])
        _opt.step()
        if (_s + 1) % 100 == 0:
            # quick val: cosine vs official teacher on val pool (causal query b=8)
            _enc.eval(); _pred.eval()
            _vx, _vy = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                             batch_size=_B, device="cuda")
            _cs = []
            with torch.no_grad():
                for _q in range(2):
                    _qv = _samp(val_pool_mw)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _pr2, _tg2 = wm_forward(_qv, _vx, _vy, _enc, _enc, _pred)
                    p = F.layer_norm(_pr2.float(), (768,))
                    t = F.layer_norm(_tg2.float(), (768,))
                    _cs.append(F.cosine_similarity(p, t, dim=-1).mean().item())
            _enc.eval(); _pred.train()
            mw_history.append({"step": _s + 1, "train_loss": float(_loss.detach()),
                               "val_cos": float(np.mean(_cs))})
            print(f"step {_s + 1:5d} | train {float(_loss.detach()):.4f} | "
                  f"val cos {float(np.mean(_cs)):.4f}", flush=True)
        if (_s + 1) % 500 == 0:
            torch.save({"step": _s + 1, "pred": _pred.state_dict()},
                       CONFIG["ckpt_dir"] / f"phase2_mw_pred{_s + 1}.pt")

    torch.save({"step": _STEPS, "pred": _pred.state_dict(), "alpha": _ALPHA},
               CONFIG["ckpt_dir"] / f"phase2_mw_pred{_STEPS}.pt")
    print("saved phase2_mw_pred", _STEPS, "| history tail", mw_history[-1] if mw_history else None)
    train_mw_ok = True

    return


@app.cell
def probe_mw(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # P1b re-probe: does motion-weighted training fix hidden-state precision?
    # Readouts (linear, trained on ETH-1 TEACHER features) for
    #  R1 body linear velocity over each target tubelet window (near k=4, far k=7)
    #  R2 body-frame displacement from context-end to tubelet window end
    # Sources: teacher | causal-mw (new) | causal (old phase2) | copy-last | persist.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4
    _NCTX = 8 // CONFIG["tubelet_size"]
    _FEET = ["LF", "LH", "RF", "RH"]


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    def _load_pred(name):
        _, _, pred = build_spike_models("cuda")
        pred.load_state_dict(torch.load(CONFIG["ckpt_dir"] / name,
                                        map_location="cuda", weights_only=True)["pred"])
        pred.eval()
        return pred


    class _Sync:
        def __init__(self, mission_dir):
            import zarr
            zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
            ze = zarr.open_group(str(mission_dir / "data" / "anymal_state_state_estimator"),
                                 mode="r")
            zd = zarr.open_group(str(mission_dir / "data" / "dlio_map_odometry"), mode="r")
            self.ts_cam = np.asarray(zc["timestamp"][:])
            self.ts_e = np.asarray(ze["timestamp"][:])
            self.twist = np.asarray(ze["twist_lin"][:])
            self.ts_d = np.asarray(zd["timestamp"][:])
            self.pos = np.asarray(zd["pose_pos"][:])
            self.orien = np.asarray(zd["pose_orien"][:])

        def _near(self, ts, arr):
            i = int(np.searchsorted(self.ts_e, ts))
            if i >= len(self.ts_e):
                i = len(self.ts_e) - 1
            if i > 0 and abs(self.ts_e[i - 1] - ts) < abs(self.ts_e[i] - ts):
                i -= 1
            return i

        def _near_d(self, ts):
            i = int(np.searchsorted(self.ts_d, ts))
            if i >= len(self.ts_d):
                i = len(self.ts_d) - 1
            if i > 0 and abs(self.ts_d[i - 1] - ts) < abs(self.ts_d[i] - ts):
                i -= 1
            return i

        def vel_window(self, f0, f1):
            i0 = max(0, self._near(self.ts_cam[f0], self.ts_e) - 1)
            i1 = min(len(self.ts_e), self._near(self.ts_cam[f1], self.ts_e) + 1)
            return self.twist[i0:max(i0 + 1, i1)].mean(0).astype(np.float32)

        def body_delta(self, f_ref, f_end):
            i_r = self._near_d(self.ts_cam[f_ref])
            i_e = self._near_d(self.ts_cam[f_end])
            x, y, z, w = self.orien[i_r]
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]], dtype=float)
            p0 = self.pos[i_r]
            p1 = self.pos[i_e]
            return (R.T @ (p1 - p0))[:2].astype(np.float32)


    def _probe(mission, n):
        mdir = Path(CONFIG["data_root"]) / mission
        sync = _Sync(mdir)
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=_STRIDE,
                                  step=CONFIG["num_frames"], max_clips=n)
        clips = []
        yv = {4: [], 7: []}
        yd = {4: [], 7: []}
        for _j, rec in enumerate(recs):
            clips.append(load_clip(mdir, rec, CONFIG["img_size"]))
            for k in (4, 7):
                yv[k].append(sync.vel_window(rec[2 * k], rec[2 * k + 1]))
                yd[k].append(sync.body_delta(rec[7], rec[2 * k + 1]))
            if (_j + 1) % 20 == 0:
                print(f"  probe {mission[:8]} {_j + 1}/{len(recs)}", flush=True)
        return clips, {k: np.array(v, dtype=np.float32) for k, v in yv.items()},            {k: np.array(v, dtype=np.float32) for k, v in yd.items()}


    _enc, _tgt = _load_frozen()
    _pred_mw = _load_pred("phase2_mw_pred1000.pt")
    _pred_old = _load_pred("phase2_stride4_pred1000.pt")

    print("building probe data ...", flush=True)
    _tr = _probe("2024-10-01-11-29-55", 120)
    _v1 = _probe("2024-11-18-13-48-19", 24)
    _v2 = _probe("2024-11-02-17-18-32", 24)


    def _feats(clips, source, k=None, b_tok=4):
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), _B):
                _sl = clips[i:i + _B]
                _b = len(_sl)
                vids = torch.stack(_sl).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if source == "teacher":
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, k]
                    elif source in ("causal_mw", "causal_old"):
                        _mx, _my = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                                         batch_size=_b, device="cuda")
                        pr, _ = wm_forward(vids, _mx, _my, _enc, _enc,
                                           _pred_mw if source == "causal_mw" else _pred_old)
                        tok = pr[:, (k - b_tok) * _P:(k - b_tok + 1) * _P]
                    else:  # copy: last context tubelet
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, _NCTX - 1]
                outs.append(tok.float().mean(dim=1))
                if (i // _B) % 10 == 0:
                    print(f"  feats {source} k={k} {min(i + _B, len(clips))}/{len(clips)}",
                          flush=True)
        return torch.cat(outs)


    def _fit(X, y, dim, iters=150):
        h = nn.Linear(768, dim).to("cuda")
        o = torch.optim.AdamW(h.parameters(), lr=1e-3)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32)).to("cuda")
        for _ in range(iters):
            _p = torch.randperm(len(X), device="cuda")[:_B * 8]
            loss = F.mse_loss(h(X[_p]), yt[_p])
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
        h.eval()
        return h


    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / den) if den > 0 else 0.0


    print("extracting ETH-1 teacher feats for head fitting ...", flush=True)
    _FX = {k: _feats(_tr[0], "teacher", k=k) for k in (4, 7)}
    _heads_v = {k: _fit(_FX[k], _tr[1][k], 3) for k in (4, 7)}
    _heads_d = {k: _fit(_FX[k], _tr[2][k], 2) for k in (4, 7)}

    probe_mw_results = {}
    for _nm, _data in (("ICE-1", _v1), ("SPX-2", _v2)):
        clips, yv, yd = _data
        row = {}
        for _k in (4, 7):
            row[_k] = {}
            for _src in ("teacher", "causal_mw", "causal_old", "copy"):
                X = _feats(clips, _src, k=_k)
                with torch.no_grad():
                    ev = _heads_v[_k](X).cpu().numpy()
                    ed = _heads_d[_k](X).cpu().numpy()
                rmse_v = float(np.sqrt(np.mean((ev - yv[_k]) ** 2)))
                corr_v = _corr(ev[:, 0], yv[_k][:, 0])
                err_d = float(np.sqrt(np.mean(np.sum((ed - yd[_k]) ** 2, axis=1))))
                row[_k][_src] = (rmse_v, corr_v, err_d)
        # persist baselines
        _p_v7 = np.concatenate([_tr[1][7][:1]] * 1)  # placeholder unused
        for _k in (4, 7):
            # persist velocity = velocity at context end (~frame 7 window) read from train? use 0-model
            # baseline: predict zero displacement and mean train velocity
            _b0_d = float(np.sqrt(np.mean(np.sum(yd[_k] ** 2, axis=1))))          # zero-disp baseline
            _mv = float(np.sqrt(np.mean((np.zeros_like(yv[_k]) - yv[_k]) ** 2)))   # zero-vel baseline
            row[_k]["persist"] = (_mv, 0.0, _b0_d)
        probe_mw_results[_nm] = row
        for _k in (4, 7):
            print(f"[{_nm} k={_k}] " + " | ".join(
                f"{s}: v {row[_k][s][0]:.3f}({row[_k][s][1]:.2f}) d {row[_k][s][2]:.3f}m"
                for s in ("teacher", "causal_mw", "causal_old", "copy", "persist")),
                flush=True)
    probe_mw_ok = True

    return


@app.cell
def transient_eval(
    CONFIG,
    F,
    Path,
    build_spike_models,
    load_clip,
    nn,
    np,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # Transient vs steady-state discriminator (P1b follow-up).
    # Clips are scored by actual motion over their window (estimator twist_lin/ang
    # sampled at camera frames): score = mean|ang_z| + (v_range / mean|v|+eps).
    # transient = top score clips, steady = low score clips. Heads fit on ETH-1
    # (mixed) teacher features; readouts evaluated per env x set x source.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4


    def _load_frozen(device="cuda"):
        enc,
 tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    def _load_pred(name):
        _, _, pred = build_spike_models("cuda")
        pred.load_state_dict(torch.load(CONFIG["ckpt_dir"] / name,
                                        map_location="cuda", weights_only=True)["pred"])
        pred.eval()
        return pred


    class _M:
        def __init__(self, mission_dir):
            import zarr
            zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
            ze = zarr.open_group(str(mission_dir / "data" / "anymal_state_state_estimator"),
                                 mode="r")
            zd = zarr.open_group(str(mission_dir / "data" / "dlio_map_odometry"), mode="r")
            self.ts_cam = np.asarray(zc["timestamp"][:])
            self.ts_e = np.asarray(ze["timestamp"][:])
            self.twist = np.asarray(ze["twist_lin"][:])
            self.ang = np.asarray(ze["twist_ang"][:])
            self.ts_d = np.asarray(zd["timestamp"][:])
            self.pos = np.asarray(zd["pose_pos"][:])
            self.orien = np.asarray(zd["pose_orien"][:])

        def _e(self, ts):
            i = int(np.searchsorted(self.ts_e, ts))
            if i >= len(self.ts_e):
                i = len(self.ts_e) - 1
            if i > 0 and abs(self.ts_e[i - 1] - ts) < abs(self.ts_e[i] - ts):
                i -= 1
            return i

        def _d(self, ts):
            i = int(np.searchsorted(self.ts_d, ts))
            if i >= len(self.ts_d):
                i = len(self.ts_d) - 1
            if i > 0 and abs(self.ts_d[i - 1] - ts) < abs(self.ts_d[i] - ts):
                i -= 1
            return i

        def profile(self, rec):
            ts = [self.ts_cam[f] for f in rec[::2]]           # 8 samples
            ie = [self._e(t) for t in ts]
            v = np.array([np.linalg.norm(self.twist[i][:2]) for i in ie])
            a = np.abs(self.ang[ie][:, 2])
            return float(a.mean()), float(v.max() - v.min()), float(v.mean())

        def vel_window(self, f0, f1):
            i0 = max(0, self._e(self.ts_cam[f0]) - 1)
            i1 = min(len(self.ts_e), self._e(self.ts_cam[f1]) + 1)
            return self.twist[i0:max(i0 + 1, i1)].mean(0).astype(np.float32)

        def body_delta(self, f_ref, f_end):
            i_r = self._d(self.ts_cam[f_ref])
            i_e = self._d(self.ts_cam[f_end])
            x, y, z, w = self.orien[i_r]
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]], dtype=float)
            return (R.T @ (self.pos[i_e] - self.pos[i_r]))[:2].astype(np.float32)


    def _select(mission, n_tr, n_st, seed=3):
        mdir = Path(CONFIG["data_root"]) / mission
        mm = _M(mdir)
        cands = []
        _max_start = len(mm.ts_cam) - 1 - (CONFIG["num_frames"] - 1) * _STRIDE
        for s in range(0, _max_start + 1, 8):
            rec = tuple(s + i * _STRIDE for i in range(CONFIG["num_frames"]))
            a, vr, vm = mm.profile(rec)
            score = a * 2.0 + (vr / (vm + 0.05))
            cands.append((score, rec, a, vr, vm))
        cands.sort(key=lambda t: t[0], reverse=True)
        tr = [c[1] for c in cands[:n_tr]]
        st = [c[1] for c in cands[-n_st:]]
        print(f"  select {mission[:8]}: cands {len(cands)} | tr score "
              f"{(cands[0][0], cands[n_tr//2][0])} | st score "
              f"{(cands[-1][0], cands[-n_st][0])}", flush=True)
        return tr, st, mm


    def _build(mission, n_tr, n_st):
        tr, st, mm = _select(mission, n_tr, n_st)
        out = {}
        for _set, recs in (("transient", tr), ("steady", st)):
            clips, yv4, yd4, yv7, yd7 = [], [], [], [], []
            for _j, rec in enumerate(recs):
                clips.append(load_clip(Path(CONFIG["data_root"]) / mission, rec,
                                       CONFIG["img_size"]))
                yv4.append(mm.vel_window(rec[8], rec[9]))
                yd4.append(mm.body_delta(rec[7], rec[9]))
                yv7.append(mm.vel_window(rec[14], rec[15]))
                yd7.append(mm.body_delta(rec[7], rec[15]))
                if (_j + 1) % 10 == 0:
                    print(f"    build {mission[:8]} {_set} {_j + 1}/{len(recs)}", flush=True)
            out[_set] = (clips, {4: np.array(yv4, np.float32), 7: np.array(yv7, np.float32)},
                         {4: np.array(yd4, np.float32), 7: np.array(yd7, np.float32)})
        return out


    _enc, _tgt = _load_frozen()
    _pred_mw = _load_pred("phase2_mw_pred1000.pt")

    print("building ETH-1 (fit set) ...", flush=True)
    _tr = _build("2024-10-01-11-29-55", 60, 40)
    print("building ICE-1 ...", flush=True)
    _v1 = _build("2024-11-18-13-48-19", 16, 16)
    print("building SPX-2 ...", flush=True)
    _v2 = _build("2024-11-02-17-18-32", 16, 16)


    def _feats(clips, source, k, b_tok=4):
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), _B):
                _sl = clips[i:i + _B]
                _b = len(_sl)
                vids = torch.stack(_sl).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if source == "teacher":
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, k]
                    elif source == "causal":
                        _mx, _my = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                                         batch_size=_b, device="cuda")
                        pr, _ = wm_forward(vids, _mx, _my, _enc, _enc, _pred_mw)
                        tok = pr[:, (k - b_tok) * _P:(k - b_tok + 1) * _P]
                    else:
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, 3]
                outs.append(tok.float().mean(dim=1))
        return torch.cat(outs)


    def _fit(X, y, dim, iters=200):
        h = nn.Linear(768, dim).to("cuda")
        o = torch.optim.AdamW(h.parameters(), lr=1e-3)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32)).to("cuda")
        for _ in range(iters):
            _p = torch.randperm(len(X), device="cuda")[:_B * 8]
            loss = F.mse_loss(h(X[_p]), yt[_p])
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
        h.eval()
        return h


    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / den) if den > 0 else 0.0


    # fit heads on ETH-1 (both sets concatenated, teacher features)
    print("fitting heads on ETH-1 teacher feats ...", flush=True)
    _Xfit = {k: torch.cat([_feats(_tr[s][0], "teacher", k) for s in ("transient", "steady")])
             for k in (4, 7)}
    _yfit_v = {k: np.concatenate([_tr[s][1][k] for s in ("transient", "steady")]) for k in (4, 7)}
    _yfit_d = {k: np.concatenate([_tr[s][2][k] for s in ("transient", "steady")]) for k in (4, 7)}
    _hv = {k: _fit(_Xfit[k], _yfit_v[k], 3) for k in (4, 7)}
    _hd = {k: _fit(_Xfit[k], _yfit_d[k], 2) for k in (4, 7)}
    print("heads fit", flush=True)

    transient_results = {}
    for _nm, _data in (("ICE-1", _v1), ("SPX-2", _v2)):
        transient_results[_nm] = {}
        for _set in ("transient", "steady"):
            _clips, _yv, _yd = _data[_set]
            transient_results[_nm][_set] = {}
            for _k in (4, 7):
                transient_results[_nm][_set][_k] = {}
                for _src in ("teacher", "causal", "copy"):
                    _X = _feats(_clips, _src, _k)
                    with torch.no_grad():
                        _ev = _hv[_k](_X).cpu().numpy()
                        _ed = _hd[_k](_X).cpu().numpy()
                    transient_results[_nm][_set][_k][_src] = (
                        float(np.sqrt(np.mean((_ev - _yv[_k]) ** 2))),
                        _corr(_ev[:, 0], _yv[_k][:, 0]),
                        float(np.sqrt(np.mean(np.sum((_ed - _yd[_k]) ** 2, axis=1)))))
                _r = transient_results[_nm][_set][_k]
                print(f"[{_nm} {_set} k={_k}] " + " | ".join(
                    f"{_s}: v {_r[_s][0]:.3f}({_r[_s][1]:.2f}) d {_r[_s][2]:.3f}m"
                    for _s in ("teacher", "causal", "copy")), flush=True)
    print("done")
    transient_ok = True

    return


@app.cell
def transient_big(
    CONFIG,
    F,
    Path,
    build_spike_models,
    load_clip,
    nn,
    np,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # A': powered transient/steady discriminator (n=60/60). Displacement (dx,dy) +
    # yaw error readouts at near/far horizons; sources teacher | causal(mw) | copy.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    def _load_pred(name):
        _, _, pred = build_spike_models("cuda")
        pred.load_state_dict(torch.load(CONFIG["ckpt_dir"] / name,
                                        map_location="cuda", weights_only=True)["pred"])
        pred.eval()
        return pred


    class _M:
        def __init__(self, mission_dir):
            import zarr
            zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
            ze = zarr.open_group(str(mission_dir / "data" / "anymal_state_state_estimator"),
                                 mode="r")
            zd = zarr.open_group(str(mission_dir / "data" / "dlio_map_odometry"), mode="r")
            self.ts_cam = np.asarray(zc["timestamp"][:])
            self.ts_e = np.asarray(ze["timestamp"][:])
            self.twist = np.asarray(ze["twist_lin"][:])
            self.ang = np.asarray(ze["twist_ang"][:])
            self.ts_d = np.asarray(zd["timestamp"][:])
            self.pos = np.asarray(zd["pose_pos"][:])
            self.orien = np.asarray(zd["pose_orien"][:])

        def _e(self, ts):
            i = int(np.searchsorted(self.ts_e, ts))
            if i >= len(self.ts_e):
                i = len(self.ts_e) - 1
            if i > 0 and abs(self.ts_e[i - 1] - ts) < abs(self.ts_e[i] - ts):
                i -= 1
            return i

        def _d(self, ts):
            i = int(np.searchsorted(self.ts_d, ts))
            if i >= len(self.ts_d):
                i = len(self.ts_d) - 1
            if i > 0 and abs(self.ts_d[i - 1] - ts) < abs(self.ts_d[i] - ts):
                i -= 1
            return i

        def profile(self, rec):
            ie = [self._e(self.ts_cam[f]) for f in rec[::2]]
            v = np.array([np.linalg.norm(self.twist[i][:2]) for i in ie])
            a = np.abs(self.ang[ie][:, 2])
            return float(a.mean()), float(v.max() - v.min()), float(v.mean())

        def vel_window(self, f0, f1):
            i0 = max(0, self._e(self.ts_cam[f0]) - 1)
            i1 = min(len(self.ts_e), self._e(self.ts_cam[f1]) + 1)
            return self.twist[i0:max(i0 + 1, i1)].mean(0).astype(np.float32)

        def dydyaw(self, f_ref, f_end):
            i_r = self._d(self.ts_cam[f_ref])
            i_e = self._d(self.ts_cam[f_end])
            def _R(q):
                x, y, z, w = q
                return np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]], dtype=float)
            Rr = _R(self.orien[i_r])
            Re = _R(self.orien[i_e])
            d = self.pos[i_e] - self.pos[i_r]
            dx, dy = (Rr.T @ d)[:2]
            dyaw = np.arctan2(Re[1, 0], Re[0, 0]) - np.arctan2(Rr[1, 0], Rr[0, 0])
            return np.array([dx, dy, dyaw], dtype=np.float32)


    def _select(mission, n_tr, n_st):
        mdir = Path(CONFIG["data_root"]) / mission
        mm = _M(mdir)
        cands = []
        _max_start = len(mm.ts_cam) - 1 - (CONFIG["num_frames"] - 1) * _STRIDE
        for _s in range(0, _max_start + 1, 6):
            rec = tuple(_s + i * _STRIDE for i in range(CONFIG["num_frames"]))
            a, vr, vm = mm.profile(rec)
            cands.append((a * 2.0 + vr / (vm + 0.05), rec))
        cands.sort(key=lambda t: t[0], reverse=True)
        print(f"  select {mission[:8]}: {len(cands)} cands | score top "
              f"{cands[0][0]:.3f} / low {cands[-1][0]:.3f}", flush=True)
        return [c[1] for c in cands[:n_tr]], [c[1] for c in cands[-n_st:]], mm


    def _build(mission, n_tr, n_st):
        tr, st, mm = _select(mission, n_tr, n_st)
        out = {}
        for _set, recs in (("transient", tr), ("steady", st)):
            clips, yv4, yd4, yv7, yd7 = [], [], [], [], []
            for _j, rec in enumerate(recs):
                clips.append(load_clip(Path(CONFIG["data_root"]) / mission, rec,
                                       CONFIG["img_size"]))
                yv4.append(mm.vel_window(rec[8], rec[9]))
                yd4.append(mm.dydyaw(rec[7], rec[9]))
                yv7.append(mm.vel_window(rec[14], rec[15]))
                yd7.append(mm.dydyaw(rec[7], rec[15]))
                if (_j + 1) % 30 == 0:
                    print(f"    build {mission[:8]} {_set} {_j + 1}/{len(recs)}", flush=True)
            out[_set] = (clips, {4: np.array(yv4, np.float32), 7: np.array(yv7, np.float32)},
                         {4: np.array(yd4, np.float32), 7: np.array(yd7, np.float32)})
        return out


    _enc, _tgt = _load_frozen()
    _pred_mw = _load_pred("phase2_mw_pred1000.pt")

    print("building ETH-1 (fit) ...", flush=True)
    _tr = _build("2024-10-01-11-29-55", 60, 60)
    print("building ICE-1 ...", flush=True)
    _v1 = _build("2024-11-18-13-48-19", 60, 60)
    print("building SPX-2 ...", flush=True)
    _v2 = _build("2024-11-02-17-18-32", 60, 60)


    def _feats(clips, source, k, b_tok=4):
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), _B):
                _sl = clips[i:i + _B]
                _b = len(_sl)
                vids = torch.stack(_sl).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if source == "teacher":
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, k]
                    elif source == "causal":
                        _mx, _my = temporal_causal_masks(8, _T, _P, CONFIG["tubelet_size"],
                                                         batch_size=_b, device="cuda")
                        pr, _ = wm_forward(vids, _mx, _my, _enc, _enc, _pred_mw)
                        tok = pr[:, (k - b_tok) * _P:(k - b_tok + 1) * _P]
                    else:
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, 3]
                outs.append(tok.float().mean(dim=1))
                if (i // _B) % 4 == 0:
                    print(f"    feats {source} k={k} {min(i + _B, len(clips))}/{len(clips)}",
                          flush=True)
        return torch.cat(outs)


    def _fit(X, y, dim, iters=250):
        h = nn.Linear(768, dim).to("cuda")
        o = torch.optim.AdamW(h.parameters(), lr=1e-3)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32)).to("cuda")
        for _ in range(iters):
            _p = torch.randperm(len(X), device="cuda")[:_B * 8]
            loss = F.mse_loss(h(X[_p]), yt[_p])
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
        h.eval()
        return h


    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / den) if den > 0 else 0.0


    print("fitting heads on ETH-1 teacher feats ...", flush=True)
    _Xfit = {k: torch.cat([_feats(_tr[s][0], "teacher", k) for s in ("transient", "steady")])
             for k in (4, 7)}
    _yfit_v = {k: np.concatenate([_tr[s][1][k] for s in ("transient", "steady")]) for k in (4, 7)}
    _yfit_d = {k: np.concatenate([_tr[s][2][k] for s in ("transient", "steady")]) for k in (4, 7)}
    _hv = {k: _fit(_Xfit[k], _yfit_v[k], 3) for k in (4, 7)}
    _hd = {k: _fit(_Xfit[k], _yfit_d[k], 3) for k in (4, 7)}
    print("heads fit", flush=True)

    transient_big_results = {}
    for _nm, _data in (("ICE-1", _v1), ("SPX-2", _v2)):
        transient_big_results[_nm] = {}
        for _set in ("transient", "steady"):
            _clips, _yv, _yd = _data[_set]
            transient_big_results[_nm][_set] = {}
            for _k in (4, 7):
                transient_big_results[_nm][_set][_k] = {}
                for _src in ("teacher", "causal", "copy"):
                    _X = _feats(_clips, _src, _k)
                    with torch.no_grad():
                        _ev = _hv[_k](_X).cpu().numpy()
                        _ed = _hd[_k](_X).cpu().numpy()
                    _vr = float(np.sqrt(np.mean(np.sum((_ev - _yv[_k]) ** 2, axis=1))))
                    _d2 = float(np.sqrt(np.mean(np.sum((_ed - _yd[_k])[:, :2] ** 2, axis=1))))
                    _yw = float(np.mean(np.abs(_ed[:, 2] - _yd[_k][:, 2]))) * 180.0 / np.pi
                    _c = _corr(_ev[:, 0], _yv[_k][:, 0])
                    transient_big_results[_nm][_set][_k][_src] = (_vr, _c, _d2, _yw)
                _r = transient_big_results[_nm][_set][_k]
                print(f"[{_nm} {_set} k={_k}] " + " | ".join(
                    f"{_s}: v {_r[_s][0]:.3f}({_r[_s][1]:.2f}) d {_r[_s][2]:.3f}m "
                    f"yaw {_r[_s][3]:.1f}deg" for _s in ("teacher", "causal", "copy")),
                    flush=True)
    print("done")
    transient_big_ok = True

    return


@app.cell
def action_train(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    torch,
):
    # B: ActionVJEPA control-signal training (frozen official encoder).
    # Per-target-window command tokens (mean cmd linear3+angular3 over the window's
    # frames) are MLP-mapped to embed dim and appended to the predictor context at
    # token indices >= 4608 (frame coordinate 8+, unused by any visual token).
    # A matched no-conditioning predictor is trained identically as control.
    # Check: val loss under TRUE future commands < SHUFFLED (wrong) commands.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4
    _STEPS = 800
    _BOUNDARIES = (6, 8, 10)
    _GRID_TOTAL = _T * _P
    _ACT_BASE = _GRID_TOTAL          # first action-token index (frame coord 8+)


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    _enc, _tgt = _load_frozen()


    def _fresh_pred():
        _, _, p = build_spike_models("cuda")
        p.train()
        return p


    def _cmd(mission_dir):
        import zarr
        g = zarr.open_group(str(mission_dir / "data" / "anymal_command_twist"), mode="r")
        ts = np.asarray(g["timestamp"][:])
        lin = np.asarray(g["linear"][:])
        ang = np.asarray(g["angular"][:])
        zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
        ts_cam = np.asarray(zc["timestamp"][:])
        return ts, lin, ang, ts_cam


    def _act_mean(ts, lin, ang, ts_cam, rec, j):
        """mean cmd over window frames [rec[2j], rec[2j+1]]."""
        t0, t1 = ts_cam[rec[2 * j]], ts_cam[rec[2 * j + 1]]
        i0 = max(0, int(np.searchsorted(ts, t0)) - 1)
        i1 = min(len(ts), int(np.searchsorted(ts, t1)) + 1)
        if i1 <= i0:
            return np.zeros(6, dtype=np.float32)
        return np.concatenate([lin[i0:i1].mean(0), ang[i0:i1].mean(0)]).astype(np.float32)


    def _pool(mission, n, with_act=True):
        mdir = Path(CONFIG["data_root"]) / mission
        ts, lin, ang, ts_cam = _cmd(mdir)
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=_STRIDE,
                                  step=CONFIG["num_frames"], max_clips=n)
        out = []
        for _k, r in enumerate(recs):
            clip = load_clip(mdir, r, CONFIG["img_size"])
            if with_act:
                act = np.stack([_act_mean(ts, lin, ang, ts_cam, r, j) for j in range(1, _T)], 0)
                out.append((clip, act.astype(np.float32)))   # act[j-1] = window j
            else:
                out.append(clip)
            if (_k + 1) % 40 == 0:
                print(f"  pool {mission[:8]} {_k + 1}/{len(recs)}", flush=True)
        return out


    print("preloading ETH-1 train + ICE-1 val (with commands) ...", flush=True)
    tr_act = _pool("2024-10-01-11-29-55", 160, with_act=True)
    va_act = _pool("2024-11-18-13-48-19", 24, with_act=True)


    class _ActMLP(nn.Module):
        def __init__(self, din=6, dout=768):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, 1024), nn.GELU(),
                                     nn.Linear(1024, dout))

        def forward(self, a):
            return self.net(a)


    def _make_batch(pool, rng, b=_B):
        idx = rng.integers(0, len(pool), size=b)
        clips = [pool[i][0] for i in idx]
        acts = np.stack([pool[i][1] for i in idx])       # (b, T-1, 6)
        return (torch.stack(clips).to("cuda"),
                torch.as_tensor(acts, dtype=torch.float32).to("cuda"))


    def _vis_masks(b_tok, b=_B):
        total = _GRID_TOTAL
        tidx = torch.arange(total, device="cuda")
        tt = tidx // _P
        ctx = tidx[tt < b_tok]
        tgt = tidx[tt >= b_tok]
        return ctx.repeat(b, 1), tgt.repeat(b, 1)


    def _act_tokens(mlp, acts, b_tok):
        """window-level actions for target tubelets j in [b_tok, T) -> [B, n, 768]."""
        nw = _T - b_tok
        a = acts[:, b_tok - 1:]                      # (B, nw, 6): j from b_tok
        return mlp(a)                                # (B, nw, 768)


    def _act_idx(b_tok, b=_B):
        nw = _T - b_tok
        idx = torch.arange(_ACT_BASE, _ACT_BASE + nw, device="cuda")
        return idx.repeat(b, 1)


    def fwd_plain(vids, mx, my, pred):
        ctx = _enc(vids, masks=[mx])
        out, _ = pred(ctx, [mx], [my], mod="video")
        return out


    def fwd_action(vids, mx, my, mx_all, pred, mlp, acts, b_tok):
        ctx = _enc(vids, masks=[mx])
        atok = _act_tokens(mlp, acts, b_tok)
        x = torch.cat([ctx, atok], dim=1)
        out, _ = pred(x, [mx_all], [my], mod="video")
        return out


    def ln_loss(pred, tgt):
        return F.smooth_l1_loss(F.layer_norm(pred.float(), (768,)),
                                F.layer_norm(tgt.float(), (768,)))


    def _teacher(vids, my):
        with torch.no_grad():
            full = _tgt(vids)
        return torch.gather(full, 1, my.unsqueeze(-1).expand(-1, -1, 768))


    def _train_plain(steps=_STEPS, seed=0, tag="plain"):
        pred = _fresh_pred()
        opt = torch.optim.AdamW(pred.parameters(), lr=CONFIG["lr"], weight_decay=0.05)
        rng = np.random.default_rng(seed)
        tr = [p for p in pred.parameters() if p.requires_grad]
        for _s in range(steps):
            _b = _BOUNDARIES[int(rng.integers(0, len(_BOUNDARIES)))]
            _b_tok = _b // 2
            _mx, _my = _vis_masks(_b_tok)
            vids, _ = _make_batch(tr_act, rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd_plain(vids, _mx, _my, pred)
                loss = ln_loss(out, _teacher(vids, _my))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tr, 1.0)
            opt.step()
            if (_s + 1) % 100 == 0:
                print(f"[{tag}] step {_s + 1} | loss {float(loss.detach()):.4f}", flush=True)
        torch.save({"pred": pred.state_dict(), "type": tag},
                   CONFIG["ckpt_dir"] / f"action_{tag}_pred{steps}.pt")
        return pred


    def _train_action(steps=_STEPS, seed=0, tag="actcond"):
        pred = _fresh_pred()
        mlp = _ActMLP().to("cuda")
        opt = torch.optim.AdamW(list(pred.parameters()) + list(mlp.parameters()),
                                lr=CONFIG["lr"], weight_decay=0.05)
        rng = np.random.default_rng(seed)
        tr = [p for p in list(pred.parameters()) + list(mlp.parameters()) if p.requires_grad]
        for _s in range(steps):
            _b = _BOUNDARIES[int(rng.integers(0, len(_BOUNDARIES)))]
            _b_tok = _b // 2
            _mx, _my = _vis_masks(_b_tok)
            vids, acts = _make_batch(tr_act, rng)
            _ai = _act_idx(_b_tok)
            _mx_all = torch.cat([_mx, _ai], dim=1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd_action(vids, _mx, _my, _mx_all, pred, mlp, acts, _b_tok)
                loss = ln_loss(out, _teacher(vids, _my))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tr, 1.0)
            opt.step()
            if (_s + 1) % 100 == 0:
                print(f"[{tag}] step {_s + 1} | loss {float(loss.detach()):.4f}", flush=True)
        torch.save({"pred": pred.state_dict(), "mlp": mlp.state_dict(), "type": tag},
                   CONFIG["ckpt_dir"] / f"action_{tag}_pred{steps}.pt")
        return pred, mlp


    print("training matched control (no conditioning) ...", flush=True)
    pred_plain = _train_plain()
    print("training ActionVJEPA (command-conditioned) ...", flush=True)
    pred_act, mlp_act = _train_action()

    # quick control-signal check on ICE-1 val
    _b_tok = 4
    _mx, _my = _vis_masks(_b_tok)
    _ai = _act_idx(_b_tok)
    _mx_all = torch.cat([_mx, _ai], dim=1)


    def _val_cos(true=True, shuffle_seed=7):
        rng = np.random.default_rng(shuffle_seed)
        coses, losses = [], []
        pred_act.eval(); mlp_act.eval()
        with torch.no_grad():
            for _ in range(3):
                vids, acts = _make_batch(va_act, rng)
                if not true:
                    acts = torch.roll(acts, shifts=1, dims=0)   # wrong-command control
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = fwd_action(vids, _mx, _my, _mx_all, pred_act, mlp_act, acts, _b_tok)
                    tg = _teacher(vids, _my)
                p = F.layer_norm(out.float(), (768,))
                t = F.layer_norm(tg.float(), (768,))
                coses.append(F.cosine_similarity(p, t, dim=-1).mean().item())
                losses.append(float(ln_loss(out, tg)))
        return float(np.mean(coses)), float(np.mean(losses))


    _c_true = _val_cos(true=True)
    _c_shuf = _val_cos(true=False)
    print(f"[control-signal check ICE-1] true-commands cos {_c_true[0]:.4f} "
          f"loss {_c_true[1]:.4f} | shuffled cos {_c_shuf[0]:.4f} loss {_c_shuf[1]:.4f} | "
          f"delta cos {(_c_true[0] - _c_shuf[0]):+.4f}")
    action_train_results = {"true": _c_true, "shuffled": _c_shuf}
    action_train_ok = True

    return


@app.cell
def action_long(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    torch,
):
    # B follow-up: is the action pathway wired but unused (short horizons), or
    # broken? (1) sensitivity: same video, true vs inverted commands -> output diff.
    # (2) retrain action + plain at STRIDE 8 (~3-6 s targets) where commands should
    # add real signal, then re-run the true-vs-shuffled control-signal check.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 8
    _STEPS = 500
    _BOUNDARIES = (6, 8, 10)
    _GRID_TOTAL = _T * _P
    _ACT_BASE = _GRID_TOTAL


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd,
 strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    def _fresh_pred():
        _, _, p = build_spike_models("cuda")
        p.train()
        return p


    class _ActMLP(nn.Module):
        def __init__(self, din=6, dout=768):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, 1024), nn.GELU(),
                                     nn.Linear(1024, dout))

        def forward(self, a):
            return self.net(a)


    def _cmd(mission_dir):
        import zarr
        g = zarr.open_group(str(mission_dir / "data" / "anymal_command_twist"), mode="r")
        ts = np.asarray(g["timestamp"][:])
        lin = np.asarray(g["linear"][:])
        ang = np.asarray(g["angular"][:])
        zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
        ts_cam = np.asarray(zc["timestamp"][:])
        return ts, lin, ang, ts_cam


    def _act_mean(ts, lin, ang, ts_cam, rec, j):
        t0, t1 = ts_cam[rec[2 * j]], ts_cam[rec[2 * j + 1]]
        i0 = max(0, int(np.searchsorted(ts, t0)) - 1)
        i1 = min(len(ts), int(np.searchsorted(ts, t1)) + 1)
        if i1 <= i0:
            return np.zeros(6, dtype=np.float32)
        return np.concatenate([lin[i0:i1].mean(0), ang[i0:i1].mean(0)]).astype(np.float32)


    def _pool(mission, n, stride):
        mdir = Path(CONFIG["data_root"]) / mission
        ts, lin, ang, ts_cam = _cmd(mdir)
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=stride,
                                  step=CONFIG["num_frames"], max_clips=n)
        out = []
        for _k, r in enumerate(recs):
            clip = load_clip(mdir, r, CONFIG["img_size"])
            act = np.stack([_act_mean(ts, lin, ang, ts_cam, r, j) for j in range(1, _T)], 0)
            out.append((clip, act.astype(np.float32)))
            if (_k + 1) % 40 == 0:
                print(f"  pool {mission[:8]} s{stride} {_k + 1}/{len(recs)}", flush=True)
        return out


    def _vis_masks(b_tok, b=_B):
        total = _GRID_TOTAL
        tidx = torch.arange(total, device="cuda")
        tt = tidx // _P
        ctx = tidx[tt < b_tok]
        tgt = tidx[tt >= b_tok]
        return ctx.repeat(b, 1), tgt.repeat(b, 1)


    def _make_batch(pool, rng):
        idx = rng.integers(0, len(pool), size=_B)
        clips = [pool[i][0] for i in idx]
        acts = np.stack([pool[i][1] for i in idx])
        return (torch.stack(clips).to("cuda"),
                torch.as_tensor(acts, dtype=torch.float32).to("cuda"))


    def _act_tokens(mlp, acts, b_tok):
        return mlp(acts[:, b_tok - 1:])


    def _teacher(vids, my):
        with torch.no_grad():
            full = _tgt(vids)
        return torch.gather(full, 1, my.unsqueeze(-1).expand(-1, -1, 768))


    def fwd_act(vids, mx, my, mx_all, pred, mlp, acts, b_tok):
        ctx = _enc(vids, masks=[mx])
        x = torch.cat([ctx, _act_tokens(mlp, acts, b_tok)], dim=1)
        out, _ = pred(x, [mx_all], [my], mod="video")
        return out


    def _ln_loss(pred, tgt):
        return F.smooth_l1_loss(F.layer_norm(pred.float(), (768,)),
                                F.layer_norm(tgt.float(), (768,)))


    def _cos_v(pred, tgt):
        p = F.layer_norm(pred.float(), (768,))
        t = F.layer_norm(tgt.float(), (768,))
        return F.cosine_similarity(p, t, dim=-1).mean().item()


    _enc, _tgt = _load_frozen()

    # ---- (1) sensitivity of the stride-4 trained action model -------------------
    print("sensitivity test on saved stride-4 action model ...", flush=True)
    _sd = torch.load(CONFIG["ckpt_dir"] / "action_actcond_pred800.pt",
                     map_location="cuda", weights_only=True)
    _pred4, _mlp4 = _fresh_pred(), _ActMLP().to("cuda")
    _pred4.load_state_dict(_sd["pred"]); _mlp4.load_state_dict(_sd["mlp"])
    _pred4.eval(); _mlp4.eval()
    va4 = _pool("2024-11-18-13-48-19", 24, 4)
    _rng4 = np.random.default_rng(3)
    _vids, _acts = _make_batch(va4, _rng4)
    _bt4 = 4
    _mx4, _my4 = _vis_masks(_bt4)
    _ai4 = torch.arange(_ACT_BASE, _ACT_BASE + (_T - _bt4), device="cuda").repeat(_B, 1)
    _mxall4 = torch.cat([_mx4, _ai4], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o_true = fwd_act(_vids, _mx4, _my4, _mxall4, _pred4, _mlp4, _acts, _bt4)
        _acts_inv = _acts.clone(); _acts_inv[:, :, :3] *= -1.0
        o_inv = fwd_act(_vids, _mx4, _my4, _mxall4, _pred4, _mlp4, _acts_inv, _bt4)
        o_zero = fwd_act(_vids, _mx4, _my4, _mxall4, _pred4, _mlp4,
                         torch.zeros_like(_acts), _bt4)
    diff_inv = float((o_true - o_inv).abs().mean())
    diff_zero = float((o_true - o_zero).abs().mean())
    print(f"sensitivity |d(true-inverted)| {diff_inv:.6f} | |d(true-zero)| {diff_zero:.6f}",
          flush=True)

    # ---- (2) long-horizon (stride 8) retrain -----------------------------------
    print("preloading stride-8 pools ...", flush=True)
    tr8 = _pool("2024-10-01-11-29-55", 140, 8)
    va8 = _pool("2024-11-18-13-48-19", 24, 8)


    def _train(pool, tag, use_actions, seed=0):
        pred = _fresh_pred()
        mlp = _ActMLP().to("cuda") if use_actions else None
        params = list(pred.parameters()) + (list(mlp.parameters()) if mlp else [])
        opt = torch.optim.AdamW(params, lr=CONFIG["lr"], weight_decay=0.05)
        tr = [p for p in params if p.requires_grad]
        rng = np.random.default_rng(seed)
        for _s in range(_STEPS):
            _b = _BOUNDARIES[int(rng.integers(0, len(_BOUNDARIES)))]
            _b_tok = _b // 2
            _mx, _my = _vis_masks(_b_tok)
            _ai = torch.arange(_ACT_BASE, _ACT_BASE + (_T - _b_tok), device="cuda").repeat(_B, 1)
            _mx_all = torch.cat([_mx, _ai], dim=1)
            vids, acts = _make_batch(pool, rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_actions:
                    out = fwd_act(vids, _mx, _my, _mx_all, pred, mlp, acts, _b_tok)
                else:
                    ctx = _enc(vids, masks=[_mx])
                    out, _ = pred(ctx, [_mx], [_my], mod="video")
                loss = _ln_loss(out, _teacher(vids, _my))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tr, 1.0)
            opt.step()
            if (_s + 1) % 100 == 0:
                print(f"[{tag}] step {_s + 1} loss {float(loss.detach()):.4f}", flush=True)
        torch.save({"pred": pred.state_dict(),
                    **({"mlp": mlp.state_dict()} if mlp else {})},
                   CONFIG["ckpt_dir"] / f"action_s8_{tag}_pred{_STEPS}.pt")
        return pred, mlp


    print("training stride-8 plain ...", flush=True)
    _p_plain, _ = _train(tr8, "plain", False)
    print("training stride-8 action ...", flush=True)
    _p_act, _m_act = _train(tr8, "actcond", True)


    def _val8(pred, mlp, use_actions, true=True, seed=11):
        rng = np.random.default_rng(seed)
        pred.eval()
        if mlp:
            mlp.eval()
        b_tok = 4
        mx, my = _vis_masks(b_tok)
        ai = torch.arange(_ACT_BASE, _ACT_BASE + (_T - b_tok), device="cuda").repeat(_B, 1)
        mx_all = torch.cat([mx, ai], dim=1)
        cs, ls = [], []
        with torch.no_grad():
            for _ in range(3):
                vids, acts = _make_batch(va8, rng)
                if not true:
                    acts = torch.roll(acts, 1, dims=0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if use_actions:
                        out = fwd_act(vids, mx, my, mx_all, pred, mlp, acts, b_tok)
                    else:
                        ctx = _enc(vids, masks=[mx])
                        out, _ = pred(ctx, [mx], [my], mod="video")
                    tg = _teacher(vids, my)
                cs.append(_cos_v(out, tg))
                ls.append(float(_ln_loss(out, tg)))
        return float(np.mean(cs)), float(np.mean(ls))


    _cp = _val8(_p_plain, None, False)
    _ct = _val8(_p_act, _m_act, True)
    _cs = _val8(_p_act, _m_act, True, true=False)
    print(f"[stride-8 ICE-1] plain cos {_cp[0]:.4f} | action true cos {_ct[0]:.4f} "
          f"| action shuffled cos {_cs[0]:.4f} | control delta {(_ct[0] - _cs[0]):+.4f}")
    action_long_results = {"plain": _cp, "act_true": _ct, "act_shuf": _cs,
                           "sens_inv": diff_inv, "sens_zero": diff_zero}
    action_long_ok = True

    return


@app.cell
def rand_vs_causal(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    random_mask,
    temporal_causal_masks,
    torch,
    wm_forward,
):
    # B': random-mask vs causal-mask predictor on the transient/far displacement
    # readout (the metric that separates models from copy-last). Trains a matched
    # RANDOM-mask predictor (50% context, LN-L1, ETH-1, stride 4), then reads out
    # teacher | causal(mw) | random | copy on far tubelet 7 of transient/steady
    # sets in held-out envs.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4
    _NCTX = 4
    _STEPS = 500


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    def _load_pred(name):
        _, _, pred = build_spike_models("cuda")
        pred.load_state_dict(torch.load(CONFIG["ckpt_dir"] / name,
                                        map_location="cuda", weights_only=True)["pred"])
        pred.eval()
        return pred


    class _M:
        def __init__(self, mission_dir):
            import zarr
            zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
            ze = zarr.open_group(str(mission_dir / "data" / "anymal_state_state_estimator"),
                                 mode="r")
            zd = zarr.open_group(str(mission_dir / "data" / "dlio_map_odometry"), mode="r")
            self.ts_cam = np.asarray(zc["timestamp"][:])
            self.ts_e = np.asarray(ze["timestamp"][:])
            self.twist = np.asarray(ze["twist_lin"][:])
            self.ang = np.asarray(ze["twist_ang"][:])
            self.ts_d = np.asarray(zd["timestamp"][:])
            self.pos = np.asarray(zd["pose_pos"][:])
            self.orien = np.asarray(zd["pose_orien"][:])

        def _e(self, ts):
            i = int(np.searchsorted(self.ts_e, ts))
            if i >= len(self.ts_e):
                i = len(self.ts_e) - 1
            if i > 0 and abs(self.ts_e[i - 1] - ts) < abs(self.ts_e[i] - ts):
                i -= 1
            return i

        def _d(self, ts):
            i = int(np.searchsorted(self.ts_d, ts))
            if i >= len(self.ts_d):
                i = len(self.ts_d) - 1
            if i > 0 and abs(self.ts_d[i - 1] - ts) < abs(self.ts_d[i] - ts):
                i -= 1
            return i

        def profile(self, rec):
            ie = [self._e(self.ts_cam[f]) for f in rec[::2]]
            v = np.array([np.linalg.norm(self.twist[i][:2]) for i in ie])
            a = np.abs(self.ang[ie][:, 2])
            return float(a.mean() * 2.0 + (v.max() - v.min()) / (v.mean() + 0.05))

        def dydyaw(self, f_ref, f_end):
            def _R(q):
                x, y, z, w = q
                return np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]], dtype=float)
            i_r, i_e = self._d(self.ts_cam[f_ref]), self._d(self.ts_cam[f_end])
            Rr, Re = _R(self.orien[i_r]), _R(self.orien[i_e])
            d = (Rr.T @ (self.pos[i_e] - self.pos[i_r]))[:2]
            dyaw = np.arctan2(Re[1, 0], Re[0, 0]) - np.arctan2(Rr[1, 0], Rr[0, 0])
            return np.array([d[0], d[1], dyaw], dtype=np.float32)


    def _sel(mission, n_tr, n_st):
        mdir = Path(CONFIG["data_root"]) / mission
        mm = _M(mdir)
        cands = []
        for _s in range(0, len(mm.ts_cam) - 1 - (15 * _STRIDE), 6):
            rec = tuple(_s + i * _STRIDE for i in range(16))
            cands.append((mm.profile(rec), rec))
        cands.sort(key=lambda t: t[0], reverse=True)
        return [c[1] for c in cands[:n_tr]], [c[1] for c in cands[-n_st:]], mm


    def _build(mission, n_tr, n_st):
        tr, st, mm = _sel(mission, n_tr, n_st)
        out = {}
        for _set, recs in (("transient", tr), ("steady", st)):
            clips, y7 = [], []
            for _j, rec in enumerate(recs):
                clips.append(load_clip(Path(CONFIG["data_root"]) / mission, rec,
                                       CONFIG["img_size"]))
                y7.append(mm.dydyaw(rec[7], rec[15]))
                if (_j + 1) % 30 == 0:
                    print(f"  build {mission[:8]} {_set} {_j + 1}/{len(recs)}", flush=True)
            out[_set] = (clips, np.array(y7, dtype=np.float32))
        return out


    _enc, _tgt = _load_frozen()
    _pred_mw = _load_pred("phase2_mw_pred1000.pt")

    # train matched random-mask predictor (50% context) on ETH-1
    print("preloading ETH-1 stride-4 pool for random training ...", flush=True)
    mdir = Path(CONFIG["data_root"]) / "2024-10-01-11-29-55"
    recs_p = build_clip_records(mdir, 16, stride=4, step=16, max_clips=120)
    tr_pool = []
    for _j, r in enumerate(recs_p):
        tr_pool.append(load_clip(mdir, r, CONFIG["img_size"]))
        if (_j + 1) % 40 == 0:
            print(f"  pool {_j + 1}/{len(recs_p)}", flush=True)

    _, _, pred_rand = build_spike_models("cuda")
    pred_rand.train()
    opt = torch.optim.AdamW(pred_rand.parameters(), lr=CONFIG["lr"], weight_decay=0.05)
    trn = [p for p in pred_rand.parameters() if p.requires_grad]
    rng = np.random.default_rng(0)
    print("training random-mask predictor ...", flush=True)
    for _s in range(_STEPS):
        idx = rng.integers(0, len(tr_pool), size=_B)
        _vids_r = torch.stack([tr_pool[i] for i in idx]).to("cuda")
        _mx, _my = random_mask(0, _T, _P, tubelet_size=2, mask_ratio=0.5,
                               seed=5000 + _s, batch_size=_B, device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = _enc(_vids_r, masks=[_mx])
            out, _ = pred_rand(ctx, [_mx], [_my], mod="video")
            with torch.no_grad():
                full = _tgt(_vids_r)
            tg = torch.gather(full, 1, _my.unsqueeze(-1).expand(-1, -1, 768))
            loss = F.smooth_l1_loss(F.layer_norm(out.float(), (768,)),
                                    F.layer_norm(tg.float(), (768,)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trn, 1.0)
        opt.step()
        if (_s + 1) % 100 == 0:
            print(f"[rand] step {_s + 1} loss {float(loss.detach()):.4f}", flush=True)
    pred_rand.eval()
    torch.save({"pred": pred_rand.state_dict(), "type": "random"},
               CONFIG["ckpt_dir"] / f"action_rand_pred{_STEPS}.pt")

    print("building probe sets ...", flush=True)
    _trs = _build("2024-10-01-11-29-55", 40, 40)
    _v1 = _build("2024-11-18-13-48-19", 40, 40)
    _v2 = _build("2024-11-02-17-18-32", 40, 40)


    def _feats(clips, source, k=7):
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), _B):
                _sl = clips[i:i + _B]
                _b = len(_sl)
                vids = torch.stack(_sl).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if source == "teacher":
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, k]
                    elif source == "causal":
                        _mx, _my = temporal_causal_masks(8, _T, _P, 2, batch_size=_b,
                                                         device="cuda")
                        pr, _ = wm_forward(vids, _mx, _my, _enc, _enc, _pred_mw)
                        tok = pr[:, (k - 4) * _P:(k - 3) * _P]
                    elif source == "random":
                        _mx, _my = random_mask(0, _T, _P, tubelet_size=2, mask_ratio=0.5,
                                               seed=777, batch_size=_b, device="cuda")
                        pr, _ = wm_forward(vids, _mx, _my, _enc, _enc, pred_rand)
                        # read tubelet-k rows from a causal query of the random model
                        _c_mx, _c_my = temporal_causal_masks(8, _T, _P, 2, batch_size=_b,
                                                             device="cuda")
                        prc, _ = wm_forward(vids, _c_mx, _c_my, _enc, _enc, pred_rand)
                        tok = prc[:, (k - 4) * _P:(k - 3) * _P]
                    else:
                        full = _enc(vids)
                        tok = full.view(vids.shape[0], _T, _P, 768)[:, 3]
                outs.append(tok.float().mean(dim=1))
                if (i // _B) % 5 == 0:
                    print(f"    feats {source} {min(i + _B, len(clips))}/{len(clips)}",
                          flush=True)
        return torch.cat(outs)


    def _fit(X, y, iters=250):
        h = nn.Linear(768, 3).to("cuda")
        o = torch.optim.AdamW(h.parameters(), lr=1e-3)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32)).to("cuda")
        for _ in range(iters):
            _p = torch.randperm(len(X), device="cuda")[:_B * 8]
            loss = F.mse_loss(h(X[_p]), yt[_p])
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
        h.eval()
        return h, yt


    print("fitting head on ETH-1 teacher features (tubelet 7) ...", flush=True)
    _Xf = torch.cat([_feats(_trs[s][0], "teacher") for s in ("transient", "steady")])
    _yf = np.concatenate([_trs[s][1] for s in ("transient", "steady")])
    _h, _ = _fit(_Xf, _yf)

    rand_causal_results = {}
    for _nm, _data in (("ICE-1", _v1), ("SPX-2", _v2)):
        rand_causal_results[_nm] = {}
        for _set in ("transient", "steady"):
            _clips, _y7 = _data[_set]
            rand_causal_results[_nm][_set] = {}
            for _src in ("teacher", "causal", "random", "copy"):
                _X = _feats(_clips, _src)
                with torch.no_grad():
                    _ed = _h(_X).cpu().numpy()
                _d2 = float(np.sqrt(np.mean(np.sum((_ed - _y7)[:, :2] ** 2, axis=1))))
                _yw = float(np.mean(np.abs(_ed[:, 2] - _y7[:, 2]))) * 180.0 / np.pi
                rand_causal_results[_nm][_set][_src] = (_d2, _yw)
            _r = rand_causal_results[_nm][_set]
            print(f"[{_nm} {_set}] " + " | ".join(
                f"{_s}: d {_r[_s][0]:.3f}m yaw {_r[_s][1]:.1f}" for _s in
                ("teacher", "causal", "random", "copy")), flush=True)
    print("done")
    rand_causal_ok = True

    return


@app.cell
def aux_action(
    CONFIG,
    F,
    Path,
    build_clip_records,
    build_spike_models,
    load_clip,
    nn,
    np,
    torch,
    vels,
):
    # C: ActionVJEPA + velocity-consistency aux head.
    # Predicted target-tubelet tokens (per window j) are mean-pooled and read out
    # to body linear velocity; supervised against observed twist_lin of that
    # window. Because velocity tracks commands (corr 0.92), this forces predicted
    # tokens to encode commanded motion -> true-vs-shuffled commands must change
    # the readout. Control-signal metric: vel-head RMSE under true vs shuffled vs
    # zero future commands on held-out ICE-1.

    _GRID = CONFIG["img_size"] // CONFIG["patch_size"]
    _P = _GRID * _GRID
    _T = CONFIG["num_frames"] // CONFIG["tubelet_size"]
    _B = CONFIG["batch_size"]
    _STRIDE = 4
    _STEPS = 500
    _BOUNDARIES = (6, 8, 10)
    _ACT_BASE = _T * _P
    _LAM = 0.5


    def _load_frozen(device="cuda"):
        enc, tgt, _ = build_spike_models(device)
        _ck = torch.load(CONFIG["ckpt_dir"] / CONFIG["vjepa_ckpt_name"],
                         map_location="cpu", weights_only=True)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in _ck["ema_encoder"].items()}
        enc.load_state_dict(_sd, strict=False)
        tgt.load_state_dict(_sd, strict=False)
        enc.eval(); tgt.eval()
        for _mm in (enc, tgt):
            for _pp in _mm.parameters():
                _pp.requires_grad_(False)
        return enc, tgt


    _enc, _tgt = _load_frozen()


    def _fresh_pred():
        _, _, p = build_spike_models("cuda")
        p.train()
        return p


    class _MLP(nn.Module):
        def __init__(self, din, dout, h=1024):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, dout))

        def forward(self, a):
            return self.net(a)


    def _ts_data(mission_dir):
        import zarr
        g = zarr.open_group(str(mission_dir / "data" / "anymal_command_twist"), mode="r")
        ts_c = np.asarray(g["timestamp"][:]); lin = np.asarray(g["linear"][:])
        ang = np.asarray(g["angular"][:])
        ze = zarr.open_group(str(mission_dir / "data" / "anymal_state_state_estimator"),
                             mode="r")
        ts_e = np.asarray(ze["timestamp"][:]); tw = np.asarray(ze["twist_lin"][:])
        zc = zarr.open_group(str(mission_dir / "data" / "hdr_front"), mode="r")
        ts_cam = np.asarray(zc["timestamp"][:])
        return ts_c, lin, ang, ts_e, tw, ts_cam


    def _win_mean(ts, arr, ts_cam, rec, j, pad=1):
        t0, t1 = ts_cam[rec[2 * j]], ts_cam[rec[2 * j + 1]]
        i0 = max(0, int(np.searchsorted(ts, t0)) - pad)
        i1 = min(len(ts), int(np.searchsorted(ts, t1)) + pad)
        if i1 <= i0:
            return None
        return arr[i0:i1].mean(0)


    def _pool(mission, n):
        mdir = Path(CONFIG["data_root"]) / mission
        ts_c, lin, ang, ts_e, tw, ts_cam = _ts_data(mdir)
        recs = build_clip_records(mdir, CONFIG["num_frames"], stride=_STRIDE,
                                  step=CONFIG["num_frames"], max_clips=n)
        out = []
        for _k, r in enumerate(recs):
            clip = load_clip(mdir, r, CONFIG["img_size"])
            acts, vels = [], []
            for _j in range(1, _T):
                a = _win_mean(ts_c, np.concatenate([lin, ang], 1), ts_cam, r, _j)
                v = _win_mean(ts_e, tw, ts_cam, r, _j)
                acts.append(a if a is not None else np.zeros(6, np.float32))
                vels.append(v if v is not None else np.zeros(3, np.float32))
            out.append((clip, np.asarray(acts, np.float32), np.asarray(vels, np.float32)))
            if (_k + 1) % 40 == 0:
                print(f"  pool {mission[:8]} {_k + 1}/{len(recs)}", flush=True)
        return out


    print("preloading ETH-1 train + ICE-1 val ...", flush=True)
    _trp = _pool("2024-10-01-11-29-55", 140)
    _vap = _pool("2024-11-18-13-48-19", 24)


    def _batch(pool, rng):
        idx = rng.integers(0, len(pool), size=_B)
        clips = [pool[i][0] for i in idx]
        acts = np.stack([pool[i][1] for i in idx])
        vels = np.stack([pool[i][2] for i in idx])
        return (torch.stack(clips).to("cuda"),
                torch.as_tensor(acts, dtype=torch.float32).to("cuda"),
                torch.as_tensor(vels, dtype=torch.float32).to("cuda"))


    def _vis_masks(b_tok, b=_B):
        tidx = torch.arange(_T * _P, device="cuda")
        tt = tidx // _P
        ctx = tidx[tt < b_tok].repeat(b, 1)
        tgt = tidx[tt >= b_tok].repeat(b, 1)
        return ctx, tgt


    def _fwd(vids, mx, my, mx_all, pred, mlp, acts, b_tok, vhead=None):
        ctx = _enc(vids, masks=[mx])
        atok = mlp(acts[:, b_tok - 1:])                     # (B, T-b_tok, 768)
        out, _ = pred(torch.cat([ctx, atok], dim=1), [mx_all], [my], mod="video")
        nw = _T - b_tok
        if vhead is not None:                                # (B, nw, 768)
            per = out.view(vids.shape[0], nw, _P, 768).mean(dim=2)
            vel = vhead(per)                                 # (B, nw, 3)
            return out, vel
        return out, None


    def _teacher(vids, my):
        with torch.no_grad():
            full = _tgt(vids)
        return torch.gather(full, 1, my.unsqueeze(-1).expand(-1, -1, 768))


    _pred = _fresh_pred()
    _mlp = _MLP(6, 768).to("cuda")
    _vhead = _MLP(768, 3, h=512).to("cuda")
    _opt = torch.optim.AdamW(list(_pred.parameters()) + list(_mlp.parameters()) +
                             list(_vhead.parameters()), lr=CONFIG["lr"], weight_decay=0.05)
    _tr = [p for p in list(_pred.parameters()) + list(_mlp.parameters()) +
           list(_vhead.parameters()) if p.requires_grad]
    _rng = np.random.default_rng(0)

    print("training aux velocity-consistent ActionVJEPA ...", flush=True)
    aux_history = []
    for _s in range(_STEPS):
        _b = _BOUNDARIES[int(_rng.integers(0, len(_BOUNDARIES)))]
        _b_tok = _b // 2
        _mx, _my = _vis_masks(_b_tok)
        _ai = torch.arange(_ACT_BASE, _ACT_BASE + (_T - _b_tok), device="cuda").repeat(_B, 1)
        _mx_all = torch.cat([_mx, _ai], dim=1)
        _vids_r, _acts_r, _vels_r = _batch(_trp, _rng)
        _opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _out_r, _vel_r = _fwd(_vids_r, _mx, _my, _mx_all, _pred, _mlp, _acts_r,
                                  _b_tok, _vhead)
            _f_loss = F.smooth_l1_loss(F.layer_norm(_out_r.float(), (768,)),
                                       F.layer_norm(_teacher(_vids_r, _my).float(), (768,)))
            _v_loss = F.mse_loss(_vel_r.float(), _vels_r[:, _b_tok - 1:])
            _loss = _f_loss + _LAM * _v_loss
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(_tr, 1.0)
        _opt.step()
        if (_s + 1) % 100 == 0:
            print(f"step {_s + 1} | feat {float(_f_loss.detach()):.4f} | "
                  f"vel {float(_v_loss.detach()):.4f}", flush=True)
            aux_history.append({"step": _s + 1, "feat": float(_f_loss.detach()),
                                "vel": float(_v_loss.detach())})

    torch.save({"pred": _pred.state_dict(), "mlp": _mlp.state_dict(),
                "vhead": _vhead.state_dict()},
               CONFIG["ckpt_dir"] / "aux_action_pred500.pt")
    print("saved aux_action_pred500.pt", flush=True)

    # ---- control-signal eval on ICE-1 ------------------------------------------
    _pred.eval(); _mlp.eval(); _vhead.eval()
    _b_tok = 4
    _mx, _my = _vis_masks(_b_tok)
    _ai = torch.arange(_ACT_BASE, _ACT_BASE + (_T - _b_tok), device="cuda").repeat(_B, 1)
    _mx_all = torch.cat([_mx, _ai], dim=1)
    _vt = vels if False else None  # placeholder
    _results = {}


    def _vel_rmse(acts_use, seed=5):
        rng = np.random.default_rng(seed)
        errs = []
        with torch.no_grad():
            for _ in range(4):
                vids, acts, vels = _batch(_vap, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, vel = _fwd(vids, _mx, _my, _mx_all, _pred, _mlp,
                                  acts_use(acts), _b_tok, _vhead)
                errs.append(float(F.mse_loss(vel.float(), vels[:, 3:]).sqrt()))
        return float(np.mean(errs))


    def _feat_cos(acts_use, seed=6):
        rng = np.random.default_rng(seed)
        cs = []
        with torch.no_grad():
            for _ in range(4):
                vids, acts, _ = _batch(_vap, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out, _ = _fwd(vids, _mx, _my, _mx_all, _pred, _mlp,
                                  acts_use(acts), _b_tok)
                    tg = _teacher(vids, _my)
                p = F.layer_norm(out.float(), (768,))
                t = F.layer_norm(tg.float(), (768,))
                cs.append(F.cosine_similarity(p, t, dim=-1).mean().item())
        return float(np.mean(cs))


    true_v = _vel_rmse(lambda a: a)
    shuf_v = _vel_rmse(lambda a: torch.roll(a, 1, dims=0), seed=7)
    zero_v = _vel_rmse(lambda a: torch.zeros_like(a), seed=8)
    true_c = _feat_cos(lambda a: a)
    shuf_c = _feat_cos(lambda a: torch.roll(a, 1, dims=0))
    print(f"[ICE-1 control-signal] velRMSE true {true_v:.4f} | shuffled {shuf_v:.4f} | "
          f"zero {zero_v:.4f} | feat cos true {true_c:.4f} | shuffled {shuf_c:.4f}")
    aux_action_results = {"vel_rmse_true": true_v, "vel_rmse_shuf": shuf_v,
                          "vel_rmse_zero": zero_v, "feat_cos_true": true_c,
                          "feat_cos_shuf": shuf_c, "history": aux_history}
    aux_action_ok = True

    return


if __name__ == "__main__":
    app.run()
