"""Checkpoint handling: download, load pretrained V-JEPA 2.1, save/resume."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch

VJEPA2_1_VITL_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"


def clean_keys(state_dict: dict) -> dict:
    """Strip 'module.' and 'backbone.' prefixes (torch.hub / DDP artifacts)."""
    out = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "backbone."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    """Download a file with HTTP range resume."""
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        # Simple existence check; range-resume handles partial files below.
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    header = {}
    if tmp.exists():
        header["Range"] = f"bytes={tmp.stat().st_size}-"

    req = urllib.request.Request(url, headers=header)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "ab") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def load_pretrained_vjepa2_1(model, ckpt_path: Path) -> dict:
    """Load the released ViT-L/16 384 checkpoint into a GoalVJEPA model.

    Loads checkpoint['ema_encoder'] into encoder + target_encoder, and
    checkpoint['predictor'] into the predictor. strict=False so our resized
    predictor head (1024 vs 1664) and any extra/missing keys are tolerated.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def _load(into: torch.nn.Module, raw: dict) -> list[str]:
        sd = clean_keys(raw)
        own = into.state_dict()
        filtered = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        missing, unexpected = into.load_state_dict(filtered, strict=False)
        return list(missing), list(unexpected)

    enc_key = next((k for k in ("ema_encoder", "encoder", "target_encoder") if k in ckpt), None)
    if enc_key is None:
        raise KeyError(f"no encoder key in checkpoint; have {list(ckpt.keys())[:10]}")
    m_enc, u_enc = _load(model.encoder, ckpt[enc_key])
    m_tgt, u_tgt = _load(model.target_encoder, ckpt[enc_key])
    m_pred, u_pred = _load(model.predictor, ckpt.get("predictor", {}))

    log = {
        "encoder_missing": m_enc, "encoder_unexpected": u_enc,
        "target_encoder_missing": m_tgt,
        "predictor_missing": m_pred, "predictor_unexpected": u_pred,
    }
    return log


def save_checkpoint(state: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def find_latest(ckpt_dir: Path, prefix: str = "step") -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob(f"{prefix}_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return cands[-1] if cands else None


def upload_to_hf(local_path: Path, repo_id: str, token: str | None, remote_path: str | None = None) -> None:
    """Upload a checkpoint to a HuggingFace repo (use for molab persistence)."""
    from huggingface_hub import upload_file

    remote_path = remote_path or local_path.name
    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
