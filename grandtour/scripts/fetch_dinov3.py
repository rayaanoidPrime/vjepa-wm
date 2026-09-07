#!/usr/bin/env python3
"""Fetch the DINOv3 ViT-L/16 (LVD-1689M) backbone used by the Path A configs.

facebookresearch/dinov3 ships its weights on a *gated* Hugging Face repo. This
helper downloads the same checkpoint from the public mirror
``PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m`` (identical file, canonical hash
``8aa4cbdd``) and aliases it to the exact basename that
``jepa-wms/app/plan_common/models/dino.py`` expects
(``dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth``), so no engine patch is needed.

Usage (after ``source grandtour/env.sh``):
  python grandtour/scripts/fetch_dinov3.py            # writes to $JEPAWM_OSSCKPT/dinov3/

Env:
  DV3_REPO  override the HF mirror repo (default PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m)
  HF_TOKEN  only needed if you point DV3_REPO at the gated facebook/dinov3
"""
import os
import sys
import shutil
from pathlib import Path

import _paths


def main() -> None:
    from huggingface_hub import hf_hub_download

    repo = os.environ.get("DV3_REPO", "PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    root = Path(os.environ.get("JEPAWM_OSSCKPT", str(_paths.JEPAWM_OSSCKPT)))
    out_dir = root / "dinov3"
    out_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo, repo_type="model")
    cands = [f for f in files if "vitl16" in f and "lvd1689m" in f and f.endswith(".pth")]
    print(f"{repo}: candidates = {cands}")
    if not cands:
        sys.exit("no dinov3_vitl16 lvd1689m .pth found in the repo")
    name = sorted(cands)[0]
    local = hf_hub_download(repo, name, repo_type="model", token=token)
    expected = out_dir / "dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"
    if os.path.abspath(local) != os.path.abspath(expected):
        shutil.copyfile(local, expected)
    print(f"downloaded {local}\naliased to {expected} ({expected.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
