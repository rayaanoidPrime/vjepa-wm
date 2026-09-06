#!/usr/bin/env python3
"""Download a subset of GrandTour missions from HuggingFace (Zarr path).

Pulls, per mission, the topics used by the JEPA-WM pipeline:
  images/hdr_front (frames), data/hdr_front (camera timestamps),
  data/anymal_command_twist (commanded twist), data/anymal_state_actuator (joints),
  data/dlio_map_odometry (base pose).

Usage:
  python download_missions.py --missions <ts> [<ts> ...]
  # default output dir: $GT_RAW (see _paths.py); override with --raw-dir
"""
import argparse
import tarfile
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

import _paths

PATTERNS = [
    "images/hdr_front*",
    "data/hdr_front*",
    "data/anymal_command_twist*",
    "data/anymal_state_actuator*",
    "data/dlio_map_odometry*",
]
REPO = "leggedrobotics/grand_tour_dataset"


def mission_ready(raw: Path, mission: str) -> bool:
    d = raw / mission
    return (
        (d / "images" / "hdr_front").exists()
        and (d / "data" / "anymal_command_twist").exists()
        and (d / "data" / "anymal_state_actuator").exists()
    )


def download(raw: Path, missions):
    raw.mkdir(parents=True, exist_ok=True)
    for m in missions:
        if mission_ready(raw, m):
            n = len(list((raw / m / "images" / "hdr_front").glob("*.jp*")))
            print(f"already present: {m} ({n} frames)")
            continue
        print(f"downloading {m} ...")
        cache = Path(snapshot_download(REPO, allow_patterns=[f"{m}/{p}" for p in PATTERNS],
                                       repo_type="dataset"))
        for tar in cache.rglob("*.tar"):
            rel = tar.relative_to(cache)
            dst = raw / rel.parent
            dst.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar) as tf:
                tf.extractall(path=dst)
        n = len(list((raw / m / "images" / "hdr_front").glob("*.jp*")))
        print(f"done {m}: {n} frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", nargs="+", required=True,
                    help="mission timestamp dirs, e.g. 2024-10-01-11-29-55")
    ap.add_argument("--raw-dir", type=Path, default=_paths.GT_RAW,
                    help=f"output root (default: {_paths.GT_RAW})")
    args = ap.parse_args()
    download(args.raw_dir, args.missions)


if __name__ == "__main__":
    main()
