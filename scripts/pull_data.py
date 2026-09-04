"""Download Grand Tour missions (camera zarrs + DLIO odom + JPEGs) from HuggingFace.

Usage:
    python scripts/pull_data.py --dataset-folder data/grandtour \
        --missions-csv configs/missions_split.csv [--missions TS1 TS2 ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vjepa_nav.data.grandtour import pull_mission_topics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-folder", type=Path, default="data/grandtour")
    ap.add_argument("--missions-csv", type=Path, default="configs/missions_split.csv")
    ap.add_argument("--missions", nargs="*", default=[])
    args = ap.parse_args()

    rows = []
    with open(args.missions_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r["Timestamp"])

    missions = args.missions or rows
    pull_mission_topics(missions, args.dataset_folder)
    print("done.")


if __name__ == "__main__":
    main()
