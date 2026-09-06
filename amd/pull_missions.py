"""Idempotent Grand Tour (ANYmal-D) data puller for train/val/all missions.

Pulls, per mission: images/hdr_front* (includes hdr_front_mask when present),
data/hdr_front, data/anymal_command_twist, data/anymal_state_state_estimator,
and (optional) data/dlio_map_odometry.

Usage:
    python amd/pull_missions.py core          # ETH-1..3 + SPX-1 + ICE-1 + SPX-2
    python amd/pull_missions.py all          # core + remaining 42 missions
    python amd/pull_missions.py 2024-10-01-11-29-55   # any timestamps
"""
import sys
from pathlib import Path

ROOT = Path("/marimo/grandtour") if Path("/marimo").exists() else Path("data") / "grandtour"
CORE = ["2024-10-01-11-29-55", "2024-10-01-11-47-44", "2024-10-01-12-00-49",
        "2024-11-02-17-10-25", "2024-11-18-13-48-19", "2024-11-02-17-18-32"]
ADD = [
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
PATTERNS = ["images/hdr_front*", "data/hdr_front*",
            "data/anymal_command_twist*", "data/anymal_state_state_estimator*"]


def ready(m):
    d = ROOT / m
    return ((d / "images" / "hdr_front").exists()
            and (d / "data" / "anymal_command_twist").exists()
            and (d / "data" / "anymal_state_state_estimator").exists())


def pull(missions):
    from huggingface_hub import snapshot_download
    import tarfile
    for m in missions:
        if ready(m):
            print("already present:", m, flush=True)
            continue
        cache = Path(snapshot_download("leggedrobotics/grand_tour_dataset",
                                       allow_patterns=[f"{m}/{p}" for p in PATTERNS],
                                       repo_type="dataset"))
        for tar in cache.rglob("*.tar"):
            rel = tar.relative_to(cache)
            dst = ROOT / rel.parent
            dst.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar) as tf:
                tf.extractall(path=dst)
        n = len(list((ROOT / m / "images" / "hdr_front").glob("*.jpeg")))
        print(f"pulled {m} ready={ready(m)} frames={n}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "core"
    if which == "core":
        pull(CORE)
    elif which == "all":
        pull(CORE + ADD)
    else:
        pull(sys.argv[1:])
