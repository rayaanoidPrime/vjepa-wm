"""Build the goal-conditioned clip dataset from Grand Tour missions.

For every usable camera frame t0 we record one clip:
    start frame (t0), K future frames evenly spaced over [t0, t0+horizon],
    the SE(2) goal (= future pose at t0+horizon, robot frame) and the
    50-waypoint trajectory (LiMo D_TEL style, fixed horizon).

Output zarr groups (one per split):
    <dataset_folder>/clips/<split>.zarr
      start_id   (N,)      int64   index of the start frame (into images/hdr_front/)
      future_ids (N, K)    int64   indices of the K future frames
      goal       (N, 3)    float32 SE(2) goal in robot base frame (x, y, yaw)
      path       (N, 50,3) float32 waypoints in robot base frame
      goal_time  (N,)      float32 fixed horizon
      mission    (N,)      str     mission timestamp
      split      (N,)      str     train/val/test

Usage:
    python scripts/build_clips.py \
      --dataset-folder data/grandtour \
      --clips-folder data/clips \
      --missions-csv configs/missions_split.csv \
      --horizon 5.0 --num-future 8 --num-waypoints 50 \
      --splits train --splits val --splits test \
      --pull 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vjepa_nav.data.grandtour import MissionSource, pull_mission_topics
from vjepa_nav.data.se2 import transform_se2_odom_to_base, convert_se2_to_transform


def load_missions_csv(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["Timestamp"], r["Split"]))
    return rows


def build_split(
    missions: list[tuple[str, str]],
    split: str,
    dataset_folder: Path,
    clips_folder: Path,
    horizon: float,
    num_future: int,
    num_waypoints: int,
    min_displacement: float,
    skip_first_frames: int,
) -> int:
    clips_folder.mkdir(parents=True, exist_ok=True)
    out = clips_folder / f"{split}.zarr"
    g = zarr.open_group(str(out), mode="w")
    st_id, ft_id, gl, pt, gt, ms = [], [], [], [], [], []

    sel_missions = [m for m, s in missions if s == split]
    for mission in sel_missions:
        src = MissionSource(dataset_folder / mission)
        n_frames = len(src)
        print(f"[{mission}] {n_frames} camera frames")
        for i in tqdm(range(skip_first_frames, n_frames), desc=mission):
            t0 = src.timestamp(i)
            t_end = t0 + horizon
            if i + num_future >= n_frames or not src.odom_covers(t_end):
                break  # not enough future footage left in this mission

            # Future frame indices (nearest camera frames to the target times).
            f_ids = [src.nearest_frame_idx(t0 + (k + 1) * horizon / num_future) for k in range(num_future)]
            if len(set(f_ids)) < num_future:
                continue  # degenerate (e.g. duplicated timestamps)

            # Goal + waypoint trajectory in the robot base frame (LiMo D_TEL).
            traj_world = src.trajectory_world(i, duration=horizon, n=num_waypoints)
            pose_world = src.odom_pose_world(i)
            path_base = transform_se2_odom_to_base(
                traj_world, convert_se2_to_transform(pose_world)
            )
            goal_base = path_base[-1]
            if np.linalg.norm(path_base[0, :2] - path_base[-1, :2]) < min_displacement:
                continue  # near-stationary

            st_id.append(i)
            ft_id.append(f_ids)
            gl.append(goal_base)
            pt.append(path_base)
            gt.append(horizon)
            ms.append(mission)

    n = len(st_id)
    if n == 0:
        print(f"[{split}] no samples")
        return 0

    def _add(name, data, chunks):
        # Assignment-based creation works on zarr v2 and v3.
        try:
            g.create_dataset(name, data=data, chunks=chunks)
        except AttributeError:  # zarr v3
            g[name] = data

    _add("start_id", np.asarray(st_id, dtype=np.int64), (4096,))
    _add("future_ids", np.asarray(ft_id, dtype=np.int64), (1024, num_future))
    _add("goal", np.asarray(gl, dtype=np.float32), (4096, 3))
    _add("path", np.asarray(pt, dtype=np.float32), (1024, num_waypoints, 3))
    _add("goal_time", np.asarray(gt, dtype=np.float32), (4096,))
    _add("mission", np.asarray(ms, dtype="S32"), (4096,))
    g.attrs["split"] = split
    g.attrs["horizon"] = horizon
    g.attrs["num_future"] = num_future
    g.attrs["num_waypoints"] = num_waypoints
    print(f"[{split}] wrote {n} samples -> {out}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-folder", type=Path, default="data/grandtour")
    ap.add_argument("--clips-folder", type=Path, default="data/clips")
    ap.add_argument("--missions-csv", type=Path, default="configs/missions_split.csv")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--num-future", type=int, default=8)
    ap.add_argument("--num-waypoints", type=int, default=50)
    ap.add_argument("--min-displacement", type=float, default=0.25)
    ap.add_argument("--skip-first-frames", type=int, default=1000)
    ap.add_argument("--pull", type=int, default=1, help="download missing data from HF")
    ap.add_argument("--missions", nargs="*", default=[], help="restrict to specific missions")
    args = ap.parse_args()

    missions = load_missions_csv(args.missions_csv)
    if args.missions:
        allowed = set(args.missions)
        missions = [m for m in missions if m[0] in allowed]

    if args.pull:
        selected = [m for m, _ in missions]
        print(f"Pulling data for {len(selected)} missions ...")
        pull_mission_topics(selected, args.dataset_folder)

    total = 0
    for split in args.splits:
        total += build_split(
            missions,
            split,
            args.dataset_folder,
            args.clips_folder,
            horizon=args.horizon,
            num_future=args.num_future,
            num_waypoints=args.num_waypoints,
            min_displacement=args.min_displacement,
            skip_first_frames=args.skip_first_frames,
        )
    print(f"Total samples: {total}")


if __name__ == "__main__":
    main()
