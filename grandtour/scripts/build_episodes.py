#!/usr/bin/env python3
"""Convert downloaded GrandTour missions into jepa-wms episode files.

Layout (mirrors franka_custom <dset>/<name>/data/<task>/<run>/episode.h5):
  $JEPAWM_DSET/grand_tour/data/<mission>/episode.h5   (one h5 per mission)

Episode contents (schema documented in docs/DATA_SCHEMA.md):
  attrs: camera, n_frames, joint_ids
  frames_bytes    uint8  concatenated JPEGs (resized to short side 512)
  frames_offsets  int64  [N+1] byte offsets of each frame
  cam_timestamp   float64 [N] camera seconds
  cmd_twist       float32 [N,3]  [vx, vy, yaw_rate] commanded twist
  joints          float32 [N,36] 12 joints x (position, velocity, torque)
  pose            float32 [N,7]  base pose xyz + quat(xyzw) (dlio odom)

Usage:
  python build_episodes.py [--missions <ts> ...]   (default: all missions under $GT_RAW)
"""
import argparse
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from PIL import Image

import _paths

JOINT_IDS = [f"{i:02d}" for i in range(12)]
SHORT = 512          # resize short side (enough margin for 224-crop at scale >= 0.5)
JPEGQ = 85


def nearest_idx(ref_ts, query_ts):
    i = np.searchsorted(ref_ts, query_ts)
    i = np.clip(i, 0, len(ref_ts) - 1)
    lo = np.maximum(i - 1, 0)
    take = np.where(np.abs(ref_ts[lo] - query_ts) <= np.abs(ref_ts[i] - query_ts), lo, i)
    return take


def build_mission(raw: Path, dset: Path, mission: str) -> str:
    base = raw / mission
    out = dset / "grand_tour" / "data" / mission
    out.mkdir(parents=True, exist_ok=True)
    ep_path = out / "episode.h5"
    if ep_path.exists():
        return f"skip (exists): {mission}"

    cam_g = zarr.open_group(str(base / "data" / "hdr_front"), mode="r")
    t_cam = np.asarray(cam_g["timestamp"][:])
    jpgs = sorted((base / "images" / "hdr_front").glob("*.jp*"))
    assert len(jpgs) == len(t_cam), f"{mission}: {len(jpgs)} jpegs vs {len(t_cam)} camera ts"

    # commanded twist at camera times
    cmd_g = zarr.open_group(str(base / "data" / "anymal_command_twist"), mode="r")
    t_cmd = np.asarray(cmd_g["timestamp"][:])
    cmd = np.concatenate([np.asarray(cmd_g["linear"][:]), np.asarray(cmd_g["angular"][:])], 1)
    ic = nearest_idx(t_cmd, t_cam)
    cmd_f = cmd[ic].astype(np.float64)

    # joints (12 x pos/vel/tor) at camera times
    act_g = zarr.open_group(str(base / "data" / "anymal_state_actuator"), mode="r")
    t_act = np.asarray(act_g["timestamp"][:])
    ia = nearest_idx(t_act, t_cam)
    cols = []
    for j in JOINT_IDS:
        cols += [np.asarray(act_g[f"{j}_state_joint_position"][:])[ia],
                 np.asarray(act_g[f"{j}_state_joint_velocity"][:])[ia],
                 np.asarray(act_g[f"{j}_state_joint_torque"][:])[ia]]
    joints = np.stack(cols, 1).astype(np.float64)

    # base pose (dlio) at camera times
    od_g = zarr.open_group(str(base / "data" / "dlio_map_odometry"), mode="r")
    t_od = np.asarray(od_g["timestamp"][:])
    io_ = nearest_idx(t_od, t_cam)
    pose = np.concatenate([np.asarray(od_g["pose_pos"][:])[io_],
                           np.asarray(od_g["pose_orien"][:])[io_]], 1).astype(np.float64)

    # frames: resize to short side SHORT, re-encode jpeg, concat bytes
    offsets = np.zeros(len(t_cam) + 1, np.int64)
    blobs, total = [], 0
    for n, jp in enumerate(jpgs):
        im = Image.open(jp).convert("RGB")
        w, h = im.size
        if min(w, h) > SHORT:
            s = SHORT / min(w, h)
            im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEGQ)
        b = buf.getvalue()
        blobs.append(b)
        total += len(b)
        offsets[n + 1] = total
    print(f"{mission}: {len(t_cam)} frames, blob {total/1e6:.1f} MB", flush=True)

    with h5py.File(ep_path, "w") as h5:
        h5.attrs["camera"] = _paths.CAMERA
        h5.attrs["n_frames"] = len(t_cam)
        h5.attrs["joint_ids"] = json.dumps(JOINT_IDS)
        h5.create_dataset("frames_bytes", data=np.frombuffer(b"".join(blobs), np.uint8),
                          chunks=(1 << 20,))
        h5.create_dataset("frames_offsets", data=offsets)
        h5.create_dataset("cam_timestamp", data=t_cam)
        h5.create_dataset("cmd_twist", data=cmd_f[:, [0, 1, 5]])   # [vx, vy, yaw_rate]
        h5.create_dataset("cmd_raw", data=cmd_f)
        h5.create_dataset("joints", data=joints)
        h5.create_dataset("pose", data=pose)
    return f"wrote {ep_path}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", nargs="*", default=None,
                    help="mission timestamp dirs (default: all under --raw-dir)")
    ap.add_argument("--raw-dir", type=Path, default=_paths.GT_RAW,
                    help=f"downloaded missions root (default: {_paths.GT_RAW})")
    ap.add_argument("--dset", type=Path, default=_paths.JEPAWM_DSET,
                    help=f"JEPAWM_DSET root (default: {_paths.JEPAWM_DSET})")
    args = ap.parse_args()

    if args.missions:
        missions = args.missions
    else:
        missions = sorted(p.name for p in args.raw_dir.iterdir() if p.is_dir())
    if not missions:
        sys.exit("no missions found under --raw-dir")
    for m in missions:
        try:
            print(build_mission(args.raw_dir, args.dset, m))
        except Exception as e:
            print(f"FAILED {m}: {e}", flush=True)
    print("BUILD DONE")


if __name__ == "__main__":
    main()
