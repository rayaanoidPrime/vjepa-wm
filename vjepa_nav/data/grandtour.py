"""Grand Tour dataset access.

Downloads the minimal set of topics per mission from HuggingFace
(`leggedrobotics/grand_tour_dataset`) and reads the camera timestamps,
DLIO odometry and JPEG frames needed to build goal-conditioned clips.

Layout on HuggingFace (mirrors the LiMo data recipe):
    <mission>/data/hdr_front.tar          -> data/hdr_front/   zarr group (timestamp)
    <mission>/data/dlio_map_odometry.tar  -> data/dlio_map_odometry/ zarr (timestamp, pose_pos, pose_orien)
    <mission>/images/hdr_front.tar        -> images/hdr_front/ <frame_id:06d>.jpeg
"""

from __future__ import annotations

import re
import shutil
import tarfile
from itertools import product
from pathlib import Path

import numpy as np
import zarr

HF_REPO_ID = "leggedrobotics/grand_tour_dataset"
HF_REVISION_MAIN = "main"

# Which mission topics we need. "hdr_front" matches both data/ and images/ tars.
DATA_TOPICS = ["hdr_front", "dlio_map_odometry"]


def _patterns_to_regex(patterns: list[str]) -> re.Pattern:
    parts = []
    for p in patterns:
        p = re.escape(p).replace(r"\*", ".*").replace(r"\?", ".")
        parts.append(f"^{p}$")
    return re.compile("|".join(parts))


def _topic_exists(mission_dir: Path, topic: str) -> bool:
    return (mission_dir / "data" / topic).exists() or (
        mission_dir / "images" / topic
    ).exists()


def _extract_to_folder(cache: Path, dest: Path, allow_patterns: list[str]) -> None:
    regex = _patterns_to_regex(allow_patterns)
    files = [f for f in cache.rglob("*") if regex.match(str(f.relative_to(cache)))]
    tar_files = [f for f in files if f.suffix == ".tar"]
    other_files = [f for f in files if f.suffix != ".tar" and f.is_file()]

    for src in tar_files:
        dst_parent = dest / src.relative_to(cache).parent
        dst_parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(src) as tar:
            tar.extractall(path=dst_parent)

    for src in other_files:
        dst = dest / src.relative_to(cache)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def pull_mission_topics(
    missions: list[str],
    dataset_folder: Path,
    topics: list[str] | None = None,
    revision: str = HF_REVISION_MAIN,
    skip_existing: bool = True,
) -> Path:
    """Download the required zarr topics + JPEGs for each mission.

    Only downloads what is missing (skip_existing=True), so re-running is cheap.
    """
    topics = topics or DATA_TOPICS
    dataset_folder = Path(dataset_folder)
    allow_patterns: list[str] = []
    for mission, topic in product(missions, topics):
        mission_dir = dataset_folder / mission
        if skip_existing and _topic_exists(mission_dir, topic):
            continue
        allow_patterns.append(f"{mission}/*{topic}*")

    if not allow_patterns:
        return dataset_folder

    from huggingface_hub import snapshot_download

    hf_cache = snapshot_download(
        repo_id=HF_REPO_ID,
        revision=revision,
        allow_patterns=allow_patterns,
        repo_type="dataset",
    )
    _extract_to_folder(Path(hf_cache), dataset_folder, allow_patterns)
    return dataset_folder


# --------------------------------------------------------------------------- #


def _quat_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _nearest_idx(timestamps: np.ndarray, t: float) -> int:
    idx = np.searchsorted(timestamps, t)
    if idx == 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    return idx - 1 if (t - timestamps[idx - 1]) <= (timestamps[idx] - t) else idx


class MissionSource:
    """Reads the camera timestamps, DLIO odometry and JPEG frames of one mission."""

    def __init__(self, mission_dir: Path) -> None:
        self.mission_dir = Path(mission_dir)

        self._z_cam = zarr.open_group(
            str(self.mission_dir / "data" / "hdr_front"), mode="r"
        )
        self._z_dlio = zarr.open_group(
            str(self.mission_dir / "data" / "dlio_map_odometry"), mode="r"
        )
        self._cam_ts = np.array(self._z_cam["timestamp"])
        self._dlio_ts = np.array(self._z_dlio["timestamp"])
        self._dlio_pos = np.array(self._z_dlio["pose_pos"])  # (M, 3)
        self._dlio_orien = np.array(self._z_dlio["pose_orien"])  # (M, 4) xyzw

    def __len__(self) -> int:
        return len(self._cam_ts)

    def timestamp(self, i: int) -> float:
        return float(self._cam_ts[i])

    def frame_path(self, i: int) -> Path:
        return self.mission_dir / "images" / "hdr_front" / f"{i:06d}.jpeg"

    def nearest_frame_idx(self, t: float) -> int:
        return _nearest_idx(self._cam_ts, t)

    def odom_pose_world(self, i: int) -> np.ndarray:
        """SE(2) pose [x, y, yaw] in the world (dlio_map) frame at camera frame i."""
        d_idx = _nearest_idx(self._dlio_ts, self.timestamp(i))
        pos = self._dlio_pos[d_idx]
        orien = self._dlio_orien[d_idx]
        R = _quat_to_matrix(orien)
        # DLIO body frame is rotated +90deg (CCW) vs ANYmal base_link.
        yaw = float(np.arctan2(R[1, 0], R[0, 0])) - np.pi / 2
        return np.array([pos[0], pos[1], yaw], dtype=np.float64)

    def trajectory_world(self, i: int, duration: float, n: int) -> np.ndarray:
        """(n, 3) SE(2) poses in world frame over [t_i, t_i + duration] (nearest odom)."""
        t0 = self.timestamp(i)
        times = np.linspace(t0, t0 + duration, n)
        idxs = np.searchsorted(self._dlio_ts, times).clip(0, len(self._dlio_ts) - 1)
        prev = (idxs - 1).clip(0)
        use_prev = np.abs(self._dlio_ts[prev] - times) < np.abs(
            self._dlio_ts[idxs] - times
        )
        idxs = np.where(use_prev, prev, idxs)
        pos = self._dlio_pos[idxs]
        orien = self._dlio_orien[idxs]
        x, y, z, w = orien[:, 0], orien[:, 1], orien[:, 2], orien[:, 3]
        yaw = np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z)) - np.pi / 2
        return np.column_stack([pos[:, 0], pos[:, 1], yaw]).astype(np.float64)

    def odom_covers(self, t: float) -> bool:
        return len(self._dlio_ts) > 0 and self._dlio_ts[-1] >= t
