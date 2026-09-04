"""Dataset + transforms for the goal-conditioned latent world model.

Each sample is: one start frame, K future frames, and the SE(2) goal.
Frames are read as JPEGs from the mission folders and resized to the model
input resolution on the fly (matches LiMo's on-disk layout, keeps the clip
zarr tiny).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import Dataset


def make_frame_transform(resolution: int = 256, train: bool = True):
    import torchvision.transforms as T

    resize = int(resolution / 224 * 256)  # ~292 for 256
    if train:
        # Mild geometric augmentation, no horizontal flip (flipping inverts the
        # goal's lateral axis y).
        return T.Compose(
            [
                T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomCrop(resolution),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ClipDataset(Dataset):
    """Reads clips built by scripts/build_clips.py."""

    def __init__(
        self,
        clips_folder: Path,
        dataset_folder: Path,
        split: str = "train",
        resolution: int = 256,
        train: bool = True,
        num_future: int | None = None,
    ) -> None:
        self.split = split
        self.dataset_folder = Path(dataset_folder)
        self.resolution = resolution
        self.num_future = num_future
        self.transform = make_frame_transform(resolution, train=train)

        z = zarr.open_group(str(Path(clips_folder) / f"{split}.zarr"), mode="r")
        self.start_id = np.asarray(z["start_id"])
        self.future_ids = np.asarray(z["future_ids"])
        self.goal = np.asarray(z["goal"])
        self.path = np.asarray(z["path"])
        if "mission" in z:
            raw = z["mission"][:]
            self.mission = [m.decode() if isinstance(m, bytes) else str(m) for m in raw]
        else:
            self.mission = None
        if self.num_future is None:
            self.num_future = self.future_ids.shape[1]
        else:
            self.future_ids = self.future_ids[:, : self.num_future]

        self._missions_dir_cache: dict[str, Path] = {}

    def __len__(self) -> int:
        return len(self.start_id)

    def _mission_dir(self, name: str) -> Path:
        d = self._missions_dir_cache.get(name)
        if d is None:
            d = self.dataset_folder / name
            self._missions_dir_cache[name] = d
        return d

    def _load_frame(self, mission: str, idx: int) -> torch.Tensor:
        p = self._mission_dir(mission) / "images" / "hdr_front" / f"{int(idx):06d}.jpeg"
        img = Image.open(p).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mission = self.mission[idx] if self.mission is not None else "unknown"
        start = self._load_frame(mission, int(self.start_id[idx]))
        future = torch.stack(
            [self._load_frame(mission, int(i)) for i in self.future_ids[idx]]
        )  # (K, 3, H, W)
        goal = torch.tensor(self.goal[idx], dtype=torch.float32)  # (3,)
        path = torch.tensor(self.path[idx], dtype=torch.float32)  # (50, 3)
        return {
            "start": start,          # (3, H, W)
            "future": future,        # (K, 3, H, W)
            "goal": goal,            # (3,)
            "path": path,            # (50, 3)
            "mission": mission,
            "idx": idx,
        }


def collate_clips(batch: list[dict]) -> dict:
    """Collate into (B, C, 1, H, W) clips for the V-JEPA image path."""
    start = torch.stack([b["start"] for b in batch]).unsqueeze(2)  # (B, 3, 1, H, W)
    future = torch.stack([b["future"] for b in batch]).unsqueeze(2)  # (B, K, 3, 1, H, W)
    return {
        "start": start,
        "future": future,
        "goal": torch.stack([b["goal"] for b in batch]),
        "path": torch.stack([b["path"] for b in batch]),
        "mission": [b["mission"] for b in batch],
        "idx": torch.tensor([b["idx"] for b in batch]),
    }
