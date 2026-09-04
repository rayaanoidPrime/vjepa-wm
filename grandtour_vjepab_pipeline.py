# ============================================================
# V-JEPA 2.1 ViT-B/384 + GRAND TOUR END-TO-END PIPELINE
# ============================================================
#
# Faithful port of the "V-JEPA 2.1 + CEAR END-TO-END PIPELINE"
# marimo cell, with the CEAR dataset swapped for the Grand Tour
# dataset (hdr_front RGB). Same task: fine-tune the V-JEPA 2.1
# ViT-B/384 encoder with a masked-latent JEPA objective, then
# predict the latent of the frame 8 steps ahead from the current
# frame's latent and decode it to a 384x384 RGB frame.
#
# Changes vs the CEAR cell (all config-level):
#   * dataset: Grand Tour hdr_front (downloaded from HuggingFace)
#   * batch size: 4 (fits the RTX Pro 6000 / 96 GB)
#   * max clips per mission: 300 (keeps the run tractable)
#   * resumable: re-running the cell skips completed stages
# No goal conditioning, no diffusion decoder.
#
# PASTE THIS WHOLE FILE INTO A SINGLE MARIMO CELL AND RUN IT.
# Requires: the V-JEPA repo at VJEPA_ROOT (default /marimo/vjepa2),
# torch, torchvision, numpy, Pillow, huggingface_hub, matplotlib.
# ============================================================

import os
import sys
import math
import copy
import random
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
# ---- ENVIRONMENT / PATHS ----------------------------------------------------
# Auto-detect a writable base directory for data + checkpoints so the same cell
# runs on molab (/marimo), Lightning AI Studio, Colab (/content) or a local
# machine. Override any path with an environment variable if you need to.
BASE_DIR = None
for _cand in (Path("/marimo"), Path("/content"), Path.home() / "vjepa_grandtour"):
    try:
        _cand.mkdir(parents=True, exist_ok=True)
        BASE_DIR = _cand
        break
    except OSError:
        continue
if BASE_DIR is None:
    raise RuntimeError(
        "Could not find a writable base directory for data/checkpoints. "
        "Set VJEPA_ROOT / GRANDTOUR_DATA_ROOT / CHECKPOINT_DIR via env vars."
    )

VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", str(BASE_DIR / "vjepa2")))
DATA_ROOT = Path(os.environ.get("GRANDTOUR_DATA_ROOT", str(BASE_DIR / "grandtour")))
CHECKPOINT_DIR = Path(
    os.environ.get("CHECKPOINT_DIR", str(BASE_DIR / "vjepa2_grandtour_checkpoints"))
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[paths] base={BASE_DIR}")
print(f"[paths] VJEPA_ROOT={VJEPA_ROOT}")
print(f"[paths] DATA_ROOT={DATA_ROOT}")
print(f"[paths] CHECKPOINT_DIR={CHECKPOINT_DIR}")

# Grand Tour (HuggingFace: leggedrobotics/grand_tour_dataset)
HF_REPO_ID = "leggedrobotics/grand_tour_dataset"

# All 48 Grand Tour missions that have hdr_front footage.
MISSIONS = [
    "2024-10-01-11-29-55", "2024-10-01-11-47-44", "2024-10-01-12-00-49",
    "2024-11-02-17-10-25", "2024-11-02-17-18-32", "2024-11-02-17-43-10",
    "2024-11-02-21-12-51", "2024-11-03-07-52-45", "2024-11-03-07-57-34",
    "2024-11-03-08-17-23", "2024-11-03-13-51-43", "2024-11-03-13-59-54",
    "2024-11-04-10-57-34", "2024-11-04-12-55-59", "2024-11-04-13-07-13",
    "2024-11-04-16-05-00", "2024-11-11-12-07-40", "2024-11-11-12-42-47",
    "2024-11-11-14-29-44", "2024-11-11-16-14-23", "2024-11-14-11-17-02",
    "2024-11-14-12-01-26", "2024-11-14-13-45-37", "2024-11-14-14-36-02",
    "2024-11-14-15-22-43", "2024-11-14-16-04-09", "2024-11-15-10-16-35",
    "2024-11-15-11-18-14", "2024-11-15-11-37-15", "2024-11-15-12-06-03",
    "2024-11-15-14-14-12", "2024-11-15-14-43-52", "2024-11-15-16-41-14",
    "2024-11-18-12-05-01", "2024-11-18-13-22-14", "2024-11-18-13-48-19",
    "2024-11-18-15-46-05", "2024-11-18-16-59-23", "2024-11-18-17-13-09",
    "2024-11-18-17-31-36", "2024-11-25-14-57-08", "2024-11-25-16-36-19",
    "2024-12-03-13-15-38", "2024-12-03-13-26-40", "2024-12-09-09-34-43",
    "2024-12-09-09-41-46", "2024-12-09-11-28-28", "2024-12-09-11-53-11",
]
# Set a subset (e.g. MISSIONS[:3]) for a quick smoke test.
MAX_CLIPS_PER_MISSION = 300  # cap clips per mission (None = use all)

# Official public V-JEPA 2.1 ViT-B/384 checkpoint.
VJEPA_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
)
VJEPA_CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
VJEPA_CHECKPOINT = VJEPA_CACHE / "vjepa2_1_vitb_dist_vitG_384.pt"

NUM_FRAMES = 64
TEMPORAL_STRIDE = 1
CLIP_STEP = 32
IMAGE_SIZE = 384
PATCH_SIZE = 16
TUBELET_SIZE = 2
EMBED_DIM = 768

# Stage 1
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-5
STAGE1_WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.996
MASK_RATIO = 0.75

# Stage 2
STAGE2_EPOCHS = 20
FUTURE_FRAME_OFFSET = 8
STAGE2_PRED_LR = 1e-4
STAGE2_DECODER_LR = 2e-4
STAGE2_WEIGHT_DECAY = 1e-4
# Cap optimizer steps per Stage-2 epoch (None = full pass over the training set).
# An epoch is otherwise len(train_loader) steps, i.e. num_train_clips // BATCH_SIZE.
STAGE2_STEPS_PER_EPOCH = None
# Gradient accumulation for Stage 2. Keep a large *effective* batch
# (BATCH_SIZE * STAGE2_GRAD_ACCUM) while limiting peak VRAM to a single
# micro-batch. E.g. BATCH_SIZE=16 + STAGE2_GRAD_ACCUM=4 == effective 64 at ~1/4
# the memory. With >1 you must call optimizer2.zero_grad(set_to_none=True) before
# the epoch (handled below). 1 = normal behaviour (step every micro-batch).
STAGE2_GRAD_ACCUM = 1

# Precompute Stage-2 latents ONCE (the encoder is frozen and clip transforms are
# deterministic, so latents are identical every run). This removes the encoder
# forward and the 64-frame clips from the Stage-2 loop: ~2-4x faster steps and
# ~4x less VRAM. Cost: one encoding pass over all clips (~30-60 min) and ~21 GB
# of fp16 latents on disk (CHECKPOINT_DIR). Set False to keep the on-the-fly
# encoder behaviour (e.g. if disk space is very tight).
PRECOMPUTE_STAGE2_LATENTS = True
STAGE2_PRECOMPUTE_BATCH = 8
STAGE2_LATENTS_CURRENT = CHECKPOINT_DIR / "stage2_latents_current.npy"
STAGE2_LATENTS_FUTURE = CHECKPOINT_DIR / "stage2_latents_future.npy"

BATCH_SIZE = 4
NUM_WORKERS = 0        # keep 0 for marimo; raise only if running as a plain script
GRAD_CLIP = 1.0
VAL_FRACTION = 0.20

# ---- Load the best Stage-1 encoder from Google Drive (optional) ------------
# Set ONE of these to fetch the Stage-1 best checkpoint before the skip/resume
# logic runs. Either way it is copied to CHECKPOINT_DIR, so the rest of the
# pipeline (skip Stage 1, load best, freeze) works unchanged.
#   STAGE1_BEST_DRIVE_PATH : filesystem path if your Drive is readable from the
#                            kernel (e.g. a molab remote-storage connection, or a
#                            file you downloaded locally).
#   STAGE1_BEST_GDRIVE_ID  : Google Drive file ID (downloads with gdown). Get it
#                            from a share link: https://drive.google.com/file/d/<ID>/view
STAGE1_BEST_DRIVE_PATH = None
STAGE1_BEST_GDRIVE_ID = None

# Resume: re-running this cell skips already-completed stages.
#   CONTINUE_STAGE1=False: if a stage1 best checkpoint exists, skip Stage 1 and
#                          use the best encoder (default).
#   CONTINUE_STAGE1=True : resume Stage 1 training from the latest
#                          stage1_epoch_XX.pth and train until STAGE1_EPOCHS.
CONTINUE_STAGE1 = False
RESUME_STAGE2 = True   # continue the latest Stage-2 epoch checkpoint if present

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE / AMP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
use_bf16 = use_amp and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
use_grad_scaler = use_amp and not use_bf16

print("=" * 72)
print("V-JEPA 2.1 ViT-B/384 + GRAND TOUR COMPLETE PIPELINE")
print("=" * 72)
print("Device:", device)
if use_amp:
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory: %.2f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print("AMP dtype:", amp_dtype)
    print("BF16 supported:", use_bf16)
else:
    print("WARNING: CUDA is not available. This pipeline is intended for GPU training.")

# ============================================================
# BASIC VALIDATION + V-JEPA REPO
# ============================================================
# Auto-clone the V-JEPA 2.1 codebase if it isn't present yet.
if not VJEPA_ROOT.exists():
    _git = shutil.which("git")
    if _git is None:
        raise FileNotFoundError(
            f"V-JEPA repository not found at {VJEPA_ROOT} and 'git' is not "
            "available on this machine. Clone it yourself and set the "
            "VJEPA_ROOT env var."
        )
    print(f"[setup] cloning facebookresearch/vjepa2 -> {VJEPA_ROOT} ...")
    VJEPA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_git, "clone", "--depth", "1",
         "https://github.com/facebookresearch/vjepa2", str(VJEPA_ROOT)],
        check=True,
    )
    print("[setup] clone complete.")

# The V-JEPA model code needs timm + einops.
for _pkg in ("timm", "einops"):
    try:
        __import__(_pkg)
    except ImportError:
        print(f"[setup] installing {_pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)

if IMAGE_SIZE % PATCH_SIZE != 0:
    raise ValueError("IMAGE_SIZE must be divisible by PATCH_SIZE.")
if NUM_FRAMES % TUBELET_SIZE != 0:
    raise ValueError("NUM_FRAMES must be divisible by TUBELET_SIZE.")
if not (0 < FUTURE_FRAME_OFFSET < NUM_FRAMES):
    raise ValueError("FUTURE_FRAME_OFFSET must be between 1 and NUM_FRAMES-1.")

os.chdir(VJEPA_ROOT)
if str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))

from app.vjepa_2_1.models import vision_transformer as vjepa_vit
from app.vjepa_2_1.models.predictor import vit_predictor

# ============================================================
# GRAND TOUR DATA PULL (hdr_front JPEGs only)
# ============================================================
def _mission_has_images(mission):
    d = DATA_ROOT / mission / "images" / "hdr_front"
    return d.exists() and any(d.glob("*.jpeg"))


def pull_grandtour_hdr_front(missions):
    """Download <mission>/images/hdr_front.tar for each missing mission."""
    missing = [m for m in missions if not _mission_has_images(m)]
    if not missing:
        print("All Grand Tour hdr_front data already present.")
        return

    from huggingface_hub import snapshot_download

    allow_patterns = [f"{m}/images/hdr_front*" for m in missing]
    print(f"Downloading hdr_front for {len(missing)} mission(s) ...")
    cache = Path(
        snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=allow_patterns,
            repo_type="dataset",
        )
    )

    for tar in cache.rglob("*.tar"):
        rel = tar.relative_to(cache)          # <mission>/images/hdr_front.tar
        dst_parent = DATA_ROOT / rel.parent
        dst_parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as tf:
            tf.extractall(path=dst_parent)
        print(f"extracted {rel}")
    print("Grand Tour data pull complete.")


pull_grandtour_hdr_front(MISSIONS)

# ============================================================
# GRAND TOUR RGB SEQUENCE DISCOVERY
# ============================================================
def find_rgb_sequences(root, missions):
    """Map mission -> sorted list of hdr_front frame paths (>= NUM_FRAMES)."""
    groups = {}
    for m in missions:
        d = root / m / "images" / "hdr_front"
        if not d.exists():
            continue
        files = []
        for p in d.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                files.append(p)
        files.sort(key=lambda p: p.name)
        if len(files) >= NUM_FRAMES:
            groups[m] = files
    return groups


sequence_frames = find_rgb_sequences(DATA_ROOT, MISSIONS)

if not sequence_frames:
    raise RuntimeError(
        f"No Grand Tour mission with at least {NUM_FRAMES} hdr_front frames found under {DATA_ROOT}."
    )

print("=" * 72)
print("GRAND TOUR RGB DATA")
print("=" * 72)
print("RGB sequences (missions):", len(sequence_frames))
for seq, frames in sorted(sequence_frames.items()):
    print(f"{seq}: {len(frames)} RGB frames")

# ============================================================
# BUILD 64-FRAME CLIPS
# ============================================================
clip_records = []

for seq, frames in sorted(sequence_frames.items()):
    max_start = len(frames) - 1 - (NUM_FRAMES - 1) * TEMPORAL_STRIDE
    records = []
    for start in range(0, max_start + 1, CLIP_STEP):
        indices = [start + i * TEMPORAL_STRIDE for i in range(NUM_FRAMES)]
        records.append((seq, frames, indices))

    if MAX_CLIPS_PER_MISSION and len(records) > MAX_CLIPS_PER_MISSION:
        step = math.ceil(len(records) / MAX_CLIPS_PER_MISSION)
        records = records[::step][:MAX_CLIPS_PER_MISSION]

    clip_records.extend(records)

if not clip_records:
    raise RuntimeError("No 64-frame clips could be constructed.")

print("=" * 72)
print("TEMPORAL CLIPS")
print("=" * 72)
print("Frames per clip:", NUM_FRAMES)
print("Temporal stride:", TEMPORAL_STRIDE)
print("Clip step:", CLIP_STEP)
print("Max clips per mission:", MAX_CLIPS_PER_MISSION)
print("Number of clips:", len(clip_records))

# ============================================================
# DATASET
# ============================================================
resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
to_tensor = transforms.ToTensor()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    tensor = to_tensor(image)
    tensor = resize(tensor)
    return tensor


class GrandTourClipDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        _, frames, indices = self.records[index]

        # [T, C, H, W]
        raw = torch.stack([load_rgb(frames[i]) for i in indices], dim=0)
        # [T, C, H, W] -> [C, T, H, W]
        video = raw.permute(1, 0, 2, 3).contiguous()
        video = (video - IMAGENET_MEAN) / IMAGENET_STD
        return video


full_dataset = GrandTourClipDataset(clip_records)

# Deterministic 80/20 train/validation split over clips (seed 42).
num_total = len(full_dataset)
num_val = max(1, int(round(num_total * VAL_FRACTION)))
num_train = num_total - num_val

split_generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(
    full_dataset, [num_train, num_val], generator=split_generator
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=use_amp, drop_last=False,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=use_amp, drop_last=False,
)

sample = full_dataset[0]
expected_shape = (3, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE)
assert tuple(sample.shape) == expected_shape, (
    f"Unexpected clip shape {tuple(sample.shape)}; expected {expected_shape}"
)

print("=" * 72)
print("V-JEPA 2.1 INPUT DATA")
print("=" * 72)
print("Total clips:", num_total)
print("Training clips:", num_train)
print("Validation clips:", num_val)
print("Single clip:", tuple(sample.shape))
print("Batch:", (BATCH_SIZE,) + tuple(sample.shape))

# ============================================================
# CHECKPOINT DOWNLOAD / LOAD
# ============================================================
def ensure_vjepa_checkpoint():
    VJEPA_CACHE.mkdir(parents=True, exist_ok=True)

    if VJEPA_CHECKPOINT.exists() and VJEPA_CHECKPOINT.stat().st_size > 500_000_000:
        print("V-JEPA checkpoint already cached: %.2f GB" % (
            VJEPA_CHECKPOINT.stat().st_size / 1024**3
        ))
        return

    if VJEPA_CHECKPOINT.exists():
        print("Removing incomplete V-JEPA checkpoint:", VJEPA_CHECKPOINT)
        VJEPA_CHECKPOINT.unlink()

    print("DOWNLOADING OFFICIAL V-JEPA 2.1 CHECKPOINT")
    torch.hub.download_url_to_file(VJEPA_CHECKPOINT_URL, str(VJEPA_CHECKPOINT), progress=True)

    if VJEPA_CHECKPOINT.stat().st_size <= 500_000_000:
        raise RuntimeError("V-JEPA checkpoint download appears incomplete.")
    print("Checkpoint downloaded: %.2f GB" % (VJEPA_CHECKPOINT.stat().st_size / 1024**3))


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


ensure_vjepa_checkpoint()

# ============================================================
# BUILD V-JEPA 2.1 ViT-B/384 ENCODER + TARGET (EMA)
# ============================================================
encoder = vjepa_vit.vit_base(
    patch_size=PATCH_SIZE,
    img_size=(IMAGE_SIZE, IMAGE_SIZE),
    num_frames=NUM_FRAMES,
    tubelet_size=TUBELET_SIZE,
    use_sdpa=True,
    use_SiLU=False,
    wide_SiLU=True,
    uniform_power=False,
    use_rope=True,
    img_temporal_dim_size=1,
    interpolate_rope=True,
)

try:
    checkpoint = torch.load(VJEPA_CHECKPOINT, map_location="cpu", weights_only=True)
except Exception as exc:
    print("weights_only=True could not read checkpoint:", repr(exc))
    print("Retrying checkpoint load with weights_only=False...")
    checkpoint = torch.load(VJEPA_CHECKPOINT, map_location="cpu", weights_only=False)

if not isinstance(checkpoint, dict):
    raise RuntimeError("Unexpected V-JEPA checkpoint format.")
if "ema_encoder" not in checkpoint:
    raise KeyError(
        "Official V-JEPA 2.1 checkpoint does not contain 'ema_encoder'. "
        f"Available keys: {list(checkpoint.keys())}"
    )

encoder.load_state_dict(clean_state_dict(checkpoint["ema_encoder"]), strict=True)
del checkpoint

encoder = encoder.to(device)

target_encoder = copy.deepcopy(encoder).to(device)
for parameter in target_encoder.parameters():
    parameter.requires_grad_(False)
target_encoder.eval()

print("=" * 72)
print("V-JEPA 2.1 ViT-B/384 BUILT")
print("=" * 72)
print("Encoder embedding dimension:", encoder.embed_dim)

# ============================================================
# V-JEPA 2.1 PREDICTOR (768-D output, fresh init)
# ============================================================
jepa_predictor = vit_predictor(
    img_size=(IMAGE_SIZE, IMAGE_SIZE),
    patch_size=PATCH_SIZE,
    use_mask_tokens=True,
    embed_dim=EMBED_DIM,
    predictor_embed_dim=384,
    out_embed_dim=EMBED_DIM,
    num_frames=NUM_FRAMES,
    tubelet_size=TUBELET_SIZE,
    depth=12,
    num_heads=12,
    num_mask_tokens=8,
    use_rope=True,
    uniform_power=False,
    use_sdpa=True,
    use_silu=False,
    wide_silu=True,
    n_output_distillation=1,
    return_all_tokens=False,
    img_temporal_dim_size=1,
    interpolate_rope=True,
).to(device)

print("Predictor output dimension:", EMBED_DIM)

# ============================================================
# TOKEN GEOMETRY
# ============================================================
TEMPORAL_TOKENS = NUM_FRAMES // TUBELET_SIZE
SPATIAL_TOKENS = IMAGE_SIZE // PATCH_SIZE
TOTAL_TOKENS = TEMPORAL_TOKENS * SPATIAL_TOKENS * SPATIAL_TOKENS

print("=" * 72)
print("TOKEN GEOMETRY")
print("=" * 72)
print("Temporal tokens:", TEMPORAL_TOKENS)
print("Spatial tokens:", SPATIAL_TOKENS, "x", SPATIAL_TOKENS)
print("Total tokens:", TOTAL_TOKENS)

# ============================================================
# MASK GENERATION
# ============================================================
def make_masks(batch_size, device, mask_ratio=MASK_RATIO):
    """Return one context-mask list and one target-mask list per batch."""
    n_target = int(round(TOTAL_TOKENS * mask_ratio))
    n_target = max(1, min(TOTAL_TOKENS - 1, n_target))
    n_context = TOTAL_TOKENS - n_target

    masks_x = []
    masks_y = []
    for _ in range(batch_size):
        permutation = torch.randperm(TOTAL_TOKENS, device=device)
        masks_x.append(permutation[:n_context])
        masks_y.append(permutation[n_context:])

    return [torch.stack(masks_x, dim=0).contiguous()], [
        torch.stack(masks_y, dim=0).contiguous()
    ]

# ============================================================
# AMP HELPERS
# ============================================================
def autocast_context():
    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)


scaler = torch.cuda.amp.GradScaler(enabled=True) if use_grad_scaler else None


def optimizer_step(loss, optimizer, parameters):
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        optimizer.step()

# ============================================================
# STAGE 1 LOSS / EMA
# ============================================================
def jepa_loss(prediction, target):
    prediction = F.layer_norm(prediction, (prediction.shape[-1],))
    target = F.layer_norm(target, (target.shape[-1],))
    return F.smooth_l1_loss(prediction, target)


@torch.no_grad()
def update_ema(online, target, momentum=EMA_MOMENTUM):
    for online_parameter, target_parameter in zip(online.parameters(), target.parameters()):
        target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)

# ============================================================
# STAGE 1: V-JEPA 2.1 GRAND TOUR FINE-TUNING
# ============================================================
STAGE1_BEST = CHECKPOINT_DIR / "vjepa21_grandtour_stage1_best.pth"


def ensure_stage1_best_from_drive():
    """Copy/download the best Stage-1 checkpoint from Google Drive into place.

    Returns the source ("local", "drive", "gdrive") or None if unavailable.
    """
    if STAGE1_BEST.exists():
        return "local"

    if STAGE1_BEST_DRIVE_PATH is not None:
        src = Path(STAGE1_BEST_DRIVE_PATH)
        if src.exists():
            import shutil

            shutil.copy2(src, STAGE1_BEST)
            print(f"[Stage 1] copied best checkpoint from Drive: {src}")
            return "drive"

    if STAGE1_BEST_GDRIVE_ID is not None:
        import subprocess
        import sys

        try:
            import gdown
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True
            )
            import gdown
        gdown.download(
            id=STAGE1_BEST_GDRIVE_ID, output=str(STAGE1_BEST), quiet=False
        )
        if STAGE1_BEST.exists() and STAGE1_BEST.stat().st_size > 0:
            print(f"[Stage 1] downloaded best checkpoint from Google Drive: {STAGE1_BEST}")
            return "gdrive"
    return None


_best_src = ensure_stage1_best_from_drive()
print("[Stage 1] best-checkpoint source:", _best_src or "none (Stage 1 will be trained)")


def latest_stage1_epoch():
    epoch = 0
    for p in CHECKPOINT_DIR.glob("vjepa21_grandtour_stage1_epoch_*.pth"):
        try:
            epoch = max(epoch, int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return epoch


SKIP_STAGE1 = (not CONTINUE_STAGE1) and STAGE1_BEST.exists()

print("=" * 72)
print("STAGE 1: V-JEPA 2.1 GRAND TOUR FINE-TUNING")
print("=" * 72)
print("Epochs:", STAGE1_EPOCHS)
print("Mask ratio:", MASK_RATIO)
print("Learning rate:", STAGE1_LR)
print("CONTINUE_STAGE1:", CONTINUE_STAGE1)
print("Stage-1 best exists:", STAGE1_BEST.exists(), "| skip stage 1:", SKIP_STAGE1)

start_epoch1 = 0
best_val_loss = float("inf")

if not SKIP_STAGE1:
    optimizer1 = torch.optim.AdamW(
        list(encoder.parameters()) + list(jepa_predictor.parameters()),
        lr=STAGE1_LR,
        weight_decay=STAGE1_WEIGHT_DECAY,
    )

    if CONTINUE_STAGE1:
        start_epoch1 = latest_stage1_epoch()
        if start_epoch1 > 0:
            ck = CHECKPOINT_DIR / f"vjepa21_grandtour_stage1_epoch_{start_epoch1:02d}.pth"
            sd = torch.load(ck, map_location=device, weights_only=True)
            encoder.load_state_dict(sd["encoder"], strict=True)
            target_encoder.load_state_dict(sd["target_encoder"], strict=True)
            jepa_predictor.load_state_dict(sd["jepa_predictor"], strict=True)
            optimizer1.load_state_dict(sd["optimizer"])
            best_val_loss = float(sd.get("val_loss", float("inf")))
            print(f"[Stage 1 resume] loaded {ck} (epoch {start_epoch1}, val {best_val_loss:.6f})")

    for epoch in range(start_epoch1, STAGE1_EPOCHS):
        encoder.train()
        jepa_predictor.train()
        target_encoder.eval()

        running_loss = 0.0

        for step, video in enumerate(train_loader):
            video = video.to(device, dtype=torch.float32, non_blocking=True)

            masks_x, masks_y = make_masks(video.shape[0], device)
            optimizer1.zero_grad(set_to_none=True)

            with autocast_context():
                with torch.no_grad():
                    target_tokens = target_encoder(video, masks=masks_y)
                context_tokens = encoder(video, masks=masks_x)

                predicted_target, _ = jepa_predictor(
                    context_tokens, masks_x, masks_y, mod="video"
                )

                if predicted_target.shape != target_tokens.shape:
                    raise RuntimeError(
                        "Stage 1 tensor mismatch: "
                        f"prediction={tuple(predicted_target.shape)}, "
                        f"target={tuple(target_tokens.shape)}"
                    )

                loss = jepa_loss(predicted_target, target_tokens)

            optimizer_step(
                loss,
                optimizer1,
                list(encoder.parameters()) + list(jepa_predictor.parameters()),
            )
            update_ema(encoder, target_encoder)
            running_loss += float(loss.detach().item())

            if step % 10 == 0:
                print(
                    f"Stage1 Epoch {epoch+1:02d}/{STAGE1_EPOCHS} | "
                    f"Step {step+1:03d}/{len(train_loader)} | Loss {loss.item():.6f}"
                )

        train_loss = running_loss / max(1, len(train_loader))

        encoder.eval()
        jepa_predictor.eval()
        target_encoder.eval()
        val_running = 0.0

        with torch.no_grad():
            for video in val_loader:
                video = video.to(device, dtype=torch.float32, non_blocking=True)
                masks_x, masks_y = make_masks(video.shape[0], device)
                with autocast_context():
                    target_tokens = target_encoder(video, masks=masks_y)
                    context_tokens = encoder(video, masks=masks_x)
                    predicted_target, _ = jepa_predictor(
                        context_tokens, masks_x, masks_y, mod="video"
                    )
                    val_loss = jepa_loss(predicted_target, target_tokens)
                val_running += float(val_loss.item())

        val_loss_value = val_running / max(1, len(val_loader))

        print(
            f"Stage1 Epoch {epoch+1:02d} complete | "
            f"Train {train_loss:.6f} | Val {val_loss_value:.6f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "jepa_predictor": jepa_predictor.state_dict(),
                "optimizer": optimizer1.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss_value,
            },
            CHECKPOINT_DIR / f"vjepa21_grandtour_stage1_epoch_{epoch+1:02d}.pth",
        )

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder": encoder.state_dict(),
                    "target_encoder": target_encoder.state_dict(),
                    "jepa_predictor": jepa_predictor.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss_value,
                },
                STAGE1_BEST,
            )

# Load the best Stage-1 encoder and freeze it.
# Falls back to the latest stage1_epoch_XX.pth if the best file is missing or
# was corrupted by an interrupt in the middle of torch.save.
def _load_stage1_state():
    if STAGE1_BEST.exists():
        try:
            return torch.load(STAGE1_BEST, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] stage1_best unreadable ({exc}); falling back to latest epoch checkpoint.")
    ck = CHECKPOINT_DIR / f"vjepa21_grandtour_stage1_epoch_{latest_stage1_epoch():02d}.pth"
    if not ck.exists():
        raise FileNotFoundError(f"No Stage-1 checkpoint found: {STAGE1_BEST} or {ck}")
    print(f"[Stage 1] using latest epoch checkpoint: {ck}")
    return torch.load(ck, map_location="cpu", weights_only=True)


best_stage1 = _load_stage1_state()
encoder.load_state_dict(best_stage1["encoder"], strict=True)
target_encoder.load_state_dict(best_stage1["target_encoder"], strict=True)
_ep = best_stage1.get("epoch")
_vl = best_stage1.get("val_loss")
print(
    "[Stage 1] loaded checkpoint from epoch %s (val_loss %s)"
    % (_ep, ("%.6f" % _vl) if _vl is not None else "?")
)
del best_stage1

for parameter in encoder.parameters():
    parameter.requires_grad_(False)
encoder.eval()
target_encoder.eval()

print("=" * 72)
print("STAGE 1 COMPLETE - BEST ENCODER RESTORED AND FROZEN")
print("=" * 72)

# ============================================================
# STAGE 2: FUTURE SPATIAL LATENT PREDICTOR
# ============================================================
class FutureSpatialPredictor(nn.Module):
    """Residual convolutional predictor over the 24x24 V-JEPA latent grid."""

    def __init__(self, dim=EMBED_DIM, hidden=768, depth=6):
        super().__init__()
        self.in_proj = nn.Conv2d(dim, hidden, kernel_size=1)
        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    nn.GroupNorm(32, hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    nn.GroupNorm(32, hidden),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out_proj(x)


class SharpRGBDecoder(nn.Module):
    """24x24 -> 384x384 RGB decoder with progressive upsampling."""

    def __init__(self, latent_dim=EMBED_DIM):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_dim, 512, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
        )
        self.up1 = self._up(512, 256)
        self.up2 = self._up(256, 128)
        self.up3 = self._up(128, 64)
        self.up4 = self._up(64, 32)
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    @staticmethod
    def _up(cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(32, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, latent):
        x = self.in_proj(latent)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.refine(x)
        return torch.sigmoid(x)


future_predictor = FutureSpatialPredictor().to(device)
decoder = SharpRGBDecoder().to(device)

optimizer2 = torch.optim.AdamW(
    [
        {"params": future_predictor.parameters(), "lr": STAGE2_PRED_LR},
        {"params": decoder.parameters(), "lr": STAGE2_DECODER_LR},
    ],
    weight_decay=STAGE2_WEIGHT_DECAY,
)

# ============================================================
# STAGE 2 HELPERS
# ============================================================
def denormalize(video):
    mean = IMAGENET_MEAN.to(video.device)
    std = IMAGENET_STD.to(video.device)
    # IMAGENET_MEAN/STD are (3,1,1,1) for the (C,T,H,W) clip layout. For images
    # shaped (B,C,H,W) or (C,H,W) the channel dim must align to dim 1 / dim 0.
    if video.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    return (video * std + mean).clamp(0.0, 1.0)


def tokens_to_spatial(tokens, temporal_index):
    """[B, 32*24*24, 768] -> [B, 768, 24, 24]."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor [B,N,D], got {tuple(tokens.shape)}")
    b, n, d = tokens.shape
    x = tokens.view(b, TEMPORAL_TOKENS, SPATIAL_TOKENS, SPATIAL_TOKENS, EMBED_DIM)
    x = x[:, temporal_index]
    return x.permute(0, 3, 1, 2).contiguous()


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred - target).pow(2) + eps**2).mean()


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))


def laplacian_loss(pred, target):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=pred.device, dtype=pred.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(pred.shape[1], 1, 1, 1)
    pred_lap = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    target_lap = F.conv2d(target, kernel, padding=1, groups=target.shape[1])
    return F.l1_loss(pred_lap, target_lap)


def ssim_value_loss(pred, target):
    c1 = 0.01**2
    c2 = 0.03**2
    mu1 = F.avg_pool2d(pred, 11, stride=1, padding=5)
    mu2 = F.avg_pool2d(target, 11, stride=1, padding=5)
    sigma1 = F.avg_pool2d(pred * pred, 11, 1, 5) - mu1 * mu1
    sigma2 = F.avg_pool2d(target * target, 11, 1, 5) - mu2 * mu2
    sigma12 = F.avg_pool2d(pred * target, 11, 1, 5) - mu1 * mu2
    numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return 1.0 - ssim_map.clamp(0.0, 1.0).mean()


def rgb_reconstruction_loss(pred, target):
    return (
        0.50 * charbonnier_loss(pred, target)
        + 0.20 * gradient_loss(pred, target)
        + 0.15 * laplacian_loss(pred, target)
        + 0.15 * ssim_value_loss(pred, target)
    )


future_token_index = FUTURE_FRAME_OFFSET // TUBELET_SIZE
if future_token_index >= TEMPORAL_TOKENS:
    raise RuntimeError("Future token index is outside the temporal token grid.")

print("=" * 72)
print("STAGE 2: FUTURE LATENT -> RGB")
print("=" * 72)
print("Current frame index:", 0)
print("Future frame index:", FUTURE_FRAME_OFFSET)
print("Future temporal token:", future_token_index)
print("Epochs:", STAGE2_EPOCHS)
print("Latent resolution:", f"{SPATIAL_TOKENS}x{SPATIAL_TOKENS}")
print("Precomputed latents:", PRECOMPUTE_STAGE2_LATENTS)

# ============================================================
# STAGE 2 DATA: one-time latent precompute (optional)
# ============================================================
def precompute_stage2_latents(encoder, dataset, current_path, future_path):
    """Encode every clip once and cache (current, future) spatial latents."""
    marker = current_path.with_suffix(".done")
    if marker.exists() and current_path.exists() and future_path.exists():
        print("[Stage 2] latent cache already exists; skipping precompute.")
        return True

    for p in (current_path, future_path, marker):
        p.unlink(missing_ok=True)

    n = len(dataset)
    print(f"[Stage 2] precomputing latents for {n} clips (one-time pass) ...")
    loader = DataLoader(
        dataset, batch_size=STAGE2_PRECOMPUTE_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False, drop_last=False,
    )
    shape = (n, EMBED_DIM, SPATIAL_TOKENS, SPATIAL_TOKENS)
    mm_cur = np.lib.format.open_memmap(str(current_path), mode="w+", dtype=np.float16, shape=shape)
    mm_fut = np.lib.format.open_memmap(str(future_path), mode="w+", dtype=np.float16, shape=shape)

    start = 0
    with torch.no_grad():
        for bi, video in enumerate(loader):
            video = video.to(device, dtype=torch.float32, non_blocking=True)
            with autocast_context():
                all_tokens = encoder(video)
            all_tokens = all_tokens.float()
            cur = tokens_to_spatial(all_tokens, 0)
            fut = tokens_to_spatial(all_tokens, future_token_index)
            b = cur.shape[0]
            mm_cur[start:start + b] = cur.cpu().numpy().astype(np.float16)
            mm_fut[start:start + b] = fut.cpu().numpy().astype(np.float16)
            start += b
            if bi % 25 == 0:
                print(f"  precompute {start}/{n}", flush=True)
    del mm_cur, mm_fut
    marker.touch()
    print("[Stage 2] latent precompute done.")
    return True


class Stage2CachedDataset(Dataset):
    """Returns (current_latent, future_latent, actual_future_frame) from cache."""

    def __init__(self, records, current_path, future_path):
        self.records = records
        self.current = np.lib.format.open_memmap(str(current_path), mode="r")
        self.future = np.lib.format.open_memmap(str(future_path), mode="r")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        _, frames, indices = self.records[i]
        current = torch.from_numpy(np.asarray(self.current[i]).copy()).float()
        future = torch.from_numpy(np.asarray(self.future[i]).copy()).float()
        target = load_rgb(frames[indices[FUTURE_FRAME_OFFSET]])  # (3,H,W) in [0,1]
        return current, future, target


if PRECOMPUTE_STAGE2_LATENTS:
    precompute_stage2_latents(
        encoder, full_dataset, STAGE2_LATENTS_CURRENT, STAGE2_LATENTS_FUTURE
    )
    stage2_full = Stage2CachedDataset(
        clip_records, STAGE2_LATENTS_CURRENT, STAGE2_LATENTS_FUTURE
    )
    _gen = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        stage2_full, [num_train, num_val], generator=_gen
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=use_amp, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_amp, drop_last=False,
    )
    print("[Stage 2] using cached latents (encoder removed from the loop).")
else:
    print("[Stage 2] using on-the-fly encoder latents.")


def get_stage2_batch(batch):
    """Normalise a Stage-2 batch to (current_latent, future_latent, actual_future).

    Handles cached batches ((B,768,24,24),(B,768,24,24),(B,3,384,384)) and
    on-the-fly video batches ((B,3,64,384,384)).
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 3 and batch[0].ndim == 4:
        current, future, target = [b.to(device, non_blocking=True) for b in batch]
        return current, future, target.float()
    video = batch.to(device, dtype=torch.float32, non_blocking=True)
    with torch.no_grad():
        with autocast_context():
            all_tokens = encoder(video)
        all_tokens = all_tokens.float()
        current = tokens_to_spatial(all_tokens, 0)
        future = tokens_to_spatial(all_tokens, future_token_index)
        target = denormalize(video[:, :, FUTURE_FRAME_OFFSET]).float()
    return current, future, target

# ============================================================
# STAGE 2 TRAINING
# ============================================================
def latest_stage2_epoch():
    epoch = 0
    for p in CHECKPOINT_DIR.glob("vjepa21_grandtour_stage2_epoch_*.pth"):
        try:
            epoch = max(epoch, int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return epoch


start_epoch2 = latest_stage2_epoch() if RESUME_STAGE2 else 0
if start_epoch2 > 0:
    ck = CHECKPOINT_DIR / f"vjepa21_grandtour_stage2_epoch_{start_epoch2:02d}.pth"
    sd = torch.load(ck, map_location=device, weights_only=True)
    future_predictor.load_state_dict(sd["future_predictor"], strict=True)
    decoder.load_state_dict(sd["decoder"], strict=True)
    optimizer2.load_state_dict(sd["optimizer"])
    print(f"[Stage 2 resume] loaded {ck} (epoch {start_epoch2})")

for epoch in range(start_epoch2, STAGE2_EPOCHS):
    future_predictor.train()
    decoder.train()

    running_loss = 0.0
    _accum = max(1, int(STAGE2_GRAD_ACCUM or 1))
    _stage2_params = list(future_predictor.parameters()) + list(decoder.parameters())

    # STAGE2_STEPS_PER_EPOCH caps *optimizer steps*; each = _accum micro-batches.
    _micro_budget = len(train_loader)
    if STAGE2_STEPS_PER_EPOCH is not None:
        _micro_budget = min(_micro_budget, int(STAGE2_STEPS_PER_EPOCH) * _accum)

    optimizer2.zero_grad(set_to_none=True)
    _micro_done = 0
    _opt_steps = 0

    for step, batch in enumerate(train_loader):
        if _micro_done >= _micro_budget:
            break
        _micro_done += 1

        current_latent, future_latent, actual_future = get_stage2_batch(batch)

        with autocast_context():
            predicted_latent = future_predictor(current_latent)
            predicted_future = decoder(predicted_latent)
            reconstructed_future = decoder(future_latent)

            predicted_flat = predicted_latent.flatten(1)
            future_flat = future_latent.flatten(1)

            latent_loss = F.smooth_l1_loss(
                F.normalize(predicted_flat, dim=1),
                F.normalize(future_flat, dim=1),
            )
            predicted_rgb_loss = rgb_reconstruction_loss(predicted_future.float(), actual_future)
            teacher_rgb_loss = rgb_reconstruction_loss(reconstructed_future.float(), actual_future)

            loss = 0.50 * latent_loss + 1.00 * predicted_rgb_loss + 0.50 * teacher_rgb_loss

        (loss / _accum).backward()
        running_loss += float(loss.item())

        # Step the optimizer every _accum micro-batches (or at the end of budget).
        if _micro_done % _accum == 0 or _micro_done == _micro_budget:
            torch.nn.utils.clip_grad_norm_(_stage2_params, GRAD_CLIP)
            optimizer2.step()
            optimizer2.zero_grad(set_to_none=True)
            _opt_steps += 1

            if _opt_steps % 10 == 0 or _micro_done == _micro_budget:
                print(
                    f"Stage2 Epoch {epoch+1:02d}/{STAGE2_EPOCHS} | "
                    f"Opt-step {_opt_steps:03d} | "
                    f"Total {loss.item():.6f} | Latent {latent_loss.item():.6f} | "
                    f"RGB {predicted_rgb_loss.item():.6f}"
                )

    stage2_train_loss = running_loss / max(1, _micro_done)

    future_predictor.eval()
    decoder.eval()
    val_running = 0.0

    with torch.no_grad():
        for batch in val_loader:
            current_latent, future_latent, actual_future = get_stage2_batch(batch)

            with autocast_context():
                predicted_latent = future_predictor(current_latent)
                predicted_future = decoder(predicted_latent)
                teacher_future = decoder(future_latent)

                val_loss = (
                    0.50 * F.smooth_l1_loss(
                        F.normalize(predicted_latent.flatten(1), dim=1),
                        F.normalize(future_latent.flatten(1), dim=1),
                    )
                    + rgb_reconstruction_loss(predicted_future.float(), actual_future)
                    + 0.50 * rgb_reconstruction_loss(teacher_future.float(), actual_future)
                )
            val_running += float(val_loss.item())

    stage2_val_loss = val_running / max(1, len(val_loader))

    print(
        f"Stage2 Epoch {epoch+1:02d} complete | "
        f"Train {stage2_train_loss:.6f} | Val {stage2_val_loss:.6f}"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "future_predictor": future_predictor.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer2.state_dict(),
            "train_loss": stage2_train_loss,
            "val_loss": stage2_val_loss,
        },
        CHECKPOINT_DIR / f"vjepa21_grandtour_stage2_epoch_{epoch+1:02d}.pth",
    )

# ============================================================
# FINAL EVALUATION
# ============================================================
print("=" * 72)
print("FINAL EVALUATION")
print("=" * 72)

encoder.eval()
future_predictor.eval()
decoder.eval()


def cosine_similarity_image(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().item()


def calculate_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(a, b):
    return 1.0 - ssim_value_loss(a, b).item()


# Average metrics over the whole validation set.
agg_pred_psnr, agg_pred_ssim, agg_pred_cos = [], [], []
agg_tea_psnr, agg_tea_ssim, agg_tea_cos = [], [], []

with torch.no_grad():
    for batch in val_loader:
        current_latent, actual_future_latent, actual_future = get_stage2_batch(batch)

        predicted_latent = future_predictor(current_latent)
        predicted_future = decoder(predicted_latent).float()
        reconstructed_future = decoder(actual_future_latent).float()

        for b in range(actual_future.shape[0]):
            pf, rf, af = predicted_future[b], reconstructed_future[b], actual_future[b]
            agg_pred_psnr.append(calculate_psnr(pf, af))
            agg_pred_ssim.append(calculate_ssim(pf, af))
            agg_pred_cos.append(cosine_similarity_image(pf, af))
            agg_tea_psnr.append(calculate_psnr(rf, af))
            agg_tea_ssim.append(calculate_ssim(rf, af))
            agg_tea_cos.append(cosine_similarity_image(rf, af))

print(f"[val avg] Predicted vs Actual  PSNR {np.mean(agg_pred_psnr):.3f} dB | "
      f"SSIM {np.mean(agg_pred_ssim):.4f} | cos {np.mean(agg_pred_cos):.4f}")
print(f"[val avg] GT-decode vs Actual  PSNR {np.mean(agg_tea_psnr):.3f} dB | "
      f"SSIM {np.mean(agg_tea_ssim):.4f} | cos {np.mean(agg_tea_cos):.4f}")

# Triplet on the first validation clip (works with cached or on-the-fly latents).
_first = val_dataset.indices[0] if hasattr(val_dataset, "indices") else 0
_seq0, _frames0, _idx0 = clip_records[_first]

_s0 = val_dataset[0]
if PRECOMPUTE_STAGE2_LATENTS:
    _b0 = [t.unsqueeze(0).to(device) for t in _s0]     # cached: (current, future, target)
else:
    _b0 = _s0.unsqueeze(0).to(device)                  # video clip

with torch.no_grad():
    current_latent, actual_future_latent, actual_future = get_stage2_batch(_b0)
    predicted_latent = future_predictor(current_latent)
    predicted_future = decoder(predicted_latent).float()
    reconstructed_future = decoder(actual_future_latent).float()
    current_frame = load_rgb(_frames0[0]).unsqueeze(0).to(device)  # frame 0, [0,1]

current_pred_cos = cosine_similarity_image(current_frame, predicted_future)
current_actual_cos = cosine_similarity_image(current_frame, actual_future)
pred_actual_cos = cosine_similarity_image(predicted_future, actual_future)
teacher_actual_cos = cosine_similarity_image(reconstructed_future, actual_future)

pred_psnr = calculate_psnr(predicted_future, actual_future)
pred_ssim = calculate_ssim(predicted_future, actual_future)
teacher_psnr = calculate_psnr(reconstructed_future, actual_future)
teacher_ssim = calculate_ssim(reconstructed_future, actual_future)

print("Current vs Predicted cosine : %.4f" % current_pred_cos)
print("Current vs Actual cosine    : %.4f" % current_actual_cos)
print("Predicted vs Actual cosine  : %.4f" % pred_actual_cos)
print("GT-latent decode vs Actual  : %.4f" % teacher_actual_cos)
print("Predicted vs Actual PSNR    : %.3f dB" % pred_psnr)
print("Predicted vs Actual SSIM    : %.4f" % pred_ssim)
print("GT-latent decode PSNR       : %.3f dB" % teacher_psnr)
print("GT-latent decode SSIM       : %.4f" % teacher_ssim)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(current_frame[0].permute(1, 2, 0).cpu().numpy())
axes[0].set_title("Current Frame (t=0)")
axes[1].imshow(predicted_future[0].permute(1, 2, 0).cpu().numpy())
axes[1].set_title(
    f"Predicted Future (t={FUTURE_FRAME_OFFSET})\n"
    f"PSNR={pred_psnr:.2f} dB | SSIM={pred_ssim:.3f}"
)
axes[2].imshow(actual_future[0].permute(1, 2, 0).cpu().numpy())
axes[2].set_title("Actual Future Frame")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# FINAL CHECKPOINT
# ============================================================
final_checkpoint = CHECKPOINT_DIR / "vjepa21_grandtour_final_sharp_decoder.pth"
torch.save(
    {
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "jepa_predictor": jepa_predictor.state_dict(),
        "future_predictor": future_predictor.state_dict(),
        "decoder": decoder.state_dict(),
        "num_frames": NUM_FRAMES,
        "temporal_stride": TEMPORAL_STRIDE,
        "clip_step": CLIP_STEP,
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "tubelet_size": TUBELET_SIZE,
        "embed_dim": EMBED_DIM,
        "future_frame_offset": FUTURE_FRAME_OFFSET,
        "num_clips": len(full_dataset),
        "train_clips": num_train,
        "val_clips": num_val,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage2_epochs": STAGE2_EPOCHS,
    },
    final_checkpoint,
)

print("=" * 72)
print("PIPELINE COMPLETE")
print("=" * 72)
print("Final checkpoint:", final_checkpoint)
print("Dataset clips:", len(full_dataset))
print("Training clips:", num_train)
print("Validation clips:", num_val)
print("Input shape:", ("B", 3, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE))
print("V-JEPA tokens:", ("B", TOTAL_TOKENS, EMBED_DIM))
print("Spatial latent:", ("B", EMBED_DIM, SPATIAL_TOKENS, SPATIAL_TOKENS))
print("Encoder frozen during Stage 2: YES")
print("=" * 72)
