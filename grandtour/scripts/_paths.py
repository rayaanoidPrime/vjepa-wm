# Shared path resolution for GrandTour pipeline scripts.
# Precedence: explicit environment variables, else defaults under the repo root.
import os
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent          # grandtour/scripts
_GT = _SCRIPTS.parent                               # grandtour/
_HOME = _GT.parent                                  # repo root

HOME = Path(os.environ.get("GT_HOME", str(_HOME)))

JEPAWM_HOME = Path(os.environ.get("JEPAWM_HOME", str(HOME / "jepa-wms")))
JEPAWM_DSET = Path(os.environ.get("JEPAWM_DSET", str(HOME / "data" / "datasets")))
JEPAWM_LOGS = Path(os.environ.get("JEPAWM_LOGS", str(HOME / "data" / "logs")))
JEPAWM_CKPT = Path(os.environ.get("JEPAWM_CKPT", str(JEPAWM_LOGS)))
JEPAWM_OSSCKPT = Path(os.environ.get("JEPAWM_OSSCKPT", str(HOME / "data" / "ossckpt")))
GT_RAW = Path(os.environ.get("GT_RAW", str(HOME / "data" / "raw")))
GT_VIZ = Path(os.environ.get("GT_VIZ", str(JEPAWM_LOGS / "grandtour_sweep" / "viz")))
TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(HOME / "data" / "torchhub")))
ENV_PY = os.environ.get("JEPA_ENV_PY", str(HOME / ".venv" / "bin" / "python"))

GT_DSET = JEPAWM_DSET / "grand_tour"                # <- data path used by jepa-wms configs
CAMERA = os.environ.get("GT_CAM", "hdr_front")


def ensure_dirs():
    for d in (JEPAWM_DSET, JEPAWM_LOGS, GT_RAW, GT_VIZ, TORCH_HOME):
        d.mkdir(parents=True, exist_ok=True)


def export_lines():
    return [
        f"GT_HOME={HOME}",
        f"JEPAWM_HOME={JEPAWM_HOME}",
        f"JEPAWM_DSET={JEPAWM_DSET}",
        f"JEPAWM_LOGS={JEPAWM_LOGS}",
        f"JEPAWM_OSSCKPT={JEPAWM_OSSCKPT}",
        f"GT_RAW={GT_RAW}",
        f"TORCH_HOME={TORCH_HOME}",
        f"JEPA_ENV_PY={ENV_PY}",
    ]
