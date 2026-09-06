#!/usr/bin/env bash
# ============================================================================
# One-shot environment setup for this repo, using `uv` for the virtualenv and
# dependency management. Works on Linux (molab / AMD MI300X ROCm / NVIDIA).
#
# Usage:
#   bash amd/setup.sh                 # auto-detect GPU vendor (or CPU fallback)
#   bash amd/setup.sh --nvidia        # force NVIDIA (default PyTorch CUDA build)
#   bash amd/setup.sh --rocm 6.2      # force AMD ROCm (default 6.2)
#   bash amd/setup.sh --pytorch-index https://download.pytorch.org/whl/rocm6.2
#
# Options:
#   AMD=1        alias for --rocm (auto version detect best-effort)
#   PYTHON=3.12  python version for the venv (default 3.12)
#   RUN_DL=1     also download checkpoints (official V-JEPA + HF step8000)
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-3.12}"
VENV=".venv"
TORCH_INDEX=""            # empty => vendor default below

# ---- parse args ------------------------------------------------------------
VENDOR="auto"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvidia) VENDOR="nvidia"; shift ;;
    --rocm)   VENDOR="rocm"; shift; [[ $# -gt 0 && "$1" != --* ]] && ROCM_VER="$1" && shift ;;
    --pytorch-index) shift; TORCH_INDEX="$1"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
ROCM_VER="${ROCM_VER:-6.2}"

if [[ "$VENDOR" == "auto" ]]; then
  if command -v rocminfo >/dev/null 2>&1 || ls /opt/rocm* >/dev/null 2>&1; then
    VENDOR="rocm"
  else
    VENDOR="nvidia"
  fi
fi
if [[ "${AMD:-0}" == "1" ]]; then VENDOR="rocm"; fi
if [[ -z "$TORCH_INDEX" ]]; then
  if [[ "$VENDOR" == "rocm" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VER}"
  else
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
  fi
fi
echo ">> vendor=$VENDOR  torch_index=$TORCH_INDEX  python=$PYTHON"

# ---- uv --------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ---- submodules (vjepa2) ---------------------------------------------------
if [[ -d .git ]]; then
  git submodule update --init --recursive || echo "!! submodule update failed (continue anyway)"
fi

# ---- venv + torch ----------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  uv venv "$VENV" --python "$PYTHON"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V

echo ">> installing torch/torchvision from $TORCH_INDEX"
uv pip install --index-url "$TORCH_INDEX" \
    --index-strategy unsafe-best-match \
    torch torchvision

# ---- repo + runtime deps ---------------------------------------------------
echo ">> installing repo + deps"
uv pip install -e ".[eval]"
uv pip install timm einops "zarr<3" huggingface_hub pillow pyyaml marimo

# ---- dirs + GPU sanity -----------------------------------------------------
mkdir -p data logs ckpts
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# ---- checkpoints (optional) ------------------------------------------------
if [[ "${RUN_DL:-0}" == "1" ]]; then
  echo ">> downloading official V-JEPA 2.1 ckpt"
  python - <<'PY'
from pathlib import Path
import torch
p = Path("ckpts/vjepa2_1_vitb_dist_vitG_384.pt")
if not p.exists() or p.stat().st_size < 1.4e9:
    torch.hub.download_url_to_file(
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        str(p))
print("official ckpt ok")
PY
  if command -v huggingface-cli >/dev/null 2>&1 \
     && [[ ! -f ckpts/big_run_step8000.pt ]] \
     && huggingface-cli whoami >/dev/null 2>&1; then
    echo ">> downloading big-run checkpoint from HF"
    huggingface-cli download rayaanoidpr/vjepa-wm-bigrun \
      big_run_step8000.pt --local-dir ckpts
  else
    echo "!! skip HF download (no huggingface-cli / not logged in / ckpt present)"
    echo "   manual: huggingface-cli download rayaanoidpr/vjepa-wm-bigrun big_run_step8000.pt --local-dir ckpts"
  fi
fi

echo
echo "==== setup complete ===="
echo "activate:  source .venv/bin/activate"
echo "data:      python amd/pull_missions.py core   (then: all)"
echo "train:     python amd/big_run.py 8000"
echo "probes:    python amd/probe.py --model ckpts/big_run_step8000.pt"
echo "see amd/README.md for the full run order"
