#!/usr/bin/env bash
# One-shot environment setup for the GrandTour JEPA-WM (jepa-wms) pipeline.
# Creates: ./.venv (python 3.10), ./jepa-wms (pinned upstream + grandtour patch),
#          ./data/{raw,datasets,logs,torchhub}, grandtour/env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

JEPA_REPO="https://github.com/facebookresearch/jepa-wms.git"
# upstream commit this branch is validated against (clone HEAD at development time)
JEPA_PIN="13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
PATCH="$ROOT/grandtour/patches/jepa_wms_grandtour.patch"
VENV="$ROOT/.venv"

echo "[1/5] uv (python 3.10 tooling)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[2/5] python 3.10 venv"
uv venv "$VENV" --python 3.10
PYBIN="$VENV/bin/python"
"$PYBIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'

echo "[3/5] jepa-wms checkout @ $JEPA_PIN + grandtour patch"
if [ ! -d jepa-wms/.git ]; then
  git clone "$JEPA_REPO" jepa-wms
fi
git -C jepa-wms fetch --depth 1 origin "$JEPA_PIN" || true
git -C jepa-wms checkout -q "$JEPA_PIN"
git -C jepa-wms apply --check "$PATCH" && git -C jepa-wms apply "$PATCH"
echo "    patched: $(git -C jepa-wms status --short | wc -l) changed/new files"

echo "[4/5] install jepa-wms (editable, no deps) + curated deps"
uv pip install --python "$PYBIN" --no-deps -e jepa-wms
uv pip install --python "$PYBIN" -r requirements.in

echo "[5/5] data dirs + env file"
mkdir -p data/raw data/datasets data/logs data/torchhub
cat > grandtour/env.sh <<ENV
# Source this file:  source grandtour/env.sh
export GT_HOME="$ROOT"
export JEPAWM_HOME="\$GT_HOME/jepa-wms"
export JEPAWM_DSET="\$GT_HOME/data/datasets"
export JEPAWM_LOGS="\$GT_HOME/data/logs"
export GT_RAW="\$GT_HOME/data/raw"
export GT_VIZ="\$GT_HOME/data/logs/grandtour_sweep/viz"
export TORCH_HOME="\$GT_HOME/data/torchhub"
export JEPA_ENV_PY="$PYBIN"
export PATH="\$GT_HOME/.venv/bin:\$PATH"
ENV

echo
echo "Done. Next:"
echo "  source grandtour/env.sh"
echo "  python grandtour/scripts/download_missions.py --help"
