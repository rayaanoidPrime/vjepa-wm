#!/usr/bin/env bash
# Launch jepa-wms training for the GrandTour world model (phase 1) or the
# decoder head (phase 2). Logs go to $JEPAWM_LOGS.
# Usage: bash run_training.sh [wm|step2] [custom cfg path relative to jepa-wms]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f grandtour/env.sh ] && source grandtour/env.sh

PHASE="${1:-wm}"
mkdir -p "$JEPAWM_LOGS"

if [ "$PHASE" = "step2" ]; then
  CFG="${2:-configs/vjepa_wm/grandtour_sweep/gt_v0_step2_vm2m.yaml}"
  LOG="$JEPAWM_LOGS/gt_step2_console.txt"
else
  CFG="${2:-configs/vjepa_wm/grandtour_sweep/gt_v0_12f_fps5_r224_dv2vits_AdaLN_d6_2roll_1n.yaml}"
  LOG="$JEPAWM_LOGS/gt_v0_console.txt"
fi

cd "$JEPAWM_HOME"
echo "launching: $JEPA_ENV_PY -m app.main --fname $CFG --debug"
nohup "$JEPA_ENV_PY" -m app.main --fname "$CFG" --debug > "$LOG" 2>&1 &
echo "pid $!  (tail -f $LOG)"
