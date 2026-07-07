#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1   # force real-time log output (no pipe buffer)

# CMA-ES tuning of controller PD gains (kp, kd) per material category.
# Each category trains ALL its target_scenes per fitness evaluation.
# 3 categories run in parallel (each uses its own gpus from yaml).
#
# Usage:
#   bash run_actuator_gain_selection.sh              # default settings
#   bash run_actuator_gain_selection.sh 30 4 15      # custom: iters popsize max_gens

ITERS="${1:-5}"
POPSIZE="${2:-4}"
MAX_GENS="${3:-10}"

# Activate conda env if available.
CONDA_ENV="${CONDA_ENV:-deformmaster}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [ -n "$CONDA_BIN" ]; then
    eval "$("$CONDA_BIN" shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "Activated conda env: $CONDA_ENV"
fi

echo "============================================"
echo "  CMA-ES Controller Gain Tuning (all cases)"
echo "  iters=$ITERS  popsize=$POPSIZE  max_gens=$MAX_GENS"
echo "  Evals per category: $((POPSIZE * MAX_GENS))"
echo "  Each eval trains ALL target_scenes in parallel"
echo "  3 categories run in PARALLEL"
echo "  Start time: $(date)"
echo "============================================"

mkdir -p actuator_gain_results

python scripts_training_eval/actuator/actuator_gain_selection.py \
    --base-config configs/cloth.yaml \
    --iters "$ITERS" --popsize "$POPSIZE" --max-generations "$MAX_GENS" \
    2>&1 | tee actuator_gain_results/cloth.log &
PID_CLOTH=$!
echo "  Cloth tuning started (PID: ${PID_CLOTH})"

python scripts_training_eval/actuator/actuator_gain_selection.py \
    --base-config configs/softbody.yaml \
    --iters "$ITERS" --popsize "$POPSIZE" --max-generations "$MAX_GENS" \
    2>&1 | tee actuator_gain_results/softbody.log &
PID_SOFT=$!
echo "  Softbody tuning started (PID: ${PID_SOFT})"

python scripts_training_eval/actuator/actuator_gain_selection.py \
    --base-config configs/rope.yaml \
    --iters "$ITERS" --popsize "$POPSIZE" --max-generations "$MAX_GENS" \
    2>&1 | tee actuator_gain_results/rope.log &
PID_ROPE=$!
echo "  Rope tuning started (PID: ${PID_ROPE})"

echo ""
echo "Waiting for all 3 categories..."

FAIL=0
wait $PID_CLOTH || { echo "  [FAIL] Cloth tuning failed"; FAIL=1; }
echo "  Cloth done."

wait $PID_SOFT || { echo "  [FAIL] Softbody tuning failed"; FAIL=1; }
echo "  Softbody done."

wait $PID_ROPE || { echo "  [FAIL] Rope tuning failed"; FAIL=1; }
echo "  Rope done."

echo ""
echo "============================================"
echo "  All CMA-ES tuning complete!"
echo "  End time: $(date)"
echo "============================================"

echo ""
echo "=== Best (raw_kp, raw_kd) per category ==="
for TAG in cloth softbody rope; do
    RESULT="actuator_gain_results/${TAG}/result.json"
    if [ -f "$RESULT" ]; then
        python -c "
import json
with open('${RESULT}') as f:
    r = json.load(f)
print(f'  ${TAG:10s}: raw_kp={r[\"best_raw_kp\"]:+.4f}  raw_kd={r[\"best_raw_kd\"]:+.4f}  loss={r[\"best_loss\"]:.6f}  ({r[\"n_evals\"]} evals, {r[\"total_min\"]:.0f} min)')
"
    else
        echo "  ${TAG}: result.json not found"
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "[WARN] Some categories failed. Check actuator_gain_results/*.log"
fi
