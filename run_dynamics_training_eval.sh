#!/bin/bash
set -euo pipefail

# Dynamics training and eval pipeline: train all → inference → render → evaluate → plot
# (system ID training; no RGB finetune. For Stage 3 RGB finetune use
# run_rgb_guided_refinement.sh instead.)
#
# Usage:
#   bash run_dynamics_training_eval.sh                # auto-detect from configs/cloth.yaml
#   bash run_dynamics_training_eval.sh output_7       # explicit output_tag
#   bash run_dynamics_training_eval.sh output_7 0     # explicit output_tag + pipeline GPU
#
# Whole-dir ablation (all 4 configs come from one ablation dir):
#   CONFIG_DIR=configs/ablations/abl_no_distribution bash run_dynamics_training_eval.sh
#
# Per-category override (mix-and-match individual yamls):
#   ROPE_CONFIG=configs/ablations/abl_no_moe/rope.yaml \
#   CLOTH_CONFIG=configs/cloth.yaml \
#   bash run_dynamics_training_eval.sh
#   (anything not set falls back to ${CONFIG_DIR}/<category>.yaml)
#
# Skip a category by setting its config to empty:
#   PACKAGE_CONFIG="" bash run_dynamics_training_eval.sh

# CONFIG_DIR env var picks the default config directory (default: configs/).
# Per-category *_CONFIG env vars override the default; pass empty to skip.
# All exported so the eval pipeline (scripts_training_eval/eval/run_dynamics_eval_pipeline.sh ->
# evaluate.sh) picks up the same config whitelist; otherwise the
# evaluator would silently fall back to globbing every case under
# data/different_types and pollute the CSV with unrelated my_* dirs.
export CONFIG_DIR="${CONFIG_DIR:-configs}"
export ROPE_CONFIG="${ROPE_CONFIG-${CONFIG_DIR}/rope.yaml}"
export CLOTH_CONFIG="${CLOTH_CONFIG-${CONFIG_DIR}/cloth.yaml}"
export SOFTBODY_CONFIG="${SOFTBODY_CONFIG-${CONFIG_DIR}/softbody.yaml}"
export PACKAGE_CONFIG="${PACKAGE_CONFIG-${CONFIG_DIR}/package.yaml}"
echo "Configs:"
echo "  rope     = ${ROPE_CONFIG:-(skipped)}"
echo "  cloth    = ${CLOTH_CONFIG:-(skipped)}"
echo "  softbody = ${SOFTBODY_CONFIG:-(skipped)}"
echo "  package  = ${PACKAGE_CONFIG:-(skipped)}"

# Auto-detect OUTPUT_TAG from the first non-empty config (cloth → rope →
# softbody → package). Reads `output_dir: './output_X/<tag>_warp'` and
# extracts 'output_X'.
if [ $# -ge 1 ]; then
    OUTPUT_TAG="$1"
else
    DETECT_FROM=""
    for c in "$CLOTH_CONFIG" "$ROPE_CONFIG" "$SOFTBODY_CONFIG" "$PACKAGE_CONFIG"; do
        [ -n "$c" ] && [ -f "$c" ] && { DETECT_FROM="$c"; break; }
    done
    if [ -z "$DETECT_FROM" ]; then
        echo "ERROR: no config available to auto-detect output_tag from"
        echo "Usage: bash run_dynamics_training_eval.sh [output_tag] [pipeline_gpu]"
        exit 1
    fi
    OUTPUT_TAG=$(python -c "
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load('${DETECT_FROM}')
print(os.path.basename(os.path.dirname(str(cfg.output_dir).rstrip('/'))))
")
    if [ -z "$OUTPUT_TAG" ]; then
        echo "ERROR: failed to auto-detect output_tag from ${DETECT_FROM}"
        exit 1
    fi
    echo "Auto-detected output_tag: ${OUTPUT_TAG} (from ${DETECT_FROM})"
fi
PIPELINE_GPU="${2:-0}"

# Activate conda env if available. conda's own deactivate.d scripts
# reference variables (e.g. CONDA_BACKUP_CXX) that aren't always set
# under `set -u`, so disable nounset just for this block.
CONDA_ENV="${CONDA_ENV:-deformmaster}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [ -n "$CONDA_BIN" ]; then
    set +u
    eval "$("$CONDA_BIN" shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
    echo "Activated conda env: $CONDA_ENV"
fi

echo "============================================"
echo "  Full Experiment: ${OUTPUT_TAG}"
echo "  Pipeline GPU: ${PIPELINE_GPU}"
echo "  Start time: $(date)"
echo "============================================"

# Step 1: Launch all training scripts in background (skip ones with empty config).
echo ""
echo ">>> Step 1: Launching training..."

declare -a PIDS=()
declare -a NAMES=()

if [ -n "$CLOTH_CONFIG" ]; then
    python scripts_training_eval/dynamics/script_train_cloth.py --config "$CLOTH_CONFIG" &
    PIDS+=("$!"); NAMES+=("cloth")
    echo "  Cloth training started (PID: ${PIDS[-1]}, config: $CLOTH_CONFIG)"
fi
if [ -n "$ROPE_CONFIG" ]; then
    python scripts_training_eval/dynamics/script_train_rope.py --config "$ROPE_CONFIG" &
    PIDS+=("$!"); NAMES+=("rope")
    echo "  Rope training started (PID: ${PIDS[-1]}, config: $ROPE_CONFIG)"
fi
if [ -n "$SOFTBODY_CONFIG" ]; then
    python scripts_training_eval/dynamics/script_train_softbody.py --config "$SOFTBODY_CONFIG" &
    PIDS+=("$!"); NAMES+=("softbody")
    echo "  Softbody training started (PID: ${PIDS[-1]}, config: $SOFTBODY_CONFIG)"
fi
if [ -n "$PACKAGE_CONFIG" ]; then
    python scripts_training_eval/dynamics/script_train_cloth.py --config "$PACKAGE_CONFIG" &
    PIDS+=("$!"); NAMES+=("package")
    echo "  Package training started (PID: ${PIDS[-1]}, config: $PACKAGE_CONFIG)"
fi

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "ERROR: all four *_CONFIG env vars are empty; nothing to train."
    exit 1
fi

# Step 2: Wait for all training to finish.
echo ""
echo ">>> Step 2: Waiting for ${#PIDS[@]} training job(s) to complete..."

FAIL=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[i]}"; name="${NAMES[i]}"
    wait "$pid" || { echo "  [FAIL] ${name} training failed"; FAIL=1; }
    echo "  ${name} training done."
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "[ERROR] Some training jobs failed. Continuing with pipeline anyway..."
fi

echo ""
echo "All training complete at $(date)"

# Step 3: Run evaluation pipeline
echo ""
echo ">>> Step 3: Running evaluation pipeline..."
bash scripts_training_eval/eval/run_dynamics_eval_pipeline.sh "${OUTPUT_TAG}" "${PIPELINE_GPU}"

echo ""
echo "============================================"
echo "  Full experiment complete!"
echo "  End time: $(date)"
echo "  Results in: results/${OUTPUT_TAG}_*.csv"
echo "============================================"
