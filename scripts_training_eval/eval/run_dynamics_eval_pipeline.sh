#!/bin/bash
set -euo pipefail

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

# Dynamics eval pipeline: inference → GS render → evaluate → plot
# (Called by run_dynamics_training_eval.sh after dynamics training finishes.)
#
# Usage:
#   bash scripts_training_eval/eval/run_dynamics_eval_pipeline.sh output_4
#   bash scripts_training_eval/eval/run_dynamics_eval_pipeline.sh output_4 0        # specify GPU
#   bash scripts_training_eval/eval/run_dynamics_eval_pipeline.sh output_4 0 "single_push_rope,single_lift_rope"  # specific scenes

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts_training_eval/eval/run_dynamics_eval_pipeline.sh <output_dir> [gpu] [scenes_csv]"
    echo ""
    echo "Steps executed:"
    echo "  1. MPM inference  → <output_dir>/mpm_inference/"
    echo "  2. GS rendering   → <output_dir>/gaussian_output_dynamic_mpm/"
    echo "  3. Evaluation     → results/<tag>_chamfer.csv, _track.csv, _render.txt"
    echo "  4. Plot comparison → results/plots/"
    exit 1
fi

OUTPUT_DIR="$1"
GPU="${2:-0}"
SCENES="${3:-}"

TAG="$(basename "$OUTPUT_DIR")"

echo "============================================"
echo "  Pipeline: ${TAG}"
echo "  GPU: ${GPU}"
echo "  Scenes: ${SCENES:-all}"
echo "============================================"

# Step 1: MPM Inference
echo ""
echo ">>> Step 1/4: MPM Inference"
SCENE_ARGS=""
if [ -n "$SCENES" ]; then
    SCENE_ARGS="--scenes $SCENES"
fi
CUDA_VISIBLE_DEVICES=$GPU python scripts_training_eval/dynamics/script_inference_mpm.py \
    --input "$OUTPUT_DIR" \
    --gpu 0 \
    $SCENE_ARGS

# Step 2: GS Rendering
echo ""
echo ">>> Step 2/4: GS Rendering"
RENDER_SCENES=""
if [ -n "$SCENES" ]; then
    RENDER_SCENES="$SCENES"
fi
bash scripts_training_eval/appearance/gs_run_simulation.sh \
    "${OUTPUT_DIR}/mpm_inference" \
    "$RENDER_SCENES" \
    "" \
    "$GPU"

# Step 3: Evaluation
echo ""
echo ">>> Step 3/4: Evaluation"
bash scripts_training_eval/eval/evaluate.sh "$OUTPUT_DIR"

# Step 4: Plot Comparison
echo ""
echo ">>> Step 4/4: Plot Comparison"
mkdir -p results/plots
python scripts_training_eval/eval/plot_results_comparison.py \
    --methods "baseline,${TAG}" \
    --results_dir results \
    --output_dir results/plots

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results: results/${TAG}_chamfer.csv"
echo "           results/${TAG}_track.csv"
echo "           results/${TAG}_render.txt"
echo "  Plots:   results/plots/"
echo "  Summary: results/plots/summary_${TAG}.txt"
echo "============================================"
