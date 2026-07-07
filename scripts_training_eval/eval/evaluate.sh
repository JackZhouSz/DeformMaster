#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <prediction_dir>"
    exit 1
fi

PREDICTION_DIR="$1"
RESULT_TAG="$(basename "$PREDICTION_DIR")"

if [ -d "$PREDICTION_DIR/mpm_inference" ]; then
    INFERENCE_DIR="$PREDICTION_DIR/mpm_inference"
else
    INFERENCE_DIR="$PREDICTION_DIR"
fi

if [ -d "$PREDICTION_DIR/gaussian_output_dynamic_mpm" ]; then
    RENDER_DIR="$PREDICTION_DIR/gaussian_output_dynamic_mpm"
elif [ -d "$PREDICTION_DIR/gaussian_output_dynamic" ]; then
    RENDER_DIR="$PREDICTION_DIR/gaussian_output_dynamic"
else
    RENDER_DIR=""
fi

echo "Evaluating predictions from: $PREDICTION_DIR"
echo "Saving results with tag: $RESULT_TAG"
echo "Resolved inference dir: $INFERENCE_DIR"
if [ -n "$RENDER_DIR" ]; then
    echo "Resolved render dir: $RENDER_DIR"
fi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()} | device_count: {torch.cuda.device_count()}')"
echo "Track evaluation currently runs on CPU (NumPy/SciPy)."

# Optional config-based case whitelist. Same env-var convention as
# run_dynamics_training_eval.sh: ROPE_CONFIG / CLOTH_CONFIG / SOFTBODY_CONFIG /
# PACKAGE_CONFIG, all default to ${CONFIG_DIR:-configs}/<name>.yaml
# and are individually skippable by setting to empty. Non-existent
# paths are silently dropped here; the python scripts also warn.
CONFIG_DIR="${CONFIG_DIR:-configs}"
ROPE_CONFIG="${ROPE_CONFIG-${CONFIG_DIR}/rope.yaml}"
CLOTH_CONFIG="${CLOTH_CONFIG-${CONFIG_DIR}/cloth.yaml}"
SOFTBODY_CONFIG="${SOFTBODY_CONFIG-${CONFIG_DIR}/softbody.yaml}"
PACKAGE_CONFIG="${PACKAGE_CONFIG-${CONFIG_DIR}/package.yaml}"
CONFIG_ARGS=()
for c in "$ROPE_CONFIG" "$CLOTH_CONFIG" "$SOFTBODY_CONFIG" "$PACKAGE_CONFIG"; do
    [ -n "$c" ] && [ -f "$c" ] && CONFIG_ARGS+=(--config "$c")
done
if [ "${#CONFIG_ARGS[@]}" -gt 0 ]; then
    echo "Case whitelist from configs: ${CONFIG_ARGS[@]}"
fi

# Run evaluation scripts with prediction directory + optional config whitelist
python scripts_training_eval/eval/evaluate_chamfer.py \
    --prediction_dir "$INFERENCE_DIR" \
    --output_file "results/${RESULT_TAG}_chamfer.csv" \
    "${CONFIG_ARGS[@]}"

python scripts_training_eval/eval/evaluate_track.py \
    --prediction_path "$INFERENCE_DIR" \
    --output_file "results/${RESULT_TAG}_track.csv" \
    "${CONFIG_ARGS[@]}"

if [ -n "$RENDER_DIR" ]; then
    python gaussian_splatting/evaluate_render.py \
        --prediction_dir "$PREDICTION_DIR" \
        --render_dir "$RENDER_DIR" \
        --output_file "results/${RESULT_TAG}_render.txt"
else
    python gaussian_splatting/evaluate_render.py \
        --prediction_dir "$PREDICTION_DIR" \
        --output_file "results/${RESULT_TAG}_render.txt"
fi

echo ""
echo "Plot comparison command:"
echo "  python scripts_training_eval/eval/plot_results_comparison.py --methods baseline,${RESULT_TAG}"
