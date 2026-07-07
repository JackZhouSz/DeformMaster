#!/bin/bash
set -o pipefail

# Single shared-GPU-pool launcher for RGB-guided refinement.
# Cases across all categories are queued together; GPU workers pull any case
# independently -> maximum saturation.
#
# Per-stage outputs:
#   training ckpts -> cfg.stage3.output_dir (per-category, set in yaml)
#   inference pkls -> ${STAGE3_DIR}/mpm_inference/<scene>/inference.pkl
#   dynamic GS     -> ${STAGE3_DIR}/gaussian_output_dynamic_mpm/<scene>/...
#   eval CSVs      -> results/${basename(STAGE3_DIR)}_{chamfer,track,render}.{csv,txt}
#
# To switch between full-seq × 3-cam (default) and full-seq × 1-cam, edit
# `stage3.rgb_finetune.cams:` in each main config under configs/.
#
# Usage:
#   bash run_rgb_guided_refinement.sh               # train -> inference -> render -> eval
#   bash run_rgb_guided_refinement.sh train         # just training
#   bash run_rgb_guided_refinement.sh eval          # just inference + render + eval
#                                          (assumes training done)

STAGE="${1:-all}"

CONDA_ENV="${CONDA_ENV:-deformmaster}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [ -n "$CONDA_BIN" ]; then
    eval "$("$CONDA_BIN" shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# gsplat JIT: use the active conda compiler/CUDA toolchain when available.
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
    export CC="${CC:-$(command -v x86_64-conda-linux-gnu-gcc)}"
fi
if command -v x86_64-conda-linux-gnu-g++ >/dev/null 2>&1; then
    export CXX="${CXX:-$(command -v x86_64-conda-linux-gnu-g++)}"
fi

# Per-category configs. Same convention as run_dynamics_training_eval.sh:
# CONFIG_DIR + ROPE_CONFIG / CLOTH_CONFIG / SOFTBODY_CONFIG / PACKAGE_CONFIG.
# Exported so the eval child shell (evaluate.sh) shares the same whitelist.
export CONFIG_DIR="${CONFIG_DIR:-configs}"
export ROPE_CONFIG="${ROPE_CONFIG-${CONFIG_DIR}/rope.yaml}"
export CLOTH_CONFIG="${CLOTH_CONFIG-${CONFIG_DIR}/cloth.yaml}"
export SOFTBODY_CONFIG="${SOFTBODY_CONFIG-${CONFIG_DIR}/softbody.yaml}"
export PACKAGE_CONFIG="${PACKAGE_CONFIG-${CONFIG_DIR}/package.yaml}"
echo "[RGB refinement] Configs:"
echo "  rope     = ${ROPE_CONFIG:-(skipped)}"
echo "  cloth    = ${CLOTH_CONFIG:-(skipped)}"
echo "  softbody = ${SOFTBODY_CONFIG:-(skipped)}"
echo "  package  = ${PACKAGE_CONFIG:-(skipped)}"

# Build --group args for the python scripts (skip empty configs).
GROUP_ARGS=()
for c in "$CLOTH_CONFIG" "$PACKAGE_CONFIG" "$ROPE_CONFIG" "$SOFTBODY_CONFIG"; do
    [ -n "$c" ] && [ -f "$c" ] && GROUP_ARGS+=(--group "$c")
done
if [ "${#GROUP_ARGS[@]}" -eq 0 ]; then
    echo "ERROR: no usable *_CONFIG yamls under ${CONFIG_DIR}"
    exit 1
fi

# STAGE3_DIR = parent of cfg.stage3.output_dir (shared across all configs).
# Override with `STAGE3_DIR=foo bash run_rgb_guided_refinement.sh` for ad-hoc runs.
STAGE3_DIR="${STAGE3_DIR:-$(python3 -c "
from omegaconf import OmegaConf
import os
configs = [c for c in [
    '${CLOTH_CONFIG}', '${PACKAGE_CONFIG}', '${ROPE_CONFIG}', '${SOFTBODY_CONFIG}'
] if c and os.path.isfile(c)]
parents = {os.path.normpath(os.path.dirname(OmegaConf.load(c).stage3.output_dir)) for c in configs}
assert len(parents) == 1, f'configs disagree on RGB refinement parent dir: {parents}'
print(parents.pop())
")}"
echo "[RGB refinement] STAGE3_DIR resolved to: ${STAGE3_DIR}"

# Inference / render GPU pool — single shared pool across all categories,
# read from the cloth config (or first available). Training reads its
# GPU pool per-category directly from each yaml's stage3.train.gpus.
# Override on the fly with `STAGE3_GPUS=0,1 bash run_rgb_guided_refinement.sh`.
STAGE3_GPUS="${STAGE3_GPUS:-$(python3 -c "
from omegaconf import OmegaConf
import os
for c in ['${CLOTH_CONFIG}', '${PACKAGE_CONFIG}', '${ROPE_CONFIG}', '${SOFTBODY_CONFIG}']:
    if c and os.path.isfile(c):
        print(OmegaConf.load(c).stage3.train.gpus); break
")}"
echo "[RGB refinement] STAGE3_GPUS (inference/render) resolved to: ${STAGE3_GPUS}"

# ---- training (per-category GPU pools, read from each yaml) ----
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "train" ]; then
    echo "=============================================="
    echo "  RGB-guided refinement training (per-category GPU pools)"
    echo "  Start: $(date)"
    echo "=============================================="
    python scripts_training_eval/rgb_refinement/script_stage3_train.py "${GROUP_ARGS[@]}"
    echo ""
    echo "Training done: $(date)"
fi

[ "${STAGE}" = "train" ] && exit 0

# ---- inference + render + eval ----

# RGB-guided refinement inference / render skip if their output already exists. When
# STAGE3_DIR points at output_custom (which already holds Stage-2 results),
# we must clear the two aggregate dirs so RGB-guided refinement actually overwrites.
echo ""
echo "[clean] removing stage-2 aggregates under ${STAGE3_DIR}/"
rm -rf "${STAGE3_DIR}/mpm_inference" "${STAGE3_DIR}/gaussian_output_dynamic_mpm"

echo ""
echo "=============================================="
echo "  Inference (multi-GPU parallel)"
echo "=============================================="
python scripts_training_eval/rgb_refinement/script_stage3_inference.py "${GROUP_ARGS[@]}" \
    --output-dir "${STAGE3_DIR}/mpm_inference" \
    --gpus "${STAGE3_GPUS}"

echo ""
echo "=============================================="
echo "  Render dynamic GS (multi-GPU parallel)"
echo "=============================================="
python scripts_training_eval/rgb_refinement/script_stage3_render.py \
    --inference-dir "${STAGE3_DIR}/mpm_inference" \
    --output-dir "${STAGE3_DIR}/gaussian_output_dynamic_mpm" \
    --gpus "${STAGE3_GPUS}"

echo ""
echo "=============================================="
echo "  Evaluation (chamfer / track / render)"
echo "=============================================="
# CONFIG_DIR + *_CONFIG are already exported above, so evaluate.sh's
# whitelist stays in sync with the configs we trained on.
bash scripts_training_eval/eval/evaluate.sh "${STAGE3_DIR}"

echo ""
echo "=============================================="
echo "  RGB-guided refinement complete: $(date)"
echo "=============================================="
