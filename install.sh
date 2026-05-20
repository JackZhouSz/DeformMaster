#!/usr/bin/env bash
# DeformMaster install helper.
#
# Assumes you have already created and activated a Python environment, and
# that CUDA is available (`nvcc --version` should work).

set -e

echo "==> Installing Python deps from requirements.txt"
pip install -r requirements.txt

echo "==> Installing CUDA rasterizer submodules (Inria, non-commercial license)"
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
pip install git+https://github.com/rahul-goel/fused-ssim
pip install git+https://gitlab.inria.fr/bkerbl/simple-knn.git

echo "==> Done. Run the smoke test:"
echo "    python inference.py --case_name double_lift_cloth_1 --config configs/planar.yaml"
