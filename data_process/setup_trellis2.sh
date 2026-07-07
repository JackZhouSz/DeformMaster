#!/usr/bin/env bash
# Init data_process/third_party/TRELLIS.2 submodule and apply compat patches.
#
# Upstream microsoft/TRELLIS.2 @5565d24 has two incompatibilities that
# this script patches:
#   1) image_feature_extractor.py: self.model.layer vs self.model.model.layer
#      (transformers API change in current trellis2 conda env)
#   2) trellis2_image_to_3d.py: unconditional rembg load triggers gated
#      HF repo briaai/RMBG-2.0 — patch to rembg_model=None since our
#      inputs already have alpha masks
#
# Run this AFTER `git clone` (or `git submodule update --init`) whenever
# the TRELLIS.2 submodule is freshly checked out.
set -e
cd "$(dirname "$0")/.."

if [ ! -f data_process/third_party/TRELLIS.2/trellis2/modules/image_feature_extractor.py ]; then
    echo "[setup_trellis2] initializing submodule..."
    git submodule update --init --recursive data_process/third_party/TRELLIS.2
fi

cd data_process/third_party/TRELLIS.2

# Check if already patched (idempotent)
if grep -q "self.model.model.layer" trellis2/modules/image_feature_extractor.py 2>/dev/null; then
    echo "[setup_trellis2] already patched, nothing to do"
    exit 0
fi

echo "[setup_trellis2] applying trellis2_compat.patch..."
git apply ../trellis2_compat.patch
echo "[setup_trellis2] done. TRELLIS 2 ready."
