# Monocular Video to DeformMaster Data

This package converts a single RGB video into the one-camera data layout used by
DeformMaster. The default path uses VGGT-Omega temporal depth and pose, then
uses MoGe2 only on frame 0 to anchor metric scale.

## Architecture

```text
RGB video
  |-- VGGT-Omega: temporal depth + camera poses
  |-- MoGe2 frame 0: metric scale anchor
  `-- output:
      color/0/*.png
      color/0.mp4
      depth/0/*.npy
      calibrate.pkl
      metadata.json
      c2w_sequence.npy
```

## Setup

Set the VGGT-Omega repository and checkpoint paths before running the default
monocular pipeline:

```bash
export DEFORMMASTER_VGGT_ROOT=/path/to/vggt-omega
export DEFORMMASTER_VGGT_CHECKPOINT=/path/to/vggt_omega_1b_512.pt
```

MoGe2 code is bundled under `moge_model/`, and its weights are downloaded on
first use.

## Usage

Use the default VGGT-Omega temporal depth path. MoGe2 is used only on frame 0
to anchor metric scale:

```bash
export DEFORMMASTER_VGGT_ROOT=/path/to/vggt-omega
export DEFORMMASTER_VGGT_CHECKPOINT=/path/to/vggt_omega_1b_512.pt
python data_process/mono_extract_pkg/scripts/extract_mono_video.py \
    --video /path/to/video.mp4 \
    --output_dir data/different_types/my_mono_cloth
```

Optional fallback: run DA3 without VGGT.

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3.git /path/to/depth-anything-3
cd /path/to/depth-anything-3
pip install -e .

python data_process/mono_extract_pkg/scripts/extract_mono_video.py \
    --video /path/to/video.mp4 \
    --output_dir data/different_types/my_mono_cloth \
    --max_frames 120 \
    --no-use_vggt
```

Use the legacy MoGe2 per-frame depth path only when you explicitly want it:

```bash
python data_process/mono_extract_pkg/scripts/extract_mono_video.py \
    --video /path/to/video.mp4 \
    --output_dir data/different_types/my_mono_cloth \
    --moge_sequence_depth
```

Then run the standard data-processing pipeline from the repository root:

```bash
xvfb-run -a python data_process/script_process_data.py \
    --config configs/data_process/data_config_mono.csv \
    --cams 0
```

## Output Format

```text
output_dir/
  color/0/*.png
  color/0.mp4
  depth/0/*.npy
  calibrate.pkl
  metadata.json
  c2w_sequence.npy
```

`depth/0/*.npy` stores uint16 depth in millimeters. `calibrate.pkl` contains a
single camera-to-world matrix for camera 0, and `c2w_sequence.npy` stores the
per-frame camera trajectory when available.
