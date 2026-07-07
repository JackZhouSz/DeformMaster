# Data Processing

This directory contains the tools for turning raw RGB-D or monocular
observations into DeformMaster training cases. The main training input is:

```text
data/different_types/<case_name>/final_data.pkl
```

Each processed case should also contain `calibrate.pkl`, `metadata.json`,
`split.json`, and `gt_track_3d.pkl`.

## Quick Commands

### Multi-Camera RGB-D Capture

Calibrate cameras once after fixing the camera rig:

```bash
python data_process/capture_calib_frames.py --output-dir calib_frames/
python data_process/calibrate_charuco.py \
    --pose-dir calib_frames/ \
    --intrinsics calib_frames/intrinsics.json \
    --output calib_frames/calibrate.pkl
```

Record one case:

```bash
python data_process/record_multi_cam.py \
    --output recorded_data/<case_name> \
    --calibrate-pkl calib_frames/calibrate.pkl \
    --frames <N> \
    --start-delay 5 \
    --preview
```

Validate calibration immediately after recording:

```bash
python data_process/verify_multicam_calibration.py \
    recorded_data/<case_name> \
    --per-cam-color
```

Copy or move the recorded case into:

```text
data/different_types/<case_name>/
```

### Monocular Video

Convert one RGB video into the same case layout:

```bash
export DEFORMMASTER_VGGT_ROOT=/path/to/vggt-omega
export DEFORMMASTER_VGGT_CHECKPOINT=/path/to/vggt_omega_1b_512.pt
python data_process/mono_extract_pkg/scripts/extract_mono_video.py \
    --video data/mono_videos/cloth.mp4 \
    --output_dir data/different_types/my_mono_cloth
```

By default this uses VGGT-Omega temporal depth and anchors metric scale with
MoGe2 frame 0. See `data_process/mono_extract_pkg/README.md` for more options.

## Batch Processing

Create a CSV file with no header:

```text
case_name,category,shape_prior,regularize,n_keypoints
```

Example:

```bash
cat > configs/data_process/data_config.csv <<'EOF'
case1,cloth,True,True,16
case2,rope,False,True,9
EOF
```

Run the processing pipeline:

```bash
xvfb-run -a python data_process/script_process_data.py \
    --config configs/data_process/data_config.csv
```

Useful options:

```bash
--gpus 0,1,2,3        # parallel GPU workers
--gpus ''             # serial mode
--cams 0              # monocular cases
--no-segment          # reuse manually authored masks
--no-overwrite        # keep existing intermediate outputs
```

`xvfb-run -a` is recommended on headless servers because Open3D and PyVista need
an X display for some visualization steps.

## Manual Masks

If GroundingDINO or SAM2 fails, annotate masks manually:

```bash
python data_process/interactive_mask_app.py \
    --base_path data/different_types \
    --case_name <case_name> \
    --camera_idx 0 \
    --port 8890
```

For a remote server, forward the port:

```bash
ssh -L 8890:localhost:8890 <user>@<server>
```

Then rerun the batch pipeline with:

```bash
--no-segment
```

Manual masks follow the same output contract as the automatic segmenter:

```text
mask/0/<obj_id>/<frame>.png
mask/metadata.json
```

## Shape Priors

When `shape_prior=True`, the pipeline expects or generates:

```text
<case_name>/shape/object.glb
```

Shape priors come from either:

- Presets matched by case/category keyword, for example
  `data/mesh/best_rope.glb`.
- TRELLIS, via `image_upscale.py`, `segment_util_image.py`, and
  `shape_prior.py`.

Override or add presets with:

```bash
--shape-prior-preset "KEYWORD:/path/to/object.glb"
```

For thin or textureless objects, automatic shape-prior alignment can be fragile.
In those cases, `shape_prior=False` and point-cloud-only training may be more
reliable.

## Pipeline Stages

The standard processing pipeline runs:

```text
1. segment.py
2. image_upscale.py                  optional, shape-prior path
3. segment_util_image.py             optional, shape-prior path
4. shape_prior.py                    optional, TRELLIS shape prior
5. dense_track.py
6. data_process_pcd.py
7. data_process_mask.py
8. data_process_track.py
9. align.py                          optional, shape-prior path
10. data_process_sample.py
11. derive_gt_track_3d.py
```

You can run these scripts manually for debugging, but
`script_process_data.py` is the preferred entry point for release use.

## `final_data.pkl` Schema

The final training file stores tensors such as:

```python
{
    "object_points": (T, N, 3),
    "controller_points": (T, C, 3),
    "surface_points": (S, 3),
    "interior_points": (I, 3),
}
```

Additional case artifacts include:

```text
split.json
mask/processed_masks.pkl
pcd/
cotracker/
shape/
track_process_data.pkl
gt_track_3d.pkl
gt_track_3d.mp4
```

`controller_mask.npy` is a legacy artifact and is not required for new cases.
Current configs normally keep all controller markers.

## Gaussian Data

Export first-frame 3DGS training data:

```bash
python data_process/export_gaussian_data.py --case_name <case_name>
```

The output lives under:

```text
data/gaussian_data/<case_name>/
```

Train static 3DGS with the scripts in `gaussian_splatting/` or the helper
scripts under `data_process/` when available. Static Gaussian outputs are read
from:

```text
gaussian_output/<case_name>/<experiment_name>/
```

## Pre-Training Checks

Before dynamics training:

- Add the case to the appropriate config under `configs/`.
- Confirm `final_data.pkl`, `split.json`, and `gt_track_3d.pkl` exist.
- Inspect point clouds and controller locations.
- Run a single-case smoke training command before launching a large batch.

Example:

```bash
python scripts_training_eval/dynamics/script_train_mpm.py \
    --case_name <case_name> \
    --config configs/cloth.yaml
```

## Common Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Empty or noisy point cloud | Bad depth or RGB-depth alignment | Recheck capture and calibration |
| Multi-camera clouds do not overlap | Incorrect extrinsics | Recalibrate cameras |
| Missing controller points | Hand/controller mask failed | Improve recording or use manual masks |
| Shape prior alignment fails | Poor mask or weak visual features | Disable shape prior or provide a better preset |
| Open3D crashes on a server | No X display | Run with `xvfb-run -a` |
| TRELLIS import/checkpoint errors | Optional dependencies missing | Install optional data-processing stack |
