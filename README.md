# DeformMaster: An Interactive Physics-Neural World Model for Deformable Objects from Videos

[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=githubpages)](https://can-lee.github.io/deformmaster-web/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.09586-b31b1b?logo=arxiv)](https://arxiv.org/abs/2605.09586)
[![Demo ckpt & data](https://img.shields.io/badge/Demo_ckpt_%26_data-Hugging_Face-ffcc4d?logo=huggingface)](https://huggingface.co/datasets/Canlee/DeformMaster-Assets)

![DeformMaster teaser](assets/teaser.png)

## Release Status

- [x] **[2026/05/20]** Inference code
- [x] **[2026/05/20]** Online interaction code
- [x] **[2026/07/07]** Checkpoints
- [x] **[2026/07/07]** Most training code (remaining auxiliary pieces will be released later)
- [x] **[2026/07/07]** Custom data and preprocessing code
- [x] **[2026/07/07]** Full configurations
- [ ] Downstream embodied application

## 1. Environment

Create and activate the DeformMaster conda environment:

```bash
conda create -y -n deformmaster python=3.10
conda activate deformmaster

# Optional: set CUDA_HOME if nvcc is not already available.
# export CUDA_HOME=/path/to/cuda
# export PATH="${CUDA_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
```

Install the DeformMaster dependencies. If you only want to explore the
interactive playground, you can skip dependencies for data processing by using
`SKIP_OPT=1`:

```bash
SKIP_OPT=1 bash ./env_install/env_install.sh

pip install gradio==6.2.0 fastapi uvicorn
python -c "import torch, warp, gradio; print('env OK', torch.__version__)"
```

For full data processing and training, run `bash ./env_install/env_install.sh`
without `SKIP_OPT=1`.

## 2. Run The Online Demo

The current interactive demo is `interactive_playground_online.py`.

![Interactive playground](assets/playground.png)

Download `playground_assets.zip` (ckpt and data) from the [playground assets](https://huggingface.co/datasets/Canlee/DeformMaster-Assets) page into the repository root and unzip it:

```bash
unzip playground_assets.zip
```

The archive expands directly into:

```text
data/
outputs/
gaussian_output/
MANIFEST.txt
```

Run the softbody demo:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python interactive_playground_online.py \
    --case_name double_lift_sloth \
    --output outputs/output_ours
```

Run the monocular-cloth demo:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python interactive_playground_online.py \
    --case_name my_mono_cloth \
    --output outputs/output_mono
```

If the port is busy, the script picks the next free port and prints the URL.

Useful options:

```bash
--n_ctrl_parts 1          # one controller
--n_ctrl_parts 2          # two controllers
--bg_blank                # white background
--bg_mono                 # use ./data/bg_mono.jpg (default for my_mono_cloth)
--settle_iters N          # initial gravity-only simulation steps; default 220
--output_dir playground_recording/physics_flow
```

Browser controls:

```text
Mouse mode: click the rendered object to bind controller particles
Keyboard mode:
  W/A/S/D/Q/E   move controller 1
  I/J/K/L/U/O   move controller 2
  R             reset
```

Recordings are saved as:

```text
playground_recording/physics_flow/<case>/
  video.mp4
  controller.npy
  flow.npy
  calibrate.pkl
  metadata.json
```

## 3. Prepare Training Data

### PhysTwin Data

For the original multi-view training cases, download the
[PhysTwin](https://github.com/Jianghanxiao/PhysTwin/tree/main) data into the
repository root:

- [data](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip): original and processed data for different cases. Case names can be found under `data/different_types`.
- [gaussian_output](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/gaussian_output.zip): static Gaussian Splatting appearance results.

After extracting `data.zip`, training cases live under:

```text
data/different_types/<case_name>/
```

Each processed case should contain:

```text
final_data.pkl
calibrate.pkl
metadata.json
split.json
gt_track_3d.pkl
```

### Monocular Video (Custom)

Convert one RGB video to the expected data layout:

```bash
export DEFORMMASTER_VGGT_ROOT=/path/to/vggt-omega
export DEFORMMASTER_VGGT_CHECKPOINT=/path/to/vggt_omega_1b_512.pt
python data_process/mono_extract_pkg/scripts/extract_mono_video.py \
    --video data/mono_videos/cloth.mp4 \
    --output_dir data/different_types/my_mono_cloth
```

Then process it into the PhysTwin data layout:

```bash
python data_process/script_process_data.py \
    --config configs/data_process/data_config_mono.csv \
    --cams 0
```

Use `--no-segment` if masks were manually annotated with
`data_process/interactive_mask_app.py`:

```bash
python data_process/script_process_data.py \
    --config configs/data_process/data_config_mono.csv \
    --cams 0 \
    --no-segment
```

For details and troubleshooting, read `data_process/README.md`.

## 4. Training

Most training code used by DeformMaster is included in this release. Remaining
auxiliary pieces will be released in later updates.

### Select Actuator Gains

Actuator/controller-related parameters were tuned with CMA-ES and are already
written into the released YAML configs.

### Train Dynamics

Train physics-neural dynamics and run the default evaluation pipeline:

```bash
bash run_dynamics_training_eval.sh
```

Optional: pass an output tag and pipeline GPU, for example
`bash run_dynamics_training_eval.sh outputs/output_ours 0`. Category configs
can be overridden with `ROPE_CONFIG`, `CLOTH_CONFIG`, `SOFTBODY_CONFIG`, and
`PACKAGE_CONFIG`.

Dynamics checkpoints are written under each config's `output_dir`, for example:

```text
outputs/output_ours/cloth_warp/<scene>/best_checkpoint.pt
outputs/output_ours/cloth_warp/<scene>/final_checkpoint.pt
outputs/output_ours/cloth_warp/<scene>/config.yaml
```

### Train Appearance

Train static Gaussian Splatting appearance models from `data/gaussian_data`:

```bash
GPUS=0,1,2 bash scripts_training_eval/appearance/gs_run.sh
```

The script writes static appearance models under:

```text
gaussian_output/<scene>/
gaussian_output_video/<scene>/
```

### RGB-Guided Refinement

RGB-guided refinement runs from dynamics checkpoints using each config's
`stage3:` block:

```bash
bash run_rgb_guided_refinement.sh
```

Optional: pass `train` or `eval` to run only one phase, and set
`STAGE3_GPUS=0,1,2` to override the inference/render GPU pool.

RGB-guided refinement writes outputs under:

```text
outputs/output_ours_finetune/<category>_warp_finetune/<scene>/
outputs/output_ours_finetune/mpm_inference/<scene>/
outputs/output_ours_finetune/gaussian_output_dynamic_mpm/<scene>/
results/output_ours_finetune_*
```

## 5. Run Inference / Rendering / Evaluation Manually

Run inference for an existing training output:

```bash
python scripts_training_eval/dynamics/script_inference_mpm.py --input outputs/output_ours --gpu 0
```

Render dynamic Gaussian Splatting from `inference.pkl`:

```bash
bash scripts_training_eval/appearance/gs_run_simulation.sh outputs/output_ours/mpm_inference "" "" 0
```

Evaluate predictions:

```bash
bash scripts_training_eval/eval/evaluate.sh outputs/output_ours
```

Run the full Stage 2 eval pipeline without retraining:

```bash
bash scripts_training_eval/eval/run_dynamics_eval_pipeline.sh outputs/output_ours 0
```
