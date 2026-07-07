#!/bin/bash

set -euo pipefail

# Render GS dynamics driven by flat MPM inference outputs:
#   {prediction_dir}/{scene_name}/inference.pkl
#
# Usage:
#   bash scripts_training_eval/appearance/gs_run_simulation.sh ./output_2/mpm_inference
#   bash scripts_training_eval/appearance/gs_run_simulation.sh ./output_2/mpm_inference "single_push_rope,single_lift_rope"
#   bash scripts_training_eval/appearance/gs_run_simulation.sh ./output_2/mpm_inference "single_push_rope" ./output_2/gaussian_output_dynamic_mpm
#   bash scripts_training_eval/appearance/gs_run_simulation.sh ./output_2/mpm_inference "" "" 0
#   bash scripts_training_eval/appearance/gs_run_simulation.sh ./output_2/mpm_inference "" "" 0 --white_background

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts_training_eval/appearance/gs_run_simulation.sh <prediction_dir> [scenes_csv] [output_dir] [gpu] [--white_background]"
    exit 1
fi

prediction_dir="${1}"
shift

white_background=0
positional_args=()
for arg in "$@"; do
    case "${arg}" in
        --white_background)
            white_background=1
            ;;
        *)
            positional_args+=("${arg}")
            ;;
    esac
done

scenes_arg="${positional_args[0]:-}"
output_dir="${positional_args[1]:-}"
gpu_arg="${positional_args[2]:-}"

if [ -n "${gpu_arg}" ]; then
    export CUDA_VISIBLE_DEVICES="${gpu_arg}"
fi

if [ -z "${output_dir}" ]; then
    prediction_parent="$(dirname "${prediction_dir}")"
    if [ "${white_background}" -eq 1 ]; then
        output_dir="${prediction_parent}/gaussian_output_dynamic_mpm_white"
    else
        output_dir="${prediction_parent}/gaussian_output_dynamic_mpm"
    fi
fi

views=("0")
exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

if [ -n "${scenes_arg}" ]; then
    IFS=',' read -ra scenes <<< "${scenes_arg}"
else
    echo "Auto-detecting scenes from ${prediction_dir} ..."
    scenes=()
    if [ -d "${prediction_dir}" ]; then
        for scene_dir in "${prediction_dir}"/*; do
            if [ ! -d "${scene_dir}" ] || [ ! -f "${scene_dir}/inference.pkl" ]; then
                continue
            fi

            scene_name="$(basename "${scene_dir}")"
            gaussian_data_dir="./data/gaussian_data/${scene_name}"
            gaussian_model_dir="./gaussian_output/${scene_name}/${exp_name}"

            if [ ! -d "${gaussian_data_dir}" ]; then
                echo "  Skipping ${scene_name} (missing gaussian data: ${gaussian_data_dir})"
                continue
            fi
            if [ ! -d "${gaussian_model_dir}" ]; then
                echo "  Skipping ${scene_name} (missing gaussian model: ${gaussian_model_dir})"
                continue
            fi

            scenes+=("${scene_name}")
            echo "  Found: ${scene_name}"
        done
    fi

    if [ "${#scenes[@]}" -eq 0 ]; then
        echo "Error: No valid scenes found in ${prediction_dir}"
        echo "Expected ${prediction_dir}/<scene_name>/inference.pkl plus matching gaussian data/model."
        exit 1
    fi
fi

mkdir -p "${output_dir}"

echo "Using prediction directory: ${prediction_dir}"
echo "Output directory: ${output_dir}"
if [ "${white_background}" -eq 1 ]; then
    echo "Using white Gaussian render background"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "Using CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo "Processing scenes: ${scenes[*]}"

for scene_name in "${scenes[@]}"; do
    gaussian_data_dir="./data/gaussian_data/${scene_name}"
    gaussian_model_dir="./gaussian_output/${scene_name}/${exp_name}"
    scene_video_path="${output_dir}/${scene_name}/0.mp4"

    if [ ! -f "${prediction_dir}/${scene_name}/inference.pkl" ]; then
        echo "Skipping ${scene_name} (missing inference: ${prediction_dir}/${scene_name}/inference.pkl)"
        continue
    fi
    if [ ! -d "${gaussian_data_dir}" ]; then
        echo "Skipping ${scene_name} (missing gaussian data: ${gaussian_data_dir})"
        continue
    fi
    if [ ! -d "${gaussian_model_dir}" ]; then
        echo "Skipping ${scene_name} (missing gaussian model: ${gaussian_model_dir})"
        continue
    fi
    if [ -f "${scene_video_path}" ]; then
        echo "Skipping ${scene_name} (already rendered: ${scene_video_path})"
        continue
    fi

    render_cmd=(
        python scripts_training_eval/appearance/gs_render_dynamics.py
        -s "${gaussian_data_dir}" \
        -m "${gaussian_model_dir}" \
        --name "${scene_name}" \
        --prediction_dir "${prediction_dir}" \
        --output_dir "${output_dir}"
    )
    if [ "${white_background}" -eq 1 ]; then
        render_cmd+=(--white_background)
    fi
    "${render_cmd[@]}"

    for view_name in "${views[@]}"; do
        python gaussian_splatting/img2video.py \
            --image_folder "${output_dir}/${scene_name}/${view_name}" \
            --video_path "${output_dir}/${scene_name}/${view_name}.mp4"
    done
done

echo ""
echo "Evaluation command:"
echo "  bash scripts_training_eval/eval/evaluate.sh $(dirname "${prediction_dir}")"
