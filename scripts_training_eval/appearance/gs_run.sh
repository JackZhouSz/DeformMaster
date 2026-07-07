output_dir="./gaussian_output"
output_video_dir="./gaussian_output_video"
scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
        "double_stretch_sloth" "double_stretch_zebra"
        "rope_double_hand"
        "single_clift_cloth_1" "single_clift_cloth_3"
        "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
        "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
        "single_push_rope" "single_push_rope_1" "single_push_rope_4"
        "single_push_sloth"
        "weird_package"
        "my_mono_cloth")

exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

# GPU pool — comma-separated. Override: GPUS=2,3,4 bash scripts_training_eval/appearance/gs_run.sh
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPU=${#GPU_ARR[@]}
echo "[gs_run] GPU pool: ${GPU_ARR[*]} (${N_GPU} workers)"

python ./gaussian_splatting/generate_interp_poses.py

train_one() {
    local scene_name="$1"
    local gpu="$2"
    local port="$3"
    echo "[GPU ${gpu}] >>> train ${scene_name} (port ${port})"
    CUDA_VISIBLE_DEVICES=${gpu} python scripts_training_eval/appearance/gs_train.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ${output_dir}/${scene_name}/${exp_name} \
        --port ${port} \
        --disable_viewer \
        --iterations 10000 \
        --lambda_depth 0.001 \
        --lambda_normal 0.0 \
        --lambda_anisotropic 0.0 \
        --lambda_seg 1.0 \
        --use_masks \
        --isotropic \
        --gs_init_opt 'hybrid' || { echo "[GPU ${gpu}] !!! ${scene_name} train FAILED"; return 1; }
    echo "[GPU ${gpu}] <<< train ${scene_name} OK"
}

render_one() {
    local scene_name="$1"
    local gpu="$2"
    echo "[GPU ${gpu}] >>> render ${scene_name}"
    CUDA_VISIBLE_DEVICES=${gpu} python scripts_training_eval/appearance/gs_render.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ${output_dir}/${scene_name}/${exp_name} || { echo "[GPU ${gpu}] !!! ${scene_name} render FAILED"; return 1; }
    python gaussian_splatting/img2video.py \
        --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
        --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
    echo "[GPU ${gpu}] <<< render ${scene_name} OK"
}

# Pass STAGE to skip a phase (e.g. STAGE=render bash scripts_training_eval/appearance/gs_run.sh re-runs only render)
STAGE="${STAGE:-all}"

# ---- Phase 1: training (parallel across GPU pool) ----
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "train" ]; then
    PORT_BASE="${PORT_BASE:-6010}"
    for i in "${!scenes[@]}"; do
        gpu=${GPU_ARR[$((i % N_GPU))]}
        port=$((PORT_BASE + i))
        train_one "${scenes[$i]}" "${gpu}" "${port}" &
        while [ "$(jobs -r | wc -l)" -ge "$N_GPU" ]; do sleep 2; done
    done
    wait
    echo "[gs_run] training phase done"
fi

# ---- Phase 2: render + img2video (serial, avoids gsplat JIT race) ----
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "render" ]; then
    for i in "${!scenes[@]}"; do
        gpu=${GPU_ARR[$((i % N_GPU))]}
        render_one "${scenes[$i]}" "${gpu}"
    done
    echo "[gs_run] render phase done"
fi

echo "[gs_run] all done"
