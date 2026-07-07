"""Multi-GPU parallel GS dynamic rendering for Stage-3.

Each worker pops a case and runs gs_render_dynamics.py on its GPU using
the baseline pretrained GS (freeze_gs=true means GS unchanged) and the
Stage-3 particle trajectories from inference.

Usage::

    python scripts_training_eval/rgb_refinement/script_stage3_render.py \
        --inference-dir output_custom/mpm_inference \
        --output-dir output_custom/gaussian_output_dynamic_mpm \
        --gpus 2,3,4,5,6,7
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from queue import Empty

GS_EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"


def worker(gpu_id, task_queue, inference_dir, output_dir, done_counter, total):
    gpu_id = str(gpu_id).strip()
    env_extra = {"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONUNBUFFERED": "1"}
    def _bump():
        with done_counter.get_lock():
            done_counter.value += 1
            return done_counter.value
    while True:
        try:
            scene_id = task_queue.get_nowait()
        except Empty:
            break

        out_check = os.path.join(output_dir, scene_id, "0")
        if os.path.isdir(out_check) and len(os.listdir(out_check)) > 0:
            n = _bump()
            print(f"[GPU {gpu_id}] [{n}/{total}] skip {scene_id} (already rendered)",
                  flush=True)
            continue

        gs_model = os.path.join("gaussian_output", scene_id, GS_EXP_NAME)
        gs_data = os.path.join("data", "gaussian_data", scene_id)
        if not os.path.isdir(gs_model):
            n = _bump()
            print(f"[GPU {gpu_id}] !!! [{n}/{total}] no gaussian_output for {scene_id}",
                  flush=True)
            continue
        if not os.path.isdir(gs_data):
            n = _bump()
            print(f"[GPU {gpu_id}] !!! [{n}/{total}] no gaussian_data for {scene_id}",
                  flush=True)
            continue

        t_start = time.time()
        print(f"[GPU {gpu_id}] >>> render {scene_id}", flush=True)
        cmd = [
            sys.executable,
            "scripts_training_eval/appearance/gs_render_dynamics.py",
            "-s", gs_data,
            "-m", gs_model,
            "--iteration", "10000",
            "--name", scene_id,
            "--prediction_dir", inference_dir,
            "--output_dir", output_dir,
        ]
        env = os.environ.copy()
        env.update(env_extra)
        log_path = f"/tmp/render10_{scene_id}.log"
        try:
            with open(log_path, "w") as log_f:
                subprocess.run(cmd, env=env, check=True,
                               stdout=log_f, stderr=subprocess.STDOUT)
            outcome = "OK"
            extra = ""
        except subprocess.CalledProcessError:
            outcome = "FAILED"
            extra = f" (see {log_path})"
        n = _bump()
        elapsed = time.time() - t_start
        print(f"[GPU {gpu_id}] <<< [{n}/{total}] {scene_id} {outcome}  ({elapsed:.0f}s){extra}",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inference-dir", default="output_custom/mpm_inference",
                        help="Dir with <case>/inference.pkl")
    parser.add_argument("--output-dir", default="output_custom/gaussian_output_dynamic_mpm")
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    # Discover cases from inference dir
    cases = []
    for entry in sorted(os.listdir(args.inference_dir)):
        pkl = os.path.join(args.inference_dir, entry, "inference.pkl")
        if os.path.isfile(pkl):
            cases.append(entry)

    os.makedirs(args.output_dir, exist_ok=True)
    total = len(cases)
    print(f"[render] {total} cases on GPUs={gpus}")

    task_queue = mp.Queue()
    for c in cases:
        task_queue.put(c)

    done_counter = mp.Value('i', 0)
    t0 = time.time()

    workers = []
    for gpu_id in gpus:
        p = mp.Process(target=worker, args=(
            gpu_id, task_queue, args.inference_dir, args.output_dir,
            done_counter, total))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
    print(f"[render] done. {done_counter.value}/{total} cases  "
          f"(wall {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
