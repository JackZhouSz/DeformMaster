"""Multi-GPU parallel inference for Stage-3.

Same shared-queue pattern as script_stage3_train.py. Each worker pops a
case from the queue, runs inference_mpm.py on its assigned GPU.

Stage-3 ckpt dir is derived from ``cfg.stage3.output_dir``; no need to pass
it on the command line. ``--output-dir`` is the aggregate directory where
all per-scene inference.pkl files land.

Usage::

    python scripts_training_eval/rgb_refinement/script_stage3_inference.py \
        --group configs/cloth.yaml \
        --group configs/rope.yaml \
        --group configs/softbody.yaml \
        --output-dir output_custom/mpm_inference \
        --gpus 2,3,4,5,6,7
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from queue import Empty

from omegaconf import OmegaConf


def worker(gpu_id, task_queue, output_dir, done_counter, total):
    gpu_id = str(gpu_id).strip()
    env_extra = {"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONUNBUFFERED": "1"}
    def _bump():
        with done_counter.get_lock():
            done_counter.value += 1
            return done_counter.value
    while True:
        try:
            scene_id, config, ckpt_dir = task_queue.get_nowait()
        except Empty:
            break

        ckpt = os.path.join(ckpt_dir, scene_id, "final_checkpoint.pt")
        if not os.path.exists(ckpt):
            n = _bump()
            print(f"[GPU {gpu_id}] !!! [{n}/{total}] MISSING {ckpt}", flush=True)
            continue

        out_pkl = os.path.join(output_dir, scene_id, "inference.pkl")
        if os.path.exists(out_pkl):
            n = _bump()
            print(f"[GPU {gpu_id}] [{n}/{total}] skip {scene_id} (inference.pkl exists)",
                  flush=True)
            continue

        t_start = time.time()
        print(f"[GPU {gpu_id}] >>> inference {scene_id}", flush=True)
        cmd = [
            sys.executable,
            "scripts_training_eval/dynamics/inference_mpm.py",
            "--case_name", scene_id,
            "--config", config,
            "--checkpoint", ckpt,
            "--output_dir", output_dir,
            "--gpu", "0",
        ]
        env = os.environ.copy()
        env.update(env_extra)
        try:
            subprocess.run(cmd, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            outcome = "OK"
            extra = ""
        except subprocess.CalledProcessError as e:
            outcome = "FAILED"
            extra = f": {e.stderr[-200:].decode(errors='ignore')}" if e.stderr else ""
        n = _bump()
        elapsed = time.time() - t_start
        print(f"[GPU {gpu_id}] <<< [{n}/{total}] {scene_id} {outcome}  ({elapsed:.0f}s){extra}",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", required=True,
                        help="config path (Stage-3 ckpt dir is read from cfg.stage3.output_dir)")
    parser.add_argument("--output-dir", default="output_custom/mpm_inference")
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    args = parser.parse_args()

    configs = list(args.group)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    tasks = []
    for config in configs:
        cfg = OmegaConf.load(config)
        scenes = list(OmegaConf.select(cfg, "target_scenes") or [])
        ckpt_dir = OmegaConf.select(cfg, "stage3.output_dir")
        if not ckpt_dir:
            print(f"[WARN] {config} has no stage3.output_dir, skipping")
            continue
        for sid in scenes:
            tasks.append((sid, config, ckpt_dir))

    os.makedirs(args.output_dir, exist_ok=True)
    total = len(tasks)
    print(f"[inference] {total} cases on GPUs={gpus}")

    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    done_counter = mp.Value('i', 0)
    t0 = time.time()

    workers = []
    for gpu_id in gpus:
        p = mp.Process(target=worker,
                       args=(gpu_id, task_queue, args.output_dir, done_counter, total))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
    print(f"[inference] done. {done_counter.value}/{total} cases  "
          f"(wall {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
