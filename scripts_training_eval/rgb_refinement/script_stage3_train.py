"""Per-category GPU-pool batch trainer for Stage-3 (RGB-finetune).

Each ``--group`` (= one config yaml) gets its own task queue and worker
pool, sized by that yaml's ``stage3.train.gpus`` and
``stage3.train.num_workers``. Categories run **independently in
parallel** — cloth tasks only land on cloth's GPUs, rope on rope's,
etc. This lets memory-heavy cases (e.g. dense Gaussian-fill softbody)
run alone on dedicated GPUs without competing with other categories.

CLI ``--gpus`` and ``--workers-per-gpu`` override every category to a
single global pool (legacy behaviour, useful for ad-hoc runs).

Stage-2 resume location is read from each config's top-level
``cfg.output_dir`` automatically.

Usage::

    python scripts_training_eval/rgb_refinement/script_stage3_train.py \\
        --group configs/cloth.yaml \\
        --group configs/rope.yaml \\
        --group configs/softbody.yaml
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time

from omegaconf import OmegaConf


def worker(gpu_id, task_queue, done_counter, total):
    gpu_id = str(gpu_id).strip()
    env_extra = {"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONUNBUFFERED": "1"}
    while True:
        # Blocking get + None sentinel: avoids the mp.Queue feeder race where
        # get_nowait() can raise Empty before put() items are visible to the
        # child process (previously caused idle GPUs at startup).
        task = task_queue.get()
        if task is None:
            break

        scene_id, config, resume_dir, iters = task
        resume = os.path.join(resume_dir, scene_id, "final_checkpoint.pt")
        if not os.path.exists(resume):
            with done_counter.get_lock():
                done_counter.value += 1
                n = done_counter.value
            print(f"[GPU {gpu_id}] !!! [{n}/{total}] MISSING {resume}", flush=True)
            continue

        t_start = time.time()
        print(f"\n[GPU {gpu_id}] >>> {scene_id} ({os.path.basename(config)})",
              flush=True)
        cmd = [
            sys.executable,
            "scripts_training_eval/rgb_refinement/train_rgb_finetune.py",
            "--case_name", scene_id,
            "--config", config,
            "--resume", resume,
            "--iters", str(iters),
        ]
        env = os.environ.copy()
        env.update(env_extra)
        try:
            subprocess.run(cmd, env=env, check=True)
            outcome = "Finished"
        except subprocess.CalledProcessError:
            outcome = "FAILED"
        with done_counter.get_lock():
            done_counter.value += 1
            n = done_counter.value
        elapsed = time.time() - t_start
        print(f"[GPU {gpu_id}] <<< [{n}/{total}] {outcome} {scene_id}  ({elapsed:.0f}s)",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", required=True,
                        help="config path (repeat for multiple categories). "
                             "Stage-2 resume dir is derived from cfg.output_dir.")
    parser.add_argument("--gpus", default=None,
                        help="Global GPU override. If set, every category "
                             "uses this single shared pool instead of its "
                             "own stage3.train.gpus.")
    parser.add_argument("--workers-per-gpu", default=None,
                        help="Global workers-per-GPU override. If set, every "
                             "category uses this value instead of its own "
                             "stage3.train.num_workers.")
    args = parser.parse_args()

    configs = list(args.group)
    if not configs:
        raise ValueError("no --group provided")

    # Build per-category groups: each gets its own queue + worker pool.
    # Group fields: name, tasks, gpus, num_workers
    groups = []
    total = 0
    for config in configs:
        cfg = OmegaConf.load(config)
        iters = int(OmegaConf.select(cfg, "stage3.train.iters") or 30)
        scenes = list(OmegaConf.select(cfg, "target_scenes") or [])
        # Stage-2 ckpts live under top-level output_dir (Stage-1+2 main config).
        resume_dir = OmegaConf.select(cfg, "output_dir")
        if not resume_dir:
            print(f"[WARN] no output_dir in {config}, skipping")
            continue
        if not scenes:
            print(f"[WARN] no target_scenes in {config}")
            continue

        if args.gpus:
            gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        else:
            gpus_str = str(OmegaConf.select(cfg, "stage3.train.gpus") or "0")
            gpus = [g.strip() for g in gpus_str.split(",") if g.strip()]
        if args.workers_per_gpu is not None:
            nw = int(args.workers_per_gpu)
        else:
            nw = int(OmegaConf.select(cfg, "stage3.train.num_workers") or 1)

        tasks = [(sid, config, resume_dir, iters) for sid in scenes]
        groups.append({
            "name": os.path.basename(config),
            "tasks": tasks,
            "gpus": gpus,
            "nw": nw,
        })
        total += len(scenes)
        print(f"[stage3] {config}: {len(scenes)} scenes "
              f"(iters={iters}, gpus={gpus}, nw={nw}, resume_dir={resume_dir})")

    if not groups:
        raise ValueError("no groups have tasks; nothing to run")

    print(f"\n[stage3] Total {total} cases across {len(groups)} categories "
          f"(per-category GPU pools)\n")

    done_counter = mp.Value('i', 0)
    t0 = time.time()

    workers = []
    queues = []  # keep refs alive in parent until join
    for grp in groups:
        q = mp.Queue()
        queues.append(q)
        for t in grp["tasks"]:
            q.put(t)
        n_workers = len(grp["gpus"]) * grp["nw"]
        for _ in range(n_workers):
            q.put(None)  # one sentinel per worker to terminate cleanly
        for gpu_id in grp["gpus"]:
            for _ in range(grp["nw"]):
                p = mp.Process(target=worker,
                               args=(gpu_id, q, done_counter, total))
                p.start()
                workers.append(p)

    for p in workers:
        p.join()

    print(f"\n[stage3] All done. {done_counter.value}/{total} cases  "
          f"(wall {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
