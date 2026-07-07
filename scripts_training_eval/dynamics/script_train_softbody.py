import os
import glob
import argparse
import multiprocessing as mp
import subprocess
import signal
from queue import Empty
import time

import sys

from omegaconf import OmegaConf


def _list_descendants(pid):
    """Recursively walk /proc/<pid>/task/<pid>/children to enumerate all
    descendant pids. Linux-only but stdlib (no psutil dep)."""
    out, stack = [], [pid]
    while stack:
        p = stack.pop()
        try:
            with open(f"/proc/{p}/task/{p}/children") as f:
                children = [int(x) for x in f.read().split()]
        except FileNotFoundError:
            continue
        out.extend(children)
        stack.extend(children)
    return out


def _install_sigkill_all_handler():
    """One Ctrl-C kills every descendant (workers + their subprocesses).
    SIGKILL bypasses any SIGINT handler the children may have installed."""
    def _handler(signum, frame):
        print("\n[INTERRUPT] killing all worker processes...", flush=True)
        for pid in _list_descendants(os.getpid()):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        os._exit(130)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

def load_runtime_config(config_path):
    cfg = OmegaConf.load(config_path)
    missing = []

    data_root = OmegaConf.select(cfg, "data.root")
    if not data_root:
        missing.append("data.root")
    iters = OmegaConf.select(cfg, "train.iters")
    if iters is None:
        missing.append("train.iters")
    gpus = OmegaConf.select(cfg, "train.gpus")
    if not gpus:
        missing.append("train.gpus")
    num_workers = OmegaConf.select(cfg, "train.num_workers")
    if num_workers is None:
        missing.append("train.num_workers")

    if missing:
        raise ValueError(f"Missing required config fields in {config_path}: {', '.join(missing)}")

    return cfg, str(data_root), int(iters), str(gpus), int(num_workers)


def worker(gpu_id, task_queue, config_path, iters):
    gpu_id = str(gpu_id).strip()
    while True:
        try:
            scene_id = task_queue.get_nowait()
        except Empty:
            break
            
        print(f"\n[GPU {gpu_id}] >>> [SOFTBODY MODE] Training Scene: {scene_id} using {config_path}", flush=True)
        
        # [FIX] Use sys.executable to ensure same python environment is used in subprocess
        cmd = ["python", "scripts_training_eval/dynamics/train_mpm.py", "--case_name", scene_id, "--iters", str(iters), "--config", config_path]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[GPU {gpu_id}] <<< Finished {scene_id} Successfully", flush=True)
        except subprocess.CalledProcessError:
            print(f"[GPU {gpu_id}] !!! Error in {scene_id}", flush=True)
        finally:
            task_queue.task_done()

def main():
    _install_sigkill_all_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    cfg, base_path, iters, gpus, num_workers = load_runtime_config(args.config)
    scenes_paths = sorted(glob.glob(os.path.join(base_path, "*")))
    all_scenes = [os.path.basename(p) for p in scenes_paths if os.path.isdir(p)]

    if 'target_scenes' in cfg and cfg.target_scenes:
        targets = list(cfg.target_scenes)
        print(f"Filter: Applying target_scenes from config: {targets}")
        valid_scenes = [s for s in all_scenes if s in targets]
    else:
        # Fallback: default softbody keywords
        valid_scenes = [s for s in all_scenes
                        if any(kw in s.lower() for kw in ['zebra', 'sloth', 'dinosor'])]

    print(f"Found {len(valid_scenes)} SOFTBODY scenes. Using GPU(s): {gpus}. Config: {args.config}", flush=True)
    task_queue = mp.JoinableQueue()
    for scene_id in valid_scenes:
        task_queue.put(scene_id)

    gpu_list = gpus.split(",")
    processes = []
    if os.name != 'nt': mp.set_start_method('fork', force=True)

    for gpu_id in gpu_list:
        for i in range(num_workers):
            p = mp.Process(target=worker, args=(gpu_id, task_queue, args.config, iters))
            p.start()
            processes.append(p)
            time.sleep(2.0)

    try:
        while not task_queue.empty() or any(p.is_alive() for p in processes):
            for p in processes: p.join(timeout=1.0)
            if task_queue.empty() and not any(p.is_alive() for p in processes): break
    except KeyboardInterrupt:
        for p in processes: p.terminate()
        os._exit(1)
    inference_input = os.path.dirname(os.path.normpath(str(cfg.output_dir)))
    print("\n[ALL DONE] All softbody scenes processed.", flush=True)
    print("Next step: export MPM inference with:", flush=True)
    print(f"  python scripts_training_eval/dynamics/script_inference_mpm.py --gpu 0 --input {inference_input}", flush=True)

if __name__ == "__main__":
    main()
