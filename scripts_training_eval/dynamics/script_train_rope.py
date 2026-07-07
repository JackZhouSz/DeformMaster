import os
import sys
import glob
import argparse
import multiprocessing as mp
import subprocess
from queue import Empty
import time

from omegaconf import OmegaConf


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

        print(f"\n[GPU {gpu_id}] >>> Training Scene: {scene_id} using {config_path}", flush=True)

        cmd = [sys.executable, "scripts_training_eval/dynamics/train_mpm.py", "--case_name", scene_id, "--iters", str(iters), "--config", config_path]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    cfg, base_path, iters, gpus, num_workers = load_runtime_config(args.config)
    scenes_paths = sorted(glob.glob(os.path.join(base_path, "*")))
    # "twine" is rope-like and trained with the rope config
    valid_scenes = [os.path.basename(p) for p in scenes_paths if os.path.isdir(p)
                    and ("rope" in os.path.basename(p).lower()
                         or "twine" in os.path.basename(p).lower())]

    if 'target_scenes' in cfg and cfg.target_scenes:
        targets = list(cfg.target_scenes)
        print(f"Filter: Applying target_scenes from config: {targets}")
        valid_scenes = [s for s in valid_scenes if s in targets]

    print(f"Found {len(valid_scenes)} ROPE scenes. Using GPU(s): {gpus}. Config: {args.config}", flush=True)

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
            time.sleep(0.5)

    try:
        while not task_queue.empty() or any(p.is_alive() for p in processes):
            for p in processes: p.join(timeout=1.0)
            if task_queue.empty() and not any(p.is_alive() for p in processes): break
    except KeyboardInterrupt:
        for p in processes: p.terminate()
        os._exit(1)
    inference_input = os.path.dirname(os.path.normpath(str(cfg.output_dir)))
    print("\n[ALL DONE] All rope scenes processed.", flush=True)
    print("Next step: export MPM inference with:", flush=True)
    print(f"  python scripts_training_eval/dynamics/script_inference_mpm.py --gpu 0 --input {inference_input}", flush=True)

if __name__ == "__main__":
    main()
