import os
import glob
import argparse
import multiprocessing as mp
import subprocess
from queue import Empty
from omegaconf import OmegaConf

"""
Example:
python scripts_training_eval/dynamics/script_train_mpm.py --gpus '0,1,2,3,4' --num_workers 3
"""

def worker(gpu_id, task_queue, config, default_iters):
    """Worker process: picks a scene from the queue and runs it on a specific GPU."""
    gpu_id = str(gpu_id).strip()
    while True:
        try:
            scene_id = task_queue.get_nowait()
        except Empty:
            break
            
        # [NEW] Determine config file based on scene_id keywords
        if any(kw in scene_id.lower() for kw in ['rope']):
            selected_config = "configs/rope.yaml"
        elif 'package' in scene_id.lower():
            selected_config = "configs/package.yaml"
        elif 'cloth' in scene_id.lower():
            selected_config = "configs/cloth.yaml"
        elif any(kw in scene_id.lower() for kw in ['zebra', 'sloth', 'dinosor']):
            selected_config = "configs/softbody.yaml"
        else:
            selected_config = config # Fallback to command line argument
            
        # [NEW] Load iters from YAML if available, otherwise use default_iters
        actual_iters = default_iters
        if os.path.exists(selected_config):
            try:
                cfg_yaml = OmegaConf.load(selected_config)
                if 'iters' in cfg_yaml:
                    actual_iters = cfg_yaml.iters
                    print(f"[GPU {gpu_id}] Found 'iters: {actual_iters}' in {selected_config}, overriding default.")
            except Exception as e:
                print(f"[GPU {gpu_id}] Warning: Could not parse iters from {selected_config}: {e}")

        print(f"\n[GPU {gpu_id}] >>> Starting MPM Training for Scene: {scene_id} using {selected_config} (iters: {actual_iters})", flush=True)
        
        # Build command
        cmd = [
            "python", "scripts_training_eval/dynamics/train_mpm.py",
            "--case_name", scene_id,
            "--iters", str(actual_iters),
            "--config", selected_config
        ]
        
        # Set CUDA_VISIBLE_DEVICES for this specific subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        try:
            # Using subprocess for better control and log capture
            subprocess.run(cmd, env=env, check=True)
            print(f"[GPU {gpu_id}] <<< Finished Scene: {scene_id} Successfully", flush=True)
        except subprocess.CalledProcessError:
            print(f"[GPU {gpu_id}] !!! Error occurred while training scene: {scene_id}", flush=True)
        finally:
            task_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="Batch training for MPM (Supports Multi-GPU Parallelism)")
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--config", type=str, default="configs/cloth.yaml")
    parser.add_argument("--iters", type=int, default=None) # Default to None to check config
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g., '0,1,2,3,4,5,6,7')")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers per GPU")
    args = parser.parse_args()
    
    # 0. Load Base Config if available to set defaults
    base_iters = 10
    base_gpus = "0,1,2,3,4,5,6,7"
    base_num_workers = 1
    
    if os.path.exists(args.config):
        try:
            cfg = OmegaConf.load(args.config)
            if 'train' in cfg:
                if 'iters' in cfg.train: base_iters = cfg.train.iters
                if 'gpus' in cfg.train: base_gpus = str(cfg.train.gpus)
                if 'num_workers' in cfg.train: base_num_workers = int(cfg.train.num_workers)
            # Support top-level iters too
            if 'iters' in cfg: base_iters = cfg.iters
        except Exception as e:
            print(f"Warning: Could not load base config {args.config} for defaults: {e}")

    # Override defaults with command line args if provided
    final_iters = args.iters if args.iters is not None else base_iters
    final_gpus = args.gpus if args.gpus is not None else base_gpus
    final_num_workers = args.num_workers if args.num_workers is not None else base_num_workers

    # 1. Collect all valid scenes
    scenes_paths = sorted(glob.glob(os.path.join(args.base_path, "*")))
    valid_scenes = []
    for scene_path in scenes_paths:
        if os.path.isdir(scene_path) and os.path.exists(os.path.join(scene_path, "final_data.pkl")):
            valid_scenes.append(os.path.basename(scene_path))

    # [NEW] Check for target_scenes in config
    if os.path.exists(args.config):
        try:
            cfg = OmegaConf.load(args.config)
            if 'target_scenes' in cfg and cfg.target_scenes:
                targets = list(cfg.target_scenes)
                print(f"Filter: Applying target_scenes from config {args.config}: {targets}")
                valid_scenes = [s for s in valid_scenes if s in targets]
        except Exception as e:
            pass

    if not valid_scenes:
        print(f"No valid scenes found in {args.base_path} (after filter)!")
        return

    print(f"Found {len(valid_scenes)} valid scenes. Using GPU(s): {final_gpus}, iters: {final_iters}, workers_per_gpu: {final_num_workers}", flush=True)

    # 2. Setup Task Queue
    task_queue = mp.JoinableQueue()
    for scene_id in valid_scenes:
        task_queue.put(scene_id)

    # 3. Start Workers (one per GPU ID)
    gpu_list = final_gpus.split(",")
    processes = []
    import time
    
    # Use 'fork' on Linux for faster process creation. 
    # It's safe since we haven't initialized CUDA in the parent.
    if os.name != 'nt':
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass

    for gpu_id in gpu_list:
        gpu_id = gpu_id.strip()
        for i in range(final_num_workers):
            p = mp.Process(target=worker, args=(gpu_id, task_queue, args.config, final_iters))
            p.start()
            processes.append(p)
            time.sleep(0.5) # Give some breathing room for initialization and logging

    # 4. Wait for completion
    try:
        # Use a timeout-based wait loop to allow KeyboardInterrupt to be caught on some systems
        while not task_queue.empty() or any(p.is_alive() for p in processes):
            for p in processes:
                p.join(timeout=1.0)
            if task_queue.empty() and not any(p.is_alive() for p in processes):
                break
    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt detected. Terminating all workers...", flush=True)
        for p in processes:
            p.terminate()
        # Use os._exit to force exit immediately and avoid being caught in cleanup loops
        os._exit(1)

    print("\n[ALL DONE] All scenes processed.", flush=True)

if __name__ == "__main__":
    main()
