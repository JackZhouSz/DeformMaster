"""
CMA-ES-based Actuator gain selection for controller PD gains (kp, kd).

For each CMA-ES sample, trains ALL target_scenes in the category
using the existing multi-GPU parallel training scripts, then reads
the average loss across all cases as fitness.

Usage:
    # Select cloth actuator gains (reads target_scenes + gpus from the yaml itself)
    python scripts_training_eval/actuator/actuator_gain_selection.py --base-config configs/cloth.yaml

    # Select rope actuator gains
    python scripts_training_eval/actuator/actuator_gain_selection.py --base-config configs/rope.yaml

    # Custom search params
    python scripts_training_eval/actuator/actuator_gain_selection.py --base-config configs/cloth.yaml \\
        --iters 30 --popsize 4 --max-generations 15

No trainer / simulator code changes. The wrapper only:
  1. Writes a temp yaml with overridden init_raw_ctrl_stiffness/damping
  2. Launches the appropriate script_train_*.py (multi-GPU parallel)
  3. Reads average Loss/Total from tensorboard across all cases
  4. Returns that as CMA-ES fitness

Dependencies:
    pip install cma
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import time

import cma
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator


# Map config tag → training script
TAG_TO_SCRIPT = {
    "cloth": "scripts_training_eval/dynamics/script_train_cloth.py",
    "rope": "scripts_training_eval/dynamics/script_train_rope.py",
    "softbody": "scripts_training_eval/dynamics/script_train_softbody.py",
    "package": "scripts_training_eval/dynamics/script_train_cloth.py",
}


def _read_avg_loss(output_dir: str, tag: str = "Loss/Total") -> float:
    """Read the average best loss across all cases under output_dir/*/mpm_train/."""
    pattern = os.path.join(output_dir, "*", "mpm_train", "*", "events.out.tfevents.*")
    events = glob.glob(pattern)
    if not events:
        return float("inf")

    # Group by case (one case can have multiple event files)
    case_best = {}
    for e in events:
        parts = e.split(os.sep)
        # output_dir / case_name / mpm_train / timestamp / events...
        case = parts[-4]
        try:
            ea = event_accumulator.EventAccumulator(e, size_guidance={"scalars": 0})
            ea.Reload()
            if tag in ea.Tags()["scalars"]:
                vals = [s.value for s in ea.Scalars(tag)]
                vals = [v for v in vals if not math.isnan(v)]
                if vals:
                    best = min(vals)
                    if case not in case_best or best < case_best[case]:
                        case_best[case] = best
        except Exception:
            pass

    if not case_best:
        return float("inf")

    # Average best loss across all cases
    avg = sum(case_best.values()) / len(case_best)
    return avg


def _train_all_cases(base_cfg, raw_ks, raw_kd, iters, output_root, sample_idx):
    """Train ALL target_scenes with overridden gains. Returns avg loss."""
    sample_dir = os.path.join(output_root, f"sample_{sample_idx:04d}")

    # Build per-sample cfg
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.mpm.init_raw_ctrl_stiffness = float(raw_ks)
    cfg.mpm.init_raw_ctrl_damping = float(raw_kd)
    cfg.train.iters = int(iters)

    # Determine tag and category output subdir
    tag = cfg.get("tag", "cloth")
    category_subdir = f"{tag}_warp"
    cfg.output_dir = os.path.join(sample_dir, category_subdir)

    # Write temp yaml
    yaml_path = os.path.join(sample_dir, "config.yaml")
    os.makedirs(sample_dir, exist_ok=True)
    OmegaConf.save(cfg, yaml_path)

    # Pick the right training script
    train_script = TAG_TO_SCRIPT.get(tag, "scripts_training_eval/dynamics/script_train_cloth.py")

    cmd = ["python", train_script, "--config", yaml_path]
    print(f"  [{tag}] → sample {sample_idx}: kp={raw_ks:+.3f} kd={raw_kd:+.3f}  "
          f"({len(cfg.get('target_scenes', []))} cases)", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [{tag}] ✗ sample {sample_idx} FAIL ({elapsed:.0f}s)", flush=True)
        with open(os.path.join(sample_dir, "stderr.log"), "w") as f:
            f.write(result.stderr)
        shutil.rmtree(sample_dir, ignore_errors=True)
        return float("inf")

    # Read average loss across all cases
    loss = _read_avg_loss(cfg.output_dir)
    n_cases = len(glob.glob(os.path.join(cfg.output_dir, "*", "mpm_train")))
    print(f"  [{tag}] ← sample {sample_idx}: loss={loss:.4f} ({n_cases} cases, {elapsed:.0f}s)", flush=True)

    # Clean up sample dir to save disk (result already recorded in history.json)
    shutil.rmtree(sample_dir, ignore_errors=True)
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", required=True,
                        help="Base yaml (e.g., configs/cloth.yaml). "
                             "target_scenes and gpus are read from this file.")
    parser.add_argument("--iters", type=int, default=30,
                        help="Trainer iters per fitness eval (default 30)")
    parser.add_argument("--popsize", type=int, default=4,
                        help="CMA-ES population size per generation")
    parser.add_argument("--max-generations", type=int, default=15,
                        help="CMA-ES max generations")
    parser.add_argument("--output-dir", default=None,
                        help="Root dir for outputs (default: actuator_gain_results/<tag>)")
    parser.add_argument("--init-kp", type=float, default=-1.05,
                        help="Initial raw_ctrl_stiffness center")
    parser.add_argument("--init-kd", type=float, default=-1.54,
                        help="Initial raw_ctrl_damping center")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Initial CMA-ES step size in raw space")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.base_config)
    tag = base_cfg.get("tag", "cloth")
    target_scenes = list(base_cfg.get("target_scenes", []))
    gpus = str(base_cfg.get("train", {}).get("gpus", "0"))

    if args.output_dir is None:
        args.output_dir = f"actuator_gain_results/{tag}"
    os.makedirs(args.output_dir, exist_ok=True)

    history = []
    sample_counter = [0]

    def fitness(raw_pair):
        raw_ks, raw_kd = float(raw_pair[0]), float(raw_pair[1])
        idx = sample_counter[0]
        sample_counter[0] += 1
        loss = _train_all_cases(base_cfg, raw_ks, raw_kd,
                                args.iters, args.output_dir, idx)
        history.append({
            "idx": idx, "raw_kp": raw_ks, "raw_kd": raw_kd,
            "loss": loss if loss != float("inf") else None,
        })
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        return loss

    es = cma.CMAEvolutionStrategy(
        x0=[args.init_kp, args.init_kd],
        sigma0=args.sigma,
        inopts={
            "maxiter": args.max_generations,
            "popsize": args.popsize,
            "verbose": -3,
        },
    )

    print("=" * 60)
    print(f"CMA-ES tuning kp/kd")
    print(f"  base config:      {args.base_config}")
    print(f"  tag:              {tag}")
    print(f"  target scenes:    {len(target_scenes)} ({', '.join(target_scenes[:3])}{'...' if len(target_scenes) > 3 else ''})")
    print(f"  gpus:             {gpus}")
    print(f"  popsize:          {args.popsize}")
    print(f"  max generations:  {args.max_generations}")
    print(f"  iters per eval:   {args.iters}")
    print(f"  total evals:      {args.popsize * args.max_generations}")
    print(f"  init:             raw_kp={args.init_kp:+.2f}  raw_kd={args.init_kd:+.2f}")
    print(f"  output dir:       {args.output_dir}")
    print("=" * 60)

    t_total = time.time()
    es.optimize(fitness)
    total_min = (time.time() - t_total) / 60

    best_x = es.result.xbest
    best_f = es.result.fbest

    print()
    print("=" * 60)
    print(f"Best:")
    print(f"  raw_kp = {best_x[0]:+.4f}")
    print(f"  raw_kd = {best_x[1]:+.4f}")
    print(f"  loss   = {best_f:.6f}")
    print(f"  total time: {total_min:.1f} min over {sample_counter[0]} evals")
    print("=" * 60)

    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump({
            "best_raw_kp": float(best_x[0]),
            "best_raw_kd": float(best_x[1]),
            "best_loss": float(best_f),
            "n_evals": sample_counter[0],
            "total_min": total_min,
            "tag": tag,
            "target_scenes": target_scenes,
            "base_config": args.base_config,
        }, f, indent=2)

    print(f"\nResult saved to {args.output_dir}/result.json")
    print(f"History: {args.output_dir}/history.json")


if __name__ == "__main__":
    main()
