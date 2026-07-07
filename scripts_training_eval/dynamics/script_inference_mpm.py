"""
Batch inference script for the Hybrid Physics-Neural MPM Simulator.

Auto-discovers trained MPM checkpoints from config-defined output directories,
runs inference using best_checkpoint.pt, and exports inference.pkl to a flat
directory structure compatible with gs_render_dynamics.py:

    {output_dir}/{case_name}/inference.pkl

Usage:
    python scripts_training_eval/dynamics/script_inference_mpm.py --gpu 0 --input output_2
    python scripts_training_eval/dynamics/script_inference_mpm.py --gpu 0 --input output_2 --scenes single_push_rope,single_lift_rope
"""

import os
import sys
import argparse
import subprocess

CATEGORY_DIRS = ("softbody_warp", "cloth_warp", "rope_warp", "package_warp")

def resolve_input_root(input_arg):
    input_path = os.path.normpath(input_arg)
    base_name = os.path.basename(input_path)
    if base_name == "mpm_inference":
        return os.path.dirname(input_path)
    if base_name in CATEGORY_DIRS:
        return os.path.dirname(input_path)
    return input_path


def normalize_output_dir_from_input(input_arg):
    return os.path.join(resolve_input_root(input_arg), "mpm_inference")


def discover_scenes(input_arg):
    """Auto-discover all scenes from checkpoint dirs under the input root."""
    input_path = os.path.normpath(input_arg)
    input_root = resolve_input_root(input_arg)
    input_base = os.path.basename(input_path)
    scenes = []
    if input_base in CATEGORY_DIRS:
        category_entries = [input_path]
    else:
        category_entries = [os.path.join(input_root, category) for category in CATEGORY_DIRS]

    for cat_dir in category_entries:
        category = os.path.basename(cat_dir)
        if not os.path.isdir(cat_dir):
            print(f"  Skipping {category}: output dir not found ({cat_dir})")
            continue
        for scene_name in sorted(os.listdir(cat_dir)):
            scene_dir = os.path.join(cat_dir, scene_name)
            ckpt_path = os.path.join(scene_dir, 'best_checkpoint.pt')
            config_path = os.path.join(scene_dir, "config.yaml")
            if not os.path.isdir(scene_dir) or not os.path.exists(ckpt_path):
                print(f"  Skipping {scene_name} (no best_checkpoint.pt)")
                continue
            if not os.path.exists(config_path):
                print(f"  Skipping {scene_name} (missing config.yaml: {config_path})")
                continue
            scenes.append((scene_name, config_path, ckpt_path))
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene names to process (default: all)")
    parser.add_argument("--input", type=str, required=True,
                        help="Training output root or category dir. Examples: output_2 -> output_2/mpm_inference, output_2/rope_warp -> output_2/mpm_inference")
    args = parser.parse_args()

    args.output_dir = normalize_output_dir_from_input(args.input)
    print(f"Scanning input: {args.input}")
    print(f"Inference output dir: {args.output_dir}")
    all_scenes = discover_scenes(args.input)
    print(f"Discovered {len(all_scenes)} scenes with best_checkpoint.pt")

    if args.scenes:
        filter_set = set(args.scenes.split(','))
        all_scenes = [(n, c, p) for n, c, p in all_scenes if n in filter_set]
        print(f"Filtered to {len(all_scenes)} scenes: {[s[0] for s in all_scenes]}")

    for i, (scene_name, config, ckpt_path) in enumerate(all_scenes):
        save_path = os.path.join(args.output_dir, scene_name, 'inference.pkl')
        if os.path.exists(save_path):
            print(f"[{i+1}/{len(all_scenes)}] {scene_name}: inference.pkl exists, skipping.")
            continue

        print(f"\n[{i+1}/{len(all_scenes)}] Processing {scene_name} ...")
        cmd = [
            sys.executable, "scripts_training_eval/dynamics/inference_mpm.py",
            "--case_name", scene_name,
            "--config", config,
            "--checkpoint", ckpt_path,
            "--output_dir", args.output_dir,
            "--gpu", args.gpu,
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"  -> Done: {scene_name}")
        except subprocess.CalledProcessError:
            print(f"  -> FAILED: {scene_name}")

    print(f"\nAll done. Inference results in: {args.output_dir}/")
    print("Next step: run GS dynamic reconstruction with:")
    print(f"  bash scripts_training_eval/appearance/gs_run_simulation.sh {args.output_dir} <scene_name1,scene_name2,...>")


if __name__ == "__main__":
    main()
