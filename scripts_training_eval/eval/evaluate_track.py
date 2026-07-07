import pickle
import glob
import csv
import json
import numpy as np
import argparse
import os
from scipy.spatial import KDTree

base_path = "./data/different_types"
output_file = "results/final_track.csv"


def resolve_prediction_path(prediction_path):
    if os.path.isfile(os.path.join(prediction_path, "inference.pkl")):
        return prediction_path

    mpm_inference_dir = os.path.join(prediction_path, "mpm_inference")
    if os.path.isdir(mpm_inference_dir):
        return mpm_inference_dir

    return prediction_path


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return np.mean(track_errors)


def _scenes_from_configs(config_paths):
    """Return the union of cfg.target_scenes across the given yaml configs.
    None signals 'no whitelist; iterate all dirs under base_path'."""
    if not config_paths:
        return None
    from omegaconf import OmegaConf
    out = set()
    for p in config_paths:
        if not os.path.isfile(p):
            print(f"  [warn] --config {p} not found, ignoring")
            continue
        cfg = OmegaConf.load(p)
        scenes = list(getattr(cfg, "target_scenes", []) or [])
        out.update(scenes)
        print(f"  [config] {p}: +{len(scenes)} scene(s)")
    return out if out else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tracking error for predictions")
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="experiments",
        help="Directory containing prediction results (default: experiments)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file path (default: results/final_track.csv)"
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to a yaml config; only cases in any cfg.target_scenes "
             "are evaluated. Repeat to merge multiple configs (e.g. one "
             "per category). When omitted, falls back to globbing every "
             "subdir under base_path (legacy behaviour).",
    )
    args = parser.parse_args()
    scene_whitelist = _scenes_from_configs(args.config)
    if scene_whitelist is not None:
        print(f"[whitelist] {len(scene_whitelist)} scene(s) from configs")
    
    prediction_path = resolve_prediction_path(args.prediction_path)
    if args.output_file is None:
        output_file = "results/final_track.csv"
    else:
        output_file = args.output_file
    
    print(f"Using prediction path: {prediction_path}")
    print(f"Output file: {output_file}")

    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(
        [
            "Case Name",
            "Train Track Error",
            "Test Track Error",
        ]
    )

    dir_names = [path for path in glob.glob(f"{base_path}/*") if os.path.isdir(path)]
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        if scene_whitelist is not None and case_name not in scene_whitelist:
            continue
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        # Cheap existence checks FIRST — skip cases that aren't ours to
        # evaluate (no inference) or that lack the required GT files.
        # Doing this before any open() means a partial dataset (e.g.
        # custom my_* dirs without split.json) doesn't crash the whole
        # eval loop and silently drop every case after the bad one.
        inference_path = f"{prediction_path}/{case_name}/inference.pkl"
        if not os.path.exists(inference_path):
            print(f"  Skipping {case_name} (inference.pkl not found in {prediction_path})")
            continue
        split_path = f"{base_path}/{case_name}/split.json"
        gt_path = f"{base_path}/{case_name}/gt_track_3d.pkl"
        if not os.path.exists(split_path):
            print(f"  Skipping {case_name} (split.json not found)")
            continue
        if not os.path.exists(gt_path):
            print(f"  Skipping {case_name} (gt_track_3d.pkl not found)")
            continue

        # Wrap the per-case work so any unexpected exception (shape
        # mismatch, KDTree failure, NaN-only GT, ...) only loses that
        # one case instead of every case after it.
        try:
            with open(split_path, "r") as f:
                split = json.load(f)
            train_frame = split["train"][1]
            test_frame = split["test"][1]

            with open(inference_path, "rb") as f:
                vertices = pickle.load(f)
            with open(gt_path, "rb") as f:
                gt_track_3d = pickle.load(f)

            # Locate the index of corresponding point index in the vertices, if nan, then ignore the points
            mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
            kdtree = KDTree(vertices[0])
            dis, idx = kdtree.query(gt_track_3d[0][mask])

            train_track_error = evaluate_prediction(
                1, train_frame, vertices, gt_track_3d, idx, mask
            )
            test_track_error = evaluate_prediction(
                train_frame, test_frame, vertices, gt_track_3d, idx, mask
            )
            writer.writerow([case_name, train_track_error, test_track_error])
        except Exception as e:
            print(f"  [error] {case_name}: {type(e).__name__}: {e}")
            continue
    file.close()
