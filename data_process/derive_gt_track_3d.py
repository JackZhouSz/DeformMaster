"""Derive a sparse k-keypoint GT track from final_data.pkl for custom cases.

For each case in --config (CSV) or a single --case_name, this picks ``k``
well-spaced object query points whose tracks are visible **across all
frames** (default 100% coverage), via farthest-point sampling on
``object_points`` at frame 0. The output (T, k, 3) tensor is saved to
``<case>/gt_track_3d.pkl`` matching the published-data convention so
``evaluate_track.py`` works without modification.

Also renders a verification video that overlays the k points (each with a
distinct color + id) onto the case's cam-0 RGB sequence:
``<case>/gt_track_3d.mp4``.

The script ONLY pulls from ``object_points`` (already separated from
``controller_points`` upstream in data_process_track.py), so the derived
GT is guaranteed to be on-object, never on the hand.

Usage::

    # single case
    python data_process/derive_gt_track_3d.py --case_name my_double_compress_rope

    # all cases in a CSV (skip if gt_track_3d.pkl already exists)
    python data_process/derive_gt_track_3d.py --config configs/data_process/data_config_custom.csv

    # force regen
    python data_process/derive_gt_track_3d.py --config configs/data_process/data_config_custom.csv \
                                             --overwrite
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

DEFAULT_BASE = "./data/different_types"


def fps(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Farthest-point sampling. Returns indices into ``points``."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    sel = [int(rng.integers(n))]
    dists = np.full(n, np.inf)
    for _ in range(k - 1):
        new_d = cdist(points, points[sel[-1:]]).squeeze(axis=1)
        dists = np.minimum(dists, new_d)
        sel.append(int(np.argmax(dists)))
    return np.array(sel, dtype=np.int64)


def render_viz(case_dir: Path, gt: np.ndarray) -> str:
    """Project ``gt`` (T, k, 3) world coords through cam 0 onto the cam-0
    video and write ``<case>/gt_track_3d.mp4``. Returns a status string."""
    import cv2
    import matplotlib.cm as cm

    meta_path = case_dir / "metadata.json"
    calib_path = case_dir / "calibrate.pkl"
    mp4_path = case_dir / "color" / "0.mp4"
    out_path = case_dir / "gt_track_3d.mp4"

    if not (meta_path.exists() and calib_path.exists() and mp4_path.exists()):
        return "skipped (missing meta / calib / cam-0 mp4)"

    K = np.array(json.load(open(meta_path))["intrinsics"][0], dtype=np.float64)
    c2w = np.array(pickle.load(open(calib_path, "rb"))[0], dtype=np.float64)
    w2c = np.linalg.inv(c2w)

    cap = cv2.VideoCapture(str(mp4_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Match data_process_sample.py / final_data.mp4: H.264 in MP4 container.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps_v, (W, H))

    T, k, _ = gt.shape
    cmap = cm.get_cmap("hsv", k)
    # BGR for cv2
    colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
              for c in cmap(np.linspace(0, 1, k))]

    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or f_idx >= T:
            break
        for i in range(k):
            pt = gt[f_idx, i]
            if np.isnan(pt).any():
                continue
            p_cam = w2c @ np.array([pt[0], pt[1], pt[2], 1.0])
            x, y, z = p_cam[:3]
            if z <= 1e-3:
                continue
            u = int(round(K[0, 0] * x / z + K[0, 2]))
            v = int(round(K[1, 1] * y / z + K[1, 2]))
            if not (0 <= u < W and 0 <= v < H):
                continue
            cv2.circle(frame, (u, v), 2, colors[i], -1)
            cv2.circle(frame, (u, v), 3, (255, 255, 255), 1)
        cv2.putText(frame, f"f={f_idx}/{T}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(frame)
        f_idx += 1

    vw.release()
    cap.release()
    return f"{f_idx} frames -> gt_track_3d.mp4"


def derive_one_case(base_path, case_name, n_kp, coverage_thresh, seed,
                    overwrite, viz):
    case_dir = Path(base_path) / case_name
    fd_path = case_dir / "final_data.pkl"
    gt_path = case_dir / "gt_track_3d.pkl"

    if not fd_path.is_file():
        return ("MISS", case_name, "final_data.pkl not found")

    if gt_path.is_file() and not overwrite:
        return ("SKIP", case_name, "gt_track_3d.pkl exists (use --overwrite)")

    with open(fd_path, "rb") as f:
        d = pickle.load(f)
    obj = np.asarray(d["object_points"])           # (T, N, 3) - object only
    vis = np.asarray(d["object_visibilities"], dtype=bool)
    T, N, _ = obj.shape

    # Candidates: visible at frame 0 AND coverage >= coverage_thresh.
    # Default coverage_thresh = 1.0 -> tracks must be visible at every frame.
    coverage = vis.sum(axis=0) / T
    candidates = np.where(vis[0] & (coverage >= coverage_thresh))[0]
    if len(candidates) < n_kp:
        return ("FAIL", case_name,
                f"only {len(candidates)}/{N} tracks meet coverage>={coverage_thresh:.2f}, "
                f"need {n_kp}")

    # FPS in 3D on frame-0 positions to get a spatially well-spread set.
    sub = fps(obj[0, candidates], n_kp, seed)
    chosen = candidates[sub]

    gt = obj[:, chosen, :].astype(np.float64).copy()
    # If user passed coverage_thresh < 1.0, occluded frames -> NaN.
    gt[~vis[:, chosen]] = np.nan
    nan_per = np.isnan(gt).any(axis=2).sum(axis=0).tolist()

    with open(gt_path, "wb") as f:
        pickle.dump(gt, f)

    detail = (f"T={T}, N={N}, candidates={len(candidates)} "
              f"(cov>={coverage_thresh:.2f}), NaN/pt={nan_per}")
    if viz:
        try:
            detail += " | viz: " + render_viz(case_dir, gt)
        except Exception as e:
            detail += f" | viz failed: {e!r}"
    return ("OK", case_name, detail)


def parse_csv(path):
    """Return list of (case_name, n_keypoints_or_None).

    Schema: case_name,category,shape_prior,regularize,n_keypoints. Only
    columns 0 and 4 are read here; the per-case n_keypoints is None when
    the column is absent or unparseable, in which case the caller falls
    back to the CLI ``--n-keypoints`` default.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            n = None
            if len(row) > 4 and row[4].strip():
                try:
                    n = int(row[4].strip())
                except ValueError:
                    n = None
            rows.append((row[0], n))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case_name", default=None,
                        help="Single-case mode (mutually exclusive with --config).")
    parser.add_argument("--config", action="append", default=[],
                        help="CSV with case_name in column 0; repeatable for "
                             "multiple CSVs.")
    parser.add_argument("--base-path", default=DEFAULT_BASE)
    parser.add_argument("--n-keypoints", type=int, default=16,
                        help="Number of GT keypoints per case (default 16).")
    parser.add_argument("--coverage-thresh", type=float, default=1.0,
                        help="Minimum fraction of frames a track must be "
                             "visible to be a candidate. Default 1.0 (every "
                             "frame); lower for heavily-occluded cases.")
    parser.add_argument("--seed", type=int, default=42,
                        help="FPS seed for reproducible keypoint selection.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate even if gt_track_3d.pkl already "
                             "exists. Default off (skip) to protect "
                             "human-annotated GT.")
    parser.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True,
                        help="Render gt_track_3d.mp4 overlay on cam-0 RGB.")
    args = parser.parse_args()

    if args.case_name and args.config:
        parser.error("specify either --case_name or --config, not both")

    if args.case_name:
        cases = [(args.case_name, None)]
    else:
        cases = []
        for c in args.config:
            cases.extend(parse_csv(c))

    if not cases:
        parser.error("no cases to process (pass --case_name or --config)")

    print(f"[derive] {len(cases)} case(s), n_keypoints default={args.n_keypoints} "
          f"(per-case from CSV col 5 if set), "
          f"coverage>={args.coverage_thresh:.2f}, viz={args.viz}, "
          f"overwrite={args.overwrite}", flush=True)

    counts = {"OK": 0, "SKIP": 0, "MISS": 0, "FAIL": 0}
    for case, n_kp_csv in cases:
        n_kp = n_kp_csv if n_kp_csv is not None else args.n_keypoints
        status, name, detail = derive_one_case(
            args.base_path, case, n_kp, args.coverage_thresh,
            args.seed, args.overwrite, args.viz,
        )
        counts[status] = counts.get(status, 0) + 1
        tag = f" (n_kp={n_kp})" if n_kp_csv is not None else ""
        print(f"[{status:4s}] {name}{tag}: {detail}", flush=True)

    print()
    print(f"[derive] summary: OK={counts['OK']}  SKIP={counts['SKIP']}  "
          f"MISS={counts['MISS']}  FAIL={counts['FAIL']}  total={len(cases)}",
          flush=True)
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
