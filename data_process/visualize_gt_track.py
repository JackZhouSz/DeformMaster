"""Overlay gt_track_3d.pkl onto cam 0's video for visual inspection.

Writes to <case>/gt_track_3d.mp4 alongside the pkl (matches
data_process/derive_gt_track_3d.py's output path), using H.264 (avc1) encoding to
match final_data.mp4. Colors are auto-assigned via the HSV colormap so
any number of keypoints renders with distinct hues.

Per frame:
  - Project k world-coord GT points through cam 0's (K_0, c2w_0)
  - Draw each point as a small colored circle (no id label)
  - NaN points are skipped for that frame

Usage:
    python data_process/visualize_gt_track.py                      # all cases
    python data_process/visualize_gt_track.py --case double_lift_cloth_1
"""
import argparse
import json
import os
import pickle

import cv2
import matplotlib.cm as cm
import numpy as np

BASE = "data/different_types"


def world_to_pixel(pt_world, K, c2w):
    """World xyz -> cam0 pixel (u, v). Returns None if behind camera."""
    w2c = np.linalg.inv(c2w)
    p_cam = w2c @ np.array([pt_world[0], pt_world[1], pt_world[2], 1.0])
    x, y, z = p_cam[:3]
    if z <= 1e-3:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return int(round(u)), int(round(v))


def visualize_case(case):
    case_dir = os.path.join(BASE, case)
    gt_path = os.path.join(case_dir, "gt_track_3d.pkl")
    if not os.path.exists(gt_path):
        return f"SKIP {case}: no gt_track_3d.pkl"

    with open(gt_path, "rb") as f:
        gt = pickle.load(f)  # (T, N, 3), typically N=9
    if gt.ndim != 3 or gt.shape[2] != 3:
        return f"SKIP {case}: unexpected gt shape {gt.shape}"

    meta = json.load(open(os.path.join(case_dir, "metadata.json")))
    K0 = np.array(meta["intrinsics"][0], dtype=np.float64)
    with open(os.path.join(case_dir, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    c2w0 = np.array(c2ws[0], dtype=np.float64)

    # Prefer mp4 (fast read), fall back to PNG sequence
    mp4_path = os.path.join(case_dir, "color", "0.mp4")
    frames_iter = None
    W = H = None
    fps = 30
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        def frames_iter_fn():
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                yield fr
            cap.release()

        frames_iter = frames_iter_fn()
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # PNG fallback: frame_idx.png sorted numerically
        png_dir = os.path.join(case_dir, "color", "0")
        pngs = sorted([p for p in os.listdir(png_dir) if p.endswith(".png")],
                      key=lambda p: int(os.path.splitext(p)[0]))

        def frames_iter_fn():
            for p in pngs:
                yield cv2.imread(os.path.join(png_dir, p))

        frames_iter = frames_iter_fn()
        if pngs:
            first = cv2.imread(os.path.join(png_dir, pngs[0]))
            H, W = first.shape[:2]

    if W is None:
        return f"SKIP {case}: no video or PNG frames"

    T = gt.shape[0]
    N = gt.shape[1]
    out_path = os.path.join(case_dir, "gt_track_3d.mp4")
    # H.264 in MP4 container, matches final_data.mp4 / data_process/derive_gt_track_3d.py.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # HSV colormap: assigns N distinct hues automatically.
    cmap = cm.get_cmap("hsv", N)
    colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))   # BGR
              for c in cmap(np.linspace(0, 1, N))]

    drawn = 0
    for f_idx, frame in enumerate(frames_iter):
        if f_idx >= T:
            break
        if frame is None:
            continue
        overlay = frame.copy()
        for i in range(N):
            pt3d = gt[f_idx, i]
            if np.isnan(pt3d).any():
                continue
            uv = world_to_pixel(pt3d, K0, c2w0)
            if uv is None:
                continue
            u, v = uv
            if not (0 <= u < W and 0 <= v < H):
                continue
            cv2.circle(overlay, (u, v), 2, colors[i], -1)
            cv2.circle(overlay, (u, v), 3, (255, 255, 255), 1)
        # Header: frame counter + case name
        cv2.putText(overlay, f"{case}  f={f_idx}/{T}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        vw.write(overlay)
        drawn += 1
    vw.release()
    return f"OK   {case}: {drawn} frames -> {out_path}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=None,
                        help="Specific case to process (default: all with gt)")
    args = parser.parse_args()

    if args.case:
        cases = [args.case]
    else:
        cases = sorted(d for d in os.listdir(BASE)
                       if os.path.isdir(os.path.join(BASE, d)))

    for c in cases:
        print(visualize_case(c), flush=True)

    print(f"\nAll outputs written to <case>/gt_track_3d.mp4 alongside the pkl.")


if __name__ == "__main__":
    main()
