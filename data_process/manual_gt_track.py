"""Gradio web UI for manual GT-track annotation via CoTracker3.

Workflow
--------
1. Pick a case from the dropdown (auto-discovered under ``--base-path``).
   The cam-0 frame-0 RGB loads in the canvas.
2. Click on the rope/object where you want each GT keypoint to live.
   Each click drops a numbered marker at that pixel; the sparse list of
   query points is what CoTracker will be told to track.
3. Hit "Run CoTracker3 + Lift to 3D". The script:
     a. Loads cam-0 ``color/0.mp4`` as a torch tensor.
     b. Runs ``cotracker3_offline`` from your queries (frame 0, x, y) →
        per-frame 2D pixel positions for each query.
     c. Lifts each 2D track to 3D world coords using cam-0 ``depth/``,
        ``metadata.json`` intrinsic, and ``calibrate.pkl`` c2w —
        matching the existing ``data_process_pcd.py`` convention.
     d. Renders a viz mp4 by reprojecting the 3D anchors back through
        cam 0 onto the RGB video (lets you spot lift failures).
4. Inspect the viz; if it looks good, hit "Save" — writes
   ``<case>/gt_track_3d.pkl`` (T, K, 3) and ``<case>/gt_track_3d.mp4``.

Usage
-----
    pip install gradio                          # if missing
    python data_process/manual_gt_track.py \\
        --base-path data/different_types \\
        --host 0.0.0.0 --port 7860

For remote servers, forward the port over SSH::

    ssh -L 7860:localhost:7860 <user>@<server>
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from threading import Lock

import numpy as np


# ---------------------------------------------------------------------------
# Cached CoTracker3 model — loaded on first run, kept on GPU.
# ---------------------------------------------------------------------------
_MODEL_LOCK = Lock()
_MODEL = None


def get_cotracker():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[manual_gt_track] loading cotracker3_offline on {device} ...",
                  flush=True)
            model = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_offline",
            ).to(device)
            model.eval()
            _MODEL = (model, device)
    return _MODEL


# ---------------------------------------------------------------------------
# Case discovery + per-case data loading.
# ---------------------------------------------------------------------------
def discover_cases(base_path):
    """Return sorted list of case names under base_path that have the
    minimum files needed for manual annotation (cam-0 mp4, depth, calib,
    intrinsic)."""
    base = Path(base_path)
    cases = []
    if not base.is_dir():
        return cases
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "color" / "0.mp4").is_file():
            continue
        if not (d / "depth" / "0").is_dir():
            continue
        if not (d / "metadata.json").is_file():
            continue
        if not (d / "calibrate.pkl").is_file():
            continue
        cases.append(d.name)
    return cases


def load_first_frame(case_dir):
    """Read frame 0 from cam-0 mp4 and return as RGB uint8 np array."""
    import cv2
    mp4 = case_dir / "color" / "0.mp4"
    cap = cv2.VideoCapture(str(mp4))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_video_tensor(case_dir):
    """Load cam-0 mp4 as a (1, T, 3, H, W) float tensor on GPU."""
    import imageio.v3 as iio
    import torch
    mp4 = case_dir / "color" / "0.mp4"
    frames = iio.imread(str(mp4), plugin="FFMPEG")          # (T, H, W, 3) uint8
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
    _, device = get_cotracker()
    return video.to(device)


def load_camera(case_dir, cam_id=0):
    """Return (intrinsic 3x3, c2w 4x4) for the chosen cam."""
    with open(case_dir / "metadata.json") as f:
        meta = json.load(f)
    intrinsic = np.array(meta["intrinsics"][cam_id], dtype=np.float64)
    with open(case_dir / "calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    c2w = np.array(c2ws[cam_id], dtype=np.float64)
    return intrinsic, c2w


# ---------------------------------------------------------------------------
# 2D → 3D lift (matches data_process_pcd.py convention).
# ---------------------------------------------------------------------------
def lift_pixel(x, y, depth, intrinsic_inv, c2w):
    """Lift a single (sub-pixel) point with depth lookup at the nearest
    integer cell. Returns (X, Y, Z) world coords or None if depth invalid."""
    H, W = depth.shape
    xi = int(round(np.clip(x, 0, W - 1)))
    yi = int(round(np.clip(y, 0, H - 1)))
    d = float(depth[yi, xi])
    if d <= 0 or not np.isfinite(d):
        return None
    cam = intrinsic_inv @ np.array([x * d, y * d, d], dtype=np.float64)
    world = c2w @ np.array([cam[0], cam[1], cam[2], 1.0], dtype=np.float64)
    return world[:3]


def lift_tracks_2d_to_3d(tracks_xy, case_dir, cam_id=0):
    """tracks_xy: (T, K, 2) numpy. Returns (T, K, 3); invalid frames/points
    forward-filled from the last valid 3D position per-track, NaN if no
    valid lift ever."""
    intrinsic, c2w = load_camera(case_dir, cam_id)
    intrinsic_inv = np.linalg.inv(intrinsic)
    T, K, _ = tracks_xy.shape
    out = np.full((T, K, 3), np.nan, dtype=np.float64)
    for t in range(T):
        depth_path = case_dir / "depth" / str(cam_id) / f"{t}.npy"
        if not depth_path.is_file():
            continue
        depth = np.load(depth_path) / 1000.0                # mm → m
        for k in range(K):
            x, y = float(tracks_xy[t, k, 0]), float(tracks_xy[t, k, 1])
            p = lift_pixel(x, y, depth, intrinsic_inv, c2w)
            if p is not None:
                out[t, k] = p
    # Forward-fill NaNs so downstream eval doesn't break on holes.
    last = np.full((K, 3), np.nan, dtype=np.float64)
    for t in range(T):
        for k in range(K):
            if np.isnan(out[t, k]).any():
                if not np.isnan(last[k]).any():
                    out[t, k] = last[k]
            else:
                last[k] = out[t, k]
    return out


# ---------------------------------------------------------------------------
# Render the GT viz overlay on cam-0 RGB (re-uses derive_gt_track_3d's idea).
# ---------------------------------------------------------------------------
def render_viz(case_dir, gt_3d, out_path):
    """Project (T, K, 3) world coords through cam 0 and write an mp4
    overlay onto the cam-0 RGB video. Returns a status string."""
    import cv2
    import matplotlib.cm as cm

    intrinsic, c2w = load_camera(case_dir, 0)
    w2c = np.linalg.inv(c2w)
    mp4_path = case_dir / "color" / "0.mp4"
    if not mp4_path.is_file():
        return "missing cam-0 mp4"
    cap = cv2.VideoCapture(str(mp4_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps_v, (W, H))

    T, K, _ = gt_3d.shape
    cmap = cm.get_cmap("hsv", K)
    colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
              for c in cmap(np.linspace(0, 1, K))]

    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or f_idx >= T:
            break
        for k in range(K):
            pt = gt_3d[f_idx, k]
            if np.isnan(pt).any():
                continue
            p_cam = w2c @ np.array([pt[0], pt[1], pt[2], 1.0])
            x, y, z = p_cam[:3]
            if z <= 1e-3:
                continue
            u = int(round(intrinsic[0, 0] * x / z + intrinsic[0, 2]))
            v = int(round(intrinsic[1, 1] * y / z + intrinsic[1, 2]))
            if not (0 <= u < W and 0 <= v < H):
                continue
            cv2.circle(frame, (u, v), 4, colors[k], -1)
            cv2.circle(frame, (u, v), 5, (255, 255, 255), 1)
            cv2.putText(frame, str(k), (u + 6, v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"f={f_idx}/{T}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(frame)
        f_idx += 1
    vw.release()
    cap.release()
    return f"{f_idx} frames"


# ---------------------------------------------------------------------------
# CoTracker invocation.
# ---------------------------------------------------------------------------
def run_cotracker_offline(case_dir, queries_xy):
    """queries_xy: (K, 2) np float, frame-0 click coords.
    Returns 2D tracks (T, K, 2) and visibilities (T, K) bool."""
    import torch
    model, device = get_cotracker()
    video = load_video_tensor(case_dir)                       # (1, T, 3, H, W)
    K = queries_xy.shape[0]
    queries = np.zeros((K, 3), dtype=np.float32)
    queries[:, 0] = 0                                         # frame 0
    queries[:, 1] = queries_xy[:, 0]                          # x
    queries[:, 2] = queries_xy[:, 1]                          # y
    queries_t = torch.from_numpy(queries).to(device)[None]    # (1, K, 3)
    with torch.no_grad():
        tracks, vis = model(video, queries=queries_t,
                            backward_tracking=True)
    tracks = tracks[0].cpu().numpy()                          # (T, K, 2)
    vis = vis[0].cpu().numpy().astype(bool)                   # (T, K)
    return tracks, vis


# ---------------------------------------------------------------------------
# Image annotation overlay (frame-0 click markers).
# ---------------------------------------------------------------------------
def annotate_frame(frame_rgb, points):
    """Draw numbered colored dots on a copy of frame_rgb."""
    import cv2
    import matplotlib.cm as cm
    out = frame_rgb.copy()
    if not points:
        return out
    K = len(points)
    cmap = cm.get_cmap("hsv", max(K, 1))
    for i, (x, y) in enumerate(points):
        c = cmap(i / max(K - 1, 1))
        bgr = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        rgb = (bgr[2], bgr[1], bgr[0])
        cv2.circle(out, (int(x), int(y)), 6, rgb, -1)
        cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 1)
        cv2.putText(out, str(i), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ---------------------------------------------------------------------------
# Gradio app.
# ---------------------------------------------------------------------------
def build_app(base_path):
    import gradio as gr

    cases = discover_cases(base_path)
    if not cases:
        print(f"[manual_gt_track] WARN: no cases discovered under {base_path}",
              file=sys.stderr)

    with gr.Blocks(title="Manual GT Track Annotation") as demo:
        gr.Markdown(
            "## Manual GT Track Annotation (CoTracker3 + 3D lift)\n\n"
            "1. Pick a case → frame 0 of cam 0 loads.  \n"
            "2. Click on the rope/object where you want each GT keypoint.  \n"
            "3. Run CoTracker3 → inspect the reprojection viz.  \n"
            "4. Save → writes `<case>/gt_track_3d.{pkl,mp4}`."
        )

        case_state = gr.State(value=None)             # str
        original_frame = gr.State(value=None)         # np.ndarray
        points_state = gr.State(value=[])             # list[(x, y)]
        gt3d_state = gr.State(value=None)             # np.ndarray (T, K, 3)

        with gr.Row():
            case_dd = gr.Dropdown(
                choices=cases, label="Case", value=cases[0] if cases else None,
                scale=4,
            )
            reload_btn = gr.Button("⟳ Refresh case list", scale=1, min_width=40)

        with gr.Row():
            with gr.Column(scale=3):
                frame_img = gr.Image(
                    label="Frame 0 — click to add query points",
                    interactive=True, show_label=True,
                )
                with gr.Row():
                    undo_btn = gr.Button("Undo last", min_width=40)
                    clear_btn = gr.Button("Clear all", min_width=40)
                run_btn = gr.Button("Run CoTracker3 + Lift to 3D",
                                    variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=2)
            with gr.Column(scale=3):
                viz_video = gr.Video(label="GT Track viz (3D → cam-0 reproject)",
                                     interactive=False)
                save_btn = gr.Button("Save gt_track_3d.{pkl,mp4} to case dir",
                                     variant="primary")
                save_status = gr.Textbox(label="Save status",
                                         interactive=False, lines=2)

        # ----- handlers -----
        def on_load_case(case):
            if not case:
                return None, None, [], "No case selected."
            case_dir = Path(base_path) / case
            frame = load_first_frame(case_dir)
            if frame is None:
                return None, None, [], f"Failed to read cam-0 mp4 in {case}"
            return frame, frame, [], (
                f"Loaded {case}; click points on the rope.")

        def on_refresh():
            new_cases = discover_cases(base_path)
            return gr.update(choices=new_cases,
                             value=new_cases[0] if new_cases else None)

        def on_click(evt: gr.SelectData, original, points):
            if original is None:
                return None, points, "Load a case first."
            x, y = float(evt.index[0]), float(evt.index[1])
            new_points = list(points) + [(x, y)]
            annotated = annotate_frame(original, new_points)
            return annotated, new_points, (
                f"{len(new_points)} query point(s).")

        def on_undo(original, points):
            if not points:
                return original, [], "Nothing to undo."
            new_points = list(points)[:-1]
            annotated = annotate_frame(original, new_points)
            return annotated, new_points, (
                f"{len(new_points)} query point(s).")

        def on_clear(original):
            if original is None:
                return None, [], "No case loaded."
            return original, [], "Cleared."

        def on_run(case, points):
            if not case:
                return None, None, "No case selected."
            if not points:
                return None, None, "No query points clicked."
            case_dir = Path(base_path) / case
            queries = np.array(points, dtype=np.float32)      # (K, 2) — (x, y)
            try:
                tracks_xy, vis = run_cotracker_offline(case_dir, queries)
            except Exception as e:
                return None, None, f"CoTracker failed: {e!r}"
            try:
                gt3d = lift_tracks_2d_to_3d(tracks_xy, case_dir, cam_id=0)
            except Exception as e:
                return None, None, f"3D lift failed: {e!r}"
            tmp_mp4 = case_dir / "gt_track_3d.preview.mp4"
            try:
                msg = render_viz(case_dir, gt3d, tmp_mp4)
            except Exception as e:
                return None, gt3d, f"Tracked but viz failed: {e!r}"
            n_nan = int(np.isnan(gt3d).any(axis=2).sum())
            T, K, _ = gt3d.shape
            return str(tmp_mp4), gt3d, (
                f"Tracked T={T} frames × K={K} pts; "
                f"viz: {msg}; lift NaNs: {n_nan}/{T * K} (forward-filled).")

        def on_save(case, gt3d):
            if not case:
                return "No case selected."
            if gt3d is None:
                return "Run CoTracker first."
            case_dir = Path(base_path) / case
            pkl = case_dir / "gt_track_3d.pkl"
            mp4 = case_dir / "gt_track_3d.mp4"
            with open(pkl, "wb") as f:
                pickle.dump(gt3d.astype(np.float64), f)
            preview = case_dir / "gt_track_3d.preview.mp4"
            if preview.is_file():
                if mp4.is_file():
                    mp4.unlink()
                preview.rename(mp4)
            return f"Saved {pkl} and {mp4}."

        # ----- wiring -----
        case_dd.change(on_load_case, [case_dd],
                       [frame_img, original_frame, points_state, status])
        reload_btn.click(on_refresh, [], [case_dd])
        frame_img.select(on_click, [original_frame, points_state],
                         [frame_img, points_state, status])
        undo_btn.click(on_undo, [original_frame, points_state],
                       [frame_img, points_state, status])
        clear_btn.click(on_clear, [original_frame],
                        [frame_img, points_state, status])
        run_btn.click(on_run, [case_dd, points_state],
                      [viz_video, gt3d_state, status])
        save_btn.click(on_save, [case_dd, gt3d_state], [save_status])

        # Trigger initial load if a default case exists.
        if cases:
            demo.load(on_load_case, inputs=[case_dd],
                      outputs=[frame_img, original_frame, points_state, status])

    return demo


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base-path",
                    default="data/different_types",
                    help="Root dir containing per-case folders.")
    ap.add_argument("--host", default="0.0.0.0",
                    help="Bind host (use 0.0.0.0 with SSH port-forward).")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Expose via gradio share link (don't use on prod).")
    args = ap.parse_args()

    demo = build_app(args.base_path)
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
