"""Skeleton-based rope regularization + GT keypoint extraction.

Per frame, extract the rope's 3D centerline from the **raw** object
point cloud (depth-lifted, mask-segmented, *not* the drift-prone
CoTracker tracks):

  1. Load <case>/pcd/<t>.npz (per-cam world-coords PCD) ∧
     <case>/mask/processed_masks.pkl[t][pos]["object"] → fused object
     PCD at frame t. This is recomputed each frame from depth+seg, so
     the skeleton has zero dependence on cross-frame point IDs.
  2. Voxel-grid downsample to a manageable point count.
  3. kNN graph (k=4) with edge-length filter (drops cross-fold spurious
     shortcuts).
  4. Two-stage Dijkstra → graph-diameter endpoints (the rope's two ends).
  5. Shortest path between them = ordered skeleton polyline.

Sample K anchor points at evenly-spaced normalized arc-lengths along
the polyline. Anchor identity across frames is preserved by matching
anchor 0 to the previous frame's anchor 0; frame 0 picks orientation
from the PCA first principal axis (deterministic).

Drift-free by construction: the skeleton comes from the raw per-frame
PCD, not from the upstream tracker's drift-prone outputs.

Outputs:
  - <case>/gt_track_3d.pkl  : (T, K, 3) anchor positions (drift-free).
  - <case>/gt_track_3d.mp4  : cam-0 RGB overlay viz.
  - <case>/track_process_data.pkl : object_points re-projected onto
    each frame's skeleton at the frame-0 arc-length parameter.
  - <case>/track_process_data_raw.pkl : backup of raw tracks (first run).

Rope-only — assumes a 1D-curve-like point cloud. For cloth/toy use
data_process/regularize_tracks.py instead.

Usage::
    python data_process/skeleton_track_rope.py \\
        --base_path ./data/different_types --case_name my_double_lift_rope \\
        --n_keypoints 9
"""
import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def load_raw_object_pcd(case_dir, frame_idx, processed_masks):
    """Load drift-free object PCD at frame t from <case>/pcd/<t>.npz +
    processed_masks. Returns (M, 3) world-frame points or None if the
    pcd file doesn't exist (e.g. cleaned up after pipeline finished)."""
    pcd_npz = case_dir / "pcd" / f"{frame_idx}.npz"
    if not pcd_npz.exists():
        return None
    data = np.load(pcd_npz)
    points = data["points"]              # (n_cams, H, W, 3)
    depth_masks = data["masks"]          # (n_cams, H, W) bool

    if frame_idx not in processed_masks:
        return None
    pm_t = processed_masks[frame_idx]

    n_cams = points.shape[0]
    pts_list = []
    for pos in range(n_cams):
        if pos not in pm_t:
            continue
        obj_mask = pm_t[pos].get("object")
        if obj_mask is None:
            continue
        m = depth_masks[pos] & obj_mask
        pts_list.append(points[pos][m].reshape(-1, 3))
    if not pts_list:
        return None
    out = np.concatenate(pts_list, axis=0)
    return out if len(out) else None


def voxel_downsample(points, voxel_size):
    """Voxel-grid downsample: one representative point per voxel cell
    (the first point that lands in the cell). Sufficient for kNN-graph
    input since exact centroid placement doesn't matter for graph
    diameter."""
    if voxel_size <= 0 or len(points) == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(idx)]


def adapt_voxel_size(points, target_count):
    """Pick a voxel size that yields roughly target_count after downsampling."""
    if len(points) <= target_count or target_count <= 0:
        return 0.0
    bbox = points.max(axis=0) - points.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    # Heuristic: cube-root scaling on bounding-box diagonal. For a 1D
    # rope this overshoots target slightly (most of the bbox is empty),
    # which is fine — extra resolution along the rope helps the skeleton.
    return diag * (target_count ** (-1.0 / 3.0)) * 0.5


def extract_skeleton(p, k=4, edge_factor=3.0):
    """Return ordered indices into p forming the rope skeleton polyline.

    Steps: kNN graph (with edge-length filter) → graph-diameter
    endpoints via two-stage Dijkstra → shortest path between them.
    """
    n = p.shape[0]
    if n < 2:
        return list(range(n))
    tree = cKDTree(p)
    dists, idxs = tree.query(p, k=min(k + 1, n))
    # drop self-edge column 0
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]
    src = np.repeat(np.arange(n), dists.shape[1])
    dst = idxs.reshape(-1)
    w = dists.reshape(-1)
    if edge_factor > 0 and len(w):
        med = float(np.median(w))
        keep = w <= edge_factor * med
        src, dst, w = src[keep], dst[keep], w[keep]
    G = csr_matrix((w, (src, dst)), shape=(n, n))
    G = G.maximum(G.T)            # symmetrize

    # Two-stage Dijkstra for graph diameter.
    d0 = dijkstra(G, indices=0, directed=False)
    finite = np.isfinite(d0)
    if not finite.any():
        return [0]
    A = int(np.argmax(np.where(finite, d0, -1.0)))
    distA, predA = dijkstra(G, indices=A, directed=False,
                            return_predecessors=True)
    finiteA = np.isfinite(distA)
    if not finiteA.any():
        return [A]
    B = int(np.argmax(np.where(finiteA, distA, -1.0)))

    # Trace path B -> A through predecessors.
    path = [B]
    while path[-1] != A:
        nxt = int(predA[path[-1]])
        if nxt < 0:               # disconnected
            break
        path.append(nxt)
    return path[::-1]             # A -> B


def polyline_arc_lengths(pts):
    """pts: (M, 3) ordered points → (arc (M,), total_length)."""
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0]), 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    return arc, float(arc[-1])


def resample_polyline(pts, arc, query_arc):
    """Linear interpolation along the polyline at given arc-lengths.

    pts: (M, 3) ordered, arc: (M,) cumulative arc-length, query_arc: (Q,)."""
    if len(arc) < 2:
        return np.tile(pts[0:1], (len(query_arc), 1))
    qa = np.clip(query_arc, arc[0], arc[-1])
    idx = np.searchsorted(arc, qa) - 1
    idx = np.clip(idx, 0, len(arc) - 2)
    seg_len = np.maximum(arc[idx + 1] - arc[idx], 1e-9)
    t = (qa - arc[idx]) / seg_len
    return pts[idx] + t[:, None] * (pts[idx + 1] - pts[idx])


def fit_smooth_spline(skel_pts, voxel_size, smoothing_factor, n_dense=1000):
    """Fit a cubic B-spline through ordered skeleton points, smoothing
    away the voxel-grid + graph-diameter discretization jitter that
    causes anchor positions to flicker frame-to-frame.

    Returns dict with keys ``tck`` (spline coefficients), ``u_dense``
    (dense parameter samples in [0, 1]), ``pts_dense`` (3D points at
    those samples), ``arc_dense`` (cumulative arc-length on the smooth
    curve), ``L`` (total length). None on failure.

    Smoothing strength scales with ``smoothing_factor * M * voxel_size**2``
    so it adapts to per-frame point density and downsample resolution.
    """
    M = len(skel_pts)
    if M < 4:                                # cubic needs >= 4 ctrl points
        return None
    # Dedup successive duplicates (splprep needs strictly increasing u).
    seg = np.linalg.norm(np.diff(skel_pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-7))
    pts = skel_pts[keep]
    if len(pts) < 4:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u_init = np.concatenate(([0.0], np.cumsum(seg)))
    L_chord = float(u_init[-1])
    if L_chord < 1e-6:
        return None
    u_init = u_init / L_chord                 # normalize to [0, 1]

    sigma = max(voxel_size, 1e-3) * 0.5       # noise estimate (half-voxel)
    s_smooth = max(smoothing_factor * len(pts) * 3.0 * sigma * sigma, 1e-9)

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]],
                         u=u_init, k=3, s=s_smooth)
    except Exception:
        return None

    u_dense = np.linspace(0.0, 1.0, n_dense)
    xyz = splev(u_dense, tck)
    pts_dense = np.stack(xyz, axis=1)
    seg_dense = np.linalg.norm(np.diff(pts_dense, axis=0), axis=1)
    arc_dense = np.concatenate(([0.0], np.cumsum(seg_dense)))
    L = float(arc_dense[-1])
    if L < 1e-6:
        return None
    return {
        "tck": tck,
        "u_dense": u_dense,
        "pts_dense": pts_dense,
        "arc_dense": arc_dense,
        "L": L,
    }


def sample_at_arc(spline, query_arc, flip=False):
    """Evaluate the smooth spline at given absolute arc-lengths.

    ``flip=True`` queries from the opposite end (used when frame-t
    skeleton's natural orientation is reversed relative to frame 0).
    Queries are clamped to [0, L]."""
    arc_dense = spline["arc_dense"]
    u_dense = spline["u_dense"]
    L = spline["L"]
    qa = np.asarray(query_arc, dtype=np.float64)
    qa = np.clip(qa if not flip else (L - qa), 0.0, L)
    u_q = np.interp(qa, arc_dense, u_dense)
    xyz = splev(u_q, spline["tck"])
    out = np.stack(xyz, axis=1) if np.ndim(xyz[0]) else np.array(xyz).reshape(1, 3)
    return out


def project_to_smooth_spline(spline, query_pts, chunk=2000):
    """For each query point, return its absolute arc-length projection
    onto the dense-sampled smooth spline (nearest dense point)."""
    pts_dense = spline["pts_dense"]
    arc_dense = spline["arc_dense"]
    Q = query_pts.shape[0]
    out = np.zeros(Q)
    for start in range(0, Q, chunk):
        end = min(start + chunk, Q)
        diff = query_pts[start:end, None, :] - pts_dense[None, :, :]
        dist2 = np.einsum("qmd,qmd->qm", diff, diff)
        best = np.argmin(dist2, axis=1)
        out[start:end] = arc_dense[best]
    return out


def project_to_polyline(pts, arc, query_pts):
    """For each query point, return its arc-length projection onto the
    polyline (closest segment, clamped). pts: (M, 3), arc: (M,),
    query_pts: (Q, 3) → (Q,) arc-length."""
    M = pts.shape[0]
    Q = query_pts.shape[0]
    if M < 2:
        return np.zeros(Q)
    best_s = np.zeros(Q)
    best_d = np.full(Q, np.inf)
    for j in range(M - 1):
        a = pts[j]
        b = pts[j + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = ((query_pts - a) @ ab) / L2
        t_clip = np.clip(t, 0.0, 1.0)
        proj = a + t_clip[:, None] * ab
        d = np.linalg.norm(query_pts - proj, axis=1)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        s_here = arc[j] + t_clip * (arc[j + 1] - arc[j])
        best_s = np.where(better, s_here, best_s)
    return best_s


def render_viz(case_dir, gt):
    """Project gt (T, K, 3) world coords through cam 0 and write
    <case>/gt_track_3d.mp4. Returns a status string."""
    import cv2
    import matplotlib.cm as cm

    meta_path = case_dir / "metadata.json"
    calib_path = case_dir / "calibrate.pkl"
    mp4_path = case_dir / "color" / "0.mp4"
    out_path = case_dir / "gt_track_3d.mp4"

    if not (meta_path.exists() and calib_path.exists() and mp4_path.exists()):
        return "skipped (missing meta / calib / cam-0 mp4)"

    K_intr = np.array(json.load(open(meta_path))["intrinsics"][0],
                      dtype=np.float64)
    c2w = np.array(pickle.load(open(calib_path, "rb"))[0], dtype=np.float64)
    w2c = np.linalg.inv(c2w)

    cap = cv2.VideoCapture(str(mp4_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps_v, (W, H))

    T, k, _ = gt.shape
    cmap = cm.get_cmap("hsv", k)
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
            u = int(round(K_intr[0, 0] * x / z + K_intr[0, 2]))
            v = int(round(K_intr[1, 1] * y / z + K_intr[1, 2]))
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


def _build_frame_spline(case_dir, t, processed_masks, fallback_pts,
                        k_nn, edge_factor, target_pcd_count,
                        smoothing_factor):
    """Load raw PCD at frame t, downsample, extract skeleton polyline,
    fit smooth spline. Returns (spline_dict, source_tag) or (None, ...).

    fallback_pts: lifted CoTracker tracks, used only if raw PCD is
    missing (e.g. pre-cleaned re-run)."""
    raw = load_raw_object_pcd(case_dir, t, processed_masks)
    if raw is None or len(raw) < 10:
        if fallback_pts is None or len(fallback_pts) < 10:
            return None, "skip"
        raw = fallback_pts
        source = "tracks"
    else:
        source = "raw"
    vs = adapt_voxel_size(raw, target_pcd_count)
    ds = voxel_downsample(raw, vs) if vs > 0 else raw
    if len(ds) < 4:
        return None, "skip"
    path = extract_skeleton(ds, k=k_nn, edge_factor=edge_factor)
    if len(path) < 4:
        return None, "skip"
    spline = fit_smooth_spline(ds[path], voxel_size=vs,
                               smoothing_factor=smoothing_factor)
    if spline is None:
        return None, "skip"
    return spline, source


def process_case(case_dir, processed_masks, track_data, K, k_nn,
                 edge_factor, target_pcd_count, smoothing_factor):
    """Returns (anchors (T, K, 3), regularized object_points (T, N, 3),
    diagnostics dict).

    Method: per-frame raw-PCD skeleton → smooth cubic B-spline (kills
    voxel-grid jitter). Frame 0's smooth-spline arc-length L_0 is the
    canonical rope length. K anchors at *absolute* arc-lengths
    linspace(0, L_0, K), which keeps anchor positions stable across
    frames (no per-frame normalized rescaling). Per-frame queries clamp
    to [0, L_t] so heavy occlusion gracefully degrades rather than
    extrapolating wildly."""
    obj_tracks = np.asarray(track_data["object_points"], dtype=np.float64)
    T, N, _ = obj_tracks.shape

    # Frame 0: build smooth spline, fix orientation by PCA.
    sp0, src0 = _build_frame_spline(
        case_dir, 0, processed_masks, obj_tracks[0],
        k_nn=k_nn, edge_factor=edge_factor,
        target_pcd_count=target_pcd_count,
        smoothing_factor=smoothing_factor,
    )
    if sp0 is None:
        raise SystemExit("[skeleton] frame-0 spline fit failed")
    L0 = sp0["L"]
    pts_d0 = sp0["pts_dense"]

    cov = np.cov(pts_d0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, int(np.argmax(eigvals))]
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-9)
    proj_a = float((pts_d0[0] - pts_d0.mean(0)) @ pc1)
    proj_b = float((pts_d0[-1] - pts_d0.mean(0)) @ pc1)
    flip0 = (proj_a > proj_b)

    eigs_sorted = np.sort(eigvals)[::-1]
    pca_ratio = float(eigs_sorted[0] / max(eigs_sorted[1], 1e-12))

    # K anchors at *absolute* arc-lengths from L_0 (constant-length curve).
    s_anchor_abs = np.linspace(0.0, L0, K)

    # Per-track frame-0 absolute arc-length: project lifted track-0
    # positions onto the smooth frame-0 spline.
    s_dense_abs = project_to_smooth_spline(sp0, obj_tracks[0])
    if flip0:
        s_dense_abs = L0 - s_dense_abs

    anchors = np.zeros((T, K, 3), dtype=np.float64)
    obj_reg = obj_tracks.copy()                       # frame 0 stays raw
    anchors[0] = sample_at_arc(sp0, s_anchor_abs, flip=flip0)

    L_per_frame = [L0]
    src_counts = {src0: 1}
    flips = int(flip0)
    fallbacks = 0
    prev_a0 = anchors[0, 0].copy()

    for t in range(1, T):
        sp_t, src_t = _build_frame_spline(
            case_dir, t, processed_masks, obj_tracks[t],
            k_nn=k_nn, edge_factor=edge_factor,
            target_pcd_count=target_pcd_count,
            smoothing_factor=smoothing_factor,
        )
        if sp_t is None:
            anchors[t] = anchors[t - 1]
            obj_reg[t] = obj_reg[t - 1]
            fallbacks += 1
            src_counts["skip"] = src_counts.get("skip", 0) + 1
            L_per_frame.append(L_per_frame[-1])
            continue
        src_counts[src_t] = src_counts.get(src_t, 0) + 1
        L_t = sp_t["L"]
        # ID consistency: which spline endpoint matches prev anchor 0?
        end_a = sp_t["pts_dense"][0]
        end_b = sp_t["pts_dense"][-1]
        flip_t = (np.linalg.norm(end_a - prev_a0)
                  > np.linalg.norm(end_b - prev_a0))
        if flip_t:
            flips += 1
        anchors[t] = sample_at_arc(sp_t, s_anchor_abs, flip=flip_t)
        obj_reg[t] = sample_at_arc(sp_t, s_dense_abs, flip=flip_t)
        prev_a0 = anchors[t, 0].copy()
        L_per_frame.append(L_t)

    L_arr = np.array(L_per_frame)
    mean_disp = float(np.linalg.norm(obj_tracks - obj_reg, axis=2).mean())
    if T > 1:
        anchor_jitter = float(
            np.linalg.norm(np.diff(anchors, axis=0), axis=2).mean())
    else:
        anchor_jitter = 0.0
    diags = {
        "L0": L0,
        "L_min": float(L_arr.min()),
        "L_max": float(L_arr.max()),
        "L_std": float(L_arr.std()),
        "pca_ratio": pca_ratio,
        "mean_displacement": mean_disp,
        "anchor_jitter": anchor_jitter,
        "n_keypoints": K,
        "n_frames": T,
        "n_dense": N,
        "flips": flips,
        "fallbacks": fallbacks,
        "src_counts": src_counts,
    }
    return anchors.astype(np.float32), obj_reg.astype(np.float32), diags


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--case_name", required=True)
    ap.add_argument("--n_keypoints", type=int, default=9,
                    help="Number of arc-length-spaced anchors / GT keypoints.")
    ap.add_argument("--k", type=int, default=4,
                    help="kNN neighbors per vertex when building the per-frame "
                         "graph for skeleton extraction.")
    ap.add_argument("--edge-factor", type=float, default=3.0,
                    help="Drop kNN edges longer than FACTOR * median NN dist "
                         "(suppresses cross-fold edges that would shortcut "
                         "the graph diameter). 0 disables.")
    ap.add_argument("--target-pcd-count", type=int, default=3000,
                    help="Voxel-downsample raw object PCD to ~this many "
                         "points before skeleton extraction. Trade-off: "
                         "higher = denser skeleton but slower kNN graph + "
                         "Dijkstra; lower = coarser skeleton.")
    ap.add_argument("--smoothing-factor", type=float, default=2.0,
                    help="Smooth-spline smoothing strength (multiplier on "
                         "M * voxel_size**2). Higher = smoother spline, "
                         "less anchor jitter, but loses sharp rope bends. "
                         "Lower = follows polyline more tightly. 0.5-5 "
                         "is the useful range.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Don't write track_process_data_raw.pkl backup.")
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip gt_track_3d.mp4 viz rendering.")
    args = ap.parse_args()

    case_dir = Path(args.base_path) / args.case_name
    pkl = case_dir / "track_process_data.pkl"
    backup = case_dir / "track_process_data_raw.pkl"
    if not pkl.is_file():
        sys.exit(f"[skeleton] missing {pkl}")
    if not args.no_backup and not backup.is_file():
        shutil.copyfile(pkl, backup)
        print(f"[skeleton] backup -> {backup}")

    pm_path = case_dir / "mask" / "processed_masks.pkl"
    if not pm_path.is_file():
        sys.exit(f"[skeleton] missing {pm_path} — run data_process_mask.py first")
    with open(pm_path, "rb") as f:
        processed_masks = pickle.load(f)

    with open(pkl, "rb") as f:
        td = pickle.load(f)

    anchors, obj_reg, diags = process_case(
        case_dir, processed_masks, td,
        K=args.n_keypoints, k_nn=args.k, edge_factor=args.edge_factor,
        target_pcd_count=args.target_pcd_count,
        smoothing_factor=args.smoothing_factor,
    )

    print(f"[skeleton] PCA ratio={diags['pca_ratio']:.1f} (1D-ness check), "
          f"K={diags['n_keypoints']}, T={diags['n_frames']}, "
          f"N={diags['n_dense']}")
    print(f"[skeleton] L0={diags['L0']:.4f}m, "
          f"per-frame L=[{diags['L_min']:.4f}, {diags['L_max']:.4f}]m "
          f"(std={diags['L_std']:.4f}m)")
    print(f"[skeleton] anchor frame-to-frame jitter="
          f"{diags['anchor_jitter']:.4f}m  "
          f"mean dense displacement={diags['mean_displacement']:.4f}m  "
          f"flips={diags['flips']}, fallbacks={diags['fallbacks']}, "
          f"src={diags['src_counts']}")

    td["object_points"] = obj_reg
    with open(pkl, "wb") as f:
        pickle.dump(td, f)
    print(f"[skeleton] wrote {pkl}")

    gt_path = case_dir / "gt_track_3d.pkl"
    with open(gt_path, "wb") as f:
        pickle.dump(anchors, f)
    print(f"[skeleton] wrote {gt_path} ({anchors.shape})")

    if not args.no_viz:
        msg = render_viz(case_dir, anchors.astype(np.float64))
        print(f"[skeleton] viz: {msg}")


if __name__ == "__main__":
    main()
