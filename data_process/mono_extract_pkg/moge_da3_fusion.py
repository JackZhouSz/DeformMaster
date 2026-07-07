"""
MoGe + DA3 Fusion: MoGe per-frame depth + DA3 inter-frame poses → metric world coordinates.

MoGe: high-quality per-frame 3D in camera coords (metric scale)
DA3:  camera poses c2w (inter-frame motion) + lower-quality depth

Fusion: estimate scale alignment, then transform MoGe camera points to world via DA3 poses.

Usage:
    python -m mono_phys_gaussian.moge_da3_fusion \
        --moge_dir results/{CASE}/moge_output \
        --da3_dir results/{CASE}/3d_efep_output \
        --output_dir results/{CASE}/fused_output
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.spatial import cKDTree


def estimate_scale_and_rotation(moge_pts_cam, da3_pts_world, c2w, mask, max_pts=10000):
    """
    Estimate scale + rotation between MoGe camera coords and DA3 world coords.

    DA3 world points = c2w @ (s * R_correction * MoGe camera points)

    In ideal case R_correction = I (both use same camera model).
    We estimate s and R_correction via Procrustes alignment.

    Args:
        moge_pts_cam: (H, W, 3) MoGe points in camera coords
        da3_pts_world: (H, W, 3) DA3 points in world coords
        c2w: (4, 4) DA3 camera-to-world for this frame
        mask: (H, W) bool, valid pixels to use
        max_pts: subsample for speed

    Returns:
        scale: float, multiply MoGe points by this
        R_correction: (3, 3) rotation correction (should be near identity)
        t_correction: (3,) translation correction
    """
    # Get valid points
    moge_flat = moge_pts_cam[mask]
    da3_flat = da3_pts_world[mask]

    # Remove NaN/inf
    valid = (np.isfinite(moge_flat).all(axis=1) &
             np.isfinite(da3_flat).all(axis=1) &
             (moge_flat[:, 2] > 0.01))  # MoGe z should be positive (forward)
    moge_flat = moge_flat[valid]
    da3_flat = da3_flat[valid]

    if len(moge_flat) < 100:
        print("  Warning: too few valid points for scale estimation")
        return 1.0, np.eye(3), np.zeros(3)

    # Subsample
    if len(moge_flat) > max_pts:
        idx = np.random.RandomState(42).choice(len(moge_flat), max_pts, replace=False)
        moge_flat = moge_flat[idx]
        da3_flat = da3_flat[idx]

    # Transform DA3 world points back to camera coords for comparison
    w2c = np.linalg.inv(c2w)
    da3_homo = np.hstack([da3_flat, np.ones((len(da3_flat), 1))])
    da3_cam = (w2c @ da3_homo.T).T[:, :3]

    # Now both moge_flat and da3_cam are in camera coords
    # Estimate scale: s * moge ≈ da3_cam
    # Use depth (z) ratio as robust scale estimate
    moge_z = moge_flat[:, 2]
    da3_z = da3_cam[:, 2]
    valid_z = (moge_z > 0.01) & (da3_z > 0.01)

    if valid_z.sum() < 50:
        return 1.0, np.eye(3), np.zeros(3)

    scale_ratios = da3_z[valid_z] / moge_z[valid_z]
    scale = np.median(scale_ratios)

    # Now estimate rotation correction via Procrustes
    # scaled_moge should match da3_cam
    scaled_moge = moge_flat * scale

    src_center = scaled_moge.mean(axis=0)
    tgt_center = da3_cam.mean(axis=0)
    src_c = scaled_moge - src_center
    tgt_c = da3_cam - tgt_center

    H_mat = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H_mat)
    R_correction = Vt.T @ U.T
    if np.linalg.det(R_correction) < 0:
        Vt[-1] *= -1
        R_correction = Vt.T @ U.T

    t_correction = tgt_center - R_correction @ src_center

    # Compute alignment quality
    aligned = (R_correction @ scaled_moge.T).T + t_correction
    residual = np.linalg.norm(aligned - da3_cam, axis=1)

    angle = np.degrees(np.arccos(np.clip((np.trace(R_correction) - 1) / 2, -1, 1)))

    print(f"  Scale: {scale:.4f}")
    print(f"  Rotation correction: {angle:.2f} deg")
    print(f"  Alignment residual: mean={residual.mean():.4f}m, median={np.median(residual):.4f}m")

    return scale, R_correction, t_correction


def fuse_moge_da3(
    moge_dir: str,
    da3_dir: str,
    output_dir: str,
    mask_dir: str = None,
):
    """
    Fuse MoGe per-frame point clouds with DA3 camera poses.

    Args:
        moge_dir: Directory with MoGe outputs (world_maps.npy)
        da3_dir: Directory with DA3 outputs (c2w.npy, world_maps_from_ply.npy)
        output_dir: Directory to save fused results
        mask_dir: Optional object mask directory for scale estimation
    """
    moge_dir = Path(moge_dir)
    da3_dir = Path(da3_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MoGe + DA3 Fusion")
    print("=" * 60)

    # Load data
    print("\nLoading MoGe point maps...")
    moge_maps = np.load(str(moge_dir / 'world_maps.npy'))  # (T, H, W, 3) camera coords
    T, H, W, _ = moge_maps.shape
    print(f"  MoGe: {moge_maps.shape}")

    print("Loading DA3 camera poses...")
    c2w = np.load(str(da3_dir / 'c2w.npy'))  # (T, 4, 4)
    print(f"  DA3 c2w: {c2w.shape}")

    # Load DA3 world maps for scale estimation
    da3_maps_path = da3_dir / 'world_maps_from_ply.npy'
    if da3_maps_path.exists():
        da3_maps = np.load(str(da3_maps_path))
        print(f"  DA3 world maps: {da3_maps.shape}")
    else:
        print("  Warning: DA3 world maps not found, using scale=1.0")
        da3_maps = None

    T = min(T, len(c2w))
    if da3_maps is not None:
        T = min(T, len(da3_maps))

    # Create mask for scale estimation (use full image or object mask)
    if mask_dir is not None:
        mask_files = sorted(Path(mask_dir).glob('mask_*.png'))
        mask_f0 = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE) > 127
        if mask_f0.shape != (H, W):
            mask_f0 = cv2.resize(mask_f0.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        # Use all pixels with valid depth
        mask_f0 = (moge_maps[0, :, :, 2] > 0.01) & np.isfinite(moge_maps[0, :, :, 2])

    # Estimate scale and rotation correction from frame 0
    print("\nEstimating scale alignment (frame 0)...")
    if da3_maps is not None:
        scale, R_corr, t_corr = estimate_scale_and_rotation(
            moge_maps[0], da3_maps[0], c2w[0], mask_f0
        )
    else:
        scale, R_corr, t_corr = 1.0, np.eye(3), np.zeros(3)
        print("  Using default scale=1.0")

    # Verify scale consistency across frames
    if da3_maps is not None:
        print("\nVerifying scale consistency...")
        scales = []
        for t in range(0, T, max(1, T // 5)):
            mask_t = (moge_maps[t, :, :, 2] > 0.01) & np.isfinite(moge_maps[t, :, :, 2])
            if da3_maps[t, :, :, 2][mask_t].sum() == 0:
                continue
            w2c_t = np.linalg.inv(c2w[t])
            da3_cam = np.zeros_like(da3_maps[t])
            pts_h = np.concatenate([da3_maps[t].reshape(-1, 3),
                                     np.ones((H*W, 1))], axis=-1)
            pts_cam = (w2c_t @ pts_h.T).T[:, :3].reshape(H, W, 3)

            moge_z = moge_maps[t, :, :, 2][mask_t]
            da3_z = pts_cam[:, :, 2][mask_t]
            valid = (moge_z > 0.01) & (da3_z > 0.01)
            if valid.sum() > 100:
                s_t = np.median(da3_z[valid] / moge_z[valid])
                scales.append(s_t)
                print(f"  Frame {t}: scale={s_t:.4f}")

        if len(scales) > 1:
            print(f"  Scale std: {np.std(scales):.4f} (mean={np.mean(scales):.4f})")

    # Apply fusion: transform MoGe camera points to world coords
    print(f"\nFusing {T} frames...")
    fused_maps = np.zeros((T, H, W, 3), dtype=np.float32)

    for t in range(T):
        # Step 1: Scale MoGe points
        pts_scaled = moge_maps[t] * scale

        # Step 2: Apply rotation correction (in camera space)
        pts_corrected = pts_scaled.reshape(-1, 3) @ R_corr.T + t_corr

        # Step 3: Transform to world coords using DA3 pose
        pts_homo = np.concatenate([pts_corrected, np.ones((H*W, 1))], axis=-1)
        pts_world = (c2w[t] @ pts_homo.T).T[:, :3]

        fused_maps[t] = pts_world.reshape(H, W, 3)

    # Save
    np.save(str(output_dir / 'world_maps_fused.npy'), fused_maps)
    np.save(str(output_dir / 'c2w.npy'), c2w[:T])
    np.save(str(output_dir / 'scale.npy'), np.array([scale]))

    # Save compatibility copy (DO NOT overwrite DA3 originals)
    # Users should symlink or copy manually when ready to use fused data
    print(f"  NOTE: To use fused data in pipeline, copy or symlink:")
    print(f"    cp {output_dir}/world_maps_fused.npy {{3d_efep_output}}/world_maps_from_ply.npy")

    print(f"\nSaved:")
    print(f"  Fused world maps: {fused_maps.shape}")
    print(f"  Scale: {scale:.4f}")
    print(f"  Compatible output: {compat_dir}/")

    # Quality check: depth range in mask
    if mask_dir is not None:
        for t in [0, T//2, T-1]:
            mask_t = cv2.imread(str(sorted(Path(mask_dir).glob('mask_*.png'))[t]),
                                cv2.IMREAD_GRAYSCALE) > 127
            if mask_t.shape != (H, W):
                mask_t = cv2.resize(mask_t.astype(np.uint8), (W, H),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
            d = fused_maps[t, :, :, 2][mask_t]
            d = d[np.isfinite(d)]
            print(f"  Frame {t} fused depth in mask: range={d.max()-d.min():.4f}m, std={d.std():.4f}m")

    return fused_maps, c2w[:T], scale


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--moge_dir', type=str, required=True)
    parser.add_argument('--da3_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, default=None)
    args = parser.parse_args()

    fuse_moge_da3(
        moge_dir=args.moge_dir,
        da3_dir=args.da3_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
    )
