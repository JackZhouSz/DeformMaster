"""Decimate a GLB to a target triangle count and bake per-vertex colors.

The curated TRELLIS-style meshes can have 100k+ triangles, which makes
gs_train initialise millions of Gaussians and run very slowly. This
script simplifies each mesh via open3d's quadric decimation and writes
the result back as a GLB with explicit vertex colors so open3d can
read them at training time.

Usage:
    python data_process/decimate_glb.py --target-tris 20000 \\
        data/gaussian_data/my_*/shape_prior.glb
"""
import argparse
import numpy as np
import open3d as o3d
import trimesh


def get_vertex_colors(m):
    n = len(m.vertices)
    if hasattr(m.visual, "to_color"):
        m.visual = m.visual.to_color()
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is None or len(vc) != n:
        vc = np.tile([180, 180, 180, 255], (n, 1)).astype(np.uint8)
    return np.asarray(vc, dtype=np.uint8)


def decimate(path, target_tris):
    m = trimesh.load(path, force="mesh", process=False)
    n_tris_in = len(m.faces)
    if n_tris_in <= target_tris:
        print(f"[skip] {path}: {n_tris_in} tris already <= target {target_tris}")
        return

    vc = get_vertex_colors(m)  # [N, 4] uint8 RGBA

    # build open3d mesh with vertex colors (RGB float in [0,1])
    o3m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices)),
        o3d.utility.Vector3iVector(np.asarray(m.faces)),
    )
    o3m.vertex_colors = o3d.utility.Vector3dVector(vc[:, :3].astype(np.float32) / 255.0)
    o3m_simp = o3m.simplify_quadric_decimation(target_number_of_triangles=int(target_tris))
    # Decimation leaves the original vertex array intact, so most vertices
    # become orphans (verts/tris ratio ~50). open3d ops like
    # sample_points_uniformly can segfault on this layout — drop them.
    o3m_simp.remove_unreferenced_vertices()
    o3m_simp.remove_duplicated_vertices()
    o3m_simp.remove_duplicated_triangles()
    o3m_simp.remove_degenerate_triangles()

    out_v = np.asarray(o3m_simp.vertices)
    out_f = np.asarray(o3m_simp.triangles)
    out_vc_rgb = (np.asarray(o3m_simp.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
    out_vc = np.concatenate(
        [out_vc_rgb, np.full((len(out_v), 1), 255, dtype=np.uint8)], axis=1
    )

    fresh = trimesh.Trimesh(
        vertices=out_v, faces=out_f,
        vertex_colors=out_vc, process=False,
    )
    fresh.export(path)
    print(f"[decimate] {path}: {n_tris_in} -> {len(out_f)} tris "
          f"(verts {len(m.vertices)} -> {len(out_v)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-tris", type=int, default=9000,
                        help="Target triangle count (default: 9000, matching "
                             "the ~5k-7k range of existing curated meshes)")
    parser.add_argument("paths", nargs="+", help="GLB files to decimate in-place")
    args = parser.parse_args()
    for p in args.paths:
        decimate(p, args.target_tris)


if __name__ == "__main__":
    main()
