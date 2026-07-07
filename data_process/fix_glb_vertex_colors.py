"""Convert GLB texture to per-vertex colors so open3d can read them.

open3d's GLB loader sometimes drops the UV texture link, leaving the mesh
with neither triangle_uvs nor vertex_colors. Downstream appearance tools may
then fail with "Mesh has no texture or valid vertex colors." Trimesh reads
GLB textures reliably, so we sample the texture at each vertex and bake the
result back into the GLB as vertex_colors.

Usage:
    python data_process/fix_glb_vertex_colors.py data/mesh/best_teddy.glb [...]
"""
import argparse
import numpy as np
import trimesh


def fix(path):
    m = trimesh.load(path, force="mesh", process=False)
    n = len(m.vertices)

    # bake whatever color source trimesh has into per-vertex colors
    if hasattr(m.visual, "to_color"):
        m.visual = m.visual.to_color()
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is None or len(vc) != n:
        vc = np.tile([180, 180, 180, 255], (n, 1)).astype(np.uint8)
    vc = np.asarray(vc, dtype=np.uint8)

    # rebuild with explicit vertex_colors so the GLB writer emits COLOR_0
    fresh = trimesh.Trimesh(
        vertices=m.vertices, faces=m.faces,
        vertex_colors=vc, process=False,
    )
    fresh.export(path)
    print(f"[fix] {path}: {n} verts, vertex_colors={vc.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="GLB files to fix in-place")
    args = parser.parse_args()
    for p in args.paths:
        fix(p)


if __name__ == "__main__":
    main()
