"""Shape prior generation, default TRELLIS 1, optional TRELLIS 2.

Self-dispatches to the right conda env (`trellis1` or `trellis2`) based on
``--version``. Process_data.py can keep calling this script with bare
``python`` — the script will re-exec itself under the correct env before
importing the pipeline.

Defaults to TRELLIS 1 because: (a) faster (~10s inference vs trellis2 17-60s),
(b) simpler env (no BiRefNet / webp / decimation patches), (c) mesh density
~7K faces is the ARAP / pytorch3d sweet spot natively. TRELLIS 2 is kept as
a fallback for cases where trellis 1 produces poor geometry.

Usage::
    python data_process/shape_prior.py \
        --img_path <case>/shape/masked_image.png \
        --output_dir <case>/shape \
        [--version 1|2]
Output: <output_dir>/object.glb
"""
import os
import sys

DEFAULT_VERSION = "1"

# --------- self-dispatch ---------------------------------------------------
# Optional separate TRELLIS environments. If these are not set, the script
# runs in the current Python environment.
TRELLIS_PY = {
    "1": os.environ.get("DEFORMMASTER_TRELLIS1_PYTHON", sys.executable),
    "2": os.environ.get("DEFORMMASTER_TRELLIS2_PYTHON", sys.executable),
}

# Pre-parse only --version to choose the env; full argparse runs after dispatch.
_version = DEFAULT_VERSION
for i, a in enumerate(sys.argv[1:]):
    if a == "--version" and i + 1 < len(sys.argv) - 1:
        _version = sys.argv[i + 2]
        break
    if a.startswith("--version="):
        _version = a.split("=", 1)[1]
        break
if _version not in TRELLIS_PY:
    print(f"[shape_prior] invalid --version {_version!r}, must be 1 or 2")
    sys.exit(2)

# Force the env vars trellis 1 needs BEFORE either dispatch or in-process import.
# Setting only before execv would miss the case where the user invokes this
# script directly with the trellis1 env's python (no dispatch needed).
if _version == "1":
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")

target_py = TRELLIS_PY[_version]
if os.path.realpath(sys.executable) != os.path.realpath(target_py):
    if os.path.isfile(target_py):
        os.execv(target_py, [target_py, os.path.abspath(__file__)] + sys.argv[1:])
    else:
        env_key = f"DEFORMMASTER_TRELLIS{_version}_PYTHON"
        print(f"[shape_prior] TRELLIS {_version} Python not found at {target_py}. "
              f"Set {env_key}=/path/to/python or run this script inside the "
              "matching TRELLIS environment.")
        sys.exit(1)

# --- from here on we're in the right trellis env ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from argparse import ArgumentParser
from PIL import Image


def _run_trellis1(args):
    """TRELLIS 1 (Microsoft, structured 3D latents, ~1B params)."""
    # Resolve user-facing paths to absolute BEFORE we chdir; otherwise the
    # subsequent chdir into TRELLIS_REPO breaks relative paths like
    # "./data/different_types/<case>/shape/masked_image.png".
    args.img_path = os.path.abspath(args.img_path)
    args.output_dir = os.path.abspath(args.output_dir)

    # Repo is vendored at data_process/TRELLIS/
    _HERE = os.path.dirname(os.path.abspath(__file__))
    TRELLIS1_REPO = os.path.abspath(os.path.join(_HERE, "TRELLIS"))
    sys.path.insert(0, TRELLIS1_REPO)
    # TRELLIS 1's from_pretrained resolves nested ckpt paths (e.g.
    # "ckpts/ss_flow_img_dit_L_16l8_fp16") relative to cwd, so we must
    # chdir into the repo before invoking it.
    os.chdir(TRELLIS1_REPO)

    import torch, random, numpy as np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    print(f"[shape_prior] Loading TRELLIS 1 ({args.model_v1})...")
    pipe = TrellisImageTo3DPipeline.from_pretrained(args.model_v1)
    pipe.cuda()

    print(f"[shape_prior] Image: {args.img_path}")
    image = Image.open(args.img_path)

    print(f"[shape_prior] Running inference...")
    out = pipe.run(image, seed=args.seed)
    mesh = out["mesh"][0]
    gaussian = out["gaussian"][0]
    print(f"[shape_prior] Raw mesh: V={len(mesh.vertices)} F={len(mesh.faces)}")

    print(f"[shape_prior] Exporting GLB (simplify={args.simplify}, texture_size={args.texture_size})...")
    glb = postprocessing_utils.to_glb(
        gaussian, mesh,
        simplify=args.simplify,
        texture_size=args.texture_size,
        verbose=False,
    )
    out_path = os.path.join(args.output_dir, "object.glb")
    glb.export(out_path)
    print(f"[shape_prior] Saved {out_path}")


def _run_trellis2(args):
    """TRELLIS.2 (Microsoft, O-Voxel, 4B params). Slower but higher detail."""
    # Repo is vendored at data_process/third_party/TRELLIS.2/
    _HERE = os.path.dirname(os.path.abspath(__file__))
    TRELLIS2_REPO = os.path.abspath(os.path.join(_HERE, "third_party",
                                                 "TRELLIS.2"))
    sys.path.insert(0, TRELLIS2_REPO)

    # TRELLIS 2 unconditionally loads briaai/RMBG-2.0 (gated HF repo) in
    # from_pretrained even though it skips rembg when input has alpha. Our
    # masked_image.png IS RGBA, so make BiRefNet a no-op.
    from trellis2.pipelines.rembg import BiRefNet
    BiRefNet.__init__ = lambda self, *a, **k: None
    BiRefNet.__call__ = lambda self, img, *a, **k: img
    BiRefNet.to = lambda self, *a, **k: None
    BiRefNet.cpu = lambda self, *a, **k: None

    import torch, random, numpy as np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    print(f"[shape_prior] Loading TRELLIS 2 ({args.model_v2})...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_v2)
    pipeline.cuda()

    print(f"[shape_prior] Image: {args.img_path}")
    image = Image.open(args.img_path)

    print(f"[shape_prior] Running inference...")
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)  # nvdiffrast face limit

    print(f"[shape_prior] Exporting GLB (decimation_target={args.decimation_target}, texture_size={args.texture_size})...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    out_path = os.path.join(args.output_dir, "object.glb")
    # extension_webp=False: pytorch3d's gltf reader (used by align.py)
    # cannot parse KHR_texture_webp.
    glb.export(out_path, extension_webp=False)
    print(f"[shape_prior] Saved {out_path}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--version", choices=["1", "2"], default=DEFAULT_VERSION,
                        help="TRELLIS version (1=fast/simple, 2=detail-heavy).")
    parser.add_argument("--model_v1", type=str,
                        default="JeffreyXiang/TRELLIS-image-large",
                        help="HF model id for TRELLIS 1.")
    parser.add_argument("--model_v2", type=str,
                        default="microsoft/TRELLIS.2-4B",
                        help="HF model id for TRELLIS 2.")
    parser.add_argument("--simplify", type=float, default=0.95,
                        help="TRELLIS 1: face decimation ratio (0.95 keeps 5%).")
    parser.add_argument("--decimation-target", type=int, default=10000,
                        help="TRELLIS 2: target face count.")
    parser.add_argument("--texture-size", type=int, default=1024,
                        help="Texture resolution (4096 OOMs pytorch3d).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible mesh generation.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.version == "1":
        _run_trellis1(args)
    else:
        _run_trellis2(args)


if __name__ == "__main__":
    main()
