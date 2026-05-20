# Third-Party Notices

DeformMaster builds on several open-source projects. The DeformMaster code
released here (`deformmaster/`, `inference.py`, `playground.py`, `configs/`,
documentation) is licensed under the MIT License (see `LICENSE`).

Vendored third-party code keeps its original license:

## `gaussian_splatting/`

Code under `gaussian_splatting/` is derived from the original
**3D Gaussian Splatting** reference implementation by the GRAPHDECO group at
Inria (https://github.com/graphdeco-inria/gaussian-splatting), and from
related upstream projects whose copyright headers are preserved inside the
individual files (Inria; PlenOctree Authors; ETH Zurich & UNC Chapel Hill).

The Inria 3D Gaussian Splatting code is released for **non-commercial,
research and evaluation use only** under the original Inria license. By using
files in `gaussian_splatting/` you agree to those upstream terms in addition
to the MIT License covering the rest of this repository. See
`gaussian_splatting/LICENSE_NOTICE.md` for details and the upstream license
link.

## `gs_render.py`

`gs_render.py` is derived from the Inria 3D Gaussian Splatting reference code
and follows the same Inria non-commercial research license.

## CUDA rasterizer dependencies (installed at runtime, not vendored)

`install.sh` pip-installs the following from upstream and they follow their
own licenses:

- `diff-gaussian-rasterization` (Inria, non-commercial)
- `fused-ssim` (Inria, non-commercial)
- `simple-knn` (Inria, non-commercial)

If you need a fully permissive (Apache-2.0) rasterizer, see
[`gsplat`](https://github.com/nerfstudio-project/gsplat); swapping the
rasterizer requires light code changes inside `gaussian_splatting/`.
