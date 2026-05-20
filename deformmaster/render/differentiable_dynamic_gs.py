"""Differentiable LBS + dynamic Gaussian Splatting rendering for training.

Composes the same low-level primitives that the inference script
``gs_render_dynamics.py`` uses (``interpolate_motions``, ``render``) but
without the inference-only wrappers that break gradient flow:
``torch.no_grad``, ``copy.deepcopy``, ``.cpu()``, the smoothing lerp pass,
and per-frame trajectory accumulation.

The intended call pattern from the trainer is one frame at a time:

    relations = init_lbs_relations(particles_t0, K=16)            # once
    gs_xyz, gs_quat = base_gs.get_xyz, base_gs.get_rotation       # frame 0
    for t in window_frames:
        gs_xyz, gs_quat = step_lbs(
            particles[t-1], particles[t], gs_xyz, gs_quat, relations,
        )
        img = render_dynamic_frame(base_gs, gs_xyz, gs_quat, cam, pipe, bg)
        loss_t = l1_loss(img['render'][:3], gt[t])
        loss_t.backward()                                         # per-frame
"""

from __future__ import annotations

from typing import Optional

import torch

from torch.utils.checkpoint import checkpoint

from gaussian_splatting.dynamic_utils import (
    interpolate_motions,
    get_topk_indices,
    knn_weights,
)
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel


class DynamicGSView:
    """Thin GaussianModel-compatible view that swaps in dynamic xyz/rotation.

    The renderer reads only a small set of @property accessors
    (``get_xyz``, ``get_rotation``, ``get_scaling``, ``get_opacity``,
    ``get_features`` / ``get_features_dc`` / ``get_features_rest``,
    ``active_sh_degree``). We override the two dynamic ones and delegate
    everything else to the pretrained base GS. No deepcopy, no parameter
    re-creation — autograd flows through the supplied ``xyz`` and
    ``rotation`` tensors straight into whatever produced them.

    Args:
        base: pretrained GaussianModel providing static attributes
            (scaling, opacity, SH features, sh degree).
        xyz: (N, 3) per-Gaussian positions for this frame.
        rotation_normalized: (N, 4) per-Gaussian rotations as already-normalized
            quaternions. ``interpolate_motions`` returns normalized quats, and
            the renderer's rotation activation is also ``F.normalize``, so
            re-normalizing a normalized quat is a no-op.
    """

    def __init__(
        self,
        base: GaussianModel,
        xyz: torch.Tensor,
        rotation_normalized: torch.Tensor,
    ):
        self._base = base
        self._dyn_xyz = xyz
        self._dyn_rotation = rotation_normalized

    # ---- dynamic accessors ----
    @property
    def get_xyz(self) -> torch.Tensor:
        return self._dyn_xyz

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._dyn_rotation

    # ---- static accessors (delegate to base) ----
    @property
    def get_scaling(self) -> torch.Tensor:
        return self._base.get_scaling

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._base.get_opacity

    @property
    def get_features_dc(self) -> torch.Tensor:
        return self._base.get_features_dc

    @property
    def get_features_rest(self) -> torch.Tensor:
        return self._base.get_features_rest

    @property
    def get_features(self) -> torch.Tensor:
        return self._base.get_features

    @property
    def active_sh_degree(self) -> int:
        return self._base.active_sh_degree

    @property
    def max_sh_degree(self) -> int:
        return self._base.max_sh_degree

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return self._base.covariance_activation(
            self.get_scaling, scaling_modifier, self._dyn_rotation,
        )


def init_lbs_relations(particles_t0: torch.Tensor, K: int = 16) -> torch.Tensor:
    """Compute the static LBS bone-relation matrix from frame-0 particles.

    Called once at trainer initialisation; the result is reused for every
    rollout step. No grad needed — relations are fixed by frame-0 topology.
    """
    with torch.no_grad():
        return get_topk_indices(particles_t0, K=K)


def step_lbs(
    particles_prev: torch.Tensor,
    particles_cur: torch.Tensor,
    gs_xyz: torch.Tensor,
    gs_quat: torch.Tensor,
    relations: torch.Tensor,
    K: int = 16,
    chunk_size: int = 20_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One LBS step: warp Gaussians from frame t-1 to frame t.

    Differentiable through ``particles_prev``, ``particles_cur``, ``gs_xyz``,
    and ``gs_quat``. We chunk over Gaussians (not over time) to bound peak
    memory; chunking is grad-safe because each chunk is independent and we
    stitch outputs with ``torch.cat`` (which is autograd-friendly).

    Args:
        particles_prev: (n_particles, 3) MPM particle positions at t-1.
        particles_cur:  (n_particles, 3) MPM particle positions at t.
        gs_xyz:         (n_gs, 3) Gaussian positions at t-1.
        gs_quat:        (n_gs, 4) Gaussian rotations at t-1 (normalized quats).
        relations:      KNN index tensor from ``init_lbs_relations``.
        K:              KNN size for per-step weights (matches relations).
        chunk_size:     max Gaussians per ``interpolate_motions`` call.

    Returns:
        (xyz_t, quat_t) at frame t, both differentiable.
    """
    motions = particles_cur - particles_prev

    # Gradient checkpoint each chunk: forward saves only inputs, backward
    # re-runs forward for that chunk to get intermediates. Peak memory drops
    # from O(n_gs * n_bones) to O(chunk_size * n_bones), critical for cases
    # with dense Gaussian filling (e.g., double_compress_pillow: 16k bones,
    # 392k Gaussians would OOM on 80GB without this).
    def _chunk_fn(p_prev, p_motions, rel, xyz_c, quat_c):
        w = knn_weights(p_prev, xyz_c, K=K)
        new_xyz_c, new_quat_c, _ = interpolate_motions(
            bones=p_prev, motions=p_motions, relations=rel,
            weights=w, xyz=xyz_c, quat=quat_c,
        )
        return new_xyz_c, new_quat_c

    out_xyz, out_quat = [], []
    n = gs_xyz.shape[0]
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        xyz_chunk = gs_xyz[s:e]
        quat_chunk = gs_quat[s:e]
        new_xyz_chunk, new_quat_chunk = checkpoint(
            _chunk_fn,
            particles_prev, motions, relations, xyz_chunk, quat_chunk,
            use_reentrant=False,
        )
        out_xyz.append(new_xyz_chunk)
        out_quat.append(new_quat_chunk)

    new_xyz = torch.cat(out_xyz, dim=0)
    new_quat = torch.nn.functional.normalize(torch.cat(out_quat, dim=0), dim=-1)
    return new_xyz, new_quat


def render_dynamic_frame(
    base_gs: GaussianModel,
    gs_xyz: torch.Tensor,
    gs_quat: torch.Tensor,
    camera,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    use_gsplat: bool = True,
    override_color: Optional[torch.Tensor] = None,
) -> dict:
    """Render one dynamic frame from (gs_xyz, gs_quat) with grad enabled.

    Args:
        base_gs: pretrained GaussianModel (provides static scale/opacity/SH).
        gs_xyz, gs_quat: dynamic state at this frame (with grad).
        camera: gaussian_splatting.scene.cameras.Camera.
        pipe: PipelineParams (or any object with the same field names).
        bg_color: (3,) tensor on the same device as the GS.
        scaling_modifier: passes through to renderer.
        use_gsplat: True selects the gsplat backend (recommended); False
            falls back to the original 3DGS rasterizer.
        override_color: optional (N, 3) per-Gaussian colour override.

    Returns:
        Renderer result dict; ``result['render']`` is the (4, H, W) RGBA tensor
        with grad through both the GS state and any tensors that produced
        ``gs_xyz`` / ``gs_quat`` (e.g. MPM particle positions).
    """
    view = DynamicGSView(base_gs, gs_xyz, gs_quat)
    return render(
        camera, view, pipe, bg_color,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
        use_gsplat=use_gsplat,
    )
