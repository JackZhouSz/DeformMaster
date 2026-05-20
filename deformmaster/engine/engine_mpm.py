"""DeformMaster inference engine.

This module exposes :class:`DeformMasterMPMEngine`, a slim runtime that loads a
trained DeformMaster checkpoint and exposes the state needed by
``inference.py`` / ``playground.py``:

* ``simulator``       — Warp-backed differentiable MPM stepper
* ``residual_net``    — particle-grid neural residual (PNPGD) if enabled
* ``data``            — dataset frames, controllers, masks for the scene
* ``auto_offset``     — auto-centering offset applied to particles & boundary
* ``patch_centers``,
  ``patch_idx``,
  ``interp_weights``  — persistent patch assignment for material parameters
* ``controller_points`` — temporally smoothed controller trajectory
* :meth:`get_current_phys_props` — maps learned ``raw_*`` parameters to
  physical quantities used by the simulator
* :meth:`load_from_checkpoint` — restores a ``best_checkpoint.pt`` produced
  by the (not-yet-released) training pipeline.

Training code lives in a separate (not yet released) repository.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    module="warp",
)

import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ..data.dataset_mpm import DeformMasterDataset
from ..model.diff_simulator.warp_solver.simulator_warp import WarpMPMSimulator
from ..model.residual_pgnd import ResidualPGND
from ..utils.mpm_utils import youngs_poisson_to_lame


class DeformMasterMPMEngine:
    """Inference-time scene engine for DeformMaster."""

    def __init__(self, cfg, scene_id, resume_path=None):
        self.cfg = cfg
        self.device = torch.device(cfg.mpm.device)
        self.scene_id = scene_id

        # 1. Dataset
        self.dataset = DeformMasterDataset(cfg, case_name=scene_id)
        if len(self.dataset) == 0:
            raise ValueError(f"Scene {scene_id} not found!")
        self.data = self.dataset[0]

        # Optional precomputed controller mask
        data_root = getattr(cfg.data, "root", "./data")
        mask_path = os.path.join(data_root, scene_id, "controller_mask.npy")
        cfg_keep_n = getattr(cfg.mpm, "controller_keep_n", None)
        mask_disabled = cfg_keep_n is not None and cfg_keep_n <= 0
        cp = self.data["controller_points"]
        self._raw_ctrl_frame0 = cp[0].clone() if isinstance(cp, torch.Tensor) else cp[0].copy()
        self._controller_mask_used = None
        if not mask_disabled and os.path.exists(mask_path):
            ctrl_mask = np.load(mask_path).astype(bool)
            self._controller_mask_used = ctrl_mask
            if isinstance(cp, torch.Tensor):
                mask_t = torch.from_numpy(ctrl_mask).to(cp.device)
                self.data["controller_points"] = cp[:, mask_t]
            else:
                self.data["controller_points"] = cp[:, ctrl_mask]

        # 2. Learnable parameters (Patch level) — populated by checkpoint
        self.n_patches = getattr(cfg.mpm, "n_patches", 64)
        self.active_experts = getattr(cfg.mpm, "active_experts", ["nh", "co", "st", "fi"])
        num_experts = len(self.active_experts)

        config_weights = getattr(cfg.mpm, "init_weights", None)
        if config_weights and len(config_weights) == num_experts:
            w_tensor = torch.tensor(config_weights, device=self.device)
            w_tensor = w_tensor / w_tensor.sum()
            init_log_weights = torch.log(w_tensor + 1e-6)
            self.log_weights = nn.Parameter(init_log_weights.unsqueeze(0).repeat(self.n_patches, 1))
        else:
            init_weights = torch.ones(self.n_patches, num_experts) / num_experts
            self.log_weights = nn.Parameter(torch.log(init_weights).to(self.device))

        init_val = getattr(cfg.mpm, "init_raw_params", 0.0)

        def _get_init(name, default):
            return getattr(cfg.mpm, f"init_raw_{name}", default)

        self.raw_E = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * _get_init("E", init_val))
        self.raw_nu = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * _get_init("nu", init_val))
        self.raw_fiber_k = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * _get_init("fiber_k", init_val))
        self.raw_yield = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * _get_init("yield", init_val))
        self.raw_viscosity = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * _get_init("viscosity", init_val))
        # Fiber direction is replaced by checkpoint or initialized via local PCA below.
        self.raw_fiber_dir = nn.Parameter(torch.randn(self.n_patches, 3, device=self.device) * 0.1)
        self.raw_ctrl_stiffness = nn.Parameter(torch.tensor([_get_init("ctrl_stiffness", 0.0)], device=self.device))
        self.raw_ctrl_damping = nn.Parameter(torch.tensor([_get_init("ctrl_damping", 0.0)], device=self.device))

        # 3. Simulator (Warp backend only).
        if not getattr(cfg.mpm, "use_warp", True):
            raise RuntimeError(
                "DeformMaster release runs the Warp backend exclusively; set "
                "`mpm.use_warp: true` in your config (PyTorch backend has been "
                "removed)."
            )
        self.simulator = WarpMPMSimulator(cfg.mpm).to(self.device)
        self.simulator.debug_mode = False

        # 4. Residual PGND (optional)
        _rcfg = getattr(cfg, "residual", None)
        _residual_enabled = _rcfg is not None and getattr(_rcfg, "enabled", True)
        if _residual_enabled:
            self.residual_cfg = _rcfg
            self.residual_net = ResidualPGND(_rcfg).to(self.device)
        else:
            self.residual_cfg = None
            self.residual_net = None

        # 5. Optional mono-capture alignment: rotate data so the cloth-plane normal -> world +Z.
        if getattr(self.cfg.data, "mono_align", False):
            init_raw = self.data["init_pos"].to(self.device)
            centered = init_raw - init_raw.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            n = Vh[-1]
            if n[1].item() < 0:
                n = -n
            target = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=n.dtype)
            v = torch.linalg.cross(n, target)
            s = torch.linalg.norm(v)
            c = torch.dot(n, target)
            if s.item() < 1e-6:
                R = torch.eye(3, device=self.device, dtype=n.dtype) if c > 0 \
                    else torch.diag(torch.tensor([1.0, -1.0, -1.0], device=self.device, dtype=n.dtype))
            else:
                vx = torch.tensor([[0.0, -v[2], v[1]],
                                   [v[2], 0.0, -v[0]],
                                   [-v[1], v[0], 0.0]], device=self.device, dtype=n.dtype)
                R = torch.eye(3, device=self.device, dtype=n.dtype) + vx + vx @ vx * ((1.0 - c) / (s * s))
            self._mono_align_R = R
            for _k in ["init_pos", "gt_surface_tracks", "controller_points"]:
                if _k in self.data:
                    self.data[_k] = self.data[_k].to(self.device) @ R.T

        # 6. Auto-centering offset (X/Y center, Z grounded at floor_margin).
        obj_pts = self.data["init_pos"].to(self.device)
        gt_pts = self.data["gt_surface_tracks"].to(self.device).view(-1, 3)
        ctrl_pts = self.data["controller_points"].to(self.device).view(-1, 3)
        gt_valid = torch.linalg.norm(gt_pts, dim=-1) > 1e-5
        ctrl_valid = torch.linalg.norm(ctrl_pts, dim=-1) > 1e-5
        gt_pts = gt_pts[gt_valid]
        ctrl_pts = ctrl_pts[ctrl_valid]

        all_pts = torch.cat([obj_pts, gt_pts, ctrl_pts], dim=0)
        p_min = all_pts.min(dim=0)[0]
        p_max = all_pts.max(dim=0)[0]
        floor_margin = getattr(self.cfg.mpm, "floor_margin", 0.05)
        init_min_z = obj_pts[:, 2].min()
        self.auto_offset = torch.zeros(3, device=self.device)
        self.auto_offset[0] = -(p_min[0] + p_max[0]) / 2.0
        self.auto_offset[1] = -(p_min[1] + p_max[1]) / 2.0
        self.auto_offset[2] = floor_margin - init_min_z

        self.simulator.base_offset = self.auto_offset
        self.simulator._apply_boundary()
        self.simulator.reset(obj_pts, controller_pos=self.data["controller_points"][0].to(self.device))

        # 7. Smooth controller trajectory.
        self.controller_points = self.data["controller_points"].to(self.device) + self.auto_offset
        T_ctrl = self.controller_points.shape[0]
        window = getattr(self.cfg.mpm, "controller_smooth_window", 5)
        if window > 1:
            smoothed = self.controller_points.clone()
            for t in range(T_ctrl):
                t_start = max(0, t - window // 2)
                t_end = min(T_ctrl, t + window // 2 + 1)
                smoothed[t] = self.controller_points[t_start:t_end].mean(dim=0)
            self.controller_points = smoothed

        # 8. Persistent patch assignment (FPS centers + KNN interpolation).
        from pytorch3d.ops import sample_farthest_points, knn_points
        xyz_static = self.data["gaussians"][:, :3].unsqueeze(0).to(self.device)
        self.patch_centers, _ = sample_farthest_points(xyz_static, K=self.n_patches)
        init_pos_centered = (self.data["init_pos"].to(self.device) + self.auto_offset).unsqueeze(0)
        dist, self.patch_idx, _ = knn_points(init_pos_centered, self.patch_centers, K=3)
        dist = torch.clamp(dist, min=1e-6)
        inv_dist = 1.0 / dist
        norm = torch.sum(inv_dist, dim=2, keepdim=True)
        self.interp_weights = (inv_dist / norm).unsqueeze(-1)

        if "fi" in self.active_experts:
            self._init_fiber_dir_from_geometry(init_pos_centered.squeeze(0))

        # 9. Apply checkpoint if provided.
        if resume_path is not None and os.path.exists(resume_path):
            self.load_from_checkpoint(resume_path)

    def _init_fiber_dir_from_geometry(self, particle_pos):
        """Initialize fiber directions via local PCA around each patch center."""
        from pytorch3d.ops import knn_points

        pca_k = min(16, particle_pos.shape[0])
        _, nbr_idx, _ = knn_points(self.patch_centers, particle_pos.unsqueeze(0), K=pca_k)
        nbr_idx = nbr_idx.squeeze(0)
        nbr_pos = particle_pos[nbr_idx]
        centroid = nbr_pos.mean(dim=1, keepdim=True)
        centered = nbr_pos - centroid
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        tangent = Vh[:, 0, :]
        with torch.no_grad():
            self.raw_fiber_dir.copy_(tangent)

    def get_current_phys_props(self):
        weights = torch.softmax(self.log_weights, dim=-1)
        bounds = self.cfg.mpm.get("material_bounds", {})
        friction = torch.tensor(self.cfg.mpm.boundary.friction, device=self.device)

        E_min, E_max = bounds.get("E", [1e3, 1e6])
        nu_min, nu_max = bounds.get("nu", [0.0, 0.45])
        fk_min, fk_max = bounds.get("fiber_k", [5e3, 5e5])
        ys_min, ys_max = bounds.get("yield_stress", [1e2, 1e5])
        visc_min, visc_max = bounds.get("plastic_viscosity", [0.1, 10.0])

        def _log_sigmoid(raw_param, min_val, max_val):
            log_min = np.log(max(min_val, 1e-6))
            log_max = np.log(max_val)
            log_val = log_min + torch.sigmoid(raw_param) * (log_max - log_min)
            return torch.exp(log_val)

        p_E = _log_sigmoid(self.raw_E, E_min, E_max)
        p_nu = nu_min + torch.sigmoid(self.raw_nu) * (nu_max - nu_min)
        p_mu, p_lam = youngs_poisson_to_lame(p_E, p_nu)
        p_fiber_k = _log_sigmoid(self.raw_fiber_k, fk_min, fk_max)
        p_fiber_dir = self.raw_fiber_dir
        p_yield = _log_sigmoid(self.raw_yield, ys_min, ys_max)
        p_visc = _log_sigmoid(self.raw_viscosity, visc_min, visc_max)

        cs_min, cs_max = bounds.get("controller_stiffness", [1e3, 1e6])
        cd_min, cd_max = bounds.get("controller_damping", [1e2, 5e4])
        p_ctrl_stiffness = _log_sigmoid(self.raw_ctrl_stiffness, cs_min, cs_max)
        p_ctrl_damping = _log_sigmoid(self.raw_ctrl_damping, cd_min, cd_max)

        return (weights, p_mu.squeeze(), p_lam.squeeze(), p_fiber_k.squeeze(), p_fiber_dir,
                friction, p_yield.squeeze(), p_E.squeeze(), p_nu.squeeze(), p_visc.squeeze(),
                p_ctrl_stiffness.squeeze(), p_ctrl_damping.squeeze())

    def load_from_checkpoint(self, resume_path):
        if not resume_path.endswith(".pt"):
            raise ValueError(
                f"Unsupported checkpoint format: {resume_path}. DeformMaster "
                "release only loads `.pt` checkpoints."
            )

        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
        state = checkpoint["model_state_dict"]
        with torch.no_grad():
            self.log_weights.copy_(state["log_weights"])
            self.raw_E.copy_(state["raw_E"])
            self.raw_nu.copy_(state["raw_nu"])
            self.raw_fiber_k.copy_(state["raw_fiber_k"])
            self.raw_fiber_dir.copy_(state["raw_fiber_dir"])
            self.raw_yield.copy_(state["raw_yield"])
            self.raw_viscosity.copy_(state["raw_viscosity"])
            if "raw_ctrl_stiffness" in state:
                self.raw_ctrl_stiffness.copy_(state["raw_ctrl_stiffness"])
            if "raw_ctrl_damping" in state:
                self.raw_ctrl_damping.copy_(state["raw_ctrl_damping"])

            if "auto_offset" in state:
                self.auto_offset = state["auto_offset"].to(self.device)
                if hasattr(self, "simulator"):
                    self.simulator.base_offset = self.auto_offset
                    self.simulator._apply_boundary()

            if "patch_centers" in state:
                self.patch_centers = state["patch_centers"].to(self.device)
                from pytorch3d.ops import knn_points
                init_pos_centered = (self.data["init_pos"].to(self.device) + self.auto_offset).unsqueeze(0)
                dist, self.patch_idx, _ = knn_points(init_pos_centered, self.patch_centers, K=3)
                dist = torch.clamp(dist, min=1e-6)
                inv_dist = 1.0 / dist
                norm = torch.sum(inv_dist, dim=2, keepdim=True)
                self.interp_weights = (inv_dist / norm).unsqueeze(-1)

            if "residual_net" in state and self.residual_net is not None:
                try:
                    self.residual_net.load_state_dict(state["residual_net"])
                except Exception as exc:
                    print(f"[RESUME] Warning: failed to load residual_net weights: {exc}")
