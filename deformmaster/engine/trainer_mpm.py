import warnings
# [FIX] Suppress Warp's UserWarning about non-leaf tensors early
warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed", module="warp")

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import taichi as ti # [FIX] Add taichi to set logging level
ti.init(arch=ti.cpu, log_level=ti.WARN) # [FIX] Suppress Taichi startup messages
import pickle
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.loss import chamfer_distance

# Use Agg backend for headless rendering
import matplotlib
matplotlib.use('Agg')

from ..model.diff_simulator.warp_solver.simulator_warp import WarpMPMSimulator
from ..data.dataset_mpm import DeformMasterDataset
from ..utils.mpm_utils import youngs_poisson_to_lame
from ..model.residual_pgnd import ResidualPGND
import warp as wp
from ..model.diff_simulator.warp_solver.warp_utils import torch2warp_vec3, torch2warp_float

class PhysExpertMPMTrainer:
    """
    Stage 1: System Identification.
    Directly optimize physical parameters for a specific scene using Differentiable MPM.
    """
    def __init__(self, cfg, scene_id, resume_path=None, for_inference=False):
        self.cfg = cfg
        self.for_inference = for_inference
        self.device = torch.device(cfg.mpm.device)
        self.scene_id = scene_id
        
        # [NEW] Save config for reproducibility (At Start)
        if not for_inference:
            os.makedirs(os.path.join(cfg.output_dir, scene_id), exist_ok=True)
            config_save_path = os.path.join(cfg.output_dir, scene_id, "config.yaml")
            OmegaConf.save(cfg, config_save_path)
            print(f"Config saved to {config_save_path}")
        
        # 1. Setup Data
        self.dataset = DeformMasterDataset(cfg, case_name=scene_id)
        if len(self.dataset) == 0:
            raise ValueError(f"Scene {scene_id} not found!")
        self.data = self.dataset[0]

        # Apply legacy per-case controller mask if (a) the mask file exists AND
        # (b) cfg.mpm.controller_keep_n is not explicitly disabled (<=0).
        # New configs normally set controller_keep_n <= 0 and keep all controllers.
        data_root = getattr(cfg.data, 'root', './data/different_types')
        mask_path = os.path.join(data_root, scene_id, 'controller_mask.npy')
        cfg_keep_n = getattr(cfg.mpm, 'controller_keep_n', None)
        mask_disabled = (cfg_keep_n is not None and cfg_keep_n <= 0)
        # Cache the raw frame-0 controllers and the mask itself for later visualization
        cp = self.data['controller_points']
        self._raw_ctrl_frame0 = cp[0].clone() if isinstance(cp, torch.Tensor) else cp[0].copy()
        self._controller_mask_used = None
        if mask_disabled:
            print(f"[INFO] Controller mask disabled (controller_keep_n={cfg_keep_n}) for {scene_id}")
        elif os.path.exists(mask_path):
            ctrl_mask = np.load(mask_path).astype(bool)  # [C]
            self._controller_mask_used = ctrl_mask
            if isinstance(cp, torch.Tensor):
                mask_t = torch.from_numpy(ctrl_mask).to(cp.device)
                self.data['controller_points'] = cp[:, mask_t]
            else:
                self.data['controller_points'] = cp[:, ctrl_mask]
            print(f"[INFO] Applied controller mask for {scene_id}: {int(ctrl_mask.sum())}/{len(ctrl_mask)}")

        self.residual_topology_idx = None
        self.residual_topology_rest = None
        self.residual_topology_mask = None
        
        # 2. Initialize Learnable Parameters (at Patch Level)
        self.n_patches = getattr(cfg.mpm, 'n_patches', 64)
        self.active_experts = getattr(cfg.mpm, 'active_experts', ['nh', 'co', 'st', 'fi'])
        num_experts = len(self.active_experts)
        
        # [NEW] Track best loss for saving best checkpoint
        self.best_loss = float('inf')
        
        # Expert Weights: [K, num_active]
        # [NEW] Load initial weights from config if available, else uniform
        config_weights = getattr(cfg.mpm, 'init_weights', None)
        if config_weights and len(config_weights) == num_experts:
            # Normalize to sum to 1 just in case
            w_tensor = torch.tensor(config_weights, device=self.device)
            w_tensor = w_tensor / w_tensor.sum()
            # Convert to log space because we optimize log_weights
            # Add epsilon to avoid log(0)
            init_log_weights = torch.log(w_tensor + 1e-6)
            # Expand to all patches: [1, num_experts] -> [n_patches, num_experts]
            self.log_weights = nn.Parameter(init_log_weights.unsqueeze(0).repeat(self.n_patches, 1))
            print(f"Initialized expert weights from config: {config_weights}")
        else:
            init_weights = torch.ones(self.n_patches, num_experts) / num_experts
            self.log_weights = nn.Parameter(torch.log(init_weights).to(self.device))
            print("Initialized expert weights uniformly.")
        
        # [REVISED] Prefer parameter-specific initial values, then fall back to init_raw_params.
        init_val = getattr(cfg.mpm, 'init_raw_params', 0.0)
        
        def get_init(name, default):
            return getattr(cfg.mpm, f'init_raw_{name}', default)

        self.raw_E = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('E', init_val))
        self.raw_nu = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('nu', init_val))
        self.raw_fiber_k = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('fiber_k', init_val))
        self.raw_yield = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('yield', init_val))
        self.raw_viscosity = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('viscosity', init_val))
        
        # fiber_dir will be re-initialized after patch assignment using local PCA;
        # placeholder here, overwritten by _init_fiber_dir_from_geometry().
        self.raw_fiber_dir = nn.Parameter(torch.randn(self.n_patches, 3, device=self.device) * 0.1)

        # Controller gains (scalar, not per-patch)
        self.raw_ctrl_stiffness = nn.Parameter(torch.tensor([get_init('ctrl_stiffness', 0.0)], device=self.device))
        self.raw_ctrl_damping = nn.Parameter(torch.tensor([get_init('ctrl_damping', 0.0)], device=self.device))

        # 3. [NEW] Resume Path Search (Moved earlier to decide log_dir)
        self.resume_checkpoint = None
        self.resume_path = resume_path
        
        # Priority: explicit resume_path > checkpoint_*.pt > optimized_params.pkl
        if self.resume_path is None:
            # 1. Check for full checkpoints (*.pt)
            log_base = os.path.join(cfg.output_dir, scene_id, 'mpm_train')
            pt_checkpoints = []
            if os.path.exists(log_base):
                for root, dirs, files in os.walk(log_base):
                    for f in files:
                        if f.startswith("checkpoint_iter_") and f.endswith(".pt"):
                            full_path = os.path.join(root, f)
                            try:
                                iter_num = int(f.split('_')[-1].split('.')[0])
                                pt_checkpoints.append((iter_num, full_path))
                            except ValueError:
                                continue
            
            # 2. Check for final checkpoint
            final_ckpt = os.path.join(cfg.output_dir, scene_id, "final_checkpoint.pt")
            if os.path.exists(final_ckpt):
                 self.resume_path = final_ckpt
            elif pt_checkpoints:
                 self.resume_path = sorted(pt_checkpoints, key=lambda x: x[0])[-1][1]
            else:
                # Fallback to old pkl logic
                opt_path = os.path.join(cfg.output_dir, scene_id, "optimized_params.pkl")
                if os.path.exists(opt_path):
                    self.resume_path = opt_path
                else:
                    # Look for params_iter_*.pkl
                    if os.path.exists(log_base):
                        checkpoints = []
                        for root, dirs, files in os.walk(log_base):
                            for f in files:
                                if f.startswith("params_iter_") and f.endswith(".pkl"):
                                    full_path = os.path.join(root, f)
                                    try:
                                        iter_num = int(f.split('_')[-1].split('.')[0])
                                        checkpoints.append((iter_num, full_path))
                                    except ValueError:
                                        continue
                        if checkpoints:
                            self.resume_path = sorted(checkpoints, key=lambda x: x[0])[-1][1]

        # [GUARD] Detect ablation directory accidentally pointing at existing checkpoint.
        # If output_dir contains "abl_" (our ablation naming convention) but a checkpoint
        # already exists, that almost certainly means a stale yaml routed training to the
        # wrong place. Fail loudly instead of silently resuming with old state.
        #
        # Skipped when:
        #   - for_inference: caller explicitly loads a checkpoint afterwards.
        #   - resume_path was provided explicitly (e.g. Stage-3 RGB finetune
        #     resuming from a Stage-2 ckpt that legitimately lives in a
        #     sibling abl_*/ tree). The guard only targets *auto-discovered*
        #     stale state inside the ablation output_dir.
        if (not for_inference
                and resume_path is None
                and self.resume_path
                and "abl_" in str(cfg.output_dir)):
            raise RuntimeError(
                f"[ABLATION GUARD] Found existing checkpoint '{self.resume_path}' inside "
                f"ablation output_dir '{cfg.output_dir}'. Refusing to resume — ablations "
                f"must start fresh. Either:\n"
                f"  (a) delete the stale dir: rm -rf {os.path.join(cfg.output_dir, scene_id)}\n"
                f"  (b) verify your config's output_dir points where you intend\n"
                f"  (c) if you really want to resume, pass --resume_path explicitly"
            )

        # 4. [NEW] Logging Initialization (Scheme A: Reuse existing log_dir if resuming)
        self.log_dir = None
        if self.resume_path and os.path.exists(self.resume_path):
            # Try to determine existing log_dir from resume_path
            # Standard path: .../mpm_train/TIMESTAMP/checkpoint_iter_N.pt
            parent_dir = os.path.dirname(self.resume_path)
            if 'mpm_train' in parent_dir:
                # If resume_path is inside a timestamp folder (not log_base itself)
                log_base = os.path.join(cfg.output_dir, scene_id, 'mpm_train')
                if parent_dir != log_base and os.path.dirname(parent_dir) == log_base:
                    self.log_dir = parent_dir
                    print(f"[RESUME] Reusing existing log directory: {self.log_dir}")
            
            # If still not found (e.g. final_checkpoint.pt in scene root), 
            # try to find the latest folder in mpm_train
            if self.log_dir is None:
                log_base = os.path.join(cfg.output_dir, scene_id, 'mpm_train')
                if os.path.exists(log_base):
                    subdirs = sorted([os.path.join(log_base, d) for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))])
                    if subdirs:
                        self.log_dir = subdirs[-1]
                        print(f"[RESUME] Found latest log directory: {self.log_dir}")

        self.writer = None
        if not for_inference:
            if self.log_dir is None:
                # Fresh start or could not find existing log_dir
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.log_dir = os.path.join(cfg.output_dir, scene_id, 'mpm_train', timestamp)
                print(f"[INFO] Created new log directory: {self.log_dir}")

            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # Warp is the only supported MPM backend; keep this guard so stale
        # configs fail explicitly.
        self.use_warp = getattr(cfg.mpm, 'use_warp', True)
        if not self.use_warp:
            raise ValueError("PyTorch MPM backend has been removed; set mpm.use_warp: true.")
        print("[INFO] Using NVIDIA Warp backend for MPM simulation.")
        self.simulator = WarpMPMSimulator(cfg.mpm).to(self.device)
        
        self.simulator.debug_mode = True # [DEBUG] Enable velocity clamping inspection
        
        # 4. [NEW] Initialize Residual PGND (optional)
        _rcfg = getattr(cfg, 'residual', None)
        _residual_enabled = _rcfg is not None and getattr(_rcfg, 'enabled', True)
        if _residual_enabled:
            self.residual_cfg = _rcfg
            arch = getattr(_rcfg, 'arch', 'pgnd')
            if arch != 'pgnd':
                raise ValueError(
                    f"Unsupported residual arch {arch!r} in the release version; "
                    "only 'pgnd' is included."
                )
            self.residual_net = ResidualPGND(_rcfg).to(self.device)
            print(f"[INFO] Residual model initialized: arch={arch}")
        else:
            self.residual_cfg = None
            self.residual_net = None
            reason = "disabled via enabled=false" if _rcfg is not None else "no residual config"
            print(f"[INFO] ResidualPGND disabled ({reason}).")

        # Actually load the checkpoint weights
        if self.resume_path and os.path.exists(self.resume_path):
            self.load_from_checkpoint(self.resume_path)

        # [STABILITY] Gradient Hook: Sanitize NaN/Inf gradients only.
        # No per-element clamp — that was destroying gradient direction and creating noise.
        # Global grad_clip_norm handles overall magnitude control.
        def nan_to_zero(grad):
            return torch.where(torch.isnan(grad) | torch.isinf(grad), torch.zeros_like(grad), grad)
            
        protected_params = [self.log_weights, self.raw_E, self.raw_nu, self.raw_fiber_k, self.raw_fiber_dir, self.raw_yield, self.raw_viscosity, self.raw_ctrl_stiffness, self.raw_ctrl_damping]
        if self.residual_net is not None:
            protected_params.extend(list(self.residual_net.parameters()))
            
        for p in protected_params:
            p.register_hook(nan_to_zero)
        
        # [Mono] Align cloth plane normal to world +Z (gravity direction).
        # Applies only when cfg.data.mono_align is truthy (mono_cloth config).
        # Rationale: for mono extractions the raw world axes are not gravity-
        # aligned; since the cloth starts flat, its own plane normal is a
        # reliable reference for "up". Rotate init_pos + gt_surface_tracks +
        # controller_points so that the plane normal → -world Z (raw world +Z
        # = gravity, matching multi-cam Z-down convention).
        if getattr(self.cfg.data, 'mono_align', False):
            init_raw = self.data['init_pos'].to(self.device)  # [N, 3]
            centered = init_raw - init_raw.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            n = Vh[-1]  # [3] plane normal
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
            for _k in ['init_pos', 'gt_surface_tracks', 'controller_points']:
                if _k in self.data:
                    self.data[_k] = self.data[_k].to(self.device) @ R.T
            init_pts = self.data['init_pos']
            ip_min = init_pts.min(dim=0)[0]
            ip_max = init_pts.max(dim=0)[0]
            print(f"[mono_align] plane normal={n.tolist()}, rotation det={torch.linalg.det(R).item():.4f}")
            print(f"[mono_align] init_pos bbox after rotate: "
                  f"X[{ip_min[0]:.3f},{ip_max[0]:.3f}] "
                  f"Y[{ip_min[1]:.3f},{ip_max[1]:.3f}] "
                  f"Z[{ip_min[2]:.3f},{ip_max[2]:.3f}], "
                  f"max_span={(ip_max - ip_min).max().item():.3f}")

        # 5. [NEW] Automatic Centering Logic
        # Calculate bounding box across all frames to find the best centering offset.
        # CoTracker marks lost points as (0,0,0) in gt_surface_tracks; including them
        # in the bbox drags p_min/p_max toward the origin and biases the auto_offset.
        # Filter those out before taking min/max.
        obj_pts = self.data['init_pos'].to(self.device) # [N, 3]
        gt_pts = self.data['gt_surface_tracks'].to(self.device).view(-1, 3) # [T*N, 3]
        ctrl_pts = self.data['controller_points'].to(self.device).view(-1, 3) # [T*C, 3]

        gt_valid = torch.linalg.norm(gt_pts, dim=-1) > 1e-5
        ctrl_valid = torch.linalg.norm(ctrl_pts, dim=-1) > 1e-5
        gt_pts = gt_pts[gt_valid]
        ctrl_pts = ctrl_pts[ctrl_valid]

        all_pts = torch.cat([obj_pts, gt_pts, ctrl_pts], dim=0)
        p_min = all_pts.min(dim=0)[0]
        p_max = all_pts.max(dim=0)[0]
        
        # [REVISED] Auto-centering Strategy
        # X and Y: Center the entire sequence to avoid side-wall collisions.
        # Z: Ground only the initial object state instead of the whole sequence.
        #    Using future GT/controller frames for min-z can lift the frame-0 pose
        #    above the floor, which shows up as an initial "free-fall" on softbody
        #    cases whose later frames dip lower than the initial rest pose.
        #    Ground is at boundary point [0,0,0] + shift 0.5 = z_grid=0.5.
        #    We want min_z_grid = 0.5 + margin. Since z_grid = z_orig + offset + 0.5:
        #    offset_z = margin - init_min_z
        floor_margin = getattr(self.cfg.mpm, 'floor_margin', 0.05)
        init_min_z = obj_pts[:, 2].min()
        self.auto_offset = torch.zeros(3, device=self.device)
        self.auto_offset[0] = -(p_min[0] + p_max[0]) / 2.0
        self.auto_offset[1] = -(p_min[1] + p_max[1]) / 2.0
        self.auto_offset[2] = floor_margin - init_min_z
        
        # Sync offset to simulator and refresh ground boundary
        self.simulator.base_offset = self.auto_offset
        self.simulator._apply_boundary()
        
        # Check if the scaled object fits in [0.05, 0.95]
        max_span = (p_max - p_min).max().item()
        
        # Log setup info to TensorBoard instead of printing to terminal
        setup_text = f"**Scene ID**: {scene_id}  \n"
        setup_text += f"**Auto-centering Offset**: {self.auto_offset.tolist()}  \n"
        setup_text += f"**Original BBox Span**: {max_span:.4f}  \n"
        
        # Establishment of controller connections is done during simulator.reset()
        # We need a quick reset to check initial connections
        self.simulator.reset(obj_pts, controller_pos=self.data['controller_points'][0].to(self.device))
        setup_text += f"**Controller Connections**: {self.simulator.num_connections}  \n"

        if self.writer is not None:
            # Save a visualization of the initial controller-particle connections to the
            # case output directory for later inspection.
            try:
                self._save_connection_visualization(obj_pts)
            except Exception as e:
                print(f"[WARN] connection visualization failed: {e}")

            self.writer.add_text('Setup/Info', setup_text, 0)

            if max_span > 0.85:
                self.writer.add_text('Setup/Warnings', f"WARNING: Scene is very large (span: {max_span:.2f}).", 0)

        # [NEW] Smooth Controller Trajectory to filter out tracking noise
        # Using a moving average window from config
        self.controller_points = (self.data['controller_points'].to(self.device) + self.auto_offset)
        T_ctrl = self.controller_points.shape[0]
        window = getattr(self.cfg.mpm, 'controller_smooth_window', 5) # Use YAML value
        
        if window > 1:
            smoothed = self.controller_points.clone()
            for t in range(T_ctrl):
                t_start = max(0, t - window // 2)
                t_end = min(T_ctrl, t + window // 2 + 1)
                smoothed[t] = self.controller_points[t_start:t_end].mean(dim=0)
            self.controller_points = smoothed
            print(f"[INFO] Controller trajectory smoothed (window={window})")

        # 6. [NEW] Deterministic Patch Assignment
        # We assign each particle to a patch once at the start. 
        # This MUST be persistent to ensure learned parameters map to the same particles.
        from pytorch3d.ops import sample_farthest_points, knn_points
        xyz_static = self.data['gaussians'][:, :3].unsqueeze(0).to(self.device)
        self.patch_centers, _ = sample_farthest_points(xyz_static, K=self.n_patches)
        
        # Calculate KNN interpolation weights for all particles
        init_pos_centered = (self.data['init_pos'].to(self.device) + self.auto_offset).unsqueeze(0)
        dist, self.patch_idx, _ = knn_points(init_pos_centered, self.patch_centers, K=3)
        dist = torch.clamp(dist, min=1e-6)
        inv_dist = 1.0 / dist
        norm = torch.sum(inv_dist, dim=2, keepdim=True)
        self.interp_weights = (inv_dist / norm).unsqueeze(-1)

        # Initialize fiber directions from local PCA (tangent estimation)
        if 'fi' in self.active_experts:
            self._init_fiber_dir_from_geometry(init_pos_centered.squeeze(0))

        if self.residual_net is not None:
            topo_k = int(getattr(self.residual_cfg, 'topology_k', 0))
            topo_edge_w = float(getattr(self.residual_cfg, 'lambda_topology_edge', 0.0))
            topo_smooth_w = float(getattr(self.residual_cfg, 'lambda_topology_smooth', 0.0))
            if topo_k > 0 and (topo_edge_w > 0.0 or topo_smooth_w > 0.0):
                topo_dist, topo_idx, _ = knn_points(init_pos_centered, init_pos_centered, K=topo_k + 1)
                self.residual_topology_idx = topo_idx.squeeze(0)[:, 1:]
                self.residual_topology_rest = torch.sqrt(torch.clamp(topo_dist.squeeze(0)[:, 1:], min=1e-8))
                topo_radius = float(getattr(self.residual_cfg, 'topology_radius', 0.0))
                if topo_radius > 0.0:
                    self.residual_topology_mask = self.residual_topology_rest <= topo_radius
                else:
                    self.residual_topology_mask = torch.ones_like(self.residual_topology_rest, dtype=torch.bool)
                print(f"[INFO] Residual topology graph initialized (k={topo_k}).")
        
        # 7. Optimizer with Parameter-level Learning Rate Ranges
        lr_ranges = self.cfg.mpm.get('lr_ranges', {})
        base_lr = self.cfg.get('train', {}).get('lr_params', 1e-3)
        
        param_list = [
            {'params': [self.log_weights], 'lr': lr_ranges.get('log_weights', [1e-3])[0]},
            {'params': [self.raw_E], 'lr': lr_ranges.get('E', [base_lr])[0]},
            {'params': [self.raw_nu], 'lr': lr_ranges.get('nu', [base_lr])[0]},
            {'params': [self.raw_fiber_k], 'lr': lr_ranges.get('fiber_k', [base_lr])[0]},
            {'params': [self.raw_fiber_dir], 'lr': lr_ranges.get('fiber_dir', [base_lr])[0]},
            {'params': [self.raw_yield], 'lr': lr_ranges.get('yield_stress', [base_lr])[0]},
            {'params': [self.raw_viscosity], 'lr': lr_ranges.get('plastic_viscosity', [base_lr])[0]},
            {'params': [self.raw_ctrl_stiffness], 'lr': lr_ranges.get('controller_stiffness', [base_lr])[0]},
            {'params': [self.raw_ctrl_damping], 'lr': lr_ranges.get('controller_damping', [base_lr])[0]},
        ]
        
        # [NEW] Add Residual PGND to optimizer
        if self.residual_net is not None:
            residual_lr = getattr(self.residual_cfg, 'lr', 1e-4)
            param_list.append({'params': self.residual_net.parameters(), 'lr': residual_lr})
            print(f"[INFO] Added ResidualPGND to optimizer with LR: {residual_lr}")
            
        self.optimizer = optim.Adam(param_list)
        
        # [NEW] Cosine Annealing Scheduler over num_iters
        # We'll initialize it in train() where num_iters is known, or default here
        self.scheduler = None 
        
        # [NEW] Session Iterations Counter for robust Early Stopping
        self.session_iters = 0
        
        # [NEW] Load optimizer/scheduler state if resuming from PT
        if hasattr(self, 'resume_checkpoint') and self.resume_checkpoint is not None:
            # We can't load scheduler yet as it's not inited, but we can load optimizer
            if 'optimizer_state_dict' in self.resume_checkpoint:
                try:
                    self.optimizer.load_state_dict(self.resume_checkpoint['optimizer_state_dict'])
                    print("[RESUME] Optimizer state loaded.")
                except Exception as e:
                    print(f"[RESUME] Warning: Failed to load optimizer state: {e}")

    def _save_connection_visualization(self, obj_pts):
        """Save a 2D top-down PNG of the controller-particle connections under
        the case output directory. Mirrors check_all_connections.py:
          - blue dots: object points
          - red X: active controllers (after mask)
          - gray X: dropped controllers (mask=False)
          - red lines: KNN connections within controller_radius
        """
        from pytorch3d.ops import knn_points

        out_path = os.path.join(self.cfg.output_dir, self.scene_id, "controller_connections.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Active (post-mask) controllers (frame 0)
        cp = self.data['controller_points']
        if isinstance(cp, torch.Tensor):
            active_ctrl = cp[0].detach().cpu().float()
        else:
            active_ctrl = torch.from_numpy(cp[0]).float()

        # Raw (pre-mask) controllers + mask, if available
        raw_ctrl = self._raw_ctrl_frame0
        if isinstance(raw_ctrl, torch.Tensor):
            raw_ctrl = raw_ctrl.detach().cpu().float()
        elif raw_ctrl is not None:
            raw_ctrl = torch.from_numpy(raw_ctrl).float()
        ctrl_mask = self._controller_mask_used  # numpy bool array or None

        if isinstance(obj_pts, torch.Tensor):
            obj_cpu = obj_pts.detach().cpu().float()
        else:
            obj_cpu = torch.from_numpy(obj_pts).float()

        K = int(getattr(self.cfg.mpm, 'controller_max_neighbors', 64))
        radius = float(getattr(self.cfg.mpm, 'controller_radius', 0.13))

        # KNN from active controllers to object points
        dist, idx, _ = knn_points(active_ctrl.unsqueeze(0), obj_cpu.unsqueeze(0), K=K)
        dist = dist.squeeze(0)
        idx = idx.squeeze(0)
        mask = dist.sqrt() < radius
        n_connections = int(mask.sum().item())

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(obj_cpu[:, 0], obj_cpu[:, 1], s=2, c='blue', alpha=0.3, label='Object')

        # Plot dropped (gray) controllers if a mask was used
        n_total_ctrl = active_ctrl.shape[0]
        if raw_ctrl is not None and ctrl_mask is not None:
            dropped = raw_ctrl[~ctrl_mask]
            if len(dropped) > 0:
                ax.scatter(dropped[:, 0], dropped[:, 1], s=50, c='gray',
                           marker='x', label='Controller (dropped)', alpha=0.4, zorder=9)
            n_total_ctrl = raw_ctrl.shape[0]

        ax.scatter(active_ctrl[:, 0], active_ctrl[:, 1], s=50, c='red',
                   marker='x', label='Controller (active)', zorder=10)

        for c_idx in range(active_ctrl.shape[0]):
            connected_p = idx[c_idx][mask[c_idx]]
            if len(connected_p) > 0:
                for p in obj_cpu[connected_p]:
                    ax.plot([active_ctrl[c_idx, 0], p[0]],
                            [active_ctrl[c_idx, 1], p[1]],
                            color='red', alpha=0.15, linewidth=1.0)

        mask_state = "loaded" if ctrl_mask is not None else "no_mask"
        ax.set_title(
            f"Case: {self.scene_id}\n"
            f"Radius: {radius}, K: {K}, mask: {mask_state}\n"
            f"Active controllers: {active_ctrl.shape[0]}/{n_total_ctrl}   Connections: {n_connections}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Connection visualization saved to {out_path}")

    def _compute_residual_topology_regularizers(self, x_phys, x_corr):
        if self.residual_topology_idx is None or self.residual_topology_mask is None:
            zero = x_phys.new_zeros(())
            return zero, zero

        nbr_idx = self.residual_topology_idx
        mask = self.residual_topology_mask
        rest_len = self.residual_topology_rest
        if not mask.any():
            zero = x_phys.new_zeros(())
            return zero, zero

        x_phys_nbr = x_phys[nbr_idx]
        x_corr_nbr = x_corr[nbr_idx]
        delta_x = x_corr - x_phys
        delta_x_nbr = delta_x[nbr_idx]

        denom = rest_len.unsqueeze(-1) + 1e-6
        smooth_term = ((delta_x.unsqueeze(1) - delta_x_nbr) / denom).pow(2).sum(dim=-1)

        phys_edge_len = torch.norm(x_phys.unsqueeze(1) - x_phys_nbr, dim=-1)
        corr_edge_len = torch.norm(x_corr.unsqueeze(1) - x_corr_nbr, dim=-1)
        edge_term = ((corr_edge_len - phys_edge_len) / (rest_len + 1e-6)).pow(2)

        smooth_term = smooth_term[mask].mean()
        edge_term = edge_term[mask].mean()

        edge_reg = edge_term * float(getattr(self.residual_cfg, 'lambda_topology_edge', 0.0))
        smooth_reg = smooth_term * float(getattr(self.residual_cfg, 'lambda_topology_smooth', 0.0))
        return edge_reg, smooth_reg

    def _init_fiber_dir_from_geometry(self, particle_pos):
        """Initialize fiber directions via local PCA around each patch center.

        For rope-like geometry, the first principal component is the local
        tangent; for cloth it is the dominant in-plane direction.  Both are
        better starting points than random vectors.

        Args:
            particle_pos: [N, 3] particle positions on ``self.device``.
        """
        from pytorch3d.ops import knn_points

        pca_k = min(16, particle_pos.shape[0])
        # [1, K_patches, 3] patch centers already computed
        _, nbr_idx, _ = knn_points(
            self.patch_centers, particle_pos.unsqueeze(0), K=pca_k
        )  # nbr_idx: [1, n_patches, pca_k]

        nbr_idx = nbr_idx.squeeze(0)  # [n_patches, pca_k]
        nbr_pos = particle_pos[nbr_idx]  # [n_patches, pca_k, 3]
        centroid = nbr_pos.mean(dim=1, keepdim=True)  # [n_patches, 1, 3]
        centered = nbr_pos - centroid  # [n_patches, pca_k, 3]

        # PCA via SVD: first right-singular vector = principal direction
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)  # Vh: [n_patches, 3, 3]
        tangent = Vh[:, 0, :]  # [n_patches, 3]

        with torch.no_grad():
            self.raw_fiber_dir.copy_(tangent)
        print(f"[INFO] Fiber directions initialized from local PCA (k={pca_k})")

    def get_current_phys_props(self):
        weights = torch.softmax(self.log_weights, dim=-1)
        
        # [STABILITY] Map to a reasonable range from config
        bounds = self.cfg.mpm.get('material_bounds', {})
        
        # [FIXED] Friction is now a constant scalar
        friction = torch.tensor(self.cfg.mpm.boundary.friction, device=self.device)
        
        # [NEW] E, nu -> mu, lam conversion
        E_min, E_max = bounds.get('E', [1e3, 1e6])
        nu_min, nu_max = bounds.get('nu', [0.0, 0.45])
        fk_min, fk_max = bounds.get('fiber_k', [5e3, 5e5])
        ys_min, ys_max = bounds.get('yield_stress', [1e2, 1e5])
        visc_min, visc_max = bounds.get('plastic_viscosity', [0.1, 10.0])

        # [REVISED] Log-space mapping for better optimization stability
        # Map raw parameter (unbounded) to [min, max] using Log-Sigmoid
        # value = exp( log(min) + sigmoid(raw) * (log(max) - log(min)) )
        
        def map_log_sigmoid(raw_param, min_val, max_val):
            log_min = np.log(max(min_val, 1e-6))
            log_max = np.log(max_val)
            log_val = log_min + torch.sigmoid(raw_param) * (log_max - log_min)
            return torch.exp(log_val)
        
        p_E = map_log_sigmoid(self.raw_E, E_min, E_max)
        # Nu is usually small [0, 0.5], keep linear sigmoid for it
        p_nu = nu_min + torch.sigmoid(self.raw_nu) * (nu_max - nu_min)
        
        # E, nu -> mu, lam conversion
        p_mu, p_lam = youngs_poisson_to_lame(p_E, p_nu)
        
        p_fiber_k = map_log_sigmoid(self.raw_fiber_k, fk_min, fk_max)
        p_fiber_dir = self.raw_fiber_dir
        
        # [NEW] Yield Stress & Viscosity for Plasticity
        p_yield = map_log_sigmoid(self.raw_yield, ys_min, ys_max)
        p_visc = map_log_sigmoid(self.raw_viscosity, visc_min, visc_max)

        # Controller gains (scalar)
        cs_min, cs_max = bounds.get('controller_stiffness', [1e3, 1e6])
        cd_min, cd_max = bounds.get('controller_damping', [1e2, 5e4])
        p_ctrl_stiffness = map_log_sigmoid(self.raw_ctrl_stiffness, cs_min, cs_max)
        p_ctrl_damping = map_log_sigmoid(self.raw_ctrl_damping, cd_min, cd_max)
        
        return (weights, p_mu.squeeze(), p_lam.squeeze(), p_fiber_k.squeeze(), p_fiber_dir,
                friction, p_yield.squeeze(), p_E.squeeze(), p_nu.squeeze(), p_visc.squeeze(),
                p_ctrl_stiffness.squeeze(), p_ctrl_damping.squeeze())

    def on_train_iteration_start(self, iter_idx: int, total_frames: int):
        return None

    def compute_optional_render_loss(self, iter_idx: int, frame_idx: int, x_curr: torch.Tensor):
        return torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def _run_forward_loss(self, init_pos, gt_tracks, T, num_supervised):
        """Run full forward simulation and return total loss (matching training loop)."""
        self.simulator.reset(init_pos, controller_pos=self.controller_points[0])
        
        w_patch, mu_patch, lam_patch, fk_patch, fdir_patch, friction, yield_patch, \
            E_patch, nu_patch, visc_patch, ctrl_stiffness_t, ctrl_damping_t = self.get_current_phys_props()
        
        def gather_and_interp(patch_data):
            flat_idx = self.patch_idx.squeeze(0).view(-1)
            gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
            return torch.sum(self.interp_weights * gathered, dim=2).squeeze(0)
        
        active_mask_wp = wp.array(
            [1 if e in self.active_experts else 0 for e in ['nh', 'co', 'st', 'fi']],
            dtype=wp.int32, device=self.simulator.warp_device)
        moe_params_wp = {
            'weights': torch2warp_float(gather_and_interp(w_patch)),
            'mu': torch2warp_float(gather_and_interp(mu_patch.unsqueeze(-1)).squeeze()),
            'lam': torch2warp_float(gather_and_interp(lam_patch.unsqueeze(-1)).squeeze()),
            'fk': torch2warp_float(gather_and_interp(fk_patch.unsqueeze(-1)).squeeze()),
            'fdir': torch2warp_vec3(torch.nn.functional.normalize(gather_and_interp(fdir_patch), dim=1, eps=1e-8)),
            'active_mask': active_mask_wp
        }
        
        warmup_frames = getattr(self.cfg.mpm, 'controller_warmup_frames', 3)
        total_loss = 0.0
        
        for t in range(T):
            c_pos_end = self.controller_points[t]
            c_pos_start = self.controller_points[t - 1] if t > 0 else c_pos_end
            v_ctrl_t = (c_pos_end - c_pos_start) / (self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame)
            
            warmup_factor = min(1.0, (t + 1) / (warmup_frames + 1e-6))
            current_stiffness = ctrl_stiffness_t * warmup_factor
            current_damping = ctrl_damping_t * warmup_factor
            kp_wp = wp.from_torch(current_stiffness.reshape(1).contiguous().float(),
                                  dtype=wp.float32, requires_grad=False)
            kd_wp = wp.from_torch(current_damping.reshape(1).contiguous().float(),
                                  dtype=wp.float32, requires_grad=False)
            
            for s in range(self.cfg.mpm.steps_per_frame):
                alpha = (s + 1) / self.cfg.mpm.steps_per_frame
                curr_target_pos = c_pos_start + alpha * (c_pos_end - c_pos_start)
                self.simulator.step(moe_params_wp,
                                    controller_pos=curr_target_pos,
                                    controller_vel=v_ctrl_t,
                                    residual_v=None,
                                    kp_wp=kp_wp, kd_wp=kd_wp)
            
            x_curr = (self.simulator.x - self.simulator.shift)[:num_supervised]
            x_gt = gt_tracks[t]
            gt_mask = torch.norm(self.data['gt_surface_tracks'][t], dim=-1) > 1e-5
            if gt_mask.any():
                x_curr_m = x_curr[gt_mask]
                x_gt_m = x_gt[gt_mask]
                track_loss = torch.mean((x_curr_m - x_gt_m) ** 2)
                cham_loss, _ = chamfer_distance(x_curr_m.unsqueeze(0), x_gt_m.unsqueeze(0))
                total_loss += (track_loss.item() + cham_loss.item()) / T
        
        return total_loss

    def train(self, num_iters=50):
        print(f"\nStarting MPM Training (System ID) for scene: {self.scene_id}")

        # [NEW] Setup Scheduler now that we know num_iters
        lr_ranges = self.cfg.mpm.get('lr_ranges', {})
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_iters, eta_min=1e-4)
        
        # [NEW] Resume Scheduler State if available
        if hasattr(self, 'resume_checkpoint') and self.resume_checkpoint is not None:
            if 'scheduler_state_dict' in self.resume_checkpoint and self.resume_checkpoint['scheduler_state_dict']:
                try:
                    self.scheduler.load_state_dict(self.resume_checkpoint['scheduler_state_dict'])
                    print("[RESUME] Scheduler state loaded.")
                except Exception as e:
                    print(f"[RESUME] Warning: Failed to load scheduler state: {e}")
            
            # [NEW] Determine start iter
            start_iter = self.resume_checkpoint.get('iter', 0)
            print(f"[RESUME] Resuming from Iter {start_iter}")
        else:
            start_iter = 0
        offset = self.auto_offset
        
        init_pos = (self.data['init_pos'].to(self.device) + offset)
        gt_tracks = (self.data['gt_surface_tracks'].to(self.device) + offset)
        num_supervised = self.data['num_supervised']
        
        pc = self.data.get('particle_counts', {})
        if pc:
            self.writer.add_text('ParticleCounts', 
                f"surface={pc['surface']}, other_surface={pc['other_surface']}, "
                f"interior={pc['interior']}, gaussian={pc['gaussian_filled']}, "
                f"**total={pc['total']}**", 0)
            for k, v in pc.items():
                self.writer.add_scalar(f'Particles/{k}', v, 0)
            tqdm.write(f"[INFO] Particles: surface={pc['surface']}, other_surface={pc['other_surface']}, "
                       f"interior={pc['interior']}, gaussian={pc['gaussian_filled']}, total={pc['total']}")

        # [Rope Length Constraint] Build topology and rest lengths
        self._rope_length_edges = None
        lambda_length = float(getattr(self.cfg.train, 'lambda_length', 0.0))
        if lambda_length > 0:
            from pytorch3d.ops import knn_points
            rope_k = int(getattr(self.cfg.train, 'length_k', 6))
            _, knn_idx, _ = knn_points(init_pos.unsqueeze(0), init_pos.unsqueeze(0), K=rope_k + 1)
            knn_idx = knn_idx.squeeze(0)[:, 1:]  # [N, K] exclude self
            N = init_pos.shape[0]
            src = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, rope_k).reshape(-1)
            dst = knn_idx.reshape(-1)
            mask = src < dst  # deduplicate
            self._rope_length_src = src[mask]
            self._rope_length_dst = dst[mask]
            diff = init_pos[self._rope_length_src] - init_pos[self._rope_length_dst]
            self._rope_rest_lengths = torch.norm(diff, dim=-1)
            self._rope_length_edges = True
            tqdm.write(f"[INFO] Rope length constraint: {self._rope_length_src.shape[0]} edges, K={rope_k}")

        # [NEW] Initialize History Buffers for ResidualPGND
        H = getattr(self.residual_cfg, 'n_history', 2) if self.residual_cfg else 2
        x_history = []
        v_history = []

        T_data = gt_tracks.shape[0]
        T = min(T_data, self.cfg.mpm.max_frames) if self.cfg.mpm.max_frames > 0 else T_data
        
        sim_pos = init_pos.unsqueeze(0)

        # Add an outer progress bar.
        # [RESUME] Start from correct iter
        last_iter = start_iter
        print("") # [FIX] Ensure first iteration output is clean
        main_pbar = tqdm(range(start_iter, num_iters), desc=f"[{self.scene_id}] Optimization Progress")

        for i in main_pbar:
            last_iter = i + 1
            self.optimizer.zero_grad()
            self._accumulated_grad_x = None
            self.simulator.reset(init_pos, controller_pos=self.controller_points[0])
            
            # Warp Tape: record last frame's final substeps for gradient flow.
            if self.use_warp:
                if not hasattr(self, '_warp_tape'):
                    self._warp_tape = wp.Tape()
                self._warp_tape.reset()
                self.simulator.tape = None

            # [NEW] Reset history with current initial state
            x_history = [init_pos.clone() for _ in range(H)]
            v_history = [torch.zeros_like(init_pos) for _ in range(H)]
            
            T_data = gt_tracks.shape[0]
            T = min(T_data, self.cfg.mpm.max_frames) if self.cfg.mpm.max_frames > 0 else T_data
            
            # Staged Optimization: Handle expert weight warmup
            weight_warmup = getattr(self.cfg.mpm, 'weight_warmup_iters', 0)

            # Elastic vs Plastic Phases
            total_iters = num_iters
            phase_split = total_iters // 2

            is_elastic_phase = (i < phase_split)
            is_plastic_phase = (i >= phase_split)

            # Dynamic Requires Grad Control
            # [rgb_finetune] When RGB finetune mode is active, freeze all physical
            # parameters: only residual_net and external GS params (handled by
            # the subclass) are updated. See RGBFinetuneTrainer.
            rgb_finetune_mode = getattr(self, 'rgb_finetune_mode', False)
            self.log_weights.requires_grad = (not rgb_finetune_mode) and (i >= weight_warmup)
            self.raw_E.requires_grad = not rgb_finetune_mode
            self.raw_nu.requires_grad = not rgb_finetune_mode
            self.raw_fiber_dir.requires_grad = False
            self.raw_fiber_k.requires_grad = False
            self.raw_yield.requires_grad = (not rgb_finetune_mode) and is_plastic_phase
            self.raw_viscosity.requires_grad = (not rgb_finetune_mode) and is_plastic_phase

            if self.residual_net is not None:
                res_warmup = getattr(self.residual_cfg, 'warmup_iters', 5)
                # In rgb_finetune mode residual_net is always unfrozen (no warmup gate)
                self.residual_net.requires_grad = rgb_finetune_mode or (i >= res_warmup)

            # [rgb_finetune] also freeze controller gains kp/kd
            if rgb_finetune_mode:
                self.raw_ctrl_stiffness.requires_grad = False
                self.raw_ctrl_damping.requires_grad = False
                self.residual_net.train()
                use_residual = (i >= res_warmup)
            else:
                use_residual = False

            res_stats = {
                'mean_mag': [], 'max_mag': [], 'ratio_to_phys': [], 'cos_sim': [],
                'topology_edge': [], 'topology_smooth': []
            }
            loss_stats = {
                'track': [], 'chamfer': [], 'render': [], 'residual_reg': [], 'length': []
            }

            total_loss = 0.0
            self.on_train_iteration_start(i, T)
            
            # [FIX] Better progress bar formatting to avoid clashing with Warp/Taichi output
            frame_pbar = tqdm(range(T), 
                              desc=f"  [{self.scene_id}] Iter {i+1}", 
                              leave=False, 
                              bar_format='{l_bar}{bar:20}{r_bar}')
            
            def gather_and_interp(patch_data):
                flat_idx = self.patch_idx.squeeze(0).view(-1)
                gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
                return torch.sum(self.interp_weights * gathered, dim=2).squeeze(0)

            # [PhysFlow-style] Compute MoE params ONCE per iteration (outside frame loop)
            # and record the setup on the tape for gradient flow.
            w_patch, mu_patch, lam_patch, fk_patch, fdir_patch, friction, yield_patch, E_patch, nu_patch, visc_patch, ctrl_stiffness_t, ctrl_damping_t = self.get_current_phys_props()
            
            p_weights = gather_and_interp(w_patch)
            p_mu = gather_and_interp(mu_patch.unsqueeze(-1)).squeeze()
            p_lam = gather_and_interp(lam_patch.unsqueeze(-1)).squeeze()
            p_fk = gather_and_interp(fk_patch.unsqueeze(-1)).squeeze()
            p_fdir = torch.nn.functional.normalize(gather_and_interp(fdir_patch), dim=1, eps=1e-8)
            p_yield = gather_and_interp(yield_patch.unsqueeze(-1)).squeeze()
            p_visc = gather_and_interp(visc_patch.unsqueeze(-1)).squeeze()

            if self.use_warp:
                if p_weights.requires_grad: p_weights.retain_grad()
                if p_mu.requires_grad: p_mu.retain_grad()
                if p_lam.requires_grad: p_lam.retain_grad()
                if p_fk.requires_grad: p_fk.retain_grad()
                if p_fdir.requires_grad: p_fdir.retain_grad()

            active_experts_list = getattr(self.cfg.mpm, 'active_experts', ['nh', 'co', 'st', 'fi'])
            expert_order = ['nh', 'co', 'st', 'fi']
            mask_active = [1 if e in active_experts_list else 0 for e in expert_order]
            active_mask_wp = wp.array(mask_active, dtype=wp.int32, device=self.simulator.warp_device)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                moe_params_wp = {
                    'weights': torch2warp_float(p_weights, requires_grad=True),
                    'mu': torch2warp_float(p_mu, requires_grad=True),
                    'lam': torch2warp_float(p_lam, requires_grad=True),
                    'fk': torch2warp_float(p_fk, requires_grad=True),
                    'fdir': torch2warp_vec3(p_fdir, requires_grad=True),
                    'active_mask': active_mask_wp
                }
            
            self.simulator.last_moe_params = moe_params_wp
            self.simulator.last_expert_inputs = {
                'weights': p_weights, 'mu': p_mu, 'lam': p_lam, 'fk': p_fk, 'fdir': p_fdir
            }

            for t in frame_pbar:
                c_pos_end = self.controller_points[t]
                c_pos_start = self.controller_points[t-1] if t > 0 else c_pos_end
                
                v_ctrl_t = (c_pos_end - c_pos_start) / (self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame)

                warmup_frames = getattr(self.cfg.mpm, 'controller_warmup_frames', 10)
                warmup_factor = min(1.0, (t + 1) / (warmup_frames + 1e-6))
                current_stiffness = ctrl_stiffness_t * warmup_factor
                current_damping = ctrl_damping_t * warmup_factor

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kp_wp = wp.from_torch(current_stiffness.reshape(1).contiguous().float(),
                                          dtype=wp.float32, requires_grad=True)
                    kd_wp = wp.from_torch(current_damping.reshape(1).contiguous().float(),
                                          dtype=wp.float32, requires_grad=True)

                x_start_frame = (self.simulator.x - self.simulator.shift).detach().unsqueeze(0)

                tape_window = min(getattr(self.cfg.mpm, 'tape_window', 20), self.cfg.mpm.steps_per_frame)
                tape_start = self.cfg.mpm.steps_per_frame - tape_window
                is_last_frame = (t == T - 1)
                if is_last_frame:
                    self._last_kp_wp = kp_wp
                    self._last_kd_wp = kd_wp
                    self._last_stiffness_t = current_stiffness
                    self._last_damping_t = current_damping

                for s in range(self.cfg.mpm.steps_per_frame):
                    alpha = (s + 1) / self.cfg.mpm.steps_per_frame
                    curr_target_pos = c_pos_start + alpha * (c_pos_end - c_pos_start)
                    
                    if is_last_frame and s >= tape_start:
                        self.simulator.tape = self._warp_tape if self.use_warp else None
                    else:
                        self.simulator.tape = None

                    x_curr = self.simulator.step(moe_params_wp, 
                                                 controller_pos=curr_target_pos, 
                                                 controller_vel=v_ctrl_t,
                                                 residual_v=None,
                                                 kp_wp=kp_wp, kd_wp=kd_wp)
                
                # [SCHEME A] Phase 2: Neural Feedback Correction
                delta_v = None
                leaf_x = self.simulator.x
                leaf_v = self.simulator.v

                if use_residual:
                    x_his_tensor = torch.stack(x_history, dim=1).unsqueeze(0)
                    v_his_tensor = torch.stack(v_history, dim=1).unsqueeze(0)
                    
                    curr_x_mpm = (self.simulator.x - self.simulator.shift).unsqueeze(0)
                    curr_v_mpm = self.simulator.v.unsqueeze(0)
                    
                    delta_v = self.residual_net(curr_x_mpm, curr_v_mpm, x_start_frame, x_his_tensor, v_his_tensor).squeeze(0)
                    
                    # [NEW] Calculate monitoring metrics before applying
                    with torch.no_grad():
                        v_mpm_mag = torch.norm(curr_v_mpm.squeeze(0), dim=-1)
                        dv_mag = torch.norm(delta_v, dim=-1)
                        
                        res_stats['mean_mag'].append(dv_mag.mean().item())
                        res_stats['max_mag'].append(dv_mag.max().item())
                        
                        # Ratio of correction to physics magnitude
                        ratio = dv_mag / (v_mpm_mag + 1e-6)
                        res_stats['ratio_to_phys'].append(ratio.mean().item())
                        
                        # Cosine similarity (directional alignment)
                        cos_sim = torch.sum(delta_v * curr_v_mpm.squeeze(0), dim=-1) / (dv_mag * v_mpm_mag + 1e-6)
                        res_stats['cos_sim'].append(cos_sim.mean().item())

                    # Apply correction to Simulator State
                    # 1. Correct Velocity
                    self.simulator.v = self.simulator.v + delta_v
                    
                    # 2. Correct Position: x = x + delta_v * (dt_frame)
                    frame_dt = self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame
                    self.simulator.x = self.simulator.x + delta_v * frame_dt
                    
                    # Final x_curr for loss calculation (after correction)
                    x_curr = self.simulator.x - self.simulator.shift
                    topo_edge_reg, topo_smooth_reg = self._compute_residual_topology_regularizers(
                        curr_x_mpm.squeeze(0), x_curr
                    )
                    with torch.no_grad():
                        res_stats['topology_edge'].append(topo_edge_reg.item())
                        res_stats['topology_smooth'].append(topo_smooth_reg.item())
                else:
                    topo_edge_reg = torch.tensor(0.0, device=self.device)
                    topo_smooth_reg = torch.tensor(0.0, device=self.device)

                # Update History Buffer (FIFO)
                if self.residual_net is not None:
                    x_history.pop(0)
                    x_history.append((self.simulator.x - self.simulator.shift).detach())
                    v_history.pop(0)
                    v_history.append(self.simulator.v.detach())
                
                # Track loss uses only the first num_supervised particles (surface with GT).
                # init_pos order: [surface, other_surf, interior, (optional) dense, new_internal],
                # so appended Gaussian particles do not affect loss indices.
                x_curr_surf = x_curr[:num_supervised]
                x_gt = gt_tracks[t]
                
                # [FIXED] Mask out zero-artifacts using RAW GT tracks (before offset)
                # to ensure we catch points at (0,0,0) regardless of auto-centering.
                gt_mask = torch.norm(self.data['gt_surface_tracks'][t], dim=-1) > 1e-5 # [N_surf]
                
                if gt_mask.any():
                    x_curr_masked = x_curr_surf[gt_mask]
                    x_gt_masked = x_gt[gt_mask]
                    
                    track_loss = torch.mean((x_curr_masked - x_gt_masked)**2)
                    cham_loss, _ = chamfer_distance(x_curr_masked.unsqueeze(0), x_gt_masked.unsqueeze(0))
                    
                    # [NEW] Residual Regularization: encourage the network to only make small corrections
                    res_reg = 0.0
                    if delta_v is not None:
                        res_reg = torch.mean(delta_v**2) * getattr(self.residual_cfg, 'lambda_reg', 0.01)
                    render_loss = self.compute_optional_render_loss(i, t, x_curr)
                    with torch.no_grad():
                        loss_stats['track'].append(track_loss.item())
                        loss_stats['chamfer'].append(cham_loss.item())
                        loss_stats['render'].append(render_loss.item())
                        loss_stats['residual_reg'].append(
                            res_reg.item() if isinstance(res_reg, torch.Tensor) else float(res_reg)
                        )
                    # Rope length preservation loss
                    # (placed after no_grad block so length_loss itself remains differentiable)
                    length_loss = torch.tensor(0.0, device=self.device)
                    if self._rope_length_edges is not None:
                        curr_diff = x_curr[self._rope_length_src] - x_curr[self._rope_length_dst]
                        curr_lengths = torch.norm(curr_diff, dim=-1)
                        length_loss = torch.mean((curr_lengths - self._rope_rest_lengths) ** 2)
                    with torch.no_grad():
                        loss_stats['length'].append(length_loss.item())

                    lambda_track = float(getattr(self.cfg.train, 'lambda_track', 1.0))
                    lambda_chamfer = float(getattr(self.cfg.train, 'lambda_chamfer', 1.0))
                    frame_loss = (
                        track_loss * lambda_track
                        + cham_loss * lambda_chamfer
                        + length_loss * lambda_length
                        + res_reg
                        + topo_edge_reg
                        + topo_smooth_reg
                        + render_loss
                    ) / T
                else:
                    # Fallback if the whole frame is empty (should not happen)
                    frame_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # Stop immediately if NaN/Inf appears.
                if not torch.isfinite(frame_loss):
                    tqdm.write(f"[ERROR] Non-finite loss ({frame_loss.item()}) detected in Frame {t}! Stopping.")
                    total_loss = float('nan')
                    break

                # PyTorch backward for ResidualPGND and tracking loss gradients
                frame_loss.backward(retain_graph=False)
                
                # Accumulate position gradients for Warp tape backward
                if self.use_warp and leaf_x.grad is not None:
                    if self._accumulated_grad_x is None:
                        self._accumulated_grad_x = leaf_x.grad.detach().clone()
                    else:
                        self._accumulated_grad_x += leaf_x.grad.detach()
                
                if i % 1 == 0:
                    self.writer.add_scalar(f'Frame_Loss_Detail/Iter_{i:03d}', frame_loss.item() * T, t)

                if i % 1 == 0 and t % 10 == 0:
                    with torch.no_grad():
                        v_sim = x_curr.detach()
                        v_gt = x_gt.detach()
                        gt_mask_viz = torch.norm(self.data['gt_surface_tracks'][t], dim=-1).to(self.device) > 1e-5
                        v_gt = v_gt[gt_mask_viz]
                        v_ctrl = curr_target_pos.detach()
                        all_v = torch.cat([v_sim, v_gt, v_ctrl], dim=0).unsqueeze(0).cpu()
                        c_sim = torch.tensor([[0, 0, 255]], device='cpu').repeat(v_sim.shape[0], 1)
                        c_gt = torch.tensor([[0, 255, 0]], device='cpu').repeat(v_gt.shape[0], 1)
                        c_ctrl = torch.tensor([[255, 0, 0]], device='cpu').repeat(v_ctrl.shape[0], 1)
                        all_c = torch.cat([c_sim, c_gt, c_ctrl], dim=0).unsqueeze(0)
                        self.writer.add_mesh(f'{self.scene_id}_Mesh/Iter_{i:03d}', vertices=all_v, colors=all_c, global_step=t)

                self.simulator.x = self.simulator.x.detach().requires_grad_()
                self.simulator.v = self.simulator.v.detach().requires_grad_()
                self.simulator.F = self.simulator.F.detach().requires_grad_()
                self.simulator.C = self.simulator.C.detach().requires_grad_()
                
                total_loss += frame_loss.item()
                
                frame_pbar.set_postfix({
                    'f_loss': f"{frame_loss.item() * T:.6f}",
                    'E': f"{E_patch.mean().item():.1e}",
                    'nu': f"{nu_patch.mean().item():.3f}"
                })

            # ======== Warp Tape Backward (ONCE per iteration) ========
            if self.use_warp and self._accumulated_grad_x is not None:
                from ..model.diff_simulator.warp_solver.mpm_utils import sum_vec3
                
                n_particles = self.simulator.solver.n_particles
                device_wp = self.simulator.warp_device
                
                grad_x_wp = torch2warp_vec3(self._accumulated_grad_x.contiguous())
                loss_wp = wp.zeros(1, dtype=float, device=device_wp, requires_grad=True)
                
                with self._warp_tape:
                    wp.launch(sum_vec3, n_particles, 
                              [self.simulator.solver.mpm_state.particle_x, grad_x_wp],
                              [loss_wp], device=device_wp)
                
                self._warp_tape.backward(loss=loss_wp)
                
                # Extract MoE parameter gradients → PyTorch
                if hasattr(self.simulator, 'last_moe_params'):
                    for key in ['weights', 'mu', 'lam', 'fk', 'fdir']:
                        wp_arr = self.simulator.last_moe_params.get(key)
                        torch_tensor = self.simulator.last_expert_inputs.get(key)
                        if wp_arr is None or torch_tensor is None or not torch_tensor.requires_grad:
                            continue
                        grad_wp = self._warp_tape.gradients.get(wp_arr)
                        if grad_wp is None and wp_arr.grad is not None:
                            grad_wp = wp_arr.grad
                        if grad_wp is not None:
                            grad_torch = wp.to_torch(grad_wp).reshape(torch_tensor.shape)
                            grad_torch = torch.nan_to_num(grad_torch, nan=0.0, posinf=0.0, neginf=0.0)
                            if torch.norm(grad_torch).item() > 0:
                                torch_tensor.backward(grad_torch, retain_graph=True)
                
                # Extract kp/kd gradients → PyTorch
                if hasattr(self, '_last_kp_wp'):
                    for name, wp_arr, torch_t in [('kp', self._last_kp_wp, self._last_stiffness_t),
                                                   ('kd', self._last_kd_wp, self._last_damping_t)]:
                        grad_wp = self._warp_tape.gradients.get(wp_arr)
                        if grad_wp is None and wp_arr.grad is not None:
                            grad_wp = wp_arr.grad
                        if grad_wp is not None and torch_t.requires_grad:
                            grad_torch = wp.to_torch(grad_wp).reshape(torch_t.shape)
                            grad_torch = torch.nan_to_num(grad_torch, nan=0.0, posinf=0.0, neginf=0.0)
                            if torch.norm(grad_torch).item() > 0:
                                torch_t.backward(grad_torch, retain_graph=True)
                
                if (i - start_iter) < 5:
                    g_parts = []
                    for name, p in [('E', self.raw_E), ('nu', self.raw_nu), ('wt', self.log_weights),
                                    ('fk', self.raw_fiber_k), ('kp', self.raw_ctrl_stiffness), ('kd', self.raw_ctrl_damping)]:
                        g = torch.norm(p.grad).item() if p.grad is not None else 0.0
                        g_parts.append(f"{name}:{g:.2e}")
                    tqdm.write(f"  [{self.scene_id}] Tape grads: {', '.join(g_parts)}")

                self._accumulated_grad_x = None

            # --- Per-parameter gradient clipping ---
            phys_params = [self.log_weights, self.raw_E, self.raw_nu, self.raw_fiber_k, self.raw_fiber_dir, self.raw_yield, self.raw_viscosity, self.raw_ctrl_stiffness, self.raw_ctrl_damping]
            per_param_clip = getattr(self.cfg.mpm, 'per_param_clip', 1.0)
            _pre_clip_norms = []
            _max_pn = 0.0
            for p in phys_params:
                if p.grad is not None:
                    _pre_clip_norms.append(torch.norm(p.grad).item())
                    pn = torch.nn.utils.clip_grad_norm_([p], max_norm=per_param_clip)
                    _max_pn = max(_max_pn, pn.item())
            phys_grad_norm_preclip = max(_pre_clip_norms) if _pre_clip_norms else 0.0
            phys_grad_norm = torch.tensor(_max_pn)
            
            res_grad_norm = torch.tensor(0.0)
            if self.residual_net is not None:
                res_clip = getattr(self.residual_cfg, 'grad_clip_norm', 1.0)
                res_grad_norm = torch.nn.utils.clip_grad_norm_(self.residual_net.parameters(), max_norm=res_clip)
            
            grad_norm = phys_grad_norm
            
            # [rgb_finetune] In rgb_finetune mode all physical params are frozen,
            # so phys_grad_norm is legitimately zero — do not treat that as an
            # error and still call optimizer.step() to update residual_net + GS.
            rgb_ft = getattr(self, 'rgb_finetune_mode', False)
            phys_is_bad = (torch.isnan(phys_grad_norm) or torch.isinf(phys_grad_norm)
                           or (phys_grad_norm == 0 and not rgb_ft))
            if phys_is_bad:
                param_status = []
                for name, p in [('weights', self.log_weights), ('E', self.raw_E), ('nu', self.raw_nu), ('fk', self.raw_fiber_k)]:
                    g_norm = torch.norm(p.grad).item() if p.grad is not None else -1.0
                    param_status.append(f"{name}:{g_norm:.1e}")
                tqdm.write(f"[WARNING] Invalid phys gradient norm ({phys_grad_norm}) in Iter {i+1}! Params: {', '.join(param_status)}. Skipping.")
                self.optimizer.zero_grad()
            else:
                with torch.no_grad():
                    w_mean = w_patch.mean(dim=0).tolist()
                    w_str = ", ".join([f"{self.active_experts[j]}:{w_mean[j]:.3f}" for j in range(len(self.active_experts))])
                    tqdm.write(f"[DEBUG] Iter {i+1} Phys Grad (pre-clip): {phys_grad_norm:.2e}, Res Grad: {res_grad_norm:.2e}")
                    tqdm.write(f"        Mean E: {E_patch.mean().item():.2e}, Mean nu: {nu_patch.mean().item():.3f}")
                    tqdm.write(f"        Mean Mu: {mu_patch.mean().item():.2f}, Mean Lam: {lam_patch.mean().item():.2f}")
                    tqdm.write(f"        Mean Weights: [{w_str}]")
                    tqdm.write(f"        Ctrl Stiffness: {ctrl_stiffness_t.item():.1f}, Ctrl Damping: {ctrl_damping_t.item():.1f}")
                    # Per-param gradient diagnostics
                    g_parts = []
                    for name, p in [('E', self.raw_E), ('nu', self.raw_nu), ('wt', self.log_weights),
                                    ('fk', self.raw_fiber_k), ('kp', self.raw_ctrl_stiffness), ('kd', self.raw_ctrl_damping)]:
                        g = torch.norm(p.grad).item() if p.grad is not None else 0.0
                        g_parts.append(f"{name}:{g:.2e}")
                    tqdm.write(f"        Param Grads: {', '.join(g_parts)}")
                self.optimizer.step()
                
                # [FIXED] Only step scheduler if the optimizer actually stepped
                if self.scheduler is not None:
                    self.scheduler.step()
                
                lr_group_names = ['Weights', 'E', 'Nu', 'Fiber_K', 'Fiber_Dir', 'Yield', 'Viscosity', 'Ctrl_Stiffness', 'Ctrl_Damping']
                if self.residual_net is not None:
                    lr_group_names.append('Residual')
                for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    name = lr_group_names[group_idx] if group_idx < len(lr_group_names) else f'Group_{group_idx}'
                    self.writer.add_scalar(f'LR/{name}', param_group['lr'], i)

            self.writer.add_scalar('GradNorm/Physics', phys_grad_norm.item(), i)
            self.writer.add_scalar('GradNorm/Physics_PreClip', phys_grad_norm_preclip, i)
            if self.residual_net is not None:
                self.writer.add_scalar('GradNorm/Residual', res_grad_norm.item(), i)

            total_loss_val = total_loss if isinstance(total_loss, float) else total_loss.item()
            current_iter_loss = total_loss_val * T
            main_pbar.set_postfix({'total_loss': f"{current_iter_loss:.6f}"})
            
            self.writer.add_scalar('Loss/Total', current_iter_loss, i)
            if len(loss_stats['track']) > 0:
                self.writer.add_scalar('Loss/Track', np.sum(loss_stats['track']), i)
                self.writer.add_scalar('Loss/Chamfer', np.sum(loss_stats['chamfer']), i)
                self.writer.add_scalar('Loss/Render', np.mean(loss_stats['render']), i)
                self.writer.add_scalar('Loss/Residual_Reg', np.mean(loss_stats['residual_reg']), i)
            if len(loss_stats.get('length', [])) > 0:
                self.writer.add_scalar('Loss/Length', np.sum(loss_stats['length']), i)
            # --- [NEW] Early Stopping Check ---
            if not hasattr(self, 'loss_history'):
                self.loss_history = []
            self.loss_history.append(current_iter_loss)
            
            # [FIXED] Increment session_iters to ensure we don't immediately early-stop on resume
            self.session_iters += 1
            
            patience = getattr(self.cfg.train, 'early_stop_patience', 10)
            min_delta = getattr(self.cfg.train, 'early_stop_min_delta', 1e-4)
            
            # Logic: Only check early stopping if we have enough total history AND
            # we have run for at least 'patience' iterations in the CURRENT session.
            if len(self.loss_history) > patience and self.session_iters > patience:
                # Check whether the best loss over the last patience iterations improved.
                recent_best = min(self.loss_history[-patience:])
                prev_best = min(self.loss_history[:-patience])
                
                if (prev_best - recent_best) < min_delta:
                    tqdm.write(f"[EARLY STOP] No significant improvement for {patience} iters in this session. Best loss: {recent_best:.6f}. Stopping.")
                    break

            if self.residual_net is not None and len(res_stats['mean_mag']) > 0:
                self.writer.add_scalar('Residual/Mean_Magnitude', np.mean(res_stats['mean_mag']), i)
                self.writer.add_scalar('Residual/Max_Magnitude', np.mean(res_stats['max_mag']), i)
                self.writer.add_scalar('Residual/Correction_to_Physics_Ratio', np.mean(res_stats['ratio_to_phys']), i)
                self.writer.add_scalar('Residual/Cosine_Similarity', np.mean(res_stats['cos_sim']), i)
                if len(res_stats['topology_edge']) > 0:
                    self.writer.add_scalar('Residual/Topology_Edge_Reg', np.mean(res_stats['topology_edge']), i)
                    self.writer.add_scalar('Residual/Topology_Smooth_Reg', np.mean(res_stats['topology_smooth']), i)
                
                if delta_v is not None:
                    self.writer.add_histogram('Residual/Delta_V_Distribution', delta_v.detach().cpu().numpy(), i)

            # Log average physical parameters to TensorBoard.
            self.writer.add_scalar('Params/E_Mean', E_patch.mean().item(), i)
            self.writer.add_scalar('Params/Nu_Mean', nu_patch.mean().item(), i)
            self.writer.add_scalar('Params/Mu_Mean', mu_patch.mean().item(), i)
            self.writer.add_scalar('Params/Lam_Mean', lam_patch.mean().item(), i)
            self.writer.add_scalar('Params/Global_Friction', friction.item(), i)
            self.writer.add_scalar('Params/Yield_Stress_Mean', yield_patch.mean().item(), i)
            self.writer.add_scalar('Params/Viscosity_Mean', visc_patch.mean().item(), i)
            
            # --- [NEW] Log additional optimized parameters ---
            # 1. Constitutive expert weights.
            w_mean = w_patch.mean(dim=0)
            for e_idx, expert_name in enumerate(self.active_experts):
                self.writer.add_scalar(f'Weights/{expert_name}', w_mean[e_idx].item(), i)
            
            # 2. Fiber strength.
            if 'fi' in self.active_experts:
                self.writer.add_scalar('Params/Fiber_K_Mean', fk_patch.mean().item(), i)
            
            # 3. Controller Gains
            self.writer.add_scalar('Params/Ctrl_Stiffness', ctrl_stiffness_t.item(), i)
            self.writer.add_scalar('Params/Ctrl_Damping', ctrl_damping_t.item(), i)
            
            # 4. Gradient norm for optimization stability.
            self.writer.add_scalar('Train/Grad_Norm', grad_norm, i)

            # --- [NEW] Save Best Checkpoint ---
            current_total_loss = total_loss_val * T
            if current_total_loss < self.best_loss:
                self.best_loss = current_total_loss
                best_path = os.path.join(self.cfg.output_dir, self.scene_id, "best_checkpoint.pt")
                self.save_checkpoint(best_path, iter=i+1)
                tqdm.write(f"[BEST] New best loss: {self.best_loss:.6f} at Iter {i+1}. Saved to {best_path}")

            if (i+1) % 50 == 0:
                self.save_checkpoint(os.path.join(self.log_dir, f"checkpoint_iter_{i+1}.pt"), iter=i+1)

        final_checkpoint_path = os.path.join(self.cfg.output_dir, self.scene_id, "final_checkpoint.pt")
        self.save_checkpoint(final_checkpoint_path, iter=last_iter)
        
        print(f"MPM Training Finished! Saved checkpoint to {final_checkpoint_path}")
        
        # --- [NEW] Qualitative Visualization ---
        print(f"Generating visualization video for {self.scene_id}...")
        video_path = os.path.join(self.cfg.output_dir, self.scene_id, "final_simulation.mp4")
        self.visualize(video_path)
        print(f"Visualization video saved to {video_path}")

    def load_from_checkpoint(self, resume_path):
        print(f"[RESUME] Loading state from {resume_path}...")
        
        if resume_path.endswith('.pt'):
            # Load Full Checkpoint
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            state = checkpoint['model_state_dict']
            with torch.no_grad():
                self.log_weights.copy_(state['log_weights'])
                self.raw_E.copy_(state['raw_E'])
                self.raw_nu.copy_(state['raw_nu'])
                self.raw_fiber_k.copy_(state['raw_fiber_k'])
                self.raw_fiber_dir.copy_(state['raw_fiber_dir'])
                self.raw_yield.copy_(state['raw_yield'])
                self.raw_viscosity.copy_(state['raw_viscosity'])
                if 'raw_ctrl_stiffness' in state:
                    self.raw_ctrl_stiffness.copy_(state['raw_ctrl_stiffness'])
                if 'raw_ctrl_damping' in state:
                    self.raw_ctrl_damping.copy_(state['raw_ctrl_damping'])
                
                # [NEW] Restore Best Loss and Loss History
                if 'best_loss' in checkpoint:
                    self.best_loss = checkpoint['best_loss']
                    print(f"[RESUME] Best loss restored: {self.best_loss:.6f}")
                if 'loss_history' in checkpoint:
                    self.loss_history = checkpoint['loss_history']
                    print(f"[RESUME] Loss history restored ({len(self.loss_history)} iters).")

                # [NEW] Restore Auto-centering Offset if available
                if 'auto_offset' in state:
                    self.auto_offset = state['auto_offset'].to(self.device)
                    # Sync to simulator
                    if hasattr(self, 'simulator'):
                        self.simulator.base_offset = self.auto_offset
                        self.simulator._apply_boundary()
                    print("[RESUME] Auto-centering offset restored.")

                # [NEW] Restore Patch Centers and Interpolation Weights if available
                # This is CRITICAL for consistent parameter mapping
                if 'patch_centers' in state:
                    self.patch_centers = state['patch_centers'].to(self.device)
                    from pytorch3d.ops import knn_points
                    init_pos_centered = (self.data['init_pos'].to(self.device) + self.auto_offset).unsqueeze(0)
                    dist, self.patch_idx, _ = knn_points(init_pos_centered, self.patch_centers, K=3)
                    dist = torch.clamp(dist, min=1e-6)
                    inv_dist = 1.0 / dist
                    norm = torch.sum(inv_dist, dim=2, keepdim=True)
                    self.interp_weights = (inv_dist / norm).unsqueeze(-1)
                    print("[RESUME] Persistent patch assignment restored.")

                # [NEW] Load ResidualPGND weights if available
                if 'residual_net' in state and self.residual_net is not None:
                    try:
                        self.residual_net.load_state_dict(state['residual_net'])
                        print("[RESUME] ResidualPGND weights loaded.")
                    except Exception as e:
                        print(f"[RESUME] Warning: Failed to load ResidualPGND weights: {e}")
            
            # We need to defer optimizer/scheduler load until train() creates them
            self.resume_checkpoint = checkpoint 
        else:
            # Load Legacy PKL
            with open(resume_path, 'rb') as f:
                ckpt = pickle.load(f)
            
            with torch.no_grad():
                if 'raw_E' in ckpt: self.raw_E.copy_(torch.from_numpy(ckpt['raw_E']).to(self.device))
                if 'raw_nu' in ckpt: self.raw_nu.copy_(torch.from_numpy(ckpt['raw_nu']).to(self.device))
                if 'raw_fiber_k' in ckpt: self.raw_fiber_k.copy_(torch.from_numpy(ckpt['raw_fiber_k']).to(self.device))
                if 'raw_fiber_dir' in ckpt: self.raw_fiber_dir.copy_(torch.from_numpy(ckpt['raw_fiber_dir']).to(self.device))
                if 'raw_yield' in ckpt: self.raw_yield.copy_(torch.from_numpy(ckpt['raw_yield']).to(self.device))
                if 'raw_viscosity' in ckpt: self.raw_viscosity.copy_(torch.from_numpy(ckpt['raw_viscosity']).to(self.device))
                if 'log_weights' in ckpt: self.log_weights.copy_(torch.from_numpy(ckpt['log_weights']).to(self.device))
            self.resume_checkpoint = None

    def test(self, checkpoint_path, output_path=None):
        """
        Load a specific checkpoint and run visualization/inference.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint {checkpoint_path} does not exist.")
            return

        self.load_from_checkpoint(checkpoint_path)
        
        if output_path is None:
             base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
             output_path = os.path.join(os.path.dirname(checkpoint_path), f"{base_name}_test.mp4")
             
        print(f"[TEST] Running inference with checkpoint: {checkpoint_path}")
        self.visualize(output_path)
        print(f"[TEST] Result saved to {output_path}")

    def visualize(self, output_path):
        """
        Run one final simulation and save as a video.
        """
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 1. Setup for final run
        self.simulator.eval()
        
        # Use automatically calculated offset
        offset = self.auto_offset
        
        init_pos = (self.data['init_pos'].to(self.device) + offset)
        controller_points = self.controller_points
        gt_tracks = (self.data['gt_surface_tracks'].to(self.device) + offset)
        num_supervised = self.data['num_supervised']
        
        # Get optimized props
        w_patch, mu_patch, lam_patch, fk_patch, fdir_patch, friction, yield_patch, E_patch, nu_patch, visc_patch, ctrl_stiff_viz, ctrl_damp_viz = self.get_current_phys_props()
        
        def gather_and_interp(patch_data):
            flat_idx = self.patch_idx.squeeze(0).view(-1)
            gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
            return torch.sum(self.interp_weights * gathered, dim=2).squeeze(0)

        p_weights = gather_and_interp(w_patch)
        p_mu = gather_and_interp(mu_patch.unsqueeze(-1)).squeeze()
        p_lam = gather_and_interp(lam_patch.unsqueeze(-1)).squeeze()
        p_fk = gather_and_interp(fk_patch.unsqueeze(-1)).squeeze()
        p_fdir = torch.nn.functional.normalize(gather_and_interp(fdir_patch), dim=1, eps=1e-8)
        p_yield = gather_and_interp(yield_patch.unsqueeze(-1)).squeeze()
        p_visc = gather_and_interp(visc_patch.unsqueeze(-1)).squeeze()
        expert_params = {'mu': p_mu, 'lam': p_lam, 'fiber_k': p_fk, 'fiber_dir': p_fdir, 'yield_stress': p_yield, 'plastic_viscosity': p_visc}

        # 2. Run Simulation
        self.simulator.reset(init_pos, controller_pos=controller_points[0])
        
        # Build moe_params_wp (same format as training loop)
        expert_order = ['nh', 'co', 'st', 'fi']
        active_experts_list = getattr(self.cfg.mpm, 'active_experts', expert_order)
        mask_active = [1 if e in active_experts_list else 0 for e in expert_order]
        active_mask_wp = wp.array(mask_active, dtype=wp.int32, device=self.simulator.warp_device)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            moe_params_wp = {
                'weights': torch2warp_float(p_weights),
                'mu': torch2warp_float(p_mu),
                'lam': torch2warp_float(p_lam),
                'fk': torch2warp_float(p_fk),
                'fdir': torch2warp_vec3(p_fdir),
                'active_mask': active_mask_wp
            }
        
        # [NEW] Initialize History Buffers for ResidualPGND
        H = getattr(self.residual_cfg, 'n_history', 2) if self.residual_cfg else 2
        x_history = [init_pos.clone() for _ in range(H)]
        v_history = [torch.zeros_like(init_pos) for _ in range(H)]

        T_data = controller_points.shape[0]
        T = min(T_data, self.cfg.mpm.max_frames) if self.cfg.mpm.max_frames > 0 else T_data
        
        temp_dir = tempfile.mkdtemp(prefix="temp_frames_", dir=output_dir)
        
        frames = []
        with torch.no_grad():
            warmup_frames = getattr(self.cfg.mpm, 'controller_warmup_frames', 10)
            for t in tqdm(range(T), desc="Rendering Frames"):
                c_pos_end = controller_points[t]
                c_pos_start = controller_points[t-1] if t > 0 else c_pos_end
                
                v_ctrl_t = (c_pos_end - c_pos_start) / (self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame)

                warmup_factor = min(1.0, (t + 1) / (warmup_frames + 1e-6))
                viz_stiff = float(ctrl_stiff_viz.item()) * warmup_factor
                viz_damp = float(ctrl_damp_viz.item()) * warmup_factor

                # [SCHEME A] Phase 1: Pure MPM Physics Solver Loop
                x_start_frame = (self.simulator.x - self.simulator.shift).detach().unsqueeze(0)

                for s in range(self.cfg.mpm.steps_per_frame):
                    alpha = (s + 1) / self.cfg.mpm.steps_per_frame
                    curr_target_pos = c_pos_start + alpha * (c_pos_end - c_pos_start)

                    x_curr = self.simulator.step(moe_params_wp, 
                                                 controller_pos=curr_target_pos, 
                                                 controller_vel=v_ctrl_t,
                                                 residual_v=None,
                                                 stiffness_override=viz_stiff,
                                                 damping_override=viz_damp)
                
                # [SCHEME A] Phase 2: Neural Feedback Correction
                if self.residual_net is not None:
                    self.residual_net.eval()
                    
                    x_his_tensor = torch.stack(x_history, dim=1).unsqueeze(0)
                    v_his_tensor = torch.stack(v_history, dim=1).unsqueeze(0)
                    
                    curr_x_mpm = (self.simulator.x - self.simulator.shift).unsqueeze(0)
                    curr_v_mpm = self.simulator.v.unsqueeze(0)
                    
                    delta_v = self.residual_net(curr_x_mpm, curr_v_mpm, x_start_frame, x_his_tensor, v_his_tensor).squeeze(0)
                    
                    self.simulator.v = self.simulator.v + delta_v
                    frame_dt = self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame
                    self.simulator.x = self.simulator.x + delta_v * frame_dt
                    
                    x_curr = self.simulator.x - self.simulator.shift

                    x_history.pop(0)
                    x_history.append((self.simulator.x - self.simulator.shift).detach())
                    v_history.pop(0)
                    v_history.append(self.simulator.v.detach())

                # Ensure memory is freed between frames
                self.simulator.x = self.simulator.x.detach()
                self.simulator.v = self.simulator.v.detach()
                self.simulator.F = self.simulator.F.detach()
                self.simulator.C = self.simulator.C.detach()
                
                # Plot frame.
                # Keep coordinates in the auto_offset-centered frame so fixed
                # [-0.5, 0.5] bounds always bracket the cloth. Subtracting
                # offset here would undo the centering and — for mono scenes
                # where mono_align only rotates, doesn't translate — push the
                # cloth toward y≈-0.8 / z≈-0.45 and out of the view window.
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Object particles
                obj_x = x_curr.detach().cpu().numpy()
                ax.scatter(obj_x[:, 0], obj_x[:, 1], obj_x[:, 2], s=1, c='blue', alpha=0.5, label='Simulation')

                # Ground Truth particles (Surface tracks)
                gt_x_raw = gt_tracks[t].detach()
                # [FIXED] Mask out zero-artifacts from GT visualization using RAW tracks
                gt_mask = torch.norm(self.data['gt_surface_tracks'][t], dim=-1) > 1e-5
                gt_x = gt_x_raw[gt_mask].cpu().numpy()
                ax.scatter(gt_x[:, 0], gt_x[:, 1], gt_x[:, 2], s=1, c='green', alpha=0.3, label='Ground Truth')

                # Controller points (already offset via self.controller_points)
                ctrl_x = c_pos_end.detach().cpu().numpy()
                if ctrl_x.shape[0] > 0:
                    ax.scatter(ctrl_x[:, 0], ctrl_x[:, 1], ctrl_x[:, 2], s=20, c='red', marker='x', label='Controller')
                    
                    # [NEW] Draw connections (if available) to show grip
                    # We need access to simulator.controller_indices and simulator.num_ctrl_points
                    if hasattr(self.simulator, 'controller_indices') and self.simulator.controller_indices is not None:
                         # Indices are flattened: [N_ctrl * K]
                        indices = self.simulator.controller_indices.view(-1).cpu().numpy()
                        # Connected object particles: [N_ctrl * K, 3]
                        conn_obj_x = obj_x[indices]
                        
                        # Controller points repeated: [N_ctrl, 3] -> [N_ctrl * K, 3]
                        K = getattr(self.cfg.mpm, 'controller_max_neighbors', 16)
                        # Be careful: actual neighbors per point might be less if filtered, but here we used KNN with fixed K
                        # However, our indices are [N_ctrl, K], so we can just repeat_interleave
                        conn_ctrl_x = np.repeat(ctrl_x, K, axis=0)
                        
                        # We also need the mask to only draw valid connections
                        if hasattr(self.simulator, 'controller_mask'):
                            mask = self.simulator.controller_mask.view(-1).cpu().numpy()
                            # Filter
                            valid_obj = conn_obj_x[mask]
                            valid_ctrl = conn_ctrl_x[mask]
                            
                            # Draw lines. To be fast, we can't do ax.plot for each line.
                            # We can plot a single line with NaNs to break segments.
                            # Format: x1, x2, nan, x3, x4, nan ...
                            n_lines = valid_obj.shape[0]
                            # Limit number of lines to avoid cluttering if too many
                            if n_lines > 500:
                                step = n_lines // 500
                                valid_obj = valid_obj[::step]
                                valid_ctrl = valid_ctrl[::step]
                                n_lines = valid_obj.shape[0]
                                
                            line_x = np.empty(n_lines * 3)
                            line_y = np.empty(n_lines * 3)
                            line_z = np.empty(n_lines * 3)
                            
                            line_x[0::3] = valid_ctrl[:, 0]
                            line_x[1::3] = valid_obj[:, 0]
                            line_x[2::3] = np.nan
                            
                            line_y[0::3] = valid_ctrl[:, 1]
                            line_y[1::3] = valid_obj[:, 1]
                            line_y[2::3] = np.nan
                            
                            line_z[0::3] = valid_ctrl[:, 2]
                            line_z[1::3] = valid_obj[:, 2]
                            line_z[2::3] = np.nan
                            
                            ax.plot(line_x, line_y, line_z, color='red', alpha=0.2, linewidth=0.5)
                
                # Fixed bounds for consistent video
                ax.set_xlim([-0.5, 0.5])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([-0.5, 0.5])
                ax.set_axis_off()
                ax.grid(False)
                # ax.set_title(f"Frame {t}")
                ax.legend(loc='upper right')
                
                frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
                plt.savefig(frame_path)
                plt.close(fig)
                frames.append(frame_path)

        # 3. Synthesize Video with System FFmpeg
        print(f"Synthesizing video to {output_path}...")
        input_pattern = os.path.abspath(os.path.join(temp_dir, 'frame_%04d.png'))
        output_abs_path = os.path.abspath(output_path)
        
        # Use absolute path to system ffmpeg to bypass conda environment's limited version
        ffmpeg_bin = "/usr/bin/ffmpeg"
        if not os.path.exists(ffmpeg_bin):
            ffmpeg_bin = "ffmpeg" # Fallback
            
        cmd = [
            ffmpeg_bin, '-y', '-loglevel', 'error', '-r', '30',
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', 
            '-pix_fmt', 'yuv420p',
            output_abs_path
        ]
        try:
            subprocess.run(cmd, check=True)
            success = os.path.exists(output_abs_path)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed with exit code {e.returncode}. See above for details.")
            success = False
        
        # 4. Cleanup
        if success:
            shutil.rmtree(temp_dir)
        else:
            print(f"Keeping temp frames at {temp_dir} for debugging.")

    def save_checkpoint(self, path, iter):
        """
        Save full training state for resumption (Optimizer, Scheduler, etc.)
        """
        model_state = {
            'log_weights': self.log_weights,
            'raw_E': self.raw_E,
            'raw_nu': self.raw_nu,
            'raw_fiber_k': self.raw_fiber_k,
            'raw_fiber_dir': self.raw_fiber_dir,
            'raw_yield': self.raw_yield,
            'raw_viscosity': self.raw_viscosity,
            'raw_ctrl_stiffness': self.raw_ctrl_stiffness,
            'raw_ctrl_damping': self.raw_ctrl_damping,
            'patch_centers': self.patch_centers,
            'auto_offset': self.auto_offset,
        }
        
        # [NEW] Include ResidualPGND weights if available
        if self.residual_net is not None:
            model_state['residual_net'] = self.residual_net.state_dict()

        checkpoint = {
            'iter': iter,
            'best_loss': self.best_loss,      # [NEW] Save best loss info
            'loss_history': getattr(self, 'loss_history', []), # [NEW] Save loss history for early stopping
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }
        torch.save(checkpoint, path)
        tqdm.write(f"[CHECKPOINT] Full training state saved to {path}")

    def save_results(self, path):
        w, mu, lam, fk, fdir, friction, ys, E, nu, visc, cs, cd = self.get_current_phys_props()
        data = {
            'scene_id': self.scene_id,
            'weights': w.detach().cpu().numpy(),
            'mu': mu.detach().cpu().numpy(),
            'lam': lam.detach().cpu().numpy(),
            'fiber_k': fk.detach().cpu().numpy(),
            'fiber_dir': fdir.detach().cpu().numpy(),
            'friction': friction.item(),
            'yield_stress': ys.detach().cpu().numpy(),
            'plastic_viscosity': visc.detach().cpu().numpy(),
            'E': E.detach().cpu().numpy(),
            'nu': nu.detach().cpu().numpy(),
            'controller_stiffness': cs.item(),
            'controller_damping': cd.item(),
            'raw_E': self.raw_E.detach().cpu().numpy(),
            'raw_nu': self.raw_nu.detach().cpu().numpy(),
            'raw_fiber_k': self.raw_fiber_k.detach().cpu().numpy(),
            'raw_fiber_dir': self.raw_fiber_dir.detach().cpu().numpy(),
            'raw_yield': self.raw_yield.detach().cpu().numpy(),
            'raw_viscosity': self.raw_viscosity.detach().cpu().numpy(),
            'raw_ctrl_stiffness': self.raw_ctrl_stiffness.detach().cpu().numpy(),
            'raw_ctrl_damping': self.raw_ctrl_damping.detach().cpu().numpy(),
            'log_weights': self.log_weights.detach().cpu().numpy()
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
