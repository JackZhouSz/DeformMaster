"""Stage-3 RGB finetune trainer — subclass of :class:`PhysExpertMPMTrainer`.

Adds photometric supervision on top of the existing chamfer + track training:
at every simulated frame, we LBS the static first-frame Gaussian Splatting
representation using the current MPM particle positions, rasterize the
resulting dynamic Gaussians through each camera, and compare the rendered
image to the recorded RGB frame. The photometric L1 loss is returned from
:meth:`compute_optional_render_loss`, which the base trainer already folds
into its frame-level loss (``render_loss`` term added at ``trainer_mpm.py:1011``).

Which parameters update
-----------------------
With ``rgb_finetune_mode = True`` the base trainer (patched above) forces
``requires_grad = False`` on all physical parameters (``raw_E``, ``raw_nu``,
``raw_fiber_*``, ``raw_yield``, ``raw_viscosity``, ``log_weights``,
``raw_ctrl_stiffness``, ``raw_ctrl_damping``); only ``residual_net`` stays
trainable from the base model. The subclass additionally adds the GS
parameters (``xyz`` / ``features_dc`` / ``features_rest`` / ``scaling`` /
``rotation`` / ``opacity``) to the optimizer.

Coordinate frames
-----------------
The simulator's ``x_curr`` lives in the ``(auto_offset + reverse_z)`` frame,
while the pretrained GS is stored in the original dataset world frame.
``_to_gs_frame`` undoes the two offsets so that LBS bones and Gaussians are
coherent.
"""

from __future__ import annotations

import glob
import os
import re
import types
from typing import List, Optional

import torch
import torch.nn.functional as F

from .trainer_mpm import PhysExpertMPMTrainer
from ..data.dataset_bridge import BridgeSequenceDataset
from ..render.differentiable_dynamic_gs import (
    init_lbs_relations,
    step_lbs,
    render_dynamic_frame,
)
from gaussian_splatting.scene.gaussian_model import GaussianModel


# The appearance-training script writes the first-frame GS with this exact
# exp_name; rendering and RGB refinement rely on it too. Kept here so
# configs can stay short ("gaussian_output_root: gaussian_output") without
# repeating the 80-char directory name every time.
_GS_EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"


def _resolve_first_frame_gs_ply(root: str, case: str,
                                exp_name: str = _GS_EXP_NAME,
                                iteration: int = -1) -> str:
    """Find the standard first-frame GS .ply for a case under `root`.

    Mirrors the layout produced by the appearance-training script::

        {root}/{case}/{exp_name}/point_cloud/iteration_{N}/point_cloud.ply

    ``iteration=-1`` picks the highest iteration folder available (matches
    ``gs_render_dynamics.py --iteration -1`` default).
    """
    model_dir = os.path.join(root, case, exp_name, "point_cloud")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"GS model dir not found: {model_dir}. Did gs_run.sh train this case?"
        )
    if iteration >= 0:
        iter_dir = os.path.join(model_dir, f"iteration_{iteration}")
    else:
        candidates = []
        for entry in os.listdir(model_dir):
            m = re.match(r"iteration_(\d+)", entry)
            if m and os.path.isdir(os.path.join(model_dir, entry)):
                candidates.append((int(m.group(1)), entry))
        if not candidates:
            raise FileNotFoundError(f"No iteration_* subfolder under {model_dir}")
        candidates.sort()
        iter_dir = os.path.join(model_dir, candidates[-1][1])
    ply = os.path.join(iter_dir, "point_cloud.ply")
    if not os.path.isfile(ply):
        raise FileNotFoundError(f"Expected GS ply not found: {ply}")
    return ply


class RGBFinetuneTrainer(PhysExpertMPMTrainer):
    """Stage-3 trainer: RGB photometric finetune on top of a dynamics ckpt."""

    def __init__(self, cfg, scene_id, resume_path: Optional[str] = None,
                 for_inference: bool = False):
        super().__init__(cfg, scene_id, resume_path=resume_path,
                         for_inference=for_inference)

        rgb_cfg = getattr(cfg, 'rgb_finetune', None)
        if rgb_cfg is None or not getattr(rgb_cfg, 'enabled', False):
            raise ValueError(
                "RGBFinetuneTrainer requires cfg.rgb_finetune.enabled=true; "
                "use PhysExpertMPMTrainer for stage-1 training."
            )

        self.rgb_finetune_mode = True       # read by patched trainer_mpm.py:784

        # Stage-3 is a fresh training stage; reset the resumed iter counter so
        # ``train()`` actually loops ``num_iters`` times instead of skipping
        # because stage-1 already reached a higher iter count.
        if self.resume_checkpoint is not None:
            self.resume_checkpoint['iter'] = 0
            # stage-2's CosineAnnealingLR last_epoch is meaningless under
            # stage-3's much smaller T_max — drop it so stage-3 anneals
            # cleanly from initial lr to eta_min over its own num_iters.
            self.resume_checkpoint['scheduler_state_dict'] = None
            # Stage-3 frame_loss includes ``render_loss`` and lives in a
            # different numeric scale than stage-2's. If we keep stage-2's
            # loss_history, the early-stop check (recent_best vs prev_best)
            # treats every stage-3 iter as "no improvement" and stops at
            # ~iter patience. Reset both so early-stop measures progress
            # within stage-3 only.
            self.best_loss = float('inf')
            self.loss_history = []
        self.rgb_cfg = rgb_cfg
        self.lambda_rgb = float(getattr(rgb_cfg, 'lambda_rgb', 1.0))
        self.bg = getattr(rgb_cfg, 'bg', 'black')
        self.lbs_k = int(getattr(rgb_cfg, 'lbs_k', 16))
        self.cams_used = list(getattr(rgb_cfg, 'cams', [0]))
        self.max_rgb_frames = int(getattr(rgb_cfg, 'max_frames', 20))
        # If freeze_gs, the base GS parameters are held fixed. RGB gradient
        # still flows through the render to residual_net via LBS, but the
        # GS state itself (xyz/scale/rot/opacity/SH) never updates. Used to
        # isolate "does RGB help dynamics alone?" from "does RGB hurt a
        # carefully-trained GS appearance?".
        self.freeze_gs = bool(getattr(rgb_cfg, 'freeze_gs', False))

        # 1. Static GS (first-frame appearance, pretrained with gs_run.sh).
        # Config only needs `gaussian_output_root` (default 'gaussian_output');
        # we resolve exp_name + highest iteration automatically. Legacy
        # `gs_ply_path` still accepted for explicit overrides.
        sh_degree = int(getattr(rgb_cfg, 'sh_degree', 3))
        explicit_ply = getattr(rgb_cfg, 'gs_ply_path', None)
        if explicit_ply:
            # Allow {case} substitution for backwards compat.
            gs_ply = str(explicit_ply).format(case=scene_id)
        else:
            gs_root = getattr(rgb_cfg, 'gaussian_output_root', 'gaussian_output')
            exp_name = getattr(rgb_cfg, 'gs_exp_name', _GS_EXP_NAME)
            iteration = int(getattr(rgb_cfg, 'gs_iteration', -1))
            gs_ply = _resolve_first_frame_gs_ply(
                gs_root, scene_id, exp_name=exp_name, iteration=iteration,
            )
        print(f"[rgb_finetune] first-frame GS ply: {gs_ply}")
        self._base_gs = GaussianModel(sh_degree)
        self._base_gs.load_ply(gs_ply)
        if self._base_gs._scaling.shape[1] == 1:
            self._base_gs.isotropic = True
            print(f"[rgb_finetune] detected isotropic GS (1 scale channel)")
        gs_params = [
            self._base_gs._xyz,
            self._base_gs._features_dc,
            self._base_gs._features_rest,
            self._base_gs._scaling,
            self._base_gs._rotation,
            self._base_gs._opacity,
        ]
        n_gs = self._base_gs._xyz.shape[0]
        n_gs_scalars = sum(p.numel() for p in gs_params)
        print(f"[rgb_finetune] GS loaded: N={n_gs:,}  scalars={n_gs_scalars:,}")

        if self.freeze_gs:
            # Fully freeze the pretrained GS. We keep using `self._base_gs` via
            # DynamicGSView in render_dynamic_frame; grad won't reach these
            # leaves since requires_grad is False.
            for p in gs_params:
                p.requires_grad_(False)
            print(f"[rgb_finetune] GS FROZEN — residual_net is the only thing RGB touches")
        else:
            gs_lr = float(getattr(rgb_cfg, 'gs_lr', 1e-3))
            self.optimizer.add_param_group({'params': gs_params, 'lr': gs_lr,
                                            'name': 'gs_rgb_finetune'})

        # 2. BridgeSequenceDataset: per-frame Camera + RGB + mask
        bridge_cfg = types.SimpleNamespace()
        bridge_cfg.data = types.SimpleNamespace(root=cfg.data.root)
        self._bridge = BridgeSequenceDataset(bridge_cfg, scene_id, split="train")

        # 3. Preload GT RGB + cameras. We pre-COMPOSE the GT so that its
        # background pixels are the same bg_color the rasterizer uses; the
        # object pixels stay at the real captured RGB. The training loss is
        # then a plain full-image L1 against this composed GT, which:
        #   - matches captured colour in the object region, AND
        #   - explicitly penalises Gaussian bleed into the background
        #     (bleed -> rendered != bg_color -> loss > 0).
        # An earlier "mask-both-sides" variant set background loss to zero on
        # both sides, which let Gaussians drift outside the object without
        # penalty (see Phase-2 mask-only retry: train PSNR -8.1 dB, IoU -0.27).
        # GT + masks are preloaded to CPU (pinned memory) and moved to GPU
        # lazily per-use. This avoids keeping ~3 GB of RGBA + mask tensors
        # resident on GPU for the entire training run (they'd only be touched
        # briefly inside compute_optional_render_loss each iter). Expected
        # overhead: ~5ms/frame CPU->GPU copy ≈ <0.5% of total training time.
        self._gt_frames: dict = {}    # {(t, cam): (3, H, W) composed GT, CPU}
        self._vis_masks: dict = {}    # {(t, cam): (1, H, W) hand-exclusion mask, CPU}
        self._cameras: dict = {}      # {(t, cam): Camera}
        max_t_data = self._bridge.frame_indices[-1] + 1
        # max_frames <= 0 means "use all available frames" (full sequence).
        if self.max_rgb_frames <= 0:
            self._window_T = max_t_data
        else:
            self._window_T = min(self.max_rgb_frames, max_t_data)
        # Compose on CPU (float32), stored in pinned memory for fast later
        # async H2D copy inside compute_optional_render_loss.
        bg_for_compose = torch.tensor(
            [1.0, 1.0, 1.0] if self.bg == 'white' else [0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
        for t in range(self._window_T):
            for c in self.cams_used:
                if (t, c) not in self._bridge.sample_to_index:
                    continue
                sample = self._bridge.get_sample(t, c)
                cam = sample.camera
                rgba = cam.original_image.float().clamp(0, 1).cpu()  # keep on CPU
                if rgba.shape[0] >= 4:
                    gt_rgb = rgba[:3]
                    mask = rgba[3:4]                              # (1, H, W) in [0,1]
                else:
                    gt_rgb = rgba[:3]
                    mask = torch.ones(1, rgba.shape[1], rgba.shape[2])
                gt_composed = gt_rgb * mask + bg_for_compose[:, None, None] * (1 - mask)
                controller_mask = sample.controller_mask.float().cpu()    # (H, W) [0,1]
                if controller_mask.dim() == 2:
                    controller_mask = controller_mask.unsqueeze(0)
                vis_mask = (1.0 - controller_mask).clamp(0, 1)
                # Pinned memory = async copy to GPU later
                self._gt_frames[(t, c)] = gt_composed.contiguous().pin_memory()
                self._vis_masks[(t, c)] = vis_mask.contiguous().pin_memory()
                self._cameras[(t, c)] = cam
        print(f"[rgb_finetune] preloaded {len(self._gt_frames)} frame×cam samples "
              f"(T<={self._window_T} × {len(self.cams_used)} cams, bg-composed)")

        # 4. LBS relations (static, frame-0 topology) — computed in GS frame
        #    from ``data['init_pos']`` (already in dataset world frame).
        init_pos_world = self.data['init_pos'].to(self.device)
        pc = self.data.get('particle_counts', {})
        n_pkl = pc.get('surface', 0) + pc.get('other_surface', 0) + pc.get('interior', 0)
        if n_pkl <= 0 or n_pkl >= init_pos_world.shape[0]:
            n_pkl = init_pos_world.shape[0]
        self._n_lbs_particles = n_pkl
        self._lbs_relations = init_lbs_relations(
            init_pos_world[:n_pkl], K=self.lbs_k,
        )
        print(f"[rgb_finetune] LBS relations built on {n_pkl} bones, K={self.lbs_k}")

        # 5. Rendering pipeline stub
        self._pipe = types.SimpleNamespace(
            convert_SHs_python=False, compute_cov3D_python=False, debug=False,
        )
        self._bg_color = torch.tensor(
            [1.0, 1.0, 1.0] if self.bg == 'white' else [0.0, 0.0, 0.0],
            device=self.device, dtype=torch.float32,
        )

        # 6. Frame-0 particle anchor (fixed reference for one-shot LBS at every
        #    frame). Re-cached from the current iter's initial state at t=0.
        self._particles_frame0: Optional[torch.Tensor] = None

    def save_finetuned_gs(self, path: Optional[str] = None) -> str:
        """Write the current (finetuned) GS state to a .ply. The base trainer
        checkpoint does not persist externally-added GS params, so this must
        be called explicitly after training to preserve RGB-loss updates.

        When freeze_gs=True the GS state never changed, but we still save so
        downstream pipelines can load a consistent path."""
        if path is None:
            path = os.path.join(self.cfg.output_dir, self.scene_id, 'finetuned_gs.ply')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._base_gs.save_ply(path)
        tag = '(unchanged — freeze_gs)' if self.freeze_gs else ''
        print(f"[rgb_finetune] Saved finetuned GS to {path} {tag}")
        return path

    # ---------------------------------------------------------------------
    # Frame-coordinate conversion: simulator frame -> GS (original world) frame
    # ---------------------------------------------------------------------
    def _to_gs_frame(self, x_sim: torch.Tensor) -> torch.Tensor:
        """Map an MPM-frame tensor back into the dataset world frame that the
        pretrained GS lives in."""
        x = x_sim[:self._n_lbs_particles] - self.auto_offset
        if getattr(self.cfg.data, 'reverse_z', False):
            x = x.clone()
            x[..., 2] = -x[..., 2]
        return x

    # ---------------------------------------------------------------------
    # Main hook: called by base trainer at trainer_mpm.py:984
    # ---------------------------------------------------------------------
    def compute_optional_render_loss(
        self, iter_idx: int, frame_idx: int, x_curr: torch.Tensor,
    ) -> torch.Tensor:
        if frame_idx == 0:
            # Re-anchor frame-0 bones each iter (detached — the reference frame
            # is a fixed input to LBS, no grad needed back to the simulator's
            # init state).
            self._particles_frame0 = self._to_gs_frame(x_curr).detach()
            return torch.tensor(0.0, device=self.device)

        if frame_idx >= self._window_T or self._particles_frame0 is None:
            return torch.tensor(0.0, device=self.device)

        cur_particles = self._to_gs_frame(x_curr)   # with grad to residual_net
        # One-shot LBS: deform from frame 0 to frame t in a single step, so
        # grads reach ``_base_gs`` parameters (xyz/rotation) on EVERY frame,
        # not just t=1. Also simpler and stateless.
        new_gs_xyz, new_gs_quat = step_lbs(
            particles_prev=self._particles_frame0,         # fixed reference
            particles_cur=cur_particles,                   # grad to residual_net
            gs_xyz=self._base_gs.get_xyz,                  # grad to base _xyz
            gs_quat=self._base_gs.get_rotation,            # grad to base _rotation
            relations=self._lbs_relations,
            K=self.lbs_k,
        )

        total = 0.0
        n_views = 0
        for c in self.cams_used:
            key = (frame_idx, c)
            if key not in self._cameras:
                continue
            result = render_dynamic_frame(
                base_gs=self._base_gs,
                gs_xyz=new_gs_xyz,
                gs_quat=new_gs_quat,
                camera=self._cameras[key],
                pipe=self._pipe,
                bg_color=self._bg_color,
                use_gsplat=True,
            )
            rendered_rgb = result['render'][:3]                   # full-image
            # Lazy GPU transfer: non_blocking works because CPU tensors are pinned.
            gt = self._gt_frames[key].to(self.device, non_blocking=True)
            vis = self._vis_masks[key].to(self.device, non_blocking=True)
            total = total + F.l1_loss(rendered_rgb * vis, gt * vis)
            n_views += 1
        if n_views == 0:
            return torch.tensor(0.0, device=self.device)
        return self.lambda_rgb * total / n_views
