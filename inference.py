"""
Inference script for the Hybrid Physics-Neural MPM Simulator.

Loads a best_checkpoint.pt (with ResidualPGND + physics params), runs the
full simulation, and exports particle trajectories as inference.pkl in the
same format as the original DeformMaster pipeline: ndarray of shape (T, N, 3)
in the ORIGINAL coordinate frame (no reverse_z, no auto_offset).

Also renders inference.mp4 with RGB overlay on the original camera view,
matching inference_warp.py's output format.

Usage:
    python inference.py --case_name double_lift_sloth --config configs/volumetric.yaml
    python inference.py --case_name double_lift_cloth_1 --config configs/planar.yaml
    python inference.py --case_name single_push_rope --config configs/linear.yaml
"""

import warnings
warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor", module="warp")

import os
import argparse
import json
import pickle
import shutil
import subprocess
import time
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from deformmaster.engine.engine_mpm import DeformMasterMPMEngine


def run_inference(trainer, save_path, tb_logdir=None):
    """
    Run full simulation and save particle trajectories as inference.pkl.
    Coordinate transform: MPM grid -> original DeformMaster frame.

    If tb_logdir is set, also logs per-frame particle clouds to TensorBoard
    (add_mesh), viewable under the "Mesh" tab. Colors use the first-frame
    GT colors for surface particles and light grey for the rest.
    """
    trainer.simulator.eval()

    writer = None
    mesh_colors_np = None
    if tb_logdir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_logdir)

    offset = trainer.auto_offset
    reverse_z = getattr(trainer.cfg.data, 'reverse_z', False)

    init_pos = trainer.data['init_pos'].to(trainer.device) + offset
    controller_points = trainer.controller_points
    T_data = controller_points.shape[0]
    T = min(T_data, trainer.cfg.mpm.max_frames) if trainer.cfg.mpm.max_frames > 0 else T_data

    # Get optimized physical properties and controller gains.
    (
        w_patch,
        mu_patch,
        lam_patch,
        fk_patch,
        fdir_patch,
        friction,
        yield_patch,
        E_patch,
        nu_patch,
        visc_patch,
        ctrl_stiffness_t,
        ctrl_damping_t,
    ) = trainer.get_current_phys_props()

    def gather_and_interp(patch_data):
        flat_idx = trainer.patch_idx.squeeze(0).view(-1)
        gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
        return torch.sum(trainer.interp_weights * gathered, dim=2).squeeze(0)

    p_weights = gather_and_interp(w_patch)
    p_mu = gather_and_interp(mu_patch.unsqueeze(-1)).squeeze()
    p_lam = gather_and_interp(lam_patch.unsqueeze(-1)).squeeze()
    p_fk = gather_and_interp(fk_patch.unsqueeze(-1)).squeeze()
    p_fdir = torch.nn.functional.normalize(gather_and_interp(fdir_patch), dim=1, eps=1e-8)
    p_yield = gather_and_interp(yield_patch.unsqueeze(-1)).squeeze()
    p_visc = gather_and_interp(visc_patch.unsqueeze(-1)).squeeze()

    import warp as wp
    from deformmaster.model.diff_simulator.warp_solver.warp_utils import torch2warp_float, torch2warp_vec3

    trainer.simulator.reset(init_pos, controller_pos=controller_points[0])

    expert_order = ['nh', 'co', 'st', 'fi']
    active_experts_list = getattr(trainer.cfg.mpm, 'active_experts', expert_order)
    mask_active = [1 if e in active_experts_list else 0 for e in expert_order]
    active_mask_wp = wp.array(mask_active, dtype=wp.int32, device=trainer.simulator.warp_device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        moe_params_wp = {
            'weights': torch2warp_float(p_weights),
            'mu': torch2warp_float(p_mu),
            'lam': torch2warp_float(p_lam),
            'fk': torch2warp_float(p_fk),
            'fdir': torch2warp_vec3(p_fdir),
            'active_mask': active_mask_wp,
        }

    # History buffers for ResidualPGND
    H = getattr(trainer.cfg.residual if hasattr(trainer.cfg, 'residual') else None, 'n_history', 2)
    x_history = [init_pos.clone() for _ in range(H)]
    v_history = [torch.zeros_like(init_pos) for _ in range(H)]

    all_positions = []

    # Only export the original PKL particles (surface + other_surface + interior),
    # excluding Gaussian-filled particles that were added for MPM volume simulation.
    # This matches the original DeformMaster inference.pkl format and avoids OOM in LBS.
    pc = trainer.data.get('particle_counts', {})
    n_pkl = pc.get('surface', 0) + pc.get('other_surface', 0) + pc.get('interior', 0)
    n_total = init_pos.shape[0]
    if n_pkl > 0 and n_pkl < n_total:
        print(f"Exporting first {n_pkl} particles (excluding {n_total - n_pkl} Gaussian-filled)")
    else:
        n_pkl = n_total
        print(f"Exporting all {n_pkl} particles")

    mono_R_cpu = None
    _mono_R = getattr(trainer, '_mono_align_R', None)
    if _mono_R is not None:
        mono_R_cpu = _mono_R.detach().cpu().float()

    def to_original_frame(pos_shifted):
        """Invert sim-frame pipeline (raw -> reverse_z -> @ R.T -> + offset):
        sim -> (- offset) -> (@ R to undo mono_align) -> (reverse_z back).
        """
        pos = (pos_shifted[:n_pkl].detach() - offset).cpu()
        if mono_R_cpu is not None:
            pos = pos @ mono_R_cpu
        if reverse_z:
            pos[..., 2] *= -1.0
        return pos

    if writer is not None:
        # Per-particle colors for TB mesh: solid blue by default.
        mesh_colors_np = np.tile(np.array([50, 120, 255], dtype=np.uint8), (n_pkl, 1))

    def tb_log_frame(pos_cpu_tensor, step):
        if writer is None:
            return
        vertices = pos_cpu_tensor.unsqueeze(0)  # (1, N, 3)
        colors = torch.from_numpy(mesh_colors_np).unsqueeze(0)  # (1, N, 3) uint8
        writer.add_mesh("mpm_particles", vertices=vertices, colors=colors, global_step=step)

    # Frame 0: save initial state before any simulation (matches original DeformMaster format)
    init_x = (trainer.simulator.x - trainer.simulator.shift)
    all_positions.append(to_original_frame(init_x))
    tb_log_frame(all_positions[-1], 0)

    with torch.no_grad():
        warmup_frames = getattr(trainer.cfg.mpm, 'controller_warmup_frames', 3)
        for t in tqdm(range(1, T), desc="MPM Inference"):
            c_pos_end = controller_points[t]
            c_pos_start = controller_points[t - 1]
            v_ctrl_t = (c_pos_end - c_pos_start) / (trainer.cfg.mpm.dt * trainer.cfg.mpm.steps_per_frame)

            x_start_frame = (trainer.simulator.x - trainer.simulator.shift).detach().unsqueeze(0)
            warmup_factor = min(1.0, (t + 1) / (warmup_frames + 1e-6))
            current_stiffness = float(ctrl_stiffness_t.item()) * warmup_factor
            current_damping = float(ctrl_damping_t.item()) * warmup_factor

            for s in range(trainer.cfg.mpm.steps_per_frame):
                alpha = (s + 1) / trainer.cfg.mpm.steps_per_frame
                curr_target_pos = c_pos_start + alpha * (c_pos_end - c_pos_start)

                x_curr = trainer.simulator.step(
                    moe_params_wp,
                    controller_pos=curr_target_pos,
                    controller_vel=v_ctrl_t,
                    residual_v=None,
                    stiffness_override=current_stiffness,
                    damping_override=current_damping,
                )

            # Neural feedback correction
            if trainer.residual_net is not None:
                trainer.residual_net.eval()
                x_his_tensor = torch.stack(x_history, dim=1).unsqueeze(0)
                v_his_tensor = torch.stack(v_history, dim=1).unsqueeze(0)
                curr_x_mpm = (trainer.simulator.x - trainer.simulator.shift).unsqueeze(0)
                curr_v_mpm = trainer.simulator.v.unsqueeze(0)
                delta_v = trainer.residual_net(
                    curr_x_mpm, curr_v_mpm, x_start_frame,
                    x_his_tensor, v_his_tensor
                ).squeeze(0)
                trainer.simulator.v = trainer.simulator.v + delta_v
                frame_dt = trainer.cfg.mpm.dt * trainer.cfg.mpm.steps_per_frame
                trainer.simulator.x = trainer.simulator.x + delta_v * frame_dt
                x_curr = trainer.simulator.x - trainer.simulator.shift

                x_history.pop(0)
                x_history.append((trainer.simulator.x - trainer.simulator.shift).detach())
                v_history.pop(0)
                v_history.append(trainer.simulator.v.detach())

            trainer.simulator.x = trainer.simulator.x.detach()
            trainer.simulator.v = trainer.simulator.v.detach()
            trainer.simulator.F = trainer.simulator.F.detach()
            trainer.simulator.C = trainer.simulator.C.detach()

            all_positions.append(to_original_frame(x_curr))
            tb_log_frame(all_positions[-1], t)

    if writer is not None:
        writer.close()
        print(f"TB mesh logs written to {tb_logdir}")

    trajectory = torch.stack(all_positions, dim=0).numpy().astype(np.float32)  # (T, N, 3)
    print(f"Trajectory shape: {trajectory.shape}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajectory, f)
    print(f"Saved inference.pkl to {save_path}")


def _start_xvfb():
    """If no DISPLAY, start an Xvfb subprocess on a free :N socket and set
    DISPLAY. Returns the subprocess handle (or None if no Xvfb was started).
    Caller is responsible for terminating it.

    qqtt/utils/visualize.py inspects DISPLAY at import time, so DISPLAY must
    be set before that module is imported.
    """
    if os.environ.get("DISPLAY"):
        return None
    if shutil.which("Xvfb") is None:
        print("WARNING: Xvfb not found on PATH; rendering will likely segfault in headless mode.")
        return None

    for n in range(99, 300):
        if not os.path.exists(f"/tmp/.X11-unix/X{n}"):
            display_num = n
            break
    else:
        print("WARNING: no free X display number found")
        return None

    proc = subprocess.Popen(
        ["Xvfb", f":{display_num}", "-screen", "0", "1280x720x24"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        if os.path.exists(f"/tmp/.X11-unix/X{display_num}"):
            break
        time.sleep(0.1)
    os.environ["DISPLAY"] = f":{display_num}"
    print(f"Started Xvfb on DISPLAY=:{display_num} (pid={proc.pid})")
    return proc


def render_inference_video(case_name, base_path, trajectory_path, video_path):
    """Render saved trajectory pkl to mp4 with RGB overlay on the source
    camera view. Mirrors inference_warp.py's output format. Uses qqtt's
    visualize_pc with the global qqtt cfg, so DISPLAY must be set before
    qqtt.utils is imported (handled by _start_xvfb in main)."""

    from qqtt.utils import cfg as qcfg, visualize_pc

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    qcfg.c2ws = np.array(c2ws)
    qcfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        meta = json.load(f)
    qcfg.intrinsics = np.array(meta["intrinsics"])
    qcfg.WH = meta["WH"]
    qcfg.overlay_path = f"{base_path}/{case_name}/color"
    qcfg.FPS = int(meta.get("fps", 30))

    with open(trajectory_path, "rb") as f:
        trajectory = pickle.load(f)  # (T_sim, N_pkl, 3) np.float32, original frame
    with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
        src = pickle.load(f)
    object_colors = np.asarray(src["object_colors"])             # (T_src, N_surf, 3)
    controller_points_np = np.asarray(src["controller_points"])  # (T_src, C, 3)

    n_surf = object_colors.shape[1]
    if trajectory.shape[1] < n_surf:
        raise ValueError(
            f"Trajectory has {trajectory.shape[1]} particles but final_data has "
            f"{n_surf} surface points; surface should be the leading prefix."
        )
    T = min(trajectory.shape[0], object_colors.shape[0], controller_points_np.shape[0])
    surface_traj = trajectory[:T, :n_surf, :]
    object_colors = object_colors[:T]
    controller_points_np = controller_points_np[:T]

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    visualize_pc(
        torch.from_numpy(surface_traj).float(),
        torch.from_numpy(object_colors).float(),
        torch.from_numpy(controller_points_np).float(),
        visualize=False,
        save_video=True,
        save_path=video_path,
    )
    print(f"Rendered inference video: {video_path}")


def main():
    parser = argparse.ArgumentParser(description="MPM Hybrid Simulator Inference -> inference.pkl + inference.mp4")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="Config YAML used during training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Default: {output_dir}/{case_name}/best_checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save inference.pkl (flat: {output_dir}/{case_name}/inference.pkl)")
    parser.add_argument("--base_path", type=str, default="./data",
                        help="Source data root containing {case_name}/{calibrate.pkl, metadata.json, color/, final_data.pkl}")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--tb_logdir", type=str, default=None,
                        help="If set, log per-frame particle mesh to TensorBoard at this dir.")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip mp4 rendering; only save inference.pkl.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = OmegaConf.load(args.config)
    cfg.mpm.device = 'cuda'

    # Resolve checkpoint path
    if args.checkpoint is None:
        ckpt_path = os.path.join(cfg.output_dir, args.case_name, "best_checkpoint.pt")
    else:
        ckpt_path = args.checkpoint

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    print(f"=== MPM Inference: {args.case_name} ===")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {ckpt_path}")

    trainer = DeformMasterMPMEngine(cfg, args.case_name)
    trainer.load_from_checkpoint(ckpt_path)

    save_path = os.path.join(args.output_dir, args.case_name, "inference.pkl")
    run_inference(trainer, save_path, tb_logdir=args.tb_logdir)

    if args.no_render:
        return

    xvfb_proc = _start_xvfb()
    try:
        video_path = os.path.join(args.output_dir, args.case_name, "inference.mp4")
        render_inference_video(args.case_name, args.base_path, save_path, video_path)
    finally:
        if xvfb_proc is not None:
            xvfb_proc.terminate()
            try:
                xvfb_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                xvfb_proc.kill()


if __name__ == "__main__":
    main()
