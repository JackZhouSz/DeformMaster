"""Stage-3 entrypoint: RGB photometric finetune.

Parallels train_mpm.py but instantiates RGBFinetuneTrainer. The config is the
same yaml used by train_mpm.py — Stage-3-specific overrides live under a
top-level ``stage3:`` block (output_dir / tag / train.iters / residual.warmup_iters
/ rgb_finetune.*) which is flattened onto the top level after loading.

Example::

    python scripts_training_eval/rgb_refinement/train_rgb_finetune.py \
        --case_name single_lift_cloth \
        --config configs/cloth.yaml \
        --resume outputs/output_ours/cloth_warp/<scene>/final_checkpoint.pt \
        --iters 30
"""

import argparse
import os
import sys

from omegaconf import OmegaConf

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deformmaster.engine.rgb_finetune_trainer import RGBFinetuneTrainer


def main():
    parser = argparse.ArgumentParser(description="Stage-3 RGB finetune")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="configs/cloth.yaml",
                        help="Same yaml as Stage-2; must contain a stage3: block.")
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to the dynamics checkpoint (.pt or .pkl)")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if 'stage3' not in cfg:
        raise ValueError(
            f"{args.config} has no top-level `stage3:` block; "
            "RGB finetune needs that overlay."
        )
    stage3 = cfg.stage3
    del cfg.stage3
    cfg = OmegaConf.merge(cfg, stage3)
    cfg.mpm.device = 'cuda'

    per_scene = OmegaConf.select(cfg, "per_scene")
    if per_scene and args.case_name in per_scene:
        overrides = per_scene[args.case_name]
        cfg = OmegaConf.merge(cfg, overrides)
        print(f"[INFO] Applied per_scene overrides for '{args.case_name}'")

    trainer = RGBFinetuneTrainer(cfg, args.case_name, resume_path=args.resume)
    trainer.train(num_iters=args.iters)

    # Persist the RGB-loss-updated GS (base trainer ckpt doesn't store these).
    trainer.save_finetuned_gs()


if __name__ == "__main__":
    main()
