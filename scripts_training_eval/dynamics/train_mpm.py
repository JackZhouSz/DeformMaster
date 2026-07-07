import os
import sys
import argparse

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import OmegaConf
from deformmaster.engine.trainer_mpm import PhysExpertMPMTrainer

def main():
    parser = argparse.ArgumentParser(description="PhysExpert Stage 1: MPM Parameter Training")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/cloth.yaml")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    cfg.mpm.device = 'cuda'

    # Apply per-scene overrides if defined in config
    per_scene = OmegaConf.select(cfg, "per_scene")
    if per_scene and args.case_name in per_scene:
        overrides = per_scene[args.case_name]
        cfg = OmegaConf.merge(cfg, overrides)
        print(f"[INFO] Applied per_scene overrides for '{args.case_name}': {OmegaConf.to_container(overrides)}")

    trainer = PhysExpertMPMTrainer(cfg, args.case_name)
    trainer.train(num_iters=args.iters)

if __name__ == "__main__":
    main()
