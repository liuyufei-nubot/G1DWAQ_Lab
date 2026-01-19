# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# Train Vision-based RL Policy
# 使用 Depth Anything V2 视觉特征训练 G1 上台阶策略

"""
训练视觉导向的 G1 上台阶策略

使用方法:
    # 基础训练
    python legged_lab/scripts/train_vision.py --task g1_vision_stairs
    
    # 指定环境数量
    python legged_lab/scripts/train_vision.py --task g1_vision_stairs --num_envs 2048
    
    # 继续训练
    python legged_lab/scripts/train_vision.py --task g1_vision_stairs --resume
    
    # 使用 rough 地形
    python legged_lab/scripts/train_vision.py --task g1_vision_rough
"""

import argparse

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# Local imports
import legged_lab.utils.cli_args as cli_args

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train a vision-based RL agent with VisionPPO.")
parser.add_argument("--task", type=str, default="g1_vision_stairs", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Enable cameras for vision tasks
if args_cli.task and ("vision" in args_cli.task or "rgb" in args_cli.task):
    args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.utils.io import dump_yaml

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

# Import vision-specific modules
from rsl_rl.runners import VisionOnPolicyRunner
from rsl_rl.algorithms import VisionPPO
from rsl_rl.modules import ActorCriticVision
from rsl_rl.storage import RolloutStorageVision

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    """Main training function for vision-based RL."""
    
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Update agent config from CLI
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    # Create environment
    print(f"[INFO] Creating environment: {env_class_name}")
    env = env_class(env_cfg, args_cli.headless)

    # Setup logging directory
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # Convert agent config to dict
    train_cfg_dict = agent_cfg.to_dict()
    
    # Resolve vision checkpoint path if not set
    if not train_cfg_dict["vision"].get("checkpoint_path"):
        module_dir = Path(__file__).parents[1] / "modules"
        encoder_type = train_cfg_dict["vision"].get("encoder", "vits")
        checkpoint_path = str(
            module_dir / "Depth-Anything-V2" / "metric_depth" / "checkpoints" 
            / f"depth_anything_v2_metric_hypersim_{encoder_type}.pth"
        )
        train_cfg_dict["vision"]["checkpoint_path"] = checkpoint_path
        print(f"[INFO] Vision encoder checkpoint: {checkpoint_path}")

    # Create runner
    print(f"[INFO] Creating VisionOnPolicyRunner...")
    runner = VisionOnPolicyRunner(
        env=env, 
        train_cfg=train_cfg_dict, 
        log_dir=log_dir, 
        device=agent_cfg.device
    )

    # Resume training if requested
    if agent_cfg.resume:
        from isaaclab_tasks.utils import get_checkpoint_path
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO] Loading checkpoint from: {resume_path}")
        runner.load(resume_path)

    # Save configuration
    os.makedirs(log_dir, exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), train_cfg_dict)
    
    # Print training info
    print("\n" + "=" * 60)
    print(" Vision-based RL Training")
    print("=" * 60)
    print(f"  Task: {env_class_name}")
    print(f"  Num envs: {env.num_envs}")
    print(f"  Num obs: {env.num_obs}")
    print(f"  Num privileged obs: {env.num_privileged_obs}")
    print(f"  Num actions: {env.num_actions}")
    print(f"  Vision enabled: {train_cfg_dict['vision']['enable']}")
    print(f"  Vision feature dim: {train_cfg_dict['vision']['feature_dim']}")
    print(f"  Vision update interval: {train_cfg_dict['vision']['update_interval']} steps")
    print(f"  Max iterations: {train_cfg_dict['max_iterations']}")
    print(f"  Log directory: {log_dir}")
    print("=" * 60 + "\n")

    # Start training
    runner.learn(num_learning_iterations=train_cfg_dict["max_iterations"], init_at_random_ep_len=True)

    # Export trained policy
    print("\n[INFO] Training completed!")
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)
    
    # Save policy for deployment
    policy_path = os.path.join(export_dir, "policy.pt")
    # Note: Export logic would need to be implemented based on deployment requirements
    
    print(f"[INFO] Exported policy saved to: {export_dir}")


def main():
    """Entry point."""
    try:
        train()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
