# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os

import cv2
import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, AMPOnPolicyRunnerMulti

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering for tasks that require RGB/depth sensing
if args_cli.task and ("sensor" in args_cli.task or "rgb" in args_cli.task or "depth" in args_cli.task):
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 6.0
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.debug_vis = False  # Disable velocity command arrows
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # env_cfg.scene.terrain_generator = None
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner | AMPOnPolicyRunnerMulti = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    # Check if using ActorCriticDepth which requires history and optionally rgb_image
    use_depth_policy = hasattr(runner.alg.policy, 'history_encoder')
    
    # Check if RGB camera is available (from env, not runner)
    use_rgb = hasattr(env, 'rgb_camera') and env.rgb_camera is not None
    print(f"[INFO] use_depth_policy: {use_depth_policy}, use_rgb: {use_rgb}")
    
    if use_depth_policy:
        # Initialize trajectory history buffer
        # Get obs_history_len from env (preferred) or runner
        obs_history_len = getattr(env, 'obs_history_len', getattr(runner, 'obs_history_len', 1))
        num_obs = runner.num_obs
        trajectory_history = torch.zeros(
            size=(env.num_envs, obs_history_len, num_obs),
            device=env.device
        )
        
        # Set policy to eval mode
        runner.eval_mode()
        
        # Create inference function that handles history and rgb
        def policy_fn(obs):
            nonlocal trajectory_history
            normalized_obs = runner.obs_normalizer(obs) if runner.empirical_normalization else obs
            
            # Get RGB image if available
            rgb_image = None
            if use_rgb and hasattr(env, 'rgb_camera') and env.rgb_camera is not None:
                rgb_raw = env.rgb_camera.data.output["rgb"]
                if rgb_raw.shape[-1] == 4:
                    rgb_raw = rgb_raw[..., :3]
                rgb_image = rgb_raw.float().to(env.device) / 255.0
            
            actions = runner.alg.policy.act_inference(normalized_obs, trajectory_history, rgb_image=rgb_image)
            
            # Update history
            trajectory_history = torch.cat((trajectory_history[:, 1:], normalized_obs.unsqueeze(1)), dim=1)
            
            return actions
        
        policy = policy_fn
    else:
        policy = runner.get_inference_policy(device=env.device)

    # Skip JIT/ONNX export for ActorCriticDepth (complex architecture)
    if not use_depth_policy:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    
    # Reset trajectory history with initial observation if using depth policy
    if use_depth_policy:
        normalized_obs = runner.obs_normalizer(obs) if runner.empirical_normalization else obs
        trajectory_history = torch.cat((trajectory_history[:, 1:], normalized_obs.unsqueeze(1)), dim=1)

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            
            # Reset history for terminated environments
            if use_depth_policy:
                reset_env_ids = dones.nonzero(as_tuple=False).flatten()
                if len(reset_env_ids) > 0:
                    trajectory_history[reset_env_ids] = 0
            
            # Display RGB image in real-time using cv2.imshow
            if hasattr(env, 'rgb_camera') and env.rgb_camera is not None:
                try:
                    rgb_raw = env.rgb_camera.data.output["rgb"]
                    lookat_id = getattr(env, 'lookat_id', 0)
                    rgb_img = rgb_raw[lookat_id].cpu().numpy()
                    # Ensure uint8 format
                    if rgb_img.dtype != 'uint8':
                        rgb_img = (rgb_img * 255).clip(0, 255).astype('uint8')
                    # Remove alpha channel if present
                    if rgb_img.shape[-1] == 4:
                        rgb_img = rgb_img[..., :3]
                    # Convert RGB to BGR for OpenCV
                    rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    # Resize for better visibility
                    rgb_img_resized = cv2.resize(rgb_img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
                    # Display in window
                    cv2.imshow("RGB Camera View", rgb_img_resized)
                    cv2.waitKey(1)  # Required for window to update
                except Exception as e:
                    pass  # Silently ignore errors


if __name__ == "__main__":
    play()
    simulation_app.close()
