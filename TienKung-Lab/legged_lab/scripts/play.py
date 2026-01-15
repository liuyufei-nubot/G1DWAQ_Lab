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
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, AMPOnPolicyRunnerMulti, DWAQOnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--terrain", type=str, default="stairs", 
                    choices=["stairs", "stairs_slope", "flat", "rough"],
                    help="Terrain type for play: stairs (纯台阶最难), stairs_slope (台阶+斜坡), flat (平地), rough (训练地形)")
parser.add_argument("--difficulty", type=float, default=0.2,
                    help="Terrain difficulty (0.0-1.0), default=1.0 (最难)")
parser.add_argument("--lighting", type=str, default="realistic",
                    choices=["realistic", "cloudy", "evening", "bright", "default"],
                    help="Lighting preset: realistic (真实户外), cloudy (多云), evening (傍晚), bright (明亮), default (默认白色)")
parser.add_argument("--terrain_color", type=str, default="mdl_shingles",
                    choices=["concrete", "grass", "sand", "dirt", "rock", "white", "dark",
                             "mdl_marble", "mdl_shingles", "mdl_aluminum"],
                    help="Terrain color: concrete/grass/sand/dirt/rock/white/dark (简单颜色), mdl_* (真实MDL材质)")

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
from legged_lab.terrains.terrain_generator_cfg import STAIRS_ONLY_HARD_CFG, STAIRS_SLOPE_HARD_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.events.randomize_dome_light = None
    env_cfg.domain_rand.events.randomize_distant_light = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 6.0
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.debug_vis = False  # Disable velocity command arrows
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # ========== 地形选择 ==========
    if args_cli.terrain == "stairs":
        # 纯台阶地形 (最难)
        env_cfg.scene.terrain_generator = STAIRS_ONLY_HARD_CFG
        env_cfg.scene.terrain_type = "generator"
        print("[INFO] 使用纯台阶地形 (最大难度)")
    elif args_cli.terrain == "stairs_slope":
        # 台阶 + 斜坡混合地形
        env_cfg.scene.terrain_generator = STAIRS_SLOPE_HARD_CFG
        env_cfg.scene.terrain_type = "generator"
        print("[INFO] 使用台阶+斜坡混合地形 (高难度)")
    elif args_cli.terrain == "flat":
        # 平地
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.terrain_type = "plane"
        print("[INFO] 使用平地地形")
    elif args_cli.terrain == "rough":
        # 使用训练时的地形配置
        print("[INFO] 使用训练地形配置 (ROUGH_TERRAINS_CFG)")
    # 如果没有指定 --terrain，使用默认的训练地形

    # env_cfg.scene.terrain_generator = None
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        # 使用命令行指定的难度
        difficulty = args_cli.difficulty
        env_cfg.scene.terrain_generator.difficulty_range = (difficulty, difficulty)
        print(f"[INFO] 地形难度: {difficulty}")

    # ========== 光照设置 ==========
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    
    # 光照预设配置
    LIGHTING_PRESETS = {
        "realistic": {  # 真实户外 - 晴朗天空
            "dome_intensity": 1000.0,
            "dome_color": (1.0, 0.98, 0.95),  # 略带暖色
            "dome_texture": f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            "distant_intensity": 2500.0,
            "distant_color": (1.0, 0.95, 0.85),  # 阳光暖色
        },
        "cloudy": {  # 多云天气
            "dome_intensity": 1200.0,
            "dome_color": (0.9, 0.92, 0.95),  # 略带蓝灰
            "dome_texture": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
            "distant_intensity": 1500.0,
            "distant_color": (0.85, 0.88, 0.92),
        },
        "evening": {  # 傍晚
            "dome_intensity": 800.0,
            "dome_color": (1.0, 0.85, 0.7),  # 暖橙色
            "dome_texture": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
            "distant_intensity": 2000.0,
            "distant_color": (1.0, 0.7, 0.5),  # 夕阳色
        },
        "bright": {  # 明亮 (适合观察细节)
            "dome_intensity": 2000.0,
            "dome_color": (1.0, 1.0, 1.0),
            "dome_texture": f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            "distant_intensity": 3500.0,
            "distant_color": (1.0, 1.0, 1.0),
        },
        "default": {  # 默认白色 (训练时用)
            "dome_intensity": 750.0,
            "dome_color": (1.0, 1.0, 1.0),
            "dome_texture": f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            "distant_intensity": 3000.0,
            "distant_color": (0.75, 0.75, 0.75),
        },
    }
    
    lighting_preset = LIGHTING_PRESETS.get(args_cli.lighting, LIGHTING_PRESETS["realistic"])
    
    # 更新场景光照配置
    env_cfg.scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=lighting_preset["distant_color"],
            intensity=lighting_preset["distant_intensity"],
        ),
    )
    env_cfg.scene.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=lighting_preset["dome_intensity"],
            color=lighting_preset["dome_color"],
            texture_file=lighting_preset["dome_texture"],
            visible_in_primary_ray=True,  # 显示天空背景
        ),
    )
    print(f"[INFO] 光照预设: {args_cli.lighting}")

    # ========== 地形颜色设置 ==========
    # 简单颜色预设 (使用 PreviewSurfaceCfg)
    TERRAIN_COLOR_PRESETS = {
        "concrete": {  # 混凝土灰色 (真实感)
            "diffuse_color": (0.5, 0.5, 0.5),
            "roughness": 0.7,
        },
        "grass": {  # 草地绿色
            "diffuse_color": (0.2, 0.45, 0.2),
            "roughness": 0.9,
        },
        "sand": {  # 沙漠黄色
            "diffuse_color": (0.76, 0.7, 0.5),
            "roughness": 0.85,
        },
        "dirt": {  # 泥土棕色
            "diffuse_color": (0.45, 0.35, 0.25),
            "roughness": 0.9,
        },
        "rock": {  # 岩石灰色
            "diffuse_color": (0.4, 0.38, 0.35),
            "roughness": 0.8,
        },
        "white": {  # 白色 (原始)
            "diffuse_color": (0.9, 0.9, 0.9),
            "roughness": 0.5,
        },
        "dark": {  # 深色
            "diffuse_color": (0.18, 0.18, 0.18),
            "roughness": 0.6,
        },
    }
    
    # MDL 材质预设 (使用 MdlFileCfg - 更真实的材质)
    # 注意: 只使用经过验证的 MDL 路径
    MDL_TERRAIN_PRESETS = {
        "mdl_marble": {  # 大理石砖 (Isaac Lab 默认使用的，确保可用)
            "mdl_path": f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            "texture_scale": (0.25, 0.25),
        },
        "mdl_shingles": {  # 瓦片地面 (anymal_c 使用的)
            "mdl_path": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            "texture_scale": (0.5, 0.5),
        },
        "mdl_aluminum": {  # 铝金属地面 (测试文件使用的)
            "mdl_path": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
            "texture_scale": (0.5, 0.5),
        },
    }
    
    # 根据选择设置地形材质
    if args_cli.terrain_color.startswith("mdl_"):
        # 使用 MDL 材质 (更真实)
        mdl_preset = MDL_TERRAIN_PRESETS.get(args_cli.terrain_color, MDL_TERRAIN_PRESETS["mdl_marble"])
        env_cfg.scene.terrain_visual_material = sim_utils.MdlFileCfg(
            mdl_path=mdl_preset["mdl_path"],
            project_uvw=True,
            texture_scale=mdl_preset["texture_scale"],
        )
        print(f"[INFO] 地形材质: {args_cli.terrain_color} (MDL)")
    else:
        # 使用简单颜色
        terrain_color = TERRAIN_COLOR_PRESETS.get(args_cli.terrain_color, TERRAIN_COLOR_PRESETS["concrete"])
        env_cfg.scene.terrain_visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=terrain_color["diffuse_color"],
            roughness=terrain_color["roughness"],
            metallic=0.0,
        )
        print(f"[INFO] 地形颜色: {args_cli.terrain_color}")

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
    
    # Check if using DWAQ policy which requires obs_history from environment
    use_dwaq_policy = hasattr(runner.alg.policy, 'cenet_forward')
    
    # Check if RGB camera is available (from env, not runner)
    use_rgb = hasattr(env, 'rgb_camera') and env.rgb_camera is not None
    print(f"[INFO] use_depth_policy: {use_depth_policy}, use_dwaq_policy: {use_dwaq_policy}, use_rgb: {use_rgb}")
    
    if use_dwaq_policy:
        # DWAQ policy needs obs_history from environment
        runner.eval_mode()
        
        def policy_fn(obs):
            # Get obs_history from environment (DWAQ env stores it internally)
            _, obs_hist = env.get_observations()
            obs_hist = obs_hist.to(env.device)
            return runner.alg.policy.act_inference(obs, obs_hist)
        
        policy = policy_fn
    elif use_depth_policy:
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

    # Skip JIT/ONNX export for ActorCriticDepth and DWAQ (complex architectures)
    if not use_depth_policy and not use_dwaq_policy:
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
            
            # Handle different env.step() return formats
            if use_dwaq_policy:
                # DWAQ: obs, privileged_obs, prev_privileged_obs, obs_hist, rewards, dones, extras
                obs, _, _, _, _, dones, _ = env.step(actions)
            else:
                # Standard: obs, _, dones, _
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
