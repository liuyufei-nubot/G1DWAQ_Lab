# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import random

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import G1_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.envs.base.base_config import EventCfg  # Import EventCfg for extension
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

@configclass
class G1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=-2.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # Penalize hip yaw/roll deviation - use 'always' version to prevent splay-footed gait
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,  # Always penalize, not just when standing
        weight=-0.3,  # Increased weight for stronger correction
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"]
            )
        },
    )
    # Penalize ankle deviation to prevent foot splay
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,  # Always penalize ankle deviation
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,  # Always penalize arm deviation, not just when standing
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_shoulder_pitch.*", ".*_elbow.*", ".*_wrist.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*"])},
    )


from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# ========== HDR Sky Textures for Domain Randomization ==========
# These HDR textures simulate different weather/lighting conditions
HDR_SKY_TEXTURES = [
    # Clear sky - outdoor sunny
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    # Cloudy sky - overcast weather
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
    # Indoor environments
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
    # Studio lighting
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
]

# ========== MDL Terrain Materials for Domain Randomization ==========
# These MDL materials simulate different ground surface types
# Note: MDL material switching at runtime is complex; for now, randomly select one at env creation
MDL_TERRAIN_MATERIALS = [
    {  # 大理石砖
        "mdl_path": f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        "texture_scale": (0.25, 0.25),
    },
    {  # 瓦片
        "mdl_path": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        "texture_scale": (0.5, 0.5),
    },
    {  # 铝金属
        "mdl_path": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
        "texture_scale": (0.5, 0.5),
    },
]

@configclass
class G1RgbEventCfg(EventCfg):
    """Extended event configuration with lighting randomization for RGB-based RL.
    
    This configuration inherits from EventCfg and adds lighting randomization
    for domain randomization in visual reinforcement learning.
    """
    
    # Randomize dome light (sky light) - affects overall scene lighting
    randomize_dome_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="reset",  # Randomize on every episode reset
        params={
            "asset_cfg": SceneEntityCfg("sky_light"),
            "intensity_range": (300.0, 2000.0),  # Wide range for robustness
            "color_variation": 0.3,  # ±30% color variation
            "textures": HDR_SKY_TEXTURES,  # List of HDR textures to sample from
            "randomize_intensity": True,
            "randomize_color": True,
            "randomize_texture": True,  # HDR texture randomization enabled
            "default_intensity": 750.0,
            "default_color": (0.75, 0.75, 0.75),
        },
    )
    
    # Randomize distant light (sun-like directional light)
    randomize_distant_light = EventTerm(
        func=mdp.randomize_distant_light,
        mode="reset",  # Randomize on every episode reset
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "intensity_range": (1000.0, 6000.0),
            "color_variation": 0.2,
            "randomize_intensity": True,
            "randomize_color": True,
            "randomize_angle": True,  # Randomize light direction for sim2real
            "default_intensity": 3000.0,
            "default_color": (0.75, 0.75, 0.75),
        },
    )


# @configclass
# class G1FlatEnvCfg(BaseEnvCfg):

#     reward = G1RewardCfg()

#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.height_scanner.prim_body_name = "torso_link"
#         self.scene.robot = G1_CFG
#         self.scene.terrain_type = "generator"
#         self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
#         self.robot.terminate_contacts_body_names = [".*torso.*"]
#         self.robot.feet_body_names = [".*ankle_roll.*"]
#         self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]


# @configclass
# class G1FlatAgentCfg(BaseAgentCfg):
#     experiment_name: str = "g1_flat"
#     wandb_project: str = "g1_flat"


@configclass
class G1RgbEnvCfg(BaseEnvCfg):

    reward = G1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # Scene configuration
        self.scene.env_spacing = 6.0
        
        # Robot configuration
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        
        # Height scanner configuration
        # 非对称 AC: height_scan 作为特权信息只给 Critic，Actor 不使用
        # 这样 Actor 可以直接部署到真实机器人 (无需 height_scan)
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = True  # Asymmetric AC
        self.scene.height_scanner.prim_body_name = "torso_link"
        
        # ========== Privileged Information for Critic (非对称 AC) ==========
        # 这些特权信息只给 Critic 使用，帮助训练更稳定
        # Actor 不使用这些信息，可以直接部署
        self.scene.privileged_info.enable_feet_info = True          # 脚部位置/速度 (12 dim)
        self.scene.privileged_info.enable_feet_contact_force = True # 脚部接触力 (6 dim)
        self.scene.privileged_info.enable_root_height = True        # 基座高度 (1 dim)
        
        # Robot-specific settings
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        
        # ========== Lighting Randomization for RGB-based Visual RL ==========
        # Use extended EventCfg with lighting randomization for sim-to-real transfer
        self.domain_rand.events = G1RgbEventCfg()
        
        # ========== Action Delay for Sim-to-Real ==========
        # Simulate real-world communication/computation delays
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {"max_delay": 3, "min_delay": 0}  # 0-3 steps delay
        # Set robot-specific mass randomization body names (after EventCfg replacement)
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        
        # ========== MDL Terrain Material Randomization (at env creation) ==========
        # Randomly select a terrain material from MDL_TERRAIN_MATERIALS list
        # This simulates different ground surfaces (concrete, tiles, metal, etc.)
        if MDL_TERRAIN_MATERIALS:
            selected_material = random.choice(MDL_TERRAIN_MATERIALS)
            self.scene.terrain_visual_material = sim_utils.MdlFileCfg(
                mdl_path=selected_material["mdl_path"],
                project_uvw=True,
                texture_scale=selected_material["texture_scale"],
            )
            print(f"[INFO] 随机选择地形材质: {selected_material['mdl_path'].split('/')[-1]}")
        
        # Reward weights
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25
        
        # Enable RGB camera - Intel RealSense D435i specifications
        # Camera position from URDF: d435_joint on torso_link
        # URDF: xyz="0.0576235 0.01753 0.41987" rpy="0 0.8307767239493009 0" (pitch=47.6° down)
        # D435i RGB camera specs: HFOV=69.4°, VFOV=42.5°
        self.scene.rgb_camera.enable_rgb_camera = True
        self.scene.rgb_camera.prim_body_name = "torso_link"  # From URDF: parent link="torso_link"
        self.scene.rgb_camera.height = 64
        self.scene.rgb_camera.width = 64
        # Position relative to torso_link (from URDF d435_joint)
        self.scene.rgb_camera.offset.pos = (0.0576235, 0.01753, 0.41987)
        # Rotation: 47.6° pitch down (from URDF: rpy="0 0.8307767239493009 0")
        # Convert RPY to quaternion: pitch = 0.8308 rad = 47.6°
        # q = (cos(pitch/2), 0, sin(pitch/2), 0) = (cos(0.4154), 0, sin(0.4154), 0)
        # q ≈ (0.9148, 0, 0.4039, 0)
        self.scene.rgb_camera.offset.rot = (0.9148, 0.0, 0.4039, 0.0)
        self.scene.rgb_camera.offset.convention = "world"
        # D435i camera intrinsics:
        # - HFOV = 69.4°, focal_length = horizontal_aperture / (2 * tan(HFOV/2))
        # - horizontal_aperture = 20.955, focal_length = 20.955 / (2 * tan(34.7°)) ≈ 15.12
        self.scene.rgb_camera.spawn.focal_length = 15.12
        self.scene.rgb_camera.spawn.horizontal_aperture = 20.955
        # Set near clipping plane to avoid rendering robot's own body parts
        self.scene.rgb_camera.spawn.clipping_range = (0.01, 100.0)  # Near=10cm to avoid self-view
        # Camera update rate: 5 simulation steps
        # update_period = step_dt * update_interval_steps = 0.02 * 5 = 0.1s (10 Hz)
        self.scene.rgb_camera.update_interval_steps = 5
        # Enable camera pose tracking - CRITICAL for body-mounted cameras!
        # Without this, camera pose is only read at initialization and won't follow the robot
        self.scene.rgb_camera.update_latest_camera_pose = True


# @configclass
# class G1RgbAgentCfg(BaseAgentCfg):
#     experiment_name: str = "g1_rgb"
#     wandb_project: str = "g1_rgb"

#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.class_name = "ActorCriticDepth"
#         self.policy.actor_hidden_dims = [512, 256, 128]
#         self.policy.critic_hidden_dims = [512, 256, 128]
#         self.policy.rnn_hidden_size = 256
#         self.policy.rnn_num_layers = 1
#         self.policy.rnn_type = "lstm"
@configclass
class G1RgbAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticDepth",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPOMulti",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AMPOnPolicyRunnerMulti"
    experiment_name = "g1_rgb"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "rgb"
    wandb_project = "rgb"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"