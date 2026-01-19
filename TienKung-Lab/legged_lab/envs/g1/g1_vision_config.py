# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# G1 Vision Environment Configuration
# 使用 Depth Anything V2 提取视觉特征，与控制解耦

import random

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import G1_CFG
from legged_lab.envs.base.base_env_config import (
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
from legged_lab.envs.base.base_config import EventCfg
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, STAIRS_TERRAINS_CFG
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


# ========== Reward Configuration (same as G1RgbEnvCfg) ==========
@configclass
class G1VisionRewardCfg(RewardCfg):
    """Reward configuration for G1 Vision environment."""
    
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
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"]
            )
        },
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
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


# ========== HDR Sky Textures for Domain Randomization ==========
HDR_SKY_TEXTURES = [
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
]

# ========== MDL Terrain Materials ==========
MDL_TERRAIN_MATERIALS = [
    {
        "mdl_path": f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        "texture_scale": (0.25, 0.25),
    },
    {
        "mdl_path": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        "texture_scale": (0.5, 0.5),
    },
    {
        "mdl_path": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
        "texture_scale": (0.5, 0.5),
    },
]


@configclass
class G1VisionEventCfg(EventCfg):
    """Event configuration with lighting randomization for vision-based RL."""
    
    randomize_dome_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sky_light"),
            "intensity_range": (300.0, 2000.0),
            "color_variation": 0.3,
            "textures": HDR_SKY_TEXTURES,
            "randomize_intensity": True,
            "randomize_color": True,
            "randomize_texture": True,
            "default_intensity": 750.0,
            "default_color": (0.75, 0.75, 0.75),
        },
    )
    
    randomize_distant_light = EventTerm(
        func=mdp.randomize_distant_light,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "intensity_range": (1000.0, 6000.0),
            "color_variation": 0.2,
            "randomize_intensity": True,
            "randomize_color": True,
            "randomize_angle": True,
            "default_intensity": 3000.0,
            "default_color": (0.75, 0.75, 0.75),
        },
    )


# ========== Vision Configuration ==========
@configclass
class VisionEncoderCfg:
    """Configuration for Depth Anything V2 vision encoder."""
    
    # Enable vision features
    enable: bool = True
    
    # Encoder type: 'vits' (24.8M), 'vitb' (97.5M), 'vitl' (335.3M)
    encoder: str = "vits"
    
    # Feature extraction settings
    feature_type: str = "cls"  # 'cls', 'avg_pool', or 'concat'
    feature_dim: int = 384  # ViT-S CLS token dimension
    
    # Feature update interval (降频更新，模拟真实部署)
    # 每 N 个 control step 更新一次视觉特征
    update_interval: int = 5
    
    # Whether to freeze the encoder (recommended for efficiency)
    freeze_encoder: bool = True
    
    # Checkpoint path (will be resolved at runtime)
    checkpoint_path: str = ""
    
    # Projection settings (done in ActorCriticVision)
    vision_latent_dim: int = 128  # Project 384 -> 128 for policy input


@configclass
class G1VisionEnvCfg(BaseEnvCfg):
    """G1 Vision Environment Configuration.
    
    使用 Depth Anything V2 提取视觉特征:
    - RGB 相机采集图像
    - DepthAnythingEncoder 提取 CLS token (384-dim)
    - VisionFeatureManager 降频更新 (每5步)
    - ActorCriticVision 融合视觉和本体特征
    
    架构设计:
    - 视觉 encoder 在 Runner 层管理，与 Policy 解耦
    - Critic 使用特权信息 (height_scan, feet_info)
    - Actor 只使用 RGB 视觉特征 + 本体感知
    """
    
    reward = G1VisionRewardCfg()
    
    # Vision encoder configuration
    vision = VisionEncoderCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
        # ========== Scene Configuration ==========
        self.scene.env_spacing = 6.0
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG  # 与g1_rgb一致
        
        # ========== Height Scanner (Critic Only - Privileged Info) ==========
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = True  # Asymmetric AC
        self.scene.height_scanner.prim_body_name = "torso_link"
        
        # ========== Privileged Information for Critic ==========
        self.scene.privileged_info.enable_feet_info = True
        self.scene.privileged_info.enable_feet_contact_force = True
        self.scene.privileged_info.enable_root_height = True
        
        # ========== Robot Configuration ==========
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        
        # ========== Domain Randomization ==========
        self.domain_rand.events = G1VisionEventCfg()
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {"max_delay": 3, "min_delay": 0}
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        
        # Terrain material randomization
        if MDL_TERRAIN_MATERIALS:
            selected_material = random.choice(MDL_TERRAIN_MATERIALS)
            self.scene.terrain_visual_material = sim_utils.MdlFileCfg(
                mdl_path=selected_material["mdl_path"],
                project_uvw=True,
                texture_scale=selected_material["texture_scale"],
            )
        
        # ========== Reward Tuning ==========
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25
        
        # ========== RGB Camera Configuration ==========
        # Intel RealSense D435i mounted on torso
        self.scene.rgb_camera.enable_rgb_camera = True
        self.scene.rgb_camera.prim_body_name = "torso_link"
        # 64x64 resolution for faster training (Depth Anything V2 will resize internally)
        self.scene.rgb_camera.height = 48
        self.scene.rgb_camera.width = 85
        # Position from URDF d435_joint
        self.scene.rgb_camera.offset.pos = (0.0576235, 0.01753, 0.41987)
        self.scene.rgb_camera.offset.rot = (0.9148, 0.0, 0.4039, 0.0)  # 47.6° pitch down
        self.scene.rgb_camera.offset.convention = "world"
        # D435i camera intrinsics
        self.scene.rgb_camera.spawn.focal_length = 15.12
        self.scene.rgb_camera.spawn.horizontal_aperture = 20.955
        self.scene.rgb_camera.spawn.clipping_range = (0.01, 100.0)
        # Camera update rate: 5 simulation steps (same as vision feature update)
        self.scene.rgb_camera.update_interval_steps = 5
        self.scene.rgb_camera.update_latest_camera_pose = True
        
        # ========== Vision Encoder Configuration ==========
        self.vision.enable = True
        self.vision.encoder = "vits"
        self.vision.feature_type = "cls"
        self.vision.feature_dim = 384
        self.vision.update_interval = 5  # 每5步更新一次视觉特征
        self.vision.freeze_encoder = True
        self.vision.vision_latent_dim = 128


@configclass
class G1VisionStairsEnvCfg(G1VisionEnvCfg):
    """G1 Vision environment with stairs terrain for stair climbing training."""
    
    def __post_init__(self):
        super().__post_init__()
        # Use stairs terrain
        self.scene.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class G1VisionRoughEnvCfg(G1VisionEnvCfg):
    """G1 Vision environment with rough terrain."""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG


# ========== Agent Configuration ==========
@configclass 
class G1VisionAgentCfg:
    """Agent configuration for VisionOnPolicyRunner.
    
    使用自定义的 VisionPPO 算法和 VisionOnPolicyRunner:
    - VisionPPO: PPO 算法，接收预提取的视觉特征
    - ActorCriticVision: 融合视觉特征的策略网络
    - RolloutStorageVision: 存储视觉特征的 rollout buffer
    """
    
    # Basic settings
    seed: int = 42
    device: str = "cuda:0"
    
    # Training settings
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    empirical_normalization: bool = False
    
    # Runner settings
    runner_class_name: str = "VisionOnPolicyRunner"
    experiment_name: str = "g1_vision"
    run_name: str = ""
    
    # Logging
    logger: str = "tensorboard"
    wandb_project: str = "g1_vision"
    neptune_project: str = "g1_vision"
    
    # Resume training
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    
    # Policy configuration (ActorCriticVision)
    policy: dict = None
    
    # Algorithm configuration (VisionPPO)
    algorithm: dict = None
    
    # Vision configuration
    vision: dict = None
    
    def __post_init__(self):
        # Policy configuration
        self.policy = {
            "class_name": "ActorCriticVision",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
            # History encoder
            "his_encoder_dims": [256, 128],
            "his_latent_dim": 64,
            # Vision feature processing
            "vision_feature_dim": 384,  # ViT-S CLS token
            "vision_latent_dim": 128,   # Projection dimension
            "use_vision_projection": True,
        }
        
        # Algorithm configuration
        self.algorithm = {
            "class_name": "VisionPPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        }
        
        # Vision configuration (for VisionOnPolicyRunner)
        self.vision = {
            "enable": True,
            "encoder": "vits",
            "feature_type": "cls",
            "feature_dim": 384,
            "update_interval": 5,
            "vision_latent_dim": 128,
            "freeze_encoder": True,
            # Checkpoint path will be resolved at runtime
            "checkpoint_path": "",
        }
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for runner."""
        return {
            "seed": self.seed,
            "device": self.device,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "save_interval": self.save_interval,
            "empirical_normalization": self.empirical_normalization,
            "policy": self.policy,
            "algorithm": self.algorithm,
            "vision": self.vision,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": self.logger,
            "wandb_project": self.wandb_project,
            "resume": self.resume,
            "load_run": self.load_run,
            "load_checkpoint": self.load_checkpoint,
        }
