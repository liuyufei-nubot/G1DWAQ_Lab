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

"""
G1 DWAQ Environment Configuration

DWAQ (Deep Variational Autoencoder for Walking) is a method that uses β-VAE to learn
latent representations from observation history for blind walking. This configuration
sets up the G1 robot for DWAQ-based training.

Key DWAQ Components:
1. dwaq_obs_history_length: Number of observation frames for VAE encoder input
2. Blind walking: No height_scan for actor (critic_only=True)
3. Velocity supervision: Critic provides velocity info for encoder training

Architecture:
- Actor: obs + latent_code -> actions
- Encoder (VAE): obs_history -> latent_code + velocity_estimate
- Decoder: latent_code -> reconstructed_obs
- Critic: privileged_obs -> value
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from legged_lab.assets.unitree import G1_CFG
from legged_lab.envs.base.base_env_config import (
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
)
import legged_lab.mdp as mdp
from legged_lab.terrains import ROUGH_TERRAINS_CFG  # 使用与g1_rgb相同的地形


@configclass
class G1DwaqRewardCfg(RewardCfg):
    """Reward configuration for G1 DWAQ blind walking.
    
    使用与 g1_rgb 完全相同的奖励配置，确保训练稳定性。
    DWAQ 的差异仅在于 VAE encoder 和 blind walking（无 height_scan）。
    """
    # ==================== 速度追踪奖励 (增强权重，解决偷懒站立问题) ====================
    # 问题: 原来权重1.0导致机器人学会"站立不动"的偷懒策略
    # 解决: 增加速度追踪权重，让"移动"比"站着"更有价值
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5})
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
        func=mdp.body_orientation_l2, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, 
        weight=-2.0
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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", 
                           ".*_shoulder_pitch.*", ".*_elbow.*", ".*_wrist.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*"])},
    )
    
    # ==================== DWAQ Core Rewards (from DreamWaQ) ====================
    # Survival bonus - 降低存活奖励，避免"偷懒站立"策略
    # 问题: 原来weight=2.0让"站着不动"获得太多奖励
    # 解决: 降低到0.5，迫使机器人必须完成速度追踪任务
    alive = RewTerm(func=mdp.alive, weight=0.5)
    
    # ==================== 偷懒惩罚 (DWAQ 专用) ====================
    # 核心问题: 机器人学会"收到移动命令但站着不动"的偷懒策略
    # 解决方案: 直接惩罚"被命令移动但实际静止"的行为
    # - cmd_threshold=0.2: 命令速度 > 0.2 m/s 时视为"需要移动"
    # - vel_threshold=0.1: 实际速度 < 0.1 m/s 时视为"静止"
    # - weight=-2.0: 每步惩罚 -2.0，与 termination_penalty=-200 形成对比
    #   (站着20秒 = 1000步 × 2.0 = -2000，远比摔倒惩罚高)
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-2.0,
        params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    )
    
    # Gait phase matching for bipedal walking - 学习正确的两足步态
    # gait_phase_contact = RewTerm(
    #     func=mdp.gait_phase_contact,
    #     weight=0.18,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "stance_threshold": 0.55},
    # )
    
    # Swing foot height control - 控制抬腿高度
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "target_height": 0.08,
        },
    )
    
    # # Base height maintenance - 维持合适的重心高度
    # base_height = RewTerm(
    #     func=mdp.base_height,
    #     weight=-1.0,
    #     params={"target_height": 0.78},
    # )
    
    # # Penalize foot velocity when in contact (prevent sliding)
    # contact_no_vel = RewTerm(
    #     func=mdp.contact_no_vel,
    #     weight=-0.4,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )
    



@configclass
class G1DwaqEnvCfg(BaseEnvCfg):
    """
    G1 DWAQ environment configuration for blind walking with VAE.
    
    This configuration enables DWAQ-based training where:
    - Actor: Uses current observation + latent code from encoder
    - Encoder: Learns to extract hidden state from observation history
    - Decoder: Reconstructs observations for VAE loss
    - Critic: Uses privileged information for value estimation
    
    The encoder learns to predict velocity and other hidden states from
    observation history, enabling blind walking on rough terrain.
    """
    
    # Use DWAQ-specific reward configuration
    reward = G1DwaqRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        
        # Robot configuration
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "generator"
        # 使用与 g1_rgb 相同的地形配置，排除地形差异的影响
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        
        # Height scanner - asymmetric AC (actor is blind, critic has terrain info)
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = True  # Actor is blind
        
        # Privileged information for Critic
        self.scene.privileged_info.enable_feet_info = True  # feet_pos + feet_vel (12 dim)
        self.scene.privileged_info.enable_feet_contact_force = True  # contact force (6 dim)
        self.scene.privileged_info.enable_root_height = True  # root height (1 dim)
        
        # DWAQ-specific: Observation history for VAE encoder
        # The encoder uses these to infer hidden environment states
        # Original DreamWaQ uses 5 frames (not 10)
        self.robot.dwaq_obs_history_length = 5  # 5 frames of observation history
        
        # Standard history for AC (can be 1 since encoder handles temporal info)
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        
        # Enable gait phase for periodic walking pattern
        self.robot.gait_phase.enable = False
        self.robot.gait_phase.period = 0.8   # 0.8s gait cycle
        self.robot.gait_phase.offset = 0.5   # 50% offset = alternating gait
        
        # Action delay for sim-to-real transfer
        self.domain_rand.action_delay.enable = False  # 暂时禁用，待训练成功后再开启
        
        # ========== 域随机化配置 ==========
        # 暂时禁用额外的域随机化，先确保DWAQ在与g1_rgb相同的条件下能训练成功
        # 待训练成功后再逐步开启
        
        # 摩擦力范围 - 使用与g1_rgb相同的默认值
        # self.domain_rand.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        # self.domain_rand.events.physics_material.params["dynamic_friction_range"] = (0.15, 1.0)
        
        # 质心偏移随机化 - 暂时禁用
        # self.domain_rand.events.randomize_com = EventTerm(...)
        
        # 执行器增益随机化 - 暂时禁用
        # self.domain_rand.events.randomize_actuator_gains = EventTerm(...)


@configclass  
class G1DwaqAgentCfg(BaseAgentCfg):
    """
    G1 DWAQ agent configuration.
    
    Uses DWAQOnPolicyRunner with ActorCritic_DWAQ policy.
    The policy includes:
    - VAE encoder for latent state extraction from observation history
    - Actor network using observation + latent code
    - Critic network using privileged observations
    - Decoder for observation reconstruction (VAE loss)
    
    Training adds autoencoder loss on top of standard PPO loss:
    - Velocity estimation loss: MSE between predicted and true velocity
    - Reconstruction loss: MSE between decoded and actual observations  
    - KL divergence: β-VAE regularization on latent space
    
    DWAQ Architecture Parameters:
    - cenet_out_dim: Output dimension of encoder = velocity_dim(3) + latent_dim(16) = 19
    - The encoder predicts velocity (3 dims) and latent representation (16 dims)
    - The latent code is concatenated with observations for actor input
    """
    experiment_name: str = "g1_dwaq"
    wandb_project: str = "g1_dwaq"
    runner_class_name: str = "DWAQOnPolicyRunner"

    def __post_init__(self):
        super().__post_init__()
        
        # Use ActorCritic_DWAQ policy with VAE encoder
        self.policy.class_name = "ActorCritic_DWAQ"
        self.policy.init_noise_std = 1.0  # 增强早期探索（原版 DreamWaQ 默认值）
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        
        # DWAQ encoder output dimension: velocity(3) + latent(16) = 19
        # This is the key DWAQ architecture parameter
        self.policy.cenet_out_dim = 19
        
        # Use DWAQPPO algorithm (PPO + autoencoder loss)
        self.algorithm.class_name = "DWAQPPO"
        # Match original DreamWaQ entropy coefficient (default is 0.005)
        self.algorithm.entropy_coef = 0.01
