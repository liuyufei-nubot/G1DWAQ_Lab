# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
G1 RGB Sim2Sim 脚本
====================

将在 Isaac Lab 中训练的 G1 机器人 g1_rgb 视觉策略迁移到 MuJoCo 仿真环境中运行。

网络架构 (ActorCriticDepth):
---------------------------
- RGB 图像 (64x64x3) -> RGBEncoder -> 128 dim
- 历史观测 (96 x T) -> HistoryEncoder -> 67 dim  
- 当前观测 (96 dim)
- Actor 输入: concat(obs, his_feature, rgb_feature) = 96 + 67 + 128 = 291 dim
- Actor 输出: 29 dim (actions)

观测结构 (96 维):
-----------------
- ang_vel (3): 角速度 (body frame)
- projected_gravity (3): 投影重力
- command (3): 速度命令 [vx, vy, yaw_rate]
- joint_pos (29): 关节位置偏差 (当前 - 默认)
- joint_vel (29): 关节速度
- action (29): 上一步动作

使用方法：
---------
python legged_lab/scripts/sim2sim_g1_rgb.py --checkpoint <model.pt>

键盘控制 (小键盘)：
------------------
- 8/2: 前进/后退
- 4/6: 左移/右移  
- 7/9: 左转/右转
- 5: 停止
"""

import argparse
import os
import sys
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pynput import keyboard


# ==================== 关节顺序定义 ====================
# MuJoCo XML 中的关节顺序 (29 DOF)
MUJOCO_DOF_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

# Isaac Lab 中的关节顺序 (按 URDF actuator 定义)
LAB_DOF_NAMES = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint'
]


# ==================== 网络定义 ====================
# 直接从 rsl_rl/modules/actor_critic_depth.py 复制，确保兼容性

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class RGBEncoder(nn.Module):
    """Simple RGB encoder without temporal stacking."""
    def __init__(self, output_dim=128, input_height=64, input_width=64) -> None:
        super().__init__()
        activation = nn.ELU()
        
        self.backbone = nn.Sequential(
            # [3, H, W]
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_height, input_width)
            dummy_output = self.backbone(dummy_input)
            flatten_size = dummy_output.shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 128),
            activation,
            nn.Linear(128, output_dim),
            activation
        )
        self.output_dim = output_dim
        
    def forward(self, rgb_image):
        # rgb_image: [batch, H, W, 3] -> need to permute to [batch, 3, H, W]
        if rgb_image.dim() == 4 and rgb_image.shape[-1] == 3:
            rgb_image = rgb_image.permute(0, 3, 1, 2)  # [B, H, W, 3] -> [B, 3, H, W]
        features = self.backbone(rgb_image)
        latent = self.fc(features)
        return latent


class ActorCriticDepth(nn.Module):
    """Actor-Critic network with RGB image encoder - Sim2Sim 版本 (只包含 Actor)"""
    is_recurrent = False
    
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        his_encoder_dims=[1024, 512, 128],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        his_latent_dim=64 + 3,
                        history_dim=570,
                        rgb_latent_dim=128,
                        rgb_height=64,
                        rgb_width=64,
                        use_rgb=True,
                        activation='elu',
                        init_noise_std=1.0,
                        max_grad_norm=10.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticDepth.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDepth, self).__init__()
        activation = get_activation(activation)

        self.his_latent_dim = his_latent_dim
        self.rgb_latent_dim = rgb_latent_dim
        self.use_rgb = use_rgb
        self.max_grad_norm = max_grad_norm        

        # RGB encoder (only if using RGB)
        if self.use_rgb:
            self.rgb_encoder = RGBEncoder(output_dim=rgb_latent_dim, input_height=rgb_height, input_width=rgb_width)
            mlp_input_dim_a = num_actor_obs + his_latent_dim + rgb_latent_dim
        else:
            self.rgb_encoder = None
            mlp_input_dim_a = num_actor_obs + his_latent_dim
        
        # History Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, his_encoder_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(his_encoder_dims)):
            if l == len(his_encoder_dims) - 1:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_latent_dim))
            else:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_encoder_dims[l + 1]))
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)
        
        # Policy (Actor)
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print(f"Actor MLP: {self.actor}")

    def act_inference(self, observations, history, rgb_image=None, **kwargs):
        """推理模式前向传播"""
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        if rgb_image is not None and self.use_rgb:
            rgb_feature = self.rgb_encoder(rgb_image)
            actor_input = torch.cat((observations, his_feature, rgb_feature), dim=-1)
        else:
            actor_input = torch.cat((observations, his_feature), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean


class G1RgbSim2SimCfg:
    """G1 RGB Sim2Sim 配置类 - 与训练配置保持一致"""

    class sim:
        sim_duration = 100.0
        num_actions = 29
        num_obs_per_step = 96      # 3+3+3+29+29+29 = 96 (无 gait phase)
        # 注意: g1_rgb 的 actor_obs_history_length = 1 (由 env 配置)
        # 但 history encoder 的 history_dim 由 runner 中的 obs_history_len * num_obs 决定
        # g1_rgb env 设置 obs_history_len = 1，所以 history_dim = 96
        actor_obs_history_length = 1  # 与 g1_rgb_config.py 保持一致
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        # G1 初始高度
        init_height = 0.793

    class camera:
        # RGB 相机配置 (与 g1_rgb_config.py 一致)
        width = 64
        height = 64
        # 相机在 torso_link 上的位置 (来自 URDF d435_joint)
        pos = (0.0576235, 0.01753, 0.41987)
        # 相机朝向: 47.6° pitch down (来自 URDF)
        # MuJoCo 使用 euler angles (roll, pitch, yaw)
        euler = (0.0, 0.8308, 0.0)  # pitch = 47.6° ≈ 0.8308 rad
        fovy = 42.5  # D435i VFOV
        # 更新间隔 (相对于控制步)
        update_interval = 5


class G1RgbMujocoRunner:
    """
    G1 RGB Sim2Sim 运行器
    
    加载 ActorCriticDepth 策略和 MuJoCo 模型，运行实时仿真控制。
    """

    def __init__(self, cfg: G1RgbSim2SimCfg, checkpoint_path: str, model_path: str, show_camera: bool = True):
        self.cfg = cfg
        self.device = torch.device("cpu")  # MuJoCo sim2sim 用 CPU
        self.show_camera = show_camera
        
        # 加载 MuJoCo 模型 (带 RGB 相机)
        print(f"[INFO] 加载 MuJoCo 模型: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        
        # 初始化 RGB 相机
        self.init_rgb_camera()
        
        # 加载策略
        print(f"[INFO] 加载 checkpoint: {checkpoint_path}")
        self.load_policy(checkpoint_path)
        
        # 初始化变量
        self.init_variables()
        self.build_joint_mappings()
        self.set_initial_pose()
        
        print(f"[INFO] 控制频率: {1.0 / (cfg.sim.dt * cfg.sim.decimation):.1f} Hz")
        print(f"[INFO] 观测维度: {cfg.sim.num_obs_per_step}")
        print(f"[INFO] 历史长度: {cfg.sim.actor_obs_history_length}")
        print(f"[INFO] RGB 相机: {cfg.camera.width}x{cfg.camera.height}")

    def init_rgb_camera(self) -> None:
        """初始化 MuJoCo RGB 相机渲染器"""
        # 查找相机 ID
        try:
            self.rgb_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_cam")
            print(f"[INFO] 找到相机: rgb_cam (id={self.rgb_cam_id})")
        except Exception:
            print("[WARN] 未找到 'rgb_cam' 相机，将使用第一个相机")
            self.rgb_cam_id = 0
        
        # 创建 RGB 渲染器
        self.rgb_renderer = mujoco.Renderer(
            self.model, 
            width=self.cfg.camera.width, 
            height=self.cfg.camera.height
        )
        
        # 初始化 RGB 图像缓冲
        self.rgb_image = np.zeros((self.cfg.camera.height, self.cfg.camera.width, 3), dtype=np.float32)
        self.camera_update_counter = 0

    def load_policy(self, checkpoint_path: str) -> None:
        """加载 ActorCriticDepth 策略"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 获取模型参数 (从 checkpoint 或使用默认值)
        model_cfg = checkpoint.get('model_cfg', {})
        
        # 构建网络参数
        num_actor_obs = self.cfg.sim.num_obs_per_step
        
        # g1_rgb: obs_history_len = 1，所以 history_dim = 96 * 1 = 96
        obs_history_len = self.cfg.sim.actor_obs_history_length
        history_dim = num_actor_obs * obs_history_len
        
        # 从 checkpoint 推断参数，或使用默认值 (与 g1_rgb_config.py 一致)
        policy_params = {
            'num_actor_obs': num_actor_obs,
            'num_critic_obs': model_cfg.get('num_critic_obs', 307),  # g1_rgb 默认
            'num_actions': self.cfg.sim.num_actions,
            'his_encoder_dims': model_cfg.get('his_encoder_dims', [1024, 512, 128]),
            'actor_hidden_dims': model_cfg.get('actor_hidden_dims', [512, 256, 128]),
            'critic_hidden_dims': model_cfg.get('critic_hidden_dims', [512, 256, 128]),
            'his_latent_dim': model_cfg.get('his_latent_dim', 67),  # 64 + 3
            'history_dim': history_dim,  # 96 for g1_rgb (obs_history_len=1)
            'rgb_latent_dim': model_cfg.get('rgb_latent_dim', 128),
            'rgb_height': self.cfg.camera.height,
            'rgb_width': self.cfg.camera.width,
            'use_rgb': True,
            'activation': model_cfg.get('activation', 'elu'),
        }
        
        print(f"[INFO] 策略参数: obs_history_len={obs_history_len}, history_dim={history_dim}")
        print(f"[INFO] his_latent_dim={policy_params['his_latent_dim']}, rgb_latent_dim={policy_params['rgb_latent_dim']}")
        
        # 创建策略网络
        self.policy = ActorCriticDepth(**policy_params)
        
        # 加载权重
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 只加载 Actor 相关的权重
        actor_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('actor') or key.startswith('history_encoder') or key.startswith('rgb_encoder'):
                actor_state_dict[key] = value
        
        # 尝试加载权重
        missing, unexpected = self.policy.load_state_dict(actor_state_dict, strict=False)
        if missing:
            print(f"[WARN] 缺少的权重: {missing}")
        if unexpected:
            print(f"[WARN] 多余的权重: {unexpected}")
        
        self.policy.eval()
        print(f"[INFO] 策略加载完成")
        
        # 加载 normalizer (如果有)
        self.obs_normalizer = None
        if 'obs_norm_mean' in checkpoint and 'obs_norm_var' in checkpoint:
            self.obs_mean = checkpoint['obs_norm_mean'].cpu().numpy()
            self.obs_var = checkpoint['obs_norm_var'].cpu().numpy()
            print(f"[INFO] 加载了观测归一化参数")
        else:
            self.obs_mean = None
            self.obs_var = None

    def init_variables(self) -> None:
        """初始化仿真变量"""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_actions = self.cfg.sim.num_actions
        
        # 关节状态
        self.dof_pos = np.zeros(self.num_actions)
        self.dof_vel = np.zeros(self.num_actions)
        
        # 动作 (Isaac Lab 顺序)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        
        # 默认关节位置 (MuJoCo 顺序) - 与 G1_CFG init_state.joint_pos 一致
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,    # 左腿
            -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,    # 右腿
            0.0, 0.0, 0.0,                        # 腰部
            0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0, # 左臂
            0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0 # 右臂
        ], dtype=np.float32)
        
        # PD 增益 (MuJoCo 顺序)
        self.kps = np.array([
            200, 150, 150, 200, 20, 20,    # 左腿
            200, 150, 150, 200, 20, 20,    # 右腿
            200, 200, 200,                  # 腰部
            100, 100, 50, 50, 40, 40, 40,  # 左臂
            100, 100, 50, 50, 40, 40, 40   # 右臂
        ], dtype=np.float32)
        
        self.kds = np.array([
            5, 5, 5, 5, 2, 2,              # 左腿
            5, 5, 5, 5, 2, 2,              # 右腿
            5, 5, 5,                        # 腰部
            2, 2, 2, 2, 2, 2, 2,           # 左臂
            2, 2, 2, 2, 2, 2, 2            # 右臂
        ], dtype=np.float32)
        
        self.episode_length_buf = 0
        
        # 速度命令
        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # 观测历史缓冲 (用于 history encoder)
        self.trajectory_history = torch.zeros(
            1, self.cfg.sim.actor_obs_history_length, self.cfg.sim.num_obs_per_step,
            dtype=torch.float32, device=self.device
        )

    def build_joint_mappings(self) -> None:
        """建立关节映射索引"""
        # MuJoCo -> Isaac Lab 索引映射
        mujoco_indices = {name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)}
        self.mujoco_to_isaac_idx = [mujoco_indices[name] for name in LAB_DOF_NAMES]
        
        # Isaac Lab -> MuJoCo 索引映射
        lab_indices = {name: idx for idx, name in enumerate(LAB_DOF_NAMES)}
        self.isaac_to_mujoco_idx = [lab_indices[name] for name in MUJOCO_DOF_NAMES]
        
        # 默认关节位置转换为 Isaac Lab 顺序
        self.default_dof_pos_isaac = self.default_dof_pos[self.mujoco_to_isaac_idx]
        
        print(f"[INFO] 关节映射建立完成")

    def set_initial_pose(self) -> None:
        """设置初始姿态"""
        # 基座位置
        self.data.qpos[0:3] = [0.0, 0.0, self.cfg.robot.init_height]
        self.data.qpos[3:7] = [1, 0, 0, 0]  # (w, x, y, z)
        
        # 关节位置
        self.data.qpos[7:7 + self.num_actions] = self.default_dof_pos.copy()
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        print(f"[INFO] 初始高度: {self.data.qpos[2]:.3f}m")

    def get_gravity_orientation(self, quat: np.ndarray) -> np.ndarray:
        """计算投影重力向量"""
        qw, qx, qy, qz = quat
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def mj29_to_lab29(self, array_mj: np.ndarray) -> np.ndarray:
        """将 MuJoCo 顺序数组转换为 Isaac Lab 顺序"""
        return array_mj[self.mujoco_to_isaac_idx]

    def lab29_to_mj29(self, array_lab: np.ndarray) -> np.ndarray:
        """将 Isaac Lab 顺序数组转换为 MuJoCo 顺序"""
        return array_lab[self.isaac_to_mujoco_idx]

    def get_obs(self) -> torch.Tensor:
        """
        计算当前观测向量 (96 维)
        
        Returns:
            当前观测 tensor
        """
        # 读取关节状态 (MuJoCo 顺序)
        dof_pos_mj = self.data.qpos[7:7 + self.num_actions].copy()
        dof_vel_mj = self.data.qvel[6:6 + self.num_actions].copy()
        
        # 基座状态
        ang_vel_body = self.data.qvel[3:6].copy()
        
        # 投影重力
        quat = self.data.qpos[3:7].copy()  # MuJoCo: (w, x, y, z)
        projected_gravity = self.get_gravity_orientation(quat)
        
        # 转换为 Isaac Lab 顺序
        joint_pos_isaac = self.mj29_to_lab29(dof_pos_mj - self.default_dof_pos)
        joint_vel_isaac = self.mj29_to_lab29(dof_vel_mj)
        
        # 构建观测 (96 维)
        obs = np.concatenate([
            ang_vel_body,
            projected_gravity,
            self.command_vel,
            joint_pos_isaac,
            joint_vel_isaac,
            np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),
        ], axis=0).astype(np.float32)
        
        # 归一化 (如果有)
        if self.obs_mean is not None and self.obs_var is not None:
            obs = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        
        # 裁剪
        obs = np.clip(obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)
        
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def update_rgb_camera(self) -> torch.Tensor:
        """更新 RGB 相机图像"""
        # 渲染 RGB 图像
        self.rgb_renderer.update_scene(self.data, camera=self.rgb_cam_id)
        rgb_raw = self.rgb_renderer.render()
        
        # 转换格式: uint8 [0, 255] -> float32 [0, 1]
        rgb_image = rgb_raw.astype(np.float32) / 255.0
        
        # 显示图像 (调试用)
        if self.show_camera:
            cv2.imshow("RGB Camera", cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        
        # 转换为 tensor: [H, W, 3] -> [1, H, W, 3]
        rgb_tensor = torch.tensor(rgb_image, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        return rgb_tensor

    def position_control(self) -> np.ndarray:
        """计算目标关节位置 (MuJoCo 顺序)"""
        actions_scaled = self.action * self.cfg.sim.action_scale
        return self.lab29_to_mj29(actions_scaled) + self.default_dof_pos

    def pd_control(self, target_q: np.ndarray) -> np.ndarray:
        """PD 控制器计算力矩"""
        q = self.data.qpos[7:7 + self.num_actions]
        dq = self.data.qvel[6:6 + self.num_actions]
        return (target_q - q) * self.kps + (0 - dq) * self.kds

    def run(self) -> None:
        """运行仿真循环"""
        self.setup_keyboard_listener()
        self.listener.start()
        
        # 创建 viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # 初始化观测历史
        for _ in range(self.cfg.sim.actor_obs_history_length):
            obs = self.get_obs()
            self.trajectory_history = torch.cat(
                (self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1
            )
        
        # 初始化 RGB 图像
        rgb_image = self.update_rgb_camera()
        
        print("\n[INFO] 键盘控制: 8/2=前后, 4/6=左右, 7/9=转向, 5=停止")
        print("[INFO] 按 Ctrl+C 退出\n")
        
        debug_counter = 0
        control_counter = 0
        
        try:
            while self.viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                # 获取当前观测
                obs = self.get_obs()
                
                # 更新观测历史
                self.trajectory_history = torch.cat(
                    (self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1
                )
                
                # 更新 RGB 相机 (按配置的更新间隔)
                control_counter += 1
                if control_counter % self.cfg.camera.update_interval == 0:
                    rgb_image = self.update_rgb_camera()
                
                # 策略推理
                with torch.no_grad():
                    actions = self.policy.act_inference(
                        obs, 
                        self.trajectory_history, 
                        rgb_image=rgb_image
                    )
                
                self.action[:] = actions.cpu().numpy().flatten()[:self.num_actions]
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                
                # 调试输出
                debug_counter += 1
                if debug_counter <= 3:
                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  obs shape: {obs.shape}")
                    print(f"  history shape: {self.trajectory_history.shape}")
                    print(f"  rgb shape: {rgb_image.shape}")
                    print(f"  command: {self.command_vel}")
                    print(f"  action (first 6): {self.action[:6]}")
                
                # 执行 decimation 步
                for _ in range(self.cfg.sim.decimation):
                    step_start = time.time()
                    
                    # PD 控制
                    target_pos = self.position_control()
                    tau = self.pd_control(target_pos)
                    self.data.ctrl[:self.num_actions] = tau
                    
                    # 物理步进
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.sync()
                    
                    # 时间控制
                    elapsed = time.time() - step_start
                    if self.cfg.sim.dt - elapsed > 0:
                        time.sleep(self.cfg.sim.dt - elapsed)
                
                self.episode_length_buf += 1
                
                # 定期打印状态
                if self.episode_length_buf % 100 == 0:
                    print(f"[INFO] t={self.data.time:.1f}s, cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, {self.command_vel[2]:.2f}], h={self.data.qpos[2]:.3f}m")
                    
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断")
        finally:
            self.listener.stop()
            self.viewer.close()
            if self.show_camera:
                cv2.destroyAllWindows()
            print("[INFO] 仿真结束")

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """调整速度命令"""
        limits = [1.0, 0.5, 1.57]
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -limits[idx], limits[idx])

    def setup_keyboard_listener(self) -> None:
        """设置键盘监听器"""
        def on_press(key):
            try:
                if key.char == "8": self.adjust_command_vel(0, 0.2)
                elif key.char == "2": self.adjust_command_vel(0, -0.2)
                elif key.char == "4": self.adjust_command_vel(1, 0.2)
                elif key.char == "6": self.adjust_command_vel(1, -0.2)
                elif key.char == "7": self.adjust_command_vel(2, 0.3)
                elif key.char == "9": self.adjust_command_vel(2, -0.3)
                elif key.char == "5": self.command_vel[:] = 0.0
            except AttributeError:
                pass
        
        self.listener = keyboard.Listener(on_press=on_press)


def create_rgb_camera_scene(base_xml_path: str, output_path: str, camera_cfg) -> str:
    """
    创建带 RGB 相机的场景文件
    
    策略:
    1. 如果 g1_29dof_rev_1_0_with_cam.xml 已存在，直接使用
    2. 如果场景使用 <include>，修改引用到 _with_cam.xml
    3. 否则创建带相机的文件
    """
    import xml.etree.ElementTree as ET
    import math
    
    base_dir = os.path.dirname(base_xml_path)
    
    # 预定义的带相机机器人文件
    robot_with_cam_path = os.path.join(base_dir, "g1_29dof_rev_1_0_with_cam.xml")
    robot_with_cam_name = "g1_29dof_rev_1_0_with_cam.xml"
    
    # 检查带相机的机器人文件是否已存在
    robot_with_cam_exists = os.path.exists(robot_with_cam_path)
    if robot_with_cam_exists:
        try:
            cam_tree = ET.parse(robot_with_cam_path)
            if cam_tree.getroot().find(".//camera[@name='rgb_cam']") is None:
                robot_with_cam_exists = False  # 文件存在但没有相机
        except:
            robot_with_cam_exists = False
    
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    
    # 检查是否使用了 <include> 标签
    include_elem = root.find(".//include")
    
    if include_elem is not None:
        # 场景文件使用 <include> 引用机器人
        if robot_with_cam_exists:
            # 直接使用已有的带相机机器人文件
            include_elem.set('file', robot_with_cam_name)
            tree.write(output_path)
            print(f"[INFO] 使用已有机器人文件: {robot_with_cam_path}")
            return output_path
        else:
            # 需要创建带相机的机器人文件
            include_file = include_elem.get('file')
            robot_xml_path = os.path.join(base_dir, include_file)
            
            if not os.path.exists(robot_xml_path):
                print(f"[ERROR] 机器人文件不存在: {robot_xml_path}")
                return base_xml_path
            
            robot_tree = ET.parse(robot_xml_path)
            robot_root = robot_tree.getroot()
            
            torso_link = robot_root.find(".//body[@name='torso_link']")
            if torso_link is not None:
                _add_camera_to_body(torso_link, camera_cfg)
                robot_tree.write(robot_with_cam_path)
                print(f"[INFO] 创建带相机的机器人文件: {robot_with_cam_path}")
            else:
                print("[WARN] 未找到 torso_link")
                return base_xml_path
            
            include_elem.set('file', robot_with_cam_name)
            tree.write(output_path)
            print(f"[INFO] 创建场景文件: {output_path}")
            return output_path
    
    else:
        # 直接包含机器人定义的 XML (如 g1_29dof_rev_1_0_daf.xml)
        existing_cams = root.findall(".//camera[@name='rgb_cam']")
        if existing_cams:
            print("[INFO] 场景中已存在 rgb_cam")
            return base_xml_path
        
        torso_link = root.find(".//body[@name='torso_link']")
        if torso_link is not None:
            _add_camera_to_body(torso_link, camera_cfg)
            tree.write(output_path)
            print(f"[INFO] 创建带相机的场景: {output_path}")
            return output_path
        else:
            print("[WARN] 未找到 torso_link")
            return base_xml_path


def _add_camera_to_body(body_elem, camera_cfg):
    """向 body 元素添加 RGB 相机"""
    import xml.etree.ElementTree as ET
    import math
    
    camera = ET.SubElement(body_elem, 'camera')
    camera.set('name', 'rgb_cam')
    camera.set('mode', 'fixed')
    
    # 位置
    pos_str = f"{camera_cfg.pos[0]} {camera_cfg.pos[1]} {camera_cfg.pos[2]}"
    camera.set('pos', pos_str)
    
    # 姿态计算
    # MuJoCo 相机: 看向 -Z, Y 是图像上方, X 是图像右方
    # 目标: 看向前下方 47.6°
    pitch_rad = camera_cfg.euler[1]
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    
    # X = (0, -1, 0) 图像右方 = 机器人右方
    # Y = (sin_p, 0, cos_p) 图像上方 = 前上方
    xyaxes_str = f"0 -1 0 {sin_p} 0 {cos_p}"
    camera.set('xyaxes', xyaxes_str)
    camera.set('fovy', str(camera_cfg.fovy))
    
    print(f"[INFO] 相机位置 (相对 torso_link): [{pos_str}]")
    print(f"[INFO] 相机姿态 (xyaxes): [{xyaxes_str}]")
    print(f"[INFO] 相机 pitch: {math.degrees(pitch_rad):.1f}°")


def get_available_scenes(mjcf_dir: str) -> dict:
    """获取可用的场景文件
    
    约定：场景文件以 _scene.xml 结尾，或者是特殊名称如 stairs.xml
    """
    scenes = {}
    
    # 搜索 mjcf 目录下的场景文件
    if os.path.isdir(mjcf_dir):
        for f in os.listdir(mjcf_dir):
            if f.endswith('.xml'):
                # 特殊场景文件
                if f in ['scene.xml', 'stairs.xml', 'flat_scene.xml', 'rough_scene.xml', 'slope_scene.xml']:
                    name = f.replace('.xml', '').replace('_scene', '')
                    scenes[name] = os.path.join(mjcf_dir, f)
                # 通用 *_scene.xml 文件
                elif f.endswith('_scene.xml'):
                    name = f.replace('_scene.xml', '')
                    scenes[name] = os.path.join(mjcf_dir, f)
    
    return scenes


def main():
    LEGGED_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    MJCF_DIR = os.path.join(LEGGED_LAB_ROOT, "legged_lab/assets/unitree/g1/mjcf")
    
    # 默认路径
    default_checkpoint = os.path.join(LEGGED_LAB_ROOT, "logs/g1_rgb/2026-01-06_14-21-16/model_30000.pt")
    # default_checkpoint = os.path.join(LEGGED_LAB_ROOT, "logs/g1_rgb/2026-01-04_14-46-50/model_47000.pt")
    default_model = os.path.join(MJCF_DIR, "g1_29dof_rev_1_0_daf.xml")
    
    # 获取可用场景
    available_scenes = get_available_scenes(MJCF_DIR)
    scene_names = list(available_scenes.keys())
    
    parser = argparse.ArgumentParser(
        description="G1 RGB Sim2Sim - ActorCriticDepth 视觉策略",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, 
                        help="模型 checkpoint 路径 (model_xxxx.pt)")
    parser.add_argument("--model", type=str, default=default_model,
                        help="MuJoCo XML 模型路径 (当未指定 --scene 时使用)")
    parser.add_argument("--scene", type=str, default=None,
                        choices=scene_names if scene_names else None,
                        help=f"场景名称，可用场景: {', '.join(scene_names) if scene_names else '无'}\n"
                             f"场景文件位于: {MJCF_DIR}")
    parser.add_argument("--scene-file", type=str, default=None,
                        help="直接指定场景文件路径 (优先级高于 --scene)")
    parser.add_argument("--duration", type=float, default=100.0, help="仿真时长 (秒)")
    parser.add_argument("--no-display", action="store_true", help="不显示 RGB 相机窗口")
    parser.add_argument("--list-scenes", action="store_true", help="列出所有可用场景")
    args = parser.parse_args()
    
    # 列出可用场景
    if args.list_scenes:
        print("\n可用场景:")
        print("-" * 40)
        for name, path in available_scenes.items():
            print(f"  {name:15} -> {os.path.basename(path)}")
        print("-" * 40)
        print(f"场景文件目录: {MJCF_DIR}")
        print("\n使用示例:")
        print(f"  python {sys.argv[0]} --scene stairs")
        print(f"  python {sys.argv[0]} --scene-file /path/to/custom_scene.xml")
        sys.exit(0)
    
    # 检查 checkpoint 文件
    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint 不存在: {args.checkpoint}")
        sys.exit(1)
    
    # 确定要加载的模型/场景文件
    if args.scene_file:
        # 直接指定场景文件
        base_model_path = args.scene_file
        if not os.path.isfile(base_model_path):
            print(f"[ERROR] 场景文件不存在: {base_model_path}")
            sys.exit(1)
    elif args.scene:
        # 使用预定义场景
        if args.scene not in available_scenes:
            print(f"[ERROR] 未知场景: {args.scene}")
            print(f"[INFO] 可用场景: {', '.join(scene_names)}")
            sys.exit(1)
        base_model_path = available_scenes[args.scene]
    else:
        # 使用默认模型
        base_model_path = args.model
        if not os.path.isfile(base_model_path):
            print(f"[ERROR] MuJoCo 模型不存在: {base_model_path}")
            sys.exit(1)
    
    # 配置
    cfg = G1RgbSim2SimCfg()
    cfg.sim.sim_duration = args.duration
    
    # 创建带相机的场景文件
    output_scene = os.path.join(MJCF_DIR, "g1_rgb_scene.xml")
    model_path = create_rgb_camera_scene(base_model_path, output_scene, cfg.camera)
    
    print("=" * 60)
    print("G1 RGB Sim2Sim")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"基础模型: {base_model_path}")
    print(f"MuJoCo 场景: {model_path}")
    if args.scene:
        print(f"场景: {args.scene}")
    print(f"历史长度: {cfg.sim.actor_obs_history_length}")
    print("=" * 60)
    
    runner = G1RgbMujocoRunner(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        model_path=model_path,
        show_camera=not args.no_display,
    )
    runner.run()


if __name__ == "__main__":
    main()
