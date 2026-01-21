# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from unitree_rl_gym Project (BSD-3-Clause license)
# with modifications by Legged Lab Project and TienKung-Lab Project (BSD-3-Clause license).

"""
G1 DWAQ 实物部署脚本
===================

DWAQ (Deep Variational Autoencoder for Walking) 策略的实物部署。

与标准部署的区别:
1. 使用 VAE Encoder 处理观测历史
2. Actor 输入 = current_obs + latent_code (非 flattened history)
3. 需要加载 checkpoint 并重建网络结构（非 JIT）

使用方法:
    python deploy_dwaq.py --config_path configs/g1_dwaq.yaml --net eno1
"""

import sys
import time
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.command_helper import (
    MotorMode,
    create_damping_cmd,
    init_cmd_go,
    init_cmd_hg,
)
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from config_dwaq import ConfigDWAQ


# ==================== DWAQ 网络定义 ====================

def get_activation(act_name: str) -> nn.Module:
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    else:
        return nn.ELU()


class ActorCritic_DWAQ(nn.Module):
    """DWAQ Actor-Critic 网络 (部署版本，仅包含 Actor + Encoder)"""
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cenet_in_dim: int,
        cenet_out_dim: int,
        obs_dim: int,
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.activation = get_activation(activation)
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(num_actor_obs, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, num_actions)
        )
        
        # VAE Encoder
        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)
        
        # Decoder (不用于推理，但需要加载权重)
        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Linear(128, obs_dim)
        )
        
        # Critic (不用于推理，但需要加载权重)
        self.critic = nn.Sequential(
            nn.Linear(num_critic_obs, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 1)
        )
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    
    def cenet_forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """VAE Encoder 前向传播（推理模式，使用均值）"""
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        # 推理时使用均值，不采样
        code = torch.cat((mean_vel, mean_latent), dim=-1)
        return code
    
    def act_inference(self, observations: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        """推理模式前向传播"""
        code = self.cenet_forward(obs_history)
        actor_input = torch.cat((code, observations), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean


class ControllerDWAQ:
    """DWAQ 策略控制器"""
    
    def __init__(self, config: ConfigDWAQ, net: str) -> None:
        ChannelFactoryInitialize(0, net)
        
        self.first_run = True
        self.config = config
        self.remote_controller = RemoteController()
        
        # 加载 DWAQ 策略
        self.load_dwaq_policy()
        
        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self.run)
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)
        self.cmd_lock = Lock()
        
        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        
        # 当前观测 (96 维)
        self.current_obs = np.zeros(config.num_obs, dtype=np.float32)
        
        # DWAQ 观测历史 (5 帧 × 96 维)
        self.dwaq_obs_history = np.zeros(
            (config.dwaq_obs_history_length, config.num_obs), 
            dtype=np.float32
        )
        
        self.clip_min_command = np.array(
            [
                self.config.command_range["lin_vel_x"][0],
                self.config.command_range["lin_vel_y"][0],
                self.config.command_range["ang_vel_z"][0],
            ],
            dtype=np.float32,
        )
        self.clip_max_command = np.array(
            [
                self.config.command_range["lin_vel_x"][1],
                self.config.command_range["lin_vel_y"][1],
                self.config.command_range["ang_vel_z"][1],
            ],
            dtype=np.float32,
        )
        
        # 预热策略
        print("[INFO] 预热策略...")
        for _ in range(50):
            with torch.inference_mode():
                obs = torch.from_numpy(self.current_obs.reshape(1, -1))
                obs_hist = torch.from_numpy(self.dwaq_obs_history.flatten().reshape(1, -1))
                self.policy.act_inference(obs, obs_hist)
        print("[INFO] 预热完成")
        
        # 初始化通信
        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()
            
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()
            
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()
            
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        else:
            raise ValueError("Invalid msg_type")
        
        self.wait_for_low_state()
        
        if config.msg_type == "hg":
            self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            self.low_cmd = init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
        
        self.publish_thread.Start()
        self.wait_for_start()
        
        self.move_to_default_pos()
        self.wait_for_control()
        
        print("[INFO] 开始 DWAQ 控制!")
        self.run_thread.Start()
    
    def load_dwaq_policy(self):
        """加载 DWAQ 策略"""
        print(f"[INFO] 加载 DWAQ 策略: {self.config.policy_path}")
        
        checkpoint = torch.load(self.config.policy_path, map_location="cpu", weights_only=False)
        
        # 构建网络参数
        num_obs = self.config.num_obs
        obs_history_len = self.config.dwaq_obs_history_length
        cenet_in_dim = num_obs * obs_history_len  # 96 * 5 = 480
        cenet_out_dim = self.config.cenet_out_dim  # 19
        num_actor_obs = num_obs + cenet_out_dim  # 96 + 19 = 115
        
        policy_params = {
            'num_actor_obs': num_actor_obs,
            'num_critic_obs': 200,  # 不重要
            'num_actions': self.config.num_actions,
            'cenet_in_dim': cenet_in_dim,
            'cenet_out_dim': cenet_out_dim,
            'obs_dim': num_obs,
            'activation': 'elu',
        }
        
        print(f"[INFO] 策略参数: num_obs={num_obs}, history={obs_history_len}, cenet_in={cenet_in_dim}")
        
        # 创建策略网络
        self.policy = ActorCritic_DWAQ(**policy_params)
        
        # 加载权重
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing, unexpected = self.policy.load_state_dict(model_state_dict, strict=False)
        if missing:
            print(f"[WARNING] 缺少的权重: {len(missing)} 个")
        
        self.policy.eval()
        print("[INFO] DWAQ 策略加载完成")
        
        # 加载 normalizer (如果有)
        self.obs_norm_mean = None
        self.obs_norm_var = None
        if 'obs_norm_mean' in checkpoint and 'obs_norm_var' in checkpoint:
            self.obs_norm_mean = checkpoint['obs_norm_mean'].cpu().numpy()
            self.obs_norm_var = checkpoint['obs_norm_var'].cpu().numpy()
            print(f"[INFO] 加载观测归一化参数")
    
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测"""
        if self.obs_norm_mean is not None and self.obs_norm_var is not None:
            return (obs - self.obs_norm_mean) / np.sqrt(self.obs_norm_var + 1e-8)
        return obs
    
    def LowStateHandler(self, msg):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
    
    def publish(self):
        with self.cmd_lock:
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
    
    def stop(self):
        print("[INFO] Select 按钮检测到，退出!")
        self.publish_thread.Wait()
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)
        sys.exit(0)
    
    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        self.mode_machine_ = self.low_state.mode_machine
        print("[INFO] 成功连接到机器人")
    
    def wait_for_start(self):
        print("[INFO] 进入零力矩状态")
        print("[INFO] 等待 Start 按钮移动到默认位置...")
        while self.remote_controller.button[KeyMap.start] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)
    
    def move_to_default_pos(self):
        print("[INFO] 移动到默认位置...")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.joint2motor_idx
        dof_size = len(dof_idx)
        
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        for i in range(num_step):
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            alpha = i / num_step
            with self.cmd_lock:
                for j in range(dof_size):
                    motor_idx = dof_idx[j]
                    target_pos = self.config.default_joint_pos[j]
                    self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
            time.sleep(self.config.control_dt)
    
    def wait_for_control(self):
        print("[INFO] 进入默认位置状态")
        print("[INFO] 等待 A 按钮开始控制...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)
    
    def run(self):
        """主控制循环"""
        # 读取关节状态
        for i in range(len(self.config.joint2motor_idx)):
            self.joint_pos[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q
            self.joint_vel[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq
        
        # 读取 IMU
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        
        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel
            )
        
        gravity_orientation = get_gravity_orientation(quat)
        joint_pos = (self.joint_pos - self.config.default_joint_pos) * self.config.dof_pos_scale
        joint_vel = self.joint_vel * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        
        # 速度命令
        command = np.array(
            [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx], 
            dtype=np.float32
        )
        command *= self.config.command_scale
        command = np.clip(command, self.clip_min_command, self.clip_max_command)
        
        # 构建当前观测 (96 维)
        num_actions = self.config.num_actions
        self.current_obs[:3] = ang_vel
        self.current_obs[3:6] = gravity_orientation
        self.current_obs[6:9] = command
        self.current_obs[9 : 9 + num_actions] = joint_pos
        self.current_obs[9 + num_actions : 9 + num_actions * 2] = joint_vel
        self.current_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        
        # 归一化当前观测
        current_obs_normalized = self.normalize_obs(self.current_obs)
        
        # 更新 DWAQ 观测历史
        if self.first_run:
            # 第一次运行，用当前观测填充整个历史
            for i in range(self.config.dwaq_obs_history_length):
                self.dwaq_obs_history[i] = current_obs_normalized.copy()
            self.first_run = False
        else:
            # 滚动更新历史
            self.dwaq_obs_history = np.roll(self.dwaq_obs_history, shift=-1, axis=0)
            self.dwaq_obs_history[-1] = current_obs_normalized.copy()
        
        # 准备输入
        obs_tensor = torch.from_numpy(current_obs_normalized.reshape(1, -1).astype(np.float32))
        obs_hist_tensor = torch.from_numpy(self.dwaq_obs_history.flatten().reshape(1, -1).astype(np.float32))
        
        # 执行策略
        with torch.inference_mode():
            action_tensor = self.policy.act_inference(obs_tensor, obs_hist_tensor)
        
        self.action = action_tensor.squeeze().numpy()
        self.action = np.clip(self.action, -100, 100)
        
        # 计算目标位置
        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale
        
        # 发送命令
        with self.cmd_lock:
            for i in range(len(self.config.joint2motor_idx)):
                self.low_cmd.motor_cmd[self.config.joint2motor_idx[i]].q = target_dof_pos[i]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="G1 DWAQ 实物部署")
    parser.add_argument("--net", type=str, default="eno1", help="网络接口")
    parser.add_argument("--config_path", type=str, default="configs/g1_dwaq.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    print("=" * 60)
    print("G1 DWAQ 实物部署")
    print("=" * 60)
    print(f"配置文件: {args.config_path}")
    print(f"网络接口: {args.net}")
    print("=" * 60)
    
    config = ConfigDWAQ(args.config_path)
    controller = ControllerDWAQ(config, args.net)
    
    try:
        while True:
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("[INFO] Select 按钮检测到，退出!")
                break
            time.sleep(0.01)
    finally:
        controller.run_thread.Wait()
        controller.publish_thread.Wait()
        with controller.cmd_lock:
            controller.low_cmd = create_damping_cmd(controller.low_cmd)
            controller.low_cmd.crc = CRC().Crc(controller.low_cmd)
            controller.lowcmd_publisher_.Write(controller.low_cmd)
        time.sleep(0.2)
        print("[INFO] 退出")
