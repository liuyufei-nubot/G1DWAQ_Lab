# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
G1 Rough Sim2Sim 脚本
=====================

将在 Isaac Lab 中训练的 G1 机器人 g1_rough 策略迁移到 MuJoCo 仿真环境中运行。

g1_rough 使用 Asymmetric Actor-Critic 训练：
- Actor: 盲走 (无 height_scan)，可直接部署到真实机器人
- Critic: 有 height_scan 作为特权信息

观测结构 (96 维 x 10 帧历史 = 960 维):
--------------------------------------
每帧观测 (96 维) - 与 g1_flat 完全相同:
- ang_vel (3): 角速度 (body frame)
- projected_gravity (3): 投影重力
- command (3): 速度命令 [vx, vy, yaw_rate]
- joint_pos (29): 关节位置偏差 (当前 - 默认)
- joint_vel (29): 关节速度
- action (29): 上一步动作

使用方法：
---------
python legged_lab/scripts/sim2sim_g1_rough.py --policy <policy.pt> --model <g1.xml>

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

import mujoco
import mujoco.viewer
import numpy as np
import torch
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


class G1RoughSim2SimCfg:
    """G1 Rough Sim2Sim 配置类 - 与训练配置保持一致"""

    class sim:
        sim_duration = 100.0
        num_actions = 29
        num_obs_per_step = 98      # 3+3+3+29+29+29+2 = 98 (Actor 无 height_scan, 有 sin/cos phase)
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        # G1 初始高度
        init_height = 0.793
    
    class gait_phase:
        enable = True          # 启用步态相位
        period = 0.8           # 步态周期 0.8 秒
        offset = 0.5           # 左右腿相位偏移 50% (交替步态)


class G1RoughMujocoRunner:
    """
    G1 Rough Sim2Sim 运行器
    
    加载策略和 MuJoCo 模型，运行实时仿真控制。
    
    注意：g1_rough 使用 Asymmetric AC，Actor 是盲走的（无 height_scan），
    所以 sim2sim 观测结构与 g1_flat 完全相同。

    Args:
        cfg: 配置对象
        policy_path: TorchScript 导出的策略路径
        model_path: MuJoCo XML 模型路径
    """

    def __init__(self, cfg: G1RoughSim2SimCfg, policy_path: str, model_path: str):
        self.cfg = cfg
        
        # 加载 MuJoCo 模型
        print(f"[INFO] 加载 MuJoCo 模型: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        
        # 加载策略
        print(f"[INFO] 加载策略: {policy_path}")
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        # 初始化变量
        self.init_variables()
        self.build_joint_mappings()
        self.set_initial_pose()
        
        print(f"[INFO] 控制频率: {1.0 / (cfg.sim.dt * cfg.sim.decimation):.1f} Hz")
        print(f"[INFO] 观测维度: {cfg.sim.num_obs_per_step} x {cfg.sim.actor_obs_history_length} = {cfg.sim.num_obs_per_step * cfg.sim.actor_obs_history_length}")
        print(f"[INFO] Actor 模式: 盲走 (无 height_scan)")
        if cfg.gait_phase.enable:
            print(f"[INFO] 步态相位: 启用 (周期={cfg.gait_phase.period}s, 偏移={cfg.gait_phase.offset})")

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
        
        # PD 增益 (MuJoCo 顺序) - 与 G1_CFG 训练配置一致
        # MuJoCo 顺序: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll, ...
        self.kps = np.array([
            200, 150, 150, 200, 20, 20,    # 左腿: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
            200, 150, 150, 200, 20, 20,    # 右腿
            200, 200, 200,                  # 腰部: waist_yaw, waist_roll, waist_pitch
            100, 100, 50, 50, 40, 40, 40,  # 左臂: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_*
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
        
        # 步态相位
        self.phase = 0.0
        self.phase_left = 0.0
        self.phase_right = self.cfg.gait_phase.offset
        
        # 观测历史
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,),
            dtype=np.float32
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
        print(f"[INFO] 默认关节位置 (MuJoCo 前6个): {self.default_dof_pos[:6]}")

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
        """计算投影重力向量
        
        Args:
            quat: MuJoCo 四元数 (w, x, y, z)
        
        Returns:
            投影重力向量 (3,)
        """
        qw, qx, qy, qz = quat
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """四元数逆旋转 - 将向量从世界坐标系转到 body 坐标系
        
        与 sim2sim.py 保持一致的实现
        
        Args:
            q: 四元数 (x, y, z, w) 格式
            v: 世界坐标系中的向量
        
        Returns:
            body 坐标系中的向量
        """
        q_w = q[-1]  # w 在最后
        q_vec = q[:3]  # xyz
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def mj29_to_lab29(self, array_mj: np.ndarray) -> np.ndarray:
        """将 MuJoCo 顺序数组转换为 Isaac Lab 顺序"""
        return array_mj[self.mujoco_to_isaac_idx]

    def lab29_to_mj29(self, array_lab: np.ndarray) -> np.ndarray:
        """将 Isaac Lab 顺序数组转换为 MuJoCo 顺序"""
        return array_lab[self.isaac_to_mujoco_idx]

    def get_obs(self) -> np.ndarray:
        """
        计算当前观测向量
        
        Returns:
            归一化和裁剪后的观测历史
        """
        # 读取关节状态 (MuJoCo 顺序)
        dof_pos_mj = self.data.qpos[7:7 + self.num_actions].copy()
        dof_vel_mj = self.data.qvel[6:6 + self.num_actions].copy()
        
        # 基座状态
        # MuJoCo 的 d.qvel[3:6] 本身就是 body frame 角速度，无需转换
        ang_vel_body = self.data.qvel[3:6].copy()
        
        # 投影重力 - 使用 get_gravity_orientation (与 deploy_mujoco.py 一致)
        quat = self.data.qpos[3:7].copy()  # MuJoCo: (w, x, y, z)
        projected_gravity = self.get_gravity_orientation(quat)
        
        # 转换为 Isaac Lab 顺序
        joint_pos_isaac = self.mj29_to_lab29(dof_pos_mj - self.default_dof_pos)
        joint_vel_isaac = self.mj29_to_lab29(dof_vel_mj)
        
        # 计算步态相位 (sin/cos 编码)
        sin_phase = np.sin(2 * np.pi * self.phase)
        cos_phase = np.cos(2 * np.pi * self.phase)
        
        # 构建观测 (98 维) - 与 g1_env.py compute_current_observations 一致
        # g1_rough Actor 没有 height_scan (盲走), 但有 sin/cos phase
        obs = np.concatenate([
            ang_vel_body,                     # 3: 角速度 (body frame)
            projected_gravity,                # 3: 投影重力 (body frame)
            self.command_vel,                 # 3: 速度命令
            joint_pos_isaac,                  # 29: 关节位置偏差
            joint_vel_isaac,                  # 29: 关节速度
            np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 29: 上一步动作
            np.array([sin_phase]),            # 1: sin(2π*phase)
            np.array([cos_phase]),            # 1: cos(2π*phase)
        ], axis=0).astype(np.float32)
        
        # 更新观测历史 (滚动更新)
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step:] = obs.copy()
        
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def position_control(self) -> np.ndarray:
        """
        计算目标关节位置 (MuJoCo 顺序)
        
        Returns:
            目标关节位置
        """
        # action 是 Isaac Lab 顺序，转换为 MuJoCo 顺序
        actions_scaled = self.action * self.cfg.sim.action_scale
        return self.lab29_to_mj29(actions_scaled) + self.default_dof_pos

    def pd_control(self, target_q: np.ndarray) -> np.ndarray:
        """
        PD 控制器计算力矩
        
        Args:
            target_q: 目标关节位置 (MuJoCo 顺序)
        
        Returns:
            关节力矩
        """
        q = self.data.qpos[7:7 + self.num_actions]
        dq = self.data.qvel[6:6 + self.num_actions]
        return (target_q - q) * self.kps + (0 - dq) * self.kds

    def run(self) -> None:
        """运行仿真循环"""
        self.setup_keyboard_listener()
        self.listener.start()
        
        # 创建 viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # 不使用稳定阶段，直接开始策略循环（与 sim2sim.py 一致）
        # 初始化观测历史
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()
        
        print("\n[INFO] 键盘控制: 8/2=前后, 4/6=左右, 7/9=转向, 5=停止")
        print("[INFO] 按 Ctrl+C 退出\n")
        
        debug_counter = 0
        
        try:
            while self.viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                # 获取观测并执行策略
                obs = self.get_obs()
                self.action[:] = self.policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()[:self.num_actions]
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                
                # 调试输出
                debug_counter += 1
                if debug_counter <= 3:
                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  obs shape: {obs.shape}")
                    print(f"  current_obs (first 9): {obs[-self.cfg.sim.num_obs_per_step:-self.cfg.sim.num_obs_per_step+9]}")
                    print(f"  command: {self.command_vel}")
                    print(f"  action (first 6, Isaac order): {self.action[:6]}")
                
                # 执行 decimation 步
                for _ in range(self.cfg.sim.decimation):
                    step_start = time.time()
                    
                    # 计算目标位置并执行 PD 控制
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
                
                # 更新步态相位
                if self.cfg.gait_phase.enable:
                    period = self.cfg.gait_phase.period
                    offset = self.cfg.gait_phase.offset
                    self.phase = (self.episode_length_buf * self.dt) % period / period
                    self.phase_left = self.phase
                    self.phase_right = (self.phase + offset) % 1.0
                
                # 定期打印状态
                if self.episode_length_buf % 100 == 0:
                    print(f"[INFO] t={self.data.time:.1f}s, cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, {self.command_vel[2]:.2f}], h={self.data.qpos[2]:.3f}m, phase={self.phase:.2f}")
                    
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断")
        finally:
            self.listener.stop()
            self.viewer.close()
            print("[INFO] 仿真结束")

    def stabilize_robot(self, duration: float = 2.0) -> None:
        """稳定阶段 - 使用 PD 控制保持默认姿态"""
        target_pos = self.default_dof_pos.copy()
        num_steps = int(duration / self.cfg.sim.dt)
        
        for i in range(num_steps):
            tau = self.pd_control(target_pos)
            self.data.ctrl[:self.num_actions] = tau
            mujoco.mj_step(self.model, self.data)
            
            if i % 10 == 0:
                self.viewer.sync()
            time.sleep(self.cfg.sim.dt)
        
        # 初始化观测历史
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()
        
        print(f"[INFO] 稳定后高度: {self.data.qpos[2]:.4f}m")

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """调整速度命令"""
        limits = [1.0, 0.5, 1.57]  # vx, vy, yaw_rate 限制
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -limits[idx], limits[idx])

    def setup_keyboard_listener(self) -> None:
        """设置键盘监听器"""
        def on_press(key):
            try:
                if key.char == "8": self.adjust_command_vel(0, 0.2)    # 前进
                elif key.char == "2": self.adjust_command_vel(0, -0.2)  # 后退
                elif key.char == "4": self.adjust_command_vel(1, 0.2)   # 左移
                elif key.char == "6": self.adjust_command_vel(1, -0.2)  # 右移
                elif key.char == "7": self.adjust_command_vel(2, 0.3)   # 左转
                elif key.char == "9": self.adjust_command_vel(2, -0.3)  # 右转
                elif key.char == "5": self.command_vel[:] = 0.0         # 停止
            except AttributeError:
                pass
        
        self.listener = keyboard.Listener(on_press=on_press)


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


def find_latest_policy(logs_dir: str, task_name: str) -> str:
    """查找最新的策略文件
    
    Args:
        logs_dir: logs 目录路径
        task_name: 任务名称 (如 'g1_rough')
    
    Returns:
        最新的 policy.pt 路径，如果找不到则返回 None
    """
    task_dir = os.path.join(logs_dir, task_name)
    if not os.path.isdir(task_dir):
        return None
    
    # 获取所有日期目录并排序
    date_dirs = []
    for d in os.listdir(task_dir):
        full_path = os.path.join(task_dir, d)
        if os.path.isdir(full_path):
            date_dirs.append((d, full_path))
    
    if not date_dirs:
        return None
    
    # 按名称排序，最新的在最后
    date_dirs.sort(key=lambda x: x[0])
    latest_dir = date_dirs[-1][1]
    
    # 查找 exported/policy.pt
    policy_path = os.path.join(latest_dir, "exported", "policy.pt")
    if os.path.isfile(policy_path):
        return policy_path
    
    return None


def main():
    LEGGED_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    MJCF_DIR = os.path.join(LEGGED_LAB_ROOT, "legged_lab/assets/unitree/g1/mjcf")
    LOGS_DIR = os.path.join(LEGGED_LAB_ROOT, "logs")
    
    # 尝试查找最新的 g1_rough 策略
    default_policy = find_latest_policy(LOGS_DIR, "g1_rough")
    if default_policy is None:
        default_policy = os.path.join(LOGS_DIR, "g1_rough/2026-01-10_19-43-12/exported/policy.pt")
    
    default_model = os.path.join(MJCF_DIR, "g1_29dof_rev_1_0_daf.xml")
    
    # 获取可用场景
    available_scenes = get_available_scenes(MJCF_DIR)
    scene_names = list(available_scenes.keys())
    
    parser = argparse.ArgumentParser(
        description="G1 Rough Sim2Sim - 盲走策略，支持加载不同地形场景",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--policy", type=str, default=default_policy, help="策略文件路径")
    parser.add_argument("--model", type=str, default=default_model, 
                        help="MuJoCo XML 模型路径 (当未指定 --scene 时使用)")
    parser.add_argument("--scene", type=str, default=None, 
                        choices=scene_names if scene_names else None,
                        help=f"场景名称，可用场景: {', '.join(scene_names) if scene_names else '无'}\n"
                             f"场景文件位于: {MJCF_DIR}")
    parser.add_argument("--scene-file", type=str, default=None,
                        help="直接指定场景文件路径 (优先级高于 --scene)")
    parser.add_argument("--duration", type=float, default=100.0, help="仿真时长 (秒)")
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
    
    # 检查策略文件
    if not os.path.isfile(args.policy):
        print(f"[ERROR] 策略文件不存在: {args.policy}")
        print(f"[INFO] 请指定正确的策略路径，例如:")
        print(f"       python {sys.argv[0]} --policy logs/g1_rough/2025-xx-xx/exported/policy.pt")
        sys.exit(1)
    
    # 确定要加载的模型/场景文件
    if args.scene_file:
        # 直接指定场景文件
        model_path = args.scene_file
        if not os.path.isfile(model_path):
            print(f"[ERROR] 场景文件不存在: {model_path}")
            sys.exit(1)
    elif args.scene:
        # 使用预定义场景
        if args.scene not in available_scenes:
            print(f"[ERROR] 未知场景: {args.scene}")
            print(f"[INFO] 可用场景: {', '.join(scene_names)}")
            sys.exit(1)
        model_path = available_scenes[args.scene]
    else:
        # 使用默认模型
        model_path = args.model
        if not os.path.isfile(model_path):
            print(f"[ERROR] MuJoCo 模型不存在: {model_path}")
            sys.exit(1)
    
    print("=" * 60)
    print("G1 Rough Sim2Sim (盲走策略)")
    print("=" * 60)
    print(f"策略: {args.policy}")
    print(f"MuJoCo 模型/场景: {model_path}")
    if args.scene:
        print(f"场景: {args.scene}")
    print(f"Actor 模式: 盲走 (无 height_scan，可部署到真实机器人)")
    print("=" * 60)
    
    cfg = G1RoughSim2SimCfg()
    cfg.sim.sim_duration = args.duration
    
    runner = G1RoughMujocoRunner(
        cfg=cfg,
        policy_path=args.policy,
        model_path=model_path,
    )
    runner.run()


if __name__ == "__main__":
    main()
