

# GeoLoco: 基于RGB相机的视觉运动控制

## 快速开始

### 训练
```bash
cd TienKung-Lab

python legged_lab/scripts/train.py --task=g1_dwaq --headless --num_envs=4096 --max_iterations=10000

python legged_lab/scripts/train.py --task=g1_rough --headless --num_envs=4096

python legged_lab/scripts/train.py --task=g1_rgb --headless --num_envs=4096
```

### 测试
```bash
python legged_lab/scripts/play.py --task=g1_rgb --load_run=<运行目录> --checkpoint=model_<迭代次数>.pt
```

### Sim2Sim
```bash
python legged_lab/scripts/sim2sim_g1_rgb.py --policy <策略文件.pt> --model <MuJoCo模型.xml> --duration 100
```

---

## G1 机器人配置

### Intel RealSense D435i RGB 相机

| 参数 | 数值 | 说明 |
|------|------|------|
| 安装位置 | `torso_link` | 相机安装在机器人躯干部位 |
| 位置偏移 (x, y, z) | (0.0576, 0.0175, 0.4199) m | 相对于躯干坐标系的偏移量 |
| 姿态 | 俯仰角 47.6° 向下 | 朝前下方观察 |
| 分辨率 | 64 × 64 像素 | 降采样以适配强化学习训练 |
| 水平视场角 | 69.4° | D435i RGB 传感器规格 |
| 垂直视场角 | 42.5° | D435i RGB 传感器规格 |
| 焦距 | 15.12 mm | 由视场角和光圈计算得出 |
| 水平光圈 | 20.955 mm | 传感器物理尺寸 |
| 更新频率 | 10 Hz | 每5个仿真步更新一次 (仿真频率50 Hz) |
| 有效距离 | 0.01 - 100.0 m | 有效感知范围 |

### 自由度配置 (29 DOF)

当前训练使用的是 **Unitree G1 29自由度版本**，基于 `g1_29dof_simple_collision.urdf` 模型。

| 部位 | 关节名称 | 数量 |
|------|----------|------|
| **腿部** | hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll (×2) | 12 |
| **腰部** | waist_yaw, waist_roll, waist_pitch | 3 |
| **肩部** | shoulder_pitch, shoulder_roll (×2) | 4 |
| **手臂** | shoulder_yaw, elbow (×2) | 4 |
| **手腕** | wrist_yaw, wrist_roll, wrist_pitch (×2) | 6 |
| **总计** | | **29** |

### 默认关节位置 (弧度)

| 关节 | 左侧 | 右侧 |
|------|------|------|
| hip_pitch | -0.20 | -0.20 |
| hip_roll | 0.0 | 0.0 |
| hip_yaw | 0.0 | 0.0 |
| knee | 0.42 | 0.42 |
| ankle_pitch | -0.23 | -0.23 |
| ankle_roll | 0.0 | 0.0 |
| shoulder_pitch | 0.35 | 0.35 |
| shoulder_roll | 0.18 | -0.18 |
| shoulder_yaw | 0.0 | 0.0 |
| elbow | 0.87 | 0.87 |
| wrist_* | 0.0 | 0.0 |

### 仿真参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 物理时间步 | 0.005 s | `SimCfg.dt` |
| 控制降采样 | 4 | `SimCfg.decimation` |
| 控制频率 | 50 Hz | 1 / (dt × decimation) |
| 动作缩放 | 0.25 | `RobotCfg.action_scale` |
| 观测历史长度 | 1 | `actor_obs_history_length` |
| 观测裁剪 | ±100.0 | `clip_observations` |
| 动作裁剪 | ±100.0 | `clip_actions` |

### 观测空间 (96 维)

| 观测项 | 维度 | 缩放因子 |
|--------|------|----------|
| 角速度 (body frame) | 3 | 1.0 |
| 重力投影 (body frame) | 3 | 1.0 |
| 速度命令 [vx, vy, yaw_rate] | 3 | 1.0 |
| 关节位置偏差 | 29 | 1.0 |
| 关节速度 | 29 | 1.0 |
| 上一步动作 | 29 | 1.0 |
| **总计** | **96** | |

### 网络架构 (ActorCriticDepth)

策略网络采用 **视觉-本体感知融合架构**，通过 CNN 提取 RGB 图像特征，与本体状态拼接后输入 Actor MLP：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RGB Image     │    │   History       │    │  Current Obs    │
│   (64×64×3)     │    │   (obs × T)     │    │    (96 dim)     │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      │
┌─────────────────┐    ┌─────────────────┐             │
│   RGBEncoder    │    │ History Encoder │             │
│ CNN → 128 dim   │    │ MLP → 67 dim    │             │
└────────┬────────┘    └────────┬────────┘             │
         │                      │                      │
         └──────────┬───────────┴──────────────────────┘
                    │
                    ▼ Concatenate
         ┌─────────────────────┐
         │  Actor Input        │
         │ 96 + 67 + 128 = 291 │
         └──────────┬──────────┘
                    ▼
              ┌──────────┐
              │  Actor   │
              │   MLP    │
              └────┬─────┘
                   ▼
              29 Actions
```

**模块说明:**

| 模块 | 输入 | 输出 | 结构 |
|------|------|------|------|
| **RGBEncoder** | 64×64×3 图像 | 128 维特征 | Conv(8,4) → MaxPool → Conv(3,1) → FC |
| **HistoryEncoder** | 96×T 历史观测 | 67 维特征 | MLP [1024, 512, 128] → 67 |
| **Actor** | 291 维拼接特征 | 29 维动作 | MLP [512, 256, 128] → 29 |

### 非对称 Actor-Critic 架构

训练采用 **非对称 AC** 设计：Actor 只使用可部署的观测，Critic 使用特权信息帮助训练。

```python
# g1_rgb_config.py
# Height scan 配置
self.scene.height_scanner.enable_height_scan = True   # 启用 height_scan
self.scene.height_scanner.critic_only = True          # 只给 Critic 使用

# 特权信息配置 (只给 Critic)
self.scene.privileged_info.enable_feet_info = True          # 脚部位置/速度 (12 dim)
self.scene.privileged_info.enable_feet_contact_force = True # 脚部接触力 (6 dim)
self.scene.privileged_info.enable_root_height = True        # 基座高度 (1 dim)
```

#### 观测维度对比

| 网络 | 观测内容 | 维度 | 说明 |
|------|---------|------|------|
| **Actor** | proprioception only | **96** | 可部署：ang_vel(3) + gravity(3) + cmd(3) + joint_pos(29) + joint_vel(29) + action(29) |
| **Critic** | proprioception + privileged | **307** | 特权信息详见下表 |

#### Critic 特权信息明细

| 特权信息 | 维度 | 说明 |
|---------|------|------|
| actor_obs | 96 | 与 Actor 相同的基础观测 |
| root_lin_vel | 3 | 基座线速度 (body frame) |
| feet_contact | 2 | 脚部接触状态 (布尔) |
| feet_pos_in_body | 6 | 脚在身体坐标系位置 (2脚×3D) |
| feet_vel_in_body | 6 | 脚在身体坐标系速度 (2脚×3D) |
| feet_contact_force | 6 | 脚部接触力 (2脚×3D) |
| root_height | 1 | 基座高度 |
| height_scan | 187 | 地形高度扫描 |
| **Critic 总维度** | **307** | |

#### Actor 网络完整输入 (ActorCriticDepth)

| 输入 | 维度 | 来源 |
|------|------|------|
| Proprioception | 96 | 本体感知观测 |
| History Feature | 67 | HistoryEncoder(96 × T) |
| RGB Feature | 128 | RGBEncoder(64×64×3) |
| **Actor MLP 总输入** | **291** | concat(obs, his, rgb) |

#### 优势

1. **Sim2Sim/Sim2Real 友好**: Actor 不依赖特权信息，可直接部署
2. **训练更稳定**: Critic 使用地形+脚部状态等特权信息，value estimation 更准确
3. **泛化性更强**: Actor 学习从 proprioception + RGB 推断地形和接触状态

---

## Sim2Sim: Isaac Lab → MuJoCo

### 原理说明

Sim2Sim 将在 Isaac Lab (PhysX) 中训练的策略迁移到 MuJoCo 仿真环境运行，用于：
- 验证策略的跨仿真器泛化能力
- 为 Sim2Real 部署做准备
- 使用 MuJoCo 进行更快速的策略评估

### 实现流程

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MuJoCo 传感器  │ ──▶ │   构建观测向量    │ ──▶ │   策略网络推理   │
│  (关节/IMU数据)  │     │  (与训练一致)     │     │   (TorchScript)  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              ▼
│   MuJoCo 执行器  │ ◀── │   位置控制转换    │ ◀────────────────────
│  (关节位置目标)  │     │  action × scale   │     动作输出
└─────────────────┘     └──────────────────┘
```

### 关键设计

1. **关节映射**: Isaac Lab 和 MuJoCo 的关节顺序可能不同，需要建立索引映射
2. **观测空间**: 必须与训练时完全一致 (96维)
3. **动作缩放**: `target = action × 0.25 + default_pos`
4. **坐标变换**: 角速度和重力需从世界坐标系转换到机身坐标系

### 使用方法

```bash
cd TienKung-Lab

# 运行 G1 Sim2Sim
python legged_lab/scripts/sim2sim_g1.py \
    --policy <策略文件.pt> \
    --model /home/lyf/code/legged_gym/resources/robots/g1/g1_29dof.xml \
    --duration 100
```

### 键盘控制

| 按键 | 功能 | 增量 |
|------|------|------|
| 8 / 2 | 前进 / 后退 | ±0.2 m/s |
| 4 / 6 | 左移 / 右移 | ±0.2 m/s |
| 7 / 9 | 左转 / 右转 | ±0.3 rad/s |
| 5 | 停止 | 归零 |

---

## G1 Sim2Sim 详细开发记录 (2026-01-07)

### 概述

本节记录了将 Isaac Lab 训练的 G1 29-DOF g1_flat 策略迁移到 MuJoCo 的完整过程，包括遇到的所有问题和解决方案。

**最终实现文件**: `TienKung-Lab/legged_lab/scripts/sim2sim_g1_flat.py`

### 问题 1: 观测维度不匹配 (963 vs 960)

#### 现象
```
RuntimeError: Expected input tensor to have size 960, but got 963
```

#### 原因分析
- **g1_flat 环境**: 96 维观测 × 10 帧历史 = **960 维**
  - ang_vel (3) + projected_gravity (3) + command (3) + joint_pos (29) + joint_vel (29) + action (29) = 96
  
- **旧 sim2sim 代码**: 基于 tienkung 环境，包含 gait phase 信息
  - 额外包含 sin(2πφ) (2) + cos(2πφ) (2) + phase_ratio (2) = **6 维**
  - 总维度: 102 × 10 = 1020 维

#### 解决方案
创建新的 `sim2sim_g1_flat.py`，移除 gait phase 相关的观测：

```python
# g1_flat 观测构建 (96 维)
obs = np.concatenate([
    ang_vel_body,           # 3: 角速度 (body frame)
    projected_gravity,      # 3: 投影重力
    self.command_vel,       # 3: 速度命令
    joint_pos_isaac,        # 29: 关节位置偏差
    joint_vel_isaac,        # 29: 关节速度
    self.action,            # 29: 上一步动作
], axis=0)  # 总计 96 维
```

---

### 问题 2: 关节顺序映射

#### 现象
动作输出到关节的映射错误，机器人行为异常。

#### 原因分析
MuJoCo XML 和 Isaac Lab URDF 的关节顺序不同：

**MuJoCo 顺序** (按身体部位分组):
```python
MUJOCO_DOF_NAMES = [
    # 左腿 (0-5)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    # 右腿 (6-11)
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    # 腰部 (12-14)
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    # 左臂 (15-21)
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    # 右臂 (22-28)
    'right_shoulder_pitch_joint', ...
]
```

**Isaac Lab 顺序** (左右交替):
```python
LAB_DOF_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
    'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    ...
]
```

#### 解决方案
建立双向索引映射：

```python
def build_joint_mappings(self):
    # MuJoCo -> Isaac Lab
    mujoco_indices = {name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)}
    self.mujoco_to_isaac_idx = [mujoco_indices[name] for name in LAB_DOF_NAMES]
    
    # Isaac Lab -> MuJoCo
    lab_indices = {name: idx for idx, name in enumerate(LAB_DOF_NAMES)}
    self.isaac_to_mujoco_idx = [lab_indices[name] for name in MUJOCO_DOF_NAMES]

def mj29_to_lab29(self, array_mj):
    """MuJoCo 顺序 → Isaac Lab 顺序 (用于观测)"""
    return array_mj[self.mujoco_to_isaac_idx]

def lab29_to_mj29(self, array_lab):
    """Isaac Lab 顺序 → MuJoCo 顺序 (用于动作)"""
    return array_lab[self.isaac_to_mujoco_idx]
```

---

### 问题 3: 角速度坐标系 ⭐重要

#### 现象
机器人在稳定阶段就倒下，高度从 0.79m 迅速降到 0.1m。

#### 错误理解
最初认为 MuJoCo 的 `d.qvel[3:6]` 是**世界坐标系**角速度，需要转换到 body frame：
```python
# ❌ 错误做法
ang_vel_world = self.data.qvel[3:6]
ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)
```

#### 正确理解
**MuJoCo 的 `d.qvel[3:6]` 本身就是 body frame 角速度！**

MuJoCo 广义坐标结构：
```
d.qpos[0:3]  - 根节点世界坐标位置 (x, y, z)
d.qpos[3:7]  - 根节点四元数 (w, x, y, z)，世界→机体旋转
d.qpos[7:]   - 关节角度 (关节坐标系)

d.qvel[0:3]  - 根节点线速度 (世界坐标系)
d.qvel[3:6]  - 根节点角速度 (机体坐标系/body frame) ✅
d.qvel[6:]   - 关节角速度 (关节坐标系)
```

#### 解决方案
直接使用 `d.qvel[3:6]`，无需坐标变换：
```python
# ✅ 正确做法
ang_vel_body = self.data.qvel[3:6].copy()
```

#### 验证方法
参考 `sim2sim_ref/deploy_mujoco/deploy_mujoco.py`:
```python
omega = d.qvel[3:6]  # 直接使用，已是 body frame
state_cmd.ang_vel = omega.copy()
```

---

### 问题 4: 投影重力计算

#### 原理
投影重力 (`projected_gravity`) 表示重力在机体坐标系中的方向，用于感知机器人姿态。

#### 实现
```python
def get_gravity_orientation(self, quat):
    """计算投影重力向量
    
    Args:
        quat: MuJoCo 四元数 (w, x, y, z)
    
    Returns:
        投影重力向量 (3,)，body frame 中的 [0, 0, -1] 向量
    """
    qw, qx, qy, qz = quat
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation
```

#### 验证
- 机器人站立时 (identity quaternion): `[0, 0, -1]` ✓
- 机器人向前倒 90°: `[1, 0, 0]` ✓

---

### 问题 5: Actuator 类型差异

#### 背景
不同的 MuJoCo 模型使用不同的 actuator 类型：

| 模型 | Actuator 类型 | ctrl 含义 | PD 控制 |
|------|---------------|-----------|---------|
| tienkung.xml | `<position kp="700"/>` | 目标位置 | MuJoCo 内置 |
| g1_29dof.xml | `<motor/>` | 力矩 | 需手动实现 |

#### tienkung.xml 示例
```xml
<actuator>
    <position name="hip_roll_l_joint" joint="hip_roll_l_joint" kp="700"/>
    <position name="ankle_pitch_l_joint" joint="ankle_pitch_l_joint" kp="30"/>
</actuator>
```

#### g1_29dof.xml 示例
```xml
<actuator>
    <motor name="left_hip_pitch_joint" joint="left_hip_pitch_joint"/>
    <motor name="left_knee_joint" joint="left_knee_joint"/>
</actuator>
```

#### 解决方案
G1 模型需要手动实现 PD 控制器：
```python
def pd_control(self, target_q):
    """PD 控制器计算力矩"""
    q = self.data.qpos[7:7 + self.num_actions]
    dq = self.data.qvel[6:6 + self.num_actions]
    return (target_q - q) * self.kps + (0 - dq) * self.kds

# 主循环中
target_pos = self.lab29_to_mj29(action * action_scale) + default_dof_pos
tau = self.pd_control(target_pos)
self.data.ctrl[:self.num_actions] = tau
```

---

### 问题 6: PD 增益配置

#### 原则
PD 增益应与 Isaac Lab 训练时的配置一致。

#### 从 G1_CFG 获取增益
```python
# TienKung-Lab/legged_lab/assets/unitree/unitree.py
actuators={
    "legs": ImplicitActuatorCfg(
        stiffness={
            ".*_hip_yaw_joint": 150.0,
            ".*_hip_roll_joint": 150.0,
            ".*_hip_pitch_joint": 200.0,
            ".*_knee_joint": 200.0,
        },
        damping={
            ".*_hip_*": 5.0,
            ".*_knee_joint": 5.0,
        },
    ),
    "feet": ImplicitActuatorCfg(
        stiffness=20.0,
        damping=2.0,
    ),
    ...
}
```

#### 转换为 MuJoCo 顺序
```python
# PD 增益 (MuJoCo 顺序)
self.kps = np.array([
    200, 150, 150, 200, 20, 20,    # 左腿: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    200, 150, 150, 200, 20, 20,    # 右腿
    200, 200, 200,                  # 腰部
    100, 100, 50, 50, 40, 40, 40,  # 左臂
    100, 100, 50, 50, 40, 40, 40   # 右臂
])

self.kds = np.array([
    5, 5, 5, 5, 2, 2,              # 左腿
    5, 5, 5, 5, 2, 2,              # 右腿
    5, 5, 5,                        # 腰部
    2, 2, 2, 2, 2, 2, 2,           # 左臂
    2, 2, 2, 2, 2, 2, 2            # 右臂
])
```

---

### 问题 7: 稳定阶段

#### 现象
添加 2 秒稳定阶段后，机器人反而在稳定阶段就倒下。

#### 原因
- 稳定阶段使用纯 PD 控制保持默认姿态
- 但 G1 的默认姿态可能不是一个稳定的平衡点
- PD 增益可能不足以在短时间内稳定机器人

#### 解决方案
**移除稳定阶段**，直接让策略从初始状态开始控制：
```python
def run(self):
    # 不使用稳定阶段，直接初始化观测历史
    for _ in range(self.cfg.sim.actor_obs_history_length):
        self.get_obs()
    
    # 直接开始策略循环
    while viewer.is_running():
        obs = self.get_obs()
        action = policy(obs)
        ...
```

#### 原理
训练好的策略本身就能让机器人保持平衡，不需要额外的稳定阶段。

---

### 问题 8: 观测历史更新

#### 两种实现方式

**方式 1: 2D buffer (walk_tienkung.py)**
```python
# buffer shape: (history_length, obs_per_step) = (10, 102)
def update_buffer(self, buffer, new_value):
    buffer[:-1] = buffer[1:]  # 向前移动
    buffer[-1] = new_value    # 新数据放最后
    return buffer

obs = self.update_buffer(self.action_obs_buffer, action_obs)
obs = obs.reshape(1, -1)  # (1, 1020)
```

**方式 2: 1D array (sim2sim_g1_flat.py)**
```python
# obs_history shape: (num_obs_per_step * history_length,) = (960,)
self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
self.obs_history[-self.cfg.sim.num_obs_per_step:] = obs.copy()
```

两种方式数学上等价，但 1D array 更简洁。

---

### 最终代码结构

```python
class G1FlatMujocoRunner:
    def __init__(self, cfg, policy_path, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.policy = torch.jit.load(policy_path)
        self.init_variables()
        self.build_joint_mappings()
        
    def get_obs(self):
        # 1. 读取关节状态 (MuJoCo 顺序)
        dof_pos_mj = self.data.qpos[7:36]
        dof_vel_mj = self.data.qvel[6:35]
        
        # 2. 读取 IMU (body frame)
        ang_vel_body = self.data.qvel[3:6]  # 直接使用，已是 body frame
        projected_gravity = self.get_gravity_orientation(self.data.qpos[3:7])
        
        # 3. 转换到 Isaac Lab 顺序
        joint_pos_isaac = self.mj29_to_lab29(dof_pos_mj - self.default_dof_pos)
        joint_vel_isaac = self.mj29_to_lab29(dof_vel_mj)
        
        # 4. 构建观测
        obs = np.concatenate([ang_vel_body, projected_gravity, command,
                              joint_pos_isaac, joint_vel_isaac, action])
        
        # 5. 更新历史
        self.obs_history = np.roll(self.obs_history, -96)
        self.obs_history[-96:] = obs
        return self.obs_history
    
    def run(self):
        while viewer.is_running():
            # 获取观测
            obs = self.get_obs()
            
            # 策略推理
            action = self.policy(torch.tensor(obs)).numpy()
            
            # 执行控制
            for _ in range(decimation):
                target_pos = self.lab29_to_mj29(action * 0.25) + self.default_dof_pos
                tau = self.pd_control(target_pos)
                self.data.ctrl[:29] = tau
                mujoco.mj_step(self.model, self.data)
```

---

### 使用方法

```bash
cd TienKung-Lab

# 运行 G1 Flat Sim2Sim
python legged_lab/scripts/sim2sim_g1_flat.py \
    --policy logs/g1_flat/<run_dir>/exported/policy.pt \
    --model legged_lab/assets/unitree/g1/mjcf/g1_29dof_rev_1_0_daf.xml \
    --duration 60

# 键盘控制
# 8: 前进  2: 后退  4: 左移  6: 右移  7: 左转  9: 右转  5: 停止
```

---

### 调试技巧

1. **打印观测值**: 检查 ang_vel, projected_gravity 是否合理
   - 站立时 projected_gravity 应为 `[0, 0, -1]`
   
2. **检查高度**: 监控 `data.qpos[2]`，稳定站立时应保持 ~0.79m

3. **验证关节映射**: 打印 MuJoCo 和 Isaac Lab 顺序的关节名，确认映射正确

4. **对比参考实现**: 参考 `sim2sim_ref/walk_tienkung/walk_tienkung.py` 的实现

---

### MuJoCo 模型

G1 的 MuJoCo XML 模型位于:
```
/home/lyf/code/legged_gym/resources/robots/g1/g1_29dof.xml
```

可用模型版本:
| 文件 | 自由度 | 说明 |
|------|--------|------|
| `g1_12dof.xml` | 12 | 仅腿部 |
| `g1_23dof.xml` | 23 | 腿+手臂，无腰部俯仰/滚转 |
| `g1_29dof.xml` | 29 | 完整版 ✅ |
| `g1_29dof_lock_waist.xml` | 29 | 锁定腰部 |
| `g1_29dof_with_hand.xml` | 29+ | 带手指 |

### 导出策略

从训练 checkpoint 导出 TorchScript 策略:
```python
# 在 play.py 或单独脚本中
import torch
# policy 是训练好的策略网络
torch.jit.save(torch.jit.script(policy), "policy.pt")
```

---

## 开发日志

### 2026-01-10: G1RoughEnvCfg 非对称 Actor-Critic 实现

#### 修改目的
实现盲走上下台阶等地形的运动能力，使 `G1RoughEnvCfg` 与 `G1RgbEnvCfg` 的观测结构保持一致。

#### 设计理念
| 配置 | Actor 输入 | Critic 特权信息 | 用途 |
|------|-----------|----------------|------|
| **G1FlatEnvCfg** | 本体感知 | lin_vel + feet_contact | 平地行走 |
| **G1RoughEnvCfg** | 本体感知（盲） | height_scan + 脚部状态 + root_height | 盲走台阶等地形 |
| **G1RgbEnvCfg** | 本体感知 + RGB | height_scan + 脚部状态 + root_height | 视觉引导地形行走 |

#### 修改内容

**1. g1_env.py**
- 添加 `_compute_feet_state()` 方法：计算脚部在 body frame 下的位置和速度
- 添加 `feet_pos_in_body` 和 `feet_vel_in_body` buffer
- 修改 `compute_current_observations()`：添加特权信息到 critic_obs
  - `feet_pos_in_body` (6 dim)
  - `feet_vel_in_body` (6 dim)
  - `feet_contact_force` (6 dim)
  - `root_height` (1 dim)
- 修改 `compute_observations()`：添加 `critic_only` 检查，当 `height_scanner.critic_only=True` 时，actor 不获得 height_scan

**2. g1_config.py - G1RoughEnvCfg**
```python
# 非对称 AC 设置
self.scene.height_scanner.enable_height_scan = True
self.scene.height_scanner.critic_only = True  # Actor 是盲的，Critic 有地形信息

# 特权信息
self.scene.privileged_info.enable_feet_info = True
self.scene.privileged_info.enable_feet_contact_force = True
self.scene.privileged_info.enable_root_height = True

# 历史长度 10（与 g1_rgb 一致）
self.robot.actor_obs_history_length = 10
self.robot.critic_obs_history_length = 10

# 动作延迟
self.domain_rand.action_delay.enable = True
```

**3. g1_config.py - G1RoughAgentCfg**
- 改用 `ActorCritic`（配合 History Encoder）
- 网络结构改为 `[512, 256, 128]`

### 2026-01-06: RGB 相机配置修复

#### 问题描述
RGB 相机画面中偶尔会出现机器人头部的边缘轮廓。

#### 问题根源
相机原本安装在 `pelvis` 上，但头部 (`head_link`) 是通过腰部关节 (`waist_yaw/roll/pitch`) 连接到 `torso_link` 的。当机器人腰部运动时，头部相对于 `pelvis` 位置会变化，导致相机可能看到头部边缘。

#### 解决方案
1. **修改相机安装位置**：从 `pelvis` 改为 `torso_link`，与 URDF 中 `d435_joint` 的定义一致
2. **使用 URDF 中定义的精确参数**：
   - 位置：`xyz="0.0576235 0.01753 0.41987"` (相对于 torso_link)
   - 旋转：`rpy="0 0.8307767239493009 0"` (pitch=47.6° 向下)
   - 四元数：`(0.9150, 0, 0.4035, 0)`
3. **设置近裁剪面**：`clipping_range=(0.1, 100.0)`，10cm 以内的物体不渲染
4. **启用相机位姿追踪**：`update_latest_camera_pose=True`，确保相机跟随机器人运动

#### URDF 中的相机定义
```xml
<joint name="d435_joint" type="fixed">
    <origin xyz="0.0576235 0.01753 0.41987" rpy="0 0.8307767239493009 0"/>
    <parent link="torso_link"/>
    <child link="d435_link"/>
</joint>
```

### 2025-12-30: 地形配置优化

#### 地形设置位置
- **配置文件**: `TienKung-Lab/legged_lab/terrains/terrain_generator_cfg.py`
- **环境配置**: `TienKung-Lab/legged_lab/envs/g1/g1_rgb_config.py` 中的 `scene.terrain_generator`

#### 当前地形配置 (ROUGH_TERRAINS_CFG)

| 地形类型 | 比例 | 类型 | 说明 |
|---------|------|------|------|
| `stairs_up_28` | 10% | Mesh | 上台阶，台阶宽28cm |
| `stairs_up_32` | 10% | Mesh | 上台阶，台阶宽32cm |
| `stairs_down_30` | 10% | Mesh | 下台阶，台阶宽30cm |
| `stairs_down_34` | 10% | Mesh | 下台阶，台阶宽34cm |
| `boxes` | 10% | Mesh | 随机高度方块网格 |
| `random_rough` | 15% | HF | 随机粗糙地形 |
| `wave` | 10% | HF | 波浪地形 |
| `slope` | 10% | HF | 斜坡地形 (新增) |
| `high_platform` | 15% | Mesh | 深坑/平台地形 |

#### 课程学习
- **状态**: ✅ 已启用 (`curriculum=True`)
- **升级条件**: 机器人行走距离 > 地形块大小的一半 (4m)
- **降级条件**: 机器人行走距离 < 命令速度 × 回合时间 × 0.5
- **初始等级**: `max_init_terrain_level = 5`

#### Play vs Train 地形差异

| 参数 | Train | Play |
|------|-------|------|
| num_rows | 10 | 5 |
| num_cols | 20 | 5 |
| curriculum | ✅ 开启 | ❌ 关闭 |
| difficulty_range | 动态 | (0.4, 0.4) 固定 |

#### Mesh 地形 vs Height Field 地形

| 特性 | Mesh (trimesh) | Height Field (HF) |
|------|----------------|-------------------|
| 数据结构 | 三角形网格 | 2D 高度图 |
| 悬空结构 | ✅ 支持 | ❌ 不支持 |
| 几何精度 | 高 | 中等 |
| 计算开销 | 较大 | 较小 |
| 适用场景 | 台阶、平台、坑洞 | 粗糙地面、斜坡 |

#### 可用地形类型参考

**Mesh 地形**:
- `MeshPyramidStairsTerrainCfg` - 上台阶
- `MeshInvertedPyramidStairsTerrainCfg` - 下台阶
- `MeshRandomGridTerrainCfg` - 随机方块
- `MeshPitTerrainCfg` - 深坑
- `MeshGapTerrainCfg` - 缝隙
- `MeshBoxTerrainCfg` - 方块
- `MeshStarTerrainCfg` - 星形

**Height Field 地形**:
- `HfRandomUniformTerrainCfg` - 随机粗糙
- `HfPyramidSlopedTerrainCfg` - 斜坡
- `HfWaveTerrainCfg` - 波浪
- `HfSteppingStonesTerrainCfg` - 踏脚石

---

### 2025-12-30: 修复外八步态问题

#### 问题描述
机器人在行走过程中出现严重的外八步态，表现为髋关节 yaw/roll 和踝关节向外旋转。

#### 根因分析
原始的 `joint_deviation_l1` 奖励函数仅在机器人静止时惩罚关节偏差：

```python
def joint_deviation_l1(env, asset_cfg):
    angle = joint_pos - default_joint_pos
    # 仅在速度指令接近零时进行惩罚
    zero_flag = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag
```

**后果**：运动过程中 (`zero_flag=False`)，髋关节 yaw/roll 和踝关节不受任何惩罚，导致关节角度无约束偏离。

#### 解决方案
修改 `g1_rgb_config.py` 中的 `G1RewardCfg`，使用 `joint_deviation_l1_always` 函数，无论运动状态如何都进行惩罚：

| 奖励项 | 修改前 | 修改后 |
|--------|--------|--------|
| `joint_deviation_hip` | `l1`, 权重=-0.15 | `l1_always`, 权重=-0.3 |
| `joint_deviation_ankle` | (包含在 `legs` 中) | 新增独立项, `l1_always`, 权重=-0.2 |
| `joint_deviation_legs` | `hip_pitch`, `knee`, `ankle` | 仅 `hip_pitch`, `knee` |

#### 实现代码
```python
# 始终惩罚髋关节 yaw/roll 偏差
joint_deviation_hip = RewTerm(
    func=mdp.joint_deviation_l1_always,
    weight=-0.3,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])}
)

# 新增：始终惩罚踝关节偏差
joint_deviation_ankle = RewTerm(
    func=mdp.joint_deviation_l1_always,
    weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle.*"])}
)
```

#### 函数参考
| 函数名 | 行为 |
|--------|------|
| `joint_deviation_l1` | 仅在 `\|v_{cmd}\| < 0.1` 时惩罚（静止状态） |
| `joint_deviation_l1_always` | 无条件惩罚（推荐用于姿态保持关节） |