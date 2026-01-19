# G1 Vision-Based Stair Climbing System

基于 Depth Anything V2 视觉特征的 G1 人形机器人上台阶强化学习系统。

## 📋 概述

本项目实现了一个视觉引导的人形机器人上台阶控制系统，核心思路是：
- 使用 **Depth Anything V2** 预训练模型提取高维视觉特征（CLS token）
- 将视觉特征与本体感知（关节状态、IMU等）融合
- 通过强化学习训练端到端的上台阶策略

### 设计理念

1. **视觉 Encoder 与控制解耦**：视觉编码器在 Runner 层管理，Policy 只接收预提取的特征
2. **降频视觉更新**：每 5 个控制步更新一次视觉特征，模拟真实部署场景（相机帧率 < 控制频率）
3. **非对称 Actor-Critic**：
   - Actor：本体感知 + 视觉特征（可部署）
   - Critic：本体感知 + 特权信息（仅训练时使用）

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         VisionOnPolicyRunner                              │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  DepthAnythingEncoder (frozen, ViT-S 24.8M params)                 │  │
│  │                                                                     │  │
│  │  RGB Image [B, 480, 640, 3] ──→ ViT-S ──→ CLS Token [B, 384]      │  │
│  │                                    │                                │  │
│  │                              (frozen weights)                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                   ↓                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  VisionFeatureManager (5-step cache)                               │  │
│  │                                                                     │  │
│  │  • step_counter: 每 5 步更新一次特征                                │  │
│  │  • 环境重置时强制更新对应环境的特征                                  │  │
│  │  • cached_features: [num_envs, 384]                                │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                   ↓                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  VisionPPO + ActorCriticVision (G1 Robot: 29 DOF)                  │  │
│  │                                                                     │  │
│  │  ┌──────────────────────────┐    ┌──────────────────────────┐     │  │
│  │  │        Actor             │    │        Critic            │     │  │
│  │  │                          │    │                          │     │  │
│  │  │  obs (96-dim)            │    │  critic_obs (307-dim)    │     │  │
│  │  │  + history_enc (64-dim)  │    │  + history_enc (64-dim)  │     │  │
│  │  │  + vision_proj (128-dim) │    │  (含 height_scan 等)     │     │  │
│  │  │  ─────────────────────   │    │  ─────────────────────   │     │  │
│  │  │  = 288-dim input         │    │  = 371-dim input         │     │  │
│  │  │        ↓                 │    │        ↓                 │     │  │
│  │  │  MLP [256, 256]          │    │  MLP [256, 256]          │     │  │
│  │  │        ↓                 │    │        ↓                 │     │  │
│  │  │  actions (29-dim)        │    │  value (1-dim)           │     │  │
│  │  │                          │    │                          │     │  │
│  │  └──────────────────────────┘    └──────────────────────────┘     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                   ↓                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  RolloutStorageVision                                              │  │
│  │                                                                     │  │
│  │  存储 vision_feature [N, 384] 而非原始 RGB 图像 [N, H, W, 3]       │  │
│  │  内存节省: 384 floats vs 480×640×3 bytes ≈ 2000x 压缩              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
TienKung-Lab/
├── legged_lab/
│   ├── envs/g1/
│   │   ├── g1_vision_config.py     # 视觉环境配置
│   │   └── g1_vision_env.py        # 视觉环境实现
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── depth_anything_encoder.py   # Depth Anything V2 封装
│   │   └── Depth-Anything-V2/          # 预训练模型
│   │       └── metric_depth/checkpoints/
│   │           └── depth_anything_v2_metric_hypersim_vits.pth
│   ├── scripts/
│   │   └── train_vision.py         # 视觉策略训练脚本
│   └── test/
│       └── test_integration.py     # 集成测试
│
└── rsl_rl/rsl_rl/
    ├── algorithms/
    │   └── vision_ppo.py           # VisionPPO 算法
    ├── runners/
    │   └── vision_on_policy_runner.py  # 视觉训练 Runner
    ├── modules/
    │   └── actor_critic_vision.py  # 视觉策略网络
    └── storage/
        └── rollout_storage_vision.py   # 视觉 Rollout 存储
```

## 🔧 核心模块详解

### 1. DepthAnythingEncoder (`legged_lab/modules/depth_anything_encoder.py`)

封装 Depth Anything V2 预训练模型，提取视觉特征。

```python
from legged_lab.modules import DepthAnythingEncoder

encoder = DepthAnythingEncoder(
    encoder='vits',           # ViT-S (24.8M params)
    freeze_encoder=True,      # 冻结预训练权重
    use_projection=False,     # 不投影，输出原始 384-dim
    feature_type='cls',       # 使用 CLS token
)

# 输入: RGB 图像 [B, H, W, 3] (uint8)
# 输出: CLS token [B, 384] (float32)
features = encoder(rgb_images)
```

**支持的特征类型：**
| 类型 | 输出维度 | 说明 |
|------|----------|------|
| `cls` | 384 | CLS token（推荐，最高效） |
| `avg_pool` | 384 | 空间平均池化 |
| `concat` | 768 | CLS + avg_pool 拼接 |

### 2. VisionFeatureManager (`legged_lab/modules/depth_anything_encoder.py`)

管理视觉特征的降频更新，模拟真实部署场景。

```python
from legged_lab.modules import VisionFeatureManager

manager = VisionFeatureManager(
    encoder=encoder,
    num_envs=4096,
    update_interval=5,    # 每 5 步更新一次
    device='cuda',
)

# 自动处理降频更新和环境重置
features = manager.step(rgb_images, dones=reset_flags)
```

**工作流程：**
1. `step_counter` 每次调用递增
2. 当 `step_counter % update_interval == 0` 时更新所有特征
3. 当某环境 `done=True` 时，强制更新该环境的特征
4. 其他情况返回缓存的特征

### 3. ActorCriticVision (`rsl_rl/modules/actor_critic_vision.py`)

接收预提取视觉特征的策略网络。

```python
from rsl_rl.modules import ActorCriticVision

policy = ActorCriticVision(
    num_actor_obs=96,        # G1: 3+3+3+29+29+29 = 96 (ang_vel + gravity + cmd + joint_pos + joint_vel + action)
    num_critic_obs=307,      # G1: 96+3+2+6+6+6+1+187 = 307 (actor_obs + privileged_info + height_scan)
    num_actions=29,          # G1: 29 DOF humanoid robot
    # History encoder (history_length=1 in current config)
    history_dim=96 * 1,      # 1步历史，每步96维
    his_encoder_dims=[256, 128],
    his_latent_dim=64,
    # Vision feature processing
    vision_feature_dim=384,   # ViT-S CLS token
    vision_latent_dim=128,    # 投影到 128 维
    use_vision_projection=True,
    # MLP
    actor_hidden_dims=[256, 256],
    critic_hidden_dims=[256, 256],
)

# Actor forward (with vision)
actions = policy.act(obs, history, vision_feature=features)

# Critic forward (without vision, uses privileged info)
values = policy.evaluate(critic_obs, history=history)
```

**G1 机器人观测维度详解：**
```
Actor obs (单步 96 维):
  - ang_vel: 3          # 角速度
  - projected_gravity: 3 # 投影重力
  - command: 3          # 速度指令 (vel_x, vel_y, ang_vel_z)
  - joint_pos: 29       # 关节位置 (相对默认位置)
  - joint_vel: 29       # 关节速度
  - action: 29          # 上一步动作
  = 96 维

Critic obs (307 维，含特权信息):
  - actor_obs: 96
  - root_lin_vel: 3      # 基座线速度
  - feet_contact: 2      # 脚部接触 (2 feet)
  - feet_pos: 6          # 脚部位置 (2×3)
  - feet_vel: 6          # 脚部速度 (2×3)
  - feet_force: 6        # 接触力 (2×3)
  - root_height: 1       # 基座高度
  - height_scan: ~187    # 高度扫描
  = ~307 维

G1 机器人关节 (29 DOF):
  - legs: hip_yaw/roll/pitch (6) + knee (2) + waist (3) = 11
  - feet: ankle_pitch/roll (4)
  - shoulders: shoulder_pitch/roll (4)
  - arms: shoulder_yaw + elbow (4)
  - wrist: wrist_yaw/roll/pitch (6)
```

**网络结构：**
```
Actor Input:
  obs (96) + history_encoded (64) + vision_projected (128) = 288 dim
  → MLP [256, 256] → actions (29)

Critic Input:
  critic_obs (307) + history_encoded (64) = 371 dim
  → MLP [256, 256] → value (1)
  
Note: Critic 不使用视觉特征，使用 height_scan 等特权信息
```

### 4. VisionPPO (`rsl_rl/algorithms/vision_ppo.py`)

PPO 算法，接收预提取的视觉特征。

```python
from rsl_rl.algorithms import VisionPPO

alg = VisionPPO(
    policy=policy,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1e-3,
    schedule='adaptive',
    desired_kl=0.01,
)

# 初始化存储
alg.init_storage(
    num_envs=4096,
    num_transitions_per_env=24,
    vision_feature_dim=384,
    ...
)

# 训练循环
actions = alg.act(obs, critic_obs, history, vision_features)
alg.process_env_step(rewards, dones, infos)
alg.compute_returns(last_critic_obs, last_history)
loss_dict = alg.update()
```

### 5. VisionOnPolicyRunner (`rsl_rl/runners/vision_on_policy_runner.py`)

完整的视觉策略训练 Runner。

```python
from rsl_rl.runners import VisionOnPolicyRunner

runner = VisionOnPolicyRunner(
    env=env,
    train_cfg={
        "policy": {...},
        "algorithm": {...},
        "vision": {
            "enable": True,
            "encoder": "vits",
            "feature_dim": 384,
            "update_interval": 5,
        },
        ...
    },
    log_dir=log_dir,
    device='cuda',
)

runner.learn(num_learning_iterations=50000)
```

### 6. RolloutStorageVision (`rsl_rl/storage/rollout_storage_vision.py`)

存储视觉特征的 Rollout Buffer。

**内存优化：**
- 存储 `vision_features [N, 384]` 而非 `rgb_images [N, 480, 640, 3]`
- 内存节省: 384 × 4 bytes vs 480 × 640 × 3 bytes ≈ **600x 压缩**

## 🚀 使用方法

### 训练视觉策略

```bash
cd TienKung-Lab

# 基础训练（台阶地形）
python legged_lab/scripts/train_vision.py --task g1_vision_stairs

# 指定环境数量
python legged_lab/scripts/train_vision.py --task g1_vision_stairs --num_envs 2048

# 使用粗糙地形
python legged_lab/scripts/train_vision.py --task g1_vision_rough

# 继续训练
python legged_lab/scripts/train_vision.py --task g1_vision_stairs --resume
```

### 运行集成测试

```bash
cd TienKung-Lab

# 运行完整测试套件
python legged_lab/test/test_integration.py
```

测试内容：
1. `test_encoder_basic`: DepthAnythingEncoder 基本功能
2. `test_feature_manager`: VisionFeatureManager 降频更新
3. `test_integration_with_actor_critic`: 完整集成测试
4. `test_end_to_end_with_images`: 端到端图像测试
5. `test_different_feature_types`: 不同特征类型测试
6. `test_vision_ppo_storage`: VisionPPO + RolloutStorageVision 集成

### 已注册的任务

| 任务名 | 环境类 | 地形 | 说明 |
|--------|--------|------|------|
| `g1_vision` | G1VisionEnv | Stairs | 基础视觉环境 |
| `g1_vision_stairs` | G1VisionEnv | Stairs | 台阶地形（推荐） |
| `g1_vision_rough` | G1VisionEnv | Rough | 粗糙地形 |

## 📊 性能指标

### Depth Anything V2 编码器性能

| Batch Size | 推理时间 (ms) | FPS |
|------------|---------------|-----|
| 1 | 7.20 | 139 |
| 4 | 32.50 | 123 |
| 16 | 113.98 | 140 |
| 64 | 467.88 | 137 |

*测试环境: NVIDIA GPU, 480×640 输入分辨率*

### 视觉更新策略

| 参数 | 值 | 说明 |
|------|-----|------|
| 控制频率 | 50 Hz | `step_dt = 0.02s` |
| 视觉更新间隔 | 5 步 | `update_interval = 5` |
| 视觉更新频率 | 10 Hz | 模拟真实相机帧率 |
| 特征维度 | 384 | ViT-S CLS token |

## 🔬 技术细节

### Depth Anything V2 模型规格

| 模型 | 参数量 | 特征维度 | 推荐用途 |
|------|--------|----------|----------|
| ViT-S | 24.8M | 384 | 实时部署（推荐） |
| ViT-B | 97.5M | 768 | 高精度场景 |
| ViT-L | 335.3M | 1024 | 研究/离线 |

### G1 机器人观测空间

**Actor 观测 (96 dim):**
- 角速度 (3)
- 投影重力 (3)
- 速度命令 (3)
- 关节位置 (29) - G1 29自由度
- 关节速度 (29)
- 上一步动作 (29)

**Critic 观测 (~307 dim):**
- Actor 观测 (96)
- 线速度 (3)
- 脚部接触 (2)
- 脚部位置 (6, 2feet×3)
- 脚部速度 (6)
- 脚部接触力 (6)
- 根部高度 (1)
- Height scan (~187, 特权信息)

### G1 机器人关节 (29 DOF)

| 部位 | 关节 | 数量 |
|------|------|------|
| 腿部 | hip_yaw, hip_roll, hip_pitch, knee | 8 (4×2) |
| 腰部 | waist_yaw, waist_roll, waist_pitch | 3 |
| 脚踝 | ankle_pitch, ankle_roll | 4 (2×2) |
| 肩部 | shoulder_pitch, shoulder_roll | 4 (2×2) |
| 手臂 | shoulder_yaw, elbow | 4 (2×2) |
| 手腕 | wrist_yaw, wrist_roll, wrist_pitch | 6 (3×2) |
| **总计** | | **29** |

### 动作空间

- 29 维关节位置目标 (G1 机器人)
- 动作缩放: `action_scale = 0.25`
- PD 控制器执行

## 📝 配置示例

### 环境配置 (`G1VisionEnvCfg`)

```python
@configclass
class G1VisionEnvCfg(BaseEnvCfg):
    reward = G1VisionRewardCfg()
    vision = VisionEncoderCfg()
    
    def __post_init__(self):
        # 场景配置
        self.scene.terrain_generator = STAIRS_TERRAINS_CFG
        
        # 非对称 AC: height_scan 只给 Critic
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = True
        
        # RGB 相机配置 (Intel RealSense D435i)
        self.scene.rgb_camera.enable_rgb_camera = True
        self.scene.rgb_camera.height = 480
        self.scene.rgb_camera.width = 640
        self.scene.rgb_camera.update_interval_steps = 5
        
        # 视觉编码器配置
        self.vision.enable = True
        self.vision.encoder = "vits"
        self.vision.feature_dim = 384
        self.vision.update_interval = 5
```

### 代理配置 (`G1VisionAgentCfg`)

```python
@configclass
class G1VisionAgentCfg:
    seed = 42
    num_steps_per_env = 24
    max_iterations = 50000
    
    policy = {
        "class_name": "ActorCriticVision",
        "actor_hidden_dims": [256, 256],
        "vision_feature_dim": 384,
        "vision_latent_dim": 128,
    }
    
    algorithm = {
        "class_name": "VisionPPO",
        "learning_rate": 1e-3,
        "schedule": "adaptive",
        "desired_kl": 0.01,
    }
    
    vision = {
        "enable": True,
        "encoder": "vits",
        "update_interval": 5,
    }
```

## 🔗 依赖关系

```
legged_lab.modules
├── DepthAnythingEncoder    # 视觉特征提取
└── VisionFeatureManager    # 降频特征管理

rsl_rl.algorithms
└── VisionPPO              # PPO 算法（视觉版本）

rsl_rl.runners
└── VisionOnPolicyRunner   # 训练 Runner

rsl_rl.modules
├── ActorCriticVision      # 策略网络
└── VisionFeatureBuffer    # 特征缓存

rsl_rl.storage
└── RolloutStorageVision   # Rollout 存储
```

## 📚 参考资料

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) - 预训练深度估计模型
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - 机器人仿真框架
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) - 强化学习库

## 📄 许可证

本项目基于 BSD-3-Clause 许可证。

---

*最后更新: 2026年1月16日*
