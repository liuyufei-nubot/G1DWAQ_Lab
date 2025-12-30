

# GeoLoco: 基于RGB相机的视觉运动控制

## 开发日志

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

---

## 快速开始

### 训练
```bash
cd TienKung-Lab
python legged_lab/scripts/train.py --task=g1_rgb --headless --num_envs=4096
```

### 测试
```bash
python legged_lab/scripts/play.py --task=g1_rgb --load_run=<运行目录> --checkpoint=model_<迭代次数>.pt
```

---

## 硬件配置

### Intel RealSense D435i RGB 相机

| 参数 | 数值 | 说明 |
|------|------|------|
| 安装位置 | `pelvis` | 相机安装在机器人骨盆部位 |
| 位置偏移 (x, y, z) | (0.0576, 0.0175, 0.4299) m | 相对于骨盆坐标系的偏移量 |
| 姿态 | 俯仰角 -45° | 朝前下方观察 |
| 分辨率 | 64 × 64 像素 | 降采样以适配强化学习训练 |
| 水平视场角 | 69.4° | D435i RGB 传感器规格 |
| 垂直视场角 | 42.5° | D435i RGB 传感器规格 |
| 焦距 | 15.12 mm | 由视场角和光圈计算得出 |
| 水平光圈 | 20.955 mm | 传感器物理尺寸 |
| 更新频率 | 10 Hz | 每5个仿真步更新一次 (仿真频率50 Hz) |
| 有效距离 | 0.1 - 10.0 m | 有效感知范围 |