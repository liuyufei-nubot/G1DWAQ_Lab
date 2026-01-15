# DWAQ 训练问题诊断与修复

## 🔴 问题诊断完成

### 训练数据对比 (1200 iterations)

| 指标 | DWAQ (修复前) | 普通 AC (同期) | 普通 AC (收敛) |
|------|-------------|-------------|-------------|
| **mean_reward** | -7.3 | -9.0 | +0.4 |
| **episode_length** | 55 步 | 50 步 | 912 步 |
| **autoencoder_loss** | **0.95 (不下降!)** | N/A | N/A |

### 🚨 发现的根本原因

#### 1. ⭐⭐⭐⭐⭐ 关键 DWAQ Rewards 被注释掉

经代码分析发现，**4个原版 DreamWaQ 核心 reward 全部被注释掉**：

```python
# ❌ alive = RewTerm(func=mdp.alive, weight=0.15)
# ❌ gait_phase_contact = RewTerm(..., weight=0.18)  
# ❌ feet_swing_height = RewTerm(..., weight=-0.2)
# ❌ base_height = RewTerm(..., weight=-1.0)
```

**影响分析：**
- **无 `alive` 奖励** → 机器人不知道"存活"是好事，无法学习维持平衡
- **无 `gait_phase_contact`** → 无法学习正确的两足交替步态
- **无 `feet_swing_height`** → 摆腿动作混乱，无法形成稳定步态
- **无 `base_height`** → 重心控制失败，容易摔倒

**为什么这导致 autoencoder_loss 不下降？**

DWAQ 的 β-VAE 编码器需要从**成功的行走经验**中学习潜在状态。但是：
```
无存活奖励 → 机器人快速摔倒 (55步)
→ 无有效行走数据
→ VAE 无法学习速度和地形特征
→ autoencoder_loss 停在 0.95 (随机噪声水平)
```

#### 2. ⚠️ 地形过难 (70% 台阶)

```python
# 原配置
DWAQ_TERRAINS_CFG:
  上台阶: 35%
  下台阶: 35%  
  其他: 30%
  总台阶比例: 70%
```

**问题：**
- 初期机器人在台阶上快速摔倒
- 无法获得足够的平地行走经验
- VAE 编码器缺少基础训练数据

#### 3. ⚠️ 初始噪声偏低 (0.8)

```python
init_noise_std = 0.8  # 低于标准值 1.0
```

**影响：**
- 早期探索不足
- 难以发现有效动作
- 收敛速度慢

---

## ✅ 修复方案

### 修复 1: 启用所有关键 DWAQ Rewards

**文件:** `legged_lab/envs/g1/g1_dwaq_config.py`

```python
# ✅ 已修复
alive = RewTerm(func=mdp.alive, weight=0.15)

gait_phase_contact = RewTerm(
    func=mdp.gait_phase_contact,
    weight=0.18,
    params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), 
            "stance_threshold": 0.55},
)

feet_swing_height = RewTerm(
    func=mdp.feet_swing_height,
    weight=-0.2,
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        "target_height": 0.08,
    },
)

base_height = RewTerm(
    func=mdp.base_height,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"), "target_height": 0.98},
)
```

### 修复 2: 降低地形难度

**文件:** `legged_lab/terrains/terrain_generator_cfg.py`

```python
# ✅ 已修复 - 修改为渐进式地形
DWAQ_TERRAINS_CFG:
  平地: 25%      # 新增，便于初期学习
  上台阶: 20%    # 从 35% 降低
  下台阶: 20%    # 从 35% 降低
  简单斜坡: 25%  # 增加
  其他: 10%
  总台阶比例: 40%  # 从 70% 降低
```

**好处：**
- 初期有足够平地练习基础步态
- 降低台阶难度，减少快速摔倒
- Curriculum learning 会逐步增加难度

### 修复 3: 增加初始噪声

**文件:** `legged_lab/envs/g1/g1_dwaq_config.py`

```python
# ✅ 已修复
self.policy.init_noise_std = 1.0  # 从 0.8 增加到 1.0
```

---

## 📊 预期效果

修复后的训练应该表现为：

| 阶段 | Iterations | 预期现象 |
|------|-----------|---------|
| **VAE 快速学习** | 0-200 | `autoencoder_loss` 从 0.95 快速下降到 0.1-0.3 |
| **基础步态** | 200-500 | `episode_length` 从 55 增长到 200+ |
| **步态优化** | 500-1500 | `mean_reward` 从 -7 提升到 -2 左右 |
| **台阶训练** | 1500-3000 | curriculum 增加台阶比例，学习爬楼 |
| **收敛** | 3000+ | `episode_length` 达到 800+，能稳定爬台阶 |

### 关键监控指标

```python
# 在 TensorBoard 中重点关注：
Loss/autoencoder       # 应在 200 iter 内快速下降
Train/mean_reward      # 应逐步上升
Train/mean_episode_length  # 应逐步增长
Curriculum/terrain_levels  # 地形难度逐步增加
```

---

## 🎯 重新训练步骤

### 1. 删除旧数据

```bash
cd /home/lyf/code/geoloco/TienKung-Lab
rm -rf logs/g1_dwaq/*  # 删除所有旧 checkpoint
```

**重要:** 由于修改了 reward 配置，旧的 checkpoint 不兼容！

### 2. 启动训练

```bash
python legged_lab/scripts/train.py \
    --task=g1_dwaq \
    --headless \
    --num_envs=4096 \
    --max_iterations=5000
```

### 3. 监控训练

```bash
# 打开 TensorBoard
tensorboard --logdir=logs/g1_dwaq
```

**重点观察:**
- **前 200 iterations**: `autoencoder_loss` 必须快速下降！
  - 如果仍然不下降 → 说明还有其他问题
- **前 500 iterations**: `episode_length` 应该开始增长
  - 如果始终很短 → 检查 reward 权重

### 4. 调试建议

如果训练 500 iterations 后仍然没有改善：

```python
# 可以尝试调整这些参数：

# 1. 增加存活奖励权重
alive.weight = 0.3  # 从 0.15 增加

# 2. 降低惩罚项权重
termination_penalty.weight = -100  # 从 -200 降低
feet_slide.weight = -0.1  # 从 -0.25 降低

# 3. 增加平地比例
flat.proportion = 0.4  # 从 0.25 增加

# 4. 降低学习率
self.algorithm.learning_rate = 5e-4  # 从 1e-3 降低
```

---

## 📋 修复清单

- [x] 启用 `alive` reward  
- [x] 启用 `gait_phase_contact` reward
- [x] 启用 `feet_swing_height` reward
- [x] 启用 `base_height` reward
- [x] 降低地形难度 (70% → 40% 台阶)
- [x] 增加平地比例 (0% → 25%)
- [x] 增加初始噪声 (0.8 → 1.0)
- [ ] 等待训练结果验证

---

## 🔍 其他潜在问题 (待验证)

如果修复后仍然有问题，需要检查：

### 1. obs_dim 参数传递

在 `dwaq_on_policy_runner.py` 中，检查 DWAQ PPO 初始化：

```python
# 需要确认 obs_dim 是否正确
self.alg = DWAQPPO(..., obs_dim=66)  # 应该是 actor_obs 的维度
```

**验证方法:**
```python
# 在环境中打印观测维度
print(f"Actor obs shape: {obs.shape}")  # 应该是 [num_envs, 66]
print(f"Critic obs shape: {critic_obs.shape}")  # 应该是 [num_envs, 307]
```

### 2. 速度监督信号

VAE 需要从 `prev_critic_obs` 提取速度：

```python
vel_target = prev_critic_obs[:, obs_dim:obs_dim+3]
# obs_dim=66 时，应该提取 [66:69]，即前3维速度
```

**如果 obs_dim 不对，速度监督会失败！**

### 3. Beta 退火策略

原版可能使用 beta 退火，但这不是必须的。如果需要：

```python
# 在 runner 中添加 beta schedule
beta = min(1.0, current_iter / 1000)  # 从 0 逐步增加到 1
loss_dict = self.alg.update(beta=beta)
```

---

## 总结

**核心问题:** 缺少关键 DWAQ rewards 导致机器人无法学习存活和步态，进而导致 VAE 编码器无法从有效数据中学习。

**修复效果预期:** 启用 rewards 后，机器人应该能在 200 iter 内学会基础存活，VAE loss 快速下降，然后逐步学习行走和爬台阶。

**如果还有问题，请提供:**
1. 修复后的训练日志 (前 500 iterations)
2. TensorBoard 截图
3. `autoencoder_loss` 的具体数值变化

祝训练顺利！🚀

