# DWAQ 19 维潜在编码详解

## 概述

DWAQ (Deep Variational Autoencoder for Walking) 使用 β-VAE 从观测历史中学习潜在表示，用于盲走 (blind walking)。

## 结构组成

```
code (19 dim) = velocity (3 dim) + latent (16 dim)
```

| 组成部分 | 维度 | 学习方式 | 用途 |
|---------|------|---------|------|
| velocity | 3 | 有监督 (MSE) | 估计机器人速度 |
| latent | 16 | 无监督 (VAE) | 编码隐式环境特征 |

---

## 1. 速度估计 (3 维) - 有监督学习

### 训练方式

Critic 观测中包含真实速度（特权信息），用来监督 Encoder 的速度预测：

```python
# dwaq_ppo.py
vel_target = critic_obs_batch[:, self.obs_dim : self.obs_dim + 3].detach()  # 真实速度
autoenc_loss = nn.MSELoss()(code_vel, vel_target)  # 速度估计 loss
```

### 作用

- **训练时**: 从观测历史学习预测速度的能力
- **推理时**: 弥补盲走（没有直接速度测量）的缺陷，为 Actor 提供速度信息

---

## 2. 潜在状态 (16 维) - 无监督学习 (β-VAE)

### 训练方式

通过 VAE 的**重构损失 + KL 散度**隐式学习：

```python
# dwaq_ppo.py

# 重构损失: Decoder 用 code 重构当前观测
decode_target = obs_batch[:, :self.obs_dim]
reconstruction_loss = nn.MSELoss()(decode, decode_target)

# KL 散度: 正则化潜在空间
kl_divergence = -0.5 * torch.sum(
    1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp()
)

# 总 autoencoder loss
autoenc_loss = (
    nn.MSELoss()(code_vel, vel_target)      # 速度估计
    + nn.MSELoss()(decode, decode_target)   # 重构
    + beta * kl_divergence                   # KL 正则化
) / num_mini_batches
```

### 16 维潜在状态可能学到的信息

| 类别 | 可能编码的信息 |
|------|--------------|
| **地形特征** | 地面摩擦力、坡度、台阶高度 |
| **动力学状态** | 惯量变化、质心偏移、负载 |
| **接触状态** | 脚是否着地、接触力分布 |
| **历史依赖** | 运动趋势、加速度变化 |

### 为什么需要这 16 维？

**核心思想**: 盲走机器人无法直接感知环境，但可以从**历史观测的变化模式**中推断隐藏信息。

**例子**:
- 如果最近几帧关节速度突然变慢 → 可能踩到高摩擦地面 → latent 编码这个信息
- 如果投影重力变化 → 可能在爬坡 → latent 编码坡度估计
- 如果某些关节反馈异常 → 可能踩空/撞到障碍 → latent 编码接触状态

---

## 3. 信息流架构

### 训练时

```
┌─────────────────────────────────────────────────────────────────┐
│                          训练时                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  观测历史 (5帧 × 96维 = 480维)                                    │
│         │                                                        │
│         ↓                                                        │
│     Encoder                                                      │
│         │                                                        │
│         ↓                                                        │
│    code (19维) ─────────────────────────────┐                    │
│         │                                    │                    │
│    ┌────┴────┐                               ↓                    │
│    ↓         ↓                           Actor                   │
│ vel(3)   latent(16)                    (obs + code → actions)    │
│    │         │                                                   │
│    ↓         ↓                                                   │
│   MSE      重构loss + KL                                         │
│ (真实速度)      │                                                 │
│                ↓                                                  │
│           Decoder                                                │
│                │                                                  │
│                ↓                                                  │
│           obs 重构                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 推理时

```
观测历史 (480维) ──→ Encoder ──→ code (19维)
                                    │
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              velocity (3)    latent (16)      当前观测 (96)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ↓
                         Actor 输入 (115维)
                                    │
                                    ↓
                            actions (29维)
```

---

## 4. 训练 vs 推理的区别

| 阶段 | 速度 (3 dim) | 潜在状态 (16 dim) |
|------|-------------|------------------|
| **训练** | MSE loss（真实速度监督） | 重构 loss + KL loss（无监督） |
| **推理** | 使用 mean (确定性) | 使用 mean (确定性) |

### 推理代码

```python
# sim2sim_g1_dwaq.py
def cenet_forward(self, obs_history):
    distribution = self.encoder(obs_history)
    mean_latent = self.encode_mean_latent(distribution)  # 不采样，使用均值
    mean_vel = self.encode_mean_vel(distribution)        # 不采样，使用均值
    code = torch.cat((mean_vel, mean_latent), dim=-1)
    return code
```

---

## 5. 网络结构详解

### Encoder (β-VAE)

```python
# 输入: obs_history [batch, 480]  (5帧 × 96维)
self.encoder = nn.Sequential(
    nn.Linear(480, 128),
    nn.ELU(),
    nn.Linear(128, 64),
    nn.ELU(),
)

# 输出 4 个分支:
self.encode_mean_vel = nn.Linear(64, 3)       # 速度均值
self.encode_logvar_vel = nn.Linear(64, 3)     # 速度方差
self.encode_mean_latent = nn.Linear(64, 16)   # 潜在均值
self.encode_logvar_latent = nn.Linear(64, 16) # 潜在方差
```

### Decoder

```python
# 输入: code [batch, 19]
self.decoder = nn.Sequential(
    nn.Linear(19, 64),
    nn.ELU(),
    nn.Linear(64, 128),
    nn.ELU(),
    nn.Linear(128, 96)  # 重构当前观测
)
```

### Actor

```python
# 输入: obs + code [batch, 96 + 19 = 115]
self.actor = nn.Sequential(
    nn.Linear(115, 512),
    nn.ELU(),
    nn.Linear(512, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, 29)  # 29 DOF actions
)
```

---

## 6. 总结

| 组件 | 作用 |
|------|------|
| **Encoder** | 从观测历史提取信息 → code (19维) |
| **速度 (3维)** | 显式学习速度估计（有监督） |
| **潜在 (16维)** | 隐式学习环境特征（VAE 无监督） |
| **Decoder** | 重构观测，训练 VAE（推理时不用） |
| **Actor** | obs + code → actions |

**核心价值**: 让盲走机器人通过历史观测**推断**无法直接感知的环境信息（摩擦力、坡度、接触状态等），实现鲁棒的 blind locomotion。

---

## 参考

- DreamWaQ 原始实现: https://github.com/Gepetto/DreamWaQ
- 代码文件: 
  - `rsl_rl/rsl_rl/modules/actor_critic_DWAQ.py`
  - `rsl_rl/rsl_rl/algorithms/dwaq_ppo.py`
  - `legged_lab/envs/g1/g1_dwaq_config.py`
