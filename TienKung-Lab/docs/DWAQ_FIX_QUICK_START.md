# DWAQ 训练问题修复 - 快速指南

## 🎯 问题根源（已确诊）

**核心问题:** 4个关键 DWAQ rewards 被注释掉 + 地形太难

导致：
- 机器人快速摔倒 (55步)
- VAE 无有效数据学习
- autoencoder_loss 停在 0.95 (随机噪声水平)

---

## ✅ 已应用的修复

### 1. 启用所有关键 Rewards
- ✅ `alive` (weight=0.15) - 存活奖励
- ✅ `gait_phase_contact` (weight=0.18) - 步态相位
- ✅ `feet_swing_height` (weight=-0.2) - 摆动高度
- ✅ `base_height` (weight=-1.0) - 躯干高度

### 2. 降低地形难度
- 台阶比例: 70% → 40%
- 新增平地: 0% → 25%
- 降低斜坡难度

### 3. 增加初始噪声
- init_noise_std: 0.8 → 1.0

---

## 🚀 重新训练步骤

```bash
cd /home/lyf/code/geoloco/TienKung-Lab

# 1. 删除旧数据（重要！）
rm -rf logs/g1_dwaq/*

# 2. 启动训练
python legged_lab/scripts/train.py \
    --task=g1_dwaq \
    --headless \
    --num_envs=4096 \
    --max_iterations=5000

# 3. 监控训练
tensorboard --logdir=logs/g1_dwaq
```

---

## 📊 预期训练进度

| 阶段 | Iterations | 关键指标 | 预期值 |
|------|-----------|---------|--------|
| **VAE学习** | 0-200 | autoencoder_loss | 0.95 → 0.1-0.3 ⬇️ |
| **存活学习** | 200-500 | episode_length | 55 → 200+ ⬆️ |
| **步态优化** | 500-1500 | mean_reward | -7 → -2 ⬆️ |
| **爬楼训练** | 1500-3000 | terrain_level | 逐步增加 |
| **收敛** | 3000+ | episode_length | 800+ |

---

## ⚠️ 故障排查

### 如果 autoencoder_loss 仍不下降 (前200 iter)

可能原因：
1. obs_dim 参数错误 → 检查 runner 中的初始化
2. 速度监督信号错误 → 检查 prev_critic_obs 维度

调试方法：
```python
# 在 dwaq_ppo.py 的 update() 中添加：
print(f"vel_target shape: {vel_target.shape}")  # 应该是 [batch, 3]
print(f"vel_target values: {vel_target[0]}")   # 应该是实际速度值
print(f"code_vel values: {code_vel[0]}")       # 应该逐渐接近 vel_target
```

### 如果 episode_length 不增长 (前500 iter)

可能需要调整 reward 权重：
```python
# 在 g1_dwaq_config.py 中：
alive.weight = 0.3  # 从 0.15 增加
termination_penalty.weight = -100  # 从 -200 降低
```

### 如果训练崩溃

检查：
- `base_height` reward 的 target_height 参数
- 确保所有 reward 函数在 mdp 模块中存在

---

## 📝 修改文件列表

1. [legged_lab/envs/g1/g1_dwaq_config.py](../legged_lab/envs/g1/g1_dwaq_config.py)
   - 启用4个关键 rewards
   - 增加 init_noise_std

2. [legged_lab/terrains/terrain_generator_cfg.py](../legged_lab/terrains/terrain_generator_cfg.py)
   - 降低台阶比例
   - 增加平地和简单地形

---

## 💡 训练成功标志

前 500 iterations 内应该看到：
- ✅ autoencoder_loss 快速下降
- ✅ episode_length 逐渐增长
- ✅ mean_reward 从负值逐步上升
- ✅ 机器人能在平地上稳定行走

如果看到这些标志 → 修复成功！继续训练即可。

如果仍然没有改善 → 提供新的训练日志继续诊断。

---

## 🔗 相关文档

- 详细诊断报告: [DWAQ_Training_Issues.md](DWAQ_Training_Issues.md)
- DWAQ vs AC 对比: [DWAQ_vs_Standard_Analysis.md](DWAQ_vs_Standard_Analysis.md)
- 域随机化配置: [Domain_Randomization_Enhancements.md](Domain_Randomization_Enhancements.md)
