# 域随机化快速对比

## ✅ 已完成的增强

### 1️⃣ 摩擦力范围扩大
- **之前:** 静态 [0.6, 1.0]  
- **现在:** 静态 [0.2, 1.25] ✅  
- **效果:** 覆盖更多地形类型(湿滑→粗糙)

### 2️⃣ 质心偏移随机化 (新增)
- **之前:** ❌ 无  
- **现在:** ±5cm (x/y/z) ✅  
- **效果:** 模拟负载、传感器偏差、装配误差

### 3️⃣ 执行器增益随机化 (新增) ⭐
- **之前:** ❌ 无  
- **现在:** Kp/Kd ±10% ✅  
- **效果:** 模拟电机老化、温度、电池变化  
- **DWAQ 关键:** VAE 学习从历史推断执行器状态！

---

## 完整域随机化清单

| 项目 | 状态 | 范围 | 模式 |
|------|------|------|------|
| 摩擦力 | ✅ | [0.2, 1.25] | startup |
| 质量 | ✅ | [-5, +5] kg | startup |
| 质心偏移 | ✅ | ±5 cm | startup |
| 执行器增益 | ✅ | Kp/Kd ±10% | startup |
| 初始姿态 | ✅ | 位置/速度全随机 | reset |
| 关节初始化 | ✅ | 50-150% 默认值 | reset |
| 外力推动 | ✅ | ±1 m/s, 10-15s | interval |
| 动作延迟 | ✅ | [0, 5] 步 | 每步 |

---

## 为什么执行器增益随机化最重要？

```
情景: 真实机器人部署时，电池从 100% → 20%

标准 Actor-Critic:
  观测(单帧) → Actor → 固定动作 
  → 执行器性能下降 → 动作幅度不足 → ❌ 摔倒

DWAQ:
  观测历史(5帧) → VAE → 推断"执行器变弱" → 潜在编码
  当前观测 + 潜在编码 → Actor → 增大动作补偿 → ✅ 继续行走
```

**结论:** DWAQ 的 β-VAE 能自适应推断和补偿硬件性能变化！

---

## 重新训练建议

```bash
cd /home/lyf/code/geoloco/TienKung-Lab

# 删除旧的 checkpoint (域随机化变化，需从头训练)
rm -rf logs/g1_dwaq/*

# 启动训练 (建议 8192 envs 加速)
python legged_lab/scripts/train.py \
    --task=g1_dwaq \
    --headless \
    --num_envs=8192 \
    --max_iterations=5000
```

### 监控要点

```python
# TensorBoard
tensorboard --logdir=logs/g1_dwaq

# 关键指标:
Loss/autoencoder        # 应在 500 iter 内快速下降
Episode/mean_reward     # 应在 2000 iter 后上升
Episode/mean_episode_length  # 应逐渐接近 max (1000 步)
```

---

## 故障排查

### autoenc_loss 不下降
- 检查 `cenet_out_dim = 19`
- 检查 `dwaq_obs_history_length = 5`
- 降低学习率 `learning_rate = 5e-4`

### episode_length 很短
- 启用更多 reward (检查注释掉的)
- 降低 `init_noise_std = 0.5`
- 增加 `entropy_coef = 0.02`

### 爬不上台阶
- 增加 `feet_air_time` reward 权重
- 检查 terrain 配置 (应该 70% stairs)
- 训练更久 (>3000 iter)

---

## 文档位置

- 详细分析: [DWAQ_vs_Standard_Analysis.md](DWAQ_vs_Standard_Analysis.md)
- 配置总结: [Domain_Randomization_Enhancements.md](Domain_Randomization_Enhancements.md)
- 代码位置: [g1_dwaq_config.py](../legged_lab/envs/g1/g1_dwaq_config.py) (行 293-328)
