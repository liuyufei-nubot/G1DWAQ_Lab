# G1 DWAQ 域随机化增强总结

## 修改时间
2026年1月14日

## 修改内容

### 1. 扩大摩擦力范围

**原版 DreamWaQ 配置:**
```python
friction_range = [0.2, 1.25]  # 静态摩擦力
```

**之前的配置:**
```python
static_friction_range = (0.6, 1.0)
dynamic_friction_range = (0.4, 0.8)
```

**修改后:**
```python
static_friction_range = (0.2, 1.25)
dynamic_friction_range = (0.15, 1.0)
```

**影响:** 更宽的摩擦力范围提升地形适应能力，覆盖湿滑地面(0.2)到粗糙表面(1.25)。

---

### 2. 添加质心偏移随机化 ⭐

**原版 DreamWaQ 配置:**
```python
randomize_com_displacement = True
com_displacement_range = [-0.05, 0.05]  # ±5cm
```

**新增配置:**
```python
self.domain_rand.events.randomize_com = EventTerm(
    func=mdp.randomize_rigid_body_com,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"),
        "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
    },
)
```

**物理意义:**
- 模拟负载不均衡(背包、工具携带)
- 传感器标定偏差
- 机械装配误差
- 影响平衡控制和步态稳定性

**DWAQ 优势:**  
β-VAE 编码器从接触力、关节扭矩历史中推断当前质心位置，无需显式感知！

---

### 3. 添加执行器增益随机化 ⭐⭐⭐

**原版 DreamWaQ 配置:**
```python
randomize_motor_strength = True
motor_strength_range = [0.9, 1.1]  # ±10%

randomize_Kp_factor = True
Kp_factor_range = [0.9, 1.1]  # ±10%

randomize_Kd_factor = True
Kd_factor_range = [0.9, 1.1]  # ±10%
```

**新增配置:**
```python
self.domain_rand.events.randomize_actuator_gains = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (0.9, 1.1),  # Kp ±10%
        "damping_distribution_params": (0.9, 1.1),    # Kd ±10%
        "operation": "scale",
        "distribution": "uniform",
    },
)
```

**物理意义:**
- 电机老化导致扭矩下降
- 温度影响电机性能(高温降低效率)
- 电池电压变化(满电 vs 低电量)
- 齿轮箱磨损、传动效率变化

**DWAQ 优势:**  
这是 **DWAQ 最关键的域随机化**！β-VAE 编码器学习从以下观测推断执行器状态：
- 期望动作 vs 实际关节速度差异
- 关节位置跟踪误差
- 接触力变化(弱执行器导致更大接触力)

通过历史推断，Actor 可以自适应调整动作幅度，补偿执行器性能变化！

---

## 完整域随机化对比

| 项目 | 原版 DreamWaQ | 之前配置 | **当前配置** |
|------|--------------|---------|-------------|
| **摩擦力** | [0.2, 1.25] | [0.6, 1.0] | ✅ **[0.2, 1.25]** |
| **质量** | [-1, +2] kg | [-5, +5] kg | ✅ [-5, +5] kg (更强) |
| **质心偏移** | ±5 cm | ❌ 无 | ✅ **±5 cm** |
| **执行器增益** | Kp/Kd ±10% | ❌ 无 | ✅ **±10%** |
| **初始姿态** | 固定 | 全随机 | ✅ 全随机 |
| **动作延迟** | 无 | [0, 5]步 | ✅ [0, 5]步 |
| **外力推动** | 禁用 | 10-15s | ✅ 10-15s |

### 评估
- ✅ **摩擦力:** 已对齐原版
- ✅ **质量:** 范围更大，更强鲁棒性
- ✅ **质心偏移:** 已添加
- ✅ **执行器增益:** 已添加(关键改进!)
- ✅ **初始化:** 当前更好
- ✅ **Sim2Real:** 动作延迟增强

---

## 为什么这些随机化对 DWAQ 至关重要？

### 标准 Actor-Critic 的困境

```
观测(单帧) → Actor → 动作
```

**问题:** 当执行器性能退化时，Actor 无法感知，持续输出相同动作 → 失败

### DWAQ 的解决方案

```
观测历史(5帧) → β-VAE编码器 → 潜在状态(16) + 速度(3)
                                          ↓
当前观测 + 潜在编码(19) → Actor → 自适应动作
```

**关键优势:**

1. **隐式状态估计**
   - 从关节位置误差历史 → 推断 Kp 偏低
   - 从速度跟踪误差历史 → 推断 Kd 偏低
   - 从接触力变化历史 → 推断质心偏移

2. **自适应控制**
   ```python
   if 编码器检测到"执行器弱":
       Actor 输出更大的动作幅度补偿
   if 编码器检测到"质心前倾":
       Actor 调整步态重心后移
   ```

3. **Sim2Real 鲁棒性**
   - 真实机器人必然有执行器性能差异
   - 训练时见过各种 Kp/Kd 组合
   - 部署时编码器自动推断真实参数并适应

---

## 训练建议

### 监控指标

在 TensorBoard 中关注:

```python
# 1. 自编码器损失 (应快速下降)
Loss/autoencoder  # vel_MSE + recon_MSE + β*KL_div
期望: 前 1000 iter 快速下降至低值

# 2. 速度估计精度 (可选 logging)
vel_error = torch.norm(code_vel - vel_target, dim=-1).mean()
期望: < 0.1 m/s (训练后期)

# 3. Episode 长度 (应逐渐增长)
Episode/mean_episode_length
期望: 逐渐达到 max_episode_length

# 4. 奖励 (应逐渐上升)
Episode/mean_reward
期望: 稳定上升
```

### 预期训练阶段

| 阶段 | Iterations | 现象 |
|------|-----------|------|
| **VAE 学习** | 0-500 | autoenc_loss 快速下降，episode_length 短 |
| **联合优化** | 500-2000 | autoenc_loss 稳定，episode_length 增长 |
| **策略优化** | 2000-5000 | mean_reward 上升，爬楼梯成功 |
| **微调** | 5000+ | 性能稳定，泛化能力提升 |

### 故障排查

**如果 autoenc_loss 不下降:**
- 检查 `cenet_out_dim = 19` (velocity 3 + latent 16)
- 检查 `dwaq_obs_history_length = 5`
- 检查 `prev_critic_obs` 维度是否包含 height_scan

**如果 episode_length 不增长:**
- 检查 reward 配置(可能需要启用注释掉的 reward)
- 降低 `init_noise_std` (当前 0.8)
- 增加 `entropy_coef` (当前 0.01)

---

## 代码修改位置

**文件:** `/home/lyf/code/geoloco/TienKung-Lab/legged_lab/envs/g1/g1_dwaq_config.py`

**修改位置:** `G1DwaqEnvCfg.__post_init__()` 方法末尾

**代码行数:** 约 293-328 行

---

## 下一步建议

1. ✅ **立即重新训练**  
   新的域随机化需要从头训练，之前的 checkpoint 不适用

2. ⚠️ **启用注释掉的奖励**  
   检查 `g1_dwaq_config.py` 中被注释的 reward:
   - `alive`: 存活奖励
   - `gait_phase_contact`: 步态相位奖励
   - `feet_swing_height`: 摆动腿高度奖励
   - `base_height`: 躯干高度奖励
   
   这些可能对 DWAQ 训练很重要！

3. 📊 **训练至少 5000 iterations**  
   DWAQ 需要更长时间收敛(相比标准 PPO)

4. 🔍 **对比实验**  
   可选: 用相同配置训练标准 AC，对比性能差异

---

## 预期效果

补全域随机化后，预期改进:

- ✅ **更强的 Sim2Real 能力**: 执行器增益随机化是关键
- ✅ **更好的平衡性**: 质心偏移训练增强动态稳定
- ✅ **更广的地形适应**: 摩擦力范围扩大
- ✅ **更高的鲁棒性**: VAE 学会推断更多隐藏状态

**DWAQ 在台阶场景的优势将更加明显！** 🚀
