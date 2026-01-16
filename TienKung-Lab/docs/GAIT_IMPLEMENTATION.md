# G1 DWAQ 步态实现指南

## 实现概述

### 问题背景
之前的步态实现有一个关键问题：**步态奖励在计算，但观测中没有步态相位信息**。这导致机器人无法学习正确的步态，反而被随机惩罚。

### 解决方案
在观测中加入步态相位信息（参照 TienKung 的实现）：
- `sin(2π × phase_left)`, `cos(2π × phase_left)` - 左腿相位
- `sin(2π × phase_right)`, `cos(2π × phase_right)` - 右腿相位
- 总共 **+4 维观测**

## 观测维数变化

| 配置 | 基础观测 | 步态观测 | 总计 |
|------|---------|---------|------|
| `--no_gait`（禁用） | N dims | 无 | N dims |
| 默认（启用） | N dims | +4 dims | N+4 dims |

**重要：不同观测维数的模型互不兼容！**

## 使用方法

### 1. 测试旧模型（无步态）
```bash
cd /home/lyf/code/geoloco/TienKung-Lab

# 加载 2026-01-15_11-21-04 (训练时无步态)
# --no_gait 禁用步态，确保观测维数匹配
python legged_lab/scripts/play.py --task=g1_dwaq \
    --load_run=2026-01-15_11-21-04 \
    --no_gait \
    --terrain=rough \
    --num_envs=4
```

### 2. 测试新模型（有步态）
```bash
# 加载启用步态的模型（如果存在）
# 不加 --no_gait，使用配置文件的默认值（enable=True）
python legged_lab/scripts/play.py --task=g1_dwaq \
    --load_run=<your_new_run> \
    --terrain=rough \
    --num_envs=4
```

### 3. 重新训练（正确的步态实现）
```bash
# 训练启用步态的模型
# 观测现在包含步态相位信息，步态奖励会有效果
python legged_lab/scripts/train.py --task=g1_dwaq \
    --headless \
    --num_envs=4096 \
    --max_iterations=10000
```

## 实现细节

### compute_current_observations()
```python
# 当 gait_phase.enable = True 时：
obs_components = [
    base_obs,  # 基础观测
    torch.sin(2π × self.leg_phase),  # sin(phase) for both legs
    torch.cos(2π × self.leg_phase),  # cos(phase) for both legs
]
current_actor_obs = torch.cat(obs_components, dim=-1)
```

### reset() 函数
```python
# 只在启用步态时重置
if self.cfg.robot.gait_phase.enable:
    self.phase[env_ids] = 0.0
    self.phase_left[env_ids] = 0.0
    self.phase_right[env_ids] = self.cfg.robot.gait_phase.offset  # 0.5
    self.leg_phase[env_ids, 0] = 0.0
    self.leg_phase[env_ids, 1] = self.cfg.robot.gait_phase.offset
```

### --no_gait 参数的作用
```python
# play.py 中的处理
if args_cli.no_gait:
    env_cfg.robot.gait_phase.enable = False
    # leg_phase 保持为 0
    # 观测中不加入步态信息
    # 兼容旧模型的观测维数
```

## 期望训练结果对比

### 旧方案（2026-01-15_11-47-04）- 效果差
- 步态奖励：+0.18（机器人随机猜测相位）
- terrain_levels：3.03（降低）
- mean_reward：-8.91（更差）
- **问题**：机器人看不到相位，被惩罚

### 新方案（启用步态 + 观测）- 预期更好
- 步态奖励：应该逐步增加（机器人能看到相位）
- terrain_levels：应该继续上升
- mean_reward：应该改善
- **优势**：机器人可以学习正确的步态

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `gait_phase.period` | 0.8s | 步态周期（两步一个周期） |
| `gait_phase.offset` | 0.5 | 左右腿相位差（50% = 交替步态） |
| `stance_threshold` | 0.55 | Phase < 0.55 = 站立，≥ 0.55 = 摆腿 |
| `gait_phase_contact.weight` | 0.2 | 步态奖励权重 |

## 故障排除

### 错误：维数不匹配
```
RuntimeError: Expected input dim 1 to have size X but got Y
```
**原因**：模型的观测维数与当前配置不匹配
**解决**：
- 旧模型（无步态）：加上 `--no_gait`
- 新模型（有步态）：不加 `--no_gait`

### 问题：步态奖励仍然很小
**原因**：可能需要更多训练迭代
**解决**：
- 检查 tensorboard 中的 `gait_phase_contact` 奖励趋势
- 确保模型在正确的配置下训练
- 可能需要调整 `gait_phase_contact.weight`

## 参考文献
- TienKung 实现: `legged_lab/envs/tienkung/tienkung_env.py` (行 375-376)
- HumanoidDreamWaq 参考: `Reference/HumanoidDreamWaq/legged_gym/legged_gym/envs/g1/g1_config.py`
