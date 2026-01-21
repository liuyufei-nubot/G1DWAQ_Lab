# G1 DWAQ 实物部署指南

## 概述

本指南说明如何将 DWAQ 训练的策略部署到 G1 实物机器人上。

## 模型版本

| 版本 | 观测维度 | 配置文件 | 说明 |
|------|---------|---------|------|
| **无步态版本** | 96 维 | `g1_dwaq_jit.yaml` | 2026-01-15_11-21-04 |
| **带步态版本** | 100 维 | `g1_dwaq_phase.yaml` | 2026-01-16_00-46-00，转向更好 |

## 两种部署方式

| 方式 | 脚本 | 配置文件 | 策略格式 | 推荐 |
|------|------|---------|---------|------|
| **方式 A (推荐)** | `deploy.py` | `g1_dwaq_jit.yaml` / `g1_dwaq_phase.yaml` | TorchScript | ✅ |
| 方式 B | `deploy_dwaq.py` | `g1_dwaq.yaml` | Checkpoint | |

**方式 A** 使用标准的 `deploy.py`，与 g1_flat/g1_rough 部署方式一致。
**方式 B** 直接加载 checkpoint，需要在部署端重建网络结构。

---

## 方式 A: 标准部署 (推荐)

### 无步态版本 (96 维)

#### 步骤 1: 导出策略为 TorchScript

```bash
cd ~/code/geoloco/TienKung-Lab
conda activate geo

# 导出 DWAQ 策略 (无步态版本，默认 96 维)
python legged_lab/scripts/export_dwaq_policy.py \
    --checkpoint logs/g1_dwaq/2026-01-15_11-21-04/model_9999.pt

# 输出: logs/g1_dwaq/2026-01-15_11-21-04/exported/policy.pt
```

#### 步骤 2: 复制导出的策略

```bash
mkdir -p ~/code/geoloco/LeggedLabDeploy/policy/g1_dwaq
cp ~/code/geoloco/TienKung-Lab/logs/g1_dwaq/2026-01-15_11-21-04/exported/policy.pt \
   ~/code/geoloco/LeggedLabDeploy/policy/g1_dwaq/
```

#### 步骤 3: 运行部署

```bash
cd ~/code/geoloco/LeggedLabDeploy
python deploy.py --config_path configs/g1_dwaq_jit.yaml --net enx6c1ff764040a
```

---

### 带步态版本 (100 维) - 推荐

带步态版本的转向能力更好 (+42% 角速度追踪)。

#### 步骤 1: 导出策略为 TorchScript

```bash
cd ~/code/geoloco/TienKung-Lab
conda activate geo

# 导出 DWAQ 策略 (带步态版本，100 维)
python legged_lab/scripts/export_dwaq_policy.py \
    --checkpoint logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt \
    --num_obs 100

# 输出: logs/g1_dwaq/2026-01-16_00-46-00/exported/policy.pt
```

#### 步骤 2: 复制导出的策略

```bash
mkdir -p ~/code/geoloco/LeggedLabDeploy/policy/g1_dwaq_phase
cp ~/code/geoloco/TienKung-Lab/logs/g1_dwaq/2026-01-16_00-46-00/exported/policy.pt \
   ~/code/geoloco/LeggedLabDeploy/policy/g1_dwaq_phase/
```

#### 步骤 3: 运行部署

```bash
cd ~/code/geoloco/LeggedLabDeploy
python deploy.py --config_path configs/g1_dwaq_phase.yaml --net enx6c1ff764040a
```

---

## 方式 B: 直接加载 Checkpoint

### 步骤 1: 复制 checkpoint

```bash
cp ~/code/geoloco/TienKung-Lab/logs/g1_dwaq/2026-01-15_11-21-04/model_7900.pt \
   ~/code/geoloco/LeggedLabDeploy/policy/g1_dwaq/
```

### 步骤 2: 修改配置文件

编辑 `configs/g1_dwaq.yaml`，更新策略路径:

```yaml
policy_path: "policy/g1_dwaq/model_7900.pt"
```

### 步骤 3: 运行部署

```bash
cd ~/code/geoloco/LeggedLabDeploy
python deploy_dwaq.py --config_path configs/g1_dwaq.yaml --net eno1
```

---

## 连接机器人

确保:
- 机器人已开机
- 网络连接正常 (通过网线)
- 确认网络接口名称 (如 `eno1`, `eth0`, `enp2s0`)

查看网络接口:
```bash
ip link show
```

### 4. 运行部署

```bash
cd ~/code/geoloco/LeggedLabDeploy

# 使用默认配置
python deploy_dwaq.py --config_path configs/g1_dwaq.yaml --net eno1

# 或指定其他网络接口
python deploy_dwaq.py --config_path configs/g1_dwaq.yaml --net eth0
```

## 控制流程

1. **启动后**: 机器人进入零力矩状态
2. **按 Start**: 机器人移动到默认站立姿态 (2秒)
3. **按 A**: 开始 DWAQ 控制
4. **摇杆控制**:
   - 左摇杆 Y: 前进/后退
   - 左摇杆 X: 左右移动
   - 右摇杆 X: 左右转向
5. **按 Select**: 紧急停止，退出程序

## 与标准部署的区别

| 项目 | 标准部署 | DWAQ 部署 |
|------|---------|----------|
| 策略格式 | TorchScript (.pt) | Checkpoint (.pt) |
| 策略输入 | 扁平化历史 (history×obs) | current_obs + obs_history |
| 历史帧数 | 10 帧 | 5 帧 |
| 网络结构 | MLP | MLP + VAE Encoder |

## DWAQ 观测结构

### 无步态版本 (96 维)

```
观测向量:
├── ang_vel (3)          # 角速度 (body frame)
├── gravity (3)          # 投影重力
├── command (3)          # 速度命令 [vx, vy, yaw_rate]
├── joint_pos (29)       # 关节位置偏差
├── joint_vel (29)       # 关节速度
└── action (29)          # 上一步动作
```

### 带步态版本 (100 维)

```
观测向量:
├── ang_vel (3)          # 角速度 (body frame)
├── gravity (3)          # 投影重力
├── command (3)          # 速度命令 [vx, vy, yaw_rate]
├── joint_pos (29)       # 关节位置偏差
├── joint_vel (29)       # 关节速度
├── action (29)          # 上一步动作
└── gait_phase (4)       # 步态相位 [sin_L, cos_L, sin_R, cos_R]
```

步态相位参数:
- `period = 0.8s` - 步态周期
- `offset = 0.5` - 左右腿相位差 (交替步态)

## DWAQ 推理流程

### 无步态版本
```
观测历史 (5×96=480) ──→ VAE Encoder ──→ latent_code (19)
                                              │
                                              ↓
当前观测 (96) ──────────────────────→ [latent_code + obs] (115)
                                              │
                                              ↓
                                           Actor
                                              │
                                              ↓
                                        actions (29)
```

### 带步态版本
```
观测历史 (5×100=500) ──→ VAE Encoder ──→ latent_code (19)
                                              │
                                              ↓
当前观测 (100) ─────────────────────→ [latent_code + obs] (119)
                                              │
                                              ↓
                                           Actor
                                              │
                                              ↓
                                        actions (29)
```

## 故障排除

### 连接失败
```
错误: 无法连接到机器人
解决: 检查网络接口名称，确保网线连接正常
```

### 维度不匹配
```
错误: RuntimeError: size mismatch
解决: 确保使用正确版本的模型 (无步态版本 96 维)
```

### 策略加载失败
```
错误: Missing keys in state_dict
解决: 确保使用完整的 checkpoint (model_xxxxx.pt)，而非 exported/policy.pt
```

## 安全注意事项

⚠️ **重要安全提示**:

1. 首次测试时使用悬挂支架
2. 确保周围有足够空间
3. 随时准备按 Select 紧急停止
4. 建议先在 sim2sim 中测试策略效果

## 参考

- DWAQ 架构说明: `TienKung-Lab/docs/DWAQ_LATENT_CODE.md`
- Sim2Sim 测试: `TienKung-Lab/legged_lab/scripts/sim2sim_g1_dwaq.py`
- 训练配置: `TienKung-Lab/legged_lab/envs/g1/g1_dwaq_config.py`
