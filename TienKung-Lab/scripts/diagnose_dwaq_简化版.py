#!/usr/bin/env python3
"""
简化版 DWAQ 配置诊断 - 通过读取源代码分析
"""

import re

print("=" * 80)
print("DWAQ 配置诊断 (代码分析版)")
print("=" * 80)

# 读取 DWAQ 配置文件
with open("/home/lyf/code/geoloco/TienKung-Lab/legged_lab/envs/g1/g1_dwaq_config.py", "r") as f:
    dwaq_config = f.read()

# 1. 检查注释掉的 rewards
print("\n[1] ⚠️  注释掉的关键 DWAQ Rewards:")
commented_rewards = {
    "alive": "存活奖励 - 鼓励机器人保持运行",
    "gait_phase_contact": "步态相位奖励 - 学习正确的两足步态",
    "feet_swing_height": "摆动高度奖励 - 控制抬腿高度",
    "base_height": "躯干高度奖励 - 维持合适的重心高度"
}

found_commented = []
for reward, desc in commented_rewards.items():
    if f"# {reward}" in dwaq_config or f"#{reward}" in dwaq_config:
        found_commented.append((reward, desc))
        print(f"  ❌ {reward}: {desc}")

if not found_commented:
    print("  ✅ 所有关键 rewards 已启用")

# 2. 检查 dwaq_obs_history_length
match = re.search(r"dwaq_obs_history_length\s*=\s*(\d+)", dwaq_config)
if match:
    hist_len = int(match.group(1))
    print(f"\n[2] 观测历史长度:")
    print(f"  dwaq_obs_history_length = {hist_len}")
    if hist_len != 5:
        print(f"  ⚠️  原版 DreamWaQ 使用 5 帧！")
else:
    print("\n[2] ⚠️  未找到 dwaq_obs_history_length 配置")

# 3. 检查 cenet_out_dim
match = re.search(r"cenet_out_dim\s*=\s*(\d+)", dwaq_config)
if match:
    cenet_dim = int(match.group(1))
    print(f"\n[3] 编码器输出维度:")
    print(f"  cenet_out_dim = {cenet_dim}")
    if cenet_dim != 19:
        print(f"  ⚠️  应该是 19 (velocity 3 + latent 16)！")
else:
    print("\n[3] ⚠️  未找到 cenet_out_dim 配置")

# 4. 检查 entropy_coef
match = re.search(r"entropy_coef\s*=\s*([0-9.]+)", dwaq_config)
if match:
    entropy = float(match.group(1))
    print(f"\n[4] 熵系数:")
    print(f"  entropy_coef = {entropy}")
    if entropy < 0.005:
        print(f"  ⚠️  熵系数可能太低，探索不足！")

# 5. 检查 init_noise_std
match = re.search(r"init_noise_std\s*=\s*([0-9.]+)", dwaq_config)
if match:
    noise_std = float(match.group(1))
    print(f"\n[5] 初始噪声:")
    print(f"  init_noise_std = {noise_std}")
    if noise_std < 1.0:
        print(f"  ⚠️  噪声可能不足，早期探索受限！")

# 6. 检查地形配置
if "DWAQ_TERRAINS_CFG" in dwaq_config:
    print(f"\n[6] 地形配置:")
    print(f"  使用 DWAQ_TERRAINS_CFG (70% 台阶)")
    print(f"  ⚠️  台阶比例过高可能导致初期难以学习！")

# 7. 对比普通 AC 配置
print("\n[7] 与普通 AC (g1_rough) 配置对比:")
with open("/home/lyf/code/geoloco/TienKung-Lab/legged_lab/envs/g1/g1_config.py", "r") as f:
    g1_config = f.read()

# 检查普通 AC 的 rewards
print("  普通 AC 启用的 rewards:")
ac_rewards = []
for reward in ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "body_orientation_l2"]:
    if f"{reward} =" in g1_config and f"# {reward}" not in g1_config:
        ac_rewards.append(reward)

for r in ac_rewards[:5]:
    print(f"    ✅ {r}")

# 8. 读取训练日志
print("\n[8] 训练数据分析 (最新 checkpoint):")
import os
latest_log = "/home/lyf/code/geoloco/TienKung-Lab/logs/g1_dwaq/2026-01-14_12-34-33"
if os.path.exists(latest_log):
    checkpoints = [f for f in os.listdir(latest_log) if f.startswith("model_") and f.endswith(".pt")]
    if checkpoints:
        iterations = [int(f.split("_")[1].split(".")[0]) for f in checkpoints]
        max_iter = max(iterations)
        print(f"  训练到 iteration: {max_iter}")
        print(f"  checkpoint 数量: {len(checkpoints)}")
        
        if max_iter > 1000:
            print(f"  ⚠️  训练了 {max_iter} iterations，但性能仍然很差！")

# 9. 总结问题
print("\n" + "=" * 80)
print("🔍 问题总结与建议")
print("=" * 80)

issues = []
if found_commented:
    issues.append("❌ 关键 DWAQ rewards 被注释掉")
if "DWAQ_TERRAINS_CFG" in dwaq_config:
    issues.append("⚠️  地形太难 (70% 台阶)")

print("\n主要问题:")
for i, issue in enumerate(issues, 1):
    print(f"{i}. {issue}")

print("\n建议修复方案:")
print("1. 启用注释掉的 rewards (alive, gait_phase_contact, etc.)")
print("2. 降低台阶比例，先用简单地形训练")
print("3. 增加初始噪声 init_noise_std=1.0")
print("4. 检查 obs_dim 参数传递是否正确")
print("=" * 80)
