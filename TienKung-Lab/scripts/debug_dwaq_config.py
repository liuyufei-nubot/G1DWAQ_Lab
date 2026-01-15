#!/usr/bin/env python3
"""
DWAQ 训练问题诊断脚本
运行此脚本检查配置和数据维度
"""

import sys
sys.path.append("/home/lyf/code/geoloco/TienKung-Lab")

print("=" * 80)
print("DWAQ 配置诊断")
print("=" * 80)

# 1. 检查环境配置
print("\n[1] 环境配置检查:")
from legged_lab.envs.g1.g1_dwaq_config import G1DwaqEnvCfg, G1DwaqAgentCfg

env_cfg = G1DwaqEnvCfg()
env_cfg.__post_init__()

agent_cfg = G1DwaqAgentCfg()
agent_cfg.__post_init__()

print(f"  dwaq_obs_history_length: {env_cfg.robot.dwaq_obs_history_length}")
print(f"  actor_obs_history_length: {env_cfg.robot.actor_obs_history_length}")
print(f"  critic_obs_history_length: {env_cfg.robot.critic_obs_history_length}")
print(f"  地形类型: {env_cfg.scene.terrain_type}")
print(f"  Height scanner enabled: {env_cfg.scene.height_scanner.enable_height_scan}")
print(f"  Height scanner critic_only: {env_cfg.scene.height_scanner.critic_only}")

# 2. 检查网络配置
print("\n[2] 网络配置检查:")
print(f"  Policy class: {agent_cfg.policy.class_name}")
print(f"  cenet_out_dim: {agent_cfg.policy.cenet_out_dim}")
print(f"  init_noise_std: {agent_cfg.policy.init_noise_std}")
print(f"  actor_hidden_dims: {agent_cfg.policy.actor_hidden_dims}")
print(f"  critic_hidden_dims: {agent_cfg.policy.critic_hidden_dims}")

# 3. 检查算法配置
print("\n[3] 算法配置检查:")
print(f"  Algorithm class: {agent_cfg.algorithm.class_name}")
print(f"  entropy_coef: {agent_cfg.algorithm.entropy_coef}")
print(f"  learning_rate: {agent_cfg.algorithm.learning_rate}")
print(f"  num_learning_epochs: {agent_cfg.algorithm.num_learning_epochs}")
print(f"  num_mini_batches: {agent_cfg.algorithm.num_mini_batches}")

# 4. 检查 Reward 配置
print("\n[4] Reward 配置检查 (注释掉的关键 rewards):")
import inspect
from legged_lab.envs.g1.g1_dwaq_config import G1DwaqRewardCfg

reward_cfg = G1DwaqRewardCfg()
source_code = inspect.getsource(G1DwaqRewardCfg)

commented_rewards = []
if "# alive" in source_code:
    commented_rewards.append("alive (存活奖励)")
if "# gait_phase_contact" in source_code:
    commented_rewards.append("gait_phase_contact (步态相位)")
if "# feet_swing_height" in source_code:
    commented_rewards.append("feet_swing_height (摆动高度)")
if "# base_height" in source_code:
    commented_rewards.append("base_height (躯干高度)")

if commented_rewards:
    print("  ⚠️  以下关键 rewards 被注释掉:")
    for r in commented_rewards:
        print(f"    - {r}")
else:
    print("  ✅ 所有关键 rewards 已启用")

# 5. 检查启用的 rewards
print("\n[5] 当前启用的 rewards:")
for attr_name in dir(reward_cfg):
    attr = getattr(reward_cfg, attr_name)
    if hasattr(attr, 'weight'):
        print(f"  {attr_name}: weight={attr.weight}")

# 6. 域随机化检查
print("\n[6] 域随机化配置:")
print(f"  physics_material: {env_cfg.domain_rand.events.physics_material.params.get('static_friction_range', 'N/A')}")
print(f"  add_base_mass: {env_cfg.domain_rand.events.add_base_mass.params.get('mass_distribution_params', 'N/A')}")

has_com_rand = hasattr(env_cfg.domain_rand.events, 'randomize_com')
has_actuator_rand = hasattr(env_cfg.domain_rand.events, 'randomize_actuator_gains')

print(f"  ✅ randomize_com: {has_com_rand}")
print(f"  ✅ randomize_actuator_gains: {has_actuator_rand}")

# 7. 观测维度估算
print("\n[7] 观测维度估算:")
print("  Actor obs (base):")
print("    - ang_vel: 3")
print("    - gravity: 3")
print("    - commands: 3")
print("    - dof_pos: 29")
print("    - dof_vel: 29")
print("    - actions (prev): 29")
print("    Total actor_obs: 3+3+3+29+29+29 = 96")
print("    ⚠️  但配置中 obs_dim 可能被硬编码为其他值！")

print("\n  Privileged obs (critic):")
print("    - actor_obs: 96")
print("    - lin_vel: 3")
print("    - contact: 2")
print("    - feet_pos: 12")
print("    - feet_vel: 12")
print("    - force: 6")
print("    - height: 1")
print("    - height_scan: 187 (if enabled)")
print("    Total privileged_obs: 96+3+2+12+12+6+1+187 = 319")

print("\n  Obs history:")
print(f"    - dwaq_obs_history: 96 × 5 = 480")

print("\n" + "=" * 80)
print("请检查以下关键问题:")
print("=" * 80)
print("1. DWAQ PPO 初始化时的 obs_dim 参数是否正确？")
print("2. 注释掉的 rewards (alive, gait_phase_contact等) 是否需要启用？")
print("3. 地形是否太难 (70% stairs) 导致无法学习？")
print("4. autoencoder loss 为 0.95 说明 VAE 完全未学习")
print("=" * 80)
