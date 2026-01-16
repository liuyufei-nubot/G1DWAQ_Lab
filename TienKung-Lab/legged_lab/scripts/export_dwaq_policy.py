# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

"""
DWAQ 策略导出脚本
================

将 DWAQ checkpoint (model_xxxx.pt) 导出为 TorchScript (JIT) 格式，
用于实物部署。

导出后的模型可以直接用标准的 deploy.py 部署，只需修改配置文件。

使用方法:
    python export_dwaq_policy.py --checkpoint logs/g1_dwaq/2026-xx-xx/model_xxxx.pt

输出:
    logs/g1_dwaq/2026-xx-xx/exported/policy.pt  (TorchScript 格式)
"""

import argparse
import os

import torch
import torch.nn as nn


def get_activation(act_name: str) -> nn.Module:
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    else:
        return nn.ELU()


class DWAQPolicyExporter(nn.Module):
    """DWAQ 策略导出器 - 将 Encoder + Actor 合并为单输入模型"""
    
    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        dwaq_obs_history_length: int,
        cenet_out_dim: int = 19,
        activation: str = "elu",
    ):
        super().__init__()
        
        self.num_obs = num_obs
        self.dwaq_obs_history_length = dwaq_obs_history_length
        self.cenet_out_dim = cenet_out_dim
        
        cenet_in_dim = num_obs * dwaq_obs_history_length
        num_actor_obs = num_obs + cenet_out_dim
        
        act = get_activation(activation)
        
        # VAE Encoder
        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            act,
            nn.Linear(128, 64),
            act,
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(64, 3)
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(num_actor_obs, 512),
            act,
            nn.Linear(512, 256),
            act,
            nn.Linear(256, 128),
            act,
            nn.Linear(128, num_actions)
        )
        
        # Normalizer (Identity by default, will be replaced if available)
        self.normalizer = nn.Identity()
    
    def forward(self, obs_history_flat: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 接收扁平化的观测历史，返回动作
        
        Args:
            obs_history_flat: [batch, dwaq_obs_history_length * num_obs]
                              最后 num_obs 个元素是当前观测
        
        Returns:
            actions: [batch, num_actions]
        """
        # 归一化
        obs_history_flat = self.normalizer(obs_history_flat)
        
        # 提取当前观测 (历史的最后一帧)
        current_obs = obs_history_flat[:, -self.num_obs:]
        
        # VAE Encoder
        distribution = self.encoder(obs_history_flat)
        mean_latent = self.encode_mean_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        code = torch.cat((mean_vel, mean_latent), dim=-1)
        
        # Actor
        actor_input = torch.cat((code, current_obs), dim=-1)
        actions = self.actor(actor_input)
        
        return actions


class EmpiricalNormalization(nn.Module):
    """用于导出的归一化模块"""
    
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


def export_dwaq_policy(
    checkpoint_path: str,
    output_path: str = None,
    num_obs: int = 96,
    num_actions: int = 29,
    dwaq_obs_history_length: int = 5,
    cenet_out_dim: int = 19,
):
    """
    导出 DWAQ 策略为 TorchScript
    
    Args:
        checkpoint_path: model_xxxx.pt 路径
        output_path: 输出路径 (默认为 checkpoint 同目录下的 exported/policy.pt)
        num_obs: 观测维度 (默认 96)
        num_actions: 动作维度 (默认 29)
        dwaq_obs_history_length: DWAQ 历史长度 (默认 5)
        cenet_out_dim: Encoder 输出维度 (默认 19)
    """
    print(f"[INFO] 加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # 创建导出器
    exporter = DWAQPolicyExporter(
        num_obs=num_obs,
        num_actions=num_actions,
        dwaq_obs_history_length=dwaq_obs_history_length,
        cenet_out_dim=cenet_out_dim,
    )
    
    # 加载权重
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 映射权重
    exporter_state = exporter.state_dict()
    for key in exporter_state:
        if key in model_state_dict:
            exporter_state[key] = model_state_dict[key]
    
    exporter.load_state_dict(exporter_state)
    
    # 加载归一化参数 (如果有)
    if 'obs_norm_mean' in checkpoint and 'obs_norm_var' in checkpoint:
        print("[INFO] 加载归一化参数")
        # 扩展归一化参数到完整历史长度
        mean = checkpoint['obs_norm_mean'].cpu()
        var = checkpoint['obs_norm_var'].cpu()
        
        # 复制到历史长度
        full_mean = mean.repeat(dwaq_obs_history_length)
        full_var = var.repeat(dwaq_obs_history_length)
        
        exporter.normalizer = EmpiricalNormalization(full_mean, full_var)
    
    exporter.eval()
    
    # 确定输出路径
    if output_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        output_dir = os.path.join(checkpoint_dir, "exported")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "policy.pt")
    
    # 导出为 TorchScript
    print(f"[INFO] 导出 TorchScript 到: {output_path}")
    
    # 测试输入
    test_input = torch.zeros(1, dwaq_obs_history_length * num_obs)
    
    # 使用 trace 而非 script (更稳定)
    traced_model = torch.jit.trace(exporter, test_input)
    traced_model.save(output_path)
    
    # 验证
    print("[INFO] 验证导出的模型...")
    loaded_model = torch.jit.load(output_path)
    test_output = loaded_model(test_input)
    print(f"[INFO] 输入形状: {test_input.shape}")
    print(f"[INFO] 输出形状: {test_output.shape}")
    print(f"[INFO] 导出成功!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="导出 DWAQ 策略为 TorchScript")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径 (model_xxxx.pt)")
    parser.add_argument("--output", type=str, default=None, help="输出路径 (默认: checkpoint目录/exported/policy.pt)")
    parser.add_argument("--num_obs", type=int, default=96, help="观测维度 (默认 96)")
    parser.add_argument("--num_actions", type=int, default=29, help="动作维度 (默认 29)")
    parser.add_argument("--history_length", type=int, default=5, help="DWAQ 历史长度 (默认 5)")
    parser.add_argument("--cenet_out_dim", type=int, default=19, help="Encoder 输出维度 (默认 19)")
    args = parser.parse_args()
    
    export_dwaq_policy(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_obs=args.num_obs,
        num_actions=args.num_actions,
        dwaq_obs_history_length=args.history_length,
        cenet_out_dim=args.cenet_out_dim,
    )


if __name__ == "__main__":
    main()
