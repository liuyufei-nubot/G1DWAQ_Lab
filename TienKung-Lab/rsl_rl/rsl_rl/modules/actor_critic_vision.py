# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Actor-Critic with pre-extracted visual features (e.g., from Depth Anything V2)
# 
# 设计理念：
# 1. 视觉 encoder 和策略网络解耦，encoder 在环境层运行
# 2. 策略网络只接收预提取的视觉特征，不包含 CNN
# 3. 支持降频特征提取（如 5 步更新一次），模拟真实部署场景
# 4. 可选的特征投影层用于维度适配和训练

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticVision(nn.Module):
    """
    Actor-Critic network with pre-extracted visual features.
    
    与 ActorCriticDepth 的区别：
    - ActorCriticDepth: 接收 rgb_image，内部用 CNN 提取特征
    - ActorCriticVision: 直接接收 vision_feature，不含 encoder
    
    使用场景：
    - Depth Anything V2 冻结 encoder 提取特征
    - DINO/DINOv2 预提取特征
    - 任何外部视觉 encoder
    
    输入格式：
    - observations: [B, num_actor_obs] 本体感知
    - history: [B, T, history_dim] 历史观测
    - vision_feature: [B, vision_feature_dim] 预提取的视觉特征
    """
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        # History encoder
        his_encoder_dims: list = [1024, 512, 128],
        his_latent_dim: int = 67,
        history_dim: int = 570,
        # Vision feature
        vision_feature_dim: int = 384,  # Depth Anything V2 ViT-S 输出
        vision_latent_dim: int = 128,   # 投影后维度
        use_vision_projection: bool = True,  # 是否使用投影层
        # Actor/Critic
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        max_grad_norm: float = 10.0,
        **kwargs
    ):
        if kwargs:
            print(f"ActorCriticVision.__init__ got unexpected arguments: {list(kwargs.keys())}")
        super().__init__()
        
        activation_fn = get_activation(activation)
        
        self.his_latent_dim = his_latent_dim
        self.vision_feature_dim = vision_feature_dim
        self.vision_latent_dim = vision_latent_dim
        self.use_vision_projection = use_vision_projection
        self.max_grad_norm = max_grad_norm
        
        # Vision feature projection (可训练)
        if use_vision_projection:
            self.vision_projection = nn.Sequential(
                nn.Linear(vision_feature_dim, vision_latent_dim),
                nn.LayerNorm(vision_latent_dim),
                nn.SiLU(),
            )
            vision_out_dim = vision_latent_dim
        else:
            self.vision_projection = nn.Identity()
            vision_out_dim = vision_feature_dim
        
        # Actor input: obs + history_feature + vision_feature
        mlp_input_dim_a = num_actor_obs + his_latent_dim + vision_out_dim
        # Critic input: critic_obs + history_feature (不使用视觉特征，使用特权信息)
        mlp_input_dim_c = num_critic_obs + his_latent_dim
        
        # History Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, his_encoder_dims[0]))
        encoder_layers.append(activation_fn)
        for l in range(len(his_encoder_dims)):
            if l == len(his_encoder_dims) - 1:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_latent_dim))
            else:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_encoder_dims[l + 1]))
                encoder_layers.append(activation_fn)
        self.history_encoder = nn.Sequential(*encoder_layers)
        
        # Actor MLP
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Critic MLP
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
        
        print(f"[ActorCriticVision] Initialized:")
        print(f"  Actor input: {mlp_input_dim_a} = obs({num_actor_obs}) + his({his_latent_dim}) + vision({vision_out_dim})")
        print(f"  Critic input: {mlp_input_dim_c} = critic_obs({num_critic_obs}) + his({his_latent_dim})")
        print(f"  Vision projection: {vision_feature_dim} -> {vision_out_dim}")
        print(f"  Actor MLP: {self.actor}")
        print(f"  Critic MLP: {self.critic}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, history, vision_feature=None, **kwargs):
        """
        采样动作
        
        Args:
            observations: [B, num_actor_obs] 本体感知
            history: [B, T, history_dim] 或 [B, T*history_dim] 历史
            vision_feature: [B, vision_feature_dim] 预提取的视觉特征
        """
        history = history.flatten(1) if history.dim() > 2 else history
        his_feature = self.history_encoder(history)
        
        if vision_feature is not None:
            vision_feat = self.vision_projection(vision_feature)
            actor_input = torch.cat((observations, his_feature, vision_feat), dim=-1)
        else:
            # 无视觉特征时，使用零填充
            vision_feat = torch.zeros(
                observations.shape[0], 
                self.vision_latent_dim if self.use_vision_projection else self.vision_feature_dim,
                device=observations.device
            )
            actor_input = torch.cat((observations, his_feature, vision_feat), dim=-1)
        
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, vision_feature=None, **kwargs):
        """推理时使用均值动作"""
        history = history.flatten(1) if history.dim() > 2 else history
        his_feature = self.history_encoder(history)
        
        if vision_feature is not None:
            vision_feat = self.vision_projection(vision_feature)
            actor_input = torch.cat((observations, his_feature, vision_feat), dim=-1)
        else:
            vision_feat = torch.zeros(
                observations.shape[0], 
                self.vision_latent_dim if self.use_vision_projection else self.vision_feature_dim,
                device=observations.device
            )
            actor_input = torch.cat((observations, his_feature, vision_feat), dim=-1)
        
        return self.actor(actor_input)
    
    def evaluate(self, critic_observations, history, **kwargs):
        """评估 value（Critic 不使用视觉特征）"""
        history = history.flatten(1) if history.dim() > 2 else history
        his_feature = self.history_encoder(history)
        critic_input = torch.cat((critic_observations, his_feature), dim=-1)
        return self.critic(critic_input)


class VisionFeatureBuffer:
    """
    视觉特征缓冲区
    
    用于实现降频特征提取：
    - encoder 每 N 步运行一次
    - 中间步骤复用缓存的特征
    
    模拟真实部署场景：
    - RGB 相机帧率可能低于控制频率
    - encoder 推理时间可能较长
    """
    
    def __init__(
        self,
        num_envs: int,
        feature_dim: int,
        update_interval: int = 5,
        device: str = 'cuda'
    ):
        """
        Args:
            num_envs: 环境数量
            feature_dim: 特征维度
            update_interval: 更新间隔（每 N 步更新一次特征）
            device: 设备
        """
        self.num_envs = num_envs
        self.feature_dim = feature_dim
        self.update_interval = update_interval
        self.device = device
        
        # 缓存的特征
        self.cached_features = torch.zeros(num_envs, feature_dim, device=device)
        # 步数计数器
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        
    def should_update(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """
        判断哪些环境需要更新特征
        
        Returns:
            mask: [num_envs] 布尔 tensor，True 表示需要更新
        """
        if env_ids is None:
            return (self.step_counter % self.update_interval) == 0
        else:
            return (self.step_counter[env_ids] % self.update_interval) == 0
    
    def update(self, new_features: torch.Tensor, env_ids: torch.Tensor = None):
        """
        更新特征缓存
        
        Args:
            new_features: [B, feature_dim] 新特征
            env_ids: 要更新的环境 ID，None 表示全部
        """
        if env_ids is None:
            self.cached_features = new_features
        else:
            self.cached_features[env_ids] = new_features
    
    def get_features(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """获取缓存的特征"""
        if env_ids is None:
            return self.cached_features
        else:
            return self.cached_features[env_ids]
    
    def step(self, dones: torch.Tensor = None):
        """
        推进一步
        
        Args:
            dones: [num_envs] 重置标志，重置的环境计数器归零
        """
        self.step_counter += 1
        if dones is not None:
            self.step_counter[dones] = 0
    
    def reset(self, env_ids: torch.Tensor = None):
        """重置缓冲区"""
        if env_ids is None:
            self.cached_features.zero_()
            self.step_counter.zero_()
        else:
            self.cached_features[env_ids] = 0
            self.step_counter[env_ids] = 0


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "silu":
        return nn.SiLU()
    else:
        print(f"Invalid activation function: {act_name}")
        return nn.ELU()
