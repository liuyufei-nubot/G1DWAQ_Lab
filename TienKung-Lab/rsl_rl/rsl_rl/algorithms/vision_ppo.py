# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# VisionPPO: PPO algorithm for ActorCriticVision with pre-extracted visual features.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticVision
from rsl_rl.storage.rollout_storage_vision import RolloutStorageVision


class VisionPPO:
    """
    PPO algorithm for ActorCriticVision with pre-extracted visual features.
    
    与 AMPPPOMulti 的区别：
    - 不包含 AMP (Adversarial Motion Prior)
    - 接收预提取的 vision_feature，而非 rgb_image
    - 更简洁的实现
    
    设计理念：
    - 视觉 encoder 在 Runner 层管理，与 Policy 解耦
    - Policy 只接收预提取的 vision_feature (384-dim CLS token)
    - Critic 不使用视觉特征，使用 privileged information
    """
    
    policy: ActorCriticVision
    """The actor critic module with vision feature support."""
    
    def __init__(
        self,
        policy: ActorCriticVision,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize VisionPPO algorithm.
        
        Args:
            policy: ActorCriticVision policy network
            num_learning_epochs: Number of epochs per update
            num_mini_batches: Number of mini-batches per epoch
            clip_param: PPO clipping parameter
            gamma: Discount factor
            lam: GAE lambda
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            learning_rate: Learning rate
            max_grad_norm: Maximum gradient norm for clipping
            use_clipped_value_loss: Whether to use clipped value loss
            schedule: Learning rate schedule ('fixed' or 'adaptive')
            desired_kl: Target KL divergence for adaptive schedule
            device: Device to run on
        """
        if kwargs:
            print(f"VisionPPO.__init__ got unexpected arguments: {list(kwargs.keys())}")
        
        self.device = device
        self.policy = policy
        self.policy.to(self.device)
        
        self.storage = None
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorageVision.Transition()
        
        # PPO hyperparameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.desired_kl = desired_kl

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list,
        critic_obs_shape: list,
        action_shape: list,
        history_len: int,
        history_dim: int,
        vision_feature_dim: int = 384
    ):
        """
        初始化 rollout storage.
        
        Args:
            num_envs: Number of parallel environments
            num_transitions_per_env: Rollout length per environment
            actor_obs_shape: Shape of actor observations
            critic_obs_shape: Shape of critic observations
            action_shape: Shape of actions
            history_len: Length of observation history
            history_dim: Dimension of each history step
            vision_feature_dim: Dimension of vision features (default 384 for ViT-S CLS token)
        """
        self.storage = RolloutStorageVision(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=action_shape,
            device=self.device,
            history_len=history_len,
            history_dim=history_dim,
            vision_feature_dim=vision_feature_dim
        )

    def test_mode(self):
        """Set policy to evaluation mode."""
        self.policy.eval()
    
    def train_mode(self):
        """Set policy to training mode."""
        self.policy.train()

    def act(
        self, 
        obs: torch.Tensor, 
        critic_obs: torch.Tensor, 
        history: torch.Tensor, 
        vision_feature: torch.Tensor = None
    ) -> torch.Tensor:
        """
        选择动作.
        
        Args:
            obs: [num_envs, num_obs] 观测
            critic_obs: [num_envs, num_critic_obs] Critic 观测
            history: [num_envs, history_len, history_dim] 历史
            vision_feature: [num_envs, vision_feature_dim] 预提取的视觉特征
            
        Returns:
            actions: [num_envs, num_actions] 选择的动作
        """
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        # Policy forward
        self.transition.actions = self.policy.act(
            obs.detach(), history, vision_feature=vision_feature
        ).detach()
        
        self.transition.values = self.policy.evaluate(
            critic_obs.detach(), history=history
        ).detach()
        
        if len(self.transition.actions.shape) > 2:
            self.transition.actions = self.transition.actions.reshape(
                self.transition.actions.shape[0], -1
            )
        
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        # Save current observations
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.history = history
        self.transition.vision_feature = vision_feature
        
        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        next_obs: torch.Tensor = None,
        next_critic_obs: torch.Tensor = None,
        **kwargs
    ):
        """
        处理环境 step 结果.
        
        Args:
            rewards: [num_envs] 奖励
            dones: [num_envs] 终止标志
            infos: 额外信息字典
            next_obs: [num_envs, num_obs] 下一观测（可选）
            next_critic_obs: [num_envs, num_critic_obs] 下一 Critic 观测（可选）
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Next observations
        if next_obs is not None:
            self.transition.next_observations = next_obs
        else:
            self.transition.next_observations = self.transition.observations
        
        if next_critic_obs is not None:
            self.transition.next_critic_observations = next_critic_obs
        else:
            self.transition.next_critic_observations = self.transition.critic_observations
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )
        
        # Record transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor, history: torch.Tensor):
        """
        计算 returns.
        
        Args:
            last_critic_obs: [num_envs, num_critic_obs] 最后一步的 Critic 观测
            history: [num_envs, history_len, history_dim] 历史
        """
        last_values = self.policy.evaluate(
            last_critic_obs.detach(), history=history
        ).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> dict:
        """
        更新 policy.
        
        Returns:
            loss_dict: 包含 value_function, surrogate, entropy 损失的字典
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        
        for batch in generator:
            (obs_batch, critic_obs_batch, actions_batch, next_obs_batch, 
             next_critic_obs_batch, history_batch, values_batch, advantages_batch, 
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch, vision_feature_batch) = batch
            
            # Forward pass
            self.policy.act(
                obs_batch.detach(), 
                history_batch.detach(), 
                vision_feature=vision_feature_batch.detach()
            )
            
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch.detach(), 
                history=history_batch.detach()
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Adaptive KL penalty
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / 
                        (2.0 * torch.square(sigma_batch)) - 0.5, 
                        axis=-1
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = values_batch + (value_batch - values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Total loss
            loss = (
                surrogate_loss +
                self.value_loss_coef * value_loss -
                self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
