# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Rollout Storage for Vision-based Policy with Pre-extracted Features
#
# 设计理念：
# 1. 存储预提取的视觉特征，而非原始 RGB 图像
# 2. 支持降频特征提取（每 N 步更新一次）
# 3. 减少显存占用：特征向量 << 原始图像

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorageVision:
    """
    Rollout Storage for Vision-based Policy with Pre-extracted Features.
    
    与 RolloutStorageEX 的区别：
    - RolloutStorageEX: 存储 rgb_image [num_envs, H, W, 3]
    - RolloutStorageVision: 存储 vision_feature [num_envs, feature_dim]
    
    优势：
    - 显存占用大幅减少：384 floats vs 480*640*3 uint8
    - 支持任意视觉 encoder 的预提取特征
    """
    
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            
            # Vision features (pre-extracted)
            self.vision_feature = None
            
            # History
            self.history = None
            
            # For next step
            self.next_observations = None
            self.next_critic_observations = None
        
        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple,
        privileged_obs_shape: tuple,
        actions_shape: tuple,
        device: str = 'cpu',
        history_len: int = 10,
        history_dim: int = 45,
        vision_feature_dim: int = 384,  # Depth Anything V2 ViT-S output
        next_obs_shape: tuple = None,
        num_critics: int = 1
    ):
        """
        Args:
            num_envs: 环境数量
            num_transitions_per_env: 每个环境的 transition 数量
            obs_shape: 观测 shape
            privileged_obs_shape: 特权观测 shape
            actions_shape: 动作 shape
            device: 设备
            history_len: 历史长度
            history_dim: 历史维度
            vision_feature_dim: 视觉特征维度 (e.g., 384 for ViT-S)
            next_obs_shape: 下一步观测 shape
            num_critics: Critic 数量
        """
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.num_critics = num_critics
        self.vision_feature_dim = vision_feature_dim

        # Core observations
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        if next_obs_shape is None:
            self.next_observations = torch.zeros(
                num_transitions_per_env, num_envs, *obs_shape, device=self.device
            )
        else:
            self.next_observations = torch.zeros(
                num_transitions_per_env, num_envs, *next_obs_shape, device=self.device
            )
        
        # History
        self.history = torch.zeros(
            num_transitions_per_env, num_envs, history_len, history_dim, device=self.device
        )
        
        # Privileged observations
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
            self.next_critic_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
            self.next_critic_observations = torch.zeros(
                num_transitions_per_env, num_envs, *obs_shape, device=self.device
            )
        
        # Vision features (pre-extracted, not raw images)
        self.vision_features = torch.zeros(
            num_transitions_per_env, num_envs, vision_feature_dim, device=self.device
        )
        
        # Rewards and dones
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, num_critics, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()
        
        # Actions
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        
        # PPO quantities
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, num_critics, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, num_critics, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, num_critics, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        # RNN hidden states
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0
        
        # Print storage info
        vision_mem = vision_feature_dim * 4 * num_envs * num_transitions_per_env / (1024 * 1024)
        print(f"[RolloutStorageVision] Initialized:")
        print(f"  num_envs: {num_envs}, steps_per_env: {num_transitions_per_env}")
        print(f"  vision_feature_dim: {vision_feature_dim}")
        print(f"  Vision feature memory: {vision_mem:.2f} MB")

    def add_transitions(self, transition: Transition):
        """添加一个 transition 到 storage"""
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.next_critic_observations[self.step].copy_(transition.next_critic_observations)
        
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        
        self.actions[self.step].copy_(transition.actions)
        self.history[self.step].copy_(transition.history)
        
        # Vision features
        if transition.vision_feature is not None:
            self.vision_features[self.step].copy_(transition.vision_feature)
        
        self.rewards[self.step].copy_(transition.rewards.view(-1, self.num_critics))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """保存 RNN hidden states"""
        if hidden_states is None or hidden_states == (None, None):
            return
        
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device)
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device)
                for i in range(len(hid_c))
            ]
        
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        """清空 storage"""
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """计算 returns 和 advantages (GAE)"""
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Normalize advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        """获取 episode 统计信息"""
        done = self.dones.clone()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """生成 mini-batch 用于训练"""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        # Flatten tensors
        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        history = self.history.flatten(0, 1)
        vision_features = self.vision_features.flatten(0, 1)

        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
            next_critic_observations = self.next_critic_observations.flatten(0, 1)
        else:
            critic_observations = observations
            next_critic_observations = next_observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                critic_obs_batch = critic_observations[batch_idx]
                next_critic_obs_batch = next_critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                history_batch = history[batch_idx]
                vision_feature_batch = vision_features[batch_idx]

                yield (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    next_obs_batch,
                    next_critic_obs_batch,
                    history_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (None, None),  # hidden states
                    None,          # masks
                    vision_feature_batch  # vision features instead of rgb_image
                )

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """生成 mini-batch 用于 RNN 训练"""
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.privileged_observations, self.dones
            )
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        # Pad vision features
        padded_vision_features, _ = split_and_pad_trajectories(
            self.vision_features.unsqueeze(-1), self.dones  # Add dummy dim for compatibility
        )
        padded_vision_features = padded_vision_features.squeeze(-1)

        mini_batch_size = self.num_envs // num_mini_batches
        
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                vision_feature_batch = padded_vision_features[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # Handle hidden states
                if self.saved_hidden_states_a is not None:
                    last_was_done_perm = last_was_done.permute(1, 0)
                    hid_a_batch = [
                        saved.permute(2, 0, 1, 3)[last_was_done_perm][first_traj:last_traj].transpose(1, 0).contiguous()
                        for saved in self.saved_hidden_states_a
                    ]
                    hid_c_batch = [
                        saved.permute(2, 0, 1, 3)[last_was_done_perm][first_traj:last_traj].transpose(1, 0).contiguous()
                        for saved in self.saved_hidden_states_c
                    ]
                    hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                    hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch
                else:
                    hid_a_batch, hid_c_batch = None, None

                yield (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (hid_a_batch, hid_c_batch),
                    masks_batch,
                    vision_feature_batch
                )

                first_traj = last_traj
