# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticDepth
from rsl_rl.storage import RolloutStorageEX as RolloutStorage
from rsl_rl.storage.replay_buffer_multi import ReplayBufferMulti


class AMPPPOMulti:
    """PPO algorithm with optional AMP support for RGB and history-based observations."""

    policy: ActorCriticDepth
    """The actor critic module."""

    def __init__(self,
                 policy,
                 discriminator=None,
                 amp_data=None,
                 amp_normalizer=None,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 disc_learning_rate=2e-5,
                 policy_learning_rate=None,
                 learning_rate=2e-5,  # Isaac Lab style parameter name
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100000,
                 amp_loader_type='16dof',
                 num_amp_frames=2,
                 use_amp=False,
                 use_rgb=False,
                 default_pos=None,
                 **kwargs
                 ):
        self.use_amp = use_amp
        self.amp_loader_type = amp_loader_type
        self.num_amp_frames = num_amp_frames
        self.use_rgb = use_rgb
        self.device = device
        if default_pos is not None: 
            self.default_pos = torch.tensor(default_pos, device=self.device)
        
        # Support both policy_learning_rate and learning_rate parameter names
        if policy_learning_rate is not None:
            self.policy_learning_rate = policy_learning_rate
        else:
            self.policy_learning_rate = learning_rate

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.disc_learning_rate = disc_learning_rate

        # Discriminator components
        if self.use_amp:
            print("**************** train with AMP ****************") 
            self.discriminator = discriminator
            self.discriminator.to(self.device)
            self.amp_transition = RolloutStorage.Transition()
            self.amp_storage = ReplayBufferMulti(
                discriminator.state_dim, amp_replay_buffer_size, self.num_amp_frames, device)
            self.amp_data = amp_data
            self.amp_normalizer = amp_normalizer
            self.optimizer_disc = optim.AdamW(self.discriminator.parameters(), lr=self.disc_learning_rate, weight_decay=1e-2)
        else:
            self.discriminator = None
            self.amp_normalizer = None
            print("**************** train without AMP ****************") 

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None # initialized later

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.policy_learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    @property
    def learning_rate(self):
        """Property for Isaac Lab compatibility."""
        return self.policy_learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, history_len, history_dim, rgb_shape=None):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device, history_len, history_dim, rgb_shape)

    def test_mode(self):
        self.policy.eval()
    
    def train_mode(self):
        self.policy.train()

    def act(self, obs, critic_obs, history):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        if isinstance(obs, tuple):
            aug_obs, rgb_image, aug_critic_obs = obs[0].detach(), obs[1].detach(), critic_obs.detach()
            self.transition.actions = self.policy.act(aug_obs, history, rgb_image).detach()
            self.transition.observations = obs[0]
            self.transition.rgb_image = obs[1]
        else:
            aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
            self.transition.actions = self.policy.act(aug_obs, history).detach()
            self.transition.observations = obs
        
        self.transition.values = self.policy.evaluate(aug_critic_obs, history=history).detach()
        if len(self.transition.actions.shape) > 2:
            self.transition.actions = self.transition.actions.reshape(self.transition.actions.shape[0], -1)
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.history = history
        self.transition.critic_observations = critic_obs
        return self.transition.actions
        
    def process_env_step(self, rewards, dones, infos, next_obs=None, next_critic_obs=None, amp_obs_frames=None, **kwargs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Use current observations as next observations if not provided
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
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        if amp_obs_frames is not None:
            self.amp_storage.insert(amp_obs_frames)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
    
    def compute_returns(self, last_critic_obs, history):
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = self.policy.evaluate(aug_last_critic_obs, history).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc
    
    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits, device=self.device))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits, device=self.device))
        return loss

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_agent_acc = 0
        mean_demo_acc = 0
        
        if self.policy.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        if self.use_amp:
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env //
                    self.num_mini_batches)
            if self.amp_loader_type == 'lafan_16dof_multi':
                amp_expert_generator = self.amp_data.feed_forward_generator_lafan_16dof_multi(
                    self.num_learning_epochs * self.num_mini_batches,
                    self.storage.num_envs * self.storage.num_transitions_per_env //
                        self.num_mini_batches)
            else:
                NotImplementedError()
                
            for sample_amp_policy, sample_amp_expert in zip(amp_policy_generator, amp_expert_generator):
                    
                expert_states = sample_amp_expert
                policy_states = sample_amp_policy
                # Discriminator loss.
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        expert_states = self.amp_normalizer.normalize_torch(expert_states.to(self.device), self.device)
                        policy_states = self.amp_normalizer.normalize_torch(policy_states, self.device)
                policy_d = self.discriminator(policy_states.flatten(1))
                expert_states = expert_states.to(self.device)
                expert_d = self.discriminator(expert_states.flatten(1))
                agent_acc, demo_acc = self._compute_disc_acc(policy_d, expert_d)
                # prediction loss
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                
                # grad penalty
                grad_pen_loss = self.discriminator.compute_grad_pen(expert_states, lambda_=5)
                
                # logit reg
                logit_weights = self.discriminator.get_disc_logit_weights()
                disc_logit_loss = torch.sum(torch.square(logit_weights))
                disc_logit_loss = 0.01 * disc_logit_loss

                # weight decay
                disc_weights = self.discriminator.get_disc_weights()
                disc_weights = torch.cat(disc_weights, dim=-1)
                disc_weight_decay = torch.sum(torch.square(disc_weights))
                disc_weight_decay = 0.0001 * disc_weight_decay

                disc_loss = amp_loss + grad_pen_loss + disc_logit_loss + disc_weight_decay
                self.optimizer_disc.zero_grad()
                disc_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.optimizer_disc.step()
                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_states.cpu().numpy())
                    self.amp_normalizer.update(expert_states.cpu().numpy())

                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
                mean_agent_acc += agent_acc.mean().item()
                mean_demo_acc += demo_acc.mean().item()
        
        for obs_batch, critic_obs_batch, actions_batch, next_obs_batch, next_critic_observations_batch, history_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rgb_image_batch, *_ in generator:

            aug_obs_batch, history_batch = obs_batch.detach(), history_batch.detach()
            if self.use_rgb:
                aug_rgb_image_batch = rgb_image_batch.detach()
                self.policy.act(aug_obs_batch, history_batch, aug_rgb_image_batch)
            else:
                self.policy.act(obs_batch, history_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.policy.evaluate(aug_critic_obs_batch, history=history_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.policy_learning_rate = max(1e-5, self.policy_learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.policy_learning_rate = min(1e-2, self.policy_learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.policy_learning_rate
                
            # Bound loss
            soft_bound = 1.0
            mu_loss_high = torch.maximum(mu_batch - soft_bound,
                                        torch.tensor(0, device=self.device)) ** 2
            mu_loss_low = torch.minimum(mu_batch + soft_bound,
                                        torch.tensor(0, device=self.device)) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
            b_loss = b_loss.mean()

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            # Compute total loss.
            loss = (
                0 * b_loss +
                surrogate_loss +
                self.value_loss_coef * value_loss -
                self.entropy_coef * entropy_batch.mean())

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            if self.device != 'cuda:0':
                for param in self.optimizer.state.values():
                    if isinstance(param, torch.Tensor):
                        pass
                        # param.data = param.data.to(self.device)
                    elif isinstance(param, dict):
                        for k, v in param.items():
                            if isinstance(v, torch.Tensor):
                                if k == "step":
                                    param[k] = v.to('cpu')
            self.optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_agent_acc /= num_updates
        mean_demo_acc /= num_updates
        
        self.storage.clear()

        # Return loss_dict in Isaac Lab style
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.use_amp:
            loss_dict["amp_loss"] = mean_amp_loss
            loss_dict["amp_grad_pen"] = mean_grad_pen_loss
            loss_dict["amp_policy_pred"] = mean_policy_pred
            loss_dict["amp_expert_pred"] = mean_expert_pred
            loss_dict["amp_agent_acc"] = mean_agent_acc
            loss_dict["amp_demo_acc"] = mean_demo_acc
        
        return loss_dict
