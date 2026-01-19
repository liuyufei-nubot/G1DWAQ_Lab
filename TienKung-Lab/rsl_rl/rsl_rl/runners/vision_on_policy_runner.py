# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# Vision On-Policy Runner with Pre-extracted Visual Features
#
# 设计理念：
# 1. 视觉 encoder (DepthAnythingEncoder) 在 Runner 层管理，与 Policy 解耦
# 2. 支持降频特征提取（每 N 步更新一次），模拟真实部署场景
# 3. Policy 只接收预提取的 vision_feature，不包含 CNN

from __future__ import annotations

import os
import time
import statistics
from collections import deque
from datetime import datetime

import numpy as np
import torch

from rsl_rl.algorithms import VisionPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticVision, EmpiricalNormalization


class VisionOnPolicyRunner:
    """
    On-policy runner with pre-extracted visual features.
    
    使用 DepthAnythingEncoder 提取视觉特征，
    VisionFeatureManager 实现降频更新，
    ActorCriticVision 接收预提取特征进行决策。
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = 'cpu'
    ):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        # Configure multi-GPU
        self._configure_multi_gpu()
        
        # Get observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        
        # Resolve privileged observations
        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"
            num_privileged_obs = extras["observations"]["critic"].shape[1]
        else:
            self.privileged_obs_type = None
            num_privileged_obs = num_obs
        
        # Get history length
        self.obs_history_len = getattr(self.env, 'obs_history_len', 1)
        
        # Vision encoder configuration
        self.vision_cfg = train_cfg.get("vision", {})
        self.use_vision = self.vision_cfg.get("enable", False)
        self.vision_feature_dim = self.vision_cfg.get("feature_dim", 384)
        self.vision_update_interval = self.vision_cfg.get("update_interval", 5)
        
        # Initialize vision encoder if enabled
        self.vision_encoder = None
        self.vision_manager = None
        if self.use_vision and hasattr(self.env, 'rgb_camera') and self.env.rgb_camera is not None:
            self._init_vision_encoder()
        
        # Build policy
        policy_cfg = dict(self.policy_cfg)
        policy_class_name = policy_cfg.pop("class_name", "ActorCriticVision")
        
        # Add vision feature dim to policy config
        policy_cfg["vision_feature_dim"] = self.vision_feature_dim
        
        policy_class = eval(policy_class_name)
        actor_critic: ActorCriticVision = policy_class(
            num_actor_obs=num_obs,
            num_critic_obs=num_privileged_obs,
            num_actions=self.env.num_actions,
            history_dim=self.obs_history_len * num_obs,
            **policy_cfg
        ).to(self.device)
        
        # Initialize algorithm
        alg_cfg = dict(self.alg_cfg)
        alg_class_name = alg_cfg.pop("class_name", "VisionPPO")
        
        # Use VisionPPO by default
        self.alg = VisionPPO(
            policy=actor_critic,
            device=self.device,
            **alg_cfg
        )
        
        # Store configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        
        # Empirical normalization
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1.0e8
            ).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[num_privileged_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)
        
        # Initialize storage
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            critic_obs_shape=[num_privileged_obs],
            action_shape=[self.env.num_actions],
            history_len=self.obs_history_len,
            history_dim=num_obs,
            vision_feature_dim=self.vision_feature_dim
        )
        
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        print(f"[VisionOnPolicyRunner] Initialized:")
        print(f"  num_obs: {num_obs}, num_privileged_obs: {num_privileged_obs}")
        print(f"  use_vision: {self.use_vision}")
        if self.use_vision:
            print(f"  vision_feature_dim: {self.vision_feature_dim}")
            print(f"  vision_update_interval: {self.vision_update_interval}")

    def _init_vision_encoder(self):
        """Initialize vision encoder and manager."""
        try:
            from legged_lab.modules import DepthAnythingEncoder, VisionFeatureManager
            
            encoder_type = self.vision_cfg.get("encoder", "vits")
            checkpoint_path = self.vision_cfg.get("checkpoint_path", None)
            
            self.vision_encoder = DepthAnythingEncoder(
                encoder=encoder_type,
                checkpoint_path=checkpoint_path,
                freeze_encoder=True,
                use_projection=False,  # Let policy do projection
                feature_type=self.vision_cfg.get("feature_type", "cls"),
                device=self.device,
                verbose=True
            )
            
            self.vision_manager = VisionFeatureManager(
                encoder=self.vision_encoder,
                num_envs=self.env.num_envs,
                update_interval=self.vision_update_interval,
                device=self.device
            )
            
            self.vision_feature_dim = self.vision_encoder.get_output_dim()
            print(f"[VisionOnPolicyRunner] Vision encoder initialized: {encoder_type}")
            
        except ImportError as e:
            print(f"[VisionOnPolicyRunner] Warning: Could not import vision modules: {e}")
            self.use_vision = False

    def _get_vision_features(self, dones: torch.Tensor = None, force_update: bool = False) -> torch.Tensor:
        """Get vision features from camera."""
        if not self.use_vision or self.vision_manager is None:
            return torch.zeros(
                self.env.num_envs, self.vision_feature_dim, device=self.device
            )
        
        # Get RGB image from camera
        rgb_raw = self.env.rgb_camera.data.output["rgb"]
        if rgb_raw.shape[-1] == 4:
            rgb_raw = rgb_raw[..., :3]  # RGBA -> RGB
        
        # VisionFeatureManager handles frequency control
        features = self.vision_manager.step(
            rgb_raw, 
            dones=dones, 
            force_update=force_update
        )
        
        return features

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop."""
        # Initialize logger
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            logger_type = self.cfg.get("logger", "tensorboard").lower()
            
            if logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            else:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        
        # Get initial observations
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs = self.obs_normalizer(obs.to(self.device))
        privileged_obs = self.privileged_obs_normalizer(privileged_obs.to(self.device))
        
        self.train_mode()
        
        # Initialize trajectory history
        self.trajectory_history = torch.zeros(
            self.env.num_envs, self.obs_history_len, self.num_obs, device=self.device
        )
        self.trajectory_history = torch.cat(
            (self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1
        )
        
        # Initialize vision features
        vision_features = self._get_vision_features(force_update=True)
        
        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    history = self.trajectory_history
                    
                    # Get action
                    actions = self.alg.act(obs, privileged_obs, history, vision_features)
                    
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs = self.obs_normalizer(obs.to(self.device))
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs
                    
                    # Update vision features (VisionFeatureManager handles frequency)
                    vision_features = self._get_vision_features(dones=dones)
                    
                    # Process env step
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    # Update trajectory history
                    reset_env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[reset_env_ids] = 0
                    self.trajectory_history = torch.cat(
                        (self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1
                    )
                    
                    # Book keeping
                    if self.log_dir is not None and not self.disable_logs:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        elif 'log' in infos:
                            ep_infos.append(infos['log'])
                        
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
            
            stop = time.time()
            collection_time = stop - start
            start = stop
            
            # Compute returns
            history = self.trajectory_history.clone()
            self.alg.compute_returns(privileged_obs.clone(), history)
            
            # Update policy
            loss_dict = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # Log
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            
            ep_infos.clear()
        
        # Save final model
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Log training statistics."""
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to writer and terminal - check if key contains '/'
                if self.writer is not None:
                    if "/" in key:
                        self.writer.add_scalar(key, value, locs['it'])
                    else:
                        self.writer.add_scalar('Episode/' + key, value, locs['it'])
                if "/" in key:
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.std.mean()
        fps = int(collection_size / (locs['collection_time'] + locs['learn_time']))

        # Log losses from loss_dict
        loss_dict = locs.get('loss_dict', {})
        if self.writer is not None:
            for key, value in loss_dict.items():
                self.writer.add_scalar('Loss/' + key, value, locs['it'])

            # Log other metrics
            self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
            self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
            self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
            self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
            self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
            if len(locs['rewbuffer']) > 0:
                self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
                self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            
            # Log vision-specific metrics if available
            if self.use_vision and self.vision_manager is not None:
                self.writer.add_scalar('Vision/update_interval', self.vision_update_interval, locs['it'])
                self.writer.add_scalar('Vision/feature_dim', self.vision_feature_dim, locs['it'])

        # Print log string
        str_title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_title.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            # -- Losses
            for key, value in loss_dict.items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- Episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_title.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            for key, value in loss_dict.items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
                       f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['tot_iter'] - locs['it'])))}\n""")
        print(log_string)

    def save(self, path: str, infos: dict | None = None):
        """Save model checkpoint."""
        saved_dict = {
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.empirical_normalization:
            saved_dict['obs_norm_state_dict'] = self.obs_normalizer.state_dict()
            saved_dict['privileged_obs_norm_state_dict'] = self.privileged_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.policy.load_state_dict(loaded_dict['model_state_dict'])
        if self.empirical_normalization:
            if 'obs_norm_state_dict' in loaded_dict:
                self.obs_normalizer.load_state_dict(loaded_dict['obs_norm_state_dict'])
            if 'privileged_obs_norm_state_dict' in loaded_dict:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict['privileged_obs_norm_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos')

    def get_inference_policy(self, device: torch.device | None = None):
        """Get inference policy for deployment."""
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        if self.empirical_normalization:
            if device is not None:
                self.obs_normalizer.to(device)
            return lambda obs, history, vision_feature: self.alg.policy.act_inference(
                self.obs_normalizer(obs), history, vision_feature=vision_feature
            )
        else:
            return self.alg.policy.act_inference

    def train_mode(self):
        self.alg.policy.train()

    def eval_mode(self):
        self.alg.policy.eval()

    def _configure_multi_gpu(self):
        """Configure multi-GPU training."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
        else:
            self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.gpu_global_rank = int(os.getenv("RANK", "0"))
        
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
