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

"""
DWAQ On-Policy Runner for training with Variational Autoencoder.

This runner implements the DWAQ (Deep Variational Autoencoder for Walking) training loop,
which extends standard PPO with a β-VAE for blind walking on rough terrain.

Key Features:
1. Observation history for VAE encoder input
2. Previous critic observations for velocity estimation supervision
3. Autoencoder loss (velocity MSE + reconstruction MSE + KL divergence)
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms import DWAQPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic_DWAQ, EmpiricalNormalization
from rsl_rl.utils import store_code_state


class DWAQOnPolicyRunner:
    """On-policy runner for DWAQ training."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        """
        Initialize the DWAQ on-policy runner.

        Args:
            env: The vectorized environment with DWAQ-specific interfaces.
            train_cfg: Training configuration dictionary containing runner, algorithm, and policy configs.
            log_dir: Directory for saving logs and checkpoints.
            device: Device to run training on ('cpu' or 'cuda:X').
        """
        self.cfg = train_cfg
        self.train_cfg = train_cfg  # keep full config for saving metadata
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Configure multi-gpu (following amp_on_policy_runner pattern)
        self._configure_multi_gpu()

        # Resolve observation dimensions from environment
        num_obs = self.env.num_obs
        num_privileged_obs = self.env.num_privileged_obs if self.env.num_privileged_obs is not None else num_obs
        num_obs_hist = self.env.num_obs_hist

        # Calculate encoder dimensions
        cenet_in_dim = num_obs_hist * num_obs
        # cenet_out_dim: 3 (velocity) + 16 (latent) = 19 by default
        # Can be overridden via policy config
        cenet_out_dim = self.policy_cfg.get("cenet_out_dim", 19)

        # Evaluate the policy class
        policy_class_name = self.policy_cfg.get("class_name", "ActorCritic_DWAQ")
        policy_class = eval(policy_class_name)

        # Filter policy config to only include ActorCritic_DWAQ supported params
        dwaq_supported_params = ["activation", "init_noise_std"]
        filtered_policy_cfg = {k: v for k, v in self.policy_cfg.items() if k in dwaq_supported_params}

        # Create actor-critic with DWAQ architecture
        # Actor input: num_obs + cenet_out_dim (observation + latent code)
        # Critic input: num_privileged_obs (privileged observations)
        policy: ActorCritic_DWAQ = policy_class(
            num_actor_obs=num_obs + cenet_out_dim,
            num_critic_obs=num_privileged_obs,
            num_actions=self.env.num_actions,
            cenet_in_dim=cenet_in_dim,
            cenet_out_dim=cenet_out_dim,
            obs_dim=num_obs,
            **filtered_policy_cfg,
        ).to(self.device)

        # Initialize algorithm
        alg_class_name = self.alg_cfg.get("class_name", "DWAQPPO")
        alg_class = eval(alg_class_name)
        # DWAQPPO only accepts these parameters (different from standard PPO)
        dwaq_alg_supported_params = [
            "num_learning_epochs", "num_mini_batches", "clip_param",
            "gamma", "lam", "value_loss_coef", "entropy_coef",
            "learning_rate", "max_grad_norm", "use_clipped_value_loss",
            "schedule", "desired_kl",
        ]
        filtered_alg_cfg = {k: v for k, v in self.alg_cfg.items() if k in dwaq_alg_supported_params}
        self.alg: DWAQPPO = alg_class(
            policy=policy,
            device=self.device,
            obs_dim=num_obs,
            **filtered_alg_cfg,
        )

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Initialize storage
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            critic_obs_shape=[num_privileged_obs],
            obs_hist_shape=[cenet_in_dim],
            action_shape=[self.env.num_actions],
        )

        # Initialize observation normalizers
        # Note: DWAQ typically doesn't use empirical normalization, but we provide Identity
        # normalizers for compatibility with play.py export functions
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # Decide whether to disable logging (for multi-gpu)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging setup
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        Run the training loop for DWAQ.

        Args:
            num_learning_iterations: Number of training iterations to run.
            init_at_random_ep_len: Whether to initialize episode lengths randomly.
        """
        # Initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self._init_logger()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations
        obs, obs_hist = self.env.get_observations()
        privileged_obs, prev_critic_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs

        # Move to device
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        prev_critic_obs = prev_critic_obs.to(self.device)
        obs_hist = obs_hist.to(self.device)

        # Switch to train mode
        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced for multi-gpu
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Rollout phase
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions using DWAQ actor
                    actions = self.alg.act(obs, critic_obs, prev_critic_obs, obs_hist)

                    # Step the environment (Isaac Lab style: 4-tuple return)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    
                    # Extract DWAQ-specific data from extras
                    obs_dict = extras.get("observations", {})
                    privileged_obs = obs_dict.get("critic", None)
                    obs_hist = obs_dict.get("obs_hist", obs_hist)
                    prev_critic_obs = obs_dict.get("prev_critic_obs", prev_critic_obs)

                    # Update critic observations
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    # Move to device
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    prev_critic_obs = prev_critic_obs.to(self.device)
                    obs_hist = obs_hist.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Process environment step
                    self.alg.process_env_step(rewards, dones, extras)

                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            # Clone critic_obs to get a normal tensor (inference tensors cannot be used for backward)
            self.alg.compute_returns(critic_obs.clone())

            loss_dict = self.alg.update()
            mean_value_loss = loss_dict["value_function"]
            mean_surrogate_loss = loss_dict["surrogate"]
            mean_autoenc_loss = loss_dict["autoencoder"]
            stop = time.time()
            learn_time = stop - start

            self.current_learning_iteration = it

            # Logging
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state on first iteration
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if hasattr(self, "logger_type") and self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save final model
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Log training statistics."""
        # Compute collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size

        # Update totals
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Episode info logging
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/autoencoder", locs["mean_autoenc_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # Log policy stats
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training stats
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if not hasattr(self, "logger_type") or self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        # Console output
        header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Autoencoder loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Autoencoder loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        """Save model checkpoint."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save normalizer state if using empirical normalization
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload to external logging service if available
        if hasattr(self, "logger_type") and self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

        # Write run metadata on first save
        self._save_run_metadata()

    def load(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        # Resolve path if it's not a .pt file
        resolved_path = self._resolve_checkpoint_path(path)

        loaded_dict = torch.load(resolved_path, weights_only=False)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        # Load normalizer state if available
        if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])

        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        """Get the inference policy function."""
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self):
        """Switch to training mode."""
        self.alg.policy.train()

    def eval_mode(self):
        """Switch to evaluation mode."""
        self.alg.policy.eval()

    def add_git_repo_to_log(self, repo_file_path: str):
        """Add a git repository to be logged."""
        self.git_status_repos.append(repo_file_path)

    # ========================
    # Helper methods
    # ========================

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )

        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

    def _init_logger(self):
        """Initialize the logger (tensorboard, wandb, or neptune)."""
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()

        if self.logger_type == "neptune":
            from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def _save_run_metadata(self):
        """Save run metadata to JSON file."""
        try:
            if self.log_dir is not None:
                run_meta_path = os.path.join(self.log_dir, "run_meta.json")
                if not os.path.exists(run_meta_path):
                    meta = {
                        "created_at": time.time(),
                        "first_saved_iter": self.current_learning_iteration,
                        "train_cfg": self.train_cfg,
                        "env_meta": {
                            "num_envs": getattr(self.env, "num_envs", None),
                            "num_obs": getattr(self.env, "num_obs", None),
                            "num_obs_hist": getattr(self.env, "num_obs_hist", None),
                            "num_privileged_obs": getattr(self.env, "num_privileged_obs", None),
                            "num_actions": getattr(self.env, "num_actions", None),
                        },
                    }
                    with open(run_meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
        except Exception:
            pass

    def _resolve_checkpoint_path(self, path: str) -> str:
        """Resolve checkpoint path, handling JSON or non-.pt files."""
        try:
            ext = os.path.splitext(path)[1]
        except Exception:
            ext = ""

        if ext in [".json", ".txt"]:
            return self._find_latest_checkpoint(os.path.dirname(path) or ".")

        # Check if file starts with '{' or '[' (JSON)
        try:
            with open(path, "rb") as f:
                first = f.read(1)
                if first in [b"{", b"["]:
                    return self._find_latest_checkpoint(os.path.dirname(path) or ".")
        except FileNotFoundError:
            raise
        except Exception:
            pass

        return path

    def _find_latest_checkpoint(self, directory: str) -> str:
        """Find the latest model checkpoint in a directory."""
        candidates = [f for f in os.listdir(directory) if f.endswith(".pt") and "model" in f]
        if candidates:
            candidates.sort()
            return os.path.join(directory, candidates[-1])
        raise FileNotFoundError(f"No checkpoint .pt files found in directory: {directory}")
