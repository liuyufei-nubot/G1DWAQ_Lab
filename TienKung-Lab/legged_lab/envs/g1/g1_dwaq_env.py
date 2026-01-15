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
G1 DWAQ Environment - Blind Walking with Variational Autoencoder

This environment implements DWAQ (Deep Variational Autoencoder for Walking) training
for the G1 robot. It follows the Isaac Lab interface style where step() returns
(obs, rewards, dones, extras), while providing DWAQ-specific data through
get_observations() and extras dict.

Key DWAQ Components:
1. Observation History Buffer: Maintains dwaq_obs_history_length frames for VAE encoder
2. Previous Critic Observations: Used for velocity estimation supervision
3. Isaac Lab Compatible: step() returns 4 values, DWAQ data in extras

Interface:
- step(actions) -> (obs, rewards, dones, extras)
  - extras["observations"]["critic"] = privileged observations
  - extras["observations"]["obs_hist"] = observation history for VAE encoder
  - extras["observations"]["prev_critic_obs"] = previous critic obs for velocity loss
- get_observations() -> (obs, extras)  
- get_privileged_observations() -> (critic_obs, prev_critic_obs)

Reference: DreamWaQ (https://github.com/Kingspider652/DreamWaQ-blind)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.utils import math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer


from legged_lab.envs.g1.g1_dwaq_config import G1DwaqEnvCfg

from legged_lab.envs.g1.g1_config import G1FlatEnvCfg, G1RoughEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv


class G1DwaqEnv(VecEnv):
    """
    G1 DWAQ environment for blind walking with β-VAE.
    
    This environment follows Isaac Lab conventions:
    - step() returns (obs, rewards, dones, extras)
    - DWAQ-specific data (obs_hist, prev_critic_obs) passed via extras
    
    The VAE encoder uses observation history to infer hidden environment
    states (friction, terrain, etc.) for blind walking on rough terrain.
    """

    def __init__(
        self,
        cfg: G1DwaqEnvCfg,
        headless: bool,
    ):
        self.cfg: G1DwaqEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dt = self.step_dt  # Alias for runner compatibility
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)
        
        # DWAQ-specific: Observation history length for VAE encoder
        self.dwaq_obs_history_length = getattr(self.cfg.robot, "dwaq_obs_history_length", 5)

        # Initialize simulation context
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        # Build scene
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        # Get robot and sensors
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # Instantiate LiDAR if enabled
        if hasattr(self.cfg.scene, "lidar") and self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]

        # Instantiate Depth Camera if enabled
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        # # Instantiate RGB Camera if enabled
        # if hasattr(self.cfg.scene, "rgb_camera") and self.cfg.scene.rgb_camera.enable_rgb_camera:
        #     self.rgb_camera: RgbCamera = self.scene.sensors["rgb_camera"]
        # else:
        #     self.rgb_camera = None

        # Command generator
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()
        # Debug: saving initial camera frames
        # Number of frames to save from the start (per env). Set to 0 to disable.
        self.debug_save_num = getattr(self.cfg, "debug_save_num", 10)
        self._debug_saved_frames = 0
        # directory to save debug frames (relative to workspace)
        self._debug_output_dir = getattr(self.cfg, "debug_output_dir", "outputs/debug_camera_frames")
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

    def init_buffers(self):
        """Initialize all internal buffers for the environment."""
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        # Initialize feet state buffers for privileged info
        num_feet = len(self.feet_cfg.body_ids)
        self.feet_pos_in_body = torch.zeros(self.num_envs, num_feet, 3, device=self.device)
        self.feet_vel_in_body = torch.zeros(self.num_envs, num_feet, 3, device=self.device)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.init_obs_buffer()

    def compute_current_observations(self):
        """Compute current step observations for actor and critic."""
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        
        # Build critic obs with privileged information (same as g1_rgb_env)
        critic_obs_list = [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact]
        
        # Compute feet info for privileged observations
        priv_cfg = self.cfg.scene.privileged_info
        if priv_cfg.enable_feet_info or priv_cfg.enable_feet_contact_force:
            self._compute_feet_state()
        
        # Add feet position and velocity in body frame (12 dim)
        if priv_cfg.enable_feet_info:
            critic_obs_list.append(self.feet_pos_in_body.reshape(self.num_envs, -1) * self.obs_scales.feet_pos)
            critic_obs_list.append(self.feet_vel_in_body.reshape(self.num_envs, -1) * self.obs_scales.feet_vel)
        
        # Add feet contact force 3D (6 dim for 2 feet)
        if priv_cfg.enable_feet_contact_force:
            feet_force = net_contact_forces[:, -1, self.feet_cfg.body_ids, :]  # (num_envs, num_feet, 3)
            critic_obs_list.append(feet_force.reshape(self.num_envs, -1) * self.obs_scales.contact_force)
        
        # Add root height (1 dim)
        if priv_cfg.enable_root_height:
            root_height = robot.data.root_pos_w[:, 2:3]  # (num_envs, 1)
            critic_obs_list.append(root_height)
        
        current_critic_obs = torch.cat(critic_obs_list, dim=-1)

        return current_actor_obs, current_critic_obs

    def _compute_feet_state(self):
        """Compute feet position and velocity in body frame."""
        robot = self.robot
        
        # Get feet body IDs
        feet_body_ids = self.feet_cfg.body_ids
        
        # Get feet positions in world frame
        feet_pos_w = robot.data.body_pos_w[:, feet_body_ids, :]  # (num_envs, num_feet, 3)
        feet_vel_w = robot.data.body_lin_vel_w[:, feet_body_ids, :]  # (num_envs, num_feet, 3)
        
        # Get root state
        root_pos_w = robot.data.root_pos_w  # (num_envs, 3)
        root_vel_w = robot.data.root_lin_vel_w  # (num_envs, 3)
        root_quat_w = robot.data.root_quat_w  # (num_envs, 4) in (w, x, y, z) format
        
        # Translate to root frame
        feet_pos_translated = feet_pos_w - root_pos_w.unsqueeze(1)  # (num_envs, num_feet, 3)
        feet_vel_translated = feet_vel_w - root_vel_w.unsqueeze(1)  # (num_envs, num_feet, 3)
        
        # Rotate to body frame using Isaac Lab's quat_apply_inverse
        num_feet = feet_pos_translated.shape[1]
        for i in range(num_feet):
            self.feet_pos_in_body[:, i, :] = math_utils.quat_apply_inverse(root_quat_w, feet_pos_translated[:, i, :])
            self.feet_vel_in_body[:, i, :] = math_utils.quat_apply_inverse(root_quat_w, feet_vel_translated[:, i, :])

    def _get_current_critic_obs_with_height_scan(self):
        """
        Get current critic observations with height scan (if enabled).
        This method does NOT update any buffers - use for prev_critic_obs.
        """
        _, current_critic_obs = self.compute_current_observations()
        
        # Add height scan if enabled (critic always gets height scan)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            current_critic_obs = torch.cat([current_critic_obs, height_scan], dim=-1)
        
        return torch.clip(current_critic_obs, -self.clip_obs, self.clip_obs)

    def compute_observations(self):
        """Compute full observations including history and sensor data."""
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)
        
        # DWAQ: Also append to observation history buffer for VAE encoder
        # Use raw observations (without height scan) for consistent encoder input
        self.dwaq_obs_history_buffer.append(current_actor_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        # Height scanner observations
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            # Critic always gets height_scan (privileged information)
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            # Actor only gets height_scan if critic_only=False (symmetric AC)
            if not self.cfg.scene.height_scanner.critic_only:
                if self.add_noise:
                    height_scan = height_scan + (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
                actor_obs = torch.cat([actor_obs, height_scan], dim=-1)
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        """Reset specified environments."""
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.dwaq_obs_history_buffer.reset(env_ids)  # DWAQ: Reset observation history
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        
        # DWAQ: Reset previous critic obs for reset environments
        # Fill with zeros to avoid using stale data
        self.prev_critic_obs[env_ids] = 0.0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        """
        Execute one environment step.
        
        Returns:
            obs: Actor observations [num_envs, num_obs]
            rewards: Reward values [num_envs]
            dones: Done flags [num_envs]
            extras: Dict containing:
                - observations.critic: Privileged observations
                - observations.obs_hist: DWAQ observation history for VAE encoder
                - observations.prev_critic_obs: Previous critic obs for velocity loss
                - time_outs: Timeout flags
                - log: Episode statistics
        
        This follows Isaac Lab convention: step() returns 4 values.
        DWAQ-specific data is passed through extras dict.
        
        DWAQ Timing (matching original DreamWaQ):
        1. At step start: save current critic_obs as prev_critic_obs (WITHOUT updating buffers)
        2. Execute action
        3. At step end: compute new observations (updates buffers)
        4. Return: new obs + prev_critic_obs from step start
        """
        # Store previous critic obs BEFORE stepping (for velocity estimation loss)
        # Use _get_current_critic_obs_with_height_scan to avoid double-appending to buffers
        self.prev_critic_obs = self._get_current_critic_obs_with_height_scan().clone()
        
        delayed_actions = self.action_buffer.compute(actions)

        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = clipped_actions * self.action_scale + self.robot.data.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1

        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        
        # DWAQ: Get observation history for VAE encoder
        obs_hist = self.dwaq_obs_history_buffer.buffer.reshape(self.num_envs, -1)
        
        # Pack extras following Isaac Lab convention
        self.extras["observations"] = {
            "critic": critic_obs,
            "obs_hist": obs_hist,  # DWAQ: for VAE encoder
            "prev_critic_obs": self.prev_critic_obs,  # DWAQ: for velocity loss
        }
        self.extras["time_outs"] = self.time_out_buf

        # Isaac Lab style: return 4 values
        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        """Check termination conditions for all environments."""
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        """Initialize observation buffers and noise vectors for DWAQ."""
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0  # commands
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0  # actions
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        # Standard observation buffers (for actor/critic history)
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        
        # DWAQ-specific: Observation history buffer for VAE encoder
        # This stores dwaq_obs_history_length frames of raw actor observations
        self.dwaq_obs_history_buffer = CircularBuffer(
            max_len=self.dwaq_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        
        # DWAQ-specific: Previous critic observations for velocity estimation loss
        # Initialize with zeros, will be updated each step before stepping
        actor_obs, critic_obs = self.compute_current_observations()
        self._num_obs = actor_obs.shape[-1]
        self._num_privileged_obs = critic_obs.shape[-1]
        
        # Calculate full privileged obs dimension (including height scan)
        height_scan_dim = 0
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan_dim = self.height_scanner.data.ray_hits_w.shape[1]
        full_privileged_obs_dim = self._num_privileged_obs + height_scan_dim
        
        # Initialize prev_critic_obs with FULL privileged obs dimension (including height scan)
        # This is required because storage is initialized with num_privileged_obs property
        self.prev_critic_obs = torch.zeros(self.num_envs, full_privileged_obs_dim, device=self.device)

    def update_terrain_levels(self, env_ids):
        """Update terrain curriculum levels based on robot progress."""
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        """
        Get current observations for DWAQ training initialization.
        
        Returns:
            obs: Actor observations [num_envs, num_obs]
            obs_hist: DWAQ observation history for VAE encoder [num_envs, num_obs * dwaq_obs_history_length]
            
        Note: This is called at training start to get initial observations.
        It updates buffers once to initialize the observation history.
        The runner calls: actions = alg.act(obs, critic_obs, prev_critic_obs, obs_hist)
        """
        actor_obs, critic_obs = self.compute_observations()
        obs_hist = self.dwaq_obs_history_buffer.buffer.reshape(self.num_envs, -1)
        
        self.extras["observations"] = {
            "critic": critic_obs,
            "obs_hist": obs_hist,
            "prev_critic_obs": self.prev_critic_obs,
        }
        return actor_obs, obs_hist
    
    def get_privileged_observations(self):
        """
        Get privileged observations for DWAQ critic and velocity loss.
        
        Returns:
            critic_obs: Privileged observations [num_envs, num_privileged_obs]
            prev_critic_obs: Previous critic observations for velocity estimation [num_envs, num_privileged_obs]
            
        Note: This method does NOT update buffers. It returns the last computed
        critic_obs (from the buffer) plus prev_critic_obs.
        prev_critic_obs contains velocity info from previous timestep,
        used to supervise the VAE encoder's velocity prediction.
        """
        # Get critic obs from buffer without updating (no append)
        critic_obs = self._get_current_critic_obs_with_height_scan()
        return critic_obs, self.prev_critic_obs
    
    # ==================== DWAQ Required Properties ====================
    
    @property
    def num_obs(self) -> int:
        """Number of actor observations (without height scan for blind walking)."""
        return self._num_obs
    
    @property
    def num_privileged_obs(self) -> int:
        """Number of privileged observations for critic (includes velocity, height scan)."""
        # Add height scan dimension if enabled
        height_scan_dim = 0
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan_dim = self.height_scanner.data.ray_hits_w.shape[1]
        return self._num_privileged_obs + height_scan_dim
    
    @property
    def num_obs_hist(self) -> int:
        """Number of observation history frames for DWAQ VAE encoder."""
        return self.dwaq_obs_history_length

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set random seed for reproducibility."""
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
