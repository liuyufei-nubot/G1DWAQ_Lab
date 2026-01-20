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

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.g1.g1_env import G1Env
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def joint_deviation_l1_always(
    env: BaseEnv | TienKungEnv | G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint deviation from default pose at all times (not just when standing).

    Use this for limbs (e.g., arms) that should maintain a default pose even during locomotion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: TienKungEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: TienKungEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def gait_feet_spd_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


# ======================== DWAQ Rewards ========================
# These rewards are adapted from the DreamWaQ project for blind walking.


def alive(env: BaseEnv) -> torch.Tensor:
    """Reward for staying alive.
    
    A simple constant reward that encourages the robot to not terminate early.
    Reference: DreamWaQ (HumanoidDreamWaq/legged_gym/envs/g1/g1_env.py)
    """
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float)


def gait_phase_contact(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, stance_threshold: float = 0.55
) -> torch.Tensor:
    """Reward for foot contact matching the expected gait phase.
    
    Rewards the robot when foot contact status matches the expected stance/swing phase.
    During stance phase (phase < stance_threshold), foot should be in contact.
    During swing phase (phase >= stance_threshold), foot should be in the air.
    
    Args:
        env: Environment with gait phase information.
        sensor_cfg: Contact sensor configuration for feet.
        stance_threshold: Phase threshold below which the foot should be in stance.
        
    Reference: DreamWaQ _reward_contact()
    
    Note: This function uses env.leg_phase which should be [num_envs, num_feet] tensor
    where leg_phase[:, 0] = phase_left and leg_phase[:, 1] = phase_right.
    The sensor_cfg.body_ids should match the same ordering (left foot first, right foot second).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # Check contact for each foot (use z-component like original DreamWaQ)
    # Original: contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
    contact = net_contact_forces[:, :, 2] > 1.0  # (num_envs, num_feet), z-direction force
    
    # Use leg_phase directly from environment
    # leg_phase shape: (num_envs, 2) where [:, 0] = left, [:, 1] = right
    leg_phase = env.leg_phase
    
    # Expected stance: phase < stance_threshold
    is_stance = leg_phase < stance_threshold
    
    # Reward: 1 if contact matches expected phase, 0 otherwise
    # XOR gives True when they don't match, so we negate it
    phase_match = ~(contact ^ is_stance)  # (num_envs, num_feet)
    
    return torch.sum(phase_match.float(), dim=-1)  # Sum over feet



def feet_swing_height(
    env: BaseEnv, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.08
) -> torch.Tensor:
    """Simple version: Penalize swing foot height deviation from fixed target.
    
    This is the original simple implementation that uses absolute z-coordinate.
    Use feet_swing_height() for terrain-aware version.
    
    Args:
        env: Environment.
        sensor_cfg: Contact sensor configuration for feet.
        asset_cfg: Robot configuration with body_ids for feet.
        target_height: Target height for swing foot (default 0.08m).
        
    Reference: DreamWaQ _reward_feet_swing_height()
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact status
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(net_contact_forces, dim=-1) > 1.0  # (num_envs, num_feet)
    
    # Get feet positions (z-coordinate)
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)
    
    # Penalize height error only during swing phase (not in contact)
    pos_error = torch.square(feet_pos_z - target_height) * (~contact).float()
    
    return torch.sum(pos_error, dim=-1)


def base_height(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.78
) -> torch.Tensor:
    """Penalize base height deviation from target.
    
    Encourages the robot to maintain a specific base height during locomotion.
    
    Args:
        env: Environment.
        asset_cfg: Robot configuration.
        target_height: Target base height (default 0.78m for G1).
        
    Reference: DreamWaQ _reward_base_height()
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    return torch.square(current_height - target_height)


def contact_no_vel(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize foot velocity when in contact with ground.
    
    During stance phase (when foot is in contact), the foot should have
    zero velocity to prevent slipping.
    
    Args:
        env: Environment.
        sensor_cfg: Contact sensor configuration for feet.
        asset_cfg: Robot configuration with body_ids for feet.
        
    Reference: DreamWaQ _reward_contact_no_vel()
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact status
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(net_contact_forces, dim=-1) > 1.0  # (num_envs, num_feet)
    
    # Get feet velocities
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    
    # Penalize velocity only when in contact
    contact_feet_vel = feet_vel * contact.unsqueeze(-1)
    penalize = torch.sum(torch.square(contact_feet_vel), dim=-1)  # Sum over xyz
    
    return torch.sum(penalize, dim=-1)  # Sum over feet


def joint_pos_limits(
    env: BaseEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9
) -> torch.Tensor:
    """Penalize joint positions near or exceeding limits.
    
    Args:
        env: Environment.
        asset_cfg: Robot configuration.
        soft_ratio: Ratio of joint limits to start penalizing (default 0.9).
        
    Reference: DreamWaQ _reward_dof_pos_limits()
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get joint limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]  # (num_envs, num_joints, 2)
    lower_limits = joint_pos_limits[:, :, 0] * soft_ratio
    upper_limits = joint_pos_limits[:, :, 1] * soft_ratio
    
    # Current joint positions
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # Penalize exceeding soft limits
    out_of_limits = torch.zeros_like(joint_pos)
    out_of_limits += (lower_limits - joint_pos).clamp(min=0.0)
    out_of_limits += (joint_pos - upper_limits).clamp(min=0.0)
    
    return torch.sum(out_of_limits, dim=-1)


def idle_when_commanded(
    env: BaseEnv | TienKungEnv | G1Env,
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize being idle when a velocity command is given.
    
    This reward function detects "lazy standing" behavior where the robot receives
    a movement command but remains stationary. It returns 1.0 when the robot should
    be moving but is not, enabling a negative weight penalty.
    
    Args:
        env: Environment instance.
        cmd_threshold: Minimum command magnitude to be considered "commanded to move".
            Commands below this threshold are ignored (robot is allowed to stand).
        vel_threshold: Maximum velocity magnitude to be considered "idle/stationary".
            If actual velocity is below this, the robot is considered not moving.
        asset_cfg: Robot configuration.
    
    Returns:
        Tensor of shape (num_envs,) with values:
        - 1.0 if commanded to move but idle (should be penalized)
        - 0.0 otherwise (no penalty)
    
    Example:
        idle_penalty = RewTerm(
            func=mdp.idle_when_commanded,
            weight=-2.0,
            params={"cmd_threshold": 0.2, "vel_threshold": 0.1}
        )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get velocity command (xy components)
    cmd_xy = env.command_generator.command[:, :2]
    cmd_magnitude = torch.linalg.norm(cmd_xy, dim=-1)
    
    # Get actual root velocity in yaw frame (same as track_lin_vel_xy uses)
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    vel_magnitude = torch.linalg.norm(vel_yaw[:, :2], dim=-1)
    
    # Detect "commanded but idle" condition
    is_commanded = cmd_magnitude > cmd_threshold  # Should be moving
    is_idle = vel_magnitude < vel_threshold       # But not moving
    
    return (is_commanded & is_idle).float()

