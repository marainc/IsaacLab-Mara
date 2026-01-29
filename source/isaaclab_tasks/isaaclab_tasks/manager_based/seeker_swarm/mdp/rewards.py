# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

"""
Seeker swarm evaluation rewards.

These reward functions are designed for seeker drones that need to reach
a target as quickly as possible. The focus is on:
1. Fast approach to target
2. Minimizing final distance
3. Efficient trajectory
4. Speed-based bonuses
"""


def distance_to_goal_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward the distance to a goal position using an exponential kernel.

    This reward computes an exponential falloff of the squared Euclidean distance
    between the commanded target position and the asset (robot) root position.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; larger values
            produce a gentler falloff.
        command_name: Name of the command to read the target pose from the
            environment's command manager. The function expects the command
            tensor to contain positions in its first three columns.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment reward
        values in [0, 1], with 1.0 when the position error is zero.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute the error
    position_error_square = torch.sum(torch.square(target_position_w - current_position), dim=1)
    return torch.exp(-position_error_square / std**2)


def ang_vel_xyz_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0
) -> torch.Tensor:
    """Penalize angular velocity magnitude with an exponential kernel.

    This reward computes exp(-||omega||^2 / std^2) where omega is the body-frame
    angular velocity of the asset. It is useful for encouraging low rotational
    rates.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; controls
            sensitivity to angular velocity magnitude.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        zero angular velocity.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute squared magnitude of angular velocity (all axes)
    ang_vel_squared = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)

    return torch.exp(-ang_vel_squared / std**2)


def lin_vel_xyz_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0
) -> torch.Tensor:
    """Penalize linear velocity magnitude with an exponential kernel.

    Computes exp(-||v||^2 / std^2) where v is the asset's linear velocity in
    world frame. Useful for encouraging the agent to reduce translational speed.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        zero linear velocity.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute squared magnitude of linear velocity (all axes)
    lin_vel_squared = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)

    return torch.exp(-lin_vel_squared / std**2)


def yaw_aligned(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.5
) -> torch.Tensor:
    """Reward alignment of the vehicle's yaw to zero using an exponential kernel.

    The function extracts the yaw (rotation about Z) from the world-frame root
    quaternion and computes exp(-yaw^2 / std^2). This encourages heading
    alignment to a zero-yaw reference.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; smaller values
            make the reward more sensitive to yaw deviations.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        perfect yaw alignment (yaw == 0).
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # extract yaw from current orientation
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    # normalize yaw to [-pi, pi] (target is 0)
    yaw = math_utils.wrap_to_pi(yaw)

    # return exponential reward (1 when yaw=0, approaching 0 when rotated)
    return torch.exp(-(yaw**2) / std**2)


##
# Seeker-specific rewards for fast target acquisition
##


def distance_to_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Returns the raw Euclidean distance to the target position.

    This is useful for tracking and evaluation metrics. Lower is better.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the seeker asset.
        command_name: Name of the command containing target position.

    Returns:
        A 1-D tensor of shape (num_envs,) containing distances in meters.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute Euclidean distance
    distance = torch.norm(target_position_w - current_position, dim=1)
    return distance


def distance_reduction_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward the rate of distance reduction to target.

    This encourages the seeker to actively close the distance to the target.
    Positive reward when getting closer, negative when moving away.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the seeker asset.
        command_name: Name of the command containing target position.

    Returns:
        A 1-D tensor of shape (num_envs,) with rate of distance change.
        Positive values mean approaching the target.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute current distance
    distance_vec = target_position_w - current_position
    distance = torch.norm(distance_vec, dim=1, keepdim=True)

    # get velocity toward target (negative of distance reduction rate)
    # velocity dot normalized direction to target
    velocity = asset.data.root_lin_vel_w
    direction_to_target = distance_vec / (distance + 1e-6)  # avoid division by zero
    velocity_toward_target = torch.sum(velocity * direction_to_target, dim=1)

    # return positive reward for approaching (distance reducing)
    return velocity_toward_target


def velocity_toward_target_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
    target_speed: float = 5.0,
) -> torch.Tensor:
    """Reward for maintaining optimal velocity toward the target.

    Encourages the seeker to move at a target speed directly toward the goal.
    Uses L2 penalty for deviation from optimal velocity.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the seeker asset.
        command_name: Name of the command containing target position.
        target_speed: Desired speed in m/s toward target.

    Returns:
        A 1-D tensor of shape (num_envs,) with negative L2 penalty.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute direction to target
    distance_vec = target_position_w - current_position
    distance = torch.norm(distance_vec, dim=1, keepdim=True)
    direction_to_target = distance_vec / (distance + 1e-6)

    # optimal velocity is target_speed in direction of target
    optimal_velocity = direction_to_target * target_speed

    # compute L2 error
    velocity = asset.data.root_lin_vel_w
    velocity_error = torch.norm(velocity - optimal_velocity, dim=1)

    # return negative L2 error (penalize deviation)
    return -velocity_error


def target_reached_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Large bonus reward when seeker reaches within threshold of target.

    This is a sparse reward that fires when the seeker successfully reaches
    the target zone. Useful for marking success in evaluation.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the seeker asset.
        command_name: Name of the command containing target position.
        threshold: Distance threshold in meters to consider target reached.

    Returns:
        A 1-D tensor of shape (num_envs,) with 1.0 when within threshold, 0.0 otherwise.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute distance
    distance = torch.norm(target_position_w - current_position, dim=1)

    # binary reward: 1.0 if within threshold, 0.0 otherwise
    return (distance < threshold).float()


def time_to_reach_penalty(
    env: ManagerBasedRLEnv,
    time_scale: float = 0.1,
) -> torch.Tensor:
    """Penalty that increases with time to encourage fast target acquisition.

    This creates urgency for the seeker to reach the target quickly.
    The penalty grows linearly with time.

    Args:
        env: The manager-based RL environment instance.
        time_scale: Scaling factor for time penalty (higher = more urgent).

    Returns:
        A 1-D tensor of shape (num_envs,) with negative time penalty.
    """
    # penalize each timestep that passes
    # encourages fast completion
    return -time_scale * torch.ones(env.num_envs, device=env.device)


def approach_angle_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward for aligning velocity direction with direction to target.

    Encourages the seeker to point and move directly toward the target
    rather than taking inefficient curved paths.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the seeker asset.
        command_name: Name of the command containing target position.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in [-1, 1].
        1.0 when moving directly toward target, -1.0 when moving directly away.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute direction to target
    distance_vec = target_position_w - current_position
    distance = torch.norm(distance_vec, dim=1, keepdim=True)
    direction_to_target = distance_vec / (distance + 1e-6)

    # compute velocity direction
    velocity = asset.data.root_lin_vel_w
    velocity_magnitude = torch.norm(velocity, dim=1, keepdim=True)
    velocity_direction = velocity / (velocity_magnitude + 1e-6)

    # cosine similarity between velocity and target direction
    alignment = torch.sum(velocity_direction * direction_to_target, dim=1)

    return alignment


def distance_to_target_tanh(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
    scale: float = 2.0,
) -> torch.Tensor:
    """Smooth reward for distance to target using tanh kernel.

    Provides dense reward signal that is less sensitive to scale than
    exponential kernel. Good for long-range seeking tasks.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the seeker asset.
        command_name: Name of the command containing target position.
        scale: Scaling factor for distance (larger = more gradual falloff).

    Returns:
        A 1-D tensor of shape (num_envs,) with values approaching 1 as distance approaches 0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute distance
    distance = torch.norm(target_position_w - current_position, dim=1)

    # tanh-based reward: approaches 1 as distance -> 0
    return 1.0 - torch.tanh(distance / scale)


def speed_toward_target_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
    min_speed: float = 1.0,
) -> torch.Tensor:
    """Bonus reward for maintaining high speed toward target.

    Encourages aggressive pursuit of the target. Only rewards when
    moving toward target (not lateral or backward movement).

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the seeker asset.
        command_name: Name of the command containing target position.
        min_speed: Minimum speed threshold to start giving bonus.

    Returns:
        A 1-D tensor of shape (num_envs,) with speed bonus (0 or positive).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute direction to target
    distance_vec = target_position_w - current_position
    distance = torch.norm(distance_vec, dim=1, keepdim=True)
    direction_to_target = distance_vec / (distance + 1e-6)

    # compute velocity component toward target
    velocity = asset.data.root_lin_vel_w
    velocity_toward_target = torch.sum(velocity * direction_to_target, dim=1)

    # only reward if moving toward target faster than min_speed
    bonus = torch.clamp(velocity_toward_target - min_speed, min=0.0)

    return bonus
