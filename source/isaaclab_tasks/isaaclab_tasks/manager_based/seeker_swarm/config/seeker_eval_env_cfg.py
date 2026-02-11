# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation-focused environment configuration for seeker swarm.

Extends the base TrackPositionNoObstaclesEnvCfg with:
- Longer episodes for thorough evaluation
- Boundary terminations (OOB box)
- Evaluation-focused reward terms
- Arming delay system
- armed_drone_commands and arm_countdown observations
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.seeker_swarm.mdp as mdp
from isaaclab_tasks.manager_based.seeker_swarm.config.seeker_swarm_env_cfg import TrackPositionNoObstaclesEnvCfg


##
# Eval-specific MDP settings
##


@configclass
class EvalObservationsCfg:
    """Observation specifications for the eval environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group — includes arming-aware target commands."""

        base_link_position = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_orientation = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        last_action = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.0, n_max=0.0))
        target_commands = ObsTerm(
            func=mdp.armed_drone_commands,
            params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        arm_state = ObsTerm(func=mdp.arm_countdown)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EvalRewardsCfg:
    """Reward terms focused on evaluation metrics while remaining training-compatible."""

    # Core pursuit rewards
    distance_to_goal_exp = RewTerm(
        func=mdp.distance_to_goal_exp,
        weight=25.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.5, "command_name": "target_pose"},
    )
    target_reached_bonus = RewTerm(
        func=mdp.target_reached_bonus,
        weight=100.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 1.0, "command_name": "target_pose"},
    )
    time_to_reach_penalty = RewTerm(
        func=mdp.time_to_reach_penalty,
        weight=-0.5,
        params={"time_scale": 0.1},
    )

    # Stabilization (reduced weights — no orientation penalty for agile drones)
    ang_vel_xyz_exp = RewTerm(
        func=mdp.ang_vel_xyz_exp,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 10.0},
    )

    # Evaluation penalties
    no_takeoff_penalty = RewTerm(
        func=mdp.no_takeoff_penalty,
        weight=-2.0,
        params={"height_threshold": 0.5, "grace_steps": 50, "asset_cfg": SceneEntityCfg("robot")},
    )
    boundary_proximity_penalty = RewTerm(
        func=mdp.boundary_proximity_penalty,
        weight=-1.0,
        params={
            "bounds_min": (-10.0, -10.0, 0.0),
            "bounds_max": (10.0, 10.0, 10.0),
            "margin": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    crash_or_oob_penalty = RewTerm(
        func=mdp.crash_or_oob_penalty,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Action smoothness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude_l2 = RewTerm(func=mdp.action_l2, weight=-0.05)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class EvalTerminationsCfg:
    """Termination terms with boundary enforcement."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -3.0})
    out_of_bounds = DoneTerm(
        func=mdp.out_of_bounds_box,
        params={
            "bounds_min": (-10.0, -10.0, 0.0),
            "bounds_max": (10.0, 10.0, 10.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


##
# Environment configuration
##


@configclass
class SeekerEvalEnvCfg(TrackPositionNoObstaclesEnvCfg):
    """Evaluation-focused seeker environment configuration.

    Extends the base config with longer episodes, boundary terminations,
    evaluation rewards, and arming delay support.
    """

    observations: EvalObservationsCfg = EvalObservationsCfg()
    rewards: EvalRewardsCfg = EvalRewardsCfg()
    terminations: EvalTerminationsCfg = EvalTerminationsCfg()

    # Arming delay config (read by SeekerEvalEnv)
    arm_delay_range: tuple[float, float] = (0.0, 2.0)
    reach_threshold: float = 1.0
    takeoff_height: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        # Longer episodes for evaluation
        self.episode_length_s = 15.0


@configclass
class SeekerEvalEnvCfg_PLAY(SeekerEvalEnvCfg):
    """Play variant for running trained policies and collecting eval metrics."""

    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play/visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise
        self.observations.policy.enable_corruption = False
