# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeker swarm environments."""

import gymnasium as gym

from .config import agents

##
# Register Gym environments.
##

gym.register(
    id="SeekerSwarm-Eval-v0",
    entry_point="isaaclab_tasks.manager_based.seeker_swarm.seeker_eval_env:SeekerEvalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.manager_based.seeker_swarm.config.seeker_eval_env_cfg:SeekerEvalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.eval_ppo_cfg:SeekerEvalPPORunnerCfg"
        ),
    },
)

gym.register(
    id="SeekerSwarm-Eval-Play-v0",
    entry_point="isaaclab_tasks.manager_based.seeker_swarm.seeker_eval_env:SeekerEvalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.manager_based.seeker_swarm.config.seeker_eval_env_cfg:SeekerEvalEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.eval_ppo_cfg:SeekerEvalPPORunnerCfg"
        ),
    },
)
