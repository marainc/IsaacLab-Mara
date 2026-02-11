# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for seeker swarm environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def out_of_bounds_box(
    env: ManagerBasedRLEnv,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when drone leaves axis-aligned bounding box.

    Compares the asset root position (relative to env origins) against the
    specified min/max bounds in all three axes.

    Args:
        env: The manager-based RL environment instance.
        bounds_min: Minimum (x, y, z) bounds of the allowed volume.
        bounds_max: Maximum (x, y, z) bounds of the allowed volume.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").

    Returns:
        A boolean tensor of shape (num_envs,) that is True for envs where the
        drone has left the bounding box.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w - env.scene.env_origins

    bounds_min_t = torch.tensor(bounds_min, device=env.device, dtype=pos.dtype)
    bounds_max_t = torch.tensor(bounds_max, device=env.device, dtype=pos.dtype)

    below_min = torch.any(pos < bounds_min_t, dim=1)
    above_max = torch.any(pos > bounds_max_t, dim=1)

    return below_min | above_max
