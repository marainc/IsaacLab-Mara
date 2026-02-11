# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeker evaluation environment with per-episode metric tracking and arming delay."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn


class SeekerEvalEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv subclass that tracks per-episode evaluation metrics.

    This environment extends the base RL env with:
    - Per-episode metrics (min distance, max approach velocity, time-to-reach, etc.)
    - Staggered arming delays (drones start unarmed and receive arm at different times)
    - Metric logging via the standard ``extras["log"]`` pipeline

    Tracked metrics (all per-env tensors):
        - ``min_distance_to_target``: Closest the drone got to target during episode
        - ``max_approach_velocity``: Peak velocity component toward target
        - ``time_to_reach``: Seconds from arm to first reaching threshold (NaN if never)
        - ``has_taken_off``: Whether drone exceeded takeoff height threshold
        - ``takeoff_time``: Seconds from arm to exceeding takeoff height
        - ``termination_reason``: 0=timeout, 1=reached, 2=crash, 3=OOB
        - ``final_distance``: Distance to target at episode end
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # Eval config params (set defaults before super().__init__ reads cfg)
        self._arm_delay_range: tuple[float, float] = getattr(cfg, "arm_delay_range", (0.0, 2.0))
        self._reach_threshold: float = getattr(cfg, "reach_threshold", 1.0)
        self._takeoff_height: float = getattr(cfg, "takeoff_height", 0.5)

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Arming state
        self.arm_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.is_armed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Metric buffers
        self.min_distance_to_target = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.max_approach_velocity = torch.zeros(self.num_envs, device=self.device)
        self.time_to_reach = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.has_taken_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.takeoff_time = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.final_distance = torch.zeros(self.num_envs, device=self.device)

        # Initialize arm delays for first episode
        self._sample_arm_delays(torch.arange(self.num_envs, device=self.device))

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one step and update evaluation metrics."""
        result = super().step(action)

        # Update arming state: flip is_armed when episode_length_buf >= arm_step
        newly_armed = (~self.is_armed) & (self.episode_length_buf >= self.arm_step)
        self.is_armed = self.is_armed | newly_armed

        # Compute current state for metrics (only for armed envs)
        self._update_metrics()

        return result

    def _update_metrics(self):
        """Update running evaluation metrics from current drone state."""
        asset: RigidObject = self.scene["robot"]
        pos = asset.data.root_pos_w - self.scene.env_origins
        command = self.command_manager.get_command("target_pose")
        target_pos = command[:, :3]

        # Current distance to target
        distance_vec = target_pos - pos
        distance = torch.norm(distance_vec, dim=1)

        # Velocity toward target
        velocity = asset.data.root_lin_vel_w
        direction = distance_vec / (distance.unsqueeze(-1) + 1e-8)
        approach_vel = torch.sum(velocity * direction, dim=1)

        # Only update metrics for armed envs
        armed = self.is_armed

        # Min distance
        self.min_distance_to_target = torch.where(
            armed & (distance < self.min_distance_to_target),
            distance,
            self.min_distance_to_target,
        )

        # Max approach velocity
        self.max_approach_velocity = torch.where(
            armed & (approach_vel > self.max_approach_velocity),
            approach_vel,
            self.max_approach_velocity,
        )

        # Time to reach (first time within threshold)
        reached_now = armed & (distance < self._reach_threshold) & torch.isnan(self.time_to_reach)
        steps_since_arm = (self.episode_length_buf - self.arm_step).float()
        time_since_arm = steps_since_arm * self.step_dt
        self.time_to_reach = torch.where(reached_now, time_since_arm, self.time_to_reach)

        # Takeoff detection
        height = pos[:, 2]
        took_off_now = armed & (~self.has_taken_off) & (height > self._takeoff_height)
        self.has_taken_off = self.has_taken_off | took_off_now
        self.takeoff_time = torch.where(took_off_now, time_since_arm, self.takeoff_time)

        # Always update final distance (will be snapshotted at reset)
        self.final_distance = distance

    def _reset_idx(self, env_ids: Sequence[int]):
        """Log metrics for finished episodes, then reset."""
        env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Determine termination reasons before super() resets everything
        # 0=timeout, 1=reached, 2=crash/OOB (any non-timeout termination)
        termination_reason = torch.zeros(len(env_ids), device=self.device, dtype=torch.long)

        # Check if timed out
        is_timeout = self.reset_time_outs[env_ids_t]
        # Check if reached target
        is_reached = self.min_distance_to_target[env_ids_t] < self._reach_threshold
        # Check if terminated (crash or OOB) - not timeout
        is_terminated = self.reset_terminated[env_ids_t]

        termination_reason = torch.where(is_timeout, torch.zeros_like(termination_reason), termination_reason)
        termination_reason = torch.where(is_reached & ~is_terminated, torch.ones_like(termination_reason), termination_reason)
        termination_reason = torch.where(is_terminated & ~is_timeout, torch.full_like(termination_reason, 2), termination_reason)

        # Snapshot metrics for logging (mean across resetting envs)
        metrics = {
            "Eval/min_distance_to_target": self.min_distance_to_target[env_ids_t].mean().item(),
            "Eval/max_approach_velocity": self.max_approach_velocity[env_ids_t].mean().item(),
            "Eval/final_distance": self.final_distance[env_ids_t].mean().item(),
            "Eval/has_taken_off_rate": self.has_taken_off[env_ids_t].float().mean().item(),
            "Eval/termination_reason_mean": termination_reason.float().mean().item(),
        }

        # Only log time_to_reach for envs that actually reached
        ttr = self.time_to_reach[env_ids_t]
        valid_ttr = ttr[~torch.isnan(ttr)]
        if len(valid_ttr) > 0:
            metrics["Eval/time_to_reach"] = valid_ttr.mean().item()
            metrics["Eval/reach_rate"] = (len(valid_ttr) / len(env_ids))

        # Only log takeoff_time for envs that took off
        tt = self.takeoff_time[env_ids_t]
        valid_tt = tt[~torch.isnan(tt)]
        if len(valid_tt) > 0:
            metrics["Eval/takeoff_time"] = valid_tt.mean().item()

        # Call super to do the actual reset (this creates self.extras["log"])
        super()._reset_idx(env_ids)

        # Inject our metrics into the log dict
        self.extras["log"].update(metrics)

        # Reset metric buffers for these envs
        self.min_distance_to_target[env_ids_t] = float("inf")
        self.max_approach_velocity[env_ids_t] = 0.0
        self.time_to_reach[env_ids_t] = float("nan")
        self.has_taken_off[env_ids_t] = False
        self.takeoff_time[env_ids_t] = float("nan")
        self.final_distance[env_ids_t] = 0.0

        # Reset arming state and sample new delays
        self.is_armed[env_ids_t] = False
        self._sample_arm_delays(env_ids_t)

    def _sample_arm_delays(self, env_ids: torch.Tensor):
        """Sample new arming delays for the given environments.

        Args:
            env_ids: Tensor of environment indices to sample delays for.
        """
        n = len(env_ids)
        delay_min, delay_max = self._arm_delay_range

        if delay_max <= delay_min:
            delays_s = torch.full((n,), delay_min, device=self.device)
        else:
            delays_s = torch.rand(n, device=self.device) * (delay_max - delay_min) + delay_min

        # Convert seconds to env steps
        delay_steps = (delays_s / self.step_dt).long()
        self.arm_step[env_ids] = delay_steps
