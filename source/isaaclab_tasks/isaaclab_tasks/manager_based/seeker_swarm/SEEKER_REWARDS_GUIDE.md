# Seeker Swarm Evaluation Rewards Guide

## Overview
This module contains reward/loss functions specifically designed for seeker swarm drones that need to reach a target as quickly as possible. The focus is on fast target acquisition, efficient trajectories, and minimizing final distance.

## Location
`isaaclab_tasks/manager_based/seeker_swarm/mdp/rewards.py`

## Seeker-Specific Reward Functions

### 1. `distance_to_target()`
**Purpose**: Raw distance metric for evaluation
- Returns Euclidean distance in meters
- Use for: Tracking performance, final distance metrics
- Lower is better
- **Type**: Metric (not typically used as reward directly)

### 2. `distance_reduction_rate()`
**Purpose**: Reward for actively approaching target
- Positive when moving toward target
- Negative when moving away
- Use for: Encouraging continuous approach
- **Weight suggestion**: 5.0 to 15.0

### 3. `velocity_toward_target_l2()`
**Purpose**: Maintain optimal velocity toward target
- Penalizes deviation from ideal approach velocity
- Parameter: `target_speed` (default 5.0 m/s)
- Use for: Smooth, controlled approach
- **Weight suggestion**: -1.0 to -5.0 (negative because it's a penalty)

### 4. `target_reached_bonus()`
**Purpose**: Sparse reward for success
- Binary: 1.0 when within threshold, 0.0 otherwise
- Parameter: `threshold` (default 1.0 m)
- Use for: Marking successful target acquisition
- **Weight suggestion**: 50.0 to 100.0 (large bonus)

### 5. `time_to_reach_penalty()`
**Purpose**: Create urgency for fast completion
- Constant penalty per timestep
- Parameter: `time_scale` (default 0.1)
- Use for: Encouraging speed over accuracy
- **Weight suggestion**: -0.1 to -1.0

### 6. `approach_angle_alignment()`
**Purpose**: Reward direct approach
- Cosine similarity between velocity and target direction
- Range: [-1, 1]
- Use for: Preventing inefficient curved paths
- **Weight suggestion**: 2.0 to 10.0

### 7. `distance_to_target_tanh()`
**Purpose**: Dense distance reward with smooth falloff
- Alternative to exponential kernel
- Parameter: `scale` (default 2.0)
- Use for: Long-range seeking with less sensitivity to exact distance
- **Weight suggestion**: 10.0 to 30.0

### 8. `speed_toward_target_bonus()`
**Purpose**: Reward high-speed pursuit
- Only counts velocity component toward target
- Parameter: `min_speed` threshold
- Use for: Aggressive seeking behavior
- **Weight suggestion**: 3.0 to 10.0

### 9. `distance_to_goal_exp()` (from original)
**Purpose**: Exponential distance reward
- Classic exponential kernel: exp(-d²/σ²)
- Parameter: `std` controls falloff
- Use for: Strong short-range attraction
- **Weight suggestion**: 15.0 to 50.0

## Original Stabilization Rewards (Still Useful)

### 10. `ang_vel_xyz_exp()`
- Penalize excessive rotation
- **Weight**: 5.0 to 15.0

### 11. `lin_vel_xyz_exp()`
- Penalize excessive overall speed (use carefully with seekers!)
- **Weight**: 1.0 to 5.0 (or 0.0 if you want max speed)

### 12. `yaw_aligned()`
- Keep drone level/aligned
- **Weight**: 1.0 to 5.0

## Recommended Reward Configurations

### Configuration 1: Fast Aggressive Seeker
**Goal**: Reach target as fast as possible, less concerned with smoothness

```python
@configclass
class AggressiveSeekerRewardsCfg:
    # Primary objective: get close fast
    distance_to_goal_exp = RewTerm(
        func=mdp.distance_to_goal_exp,
        weight=30.0,
        params={"std": 2.0}
    )

    # Bonus for speed
    speed_toward_target_bonus = RewTerm(
        func=mdp.speed_toward_target_bonus,
        weight=8.0,
        params={"min_speed": 2.0}
    )

    # Large success bonus
    target_reached_bonus = RewTerm(
        func=mdp.target_reached_bonus,
        weight=100.0,
        params={"threshold": 1.0}
    )

    # Time penalty for urgency
    time_to_reach_penalty = RewTerm(
        func=mdp.time_to_reach_penalty,
        weight=-0.5
    )

    # Direct approach
    approach_angle_alignment = RewTerm(
        func=mdp.approach_angle_alignment,
        weight=5.0
    )

    # Minimal stabilization
    ang_vel_xyz_exp = RewTerm(
        func=mdp.ang_vel_xyz_exp,
        weight=3.0,
        params={"std": 5.0}
    )
```

### Configuration 2: Smooth Efficient Seeker
**Goal**: Reach target quickly but with smooth trajectory

```python
@configclass
class SmoothSeekerRewardsCfg:
    # Distance reward
    distance_to_target_tanh = RewTerm(
        func=mdp.distance_to_target_tanh,
        weight=20.0,
        params={"scale": 3.0}
    )

    # Optimal velocity maintenance
    velocity_toward_target_l2 = RewTerm(
        func=mdp.velocity_toward_target_l2,
        weight=-2.0,
        params={"target_speed": 4.0}
    )

    # Success bonus
    target_reached_bonus = RewTerm(
        func=mdp.target_reached_bonus,
        weight=50.0,
        params={"threshold": 0.5}
    )

    # Distance reduction
    distance_reduction_rate = RewTerm(
        func=mdp.distance_reduction_rate,
        weight=10.0
    )

    # Alignment
    approach_angle_alignment = RewTerm(
        func=mdp.approach_angle_alignment,
        weight=8.0
    )

    # Stabilization
    ang_vel_xyz_exp = RewTerm(
        func=mdp.ang_vel_xyz_exp,
        weight=8.0,
        params={"std": 3.0}
    )

    # Action penalties
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05
    )
```

### Configuration 3: Evaluation/Competition Mode
**Goal**: Optimize for metrics - fastest time to threshold

```python
@configclass
class CompetitionSeekerRewardsCfg:
    # Sparse success reward
    target_reached_bonus = RewTerm(
        func=mdp.target_reached_bonus,
        weight=1000.0,  # Very large
        params={"threshold": 0.5}
    )

    # Heavy time penalty
    time_to_reach_penalty = RewTerm(
        func=mdp.time_to_reach_penalty,
        weight=-1.0
    )

    # Direct approach
    distance_to_goal_exp = RewTerm(
        func=mdp.distance_to_goal_exp,
        weight=25.0,
        params={"std": 1.0}
    )

    # Speed bonus
    speed_toward_target_bonus = RewTerm(
        func=mdp.speed_toward_target_bonus,
        weight=10.0,
        params={"min_speed": 3.0}
    )

    # Crash penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-100.0
    )
```

## Evaluation Metrics to Track

When running seeker swarm training/evaluation, track these metrics:

1. **Time to Reach** (primary metric)
   - Average timesteps until reaching threshold
   - Success rate within time limit

2. **Final Distance**
   - Distance to target at episode end
   - Percentage within success threshold

3. **Approach Efficiency**
   - Average velocity toward target
   - Path length vs. straight-line distance

4. **Speed Metrics**
   - Peak velocity toward target
   - Average velocity magnitude

5. **Stability Metrics**
   - Angular velocity magnitude
   - Number of oscillations

## Usage with Falcon Control

Since your seekers will run Falcon control code, these reward functions will evaluate how well the Falcon controller performs at the seeking task. The rewards will shape the high-level commands or parameters that Falcon receives, not directly control the thrust commands.

## Next Steps

1. **Create environment config** - Set up scene with target asset
2. **Configure reward weights** - Choose one of the configurations above
3. **Add termination conditions** - Define success (reached target) and failure (timeout, crash)
4. **Set up observations** - Include target position/direction in observation space
5. **Integrate with Falcon** - Connect Falcon controller as action space
