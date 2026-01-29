# Seeker Swarm Task Module

## Overview
This module contains evaluation and reward functions for seeker swarm drones that need to reach a target as quickly as possible. Designed for use with Falcon control code in Isaac Lab.

## Directory Structure
```
seeker_swarm/
├── README.md                      # This file
├── SEEKER_REWARDS_GUIDE.md       # Detailed reward function documentation
├── __init__.py                    # Module initialization
├── config/
│   └── seeker_swarm_env_cfg.py   # Environment configuration (to be customized)
└── mdp/
    ├── __init__.py
    ├── rewards.py                 # Seeker-specific reward/loss functions
    ├── observations.py            # Observation functions
    └── commands/
        ├── __init__.py
        ├── commands_cfg.py
        └── drone_pose_command.py
```

## Key Features

### Reward Functions (12 total)
The `mdp/rewards.py` file contains:

**Original Stabilization Rewards (4)**
1. `distance_to_goal_exp()` - Exponential distance reward
2. `ang_vel_xyz_exp()` - Angular velocity penalty
3. `lin_vel_xyz_exp()` - Linear velocity penalty
4. `yaw_aligned()` - Yaw alignment reward

**New Seeker-Specific Rewards (8)**
5. `distance_to_target()` - Raw distance metric
6. `distance_reduction_rate()` - Reward for approaching
7. `velocity_toward_target_l2()` - Optimal velocity maintenance
8. `target_reached_bonus()` - Success bonus
9. `time_to_reach_penalty()` - Time urgency penalty
10. `approach_angle_alignment()` - Direct approach reward
11. `distance_to_target_tanh()` - Smooth distance reward
12. `speed_toward_target_bonus()` - High-speed pursuit bonus

See `SEEKER_REWARDS_GUIDE.md` for detailed documentation and usage examples.

## Design Philosophy

### Evaluation Focus
These functions are designed as **loss/evaluation functions** for training seekers to:
1. **Minimize time to target** - Time penalties create urgency
2. **Minimize final distance** - Distance rewards encourage close approach
3. **Maximize approach efficiency** - Alignment rewards prevent wasteful trajectories
4. **Reward successful acquisition** - Sparse bonuses for reaching threshold

### Multi-Seeker Support
- Each seeker is evaluated independently
- No inter-seeker collision or interaction rewards (as specified)
- All functions work with vectorized environments (num_envs parallel simulations)
- Each environment can have multiple seekers with separate reward terms

## Integration with Falcon Control

These reward functions evaluate the **performance** of the Falcon control system:
- Falcon handles low-level thrust/attitude control
- Rewards shape high-level behavior (goal-seeking)
- Loss functions provide training signal for any learnable components

## Recommended Reward Configurations

### Fast Aggressive Seeker
Best for: Pure speed, minimal smoothness constraints
```python
distance_to_goal_exp: weight=30.0
speed_toward_target_bonus: weight=8.0
target_reached_bonus: weight=100.0
time_to_reach_penalty: weight=-0.5
```

### Smooth Efficient Seeker
Best for: Balanced speed and trajectory quality
```python
distance_to_target_tanh: weight=20.0
velocity_toward_target_l2: weight=-2.0
distance_reduction_rate: weight=10.0
approach_angle_alignment: weight=8.0
```

### Competition/Evaluation Mode
Best for: Optimizing specific metrics
```python
target_reached_bonus: weight=1000.0
time_to_reach_penalty: weight=-1.0
speed_toward_target_bonus: weight=10.0
```

## Next Steps

1. **Customize environment config** (`config/seeker_swarm_env_cfg.py`)
   - Define scene with target asset
   - Set number of seekers per environment
   - Configure observation space
   - Choose reward configuration

2. **Define termination conditions**
   - Success: reached within threshold
   - Failure: timeout or crash
   - Episode length

3. **Set up observations**
   - Target position/direction
   - Current velocity
   - Distance to target

4. **Integrate with Falcon**
   - Connect Falcon as action handler
   - Configure action space
   - Set up command interface

5. **Training/Evaluation**
   - Run parallel environments
   - Track metrics (time to reach, success rate, etc.)
   - Tune reward weights

## Key Metrics to Track

During training/evaluation, monitor:
- **Time to Reach** - Primary performance metric
- **Success Rate** - Percentage reaching threshold
- **Final Distance** - Distance at episode end
- **Average Velocity Toward Target** - Approach efficiency
- **Episode Reward** - Overall performance indicator

## File Modification Status

✅ **Created (copies from drone_arl)**:
- `seeker_swarm/mdp/rewards.py` - Modified with 8 new functions
- `seeker_swarm/mdp/observations.py` - Ready for customization
- `seeker_swarm/mdp/commands/` - Command generation
- `seeker_swarm/config/seeker_swarm_env_cfg.py` - Ready for customization

📝 **Original files remain untouched**:
- `drone_arl/` directory completely unchanged
- All modifications are in the new `seeker_swarm/` module

## Contact/Notes

These evaluation functions were designed for seeker swarm training in Isaac Lab.
The focus is on fast target acquisition with Falcon control integration.
