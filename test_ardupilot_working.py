#!/usr/bin/env python3
"""Test ArduPilot integration with comprehensive logging."""

import sys
import time
import torch
import traceback
from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Import after app launch
import carb
from seeker_swarm.tasks.direct.seeker_swarm.seeker_swarm_env import SeekerSwarmEnv
from seeker_swarm.tasks.direct.seeker_swarm.seeker_swarm_env_cfg import SeekerSwarmEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat

def print_drone_stats(env, step, elapsed_time):
    """Print comprehensive stats for all drones."""
    print(f"\n{'='*80}")
    print(f"Step {step:5d} | Time: {elapsed_time:.1f}s / 600s")
    print('='*80)

    for env_id in range(env.num_envs):
        # Position and velocity
        pos = env.scene["robot"].data.root_pos_w[env_id]
        vel = env.scene["robot"].data.root_lin_vel_w[env_id]

        # Attitude (quaternion -> Euler angles)
        quat = env.scene["robot"].data.root_quat_w[env_id].unsqueeze(0)  # Add batch dim
        roll, pitch, yaw = euler_xyz_from_quat(quat)
        roll_deg = torch.rad2deg(roll).item()
        pitch_deg = torch.rad2deg(pitch).item()
        yaw_deg = torch.rad2deg(yaw).item()

        # Thrust (if available from ArduPilot manager)
        if hasattr(env, 'ardupilot_manager') and env.ardupilot_manager is not None:
            # Get motor thrusts from ArduPilot (N, 4 tensor)
            motor_thrusts = env.ardupilot_manager.motor_thrusts
            if motor_thrusts is not None and len(motor_thrusts) > env_id:
                # Sum of all motor thrusts for this drone
                total_thrust = motor_thrusts[env_id].sum().item()
                # Get individual motor thrusts
                motor_0 = motor_thrusts[env_id][0].item()
                motor_1 = motor_thrusts[env_id][1].item()
                motor_2 = motor_thrusts[env_id][2].item()
                motor_3 = motor_thrusts[env_id][3].item()
            else:
                total_thrust = 0.0
                motor_0 = motor_1 = motor_2 = motor_3 = 0.0
        else:
            total_thrust = 0.0
            motor_0 = motor_1 = motor_2 = motor_3 = 0.0

        print(f"Drone {env_id}:")
        print(f"  Pos: [{pos[0]:+7.3f}, {pos[1]:+7.3f}, {pos[2]:+7.3f}] m")
        print(f"  Vel: [{vel[0]:+7.3f}, {vel[1]:+7.3f}, {vel[2]:+7.3f}] m/s")
        print(f"  Att: [R:{roll_deg:+7.2f}°, P:{pitch_deg:+7.2f}°, Y:{yaw_deg:+7.2f}°]")
        print(f"  Thrust: {total_thrust:8.4f} N (Motors: [{motor_0:.4f}, {motor_1:.4f}, {motor_2:.4f}, {motor_3:.4f}])")

def main():
    # Configure environment
    cfg = SeekerSwarmEnvCfg()
    cfg.scene.num_envs = 2
    cfg.use_ardupilot = True
    cfg.ardupilot_dir = "/home/sam/repos/falcon/submodules/ardupilot"
    cfg.ardupilot_autolaunch = True

    carb.log_info("\n" + "="*80)
    carb.log_info("ArduPilot Integration Test - 10 Minute Run")
    carb.log_info("="*80)
    carb.log_info(f"Drones: {cfg.scene.num_envs} | ArduPilot: {cfg.use_ardupilot}")
    carb.log_info(f"ArduPilot path: {cfg.ardupilot_dir}")
    carb.log_info(f"Duration: 10 minutes (600 seconds)")
    carb.log_info(f"Stats printed every ~2-3 seconds")
    carb.log_info("="*80 + "\n")

    # Create environment
    carb.log_info("[1/4] Creating environment...")
    env = SeekerSwarmEnv(cfg)
    carb.log_info("✓ Environment created\n")

    # Wait for ArduPilot SITL to initialize
    carb.log_info("[2/4] Waiting for ArduPilot SITL to initialize (10 seconds)...")
    for i in range(10, 0, -1):
        carb.log_info(f"  ...{i}s")
        time.sleep(1)
    carb.log_info("✓ ArduPilot ready\n")

    # Reset environment
    carb.log_info("[3/4] Resetting environment...")
    obs, _ = env.reset()
    carb.log_info(f"✓ Reset complete. Obs shape: {obs['policy'].shape}\n")

    # Run simulation for 10 minutes
    carb.log_info("[4/4] Running simulation for 10 minutes...")
    carb.log_info("      Press Ctrl+C to stop early\n")

    duration = 600.0  # 10 minutes in seconds
    print_interval = 2.0  # Print every 2 seconds

    start_time = time.time()
    last_print_time = start_time
    step = 0

    try:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Check if 10 minutes elapsed
            if elapsed_time >= duration:
                break

            # Zero actions (ArduPilot runs its own stabilization)
            actions = torch.zeros((cfg.scene.num_envs, env.action_space.shape[0]), device=env.device)

            # Step simulation
            obs, rewards, dones, truncated, info = env.step(actions)
            step += 1

            # Print stats periodically
            if current_time - last_print_time >= print_interval:
                print_drone_stats(env, step, elapsed_time)
                last_print_time = current_time

        # Final stats
        print_drone_stats(env, step, elapsed_time)

    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"\n\n⚠️  Test interrupted by user after {elapsed_time:.1f}s")
        print_drone_stats(env, step, elapsed_time)

    # Cleanup
    carb.log_info("\n[5/5] Closing environment...")
    env.close()
    carb.log_info("✓ Environment closed\n")

    # Summary
    elapsed_time = time.time() - start_time
    carb.log_info("="*80)
    carb.log_info("🎉 TEST COMPLETE!")
    carb.log_info("="*80)
    carb.log_info(f"Total steps: {step}")
    carb.log_info(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    carb.log_info(f"Average FPS: {step/elapsed_time:.1f}")
    carb.log_info("\nArduPilot SITL integration ran successfully!")
    carb.log_info("Check the logs above for drone positions, velocities, attitudes, and thrusts.")
    carb.log_info("="*80 + "\n")

if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user (Ctrl+C)")
        exit_code = 130
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}\n")
        traceback.print_exc()
        exit_code = 1
    finally:
        print("\nClosing simulation...")
        simulation_app.close()
        print("Done.\n")

    sys.exit(exit_code)
