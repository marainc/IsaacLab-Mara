#!/usr/bin/env python3
"""Test ArduPilot integration with comprehensive logging."""

import argparse
import sys
import time
import torch
import traceback
from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Test ArduPilot integration with Isaac Lab")
parser.add_argument("--num_drones", type=int, default=12, help="Number of drones to simulate (default: 3)")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
parser.add_argument("--duration", type=int, default=600, help="Test duration in seconds (default: 600)")
parser.add_argument("--spinning_rotors", action="store_true", help="Enable visual rotor spinning (kinematic, no physics feedback)")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# Import after app launch
import carb
from seeker_swarm.tasks.direct.seeker_swarm.seeker_swarm_env import SeekerSwarmEnv
from seeker_swarm.tasks.direct.seeker_swarm.seeker_swarm_env_cfg import SeekerSwarmEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat

def print_drone_stats(env, step, elapsed_time):
    """Print comprehensive stats for all drones."""
    # Calculate physics rate metrics
    if elapsed_time > 0 and step > 0:
        actual_physics_rate = step / elapsed_time
    else:
        actual_physics_rate = 0.0

    physics_dt = env.physics_dt if hasattr(env, 'physics_dt') else env.step_dt
    target_physics_rate = 1.0 / physics_dt if physics_dt > 0 else 0.0

    print(f"\n{'='*80}")
    print(f"Step {step:5d} | Time: {elapsed_time:.1f}s / 600s")
    print(f"Physics: dt={physics_dt*1000:.2f}ms | Target={target_physics_rate:.1f}Hz | Actual={actual_physics_rate:.1f}Hz")
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
    cfg.scene.num_envs = args_cli.num_drones
    cfg.use_ardupilot = True
    cfg.ardupilot_dir = "/home/sam/repos/falcon/submodules/ardupilot"
    cfg.ardupilot_autolaunch = True
    cfg.enable_rotor_spinning = args_cli.spinning_rotors  # Enable visual rotor spinning if flag passed

    carb.log_info("\n" + "="*80)
    carb.log_info(f"ArduPilot Integration Test - {args_cli.duration}s Run")
    carb.log_info("="*80)
    carb.log_info(f"Drones: {cfg.scene.num_envs} | ArduPilot: {cfg.use_ardupilot}")
    carb.log_info(f"ArduPilot path: {cfg.ardupilot_dir}")
    carb.log_info(f"Duration: {args_cli.duration} seconds ({args_cli.duration/60:.1f} minutes)")
    carb.log_info(f"Stats printed every ~2-3 seconds")
    carb.log_info("="*80 + "\n")

    # Create environment
    carb.log_info("[1/4] Creating environment...")
    env = SeekerSwarmEnv(cfg)
    carb.log_info("✓ Environment created\n")

    # Wait for ArduPilot SITL to initialize
    carb.log_info("[2/4] Waiting for ArduPilot SITL to initialize (5 seconds)...")
    for i in range(5, 0, -1):
        carb.log_info(f"  ...{i}s")
        time.sleep(1)
    carb.log_info("✓ ArduPilot ready\n")

    # Reset environment
    carb.log_info("[3/4] Resetting environment...")
    obs, _ = env.reset()
    carb.log_info(f"✓ Reset complete. Obs shape: {obs['policy'].shape}\n")

    # Run simulation
    carb.log_info(f"[4/4] Running simulation for {args_cli.duration}s ({args_cli.duration/60:.1f} minutes)...")
    carb.log_info("      Press Ctrl+C to stop early\n")

    duration = float(args_cli.duration)
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
