#!/usr/bin/env python3
"""
MAVLink Command Broadcaster for ArduPilot Multi-Drone Simulation

This script sends the same MAVLink command to multiple ArduPilot SITL instances.
Use it while the Isaac Lab simulation is running to control all drones at once.

Usage:
    # Arm all drones
    python3 mavlink_broadcast.py --num_drones 3 --command arm

    # Disarm all drones
    python3 mavlink_broadcast.py --num_drones 3 --command disarm

    # Set mode to GUIDED
    python3 mavlink_broadcast.py --num_drones 3 --command mode --mode GUIDED

    # Set mode to LOITER
    python3 mavlink_broadcast.py --num_drones 3 --command mode --mode LOITER

    # Takeoff to 5 meters
    python3 mavlink_broadcast.py --num_drones 3 --command takeoff --altitude 5

    # Land
    python3 mavlink_broadcast.py --num_drones 3 --command land

    # Set throttle (RC override) - value 1000-2000, 1500=center
    python3 mavlink_broadcast.py --num_drones 3 --command throttle --value 1600

    # Clear RC overrides
    python3 mavlink_broadcast.py --num_drones 3 --command clear_rc

    # Send custom RC channels (Roll, Pitch, Throttle, Yaw) - values 1000-2000
    python3 mavlink_broadcast.py --num_drones 3 --command rc --channels 1500,1500,1600,1500

Port mapping (each drone uses a unique port):
    Drone 0: tcp:127.0.0.1:14550
    Drone 1: tcp:127.0.0.1:14560
    Drone 2: tcp:127.0.0.1:14570
    ...
    Drone N: tcp:127.0.0.1:(14550 + N*10)
"""

import argparse
import sys
import time
from typing import List

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not found. Install with: pip install pymavlink")
    sys.exit(1)


class DroneConnection:
    """Represents a connection to a single drone."""

    def __init__(self, drone_id: int, port: int, timeout: float = 3.0):
        self.drone_id = drone_id
        self.port = port
        self.connection = None
        self.timeout = timeout

    def connect(self) -> bool:
        """Connect to the drone via MAVLink."""
        try:
            connection_string = f'tcp:127.0.0.1:{self.port}'
            print(f"[Drone {self.drone_id}] Connecting to {connection_string}...", end=' ')

            self.connection = mavutil.mavlink_connection(
                connection_string,
                source_system=255,
                source_component=0
            )

            # Wait for heartbeat
            self.connection.wait_heartbeat(timeout=self.timeout)
            print(f"✓ Connected (sysid={self.connection.target_system})")
            return True

        except Exception as e:
            print(f"✗ Failed: {e}")
            return False

    def arm(self):
        """Arm the drone."""
        self.connection.arducopter_arm()

    def disarm(self):
        """Disarm the drone."""
        self.connection.arducopter_disarm()

    def set_mode(self, mode: str):
        """Set flight mode (e.g., GUIDED, LOITER, STABILIZE, ALT_HOLD)."""
        mode_id = self.connection.mode_mapping().get(mode.upper())
        if mode_id is None:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.connection.mode_mapping().keys())}")

        self.connection.mav.set_mode_send(
            self.connection.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

    def takeoff(self, altitude: float):
        """Takeoff to specified altitude (meters)."""
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # confirmation
            0, 0, 0, 0, 0, 0,  # params 1-6
            altitude  # param 7: altitude
        )

    def land(self):
        """Land the drone."""
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,  # confirmation
            0, 0, 0, 0, 0, 0, 0
        )

    def set_rc_channels(self, channels: List[int]):
        """Set RC channel overrides.

        Args:
            channels: List of 8 channel values (1000-2000). Use 0 to skip a channel.
                     Typically: [roll, pitch, throttle, yaw, mode, 0, 0, 0]
        """
        if len(channels) < 8:
            channels.extend([0] * (8 - len(channels)))

        self.connection.mav.rc_channels_override_send(
            self.connection.target_system,
            self.connection.target_component,
            *channels[:8]
        )

    def clear_rc_override(self):
        """Clear all RC overrides."""
        self.set_rc_channels([0] * 8)


def broadcast_command(num_drones: int, command: str, **kwargs):
    """Broadcast a command to all drones.

    Args:
        num_drones: Number of drones to control
        command: Command to send (arm, disarm, mode, takeoff, land, throttle, rc, clear_rc)
        **kwargs: Additional command-specific arguments
    """
    print(f"\n{'='*70}")
    print(f"MAVLink Command Broadcaster")
    print(f"{'='*70}")
    print(f"Command: {command}")
    print(f"Target drones: {num_drones}")
    print(f"{'='*70}\n")

    # Connect to all drones
    drones: List[DroneConnection] = []

    for i in range(num_drones):
        port = 5762 + i * 10
        drone = DroneConnection(i, port)

        if drone.connect():
            drones.append(drone)
        else:
            print(f"[WARNING] Skipping drone {i} (connection failed)")

    if not drones:
        print("\n❌ No drones connected. Is the simulation running?")
        return 1

    print(f"\n✓ Connected to {len(drones)}/{num_drones} drones\n")

    # Send command to all connected drones
    print(f"Sending command '{command}' to all drones...")

    try:
        for drone in drones:
            if command == "arm":
                drone.arm()
                print(f"[Drone {drone.drone_id}] Sent ARM command")

            elif command == "disarm":
                drone.disarm()
                print(f"[Drone {drone.drone_id}] Sent DISARM command")

            elif command == "mode":
                mode = kwargs.get('mode')
                if not mode:
                    raise ValueError("--mode argument required for 'mode' command")
                drone.set_mode(mode)
                print(f"[Drone {drone.drone_id}] Set mode to {mode}")

            elif command == "takeoff":
                altitude = kwargs.get('altitude', 5.0)
                drone.takeoff(altitude)
                print(f"[Drone {drone.drone_id}] Sent TAKEOFF to {altitude}m")

            elif command == "land":
                drone.land()
                print(f"[Drone {drone.drone_id}] Sent LAND command")

            elif command == "throttle":
                value = kwargs.get('value', 1500)
                if not (1000 <= value <= 2000):
                    raise ValueError("Throttle value must be between 1000-2000")
                # RC channel 3 is throttle
                drone.set_rc_channels([1500, 1500, value, 1500, 0, 0, 0, 0])
                print(f"[Drone {drone.drone_id}] Set throttle to {value}")

            elif command == "rc":
                channels_str = kwargs.get('channels')
                if not channels_str:
                    raise ValueError("--channels argument required for 'rc' command (e.g., 1500,1500,1600,1500)")
                channels = [int(x) for x in channels_str.split(',')]
                drone.set_rc_channels(channels)
                print(f"[Drone {drone.drone_id}] Set RC channels to {channels}")

            elif command == "clear_rc":
                drone.clear_rc_override()
                print(f"[Drone {drone.drone_id}] Cleared RC overrides")

            else:
                raise ValueError(f"Unknown command: {command}")

        print(f"\n✓ Command sent to {len(drones)} drones successfully!")

        # Wait briefly for commands to be received
        time.sleep(0.1)

    except Exception as e:
        print(f"\n❌ Error sending command: {e}")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Broadcast MAVLink commands to multiple ArduPilot drones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Arm all 3 drones
  %(prog)s --num_drones 3 --command arm

  # Set all drones to GUIDED mode and takeoff to 5m
  %(prog)s --num_drones 3 --command mode --mode GUIDED
  %(prog)s --num_drones 3 --command arm
  %(prog)s --num_drones 3 --command takeoff --altitude 5

  # Land all drones
  %(prog)s --num_drones 3 --command land

  # Send 60%% throttle to all drones
  %(prog)s --num_drones 3 --command throttle --value 1600

  # Clear RC overrides
  %(prog)s --num_drones 3 --command clear_rc

Available modes:
  STABILIZE, ACRO, ALT_HOLD, AUTO, GUIDED, LOITER, RTL, CIRCLE,
  LAND, DRIFT, SPORT, FLIP, AUTOTUNE, POSHOLD, BRAKE, THROW,
  AVOID_ADSB, GUIDED_NOGPS, SMART_RTL, FLOWHOLD, FOLLOW, ZIGZAG
        """
    )

    parser.add_argument(
        '--num_drones',
        type=int,
        required=True,
        help='Number of drones to control'
    )

    parser.add_argument(
        '--command',
        type=str,
        required=True,
        choices=['arm', 'disarm', 'mode', 'takeoff', 'land', 'throttle', 'rc', 'clear_rc'],
        help='Command to send to all drones'
    )

    parser.add_argument(
        '--mode',
        type=str,
        help='Flight mode (for "mode" command): GUIDED, LOITER, STABILIZE, etc.'
    )

    parser.add_argument(
        '--altitude',
        type=float,
        default=5.0,
        help='Takeoff altitude in meters (for "takeoff" command, default: 5.0)'
    )

    parser.add_argument(
        '--value',
        type=int,
        help='PWM value 1000-2000 (for "throttle" command, default: 1500)'
    )

    parser.add_argument(
        '--channels',
        type=str,
        help='Comma-separated RC channel values (for "rc" command, e.g., 1500,1500,1600,1500)'
    )

    args = parser.parse_args()

    # Run the broadcaster
    return broadcast_command(
        num_drones=args.num_drones,
        command=args.command,
        mode=args.mode,
        altitude=args.altitude,
        value=args.value,
        channels=args.channels
    )


if __name__ == '__main__':
    sys.exit(main())
