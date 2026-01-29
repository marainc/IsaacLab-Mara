#!/bin/bash
# Convenience script to run ArduPilot integration test with proper environment setup
#
# Usage:
#   ./run_ardupilot_test.sh [OPTIONS]
#
# Options:
#   -n, --num_drones N    Number of drones to simulate (default: 12)
#   -d, --duration N      Test duration in seconds (default: 600)
#   -h, --headless        Run in headless mode (no GUI)
#   --help                Show this help message
#
# Examples:
#   ./run_ardupilot_test.sh                              # 12 drones, 10 minutes
#   ./run_ardupilot_test.sh -n 3                         # 3 drones, 10 minutes
#   ./run_ardupilot_test.sh -n 5 -d 300                  # 5 drones, 5 minutes
#   ./run_ardupilot_test.sh -n 10 -d 120 --headless      # 10 drones, 2 minutes, no GUI

# Parse command line arguments
NUM_DRONES=""
DURATION=""
HEADLESS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num_drones)
            NUM_DRONES="--num_drones $2"
            shift 2
            ;;
        -d|--duration)
            DURATION="--duration $2"
            shift 2
            ;;
        -h|--headless)
            HEADLESS="--headless"
            shift
            ;;
        --help)
            echo "ArduPilot SITL Integration Test"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --num_drones N    Number of drones to simulate (default: 12)"
            echo "  -d, --duration N      Test duration in seconds (default: 600)"
            echo "  -h, --headless        Run in headless mode (no GUI)"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # 12 drones, 10 minutes"
            echo "  $0 -n 3                         # 3 drones, 10 minutes"
            echo "  $0 -n 5 -d 300                  # 5 drones, 5 minutes"
            echo "  $0 -n 10 -d 120 --headless      # 10 drones, 2 minutes, no GUI"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ArduPilot SITL Integration Test"
echo "=========================================="
echo ""

# Set Pegasus path
export PYTHONPATH="/home/sam/repos/PegasusSimulator/extensions/pegasus.simulator:$PYTHONPATH"

# Check ArduPilot
ARDUPILOT_DIR="/home/sam/repos/falcon/submodules/ardupilot"
if [ ! -f "$ARDUPILOT_DIR/build/sitl/bin/arducopter" ]; then
    echo "⚠️  ArduPilot SITL not found!"
    echo ""
    echo "Building ArduPilot SITL..."
    cd "$ARDUPILOT_DIR"
    ./waf configure --board sitl
    ./waf copter
    echo ""
fi

# Run test
cd /home/sam/repos/IsaacLab
echo "Starting test..."
if [ -n "$NUM_DRONES" ] || [ -n "$DURATION" ] || [ -n "$HEADLESS" ]; then
    echo "Parameters: $NUM_DRONES $DURATION $HEADLESS"
fi
echo ""
echo "=========================================="
echo "NOTES:"
echo "  - ArduPilot processes run silently in background (no pop-ups)"
echo "  - Startup is parallelized (~5s for all drones)"
echo "  - Press Ctrl+C to stop early"
echo "  - Logs: /tmp/ardupilot_logs/vehicle_N/"
echo "=========================================="
echo ""

./isaaclab.sh -p test_ardupilot_working.py $NUM_DRONES $DURATION $HEADLESS
