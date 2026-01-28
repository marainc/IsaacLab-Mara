#!/bin/bash
# Convenience script to run ArduPilot integration test with proper environment setup

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
echo ""
echo "=========================================="
echo "IMPORTANT:"
echo "  1. A simulation window will open - DON'T CLOSE IT!"
echo "  2. Two GNOME Terminal windows will pop up (ArduPilot SITL) - DON'T CLOSE THEM!"
echo "  3. The test will wait 10 seconds for SITL to initialize"
echo "  4. Then run 200 simulation steps (~10 seconds)"
echo "  5. It will automatically close when done"
echo "=========================================="
echo ""
# echo "Press Enter to continue..."
# read

./isaaclab.sh -p test_ardupilot_working.py
