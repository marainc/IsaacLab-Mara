#!/bin/bash
# Kill all simulation-related processes

echo "=========================================="
echo "Hunting and Killing Simulation Processes"
echo "=========================================="
echo ""

# Function to kill processes matching a pattern
kill_pattern() {
    local pattern=$1
    local name=$2
    echo "Checking for $name processes..."

    pids=$(ps aux | grep -i "$pattern" | grep -v grep | awk '{print $2}')

    if [ -z "$pids" ]; then
        echo "  ✓ No $name processes found"
    else
        echo "  Found PIDs: $pids"
        for pid in $pids; do
            echo "    Killing PID $pid..."
            kill -9 $pid 2>/dev/null || true
        done
        echo "  ✓ Killed $name processes"
    fi
    echo ""
}

# Kill ArduPilot processes
kill_pattern "arducopter" "ArduCopter"
kill_pattern "sim_vehicle" "sim_vehicle.py"
kill_pattern "mavproxy" "MAVProxy"

# Kill Isaac Sim processes - MORE SPECIFIC PATTERNS
kill_pattern "isaac-sim" "Isaac Sim"
kill_pattern "omni.isaac" "Omniverse Isaac"
kill_pattern "omniverse.*isaac" "Omniverse (Isaac-related)"

# Kill Python processes running our test scripts
echo "Checking for test script processes..."
pids=$(ps aux | grep -E "test_ardupilot|isaaclab.sh" | grep -v grep | awk '{print $2}')
if [ -z "$pids" ]; then
    echo "  ✓ No test script processes found"
else
    echo "  Found PIDs: $pids"
    for pid in $pids; do
        echo "    Killing PID $pid..."
        kill -9 $pid 2>/dev/null || true
    done
    echo "  ✓ Killed test script processes"
fi
echo ""

# Give processes time to die
echo "Waiting 2 seconds for processes to terminate..."
sleep 2

# Verify cleanup
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

remaining=$(ps aux | grep -E "arducopter|sim_vehicle|mavproxy|isaac-sim|omni.isaac|test_ardupilot" | grep -v grep)
if [ -z "$remaining" ]; then
    echo "✓ All processes terminated successfully"
else
    echo "⚠ Some processes still running:"
    echo "$remaining"
fi

echo ""
echo "Checking ports..."
ports_in_use=$(lsof -i :14550 -i :14560 -i :9002 -i :9012 -i :5760 -i :5770 2>/dev/null)
if [ -z "$ports_in_use" ]; then
    echo "✓ All ports are free"
else
    echo "⚠ Some ports still in use:"
    echo "$ports_in_use"
fi

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
