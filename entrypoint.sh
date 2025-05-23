#!/bin/bash
set -e
source /opt/ros/${ROS_DISTRO}/setup.bash
cd /home/user/task22_ws

# Build the workspace if src directory exists
if [ -d "src" ]; then
    echo "Building task22_ws workspace..."
    colcon build
    echo "Build completed!"
    
    # Source the workspace setup if build was successful
    if [ -f "install/setup.bash" ]; then
        source install/setup.bash
    fi
else
    echo "No src directory found in task22_ws, skipping build"
fi

exec "$@"