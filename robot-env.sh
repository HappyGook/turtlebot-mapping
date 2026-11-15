#!/usr/bin/env bash
# robot-env.sh - environment for the TurtleBot3 SBC (Raspberry Pi)
# Usage (must be sourced, not executed):   source ~/robot_env.sh
# echo "source ~/robot-env.sh" >> ~/.bashrc <- so that it is always loaded

# --- ROS 2 base installation ---
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "[robot_env] WARNING: /opt/ros/humble/setup.bash not found"
fi

# --- Optional: TurtleBot3 workspace overlay (if you built one) ---
if [ -f "$HOME/turtlebot3_ws/install/setup.bash" ]; then
    source "$HOME/turtlebot3_ws/install/setup.bash"
fi

# --- ROS / TurtleBot3 configuration (must match the remote PC) ---
export ROS_DOMAIN_ID=54
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-03
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# --- libcamera built from source (installed to /usr/local) ---
export LD_LIBRARY_PATH=/usr/local/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}

echo "[robot_env] ROS_DOMAIN_ID=$ROS_DOMAIN_ID MODEL=$TURTLEBOT3_MODEL LDS=$LDS_MODEL RMW=$RMW_IMPLEMENTATION"