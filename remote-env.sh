#!/usr/bin/env bash
# remote_env.sh - environment for the remote Ubuntu PC/VM
# Usage (must be sourced, not executed):
#   source ~/remote_env.sh <ROBOT_IP> [NETWORK_INTERFACE]
# Example:
#   source ~/remote-env.sh 192.168.1.42 enp0s1

ROBOT_IP="$1"
IFACE="${2:-enp0s1}"

if [ -z "$ROBOT_IP" ]; then
    echo "Usage: source remote_env.sh <ROBOT_IP> [NETWORK_INTERFACE]"
    return 1 2>/dev/null || exit 1
fi

if ! [[ "$ROBOT_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "[remote_env] ERROR: '$ROBOT_IP' is not a valid IPv4 address"
    return 1 2>/dev/null || exit 1
fi

if ! ip link show "$IFACE" >/dev/null 2>&1; then
    echo "[remote_env] ERROR: interface '$IFACE' not found. Available:"
    ip -o link show | awk -F': ' '{print "  " $2}'
    return 1 2>/dev/null || exit 1
fi

# --- ROS 2 base installation ---
source /opt/ros/humble/setup.bash
[ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"

# --- ROS / TurtleBot3 configuration (must match the robot) ---
export ROS_DOMAIN_ID=54
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-03
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# --- CycloneDDS: unicast discovery, multicast off ---
CYCLONE_FILE="$HOME/cyclonedds.xml"
cat > "$CYCLONE_FILE" <<EOF
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="${IFACE}"/>
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>50</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="${ROBOT_IP}"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF
export CYCLONEDDS_URI="file://${CYCLONE_FILE}"

# --- Restart the ROS 2 daemon so it picks up the new settings ---
ros2 daemon stop >/dev/null 2>&1

echo "[remote_env] robot=$ROBOT_IP iface=$IFACE domain=$ROS_DOMAIN_ID"
echo "[remote_env] config written to $CYCLONE_FILE"
echo "[remote_env] test with: ros2 topic list"