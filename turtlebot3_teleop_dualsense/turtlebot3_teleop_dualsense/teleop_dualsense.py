#!/usr/bin/env python
#
# DualSense teleoperation for TurtleBot3 - receiver side
# (runs where ROS 2 runs).
#
#   R2 (axis 5)        throttle, forward
#   L2 (axis 4)        brake / reverse  (forward speed = R2 - L2)
#   left stick X (0)   steering
#
# Same conventions as turtlebot3_teleop/teleop_keyboard (ROBOTIS, Apache-2.0):
# 'cmd_vel' topic, velocity limits per TURTLEBOT3_MODEL, Twist on Humble and
# TwistStamped on newer distros (decided via ROS_DISTRO).
#
# tools/dualsense_sender.py runs on the machine the DualSense is attached to
# and streams its state as JSON over UDP.
# This node only needs the Python standard library.
#
# Safety:
#   - If no packet arrives for LINK_TIMEOUT seconds, the robot is stopped and
#     nothing is published until the link is back.
#   - Triggers are ignored until each has been seen at rest (-1.0) once. Some
#     controllers report 0.0 for an untouched trigger, which would read as
#     half throttle.

import json
import os
import socket
import time

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

# Optional environment variables
UDP_PORT = int(os.environ.get('DUALSENSE_PORT', '9999'))
ALLOWED_IP = os.environ.get('DUALSENSE_ALLOWED_IP', '')
LINK_TIMEOUT = 0.5

# Axis numbers as reported by pygame on the sender machine
# (check with: python3 dualsense_sender.py --probe).
AXIS_STEER = 0       # left stick X: left = -1.0, right = +1.0
AXIS_BRAKE = 4       # L2: released = -1.0, fully pressed = +1.0
AXIS_THROTTLE = 5    # R2: released = -1.0, fully pressed = +1.0

STEER_DEADZONE = 0.08       # stick units
TRIGGER_DEADZONE = 0.05     # after normalising a trigger to 0..1
TRIGGER_REST = -0.9         # a trigger at or below this counts as "released"

# --- feel / tuning ---
STEER_SCALE = 0.6    # fraction of the max angular velocity at full steering lock
LIN_SLOP = 0.02      # max change of linear velocity per tick (20 Hz) = 0.4 m/s^2
ANG_SLOP = 0.4       # max change of angular velocity per tick (20 Hz) = 8 rad/s^2

DEBUG_PERIOD = 0.2   # seconds; at most one debug line per period (ROS parameter 'debug')

MSG = """
Control Your TurtleBot3 with the DualSense
-----------------------------------------------------------
  R2            : throttle (forward)
  L2            : brake / reverse
  left stick X  : steer
  release both triggers -> robot rolls to a stop

Waiting for sender on UDP port {port} ...   CTRL-C to quit
"""


def make_simple_profile(output_vel, input_vel, slop):
    if input_vel > output_vel:
        output_vel = min(input_vel, output_vel + slop)
    elif input_vel < output_vel:
        output_vel = max(input_vel, output_vel - slop)
    else:
        output_vel = input_vel
    return output_vel


def constrain(input_vel, low_bound, high_bound):
    return max(low_bound, min(high_bound, input_vel))


def deadzone(value, threshold):
    return 0.0 if abs(value) < threshold else value


class TeleopDualSense(Node):

    def __init__(self):
        super().__init__('teleop_dualsense')

        self.model = os.environ.get('TURTLEBOT3_MODEL', 'burger')
        if self.model == 'burger':
            self.max_lin, self.max_ang = BURGER_MAX_LIN_VEL, BURGER_MAX_ANG_VEL
        else:
            self.max_lin, self.max_ang = WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_ANG_VEL

        self.use_twist = os.environ.get('ROS_DISTRO') == 'humble'
        msg_type = Twist if self.use_twist else TwistStamped
        self.pub = self.create_publisher(msg_type, 'cmd_vel', 10)

        self.target_lin = 0.0
        self.target_ang = 0.0
        self.control_lin = 0.0
        self.control_ang = 0.0

        # debug output: ros2 run ... teleop_dualsense --ros-args -p debug:=false to silence
        self.declare_parameter('debug', True)
        self.debug = self.get_parameter('debug').value
        self.pedals = (0.0, 0.0, 0.0)   # throttle, brake, steer after deadzones
        self.debug_last = 0.0
        self.debug_prev = None

        # latest controller state received over UDP
        self.axes = []
        self.rested = {AXIS_THROTTLE: False, AXIS_BRAKE: False}
        self.armed_announced = False
        self.linked = False
        self.last_rx = 0.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', UDP_PORT))
        self.sock.setblocking(False)

        print(MSG.format(port=UDP_PORT))
        self.create_timer(0.05, self._tick)  # 20 Hz

    # -- input ---------------------------------------------------------------
    def _poll_network(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                return
            if ALLOWED_IP and addr[0] != ALLOWED_IP:
                continue
            try:
                state = json.loads(data)
            except ValueError:
                continue
            if not isinstance(state, dict) or not isinstance(state.get('axes'), list):
                continue
            if not self.linked:
                self.get_logger().info(
                    f'Controller link up ({addr[0]}) - release both triggers to arm')
                self.linked = True
                self.rested = {AXIS_THROTTLE: False, AXIS_BRAKE: False}
                self.armed_announced = False
            self.last_rx = time.monotonic()
            self.axes = state['axes']

    def _axis(self, index):
        try:
            return float(self.axes[index])
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _trigger(self, index):
        """Trigger as 0.0 (released); 1.0 (fully pressed); 0.0 until seen at rest."""
        value = self._axis(index)
        if value <= TRIGGER_REST:
            self.rested[index] = True
        if not self.rested[index]:
            return 0.0
        return deadzone((value + 1.0) / 2.0, TRIGGER_DEADZONE)

    def _update_targets(self):
        throttle = self._trigger(AXIS_THROTTLE)
        brake = self._trigger(AXIS_BRAKE)
        steer = -deadzone(self._axis(AXIS_STEER), STEER_DEADZONE)  # stick left = turn left

        if not self.armed_announced and all(self.rested.values()):
            self.armed_announced = True
            print('Triggers spotted')

        self.target_lin = constrain((throttle - brake) * self.max_lin,
                                    -self.max_lin, self.max_lin)
        self.target_ang = constrain(steer * self.max_ang * STEER_SCALE,
                                    -self.max_ang, self.max_ang)
        self.pedals = (throttle, brake, steer)

    def _debug_print(self):
        now = time.monotonic()
        if now - self.debug_last < DEBUG_PERIOD:
            return
        self.debug_last = now
        throttle, brake, steer = self.pedals
        line = (f'thr {throttle:4.2f}  brk {brake:4.2f}  steer {steer:+5.2f} | '
                f'target {self.target_lin:+5.2f} m/s {self.target_ang:+5.2f} rad/s | '
                f'cmd {self.control_lin:+5.2f} m/s {self.control_ang:+5.2f} rad/s')
        if not all(self.rested.values()):
            line += (f' | NOT ARMED (raw L2 {self._axis(AXIS_BRAKE):+.2f}, '
                     f'R2 {self._axis(AXIS_THROTTLE):+.2f})')
        if line != self.debug_prev:   # only print when something changed
            print(line)
            self.debug_prev = line

    def _stop(self):
        self.target_lin = self.target_ang = 0.0
        self.control_lin = self.control_ang = 0.0

    # -- output --------------------------------------------------------------
    def _publish(self, lin, ang):
        if self.use_twist:
            twist = Twist()
            twist.linear.x = lin
            twist.angular.z = ang
            self.pub.publish(twist)
        else:
            ts = TwistStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id = ''
            ts.twist.linear.x = lin
            ts.twist.angular.z = ang
            self.pub.publish(ts)

    def _tick(self):
        self._poll_network()

        if self.linked and time.monotonic() - self.last_rx > LINK_TIMEOUT:
            self.get_logger().warn('Controller link lost')
            self.linked = False
            self._stop()
            self._publish(0.0, 0.0)   # one explicit stop, then stay silent
            return
        if not self.linked:
            return   # publish nothing, so other cmd_vel sources (e.g. Nav2) are not disturbed

        self._update_targets()
        self.control_lin = make_simple_profile(
            self.control_lin, self.target_lin, LIN_SLOP)
        self.control_ang = make_simple_profile(
            self.control_ang, self.target_ang, ANG_SLOP)
        self._publish(self.control_lin, self.control_ang)
        if self.debug:
            self._debug_print()

    def shutdown(self):
        self._publish(0.0, 0.0)
        self.sock.close()


def main():
    rclpy.init()
    node = TeleopDualSense()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()