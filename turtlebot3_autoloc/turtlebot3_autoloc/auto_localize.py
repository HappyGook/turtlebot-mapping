"""Autonomous global localization for TurtleBot3 with Nav2 AMCL (ROS 2 Humble).

Flow:
  1. Wait until AMCL is active (its global-localization service is available).
  2. Send a dummy /initialpose so AMCL starts publishing map->odom and /amcl_pose
     (this also lets the rest of Nav2 finish starting up).
  3. Call /reinitialize_global_localization: particles are spread over the whole map.
  4. Rotate in place via /cmd_vel until the /amcl_pose covariance is small
     (at least one full turn, up to a timeout).
  5. Stop the robot and exit. Nav2 stays running for goal navigation.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Empty


class AutoLocalize(Node):

    def __init__(self):
        super().__init__('auto_localize')

        self.declare_parameter('angular_speed', 0.4)      # rad/s while turning
        self.declare_parameter('step_angle', 0.785)       # rad per step (45 deg)
        self.declare_parameter('pause', 1.0)              # s standing still per step
        self.declare_parameter('updates_per_stop', 3)     # forced AMCL updates per stop
        self.declare_parameter('laser_max_range', 1.5)    # m, <= 0 keeps AMCL's value
        self.declare_parameter('timeout', 90.0)           # s
        self.declare_parameter('max_xy_std', 0.15)        # m
        self.declare_parameter('max_yaw_std', 0.15)       # rad
        self.declare_parameter('global_loc_service', '/reinitialize_global_localization')

        p = lambda name: self.get_parameter(name).value
        self.speed = p('angular_speed')
        self.step_angle = p('step_angle')
        self.pause = p('pause')
        self.updates_per_stop = p('updates_per_stop')
        self.laser_max_range = p('laser_max_range')
        self.timeout = p('timeout')
        self.max_xy_std = p('max_xy_std')
        self.max_yaw_std = p('max_yaw_std')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_cb, 10)
        self.global_loc = self.create_client(Empty, p('global_loc_service'))
        self.nomotion = self.create_client(Empty, '/request_nomotion_update')
        self.amcl_params = self.create_client(SetParameters, '/amcl/set_parameters')

        self.pose = None

    # ---------- helpers ----------

    def pose_cb(self, msg):
        self.pose = msg

    def converged(self):
        if self.pose is None:
            return False
        cov = self.pose.pose.covariance  # row-major 6x6: x, y, z, roll, pitch, yaw
        xy_std = math.sqrt(max(cov[0], cov[7]))
        yaw_std = math.sqrt(cov[35])
        return xy_std < self.max_xy_std and yaw_std < self.max_yaw_std

    def wait(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result()

    def drive(self, angular):
        twist = Twist()
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def stop(self):
        for _ in range(5):   # make sure the stop command arrives
            self.drive(0.0)
            self.wait(0.05)

    # ---------- steps ----------

    def limit_laser_range(self):
        if self.laser_max_range <= 0:
            return
        req = SetParameters.Request()
        req.parameters = [
            Parameter('laser_max_range', Parameter.Type.DOUBLE,
                      float(self.laser_max_range)).to_parameter_msg()]
        if not self.amcl_params.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Cannot reach /amcl/set_parameters, keeping laser range.')
            return
        result = self.call(self.amcl_params, req)
        if result and result.results[0].successful:
            self.get_logger().info(f'AMCL laser_max_range set to {self.laser_max_range} m')
        else:
            self.get_logger().warn('AMCL rejected laser_max_range, set it in the Nav2 params file.')

    def send_dummy_initial_pose(self):
        while self.init_pub.get_subscription_count() == 0:
            self.wait(0.2)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'          # stamp 0 = "use latest transform"
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.07
        self.init_pub.publish(msg)
        self.wait(1.0)

    def turn_step(self):
        """Rotate by step_angle (time-based), then stand still."""
        end = time.time() + self.step_angle / self.speed
        while time.time() < end:
            self.drive(self.speed)
            self.wait(0.05)
        self.stop()

    def update_while_still(self):
        """Let AMCL integrate several scans taken while the robot is not moving."""
        self.wait(0.3)   # let the robot settle, get a fresh scan
        per_update = max(self.pause / self.updates_per_stop, 0.25)
        for _ in range(self.updates_per_stop):
            self.call(self.nomotion, Empty.Request())
            self.wait(per_update)   # next scan triggers the forced update

    def run(self):
        self.get_logger().info('Waiting for AMCL ...')
        self.global_loc.wait_for_service()
        self.nomotion.wait_for_service()

        self.limit_laser_range()
        self.send_dummy_initial_pose()

        self.get_logger().info('Spreading particles over the whole map ...')
        self.call(self.global_loc, Empty.Request())
        self.wait(0.5)
        self.pose = None

        steps_per_turn = math.ceil(2 * math.pi / self.step_angle)
        start = time.time()
        step = 0
        self.get_logger().info('Localizing: turn, stop, update ...')
        while rclpy.ok() and time.time() - start < self.timeout:
            self.update_while_still()
            if step >= steps_per_turn and self.converged():
                break
            self.turn_step()
            step += 1
        self.stop()

        if self.converged():
            pose = self.pose.pose.pose
            yaw = 2 * math.atan2(pose.orientation.z, pose.orientation.w)
            self.get_logger().info(
                f'Localized after {step} steps at x={pose.position.x:.2f}, '
                f'y={pose.position.y:.2f}, yaw={math.degrees(yaw):.0f} deg')
        else:
            self.get_logger().warn(
                'Not converged. Move the robot a bit and run this node again, '
                'or set the pose manually with "2D Pose Estimate" in RViz.')


def main():
    rclpy.init()
    node = AutoLocalize()
    try:
        node.run()
    except KeyboardInterrupt:
        node.drive(0.0)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()