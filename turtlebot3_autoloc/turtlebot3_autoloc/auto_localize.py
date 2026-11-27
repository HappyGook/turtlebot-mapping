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
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_srvs.srv import Empty


class AutoLocalize(Node):

    def __init__(self):
        super().__init__('auto_localize')

        self.declare_parameter('angular_speed', 0.5)      # rad/s
        self.declare_parameter('timeout', 60.0)           # s
        self.declare_parameter('max_xy_std', 0.15)        # m
        self.declare_parameter('max_yaw_std', 0.15)       # rad
        self.declare_parameter('global_loc_service', '/reinitialize_global_localization')

        self.speed = self.get_parameter('angular_speed').value
        self.timeout = self.get_parameter('timeout').value
        self.max_xy_std = self.get_parameter('max_xy_std').value
        self.max_yaw_std = self.get_parameter('max_yaw_std').value
        service_name = self.get_parameter('global_loc_service').value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_cb, 10)
        self.global_loc = self.create_client(Empty, service_name)

        self.pose = None

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

    def rotate(self, angular):
        twist = Twist()
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def run(self):
        self.get_logger().info('Waiting for AMCL ...')
        self.global_loc.wait_for_service()

        self.send_dummy_initial_pose()

        self.get_logger().info('Spreading particles over the whole map ...')
        rclpy.spin_until_future_complete(self, self.global_loc.call_async(Empty.Request()))
        self.wait(0.5)
        self.pose = None

        min_spin_time = 2 * math.pi / self.speed   # at least one full turn
        start = time.time()
        self.get_logger().info('Rotating to localize ...')
        while rclpy.ok():
            elapsed = time.time() - start
            if elapsed > min_spin_time and self.converged():
                break
            if elapsed > self.timeout:
                break
            self.rotate(self.speed)
            self.wait(0.1)

        for _ in range(5):   # make sure the stop command arrives
            self.rotate(0.0)
            self.wait(0.05)

        if self.converged():
            p = self.pose.pose.pose
            yaw = 2 * math.atan2(p.orientation.z, p.orientation.w)
            self.get_logger().info(
                f'Localized at x={p.position.x:.2f}, y={p.position.y:.2f}, '
                f'yaw={math.degrees(yaw):.0f} deg')
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
        node.rotate(0.0)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
