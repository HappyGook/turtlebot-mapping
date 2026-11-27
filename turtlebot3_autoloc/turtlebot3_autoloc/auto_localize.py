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
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.msg import ParticleCloud
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Empty


class AutoLocalize(Node):

    def __init__(self):
        super().__init__('auto_localize')

        self.declare_parameter('angular_speed', 0.4)      # rad/s while turning
        self.declare_parameter('step_angle', 0.785)       # rad per step (45 deg)
        self.declare_parameter('pause', 1.0)              # s standing still per step
        self.declare_parameter('updates_per_stop', 2)     # forced AMCL updates per stop
        # convergence
        self.declare_parameter('max_xy_std', 0.15)        # m
        self.declare_parameter('max_yaw_std', 0.15)       # rad
        self.declare_parameter('max_drift', 0.15)         # m, allowed jump while confirming
        self.declare_parameter('turns_before_rescatter', 3)
        self.declare_parameter('timeout', 180.0)          # s
        # AMCL tuning (<= 0 keeps AMCL's own value)
        self.declare_parameter('laser_max_range', 3.5)    # m
        self.declare_parameter('laser_min_range', -1.0)   # m, halls are 30-60 cm: keep close hits
        # Likelihood-field sensor model. AMCL's defaults (sigma_hit 0.2, z_hit/z_rand 0.5)
        # are blurrier than the 30-60 cm halls: with them almost every pose in the box
        # maze "fits" and the heading never resolves (last run: uniform headings).
        self.declare_parameter('sigma_hit', 0.06)         # m, ~1 map cell + lidar noise
        self.declare_parameter('z_hit', 0.85)
        self.declare_parameter('z_rand', 0.15)
        self.declare_parameter('laser_likelihood_max_dist', 0.5)  # m
        self.declare_parameter('max_particles', 10000)
        self.declare_parameter('min_particles', 2000)
        self.declare_parameter('recovery_alpha_slow', 0.001)
        self.declare_parameter('recovery_alpha_fast', 0.1)
        # debugging (see loc_debug.py)
        self.declare_parameter('debug', True)
        self.declare_parameter('debug_dir', '~/.ros/autoloc_debug')
        self.declare_parameter('debug_png', True)            # needs matplotlib
        self.declare_parameter('debug_global_search', True)  # brute-force scan match
        self.declare_parameter('scan_topic', '/scan')
        # optional ground truth in the map frame (nan = unknown), e.g. read off RViz
        self.declare_parameter('true_x', float('nan'))
        self.declare_parameter('true_y', float('nan'))
        self.declare_parameter('true_yaw', float('nan'))   # rad

        self.p = lambda name: self.get_parameter(name).value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_cb, 10)
        self.create_subscription(ParticleCloud, '/particle_cloud', self.cloud_cb,
                                 qos_profile_sensor_data)
        self.global_loc = self.create_client(Empty, '/reinitialize_global_localization')
        self.nomotion = self.create_client(Empty, '/request_nomotion_update')
        self.amcl_params = self.create_client(SetParameters, '/amcl/set_parameters')

        self.pose = None
        self.cloud = None

        self.dbg = None
        if self.p('debug'):
            from turtlebot3_autoloc.loc_debug import LocalizationDebugger
            self.dbg = LocalizationDebugger(self)

    # ---------- helpers ----------

    def pose_cb(self, msg):
        self.pose = msg

    def cloud_cb(self, msg):
        self.cloud = msg

    def converged(self):
        if self.pose is None:
            return False
        cov = self.pose.pose.covariance  # row-major 6x6: x, y, z, roll, pitch, yaw
        xy_std = math.sqrt(max(cov[0], cov[7]))
        yaw_std = math.sqrt(cov[35])
        return xy_std < self.p('max_xy_std') and yaw_std < self.p('max_yaw_std')

    def xy(self):
        pos = self.pose.pose.pose.position
        return pos.x, pos.y

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

    def log_cloud(self, label):
        if self.cloud is None or not self.cloud.particles:
            self.get_logger().info(f'{label}: no particle cloud received yet')
            return
        xs = [pt.pose.position.x for pt in self.cloud.particles]
        ys = [pt.pose.position.y for pt in self.cloud.particles]
        self.get_logger().info(
            f'{label}: {len(xs)} particles, x {min(xs):.1f}..{max(xs):.1f} m, '
            f'y {min(ys):.1f}..{max(ys):.1f} m')

    # ---------- steps ----------

    def tune_amcl(self):
        wanted = [
            ('laser_max_range', Parameter.Type.DOUBLE, float(self.p('laser_max_range'))),
            ('laser_min_range', Parameter.Type.DOUBLE, float(self.p('laser_min_range'))),
            ('sigma_hit', Parameter.Type.DOUBLE, float(self.p('sigma_hit'))),
            ('z_hit', Parameter.Type.DOUBLE, float(self.p('z_hit'))),
            ('z_rand', Parameter.Type.DOUBLE, float(self.p('z_rand'))),
            ('laser_likelihood_max_dist', Parameter.Type.DOUBLE,
             float(self.p('laser_likelihood_max_dist'))),
            ('max_particles', Parameter.Type.INTEGER, int(self.p('max_particles'))),
            ('min_particles', Parameter.Type.INTEGER, int(self.p('min_particles'))),
            ('recovery_alpha_slow', Parameter.Type.DOUBLE, float(self.p('recovery_alpha_slow'))),
            ('recovery_alpha_fast', Parameter.Type.DOUBLE, float(self.p('recovery_alpha_fast'))),
        ]
        wanted = [w for w in wanted if w[2] > 0]
        if not wanted:
            return
        if not self.amcl_params.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Cannot reach /amcl/set_parameters, AMCL left untuned.')
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(n, t, v).to_parameter_msg() for n, t, v in wanted]
        result = self.call(self.amcl_params, req)
        if result is None:
            self.get_logger().warn('No answer from /amcl/set_parameters.')
            return
        for (name, _, value), res in zip(wanted, result.results):
            if res.successful:
                self.get_logger().info(f'AMCL {name} = {value}')
            else:
                self.get_logger().warn(f'AMCL rejected {name}: {res.reason} '
                                       '-> set it in the Nav2 params file instead')

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

    def scatter(self):
        self.get_logger().info('Spreading particles over the whole map ...')
        self.call(self.global_loc, Empty.Request())
        self.pose = None
        self.cloud = None
        self.wait(0.5)

    def turn_step(self):
        """Rotate by step_angle (time-based), then stand still."""
        end = time.time() + self.p('step_angle') / self.p('angular_speed')
        while time.time() < end:
            self.drive(self.p('angular_speed'))
            self.wait(0.05)
        self.stop()

    def update_while_still(self):
        """Let AMCL integrate a few scans taken while the robot is not moving."""
        self.wait(0.3)   # let the robot settle, get a fresh scan
        n = max(int(self.p('updates_per_stop')), 1)
        per_update = max(self.p('pause') / n, 0.25)
        for _ in range(n):
            self.call(self.nomotion, Empty.Request())
            self.wait(per_update)   # next scan triggers the forced update

    def run(self):
        self.get_logger().info('Waiting for AMCL ...')
        self.global_loc.wait_for_service()
        self.nomotion.wait_for_service()

        self.tune_amcl()
        if self.dbg:
            self.dbg.setup()
        self.send_dummy_initial_pose()
        self.scatter()

        steps_per_turn = math.ceil(2 * math.pi / self.p('step_angle'))
        rescatter_after = steps_per_turn * int(self.p('turns_before_rescatter'))
        start = time.time()
        steps_since_scatter = 0
        confirmed = 0          # consecutive converged + stable stops
        anchor = None          # pose where convergence started
        step = 0

        self.get_logger().info('Localizing: turn, stop, update ...')
        while rclpy.ok() and time.time() - start < self.p('timeout'):
            self.update_while_still()
            if steps_since_scatter == 0:
                self.log_cloud('After first update')
            if self.dbg:
                self.dbg.on_stop(step, 'after scatter' if steps_since_scatter == 0 else '')
            step += 1

            if self.converged():
                if anchor is None:
                    anchor = self.xy()
                    self.get_logger().info(
                        f'Candidate at x={anchor[0]:.2f}, y={anchor[1]:.2f}, confirming ...')
                if math.dist(anchor, self.xy()) < self.p('max_drift'):
                    confirmed += 1
                else:   # jumped somewhere else -> start confirming again
                    anchor, confirmed = self.xy(), 1
                if confirmed > steps_per_turn:   # stable for a full extra turn
                    break
            else:
                anchor, confirmed = None, 0
                if steps_since_scatter >= rescatter_after:
                    self.get_logger().warn(
                        f'Not converged after {steps_since_scatter} steps, re-scattering')
                    self.scatter()
                    steps_since_scatter = 0
                    continue

            self.turn_step()
            steps_since_scatter += 1
        self.stop()

        if confirmed > steps_per_turn:
            pose = self.pose.pose.pose
            yaw = 2 * math.atan2(pose.orientation.z, pose.orientation.w)
            self.get_logger().info(
                f'Localized at x={pose.position.x:.2f}, y={pose.position.y:.2f}, '
                f'yaw={math.degrees(yaw):.0f} deg')
        else:
            self.get_logger().warn(
                'Not localized. Move the robot a bit and run this node again, '
                'or set the pose manually with "2D Pose Estimate" in RViz.')
        if self.dbg:
            self.dbg.finish(confirmed > steps_per_turn)


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