import math
import subprocess
from enum import Enum, auto

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cartographer_ros_msgs.srv import FinishTrajectory, WriteState
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy import ndimage
from tf2_ros import Buffer, TransformException, TransformListener

UNKNOWN, FREE = -1, 0


class State(Enum):
    WAIT_FOR_DATA = auto()
    SPIN = auto()
    SELECT = auto()
    NAVIGATING = auto()
    DONE = auto()


"""Frontier-based autonomous exploration for ROS 2 Humble.

Runs on the remote PC next to Cartographer and Nav2.

Inputs : /map (OccupancyGrid), TF map -> base_link
Outputs: NavigateToPose goals (action /navigate_to_pose),
         optional Spin goal at start (action /spin),
         at the end: Cartographer .pbstream + Nav2 map files.

State machine:  WAIT_FOR_DATA -> SPIN -> SELECT -> NAVIGATING -> SELECT ... -> DONE
"""
class Explorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # ---- parameters -------------------------------------------------
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('min_frontier_size_m', 0.30)   # ~ corridor width
        self.declare_parameter('gain_weight', 0.5)            # bonus per metre of frontier length
        self.declare_parameter('blacklist_radius_m', 0.4)
        self.declare_parameter('tick_period_s', 1.0)
        self.declare_parameter('initial_spin', True)
        self.declare_parameter('empty_ticks_to_finish', 3)    # robustness against flicker
        # [x_min, y_min, x_max, y_max] in map frame; empty = unbounded.
        # Use this to keep the robot inside the building (gaps in walls!).
        self.declare_parameter('bounds', [-100.0, -100.0, 100.0, 100.0])
        self.declare_parameter('save_prefix', 'lab_map')

        p = self.get_parameter
        self.robot_frame = p('robot_frame').value
        self.map_frame = p('map_frame').value
        self.min_frontier_m = p('min_frontier_size_m').value
        self.gain_weight = p('gain_weight').value
        self.blacklist_radius = p('blacklist_radius_m').value
        self.initial_spin = p('initial_spin').value
        self.empty_ticks_needed = p('empty_ticks_to_finish').value

        b = list(p('bounds').value)
        if len(b) != 4:
            raise ValueError('bounds must be [x_min, y_min, x_max, y_max]')
        if any(b):  # not all zeros -> bounded
            if not (b[0] < b[2] and b[1] < b[3]):
                raise ValueError('bounds need x_min < x_max and y_min < y_max')
            self.bounds = b
        else:
            self.bounds = []   # treated as unbounded
        self.get_logger().info(f'Bounds set to: {self.bounds}')

        self.save_prefix = p('save_prefix').value

        # ---- ROS interfaces ---------------------------------------------

        # Subscribe to map topic
        map_qos = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.BEST_EFFORT) ## TODO: check QoS
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        # Buffer collects tf topics to get where robot is each time when asked
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, '/spin')

        # ---- internal state ---------------------------------------------
        self.state = State.WAIT_FOR_DATA
        self.map_msg = None
        self.blacklist = []            # [(x, y)] in map frame
        self.target = None             # (x, y) of the active goal
        self.goal_handle = None
        self.cancelling = False
        self.empty_ticks = 0
        self.done = False

        # tick is made each ['tick_period_s']
        self.create_timer(p('tick_period_s').value, self._tick)
        self.get_logger().info('frontier_explorer started')

    # ================= callbacks =========================================
    def _on_map(self, msg):
        self.map_msg = msg # stores newest occupancy grid

    def _tick(self): # switches between different exploration stages based on state
        if self.state == State.WAIT_FOR_DATA:
            self._tick_wait()
        elif self.state == State.SELECT:
            self._tick_select()
        elif self.state == State.NAVIGATING:
            self._tick_navigating()
        # DONE does nothing, since SPIN is advanced by its result callback

    # ================= state handlers ====================================
    def _tick_wait(self):
        if self.map_msg is None or self._robot_xy() is None:
            return
        if not (self.nav_client.server_is_ready()):
            self.get_logger().info('waiting for /navigate_to_pose ...',
                                   throttle_duration_sec=5.0)
            return
        if self.initial_spin and self.spin_client.server_is_ready():
            self._start_spin()
        else:
            self.state = State.SELECT

    def _start_spin(self):
        self.state = State.SPIN
        goal = Spin.Goal()
        goal.target_yaw = 0 # or 2.0 * math.pi
        goal.time_allowance = Duration(seconds=20).to_msg()
        self.get_logger().info('initial spin')
        fut = self.spin_client.send_goal_async(goal)
        fut.add_done_callback(self._on_spin_accepted)

    def _on_spin_accepted(self, fut):
        handle = fut.result()
        if not handle.accepted:
            self.state = State.SELECT
            return
        handle.get_result_async().add_done_callback(
            lambda _: setattr(self, 'state', State.SELECT))

    def _tick_select(self): # selecting the next goal
        frontiers = self._find_frontiers()
        if not frontiers:
            self.empty_ticks += 1
            if self.empty_ticks >= self.empty_ticks_needed:
                self.get_logger().info('no frontiers left -> exploration finished')
                self.state = State.DONE
                self.done = True
            return
        self.empty_ticks = 0

        best = min(frontiers, key=lambda f: f['score'])
        self.target = best['goal']
        self._send_nav_goal(*self.target)
        self.state = State.NAVIGATING

    def _tick_navigating(self):
        # Re-validate: if the target got explored meanwhile (or map shifted
        # after a loop closure), cancel and re-select.
        if self.cancelling or self.goal_handle is None:
            return
        if not self._is_frontier_cell(*self.target):
            self.get_logger().info('target no longer a frontier -> cancel')
            self.cancelling = True
            self.goal_handle.cancel_goal_async()

    # ================= navigation ========================================
    def _send_nav_goal(self, x, y):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        # Face the unknown: simple heuristic = keep heading from robot to goal
        rxy = self._robot_xy()
        yaw = math.atan2(y - rxy[1], x - rxy[0]) if rxy else 0.0
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f'goal -> ({x:.2f}, {y:.2f})')
        self.goal_handle = None
        self.cancelling = False
        self.nav_client.send_goal_async(goal).add_done_callback(self._on_goal_response)

    def _on_goal_response(self, fut):
        handle = fut.result()
        if not handle.accepted:
            self.get_logger().warn('goal rejected -> blacklist')
            self._blacklist_target()
            self.state = State.SELECT
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_nav_result)

    def _on_nav_result(self, fut):
        status = fut.result().status
        if status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn('goal aborted -> blacklist')
            self._blacklist_target()
        # SUCCEEDED / CANCELED need no blacklisting
        self.goal_handle = None
        self.cancelling = False
        self.state = State.SELECT

    def _blacklist_target(self):
        if self.target is not None:
            self.blacklist.append(self.target)

    # ================= frontier detection ================================
    def _grid(self):
        m = self.map_msg
        data = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        return data, m.info

    def _frontier_mask(self, data):
        """Free cells that have at least one unknown 4-neighbour."""
        free =( data == FREE )
        unk = ( data == UNKNOWN )
        near_unk = np.zeros_like(unk)
        near_unk[1:, :] |= unk[:-1, :]
        near_unk[:-1, :] |= unk[1:, :]
        near_unk[:, 1:] |= unk[:, :-1]
        near_unk[:, :-1] |= unk[:, 1:]
        return free & near_unk

    def _find_frontiers(self):
        if self.map_msg is None:
            return []
        rxy = self._robot_xy()
        if rxy is None:
            return []
        data, info = self._grid()
        res = info.resolution

        # groups touching frontier cells into numbered clusters
        labels, n = ndimage.label(self._frontier_mask(data),
                                  structure=np.ones((3, 3))) # 8-connectivity so diagonal neighbours count

        # converts 30 cm minimum into cells, smaller clusters (cause noise) are dropped
        min_cells = max(1, int(self.min_frontier_m / res))

        result = []
        for lbl in range(1, n + 1):
            ys, xs = np.nonzero(labels == lbl)
            if len(xs) < min_cells:
                continue
            # goal = member cell closest to the cluster centroid
            cx, cy = xs.mean(), ys.mean()
            k = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
            gx = info.origin.position.x + (xs[k] + 0.5) * res
            gy = info.origin.position.y + (ys[k] + 0.5) * res
            if not self._allowed(gx, gy):
                continue
            dist = math.hypot(gx - rxy[0], gy - rxy[1])
            score = dist - self.gain_weight * len(xs) * res   # lower = better
            result.append({'goal': (gx, gy), 'score': score})
        return result

    def _is_frontier_cell(self, x, y):
        data, info = self._grid()
        res = info.resolution
        ix = int((x - info.origin.position.x) / res)
        iy = int((y - info.origin.position.y) / res)
        if not (0 <= ix < info.width and 0 <= iy < info.height):
            return False
        # look at a small window so tiny map shifts do not cancel the goal
        win = self._frontier_mask(data)[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        return bool(win.any())

    def _allowed(self, x, y):

        # cell passes if it is inside [bounds]
        if len(self.bounds) == 4:
            xmin, ymin, xmax, ymax = self.bounds
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                return False
        return all(math.hypot(x - bx, y - by) > self.blacklist_radius
                   for bx, by in self.blacklist)

    # ================= TF ================================================
    def _robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, rclpy.time.Time())
        except TransformException:
            return None
        return t.transform.translation.x, t.transform.translation.y

    # ================= saving ============================================
    def save_map(self):
        """Call after the spin loop ends (blocking calls are fine here)."""
        pbstream = f'{self.save_prefix}.pbstream'

        fin = self.create_client(FinishTrajectory, '/finish_trajectory')
        wr = self.create_client(WriteState, '/write_state')
        if fin.wait_for_service(5.0):
            req = FinishTrajectory.Request()
            req.trajectory_id = 0
            self._call(fin, req)
        if wr.wait_for_service(5.0):
            req = WriteState.Request()
            req.filename = pbstream   # path is on the machine running cartographer_node
            req.include_unfinished_submaps = True
            self._call(wr, req)

        # .yaml/.pgm for Nav2 map_server + AMCL
        subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                        '-f', self.save_prefix], check=False)
        self.get_logger().info(f'saved {pbstream} and {self.save_prefix}.yaml/.pgm')

    def _call(self, client, req):
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)
        return fut.result()


def main():
    rclpy.init()
    node = Explorer()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.done:
            node.save_map()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
