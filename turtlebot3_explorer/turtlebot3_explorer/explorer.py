"""Frontier-based autonomous exploration for ROS 2 Humble.

Runs on the remote PC next to Cartographer and Nav2.

Inputs : /map (OccupancyGrid), TF map -> robot_frame
Outputs: NavigateToPose goals (action /navigate_to_pose),
         optional Spin goal at start (action /spin),
         at the end: Cartographer .pbstream + Nav2 map files.

State machine:  WAIT_FOR_DATA -> SPIN -> SELECT -> NAVIGATING -> SELECT ... -> DONE

Goal selection: only frontiers that are reachable over known, safe free
cells are candidates; the score is the path length to them (nearest first),
with a small bonus for large frontiers.
"""
import os

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

def frontier_mask(data, free_max):
    """Known free cells that have at least one unknown 4-neighbour."""
    free = (data >= 0) & (data < free_max)
    unk = (data == UNKNOWN)
    near_unk = np.zeros_like(unk)
    near_unk[1:, :] |= unk[:-1, :]
    near_unk[:-1, :] |= unk[1:, :]
    near_unk[:, 1:] |= unk[:, :-1]
    near_unk[:, :-1] |= unk[:, 1:]
    return free & near_unk

def reachable_distance(data,res , origin_xy, start_xy, free_max, distance):
    height, width = data.shape
    delta_x = int((start_xy[0] - origin_xy[0])/res)
    delta_y = int((start_xy[1] - origin_xy[1])/res)
    if not (0 <= delta_x<width and 0 <= delta_y<height):
        return None

    r = max(0, int(math.ceil(distance/res)))
    yy,xx = np.ogrid[-r:r+1, -r:r+1]
    disc = (xx * xx + yy * yy) <= r * r
    blocked = ndimage.binary_dilation(data >= free_max, structure=disc)
    free = (data >= 0) & (data < free_max)
    ok = free & ~blocked

    # the robot may stand inside its own inflation zone (e.g. next to a wall)
    y0, y1 = max(0, delta_y - r), min(height, delta_y + r + 1)
    x0, x1 = max(0, delta_x - r), min(width, delta_x + r + 1)
    ok[y0:y1, x0:x1] |= free[y0:y1, x0:x1]
    ok[delta_y, delta_x] = True

    dist = np.full(data.shape, np.inf)
    dist[delta_y, delta_x] = 0.0
    visited = np.zeros(data.shape, dtype=bool)
    visited[delta_y, delta_x] = True
    wave = visited.copy()
    cross = ndimage.generate_binary_structure(2, 1)
    step = 0

    while wave.any():
        step +=1
        wave = ndimage.binary_dilation(wave, structure=cross) & ok & ~visited
        dist[wave] = step*res
        visited |= wave

    return dist


class Explorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # ---- parameters -------------------------------------------------
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('min_frontier_size_m', 0.30)   # ~ corridor width
        self.declare_parameter('gain_weight', 0.1)            # bonus per metre of frontier length
        self.declare_parameter('blacklist_radius_m', 0.4)
        self.declare_parameter('tick_period_s', 1.0)
        self.declare_parameter('initial_spin', True)
        self.declare_parameter('empty_ticks_to_finish', 3)    # robustness against flicker
        # [x_min, y_min, x_max, y_max] in map frame; empty = unbounded.
        # Use this to keep the robot inside the building (gaps in walls!).
        self.declare_parameter('bounds', [-100.0, -100.0, 100.0, 100.0])
        # Automatic bounds: box around the walls that border the drivable
        # area, plus a small margin. 'bounds' above stays as a hard cap.
        self.declare_parameter('auto_bounds', True)
        self.declare_parameter('auto_bounds_margin_m', 0.15)
        self.declare_parameter('wall_near_m', 0.30)  # wall counts if this close to drivable space
        self.declare_parameter('save_prefix', 'lab_map')
        self.declare_parameter('free_max', 50) # treat cells with more than this as occupied
        self.declare_parameter('inflate_meters', 0.15)
        # TurtleBot3 Burger footprint radius is ~0.105 m
        self.declare_parameter('robot_radius_m', 0.11)
        # must be larger than Nav2's xy_goal_tolerance (0.25 m in the TB3
        # params), otherwise the goal "succeeds" instantly without moving
        self.declare_parameter('min_goal_dist_meters', 0.35)
        self.declare_parameter('goal_timeout_s', 60.0) # TODO: tune, max map building time = 20 min
        # attempts on the same area (counted over the WHOLE run, not just
        # consecutively) before it is blacklisted
        self.declare_parameter('max_repeat_goals', 3)
        self.declare_parameter('attempt_radius_m', 0.5)

        p = self.get_parameter
        self.robot_frame = p('robot_frame').value
        self.map_frame = p('map_frame').value
        self.min_frontier_m = p('min_frontier_size_m').value
        self.gain_weight = p('gain_weight').value
        self.blacklist_radius = p('blacklist_radius_m').value
        self.initial_spin = p('initial_spin').value
        self.empty_ticks_needed = p('empty_ticks_to_finish').value
        self.free_max = p('free_max').value
        self.inflate_m = p('inflate_meters').value
        self.min_goal_dist = p('min_goal_dist_meters').value
        self.goal_timeout = p('goal_timeout_s').value
        self.max_repeat = p('max_repeat_goals').value
        self.robot_radius = p('robot_radius_m').value
        self.attempt_radius = p('attempt_radius_m').value
        if self.inflate_m < self.robot_radius:
            self.get_logger().warn(
                f'inflate_meters={self.inflate_m} < robot_radius_m={self.robot_radius}: '
                f'using {self.robot_radius} m (Nav2 cannot plan through narrower gaps)')

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

        self.auto_bounds = p('auto_bounds').value
        self.auto_margin = p('auto_bounds_margin_m').value
        self.wall_near = p('wall_near_m').value
        self.building_box = None   # [x_min, y_min, x_max, y_max], updated every select

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
        self.goal_started = None  # clock time the active goal was sent
        self.cancelling = False
        self.timed_out = False
        self.attempts = []       # [[x, y, count]] per area, for repeat detection
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

        data, _ = self._grid()
        if (data >= self.free_max).sum() < 20:      # no walls seen yet -> map not ready
            self.get_logger().info('waiting for walls in the map ...',
                                   throttle_duration_sec=5.0)
            return
        if not self.nav_client.server_is_ready():
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
        goal.target_yaw = 2.0 * math.pi
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
        if frontiers is None:
            return
        if not frontiers:
            self.empty_ticks += 1
            if self.empty_ticks >= self.empty_ticks_needed:
                self.get_logger().info('no frontiers left -> exploration finished')
                self.state = State.DONE
                self.done = True
            return
        self.empty_ticks = 0

        best = min(frontiers, key=lambda f: f['score'])

        if self._count_attempt(best['goal']) > self.max_repeat:
            self.get_logger().warn(
                f'area around ({best["goal"][0]:.2f}, {best["goal"][1]:.2f}) '
                f'tried {self.max_repeat}x -> blacklist')
            self.blacklist.append(best['goal'])
            return  # reselect on next tick

        self.target = best['goal']
        self._send_nav_goal(*self.target)
        self.state = State.NAVIGATING

    def _tick_navigating(self):
        # Re-validate: if the target got explored meanwhile (or map shifted
        # after a loop closure), cancel and re-select.
        if self.cancelling or self.goal_handle is None:
            return

        # stuck / too slow: Nav2 recoveries can loop for a long time
        elapsed = (self.get_clock().now() - self.goal_started).nanoseconds * 1e-9
        if elapsed > self.goal_timeout:
            self.get_logger().warn(f'goal timeout after {elapsed:.0f}s -> cancel + blacklist')
            self.timed_out = True
            self.cancelling = True
            self.goal_handle.cancel_goal_async()
            return

        # target explored meanwhile (or map shifted after a loop closure)
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
        rxy = self._robot_xy()
        yaw = math.atan2(y - rxy[1], x - rxy[0]) if rxy else 0.0
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f'goal -> ({x:.2f}, {y:.2f})')
        self.goal_handle = None
        self.cancelling = False
        self.timed_out = False
        self.goal_started = self.get_clock().now()
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
        elif self.timed_out:
            self._blacklist_target()
        # SUCCEEDED / CANCELED (not timeout) need no blacklisting
        self.goal_handle = None
        self.cancelling = False
        self.timed_out = False
        self.state = State.SELECT

    def _count_attempt(self, goal):
        """Increment and return the attempt count for the area around goal."""
        for a in self.attempts:
            if math.hypot(goal[0] - a[0], goal[1] - a[1]) < self.attempt_radius:
                a[2] += 1
                return a[2]
        self.attempts.append([goal[0], goal[1], 1])
        return 1

    def _blacklist_target(self):
        if self.target is not None:
            self.blacklist.append(self.target)

    # ================= frontier detection ================================
    def _grid(self):
        m = self.map_msg
        data = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        return data, m.info

    def _find_frontiers(self):
        """List of candidate dicts, [] if none, None on a transient error."""
        if self.map_msg is None:
            self.get_logger().warn('Frontiers not found: map_msg is None')
            return None
        rxy = self._robot_xy()
        if rxy is None:
            self.get_logger().warn('Frontiers not found: coordinates could not be determined')
            return None
        data, info = self._grid()
        # never plan with less clearance than the robot's own radius
        relaxed = self.robot_radius
        for clearance in (max(self.inflate_m, relaxed), relaxed):
            res = info.resolution
            origin = (info.origin.position.x, info.origin.position.y)

            dist = reachable_distance(data, res, origin, rxy, self.free_max, clearance)
            if dist is None:
                self.get_logger().warn('Frontiers not found: robot outside map')
                return None

            raw = frontier_mask(data, self.free_max)
            reach = np.isfinite(dist)
            snap = int(math.ceil(clearance / res)) + 2
            near = ndimage.binary_dilation(reach, structure=np.ones((3, 3)), iterations=snap)
            mask = raw & near
            labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
            min_cells = max(1, int(self.min_frontier_m / res))
            _, (ny, nx) = ndimage.distance_transform_edt(~reach, return_indices=True)
            if self.auto_bounds:
                self.building_box = self._building_box(data, reach, origin, res, clearance)

            result = []
            for lbl in range(1, n + 1):
                ys, xs = np.nonzero(labels == lbl)
                if len(xs) < min_cells:
                    continue
                cx, cy = xs.mean(), ys.mean()
                k = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
                sy, sx = ny[ys[k], xs[k]], nx[ys[k], xs[k]]
                gx = origin[0] + (sx + 0.5) * res
                gy = origin[1] + (sy + 0.5) * res
                path_len = dist[sy, sx]
                if path_len < self.min_goal_dist:
                    continue
                if not self._allowed(gx, gy):
                    continue
                score = path_len - self.gain_weight * len(xs) * res
                result.append({'goal': (gx, gy), 'score': score})

            self.get_logger().info(
                f'[clearance {clearance:.2f} m] frontier cells: {int(raw.sum())} raw, '
                f'{int(mask.sum())} near drivable ({int(reach.sum())} drivable) '
                f'-> {n} clusters, {len(result)} usable')
            if result:
                if clearance < max(self.inflate_m, relaxed):
                    self.get_logger().warn(
                        f'no frontier with normal clearance -> relaxed to {relaxed:.2f} m')
                return result
        return []

    def _is_frontier_cell(self, x, y):
        data, info = self._grid()
        res = info.resolution
        ix = int((x - info.origin.position.x) / res)
        iy = int((y - info.origin.position.y) / res)
        if not (0 <= ix < info.width and 0 <= iy < info.height):
            return False
        # look at a small window so tiny map shifts do not cancel the goal
        win = frontier_mask(data, self.free_max)[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        return bool(win.any())

    def _building_box(self, data, reach, origin, res, clearance):
        """Bounding box of wall cells that border the drivable area.

        Only walls next to space the robot can reach are counted, so
        furniture seen far away through a small gap does not grow the box.
        Returns None while too few walls are known (then only 'bounds' applies).
        """
        k = int(math.ceil((clearance + self.wall_near) / res))
        near = ndimage.binary_dilation(reach, structure=np.ones((3, 3)), iterations=k)
        ys, xs = np.nonzero((data >= self.free_max) & near)
        if len(xs) < 20:
            return None
        m = self.auto_margin
        box = [origin[0] + xs.min() * res - m, origin[1] + ys.min() * res - m,
               origin[0] + (xs.max() + 1) * res + m, origin[1] + (ys.max() + 1) * res + m]
        self.get_logger().info(
            'building box: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(*box),
            throttle_duration_sec=10.0)
        return box

    def _allowed(self, x, y):

        # hard cap from the 'bounds' parameter
        if len(self.bounds) == 4:
            xmin, ymin, xmax, ymax = self.bounds
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                return False
        # automatic bounds learned from the walls seen so far
        if self.building_box is not None:
            xmin, ymin, xmax, ymax = self.building_box
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
        """Call after the spin loop ends (blocking calls are fine here).

        Order: .pgm/.yaml first (needs a live /map), then finish the
        Cartographer trajectory and write the .pbstream. Every step is
        checked; nothing is reported as saved unless the file exists.
        Returns True only if all three files were written.
        """
        log = self.get_logger()
        prefix = os.path.abspath(os.path.expanduser(self.save_prefix))
        out_dir = os.path.dirname(prefix)

        # map_saver / GraphicsMagick do NOT create missing directories and
        # only report "Magick: Unable to open file" -> make sure it exists
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            log.error(f'cannot create map directory {out_dir}: {e}')
            return False
        if not os.access(out_dir, os.W_OK):
            log.error(f'map directory {out_dir} is not writable '
                      f'(owned by root from an earlier sudo/docker run?)')
            return False
        for ext in ('.pgm', '.yaml', '.pbstream'):
            f = prefix + ext
            if os.path.exists(f) and not os.access(f, os.W_OK):
                log.error(f'{f} exists and is read-only -> remove it or pick another prefix')
                return False

        ok = True

        # 1) .yaml/.pgm for Nav2 map_server + AMCL
        proc = subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                               '-f', prefix], check=False)
        if proc.returncode != 0 or not os.path.isfile(prefix + '.pgm'):
            log.error(f'map_saver_cli failed (exit code {proc.returncode})')
            ok = False

        # 2) .pbstream (path is on the machine running cartographer_node)
        fin = self.create_client(FinishTrajectory, '/finish_trajectory')
        wr = self.create_client(WriteState, '/write_state')
        if fin.wait_for_service(timeout_sec=5.0):
            req = FinishTrajectory.Request()
            req.trajectory_id = 0
            res = self._call(fin, req)
            if res is None or res.status.code != 0:
                log.warn(f'finish_trajectory: {self._status_text(res)}')
        else:
            log.warn('/finish_trajectory not available')

        if wr.wait_for_service(timeout_sec=5.0):
            req = WriteState.Request()
            req.filename = prefix + '.pbstream'
            req.include_unfinished_submaps = True
            res = self._call(wr, req)
            if res is None or res.status.code != 0:
                log.error(f'write_state: {self._status_text(res)}')
                ok = False
        else:
            log.error('/write_state not available -> no .pbstream')
            ok = False

        if ok:
            log.info(f'saved {prefix}.pbstream and {prefix}.yaml/.pgm')
        else:
            log.error(f'map saving incomplete, check files with prefix {prefix}')
        return ok

    @staticmethod
    def _status_text(res):
        if res is None:
            return 'no response (timeout)'
        return f'code {res.status.code}: {res.status.message}'

    def _call(self, client, req):
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)
        return fut.result()

    def _dump_map(self, data, raw, reach, rxy, info):
        res = info.resolution
        rx = int((rxy[0] - info.origin.position.x) / res)
        ry = int((rxy[1] - info.origin.position.y) / res)
        rows = []
        for y in range(data.shape[0] - 1, -1, -1):
            row = ''
            for x in range(data.shape[1]):
                v = data[y, x]
                if (x, y) == (rx, ry):
                    c = 'R'
                elif raw[y, x]:
                    c = 'F'
                elif reach[y, x]:
                    c = 'o'
                elif v < 0:
                    c = ' '
                elif v >= 65:
                    c = '#'
                elif v >= self.free_max:
                    c = '+'
                else:
                    c = '.'
                row += c
            rows.append(row)
        self.get_logger().info('map (top = +y):\n' + '\n'.join(rows),
                               throttle_duration_sec=20.0)


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