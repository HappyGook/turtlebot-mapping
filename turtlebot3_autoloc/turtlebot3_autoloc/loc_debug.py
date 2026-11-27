"""Debug helper for auto_localize.

Logs what AMCL is configured with, what it sees (map, scan, TF) and what its
particle cloud does after every stop. Additionally it scores the current scan
against the map *independently of AMCL* (same likelihood-field model AMCL uses)
so we can tell apart:

  * AMCL has (almost) no laser information   -> few usable beams
  * map is not what AMCL expects             -> map cell values other than 0/100/-1
  * particle depletion                       -> the global best scan match is
                                                somewhere AMCL did not end up
  * model/map/TF problem                     -> the global best match is the
                                                same wrong spot AMCL picks
  * mirrored / flipped scan                  -> mirrored scan matches better

Optional ground truth (true_x, true_y, true_yaw parameters, in the map frame)
makes the report much more useful: it counts particles near the real pose and
scores the scan there.

Snapshots (PNG if matplotlib is available + one .npz with everything) are
written to debug_dir/<timestamp>/.
"""

import json
import math
import os
import time

import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.msg import ParticleCloud
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

AMCL_PARAMS = [
    'base_frame_id', 'odom_frame_id', 'global_frame_id', 'scan_topic', 'map_topic',
    'robot_model_type', 'laser_model_type', 'laser_max_range', 'laser_min_range',
    'max_beams', 'z_hit', 'z_rand', 'z_short', 'z_max', 'sigma_hit',
    'laser_likelihood_max_dist', 'do_beamskip', 'min_particles', 'max_particles',
    'pf_err', 'pf_z', 'resample_interval', 'recovery_alpha_slow', 'recovery_alpha_fast',
    'update_min_a', 'update_min_d', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5',
    'tf_broadcast', 'transform_tolerance', 'set_initial_pose',
]

COLLAPSE_STD = 0.5   # m, cloud xy std below this counts as "collapsed"


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def param_value(v):
    return {
        ParameterType.PARAMETER_BOOL: lambda: v.bool_value,
        ParameterType.PARAMETER_INTEGER: lambda: v.integer_value,
        ParameterType.PARAMETER_DOUBLE: lambda: v.double_value,
        ParameterType.PARAMETER_STRING: lambda: v.string_value,
    }.get(v.type, lambda: None)()


def rotate(q, v):
    """Rotate 3-vector(s) v (..., 3) by quaternion q (geometry_msgs Quaternion)."""
    u = np.array([q.x, q.y, q.z])
    t = 2.0 * np.cross(u, v)
    return v + q.w * t + np.cross(u, t)


class GridMap:
    """Occupancy grid interpreted exactly like Nav2 AMCL does (0 free, 100 occ, else unknown)."""

    def __init__(self, msg, max_dist):
        info = msg.info
        self.res = info.resolution
        self.w, self.h = info.width, info.height
        self.ox, self.oy = info.origin.position.x, info.origin.position.y
        self.origin_yaw = yaw_of(info.origin.orientation)
        self.data = np.array(msg.data, dtype=np.int16).reshape(self.h, self.w)
        self.free = self.data == 0
        self.occ = self.data == 100
        self.max_dist = max_dist
        self.dist = self._distance_field()

    def _distance_field(self):
        if not self.occ.any():
            return np.full(self.data.shape, self.max_dist)
        try:
            from scipy.ndimage import distance_transform_edt
            d = distance_transform_edt(~self.occ) * self.res
        except ImportError:   # slow-ish fallback: brushfire with 8-neighbourhood
            d = np.full(self.data.shape, np.inf)
            d[self.occ] = 0.0
            steps = int(math.ceil(self.max_dist / self.res)) + 1
            for _ in range(steps):
                p = np.pad(d, 1, constant_values=np.inf)
                cand = np.minimum.reduce([
                    p[1:-1, :-2] + self.res, p[1:-1, 2:] + self.res,
                    p[:-2, 1:-1] + self.res, p[2:, 1:-1] + self.res,
                    p[:-2, :-2] + self.res * 1.414, p[:-2, 2:] + self.res * 1.414,
                    p[2:, :-2] + self.res * 1.414, p[2:, 2:] + self.res * 1.414])
                d = np.minimum(d, cand)
        return np.minimum(d, self.max_dist)

    def cells(self, x, y):
        gx = np.floor((np.asarray(x) - self.ox) / self.res).astype(np.int64)
        gy = np.floor((np.asarray(y) - self.oy) / self.res).astype(np.int64)
        valid = (gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h)
        return gx, gy, valid

    def dist_at(self, x, y):
        gx, gy, valid = self.cells(x, y)
        out = np.full(gx.shape, self.max_dist)
        out[valid] = self.dist[gy[valid], gx[valid]]
        return out

    def is_free(self, x, y):
        gx, gy, valid = self.cells(x, y)
        out = np.zeros(gx.shape, dtype=bool)
        out[valid] = self.free[gy[valid], gx[valid]]
        return out

    def extent(self):
        return [self.ox, self.ox + self.w * self.res, self.oy, self.oy + self.h * self.res]


class LocalizationDebugger:

    def __init__(self, node):
        self.node = node
        self.log = node.get_logger()
        p = node.p
        self.truth = tuple(float(p(n)) for n in ('true_x', 'true_y', 'true_yaw'))
        self.has_truth = not (math.isnan(self.truth[0]) or math.isnan(self.truth[1]))
        self.png = bool(p('debug_png'))
        self.global_search = bool(p('debug_global_search'))
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.dir = os.path.join(os.path.expanduser(p('debug_dir')), stamp)
        os.makedirs(self.dir, exist_ok=True)

        self.amcl = {}
        self.map_msg = None
        self.map = None
        self.scan = None
        self.odom_yaw = None
        self.cloud = None
        self.pose = None
        self.cloud_msgs = 0
        self.pose_msgs = 0
        self.last_cloud_t = None
        self.last_pose_t = None
        self.laser = None        # (x, y, quaternion) of laser in base frame
        self.laser_frame = None
        self.prev_odom_yaw = None
        self.collapsed_at = None
        self.snapshots = []
        self.diagnoses = []
        self.peaks = []

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(OccupancyGrid, '/map', self._map_cb, latched)
        node.create_subscription(LaserScan, p('scan_topic'), self._scan_cb,
                                 qos_profile_sensor_data)
        node.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        node.create_subscription(ParticleCloud, '/particle_cloud', self._cloud_cb,
                                 qos_profile_sensor_data)
        node.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._pose_cb, 10)
        self.get_params = node.create_client(GetParameters, '/amcl/get_parameters')

        try:
            from tf2_ros import Buffer, TransformListener
            self.tf = Buffer()
            self._tfl = TransformListener(self.tf, node)
        except ImportError:
            self.tf = None
            self.log.warn('[dbg] tf2_ros not available: no TF checks, laser assumed at base origin')

        self.log.info(f'[dbg] debug output -> {self.dir}')
        if self.has_truth:
            self.log.info(f'[dbg] ground truth x={self.truth[0]:.2f} y={self.truth[1]:.2f} '
                          f'yaw={self.truth[2]:.2f} rad (nan = unknown)')

    # ---------- callbacks ----------

    def _map_cb(self, msg):
        self.map_msg = msg

    def _scan_cb(self, msg):
        self.scan = msg

    def _odom_cb(self, msg):
        self.odom_yaw = yaw_of(msg.pose.pose.orientation)

    def _cloud_cb(self, msg):
        self.cloud = msg
        self.cloud_msgs += 1
        self.last_cloud_t = time.time()

    def _pose_cb(self, msg):
        self.pose = msg
        self.pose_msgs += 1
        self.last_pose_t = time.time()

    # ---------- setup ----------

    def setup(self):
        """Call after AMCL is up and tuned."""
        self._read_amcl_params()
        self._wait_for('map', lambda: self.map_msg is not None, 10.0)
        self._wait_for('scan', lambda: self.scan is not None, 5.0)
        self._wait_for('odom', lambda: self.odom_yaw is not None, 5.0)
        self._check_scan()
        self._check_map()
        self._check_tf()

    def _wait_for(self, what, cond, timeout):
        end = time.time() + timeout
        while not cond() and time.time() < end:
            self.node.wait(0.1)
        if not cond():
            self.log.warn(f'[dbg] no {what} received within {timeout:.0f} s')

    def a(self, name, default):
        v = self.amcl.get(name)
        return default if v is None else v

    def _read_amcl_params(self):
        if not self.get_params.wait_for_service(timeout_sec=5.0):
            self.log.warn('[dbg] /amcl/get_parameters not available')
            return
        req = GetParameters.Request()
        req.names = AMCL_PARAMS
        res = self.node.call(self.get_params, req)
        if res is None:
            self.log.warn('[dbg] no answer from /amcl/get_parameters')
            return
        self.amcl = {n: param_value(v) for n, v in zip(AMCL_PARAMS, res.values)}
        self.log.info('[dbg] AMCL params: ' + ', '.join(f'{k}={v}' for k, v in self.amcl.items()))
        if self.amcl.get('laser_model_type') not in (None, 'likelihood_field'):
            self.log.warn(f'[dbg] laser_model_type={self.amcl["laser_model_type"]}: the '
                          'independent scan check below models likelihood_field only')
        if self.a('resample_interval', 1) == 1:
            self.log.info('[dbg] resample_interval=1: every forced update resamples '
                          '(published weights will look uniform)')

    def _range_limits(self, scan):
        rmax = scan.range_max
        lmax = self.a('laser_max_range', -1.0)
        if lmax > 0:
            rmax = min(rmax, lmax)
        rmin = max(scan.range_min, self.a('laser_min_range', -1.0))
        return rmin, rmax

    def _check_scan(self):
        s = self.scan
        if s is None:
            return
        r = np.array(s.ranges, dtype=float)
        finite = np.isfinite(r) & (r > s.range_min) & (r < s.range_max)
        rmin, rmax = self._range_limits(s)
        used = finite & (r > rmin) & (r < rmax)
        self.log.info(
            f'[dbg] scan frame={s.header.frame_id} beams={len(r)} '
            f'angle {math.degrees(s.angle_min):.0f}..{math.degrees(s.angle_max):.0f} deg, '
            f'sensor range {s.range_min:.2f}..{s.range_max:.2f} m, AMCL uses <{rmax:.2f} m')
        self.log.info(
            f'[dbg] scan beams: {finite.sum()}/{len(r)} valid, {used.sum()} inside AMCL range '
            f'({100.0 * used.sum() / max(len(r), 1):.0f}%), median valid range '
            f'{np.median(r[finite]) if finite.any() else float("nan"):.2f} m')
        # range histogram + where the very short returns are (self-occlusion check)
        edges = [0.0, 0.15, 0.3, 0.5, 1.0, 1.5, 2.5, 3.5, np.inf]
        hist, _ = np.histogram(r[finite], bins=edges)
        self.log.info('[dbg] scan range histogram: ' + ', '.join(
            f'{lo:g}-{hi:g} m: {n}' for lo, hi, n in zip(edges[:-1], edges[1:], hist)))
        ang = s.angle_min + np.arange(len(r)) * s.angle_increment
        near = finite & (r < 0.3)
        if near.any():
            sectors = np.unique(np.floor(((ang[near] % (2 * np.pi)) / (np.pi / 4))).astype(int))
            self.log.info(f'[dbg] {near.sum()} beams < 0.3 m, in {len(sectors)}/8 '
                          f'45-deg sectors {sectors.tolist()} (0 = straight ahead, CCW)')
            if near.sum() > 0.3 * finite.sum():
                self.log.warn(
                    '[dbg] most beams hit something within 0.3 m -> the lidar is probably '
                    'seeing the robot itself (camera / mount / cables) or the robot is boxed '
                    'in. Such returns turn with the robot and carry no map information.')
        if used.sum() < 0.2 * len(r):
            self.log.warn(
                f'[dbg] only {used.sum()} beams are shorter than {rmax:.2f} m -> AMCL gets '
                'very little information per update (laser_max_range too small?)')

    def _check_map(self):
        if self.map_msg is None:
            return
        self.map = GridMap(self.map_msg, self.a('laser_likelihood_max_dist', 2.0))
        m = self.map
        d = m.data
        n = d.size
        unknown = (d == -1).sum()
        other = n - m.free.sum() - m.occ.sum() - unknown
        self.log.info(
            f'[dbg] map {m.w}x{m.h} @ {m.res:.3f} m, origin ({m.ox:.2f}, {m.oy:.2f}), '
            f'extent x {m.extent()[0]:.1f}..{m.extent()[1]:.1f} y {m.extent()[2]:.1f}..'
            f'{m.extent()[3]:.1f}')
        self.log.info(
            f'[dbg] map cells: free(0)={m.free.sum()} ({m.free.sum() * m.res ** 2:.1f} m2), '
            f'occupied(100)={m.occ.sum()}, unknown(-1)={unknown}, other values={other}')
        if other > 0.01 * n:
            vals, counts = np.unique(d[(d != 0) & (d != 100) & (d != -1)], return_counts=True)
            top = sorted(zip(counts, vals), reverse=True)[:8]
            self.log.warn(
                '[dbg] map has cell values other than 0/100/-1 (e.g. '
                + ', '.join(f'{v}:{c}' for c, v in top)
                + '). Nav2 AMCL treats them as UNKNOWN: they are neither walls nor '
                'spawn space. Use "mode: trinary" in the map yaml.')
        if m.occ.sum() < 100:
            self.log.warn('[dbg] almost no occupied cells -> AMCL has nothing to match against')
        if abs(m.origin_yaw) > 1e-3:
            self.log.warn(f'[dbg] map origin yaw {m.origin_yaw:.3f} rad: Nav2 AMCL ignores it')
        if self.has_truth:
            free = m.is_free(np.array([self.truth[0]]), np.array([self.truth[1]]))[0]
            self.log.info(f'[dbg] true position is on a {"FREE" if free else "NON-free"} cell'
                          + ('' if free else ' -> AMCL never spawns particles there!'))
        # per-free-cell particle density right after global init
        mp = self.a('max_particles', 0) or 0
        if mp:
            area = m.free.sum() * m.res ** 2
            self.log.info(f'[dbg] {mp} particles over {area:.1f} m2 -> '
                          f'{mp / max(area, 1e-9):.0f} particles/m2 '
                          f'(~{mp / max(area, 1e-9) / 36:.1f} per m2 per 10 deg heading)')

    def _check_tf(self):
        if self.tf is None or self.scan is None:
            return
        base = self.a('base_frame_id', 'base_footprint')
        odom = self.a('odom_frame_id', 'odom')
        glob = self.a('global_frame_id', 'map')
        self.laser_frame = self.scan.header.frame_id
        self._wait_for('TF base->laser',
                       lambda: self.tf.can_transform(base, self.laser_frame, Time()), 5.0)
        try:
            t = self.tf.lookup_transform(base, self.laser_frame, Time())
            tr, q = t.transform.translation, t.transform.rotation
            self.laser = (tr.x, tr.y, q)
            fwd = rotate(q, np.array([1.0, 0.0, 0.0]))
            up = rotate(q, np.array([0.0, 0.0, 1.0]))
            self.log.info(
                f'[dbg] laser {self.laser_frame} in {base}: x={tr.x:.3f} y={tr.y:.3f} '
                f'z={tr.z:.3f}, yaw={math.degrees(math.atan2(fwd[1], fwd[0])):.1f} deg, '
                f'{"upright" if up[2] > 0 else "UPSIDE DOWN"}')
        except Exception as e:  # noqa: BLE001
            self.log.warn(f'[dbg] TF {base} <- {self.laser_frame} failed: {e}')
        for target, source in ((odom, base), (glob, odom)):
            try:
                t = self.tf.lookup_transform(target, source, Time())
                age = self.node.get_clock().now() - Time.from_msg(t.header.stamp)
                self.log.info(f'[dbg] TF {target} <- {source} ok '
                              f'(age {age.nanoseconds / 1e9:.2f} s)')
            except Exception as e:  # noqa: BLE001
                self.log.warn(f'[dbg] TF {target} <- {source} missing: {e}')

    # ---------- per stop ----------

    def cloud_array(self):
        if self.cloud is None or not self.cloud.particles:
            return None
        return np.array([[pt.pose.position.x, pt.pose.position.y,
                          yaw_of(pt.pose.orientation), pt.weight]
                         for pt in self.cloud.particles])

    def on_stop(self, step, label=''):
        """Call after each still-stand update."""
        now = time.time()
        msgs = (f'{self.cloud_msgs} cloud / {self.pose_msgs} pose msgs since last stop')
        self.cloud_msgs = self.pose_msgs = 0
        parts = [f'[dbg] step {step}{" " + label if label else ""}: {msgs}']

        if self.odom_yaw is not None:
            if self.prev_odom_yaw is not None:
                turned = math.degrees(angle_diff(self.odom_yaw, self.prev_odom_yaw))
                parts.append(f'odom turned {turned:+.0f} deg')
            self.prev_odom_yaw = self.odom_yaw
        if self.last_cloud_t is not None and now - self.last_cloud_t > 2.0:
            parts.append(f'LAST CLOUD {now - self.last_cloud_t:.1f} s AGO')
        self.log.info(', '.join(parts))

        pts = self.cloud_array()
        if pts is None:
            self.log.warn('[dbg]   no particle cloud yet')
            return
        x, y, th, w = pts.T
        sxy = math.sqrt(x.var() + y.var())
        R = math.hypot(np.cos(th).mean(), np.sin(th).mean())
        syaw = math.sqrt(-2.0 * math.log(max(R, 1e-12)))
        wn = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        ess = 1.0 / (wn ** 2).sum()
        line = (f'[dbg]   cloud {len(x)} pts, mean ({x.mean():.2f}, {y.mean():.2f}), '
                f'std xy {sxy:.2f} m yaw {math.degrees(syaw):.0f} deg, '
                f'ESS {ess:.0f} ({"uniform weights" if w.max() - w.min() < 1e-12 else "weighted"})')
        if self.map is not None:
            line += f', on free cells {100.0 * self.map.is_free(x, y).mean():.0f}%'
        self.log.info(line)

        # coarse clusters: 0.5 m bins
        keys = np.stack([np.floor(x / 0.5), np.floor(y / 0.5)], axis=1)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        mass = np.bincount(inv.ravel(), weights=wn, minlength=len(uniq))
        order = np.argsort(mass)[::-1][:3]
        self.log.info('[dbg]   top 0.5 m bins: ' + ', '.join(
            f'({(uniq[i][0] + 0.5) * 0.5:.2f}, {(uniq[i][1] + 0.5) * 0.5:.2f}) '
            f'{100.0 * mass[i]:.0f}%' for i in order) + f' | {len(uniq)} bins occupied')

        if self.has_truth:
            d = np.hypot(x - self.truth[0], y - self.truth[1])
            near = d < 0.5
            if not math.isnan(self.truth[2]):
                near &= np.abs(angle_diff(th, self.truth[2])) < math.radians(30)
            self.log.info(f'[dbg]   particles near truth (0.5 m{", 30 deg" if not math.isnan(self.truth[2]) else ""}): '
                          f'{near.sum()}, closest {d.min():.2f} m')
            if near.sum() == 0 and self.collapsed_at is None:
                self.log.warn('[dbg]   NO particle near the true pose -> AMCL cannot converge '
                              'there without new random particles')

        if self.pose is not None:
            pp = self.pose.pose.pose
            cov = self.pose.pose.covariance
            self.log.info(
                f'[dbg]   amcl_pose ({pp.position.x:.2f}, {pp.position.y:.2f}, '
                f'{math.degrees(yaw_of(pp.orientation)):.0f} deg), std x {math.sqrt(cov[0]):.2f} '
                f'y {math.sqrt(cov[7]):.2f} yaw {math.degrees(math.sqrt(cov[35])):.0f} deg')

        self.snapshots.append({'step': step, 'label': label, 't': now, 'particles': pts,
                               'pose': self._pose_tuple(), 'odom_yaw': self.odom_yaw,
                               'scan': None if self.scan is None else
                               np.array(self.scan.ranges, dtype=np.float32)})
        self._save_png(f'step_{step:03d}', f'step {step} {label}', pts)

        if step == 0:
            self.diagnose('initial (before turning)')
        elif self.collapsed_at is None and sxy < COLLAPSE_STD:
            self.collapsed_at = step
            self.log.warn(f'[dbg] cloud COLLAPSED at step {step} around '
                          f'({x.mean():.2f}, {y.mean():.2f}) -> checking that spot')
            self.diagnose(f'collapse at step {step}')

    def _pose_tuple(self):
        if self.pose is None:
            return None
        pp = self.pose.pose.pose
        return (pp.position.x, pp.position.y, yaw_of(pp.orientation))

    # ---------- independent scan-to-map check ----------

    def _beams(self, mirrored=False):
        """Selected beams like AMCL (max_beams, range limits) -> (ranges, base-frame angles)."""
        s = self.scan
        r = np.array(s.ranges, dtype=float)
        n = len(r)
        max_beams = int(self.a('max_beams', 60))
        step = max((n - 1) // max(max_beams - 1, 1), 1)
        idx = np.arange(0, n, step)
        ang = s.angle_min + idx * s.angle_increment
        if mirrored:
            ang = -ang
        r = r[idx]
        rmin, rmax = self._range_limits(s)
        keep = np.isfinite(r) & (r > rmin) & (r < rmax)
        r, ang = r[keep], ang[keep]
        # beam directions into the base frame (handles yawed / upside-down lasers)
        if self.laser is not None:
            v = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
            v = rotate(self.laser[2], v)
            ang = np.arctan2(v[:, 1], v[:, 0])
        return r, ang, rmax, len(idx)

    def score(self, xs, ys, yaws, beams):
        """AMCL likelihood-field weight for poses (arrays of equal length)."""
        r, ang, rmax, _ = beams
        z_hit = self.a('z_hit', 0.5)
        z_rand = self.a('z_rand', 0.5)
        sigma = self.a('sigma_hit', 0.2)
        lx, ly = (self.laser[0], self.laser[1]) if self.laser is not None else (0.0, 0.0)
        c, s = np.cos(yaws), np.sin(yaws)
        px = xs + c * lx - s * ly
        py = ys + s * lx + c * ly
        a = yaws[:, None] + ang[None, :]
        ex = px[:, None] + r[None, :] * np.cos(a)
        ey = py[:, None] + r[None, :] * np.sin(a)
        d = self.map.dist_at(ex, ey)
        pz = z_hit * np.exp(-d * d / (2 * sigma * sigma)) + z_rand / rmax
        return 1.0 + (pz ** 3).sum(axis=1)

    def _search(self, xs, ys, yaws, beams):
        """Best score (and yaw) for every position over the given yaw set."""
        best = np.full(len(xs), -np.inf)
        best_yaw = np.zeros(len(xs))
        for yaw in yaws:
            sc = self.score(xs, ys, np.full(len(xs), yaw), beams)
            better = sc > best
            best[better] = sc[better]
            best_yaw[better] = yaw
        return best, best_yaw

    def _global_peaks(self, beams, k=5, sep=0.5):
        m = self.map
        gy, gx = np.nonzero(m.free)
        stride = max(1, int(math.ceil(math.sqrt(len(gx) / 15000.0))))
        sel = (gx % stride == 0) & (gy % stride == 0)
        xs = m.ox + (gx[sel] + 0.5) * m.res
        ys = m.oy + (gy[sel] + 0.5) * m.res
        yaws = np.radians(np.arange(0, 360, 5))
        best, best_yaw = self._search(xs, ys, yaws, beams)
        order = np.argsort(best)[::-1]
        peaks = []
        for i in order:
            if all(math.hypot(xs[i] - p[0], ys[i] - p[1]) > sep for p in peaks):
                peaks.append((xs[i], ys[i], best_yaw[i], best[i]))
                if len(peaks) == k:
                    break
        return peaks, stride * m.res

    def diagnose(self, label):
        if self.map is None or self.scan is None:
            self.log.warn('[dbg] diagnose skipped: need map and scan')
            return
        t0 = time.time()
        beams = self._beams()
        r, _, rmax, n_sel = beams
        self.log.info(f'[dbg] === scan check: {label} === {len(r)}/{n_sel} selected beams '
                      f'usable (< {rmax:.2f} m)')
        if len(r) == 0:
            self.log.warn('[dbg]   no usable beams: every particle gets the same weight, '
                          'resampling just drifts the cloud randomly')
            return
        perfect = 1.0 + len(r) * (self.a('z_hit', 0.5) + self.a('z_rand', 0.5) / rmax) ** 3
        result = {'label': label, 'usable_beams': int(len(r)), 'perfect': perfect}

        est = self._pose_tuple()
        if est is not None:
            sc = self.score(np.array([est[0]]), np.array([est[1]]), np.array([est[2]]), beams)[0]
            result['amcl'] = (est, sc)
            self.log.info(f'[dbg]   AMCL estimate ({est[0]:.2f}, {est[1]:.2f}, '
                          f'{math.degrees(est[2]):.0f} deg): score {sc:.2f} '
                          f'(perfect match {perfect:.2f})')

        if self.has_truth:
            tx, ty, tyaw = self.truth
            off = np.arange(-0.25, 0.25 + 1e-9, 0.05)
            gx, gy = np.meshgrid(tx + off, ty + off)
            yaws = (np.radians(np.arange(0, 360, 2)) if math.isnan(tyaw)
                    else tyaw + np.radians(np.arange(-20, 21, 2)))
            best, best_yaw = self._search(gx.ravel(), gy.ravel(), yaws, beams)
            i = int(np.argmax(best))
            result['truth'] = ((gx.ravel()[i], gy.ravel()[i], best_yaw[i]), best[i])
            self.log.info(f'[dbg]   best near TRUTH ({gx.ravel()[i]:.2f}, {gy.ravel()[i]:.2f}, '
                          f'{math.degrees(best_yaw[i]):.0f} deg): score {best[i]:.2f}')

        if self.global_search:
            peaks, cell = self._global_peaks(beams)
            self.peaks = peaks
            result['peaks'] = peaks
            self.log.info(f'[dbg]   global scan match (grid {cell:.2f} m / 5 deg), top peaks:')
            for j, (px, py, pyaw, ps) in enumerate(peaks):
                note = ''
                if est is not None and math.hypot(px - est[0], py - est[1]) < 0.5:
                    note += ' <- AMCL is here'
                if self.has_truth and math.hypot(px - self.truth[0], py - self.truth[1]) < 0.5:
                    note += ' <- TRUTH'
                self.log.info(f'[dbg]     #{j + 1} ({px:.2f}, {py:.2f}, '
                              f'{math.degrees(pyaw):.0f} deg) score {ps:.2f}{note}')
            mpeaks, _ = self._global_peaks(self._beams(mirrored=True), k=1)
            result['mirrored_best'] = mpeaks[0][3] if mpeaks else None
            if mpeaks:
                self.log.info(f'[dbg]   mirrored scan best score {mpeaks[0][3]:.2f} at '
                              f'({mpeaks[0][0]:.2f}, {mpeaks[0][1]:.2f})')
                if peaks and mpeaks[0][3] > 1.1 * peaks[0][3]:
                    self.log.warn('[dbg]   MIRRORED scan fits the map clearly better -> '
                                  'scan direction / laser TF is probably flipped')
            if len(peaks) > 1 and peaks[1][3] > 0.95 * peaks[0][3]:
                self.log.warn('[dbg]   top peaks score nearly the same -> map is ambiguous '
                              'from this spot, rotating in place may never decide')

        self._interpret(result)
        self.diagnoses.append(result)
        self.log.info(f'[dbg]   (scan check took {time.time() - t0:.1f} s)')
        if self.cloud is not None:
            self._save_png(f'diag_{len(self.diagnoses):02d}', f'scan check: {label}',
                           self.cloud_array())

    def _interpret(self, res):
        hints = []
        amcl = res.get('amcl')
        truth = res.get('truth')
        peaks = res.get('peaks') or []
        if res['usable_beams'] < 10:
            hints.append('very few usable beams -> raise laser_max_range '
                         '(sensor max is ~3.5 m on LDS-01/02)')
        if truth and amcl and math.hypot(truth[0][0] - amcl[0][0], truth[0][1] - amcl[0][1]) > 0.5:
            if truth[1] > amcl[1]:
                hints.append('scan fits the TRUE pose better than AMCL\'s pose -> particle '
                             'depletion: AMCL threw away the right hypothesis (too few '
                             'particles near truth, too many resamples per stop)')
            else:
                hints.append('scan fits AMCL\'s WRONG pose better than the true pose -> the '
                             'map / scan / laser TF / laser_max_range make the wrong spot '
                             'look right; more particles will not fix that')
        if peaks and amcl and math.hypot(peaks[0][0] - amcl[0][0], peaks[0][1] - amcl[0][1]) > 0.5:
            if peaks[0][3] > amcl[1] * 1.05:
                hints.append(f'global best match ({peaks[0][0]:.2f}, {peaks[0][1]:.2f}) is not '
                             'where AMCL is and scores higher -> AMCL converged to a local optimum')
        for h in hints:
            self.log.warn(f'[dbg]   HINT: {h}')
        res['hints'] = hints

    # ---------- output ----------

    def _save_png(self, name, title, pts):
        if not self.png or self.map is None:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.log.warn('[dbg] matplotlib missing, PNG output disabled')
            self.png = False
            return
        m = self.map
        img = np.full(m.data.shape, 0.6)
        img[m.free] = 1.0
        img[m.occ] = 0.0
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', extent=m.extent(),
                  interpolation='nearest')
        if pts is not None:
            sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 4000),
                                                       replace=False)]
            ax.scatter(sub[:, 0], sub[:, 1], s=1, c='tab:orange', alpha=0.5, label='particles')
        for j, (px, py, pyaw, _) in enumerate(self.peaks):
            ax.plot(px, py, 'x', color='tab:blue', ms=10, mew=2)
            ax.annotate(f'#{j + 1}', (px, py), color='tab:blue', xytext=(4, 4),
                        textcoords='offset points')
        est = self._pose_tuple()
        if est is not None:
            ax.arrow(est[0], est[1], 0.3 * math.cos(est[2]), 0.3 * math.sin(est[2]),
                     width=0.04, color='red', label='amcl_pose')
        if self.has_truth:
            tx, ty, tyaw = self.truth
            ax.plot(tx, ty, 'o', color='lime', ms=10, mec='k', label='truth')
            if not math.isnan(tyaw):
                ax.arrow(tx, ty, 0.3 * math.cos(tyaw), 0.3 * math.sin(tyaw), width=0.04,
                         color='lime')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        fig.savefig(os.path.join(self.dir, name + '.png'), dpi=100)
        plt.close(fig)

    def finish(self, localized):
        self.diagnose('final (localized)' if localized else 'final (NOT localized)')
        data = {f'particles_{i:03d}': s['particles'] for i, s in enumerate(self.snapshots)}
        data.update({f'scan_{i:03d}': s['scan'] for i, s in enumerate(self.snapshots)
                     if s['scan'] is not None})
        if self.scan is not None:
            data['scan_angles'] = (self.scan.angle_min + np.arange(len(self.scan.ranges))
                                   * self.scan.angle_increment)
        if self.map is not None:
            data['map'] = self.map.data
        meta = {
            'amcl_params': self.amcl,
            'truth': self.truth,
            'map_origin': None if self.map is None else
            [self.map.ox, self.map.oy, self.map.res],
            'snapshots': [{k: v for k, v in s.items() if k not in ('particles', 'scan')}
                          for s in self.snapshots],
            'diagnoses': self.diagnoses,
            'collapsed_at_step': self.collapsed_at,
            'localized': localized,
        }
        data['meta'] = np.array(json.dumps(meta, default=float))
        path = os.path.join(self.dir, 'debug.npz')
        np.savez_compressed(path, **data)
        self.log.info(f'[dbg] saved {len(self.snapshots)} snapshots to {path}')
