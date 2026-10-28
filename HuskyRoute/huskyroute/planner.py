"""planner.py — the real Nav2 global planner on a saved grid, without a robot.

Planner.start() writes a Nav2 params file (footprint, inflation, planner plugin) and launches map_server +
planner_server in this process's ROS domain; plan() sends ComputePathToPose with an explicit start pose, so no robot or
localization is needed. The global costmap is the saved map + inflation only: live obstacles are not simulated.

Must be imported after ROS_DOMAIN_ID is set (server.py does that), because rclpy reads it at init.
"""
from __future__ import annotations

import base64
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from .grid import png_rgba

LAUNCH = Path(__file__).resolve().parent / "launch" / "planner.launch.py"

# Husky A200 body 0.99 x 0.67 m; the footprint adds a small margin (half length, half width)
HUSKY_HALF = (0.495, 0.335)
DEFAULTS = {"planner": "hybrid", "half_length": 0.51, "half_width": 0.355, "inflation_radius": 0.6,
            "cost_scaling": 3.0, "reverse": True, "min_turn_radius": 0.30, "prefer_driven": True, "allow_unknown": False}


def nav2_params(o, resolution):
    """planner_server + global_costmap parameters (Nav2 Humble names)"""
    hl, hw = float(o["half_length"]), float(o["half_width"])
    common = {"tolerance": 0.25, "downsample_costmap": False, "allow_unknown": bool(o["allow_unknown"]),
              "max_iterations": 1000000, "max_on_approach_iterations": 1000, "max_planning_time": 5.0}
    if o["planner"] == "hybrid":                    # heading-aware, the footprint is checked along the path
        gb = {"plugin": "nav2_smac_planner/SmacPlannerHybrid", **common,
              "motion_model_for_search": "REEDS_SHEPP" if o["reverse"] else "DUBIN",
              "angle_quantization_bins": 72, "analytic_expansion_ratio": 3.5, "analytic_expansion_max_length": 3.0,
              "minimum_turning_radius": float(o["min_turn_radius"]), "reverse_penalty": 2.0, "change_penalty": 0.0,
              "non_straight_penalty": 1.2, "cost_penalty": 2.0, "retrospective_penalty": 0.015,
              "lookup_table_size": 20.0, "cache_obstacle_heuristic": False, "smooth_path": True}
    elif o["planner"] == "2d":                      # grid A*, the robot as a circle (inflation) only
        gb = {"plugin": "nav2_smac_planner/SmacPlanner2D", **common, "cost_travel_multiplier": 2.0,
              "use_final_approach_orientation": False}
    else:                                           # NavFn (Dijkstra), the Nav2 default
        gb = {"plugin": "nav2_navfn_planner/NavfnPlanner", "tolerance": 0.25, "use_astar": False,
              "allow_unknown": bool(o["allow_unknown"])}
    fp = f"[[{hl}, {hw}], [{hl}, {-hw}], [{-hl}, {-hw}], [{-hl}, {hw}]]"
    return {
        "planner_server": {"ros__parameters": {"use_sim_time": False, "expected_planner_frequency": 1.0,
                                               "planner_plugins": ["GridBased"], "GridBased": gb}},
        "global_costmap": {"global_costmap": {"ros__parameters": {
            "use_sim_time": False, "global_frame": "map", "robot_base_frame": "base_link",
            "update_frequency": 1.0, "publish_frequency": 1.0, "rolling_window": False, "resolution": resolution,
            "track_unknown_space": True, "always_send_full_costmap": True, "footprint": fp,
            "trinary_costmap": False,                # keep the "prefer the driven path" costs of the _pref map
            "plugins": ["static_layer", "inflation_layer"],
            "static_layer": {"plugin": "nav2_costmap_2d::StaticLayer", "map_subscribe_transient_local": True,
                             "trinary_costmap": False},
            "inflation_layer": {"plugin": "nav2_costmap_2d::InflationLayer",
                                "inflation_radius": float(o["inflation_radius"]),
                                "cost_scaling_factor": float(o["cost_scaling"])}}}},
    }


class Planner:
    def __init__(self, work_dir):
        import rclpy
        from nav2_msgs.action import ComputePathToPose
        from nav_msgs.msg import OccupancyGrid
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.proc, self.key, self.lock = None, None, threading.Lock()
        self.cost, self.cost_raw, self.cost_at = None, None, 0.0
        self.Goal = ComputePathToPose.Goal
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("huskyroute_planner_client")
        self.ac = ActionClient(self.node, ComputePathToPose, "/compute_path_to_pose")
        q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.node.create_subscription(OccupancyGrid, "/global_costmap/costmap", self._on_cost, q)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    # ── planner process ──
    def start(self, map_yaml, opts):
        """(re)launch map_server + planner_server for this grid and these options; no-op if unchanged"""
        o = {**DEFAULTS, **{k: v for k, v in (opts or {}).items() if k in DEFAULTS}}
        grid = str(map_yaml)
        if o["prefer_driven"] and Path(grid.replace(".yaml", "_pref.yaml")).exists():
            grid = grid.replace(".yaml", "_pref.yaml")
        key = (grid, tuple(sorted(o.items())))
        with self.lock:
            if key == self.key and self.proc and self.proc.poll() is None:
                return o
            self.stop()
            res = float(yaml.safe_load(open(map_yaml))["resolution"])
            params = self.work / "nav2_params.yaml"
            yaml.safe_dump(nav2_params(o, res), open(params, "w"), sort_keys=False)
            self.cost = self.cost_raw = None
            self.proc = subprocess.Popen(["ros2", "launch", str(LAUNCH), f"params:={params}", f"map:={grid}"],
                                         stdout=open(self.work / "planner.log", "w"), stderr=subprocess.STDOUT,
                                         start_new_session=True)
            self.key = key
        return o

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc, self.key = None, None

    def status(self):
        if not self.proc:
            return "stopped"
        if self.proc.poll() is not None:
            return "failed"
        return "ready" if self.ac.server_is_ready() and self.cost_raw is not None else "starting"

    def log_tail(self, n=15):
        p = self.work / "planner.log"
        return "".join(open(p, errors="replace").readlines()[-n:]) if p.exists() else ""

    # ── costmap (for the picture and for explaining a failure) ──
    def _on_cost(self, m):
        c = np.array(m.data, np.int16).reshape(m.info.height, m.info.width)
        o = np.zeros(c.shape + (4,), np.uint8)
        f = (c > 0) & (c < 99)
        # faint below 35 (the "prefer the driven path" extra cost tops out at 30), clear above (obstacle clearance)
        o[f, 0], o[f, 1], o[f, 2] = 220, 60, 60
        o[f, 3] = np.where(c[f] <= 35, c[f] * 1.0, 70 + c[f] * 1.5).astype(np.uint8)
        o[c >= 99] = (150, 20, 20, 220)
        with self.lock:
            self.cost_raw = {"c": c, "res": m.info.resolution, "ox": m.info.origin.position.x, "oy": m.info.origin.position.y}
            self.cost = {"png": base64.b64encode(png_rgba(o[::-1])).decode(), "res": m.info.resolution,
                         "w": m.info.width, "h": m.info.height, "ox": m.info.origin.position.x,
                         "oy": m.info.origin.position.y}
            self.cost_at = time.time()

    def cell_cost(self, x, y):
        """costmap value under (x, y): -1 unknown, 0 free .. 253 inscribed, 254 lethal; None outside the map"""
        r = self.cost_raw
        if r is None:
            return None
        i, j = int((y - r["oy"]) / r["res"]), int((x - r["ox"]) / r["res"])
        if not (0 <= i < r["c"].shape[0] and 0 <= j < r["c"].shape[1]):
            return None
        v = int(r["c"][i, j])
        return -1 if v < 0 else round(v * 2.54)       # published 0..100 -> 0..254

    def explain(self, x, y, what):
        """why a start/goal is bad, or '' if it looks fine (what = subject with its particle, e.g. '목표가')"""
        v = self.cell_cost(x, y)
        if v is None:
            return f"{what} 지도 밖입니다"
        if v == -1:
            return f"{what} 지도에서 본 적 없는 곳(회색)입니다"
        if v >= 253:
            return f"{what} 장애물 위이거나 차체가 장애물에 닿는 곳입니다"
        return ""

    # ── planning ──
    def plan(self, s, g, timeout=15.0):
        """ComputePathToPose from s to g ([x, y, yaw]) -> (path [[x, y, yaw], ...] or None, error text, ms)"""
        if not self.ac.wait_for_server(timeout_sec=1.0):
            return None, "계획기가 아직 준비되지 않았습니다 (몇 초 뒤 다시 시도)", 0
        goal = self.Goal()
        goal.use_start, goal.planner_id = True, "GridBased"
        for ps, (x, y, yaw) in ((goal.start, s), (goal.goal, g)):
            ps.header.frame_id = "map"
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        ev, box, t0 = threading.Event(), {}, time.time()

        def got_goal(f):
            h = f.result()
            if not h.accepted:
                box["err"] = "계획기가 요청을 거절했습니다"
                ev.set()
                return
            h.get_result_async().add_done_callback(lambda r: (box.update(r=r.result()), ev.set()))
        self.ac.send_goal_async(goal).add_done_callback(got_goal)
        if not ev.wait(timeout):
            return None, f"계획기 응답 없음 ({timeout:.0f} s)", round(1e3 * (time.time() - t0))
        ms = round(1e3 * (time.time() - t0))
        if "err" in box:
            return None, box["err"], ms
        P = box["r"].result.path.poses
        if len(P) < 2:
            why = self.explain(*s[:2], "출발점이") or self.explain(*g[:2], "목표가")
            return None, why or "갈 수 있는 경로를 찾지 못했습니다 (막혀 있거나 차체가 지나갈 틈이 없음)", ms
        out = []
        for q in P:
            o = q.pose.orientation
            out.append([q.pose.position.x, q.pose.position.y, 2 * math.atan2(o.z, o.w)])
        return out, "", ms

    def clearance(self, path):
        """smallest costmap value along the path (0..254) — how close it comes to obstacles"""
        vals = [self.cell_cost(x, y) for x, y, _ in path]
        vals = [v for v in vals if v is not None and v >= 0]
        return max(vals) if vals else None
