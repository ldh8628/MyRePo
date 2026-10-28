"""command line: the same steps as the web page (python -m huskyroute.server), without the page.

    python -m huskyroute map   <bag folder> --name <map name> [--sensor-height 0.7] [--rate 1.0]
    python -m huskyroute route <map name> --start=x,y,deg --goal=x,y,deg [--planner hybrid|2d|navfn]
"""
import argparse
import json
import math
import os
import time
from pathlib import Path


def pose(s):
    x, y, *d = [float(v) for v in s.split(",")]
    return [x, y, math.radians(d[0] if d else 0.0)]


def main():
    ap = argparse.ArgumentParser(description="Husky route test on a hdl_graph_slam map")
    ap.add_argument("--maps", default="maps", help="folder the maps are kept in")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="LiDAR bag -> hdl_graph_slam -> 2D grid")
    m.add_argument("bag")
    m.add_argument("--name", required=True)
    m.add_argument("--topic", help="PointCloud2 topic (default: the first one in the bag)")
    m.add_argument("--sensor-height", type=float, help="LiDAR height above the floor [m] (default: estimated)")
    m.add_argument("--rate", type=float, default=1.0, help="bag play rate")
    m.add_argument("--kf-trans", type=float, default=0.5, help="keyframe spacing [m]")
    m.add_argument("--far", type=float, default=30.0, help="ignore points farther than this [m]")
    r = sub.add_parser("route", help="plan one route with Nav2 on a saved map")
    r.add_argument("name")
    r.add_argument("--start", required=True, type=pose, help="x,y,heading in degrees (write --start=-1,2,90 when it starts with a minus)")
    r.add_argument("--goal", required=True, type=pose, help="x,y,heading in degrees (write --goal=-1,2,90 when it starts with a minus)")
    r.add_argument("--planner", default="hybrid", choices=["hybrid", "2d", "navfn"])
    a = ap.parse_args()
    maps = Path(a.maps).resolve()

    if a.cmd == "map":
        from . import mapping
        res = mapping.build(str(Path(a.bag).resolve()), maps, a.name, topic=a.topic, rate=a.rate, kf_trans=a.kf_trans,
                            far=a.far, sensor_height=a.sensor_height,
                            log=lambda f, msg: print(f"[{f * 100:3.0f}%] {msg}", flush=True))
        print(json.dumps(res["grid"]))
        print("->", maps / a.name)
        return

    os.environ["ROS_DOMAIN_ID"] = os.environ.get("HUSKYROUTE_PLAN_DOMAIN", "62")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    from .planner import Planner
    p = Planner(maps / ".planner")
    try:
        p.start(maps / a.name / f"{a.name}.yaml", {"planner": a.planner})
        for _ in range(60):
            if p.status() in ("ready", "failed"):
                break
            time.sleep(0.5)
        if p.status() != "ready":
            raise SystemExit("planner did not start:\n" + p.log_tail())
        path, err, ms = p.plan(a.start, a.goal)
        if path is None:
            raise SystemExit(f"no path: {err}")
        length = sum(math.hypot(q[0] - o[0], q[1] - o[1]) for o, q in zip(path, path[1:]))
        print(json.dumps({"length": round(length, 2), "poses": len(path), "max_cost": p.clearance(path), "ms": ms,
                          "path": [[round(v, 3) for v in q] for q in path]}))
    finally:
        p.stop()


if __name__ == "__main__":
    main()
