"""server.py — local web page: LiDAR bag -> hdl_graph_slam map -> place the Husky, click a goal, see Nav2's path.

    python -m huskyroute.server [--port 8798] [--maps maps] [--root ~]      then open http://localhost:8798

Run it in a shell where ROS 2 Humble, Nav2 and the hdl_graph_slam workspace are sourced (README).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import signal
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml

# the planner's ROS domain, set before rclpy is imported (planner.py); hdl mapping runs in another (mapping.py)
os.environ["ROS_DOMAIN_ID"] = os.environ.get("HUSKYROUTE_PLAN_DOMAIN", "62")
os.environ["ROS_LOCALHOST_ONLY"] = "1"

from . import mapping                                   # noqa: E402
from .grid import png_rgba                              # noqa: E402
from .planner import DEFAULTS, HUSKY_HALF, Planner      # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
MAPS, ROOT = Path("maps"), Path.home()
JOBS: dict[str, dict] = {}
PLANNER: Planner | None = None
CURRENT = {"name": None, "opts": dict(DEFAULTS)}
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def map_list():
    out = []
    for d in sorted(MAPS.glob("*/info.json")):
        try:
            m = json.load(open(d))
            out.append({"name": m["name"], "made": m.get("made"), "bag": m.get("bag"),
                        "track_length": m["grid"].get("track_length"), "keyframes": m["grid"].get("keyframes")})
        except Exception:  # noqa: BLE001 - a half-written map folder
            pass
    return out


def map_data(name):
    """grid as an RGBA PNG for the page, with its placement, the driven path and the build settings"""
    d = MAPS / name
    meta = yaml.safe_load(open(d / f"{name}.yaml"))
    raw = open(d / meta["image"], "rb").read()
    hdr = raw.split(b"\n", 3)
    w, h = map(int, hdr[1].split())
    g = np.frombuffer(hdr[3], np.uint8)[: w * h].reshape(h, w)
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 255
    img[g > 250] = (236, 239, 243, 255)
    img[g < 100] = (40, 44, 52, 255)
    img[(g >= 100) & (g <= 250)] = (176, 182, 190, 255)
    info = json.load(open(d / "info.json"))
    return {"name": name, "res": float(meta["resolution"]), "ox": float(meta["origin"][0]), "oy": float(meta["origin"][1]),
            "w": w, "h": h, "png": base64.b64encode(png_rgba(img)).decode(),
            "track": json.load(open(d / f"{name}_track.json")), "info": info}


def run_build(jid, args):
    job = JOBS[jid]

    def log(frac, msg):
        job["progress"] = frac
        if job["log"] and job["log"][-1].startswith("hdl_graph_slam:") and msg.startswith("hdl_graph_slam:"):
            job["log"][-1] = msg                          # one progress line, not one per second
        else:
            job["log"].append(msg)
    try:
        job["result"] = mapping.build(log=log, cancel=job["cancel"], maps_dir=MAPS, **args)
        job["state"] = "done"
    except Exception as e:  # noqa: BLE001 (shown on the page)
        job["state"], job["error"] = "error", str(e)
        job["log"].append(traceback.format_exc(limit=3))


def route(d):
    s = [float(d["sx"]), float(d["sy"]), float(d["syaw"])]
    g = [float(d["gx"]), float(d["gy"]), float(d["gyaw"])]
    straight = math.hypot(g[0] - s[0], g[1] - s[1])
    path, err, ms = PLANNER.plan(s, g)
    if path is None:
        return {"ok": False, "why": err, "ms": ms, "straight": round(straight, 2)}
    length = float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))
    rev = 0.0                                             # length driven backwards (Hybrid-A* with reversing)
    for a, b in zip(path, path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx * math.cos(a[2]) + dy * math.sin(a[2]) < -1e-3:
            rev += math.hypot(dx, dy)
    return {"ok": True, "path": [[round(x, 3), round(y, 3), round(t, 3)] for x, y, t in path], "length": round(length, 2),
            "straight": round(straight, 2), "reverse": round(rev, 2), "max_cost": PLANNER.clearance(path), "ms": ms}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/browse":
                p = Path(q.get("path") or ROOT).expanduser()
                if not p.is_dir():
                    return self._send(404, {"error": f"not a folder: {p}"})
                dirs = sorted(x.name for x in p.iterdir() if x.is_dir() and not x.name.startswith("."))
                return self._send(200, {"path": str(p), "parent": str(p.parent), "dirs": dirs,
                                        "is_bag": (p / "metadata.yaml").is_file()})
            if u.path == "/api/bag":
                return self._send(200, mapping.bag_info(q["path"]))
            if u.path == "/api/maps":
                return self._send(200, {"maps": map_list(), "current": CURRENT})
            if u.path == "/api/map":
                if not NAME_RE.match(q.get("name", "")) or not (MAPS / q["name"] / "info.json").is_file():
                    return self._send(404, {"error": "no such map"})
                return self._send(200, map_data(q["name"]))
            if u.path == "/api/planner":
                return self._send(200, {"status": PLANNER.status(), "map": CURRENT["name"], "opts": CURRENT["opts"],
                                        "log": PLANNER.log_tail() if PLANNER.status() == "failed" else ""})
            if u.path == "/api/cost":
                with PLANNER.lock:
                    return self._send(200, PLANNER.cost or {})
            if u.path == "/api/job":
                job = JOBS.get(q.get("id", ""))
                if not job:
                    return self._send(404, {"error": "no such job"})
                return self._send(200, {k: job[k] for k in ("state", "progress", "log", "error", "result", "name")})
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/api/build":
                if any(j["state"] == "running" for j in JOBS.values()):
                    return self._send(409, {"error": "a map is already being made"})
                name = b.get("name", "")
                if not NAME_RE.match(name):
                    return self._send(400, {"error": "map name: letters, digits, _ and - only"})
                if (MAPS / name / "info.json").exists() and not b.get("overwrite"):
                    return self._send(409, {"error": f"map {name} already exists"})
                bag = str(Path(b.get("bag", "")).expanduser().resolve())
                mapping.bag_info(bag)                       # raises with a readable message if it is not a bag
                sh = b.get("sensor_height")
                args = {"bag": bag, "name": name, "topic": b.get("topic") or None, "rate": float(b.get("rate", 1.0)),
                        "kf_trans": float(b.get("kf_trans", 0.5)), "kf_angle": float(b.get("kf_angle", 0.3)),
                        "far": float(b.get("far", 30.0)), "sensor_height": float(sh) if sh not in (None, "") else None}
                jid = uuid.uuid4().hex[:8]
                JOBS[jid] = {"state": "running", "progress": 0.0, "log": [], "error": None, "result": None,
                             "name": name, "cancel": threading.Event()}
                threading.Thread(target=run_build, args=(jid, args), daemon=True).start()
                return self._send(200, {"id": jid})
            if u.path == "/api/cancel":
                job = JOBS.get(b.get("id", ""))
                if job:
                    job["cancel"].set()
                return self._send(200, {})
            if u.path == "/api/regrid":
                name = b.get("name", "")
                if not NAME_RE.match(name):
                    return self._send(400, {"error": "bad name"})
                sh = b.get("sensor_height")
                opts = {k: float(b[k]) for k in ("obst_min", "obst_max") if b.get(k) not in (None, "")}
                mapping.regrid(MAPS, name, sensor_height=float(sh) if sh not in (None, "") else None, grid_opts=opts,
                               log=lambda m: None)
                if CURRENT["name"] == name:
                    PLANNER.key = None                      # same file names: force the planner to reload the grid
                    PLANNER.start(MAPS / name / f"{name}.yaml", CURRENT["opts"])
                return self._send(200, map_data(name))
            if u.path == "/api/delete":
                name = b.get("name", "")
                if NAME_RE.match(name) and (MAPS / name / "info.json").is_file() and CURRENT["name"] != name:
                    shutil.rmtree(MAPS / name)
                return self._send(200, {"maps": map_list()})
            if u.path == "/api/use":
                name = b.get("name", "")
                if not NAME_RE.match(name) or not (MAPS / name / "info.json").is_file():
                    return self._send(404, {"error": "no such map"})
                opts = {**CURRENT["opts"], **{k: b["opts"][k] for k in DEFAULTS if k in (b.get("opts") or {})}}
                CURRENT["opts"] = PLANNER.start(MAPS / name / f"{name}.yaml", opts)
                CURRENT["name"] = name
                return self._send(200, {"status": PLANNER.status(), "opts": CURRENT["opts"]})
            if u.path == "/api/route":
                if PLANNER.status() != "ready":
                    return self._send(200, {"ok": False, "why": "계획기가 아직 준비되지 않았습니다 (잠시 뒤 다시)"})
                return self._send(200, route(b))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(HTTPStatus.BAD_REQUEST, {"error": f"{type(e).__name__}: {e}"})


def main():
    global MAPS, ROOT, PLANNER
    ap = argparse.ArgumentParser(description="Husky route test on a hdl_graph_slam map (web page)")
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--maps", default="maps", help="folder the maps are kept in")
    ap.add_argument("--root", default=str(Path.home()), help="folder the bag browser starts in")
    a = ap.parse_args()
    MAPS, ROOT = Path(a.maps).resolve(), Path(a.root).expanduser()
    MAPS.mkdir(parents=True, exist_ok=True)
    for tool in ("ros2",):
        if not shutil.which(tool):
            raise SystemExit("ros2 not found: source ROS 2 Humble (and the hdl_graph_slam workspace) first")
    PLANNER = Planner(MAPS / ".planner")

    def on_term(*_):
        raise KeyboardInterrupt                         # so the planner processes are stopped below
    signal.signal(signal.SIGTERM, on_term)
    print(f"huskyroute: http://{a.host}:{a.port}  maps in {MAPS}  (Ctrl+C to stop)", flush=True)
    try:
        ThreadingHTTPServer((a.host, a.port), H).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        PLANNER.stop()
        for j in JOBS.values():
            j["cancel"].set()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
