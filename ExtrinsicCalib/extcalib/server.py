"""server.py — local web page for the calibration: pick two bags, press run, read the result.

    python -m extcalib.server [--port 8795] [--root /data]      then open http://localhost:8795

The page (web/index.html) only talks to this server; all computation runs here, one job at a time per request.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import pipeline
from .bagio import topics

WEB = Path(__file__).resolve().parent / "web"
JOBS: dict[str, dict] = {}
ROOT = Path("/")
OUT = Path("results")


def guess(tp: dict) -> dict:
    """sensor type and default topics from a bag's topic list"""
    by = lambda typ: sorted(t for t, (m, _) in tp.items() if m == typ)
    clouds, images, infos = by("sensor_msgs/msg/PointCloud2"), by("sensor_msgs/msg/Image"), by("sensor_msgs/msg/CameraInfo")
    depth = [t for t in images if "depth" in t]
    color = [t for t in images if "depth" not in t and ("color" in t or "rgb" in t or "image" in t)]
    cinfo = [t for t in infos if "depth" not in t] or infos
    return {"type": "rgbd" if depth and color else ("lidar" if clouds else None),
            "points": clouds, "color": color, "depth": depth, "info": cinfo,
            "counts": {t: c for t, (_, c) in tp.items()}}


def run_job(jid, a, b):
    job = JOBS[jid]

    def log(stage, frac, msg):
        job["progress"] = frac
        job["stage"] = stage
        job["log"].append(msg)
    try:
        res = pipeline.run(a, b, log=log)
        out = OUT / f"extrinsic_{Path(a['path']).name}_{Path(b['path']).name}_{jid}.json"
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            pipeline.save(res, out)
            res["saved_to"] = str(out)
        except OSError:
            res["saved_to"] = None
        job["result"], job["state"] = res, "done"
    except Exception as e:  # noqa: BLE001  (shown on the page)
        job["state"], job["error"] = "error", f"{e}"
        job["log"].append(traceback.format_exc(limit=3))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
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
                dirs = sorted(d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith("."))
                is_bag = any(p.glob("metadata.yaml")) or any(p.glob("*.db3")) or any(p.glob("*.mcap"))
                return self._send(200, {"path": str(p), "parent": str(p.parent), "dirs": dirs, "is_bag": is_bag})
            if u.path == "/api/topics":
                return self._send(200, guess(topics(q["path"])))
            if u.path == "/api/job":
                job = JOBS.get(q.get("id", ""))
                if not job:
                    return self._send(404, {"error": "no such job"})
                return self._send(200, {k: job[k] for k in ("state", "progress", "stage", "log", "error", "result")})
            if u.path == "/api/download":
                job = JOBS.get(q.get("id", ""))
                if not job or not job.get("result"):
                    return self._send(404, {"error": "no result"})
                body = json.dumps({k: v for k, v in job["result"].items() if k != "figures"}, indent=1).encode()
                return self._send(200, body, extra={"Content-Disposition": 'attachment; filename="extrinsic.json"'})
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/run":
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        a, b = body.get("a"), body.get("b")
        for s in (a, b):
            if not s or not s.get("path") or not Path(s["path"]).exists():
                return self._send(400, {"error": f"bag not found: {s and s.get('path')}"})
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"state": "running", "progress": 0.0, "stage": "start", "log": [], "error": None, "result": None}
        threading.Thread(target=run_job, args=(jid, a, b), daemon=True).start()
        self._send(200, {"id": jid})


def main():
    global ROOT, OUT
    ap = argparse.ArgumentParser(description="extrinsic calibration web page")
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--root", default=os.path.expanduser("~"), help="folder the file browser starts in")
    ap.add_argument("--out", default="results", help="folder the result JSON files are written to")
    a = ap.parse_args()
    ROOT, OUT = Path(a.root), Path(a.out).resolve()
    print(f"extcalib: http://{a.host}:{a.port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
