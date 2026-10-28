"""server.py — local web page: give a CAD model (upload or a path), get ranked YOLO-World / YOLOE prompts.

    python -m cadprompt.server [--port 8797]      then open http://localhost:8797
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import pipeline

WEB = Path(__file__).resolve().parent / "web"
UPLOADS = Path(tempfile.gettempdir()) / "cadprompt_uploads"
OUT = Path("results")
JOBS: dict[str, dict] = {}
CAD_EXT = (".obj", ".ply", ".glb", ".gltf", ".stl", ".off")


def run_job(jid, args):
    job = JOBS[jid]

    def log(stage, frac, msg):
        job["progress"], job["stage"] = frac, stage
        job["log"].append(msg)
    try:
        res = pipeline.run(log=log, **args)
        res["saved_to"] = str(pipeline.save(res, OUT / f"{Path(args['cad_path']).stem}_{jid}"))
        res.pop("_mosaic_png", None)
        job["result"], job["state"] = res, "done"
    except Exception as e:  # noqa: BLE001  (shown on the page)
        job["state"], job["error"] = "error", str(e)
        job["log"].append(traceback.format_exc(limit=4))


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
        if u.path in ("/", "/index.html"):
            return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/job":
            job = JOBS.get(q.get("id", ""))
            if not job:
                return self._send(404, {"error": "no such job"})
            return self._send(200, {k: job[k] for k in ("state", "progress", "log", "error", "result")})
        if u.path == "/api/download":
            job = JOBS.get(q.get("id", ""))
            if not job or not job.get("result"):
                return self._send(404, {"error": "no result"})
            body = json.dumps({k: v for k, v in job["result"].items() if k != "images"}, indent=1).encode()
            return self._send(200, body, extra={"Content-Disposition": 'attachment; filename="prompts.json"'})
        self._send(404, {"error": "not found"})

    def do_PUT(self):
        """one uploaded file: /api/upload?box=<id>&name=<file name>, body = the file"""
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path != "/api/upload":
            return self._send(404, {"error": "not found"})
        box = re.sub(r"[^a-z0-9]", "", q.get("box", ""))[:16] or uuid.uuid4().hex[:8]
        name = Path(q.get("name", "file")).name                     # no directories from the client
        d = UPLOADS / box
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        cads = sorted(str(p) for p in d.iterdir() if p.suffix.lower() in CAD_EXT)
        self._send(200, {"box": box, "files": sorted(p.name for p in d.iterdir()), "cad": cads})

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self._send(404, {"error": "not found"})
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        cad = str(Path(b["cad_path"]).resolve()) if b.get("cad_path") else ""   # detectors chdir to weights/
        if not cad or not Path(cad).is_file():
            return self._send(400, {"error": f"CAD file not found: {cad}"})
        dets = [d for d in b.get("detectors", ["yolo-world", "yoloe"]) if d in ("yolo-world", "yoloe")]
        if not dets:
            return self._send(400, {"error": "choose at least one detector"})
        args = {"cad_path": cad, "name": b.get("name") or None, "detectors": dets,
                "weights": {k: v for k, v in (b.get("weights") or {}).items() if v},
                "n_views": int(b.get("n_views", 24)), "n_tests": int(b.get("n_tests", 48)),
                "thr": float(b.get("thr", 0.1)),
                "backgrounds": str(Path(b["backgrounds"]).resolve()) if b.get("backgrounds") else None,
                "extra_prompts": [p for p in (b.get("extra_prompts") or "").split(",") if p.strip()]}
        if args["backgrounds"] and not Path(args["backgrounds"]).is_dir():
            return self._send(400, {"error": f"background folder not found: {args['backgrounds']}"})
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"state": "running", "progress": 0.0, "stage": "start", "log": [], "error": None, "result": None}
        threading.Thread(target=run_job, args=(jid, args), daemon=True).start()
        self._send(200, {"id": jid})


def main():
    global OUT
    ap = argparse.ArgumentParser(description="CAD -> YOLO-World / YOLOE prompt recommender (web page)")
    ap.add_argument("--port", type=int, default=8797)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="results", help="folder for prompts.json and the visual-prompt mosaic")
    a = ap.parse_args()
    OUT = Path(a.out).resolve()
    print(f"cadprompt: http://{a.host}:{a.port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
