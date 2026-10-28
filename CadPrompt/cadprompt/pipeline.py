"""pipeline.py — CAD model in, ranked prompts out.

render views -> candidates (name, colours, shape, CLIP nouns) -> test images (renders on backgrounds) ->
score every candidate with YOLO-World and/or YOLOE (text), and YOLOE's visual prompt from a render mosaic.
"""
from __future__ import annotations

import base64
import io
import json
import time

import numpy as np
from PIL import Image

from . import candidates as cand
from .compose import Backgrounds, background_only, composite
from .detect import Detector, evaluate, mosaic
from .render import Model, render

DEFAULT_WEIGHTS = {"yolo-world": "yolov8s-worldv2.pt", "yoloe": "yoloe-11s-seg.pt"}


def _b64(img: Image.Image, fmt="JPEG", q=85):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=q)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _sheet(views, n=12, cell=128):
    pick = np.linspace(0, len(views) - 1, min(n, len(views))).astype(int)
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for j, i in enumerate(pick):
        rgb, m, _ = views[i]
        im = rgb.copy()
        im[~m] = 255
        sheet.paste(Image.fromarray(im).resize((cell, cell)), ((j % cols) * cell, (j // cols) * cell))
    return sheet


def run(cad_path, name=None, detectors=("yolo-world", "yoloe"), weights=None, n_views=24, n_tests=48, n_backs=24,
        thr=0.1, backgrounds=None, extra_prompts=(), vocab_file=None, log=print):
    t0 = time.time()
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    log("load", 0.02, f"loading {cad_path}")
    model = Model(cad_path)
    log("render", 0.05, f"{len(model.F)} triangles, {'coloured' if model.coloured else 'NO colour (grey renders)'}; "
                        f"rendering {n_views} views")
    views = render(model, n_views=n_views)
    log("candidates", 0.20, "prompt candidates (name, colour, shape, CLIP nouns)")
    cands, evidence = cand.build(model, views, name=name, vocab_file=vocab_file)
    for p in extra_prompts:
        if p.strip() and p.strip() not in {c["prompt"] for c in cands}:
            cands.append({"prompt": p.strip(), "source": "user"})
    bg = Backgrounds(backgrounds, seed=0)
    tests = composite(views, n_images=n_tests, bg=bg, seed=0)
    backs = background_only(n_backs, bg=Backgrounds(backgrounds, seed=1), seed=1)
    log("score", 0.25, f"{len(cands)} candidates x {len(detectors)} detector(s) on {len(tests)} test + {len(backs)} empty images")

    rows, total, done = [], len(cands) * len(detectors) + (2 if "yoloe" in detectors else 0), 0
    for det in detectors:
        d = Detector(det, weights[det])
        for c in cands:
            r = evaluate(d.text_model(c["prompt"]), tests, backs, thr=thr)
            rows.append({**c, "detector": det, "mode": "text", **r})
            done += 1
            log("score", 0.25 + 0.7 * done / total, f"{det:10s} {r['score']:+.2f} hit {r['hit_rate']:.2f} "
                                                      f"false {r['false_alarm_rate']:.2f}  '{c['prompt']}'")
    ref, boxes = mosaic(views)
    if "yoloe" in detectors:
        d = Detector("yoloe", weights["yoloe"])
        best_text = max((r for r in rows if r["detector"] == "yoloe"), key=lambda r: (r["score"], r["mean_conf"]))["prompt"]
        for text, label in ((None, "visual (render mosaic)"), (best_text, f"visual + text '{best_text}'")):
            r = evaluate(d.visual_model(ref, boxes, text=text), tests, backs, thr=thr)
            rows.append({"prompt": label, "source": "visual prompt", "detector": "yoloe",
                         "mode": "visual" if text is None else "visual+text", **r})
            done += 1
            log("score", 0.25 + 0.7 * done / total, f"yoloe      {r['score']:+.2f} hit {r['hit_rate']:.2f} "
                                                      f"false {r['false_alarm_rate']:.2f}  {label}")
    rows.sort(key=lambda r: (r["score"], r["mean_conf"]), reverse=True)
    best = {}
    for det in detectors:
        b = [r for r in rows if r["detector"] == det and r["mode"] == "text" and r["score"] > 0]
        best[det] = b[0]["prompt"] if b else None      # None: no candidate found the object on the test images
    res = {
        "cad": str(cad_path), "name": name or model.name, "triangles": int(len(model.F)), "coloured": model.coloured,
        "settings": {"views": n_views, "test_images": len(tests), "empty_images": len(backs), "conf_threshold": thr,
                     "backgrounds": backgrounds or "generated", "weights": {k: weights[k] for k in detectors}},
        "evidence": evidence, "recommended": best, "ranking": rows,
        "visual_prompt": {"boxes": boxes, "note": "reference image = render mosaic; use with YOLOE "
                                                  "predict(..., visual_prompts={'bboxes': boxes, 'cls': [0]*n}, refer_image=mosaic)"},
        "seconds": round(time.time() - t0, 1),
        "images": {"views": _b64(_sheet(views)), "mosaic": _b64(ref), "examples": [_b64(t[0].resize((256, 256))) for t in tests[:6]]},
        "_mosaic_png": ref,
    }
    log("done", 1.0, f"done in {res['seconds']} s — best: " + ", ".join(f"{k}: '{v}'" if v else f"{k}: none" for k, v in best.items()))
    return res


def save(res, out_dir):
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["_mosaic_png"].save(out / "visual_prompt_mosaic.png")
    with open(out / "prompts.json", "w") as f:
        json.dump({k: v for k, v in res.items() if k not in ("images", "_mosaic_png")}, f, indent=1)
    return out
