"""detect.py — score each prompt candidate with the detector it is meant for.

Every candidate is run alone (one class), so candidates never suppress each other, on
  * test images: renders pasted on backgrounds, box known  -> hit = a box with IoU >= 0.5 and conf >= thr
  * background-only images                                  -> false alarm = any box with conf >= thr
score = hit rate - 0.5 x false-alarm rate (mean matched confidence breaks ties).

YOLOE can also take a visual prompt: a reference image with boxes. Here the reference is a 3x3 mosaic of the
renders, so the prompt needs no photo of the real object.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
from PIL import Image

# detector weights and the text encoders ultralytics fetches on first use land here, not in the working folder
WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _boxes(model, img, conf):
    r = model.predict(np.asarray(img)[..., ::-1], conf=conf, imgsz=640, verbose=False, agnostic_nms=False)[0]
    return list(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()))


def evaluate(model, tests, backs, thr=0.1, floor=0.01):
    hits, confs, fa = 0, [], 0
    for img, box in tests:
        best = max((c for b, c in _boxes(model, img, floor) if iou(b, box) >= 0.5), default=0.0)
        confs.append(best)
        hits += best >= thr
    for img in backs:
        fa += any(c >= thr for _, c in _boxes(model, img, thr))
    hr, far = hits / max(len(tests), 1), fa / max(len(backs), 1)
    return {"hit_rate": round(hr, 3), "false_alarm_rate": round(far, 3), "mean_conf": round(float(np.mean(confs)), 3),
            "score": round(hr - 0.5 * far, 3)}


class Detector:
    def __init__(self, kind, weights, device=None):
        import torch
        self.kind = kind
        self.weights = str(Path(weights).resolve()) if Path(weights).is_file() else weights   # a bare name downloads
        WEIGHTS.mkdir(exist_ok=True)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def _fresh(self):
        with contextlib.chdir(WEIGHTS):
            return self._load()

    def _load(self):
        if self.kind == "yolo-world":
            from ultralytics import YOLOWorld
            return YOLOWorld(self.weights)
        from ultralytics import YOLOE
        return YOLOE(self.weights)

    def text_model(self, prompt):
        m = self._fresh()                            # fresh each time: set_classes moves the text tower between devices
        with contextlib.chdir(WEIGHTS):
            if self.kind == "yolo-world":
                m.set_classes([prompt])
            else:
                m.set_classes([prompt], m.get_text_pe([prompt]))
        return m

    def visual_model(self, ref_img, boxes, text=None):
        """YOLOE with one class from a visual prompt (optionally averaged with a text prompt's embedding)"""
        import torch
        from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor, YOLOEVPSegPredictor
        m = self._fresh()
        pc = YOLOEVPSegPredictor if m.task == "segment" else YOLOEVPDetectPredictor
        p = pc(overrides={"task": m.task, "mode": "predict", "save": False, "verbose": False, "batch": 1, "imgsz": 640})
        m.model.model[-1].nc = 1
        m.model.names = ["object0"]
        p.set_prompts({"bboxes": np.asarray(boxes, float), "cls": np.zeros(len(boxes), int)})
        p.setup_model(model=m.model)
        vpe = p.get_vpe(np.asarray(ref_img)[..., ::-1].copy())
        m = self._fresh()                            # the VP predictor rewired the head
        if text:
            with contextlib.chdir(WEIGHTS):
                t = m.get_text_pe([text]).to(vpe.device, vpe.dtype)
            vpe = torch.nn.functional.normalize(vpe + t, dim=-1)
        m.set_classes(["object0"], vpe)
        return m


def mosaic(views, size=640, n=9):
    """reference image for a visual prompt: n renders spread over the views on a grey canvas, and their boxes"""
    k = int(np.ceil(np.sqrt(n)))
    cell = size // k
    can = np.full((size, size, 3), 114, np.uint8)
    boxes = []
    pick = np.linspace(0, len(views) - 1, n).astype(int)
    for j, i in enumerate(pick):
        rgb, m, _ = views[i]
        ys, xs = np.nonzero(m)
        crop, cm = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1], m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        s = 0.85 * cell / max(crop.shape[:2])
        w, h = max(4, round(crop.shape[1] * s)), max(4, round(crop.shape[0] * s))
        crop = np.asarray(Image.fromarray(crop).resize((w, h), Image.BILINEAR))
        cm = np.asarray(Image.fromarray(cm.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        oy, ox = (j // k) * cell + (cell - h) // 2, (j % k) * cell + (cell - w) // 2
        roi = can[oy:oy + h, ox:ox + w]
        roi[cm] = crop[cm]
        boxes.append([ox, oy, ox + w, oy + h])
    return Image.fromarray(can), boxes
