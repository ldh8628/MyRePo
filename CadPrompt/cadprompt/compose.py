"""compose.py — test images: object renders pasted onto backgrounds at random size, place and lighting.

The detector never sees the renders on black: each test image is a background (the user's photos when given,
otherwise generated clutter) with one render pasted in, and its box is known. Background-only images of the same
kind measure how often a prompt fires on nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _generated_background(rng, size):
    """smooth colour field + random rectangles and ellipses: stands in for an indoor scene"""
    h = w = size
    c0, c1 = rng.uniform(40, 220, 3), rng.uniform(40, 220, 3)
    t = np.linspace(0, 1, h)[:, None, None]
    img = (c0 * (1 - t) + c1 * t) * np.ones((h, w, 3))
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(rng.integers(6, 16)):
        col = rng.uniform(20, 235, 3)
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        rw, rh = rng.integers(w // 12, w // 2), rng.integers(h // 12, h // 2)
        if rng.random() < 0.6:
            m = (xx >= x0) & (xx < x0 + rw) & (yy >= y0) & (yy < y0 + rh)
        else:
            m = ((xx - x0) / rw) ** 2 + ((yy - y0) / rh) ** 2 < 1
        img[m] = img[m] * 0.3 + col * 0.7
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))


class Backgrounds:
    def __init__(self, folder=None, size=640, seed=0):
        self.size, self.rng = size, np.random.default_rng(seed)
        self.files = []
        if folder:
            self.files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    def next(self):
        if self.files:
            img = Image.open(self.files[self.rng.integers(len(self.files))]).convert("RGB")
            s = self.size / min(img.size)
            img = img.resize((max(self.size, round(img.width * s)), max(self.size, round(img.height * s))))
            x = self.rng.integers(0, img.width - self.size + 1)
            y = self.rng.integers(0, img.height - self.size + 1)
            return img.crop((x, y, x + self.size, y + self.size))
        return _generated_background(self.rng, self.size)


def composite(views, n_images=48, bg=None, size=640, scale=(0.15, 0.45), seed=0):
    """list of (PIL image, box [x0,y0,x1,y1]) — each view used in turn, with its own size/place/brightness"""
    rng = np.random.default_rng(seed)
    bg = bg or Backgrounds(size=size, seed=seed)
    out = []
    for k in range(n_images):
        rgb, mask, _ = views[k % len(views)]
        ys, xs = np.nonzero(mask)
        if len(ys) < 50:
            continue
        crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float64)
        cm = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crop = np.clip(crop * rng.uniform(0.7, 1.25) + rng.uniform(-15, 15), 0, 255)
        s = rng.uniform(*scale) * size / max(crop.shape[:2])
        ow, oh = max(8, round(crop.shape[1] * s)), max(8, round(crop.shape[0] * s))
        obj = Image.fromarray(crop.astype(np.uint8)).resize((ow, oh), Image.BILINEAR)
        alpha = Image.fromarray((cm * 255).astype(np.uint8)).resize((ow, oh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.8))
        canvas = bg.next()
        x, y = int(rng.integers(0, size - ow + 1)), int(rng.integers(0, size - oh + 1))
        canvas.paste(obj, (x, y), alpha)
        out.append((canvas, [x, y, x + ow, y + oh]))
    return out


def background_only(n_images=24, bg=None, size=640, seed=1):
    bg = bg or Backgrounds(size=size, seed=seed)
    return [bg.next() for _ in range(n_images)]
