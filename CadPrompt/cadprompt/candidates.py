"""candidates.py — prompt candidates from what the CAD model itself offers: its name, colours, shape, and the nouns
CLIP finds closest to its renders. They are only candidates; detect.py measures which ones the detector answers.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# colour words a detector's text encoder knows; RGB centres
PALETTE = {
    "black": (25, 25, 25), "white": (240, 240, 240), "grey": (128, 128, 128), "silver": (190, 190, 195),
    "red": (200, 30, 35), "dark red": (120, 25, 30), "orange": (240, 130, 30), "yellow": (240, 210, 40),
    "beige": (215, 195, 160), "brown": (120, 75, 40), "green": (50, 150, 60), "dark green": (30, 80, 45),
    "light green": (150, 210, 120), "blue": (40, 80, 200), "light blue": (130, 180, 230), "navy blue": (30, 40, 90),
    "purple": (120, 60, 160), "pink": (240, 150, 180),
}


def name_words(name: str) -> str:
    """'AlphabetSoup' / 'obj_000003' / 'mustard_bottle' -> 'alphabet soup' / '' / 'mustard bottle'"""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", re.sub(r"([a-z])([A-Z])", r"\1 \2", Path(name).stem))
    words = [w.lower() for w in re.split(r"[\s_\-.]+", s) if w and not any(c.isdigit() for c in w)]
    words = [w for w in words if w not in {"obj", "model", "mesh", "textured", "high", "low", "cad", "scan"}]
    return " ".join(words)


def colour_words(views, k=3):
    """the main colours of the object over its renders, by area: [(word, share), ...]"""
    px = np.concatenate([rgb[m] for rgb, m, _ in views]).astype(np.float64)
    if len(px) > 60000:
        px = px[np.random.default_rng(0).choice(len(px), 60000, replace=False)]
    px = px / np.maximum(px.max(1, keepdims=True) / 255, 0.6)    # undo some of the shading before naming
    names = list(PALETTE)
    cent = np.array([PALETTE[n] for n in names], np.float64)
    lab = np.argmin(((px[:, None, :] - cent[None]) ** 2).sum(-1), 1)
    cnt = np.bincount(lab, minlength=len(names)) / len(px)
    order = np.argsort(cnt)[::-1][:k]
    return [(names[i], round(float(cnt[i]), 3)) for i in order if cnt[i] > 0.08]


def shape_words(model):
    """coarse shape from the bounding box and how much of it the object fills"""
    e = np.sort(model.extent)[::-1]
    a, b, c = e / e[0]
    try:
        import trimesh
        vol = trimesh.Trimesh(model.V, model.F, process=False).convex_hull.volume
    except Exception:  # noqa: BLE001
        vol = np.prod(e)
    fill = vol / max(np.prod(e), 1e-12)
    if c > 0.8 and fill < 0.62:
        return ["ball", "round object"]
    if c < 0.25 and b > 0.5:
        return ["flat object", "plate"]
    if b < 0.35:
        return ["long object", "stick"]
    if fill < 0.85 and (abs(b - c) < 0.15 or abs(a - b) < 0.15):
        return ["cylinder", "can", "bottle"]
    return ["box", "carton"]


def default_vocab():
    """LVIS (1203) and Objects365 (365) class names as shipped with ultralytics: a broad, standard list, so the
    nouns are not chosen with any test object in mind"""
    import ultralytics
    import yaml
    d = Path(ultralytics.__file__).parent / "cfg" / "datasets"
    out = []
    for f in ("lvis.yaml", "Objects365.yaml"):
        for n in yaml.safe_load(open(d / f))["names"].values():
            w = n.split("/")[0].replace("_", " ").strip().lower()      # first LVIS synonym
            if w and w not in out and w not in ("person", "other shoes"):
                out.append(w)
    return out


def clip_nouns(views, vocab_file=None, top=6, device=None):
    """nouns from the vocabulary that CLIP (ViT-B/32, the text tower YOLO-World uses) finds closest to the renders.
    vocab_file: one noun per line ('#' comments) to replace the default LVIS + Objects365 list"""
    import clip  # ultralytics' CLIP package
    import torch
    from PIL import Image
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = ([l.strip() for l in open(vocab_file) if l.strip() and not l.startswith("#")] if vocab_file
             else default_vocab())
    model, pre = clip.load("ViT-B/32", device=device)
    imgs = []
    for rgb, m, _ in views:
        ys, xs = np.nonzero(m)
        crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        crop[~m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]] = 255   # white backdrop
        imgs.append(pre(Image.fromarray(crop)))
    with torch.no_grad():
        fi = model.encode_image(torch.stack(imgs).to(device)).float()
        ft = model.encode_text(clip.tokenize([f"a photo of a {w}" for w in vocab]).to(device)).float()
        fi = torch.nn.functional.normalize(fi, dim=-1).mean(0)
        ft = torch.nn.functional.normalize(ft, dim=-1)
        p = (100 * ft @ torch.nn.functional.normalize(fi, dim=-1)).softmax(0).cpu().numpy()
    order = np.argsort(p)[::-1][:top]
    return [(vocab[i], round(float(p[i]), 3)) for i in order]


def build(model, views, name=None, vocab_file=None, max_candidates=24):
    """candidate list with where each came from, plus the evidence used"""
    nm = name_words(name or model.name)
    cols = colour_words(views)
    shapes = shape_words(model)
    nouns = clip_nouns(views, vocab_file)
    c1 = cols[0][0] if cols else ""
    out = []

    def add(p, src):
        p = " ".join(p.split())
        if p and p not in {o["prompt"] for o in out}:
            out.append({"prompt": p, "source": src})
    if nm:
        add(nm, "name")
        if c1:
            add(f"{c1} {nm}", "colour + name")
    for n, _ in nouns[:4]:
        add(n, "CLIP noun")
        if c1:
            add(f"{c1} {n}", "colour + CLIP noun")
    if len(cols) > 1:
        add(f"{c1} and {cols[1][0]} {nouns[0][0]}", "two colours + CLIP noun")
    for s in shapes:
        add(s, "shape")
        if c1:
            add(f"{c1} {s}", "colour + shape")
    add("object", "generic")
    return out[:max_candidates], {"name_words": nm, "colours": cols, "shape": shapes, "clip_nouns": nouns}
