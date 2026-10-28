"""grid.py — hdl_graph_slam dump (keyframe clouds + optimized poses) -> Nav2 map_server grid (PGM + YAML).

Map frame = hdl `map` (origin at the first LiDAR pose). The floor is the densest height bin below the sensor, or the
sensor height the user gives (a LiDAR mounted high sees little floor). Per keyframe, every point is a ray from the
keyframe position. Hits and passes are counted per voxel (cell x height bin inside [floor + obst_min, floor + obst_max]),
with the ray's own height at each cell:
  voxel occupied: hits >= min_hits and hits / (hits + passes) >= occ_ratio
  cell occupied:  any of its voxels is;   free: a ray crossed the column inside the band and no voxel is occupied
So a ray flying OVER a low obstacle to a wall behind it does not erase the obstacle.
Cells hit from fewer than min_kf keyframes are dropped: things that moved (people walking along) are seen from one or
two keyframes, fixed things from many.

Also writes <name>_pref.png/.yaml (map_server "scale" mode): free cells near the path driven while mapping (the
keyframe positions) cost 0, farther ones a little more, so a planner on that map PREFERS the driven path.
"""
from __future__ import annotations

import glob
import json
import os
import struct
import zlib

import numpy as np


def png_rgba(a):
    """HxWx4 uint8 -> PNG bytes (numpy + zlib only)"""
    h, w = a.shape[:2]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * 4)], 1).tobytes()
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def read_pcd(path):
    """binary PCD -> Nx3 float64 (x, y, z)"""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            k, _, v = f.readline().decode("ascii").strip().partition(" ")
            hdr[k] = v.split()
            if k == "DATA":
                break
        fields, sizes, types, n = hdr["FIELDS"], list(map(int, hdr["SIZE"])), hdr["TYPE"], int(hdr["POINTS"][0])
        if hdr["DATA"][0] != "binary":
            raise ValueError(f"{path}: only binary PCD is read")
        dt = np.dtype([(fn, {"F": "f", "U": "u", "I": "i"}[t] + str(s)) for fn, s, t in zip(fields, sizes, types)])
        a = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


def read_dump(d):
    """[(T_map_keyframe 4x4, cloud Nx3 in the keyframe frame)] from hdl_graph_slam's dump_graph folder"""
    out = []
    for kd in sorted(glob.glob(os.path.join(d, "[0-9]" * 6))):
        L = open(os.path.join(kd, "data")).read().split("\n")
        i = L.index("estimate")
        T = np.array([list(map(float, L[i + 1 + r].split())) for r in range(4)])
        out.append((T, read_pcd(os.path.join(kd, "cloud.pcd"))))
    if not out:
        raise ValueError(f"no keyframes in {d}")
    return out


def _grow(m):
    """boolean mask grown by one cell in all eight directions"""
    p = np.pad(m, 1)
    return p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]


def _tidy(occ, speckle, close):
    """drop occupied blobs smaller than `speckle` cells, then fill gaps of up to `close` cells inside obstacles"""
    from scipy import ndimage
    occ = occ.copy()
    if speckle > 1:
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        size = np.bincount(lab.ravel(), minlength=n + 1)
        occ &= size[lab] >= speckle
    if close > 0:
        occ |= ndimage.binary_closing(occ, structure=np.ones((3, 3)), iterations=close)
    return occ


def make_grid(dump, out_prefix, res=0.05, obst_min=0.15, obst_max=1.2, sensor_height=None, floor_tol=0.08,
              occ_ratio=0.1, min_hits=2, zres=0.15, min_kf=3, speckle=3, close=1,
              pref_width=0.5, pref_ramp=0.75, pref_penalty=30.0, log=print):
    """dump folder -> <out_prefix>.pgm/.yaml (+ _pref.png/.yaml); returns a summary dict"""
    kfs = read_dump(dump)
    world = [(T, c @ T[:3, :3].T + T[:3, 3]) for T, c in kfs]
    p = np.concatenate([w for _, w in world])
    if sensor_height is not None:
        floor = float(kfs[0][0][2, 3]) - sensor_height
    else:
        below = p[p[:, 2] < -0.1, 2]
        if len(below) < 1000:
            raise ValueError("almost no points below the sensor: give the sensor height above the floor")
        h, e = np.histogram(below, bins=np.arange(below.min(), -0.1, 0.02))
        floor = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    log(f"keyframes {len(kfs)}, points {len(p)}, floor z {floor:.3f} m (sensor ~{-floor:.2f} m above it)")

    pad = 1.0
    x0, y0 = p[:, 0].min() - pad, p[:, 1].min() - pad
    W = int(np.ceil((p[:, 0].max() + pad - x0) / res))
    H = int(np.ceil((p[:, 1].max() + pad - y0) / res))
    if W * H > 4e7:
        raise ValueError(f"grid {W}x{H} is too large: points far from the robot? use a coarser resolution")
    NZ = max(1, int(np.ceil((obst_max - obst_min) / zres)))
    hits = np.zeros((H, W, NZ), np.int32)
    passes = np.zeros((H, W, NZ), np.int32)
    kf_hit = np.zeros((H, W), np.int16)
    zbin = lambda hh: np.clip(((hh - obst_min) / zres).astype(int), 0, NZ - 1)
    step = res * 0.5
    for T, w in world:
        dz = w[:, 2] - floor
        keep = (dz > -floor_tol) & (dz < obst_max)                     # floor + obstacle band; no ceiling
        w, dz = w[keep], dz[keep]
        o = T[:2, 3]
        v = w[:, :2] - o
        dist = np.linalg.norm(v, axis=1)
        n = np.maximum(1, (dist / step).astype(int))
        k = np.repeat(np.arange(len(w)), n)                            # samples along each ray, one cell short of its end
        t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
        t *= np.repeat(np.clip(1 - res / np.maximum(dist, 1e-6), 0, 1), n)
        s_xy = o + v[k] * t[:, None]
        cx = ((s_xy[:, 0] - x0) / res).astype(int)
        cy = ((s_xy[:, 1] - y0) / res).astype(int)
        hs = (T[2, 3] - floor) + (dz[k] - (T[2, 3] - floor)) * t       # height of the ray above the floor
        inband = (hs >= obst_min) & (hs < obst_max)
        cell = np.unique(np.stack([cy[inband], cx[inband], zbin(hs[inband]), k[inband]], 1), axis=0)   # one pass per ray per voxel
        np.add.at(passes, (cell[:, 0], cell[:, 1], cell[:, 2]), 1)
        m = dz > obst_min
        hi = (((w[m, 1] - y0) / res).astype(int), ((w[m, 0] - x0) / res).astype(int), zbin(dz[m]))
        np.add.at(hits, hi, 1)
        h1 = np.zeros((H, W), bool)
        h1[hi[0], hi[1]] = True
        kf_hit += _grow(h1)
    occ = ((hits >= min_hits) & (hits >= occ_ratio * (hits + passes))).any(2)
    seen = passes.sum(2) > 0
    moved = 0
    if min_kf > 0 and len(kfs) >= 6:
        mv = occ & (kf_hit < min_kf)
        moved = int(mv.sum())
        occ &= ~mv
    occ = _tidy(occ, speckle, close)
    free = seen & ~occ
    img = np.full((H, W), 205, np.uint8)                               # unknown
    img[free] = 254
    img[occ] = 0
    name = os.path.basename(out_prefix)
    with open(out_prefix + ".pgm", "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (W, H))
        f.write(img[::-1].tobytes())                                   # PGM row 0 = top = max y
    with open(out_prefix + ".yaml", "w") as f:
        f.write(f"image: {name}.pgm\nmode: trinary\nresolution: {res}\norigin: [{x0:.3f}, {y0:.3f}, 0.0]\n"
                f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")

    # driven path = keyframe positions; cost rises away from it
    track = np.array([T[:2, 3] for T, _ in kfs])
    fy, fx = np.nonzero(free)
    cxy = np.column_stack([x0 + (fx + 0.5) * res, y0 + (fy + 0.5) * res])
    dmin = np.full(len(cxy), np.inf)
    dense = [track[0]]
    for a_, b_ in zip(track, track[1:]):                               # path points every 10 cm
        n_ = max(1, int(np.hypot(*(b_ - a_)) / 0.1))
        dense += list(a_ + (b_ - a_) * (np.arange(1, n_ + 1) / n_)[:, None])
    dense = np.array(dense)
    for i in range(0, len(dense), 64):
        dmin = np.minimum(dmin, np.sqrt(((cxy[:, None, :] - dense[None, i:i + 64, :]) ** 2).sum(2)).min(1))
    cost = pref_penalty * np.clip((dmin - pref_width) / max(pref_ramp, 1e-6), 0, 1)
    FREE_T, OCC_T = 0.02, 0.65                                         # scale mode: cost = 100 (occ - FREE_T) / (OCC_T - FREE_T)
    occv = np.where(cost < 0.5, 0.0, FREE_T + cost / 100.0 * (OCC_T - FREE_T))
    g = np.zeros((H, W, 4), np.uint8)                                  # alpha 0 = unknown
    g[fy, fx, :3] = np.round(255 * (1 - occv))[:, None].astype(np.uint8)
    g[fy, fx, 3] = 255
    g[occ] = (0, 0, 0, 255)
    with open(out_prefix + "_pref.png", "wb") as f:
        f.write(png_rgba(g[::-1]))
    with open(out_prefix + "_pref.yaml", "w") as f:
        f.write(f"image: {name}_pref.png\nmode: scale\nresolution: {res}\norigin: [{x0:.3f}, {y0:.3f}, 0.0]\n"
                f"negate: 0\noccupied_thresh: {OCC_T}\nfree_thresh: {FREE_T}\n")
    with open(out_prefix + "_track.json", "w") as f:
        json.dump([[round(float(x), 3), round(float(y), 3)] for x, y in dense[::2]], f)
    s = {"keyframes": len(kfs), "floor_z": round(float(floor), 3), "size": [W, H], "res": res,
         "origin": [round(x0, 3), round(y0, 3)], "occupied": int(occ.sum()), "free": int(free.sum()),
         "unknown": int((img == 205).sum()), "moved_dropped": moved,
         "track_length": round(float(np.hypot(*np.diff(track, axis=0).T).sum()), 1)}
    log(f"grid {W}x{H} @ {res} m: occupied {s['occupied']}, free {s['free']}, dropped as moving {moved}")
    return s
