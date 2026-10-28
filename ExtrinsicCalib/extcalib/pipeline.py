"""pipeline.py — the whole calibration for one sensor pair, with progress reports.

  sensor a : the reference (LiDAR, or the first RGB-D camera)
  sensor b : an RGB-D camera
Result X = T_a_b (pose of b in a; maps b coordinates into a) and tau (b's stamps are tau seconds late).

Steps: own trajectories -> time offset -> hand-eye from motion -> floor (height and tilt, planar motion only)
-> checks (even/odd split, LiDAR points on the depth surface) -> figures.
"""
from __future__ import annotations

import base64
import io
import json
import time

import numpy as np

from . import floor, refine
from .bagio import LidarStream, RGBDStream
from .handeye import Traj, _angle_deg, motion_pairs, solve, split_check, time_offset
from .odometry import lidar_odometry, rgbd_odometry


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _open(spec):
    if spec["type"] == "lidar":
        return LidarStream(spec["path"], topic=spec.get("points_topic") or "/velodyne_points")
    return RGBDStream(spec["path"], color_topic=spec.get("color_topic") or "/camera/camera/color/image_raw",
                      depth_topic=spec.get("depth_topic") or "/camera/camera/aligned_depth_to_color/image_raw",
                      info_topic=spec.get("info_topic") or "/camera/camera/color/camera_info",
                      depth_scale=float(spec.get("depth_scale") or 0.001))


def run(a_spec, b_spec, log=print, every_s=0.1, scale=0.5):
    """a_spec / b_spec: {"type": "lidar"|"rgbd", "path": bag dir, topics...}. Returns the result dict."""
    if b_spec["type"] != "rgbd":
        raise ValueError("sensor b must be an RGB-D camera")
    t_start = time.time()
    sa, sb = _open(a_spec), _open(b_spec)

    def odom(spec, s, tag, lo, hi):
        log(tag, lo, f"{tag}: own trajectory")
        if spec["type"] == "lidar":
            ts, T, seg = lidar_odometry(s)
        else:
            ts, T, seg = rgbd_odometry(s, every_s=every_s, scale=scale)
        log(tag, hi, f"{tag}: {len(ts)} poses, {int(seg.max()) + 1} segment(s)")
        return Traj(ts, T, seg), int(seg.max()) + 1
    ta, nseg_a = odom(a_spec, sa, "a", 0.02, 0.35)
    tb, nseg_b = odom(b_spec, sb, "b", 0.36, 0.70)

    log("sync", 0.72, "time offset from rotation rates")
    tau, tinfo = time_offset(ta, tb)
    A, B = motion_pairs(ta, tb, tau)
    if len(A) < 50:
        raise RuntimeError(f"only {len(A)} moving windows overlap in time: is it one simultaneous recording?")
    log("handeye", 0.76, f"tau {tau:+.3f} s, hand-eye on {len(A)} motion pairs")
    X0, hinfo = solve(A, B)
    _, sinfo = solve(A, B, scale_free=True, X0=X0)
    hinfo["scale_b_over_a"] = sinfo["scale_b_over_a"]

    X, finfo, lmap = X0, None, None
    if hinfo["planar_motion"]:
        up_a = np.array(hinfo["unobservable_translation_dirs_in_a"][0])
        try:
            log("floor", 0.80, "floor of b (height, tilt)")
            nb, hb, ib = floor.camera_floor(sb, X0[:3, :3].T @ up_a)
            log("floor", 0.85, "floor of a")
            if a_spec["type"] == "lidar":
                lmap = refine.LidarMap(sa, ta)
                na, ha, ia = floor.lidar_floor(lmap, ta, X0[:3, :3] @ nb)
            else:
                na, ha, ia = floor.camera_floor(sa, X0[:3, :3] @ nb, signed_hint=True)
            # on a flat floor each sensor turns about its floor normal: a larger angle means another plane was taken
            off = {"a": _deg(up_a, na), "b": _deg(X0[:3, :3].T @ up_a, nb)}
            if max(off.values()) > 3.0:
                raise RuntimeError(f"floor normal and turning axis differ by {off} deg")
            X1, tilt = floor.apply_floor(X0, na, ha, nb, hb)
            X = floor.planar_refit(X1, A, B, na)
            finfo = {"used": True, "a": ia, "b": ib, "tilt_correction_deg": tilt, "turning_axis_vs_floor_deg": off}
        except RuntimeError as e:
            finfo = {"used": False, "reason": str(e)}
            log("floor", 0.86, f"floor not used: {e}")

    log("check", 0.90, "checks")
    checks = {"even_odd_split": split_check(A, B, X)}
    figs = {"tau": _fig_tau(tinfo), "top": _fig_top(ta, tb, X, tau)}
    if a_spec["type"] == "lidar":
        pairs, _ = refine.lidar_camera_pairs(sb, sa, ta, tau, n=40)
        if pairs:
            sc = [refine.score(np.linalg.inv(X), pairs, tol=t) for t in (0.02, 0.03, 0.05)]
            checks["lidar_on_depth"] = {"frames": len(pairs), "within_2cm": round(sc[0][0], 3), "within_3cm": round(sc[1][0], 3),
                                        "within_5cm": round(sc[2][0], 3), "median_cm": round(sc[1][1] * 100, 2)}
        figs["overlay"] = _fig_overlay(sa, sb, ta, X, tau)
    Xi = np.linalg.inv(X)
    from scipy.spatial.transform import Rotation as R
    q = R.from_matrix(X[:3, :3]).as_quat()
    res = {
        "sensor_a": {k: v for k, v in a_spec.items()}, "sensor_b": {k: v for k, v in b_spec.items()},
        "T_a_b": np.round(X, 6).tolist(), "T_b_a": np.round(Xi, 6).tolist(),
        "translation_m": np.round(X[:3, 3], 4).tolist(), "quaternion_xyzw": np.round(q, 6).tolist(),
        "rpy_deg": np.round(R.from_matrix(X[:3, :3]).as_euler("xyz", degrees=True), 3).tolist(),
        "tau_s": round(tau, 4),
        "meaning": "T_a_b maps points from b's frame into a's frame; b's stamps are tau_s seconds late (b real time = stamp - tau_s)",
        "time_offset": {k: v for k, v in tinfo.items() if k != "scan"},
        "handeye": hinfo, "floor": finfo, "checks": checks,
        "segments": {"a": nseg_a, "b": nseg_b},
        "change_by_floor": {"cm": round(float(np.linalg.norm(X[:3, 3] - X0[:3, 3])) * 100, 2),
                            "deg": round(float(_angle_deg(np.linalg.inv(X0) @ X)), 3)},
        "seconds": round(time.time() - t_start, 1),
        "warnings": _warnings(tinfo, hinfo, finfo, checks, nseg_b),
        "figures": figs,
    }
    log("done", 1.0, f"done in {res['seconds']} s")
    return res


def _deg(u, v):
    return round(float(np.degrees(np.arccos(np.clip(abs(u @ v) / np.linalg.norm(u) / np.linalg.norm(v), 0, 1)))), 3)


def _warnings(tinfo, hinfo, finfo, checks, nseg_b):
    w = []
    if tinfo["next_best_ratio"] < 1.2:
        w.append("time offset is weakly determined (another offset fits almost as well): drive with more turns")
    if hinfo["rot_residual_median_deg"] > 1.5:
        w.append(f"rotation residual {hinfo['rot_residual_median_deg']} deg is high: odometry of one sensor is poor")
    if abs(hinfo["scale_b_over_a"] - 1) > 0.03:
        w.append(f"b's odometry moves {hinfo['scale_b_over_a']:.3f}x as far as a's: check the depth scale of the camera")
    if hinfo["planar_motion"] and not (finfo or {}).get("used"):
        w.append("planar motion and no reliable floor: the height difference is NOT determined (left at 0) and the "
                 f"tilt comes from the turning axes only ({(finfo or {}).get('reason', '')})")
    s = checks["even_odd_split"]
    if s["even_vs_odd_cm"] > 2 or s["even_vs_odd_deg"] > 0.5:
        w.append(f"halves of the data disagree by {s['even_vs_odd_cm']} cm / {s['even_vs_odd_deg']} deg")
    if nseg_b > 20:
        w.append(f"camera odometry broke {nseg_b - 1} times (fast motion, dark or textureless views)")
    return w


def _fig_tau(tinfo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = zip(*tinfo["scan"])
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(x, y, color="#2563eb", lw=1.6)
    ax.axvline(tinfo["tau_s"], color="#dc2626", lw=1, ls="--")
    ax.set_xlabel("time offset tau [s]"); ax.set_ylabel("rotation mismatch [deg / 1 s]")
    ax.grid(alpha=0.3)
    return _png(fig)


def _fig_top(ta, tb, X, tau, win=1.0):
    """per 1-s window: a's own motion and b's motion carried into a by the result (X B X^-1).
    Odometry drift does not enter: each window is compared on its own."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.arange(max(ta.t[0], tb.t[0] - tau) + 0.1, min(ta.t[-1], tb.t[-1] - tau) - win - 0.1, 0.2)
    A, oka = ta.rel(g, win)
    B, okb = tb.rel(g + tau, win)
    P = X @ B @ np.linalg.inv(X)
    ok = oka & okb
    x = g - g[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True)
    for ax, fa, fb, lab in ((ax1, np.linalg.norm(A[:, :3, 3], axis=1), np.linalg.norm(P[:, :3, 3], axis=1), "move [m / 1 s]"),
                            (ax2, _angle_deg(A), _angle_deg(P), "turn [deg / 1 s]")):
        fa, fb = np.where(ok, fa, np.nan), np.where(ok, fb, np.nan)
        ax.plot(x, fa, color="#2563eb", lw=1.4, label="sensor a")
        ax.plot(x, fb, color="#ea580c", lw=1.0, ls="--", label="sensor b through the result")
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    err = np.linalg.norm(A[ok, :3, 3] - P[ok, :3, 3], axis=1)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"1-s motion, position mismatch median {np.median(err) * 100:.1f} cm", fontsize=9)
    ax2.set_xlabel("time [s]")
    return _png(fig)


def _fig_overlay(lid, cam, ta, X, tau, n=3):
    """LiDAR points drawn on camera colour images (coloured by range)"""
    import cv2
    Tcl = np.linalg.inv(X)
    times = []
    for t, bgr, _ in cam.frames(every_s=5.0):
        times.append((t, bgr))
    pick = [times[i] for i in np.linspace(len(times) * 0.2, len(times) * 0.8, n).astype(int)] if len(times) >= n else times
    scans = {}
    targets = {int(ta.ns[int(np.argmin(np.abs(ta.t - (t / 1e9 - tau))))]): (t, bgr) for t, bgr in pick}
    for t, xyz, _ in lid.scans():
        if t in targets:
            scans[t] = xyz
    tiles = []
    for ts_l, (t, bgr) in targets.items():
        if ts_l not in scans:
            continue
        M = np.linalg.inv(ta.at(t / 1e9 - tau)) @ ta.at(ts_l / 1e9)
        Xl = (M[:3, :3] @ scans[ts_l].T).T + M[:3, 3]
        Q = (Tcl[:3, :3] @ Xl.T).T + Tcl[:3, 3]
        Q = Q[Q[:, 2] > 0.3]
        uv = cv2.projectPoints(Q, np.zeros(3), np.zeros(3), cam.info.K, cam.info.D)[0].reshape(-1, 2).round().astype(int)
        h, w = bgr.shape[:2]
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        img = bgr.copy()
        col = cv2.applyColorMap((np.clip(Q[ok, 2] / 6.0, 0, 1) * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        for (x, y), c in zip(uv[ok], col):
            cv2.circle(img, (int(x), int(y)), 2, c.tolist(), -1)
        tiles.append(img)
    if not tiles:
        return None
    img = np.vstack(tiles) if len(tiles) > 1 else tiles[0]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def save(res, path):
    out = {k: v for k, v in res.items() if k != "figures"}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
