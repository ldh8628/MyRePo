"""floor.py — the floor as seen by each sensor: its normal (up) and the sensor's height above it.

For a rig on a ground robot the motion is planar, so the hand-eye step cannot see the height difference between
two sensors, and sees their relative tilt only through the common turning axis. The floor gives both directly:
  n_a = R_ab n_b          (up agrees)
  n_a . t_ab = h_b - h_a  (height difference)
Indoors a 16-beam LiDAR hits the floor only metres away, so its floor comes from the scans merged over the drive.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .handeye import Traj


def trajectory_up(traj: Traj):
    """normal of the plane the trajectory lies in (world frame), and the rms off that plane"""
    P = traj.T[:, :3, 3]
    c = P - P.mean(0)
    n = np.linalg.svd(c, full_matrices=False)[2][2]
    return n, float(np.sqrt(np.mean((c @ n) ** 2)))


def lidar_floor(lmap, traj: Traj, up_hint):
    """(up in the lidar frame, lidar height above the floor, info) from the merged LiDAR map.

    `up_hint` (lidar frame) says which side is down; it comes from a camera's floor carried over by the hand-eye
    rotation, because a camera sees the floor plainly in most frames while a 16-beam LiDAR hits it only metres
    away. Of the horizontal surfaces below the sensor (floor, table tops, ...), the floor holds the most points
    once the scans of the whole drive are merged."""
    n_w, rms = trajectory_up(traj)
    Rl = traj.T[:, :3, :3]
    if np.median(np.einsum("nji,j->ni", Rl, n_w) @ up_hint) < 0:
        n_w = -n_w
    P = np.vstack([b[0] for b in lmap.blocks if b])
    N = np.vstack([b[1] for b in lmap.blocks if b])
    h = P[np.abs(N @ n_w) > 0.95] @ n_w - traj.T[:, :3, 3].mean(0) @ n_w    # height relative to the sensor path
    hist, e = np.histogram(-h[h < -0.1], bins=np.arange(0.1, 6.0, 0.04))
    if hist.max(initial=0) < 200:
        raise RuntimeError("lidar floor: almost no horizontal surface below the sensor in the merged scans")
    k = int(np.argmax(hist))
    d = -h[h < -0.1]
    sel = np.abs(d - (e[k] + 0.02)) < 0.06
    height = float(np.median(d[sel]))
    up_sensor = np.median(np.einsum("nji,j->ni", Rl, n_w), axis=0)   # world up seen from each lidar pose
    up_sensor /= np.linalg.norm(up_sensor)
    order = np.argsort(hist)[::-1][:4]
    return up_sensor, height, {"height_m": round(height, 4), "floor_points": int(sel.sum()),
                               "height_spread_cm": round(float(np.std(d[sel])) * 100, 2), "path_plane_rms_cm": round(rms * 100, 2),
                               "next_bands_below_m": [(round(float(e[i] + 0.02), 2), int(hist[i])) for i in order[1:]]}


def camera_floor(cam, up_hint, signed_hint=False, every_s=0.5, n=200, zmax=3.5, max_tilt_deg=5.0, min_frames=30):
    """(up in the camera frame, camera height above the floor, info) by RANSAC on depth points below the camera.
    `up_hint` (camera frame) only chooses which plane is the floor; the measured normal replaces it. With
    `signed_hint` (e.g. the LiDAR's floor normal carried over by the hand-eye result) its sign is trusted."""
    from .refine import floor_heights
    # a hint from a turning axis has no sign: try both, the floor is seen in more frames than a ceiling
    tries = [floor_heights(cam, s_ * up_hint, every_s=every_s, n=n, zmax=zmax, max_tilt_deg=max_tilt_deg)
             for s_ in ((1.0,) if signed_hint else (1.0, -1.0))]
    fl = max(tries, key=len)
    Ns = np.array([f[1] for f in fl]).reshape(-1, 3); hs = np.array([f[2] for f in fl])
    keep = np.abs(hs - np.median(hs)) < 0.05 if len(hs) else np.zeros(0, bool)
    if keep.sum() < min_frames:                               # few frames, or planes at many heights (tables, ...)
        raise RuntimeError(f"only {int(keep.sum())} frames show one consistent floor plane within {max_tilt_deg} deg "
                           f"of the turning axis ({len(fl)} with some plane)")
    n = np.median(Ns[keep], axis=0)
    n /= np.linalg.norm(n)
    return n, float(np.median(hs[keep])), {"frames": int(keep.sum()), "of": len(fl), "height_m": round(float(np.median(hs[keep])), 4),
                                           "height_mad_cm": round(float(np.median(np.abs(hs[keep] - np.median(hs[keep])))) * 100, 2)}


def apply_floor(X, n_a, h_a, n_b, h_b):
    """put the floor constraints into X = T_a_b: tilt so that R n_b = n_a, height so that n_a . t = h_b - h_a"""
    Xn = X.copy()
    Rfix, _ = R.align_vectors([n_a], [X[:3, :3] @ n_b])
    Xn[:3, :3] = Rfix.as_matrix() @ X[:3, :3]
    t = X[:3, 3]
    Xn[:3, 3] = t - (t @ n_a) * n_a + (h_b - h_a) * n_a
    return Xn, round(float(np.degrees(np.linalg.norm(Rfix.as_rotvec()))), 3)


def planar_refit(X, A, B, n_a, w_t=0.5):
    """re-solve the turn about n_a and the horizontal offset with tilt and height held from the floor"""
    from scipy.optimize import least_squares
    from .handeye import motion_residuals
    h = X[:3, 3] @ n_a

    def make(p):
        Xn = np.eye(4)
        Xn[:3, :3] = R.from_rotvec(n_a * p[0]).as_matrix() @ X[:3, :3]
        t = X[:3, 3] + p[1:4]
        Xn[:3, 3] = t - (t @ n_a) * n_a + h * n_a
        return Xn
    r = least_squares(lambda p: motion_residuals(make(p), A, B, w_t), np.zeros(4), loss="huber", f_scale=0.02)
    return make(r.x)
