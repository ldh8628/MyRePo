"""refine.py — geometry helpers: depth surfaces, the LiDAR-on-depth check, merged LiDAR scans, floor planes.

The LiDAR-on-depth score is a check, not the solver: indoors a 16-beam LiDAR mostly sees vertical walls, and a
6-DoF LiDAR-to-depth fit moved the height by 40-50 cm on the 260901 recordings while the floor planes (floor.py)
agreed with an independent calibration within 5 cm.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from .handeye import Traj


class DepthSurface:
    """3D points and normals of one depth image (camera optical frame), depth edges removed"""

    def __init__(self, depth_m, K, D, stride=2, zmax=4.5):
        h, w = depth_m.shape
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z = depth_m[::stride, ::stride].astype(np.float64)
        und = cv2.undistortPoints(np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float32).reshape(-1, 1, 2), K, D).reshape(-1, 2)
        P = np.column_stack([und[:, 0] * z.ravel(), und[:, 1] * z.ravel(), z.ravel()]).reshape(z.shape + (3,))
        du, dv = np.zeros_like(P), np.zeros_like(P)
        du[:, 1:-1] = P[:, 2:] - P[:, :-2]
        dv[1:-1] = P[2:] - P[:-2]
        N = np.cross(du, dv)
        nn = np.linalg.norm(N, axis=2, keepdims=True)
        ok = (z > 0.3) & (z < zmax) & (nn[..., 0] > 1e-9)
        jump = np.zeros_like(z, bool)
        jump[:, 1:-1] |= np.abs(z[:, 2:] - z[:, :-2]) > 0.05 * z[:, 1:-1] + 0.02
        jump[1:-1] |= np.abs(z[2:] - z[:-2]) > 0.05 * z[1:-1] + 0.02
        ok &= ~jump
        self.P, self.N = P[ok], (N / np.maximum(nn, 1e-12))[ok]
        self.tree = cKDTree(self.P)
        self.K, self.D, self.w, self.h, self.zmax = K, D, w, h, zmax
        self.valid = (depth_m > 0.3) & (depth_m < zmax)       # pixels with a usable depth reading


def _in_view(Q, F, margin=5):
    """points that project into the image onto a pixel with a depth reading, within the depth range"""
    Q = Q[(Q[:, 2] > 0.4) & (Q[:, 2] < F.zmax)]
    if not len(Q):
        return Q
    uv = cv2.projectPoints(Q, np.zeros(3), np.zeros(3), F.K, F.D)[0].reshape(-1, 2)
    ok = (uv[:, 0] > margin) & (uv[:, 0] < F.w - margin) & (uv[:, 1] > margin) & (uv[:, 1] < F.h - margin)
    Q, uv = Q[ok], uv[ok].astype(int)
    return Q[F.valid[uv[:, 1], uv[:, 0]]]


def score(T_cam_lidar, pairs, cap=8000, tol=0.03):
    """share of in-view LiDAR points within `tol` of the depth surface, and median point-to-plane distance"""
    share, med = [], []
    for L, F in pairs:
        Q = _in_view((T_cam_lidar[:3, :3] @ L.T).T + T_cam_lidar[:3, 3], F)
        if len(Q) < 200:
            continue
        if len(Q) > cap:
            Q = Q[np.random.default_rng(0).choice(len(Q), cap, replace=False)]
        d, idx = F.tree.query(Q, distance_upper_bound=0.3)
        ok = np.isfinite(d)
        res = np.abs(np.sum((Q[ok] - F.P[idx[ok]]) * F.N[idx[ok]], axis=1))
        share.append((res < tol).sum() / len(Q))
        med.append(np.median(res) if len(res) else 0.3)
    return (float(np.mean(share)), float(np.median(med))) if share else (0.0, 1.0)


def lidar_camera_pairs(cam, lid, lid_traj: Traj, tau, n=80, every_s=0.5, max_rate_dps=10.0, max_speed=0.3,
                       zmax=4.5, stride=2):
    """(LiDAR points in the lidar frame AT the camera frame's real time, depth surface of that frame).

    lid: LidarStream; lid_traj: its odometry (stamps = scan stamps); tau: camera stamps are tau late against the lidar.
    Each scan is moved to the camera instant with the LiDAR odometry; slow-motion frames keep that move small."""
    st = lid_traj.t
    cand = []
    for t, _, depth in cam.frames(every_s=every_s, color=False):
        tr = t / 1e9 - tau                                    # real time of the camera frame
        if tr < lid_traj.t[0] + 0.2 or tr > lid_traj.t[-1] - 0.2:
            continue
        M = np.linalg.inv(lid_traj.at(tr - 0.1)) @ lid_traj.at(tr + 0.1)
        rate = np.degrees(np.linalg.norm(R.from_matrix(M[:3, :3]).as_rotvec())) / 0.2
        speed = np.linalg.norm(M[:3, 3]) / 0.2
        if rate > max_rate_dps or speed > max_speed:
            continue
        cand.append((tr, depth))
    pick = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, min(n, len(cand))).astype(int))] if cand else []
    want = {}
    for tr, depth in pick:
        k = int(np.argmin(np.abs(st - tr)))
        if abs(st[k] - tr) <= 0.1:
            want.setdefault(int(lid_traj.ns[k]), []).append((tr, depth))
    pairs = []
    for t, X, _ in lid.scans():
        for tr, depth in want.get(t, []):
            M = np.linalg.inv(lid_traj.at(tr)) @ lid_traj.at(t / 1e9)  # scan pose -> lidar pose at the camera instant
            r = np.linalg.norm(X, axis=1)
            X = X[(r > 0.5) & (r < 8.0)]
            pairs.append(((M[:3, :3] @ X.T).T + M[:3, 3], DepthSurface(depth, cam.info.K, cam.info.D, stride=stride, zmax=zmax)))
    return pairs, len(cand)


def floor_heights(cam, up_cam, every_s=0.5, n=200, zmax=3.5, max_tilt_deg=10.0):
    """per frame: unit floor normal (pointing up, camera frame) and camera height above it, by RANSAC on the depth
    points below the camera. `up_cam` (from the hand-eye rotation axis) only picks the right plane."""
    import open3d as o3d
    K, out = cam.info.K, []
    for t, _, depth in cam.frames(every_s=every_s, color=False):
        z = depth[::4, ::4]
        vv, uu = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
        ok = (z > 0.3) & (z < zmax)
        P = np.column_stack([(uu[ok] - K[0, 2]) / K[0, 0] * z[ok], (vv[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]])
        P = P[P @ up_cam < -0.15]                            # below the camera
        if len(P) < 500:
            continue
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        (a, b, c, d), inl = pc.segment_plane(0.02, 3, 200)
        nrm = np.array([a, b, c])
        if nrm @ up_cam < 0:
            nrm, d = -nrm, -d
        if np.degrees(np.arccos(np.clip(nrm @ up_cam, -1, 1))) > max_tilt_deg or len(inl) < 0.15 * len(P):
            continue
        out.append((t, nrm, float(d)))                       # floor: nrm . p + d = 0, camera height = d
        if len(out) >= n:
            break
    return out


class LidarMap:
    """LiDAR scans placed with the LiDAR odometry and merged per time block (blocks keep odometry drift local).
    Over a drive the floor around the path is filled in, which a single scan of a 16-beam LiDAR never sees close by."""

    def __init__(self, lid, lid_traj: Traj, every=3, voxel=0.05, block_s=20.0, r_max=12.0):
        import open3d as o3d
        st = lid_traj.t
        self.edges = np.arange(st[0], st[-1] + block_s, block_s / 2)       # blocks of block_s, half overlapping
        self.block_s = block_s
        pts = {k: [] for k in range(len(self.edges))}
        for i, (t, X, _) in enumerate(lid.scans()):
            if i % every:
                continue
            r = np.linalg.norm(X, axis=1)
            X = X[(r > 0.8) & (r < r_max)]
            M = lid_traj.at(t / 1e9)
            W = (M[:3, :3] @ X.T).T + M[:3, 3]
            for k, e in enumerate(self.edges):
                if e <= t / 1e9 < e + block_s:
                    pts[k].append(W)
        self.blocks = []
        for k in range(len(self.edges)):
            if not pts[k]:
                self.blocks.append(None)
                continue
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(pts[k]))).voxel_down_sample(voxel)
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
            P, N = np.asarray(pc.points), np.asarray(pc.normals)
            self.blocks.append((P, N, cKDTree(P)))

    def block(self, t):
        """the block whose centre is nearest to t"""
        k = int(np.argmin(np.abs(self.edges + self.block_s / 2 - t)))
        return self.blocks[k]
