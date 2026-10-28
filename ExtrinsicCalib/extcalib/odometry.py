"""odometry.py — each sensor's own trajectory, the input of the hand-eye step.

  RGB-D : Open3D RGB-D odometry, photometric term (depth only lifts pixels to 3D), frame to frame at `every_s`.
          The hybrid (photometric + geometric) term under-reads translation: 0.69 of the LiDAR's on straight
          stretches of a 260901 indoor drive, against 1.00 for the photometric term and for ORB-SLAM3.
  LiDAR : KISS-ICP (deskew from the previous motion, adaptive threshold, local voxel map).

Both return (stamps_ns (N,), T_world_sensor (N,4,4), segment id (N,)). Only short-range relative motion is used later, so
odometry drift over the whole recording does not matter; a broken step does, and starts a new segment.
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np
import open3d as o3d

from .bagio import LidarStream, RGBDStream


def _progress(tag, i, t0, every=200):
    if i % every == 0 and i:
        print(f"  [{tag}] {i} frames, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)


def rgbd_odometry(cam: RGBDStream, every_s=0.1, depth_max=4.0, scale=0.5, t_from=None, t_to=None):
    """frame-to-frame RGB-D odometry on images resized by `scale`. A failed or implausible step (> 0.3 m or > 25 deg
    per step) is not chained: the pose is held and the segment id goes up, so later stages never use a pair that
    crosses the break."""
    K = cam.info.K * np.array([[scale], [scale], [1.0]])
    w, h = round(cam.info.width * scale), round(cam.info.height * scale)
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    opt = o3d.pipelines.odometry.OdometryOption(depth_max=depth_max, depth_diff_max=0.07)
    jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()
    ts, poses, segs, prev, T, seg, t0 = [], [], [], None, np.eye(4), 0, time.time()
    for i, (t, bgr, depth) in enumerate(cam.frames(every_s=every_s, t_from=t_from, t_to=t_to)):
        if scale != 1.0:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(bgr[..., ::-1])), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=depth_max, convert_rgb_to_intensity=True)
        if prev is not None:
            ok, Tst, _ = o3d.pipelines.odometry.compute_rgbd_odometry(rgbd, prev, intr, np.eye(4), jac, opt)
            ang = np.degrees(np.arccos(np.clip((np.trace(Tst[:3, :3]) - 1) / 2, -1, 1)))
            if ok and np.linalg.norm(Tst[:3, 3]) < 0.3 and ang < 25:
                T = T @ Tst                         # Tst maps the current frame into the previous one
            else:
                seg += 1
        ts.append(t); poses.append(T.copy()); segs.append(seg); prev = rgbd
        _progress("rgbd", i, t0)
    return np.array(ts, np.int64), np.array(poses), np.array(segs)


def lidar_odometry(lid: LidarStream, voxel=0.10, min_range=0.5, max_range=30.0, t_from=None, t_to=None):
    """KISS-ICP over every scan. Scans with a header stamp already seen are dropped (some recorders store each
    message twice, which breaks the constant-velocity deskew)."""
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP
    cfg = load_config(None)
    cfg.data.min_range, cfg.data.max_range, cfg.data.deskew = min_range, max_range, True
    cfg.mapping.voxel_size = voxel
    odom = KissICP(cfg)
    ts, poses, seen, t0 = [], [], set(), time.time()
    for i, (t, xyz, tt) in enumerate(lid.scans(t_from=t_from, t_to=t_to)):
        if t in seen or len(xyz) < 100:
            continue
        seen.add(t)
        if tt is None or np.ptp(tt) <= 0:
            tt = np.zeros(len(xyz))
        else:
            tt = (tt - tt.min()) / np.ptp(tt)        # KISS-ICP wants 0..1 within the sweep
        odom.register_frame(xyz, tt)
        ts.append(t); poses.append(odom.last_pose.copy())
        _progress("lidar", i, t0)
    return np.array(ts, np.int64), np.array(poses), np.zeros(len(ts), int)


def write_tum(path, ts, T):
    from scipy.spatial.transform import Rotation
    with open(path, "w") as f:
        for t, M in zip(ts, T):
            q = Rotation.from_matrix(M[:3, :3]).as_quat()
            f.write(f"{t / 1e9:.9f} {M[0, 3]:.6f} {M[1, 3]:.6f} {M[2, 3]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")
