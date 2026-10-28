"""handeye.py — time offset and extrinsic between two rigidly mounted sensors from their own trajectories.

X = T_a_b, the pose of sensor b in sensor a's frame (maps b coordinates into a). For every short window the motion
of a (A = Ta(t)^-1 Ta(t+w)) and of b over the same real time interval satisfy A X = X B.

Ground robots move in a plane: every rotation axis is (nearly) the same vertical, so
  * the rotation about that axis is fixed only through the translations, and
  * the translation along it (the height difference) is not observable at all.
The solver detects this, keeps the unobservable part at its prior (0) and reports it; the geometric refinement
(refine.py) measures it from the sensors' data instead.

Time: tb_true = ta + tau, i.e. sensor b's stamps are `tau` late; b's trajectory is read at t + tau.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp


class Traj:
    """trajectory with interpolation; pairs across an odometry break (segment change) are rejected"""

    def __init__(self, ts_ns, T, seg=None):
        o = np.argsort(ts_ns)
        self.ns = np.asarray(ts_ns, np.int64)[o]
        self.t = self.ns / 1e9
        self.T = np.asarray(T)[o]
        self.seg = (np.zeros(len(self.t)) if seg is None else np.asarray(seg))[o]
        self.slerp = Slerp(self.t, R.from_matrix(self.T[:, :3, :3]))

    def at(self, t):
        t = np.asarray(t, np.float64)
        M = np.zeros(t.shape + (4, 4))
        M[..., :3, :3] = self.slerp(np.clip(t, self.t[0], self.t[-1])).as_matrix()
        for k in range(3):
            M[..., k, 3] = np.interp(t, self.t, self.T[:, k, 3])
        M[..., 3, 3] = 1
        return M

    def same_seg(self, t0, t1):
        i0 = np.searchsorted(self.t, t0) - 1
        i1 = np.searchsorted(self.t, t1)
        ok = (i0 >= 0) & (i1 < len(self.t))
        i0, i1 = np.clip(i0, 0, len(self.t) - 1), np.clip(i1, 0, len(self.t) - 1)
        return ok & (self.seg[i0] == self.seg[i1])

    def rel(self, t, w):
        A0, A1 = self.at(t), self.at(t + w)
        return np.linalg.inv(A0) @ A1, self.same_seg(t, t + w)


def _angle_deg(M):
    return np.degrees(np.linalg.norm(R.from_matrix(M[..., :3, :3]).as_rotvec(), axis=-1))


def time_offset(a: Traj, b: Traj, search_s=0.5, win=1.0, step=0.2):
    """tau that makes the rotation angle per `win` seconds agree (frame-independent, so it needs no extrinsic)"""
    lo, hi = max(a.t[0], b.t[0]) + search_s + 0.1, min(a.t[-1], b.t[-1]) - search_s - win - 0.1
    g = np.arange(lo, hi, step)
    Ar, oka = a.rel(g, win)
    aa = _angle_deg(Ar)

    def cost(tau):
        Br, okb = b.rel(g + tau, win)
        ok = oka & okb & (aa > 1.0)               # still periods carry no timing information
        return float(np.median(np.abs(aa[ok] - _angle_deg(Br)[ok]))) if ok.sum() > 20 else np.inf

    taus = np.arange(-search_s, search_s + 1e-9, 0.01)
    c = np.array([cost(x) for x in taus])
    k = int(np.argmin(c))
    fine = np.arange(taus[k] - 0.012, taus[k] + 0.0121, 0.001)
    cf = np.array([cost(x) for x in fine])
    tau = float(fine[int(np.argmin(cf))])
    second = np.min(c[np.abs(taus - taus[k]) > 0.1]) if np.any(np.abs(taus - taus[k]) > 0.1) else np.inf
    return tau, {"tau_s": round(tau, 4), "median_angle_err_deg": round(float(cf.min()), 3),
                 "next_best_ratio": round(float(second / max(cf.min(), 1e-9)), 2),
                 "scan": [(round(float(x), 3), round(float(y), 3)) for x, y in zip(taus, c) if np.isfinite(y)]}


def motion_pairs(a: Traj, b: Traj, tau, windows=(0.5, 1.0, 2.0, 3.0), step=0.25, min_deg=1.0, min_m=0.05):
    """(A, B) relative motions over the same real interval, both inside one odometry segment and not at rest"""
    lo, hi = max(a.t[0], b.t[0] - tau) + 0.05, min(a.t[-1], b.t[-1] - tau) - 0.05
    As, Bs = [], []
    for w in windows:
        g = np.arange(lo, hi - w, step)
        A, oka = a.rel(g, w)
        B, okb = b.rel(g + tau, w)
        moving = (_angle_deg(A) > min_deg) | (np.linalg.norm(A[:, :3, 3], axis=1) > min_m)
        ok = oka & okb & moving
        As.append(A[ok]); Bs.append(B[ok])
    return np.concatenate(As), np.concatenate(Bs)


def _residuals(x, A, B, w_t, prior, scale_free, fix_dir=None):
    RX = R.from_rotvec(x[:3]).as_matrix()
    tX = x[3:6]
    s = x[6] if scale_free else 1.0
    RA, tA, RB, tB = A[:, :3, :3], A[:, :3, 3], B[:, :3, :3], B[:, :3, 3]
    E = RA @ RX @ np.transpose(RB, (0, 2, 1)) @ RX.T
    r_rot = R.from_matrix(E).as_rotvec()
    r_t = (np.einsum("nij,j->ni", RA, tX) + tA - (s * tB) @ RX.T - tX) * w_t
    pri = prior * tX if fix_dir is None else np.append(prior * tX, 10.0 * (tX @ fix_dir))
    return np.concatenate([r_rot.ravel(), r_t.ravel(), pri])


def motion_residuals(X, A, B, w_t=0.5):
    """A X = X B residuals for a given X: rotation (rad) and weighted translation (m * w_t), per pair"""
    x = np.concatenate([R.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3]])
    return _residuals(x, A, B, w_t, 0.0, False)[:6 * len(A)]


def solve(A, B, w_t=0.5, prior=1e-3, scale_free=False, X0=None):
    """robust least squares for X (and optionally a scale of b's translations). Returns X, info."""
    al = R.from_matrix(A[:, :3, :3]).as_rotvec()
    be = R.from_matrix(B[:, :3, :3]).as_rotvec()
    big = (np.linalg.norm(al, axis=1) > np.radians(2)) & (np.linalg.norm(be, axis=1) > np.radians(2))
    sv = np.linalg.svd(al[big] / np.linalg.norm(al[big], axis=1, keepdims=True), compute_uv=False) if big.sum() > 5 else np.zeros(3)
    planar = bool(sv[1] < 0.15 * sv[0]) if big.sum() > 5 else True
    def axis(v):
        _, _, vt = np.linalg.svd(v, full_matrices=False)
        return vt[0] if (v @ vt[0]).sum() > 0 else -vt[0]

    na = axis(al[big]) if planar else None          # planar: the translation along it is held at 0
    if X0 is not None:
        starts = [R.from_matrix(X0[:3, :3])]
    elif planar:
        # line up the (common) rotation axes, then try every turn about that axis: the translations decide
        Rab, _ = R.align_vectors([na], [axis(be[big])])
        starts = [R.from_rotvec(na * np.radians(d)) * Rab for d in range(0, 360, 15)]
    else:
        starts = [R.align_vectors(al[big], be[big])[0]]
    best = None
    for Rs in starts:
        x0 = np.concatenate([Rs.as_rotvec(), np.zeros(3)] + ([[1.0]] if scale_free else []))
        if X0 is not None:
            x0[3:6] = X0[:3, 3]
        r = least_squares(_residuals, x0, args=(A, B, w_t, prior, scale_free, na), loss="huber", f_scale=0.02)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R.from_rotvec(x[:3]).as_matrix(), x[3:6]
    n = len(A)
    res = _residuals(x, A, B, w_t, 0.0, scale_free)[:6 * n]
    rot_err = np.degrees(np.linalg.norm(res[:3 * n].reshape(-1, 3), axis=1))
    t_err = np.linalg.norm(res[3 * n:].reshape(-1, 3), axis=1) / w_t
    # observability: singular values of the data Jacobian (prior rows excluded) over the 6 pose parameters
    J = best.jac[:6 * n, :6]
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    weak = [Vt[i] for i in range(6) if s[i] < 1e-2 * s[0]]
    unobs_t = [na] if planar else [v[3:] / np.linalg.norm(v[3:]) for v in weak if np.linalg.norm(v[3:]) > 0.9]
    info = {"pairs": int(n), "planar_motion": planar, "rotation_axis_spread": [round(float(v), 3) for v in sv],
            "rot_residual_median_deg": round(float(np.median(rot_err)), 3),
            "trans_residual_median_cm": round(float(np.median(t_err)) * 100, 2),
            "unobservable_translation_dirs_in_a": [np.round(v, 3).tolist() for v in unobs_t]}
    if scale_free:
        info["scale_b_over_a"] = round(float(x[6]), 4)
    return X, info


def split_check(A, B, X, **kw):
    """solve on even and odd pairs separately and report how far the two answers are apart"""
    Xe, _ = solve(A[0::2], B[0::2], X0=X, **kw)
    Xo, _ = solve(A[1::2], B[1::2], X0=X, **kw)
    D = np.linalg.inv(Xe) @ Xo
    return {"even_vs_odd_cm": round(float(np.linalg.norm(Xe[:3, 3] - Xo[:3, 3])) * 100, 2),
            "even_vs_odd_deg": round(float(_angle_deg(D)), 3)}
