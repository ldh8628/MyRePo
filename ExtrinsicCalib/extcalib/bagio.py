"""bagio.py — read RGB-D frames and LiDAR clouds from a ROS 2 bag without a ROS install (rosbags).

Times are the message header stamps in nanoseconds: the bags are assumed to be recorded on one clock
(or already shifted onto one), which is what "simultaneous recording" means here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite3

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class _Bag:
    """Raw messages of a bag directory (or a single .db3 / .mcap file).

    .db3 files are read with sqlite directly: rosbags matches the db's topics against metadata.yaml including the
    QoS text, and bags whose metadata was rewritten (QoS '') then yield no messages at all.
    """

    def __init__(self, path):
        p = Path(path)
        self.db3 = sorted(p.glob("*.db3")) if p.is_dir() else ([p] if p.suffix == ".db3" else [])
        self.path = p

    def topics(self) -> dict[str, tuple[str, int]]:
        if self.db3:
            out = {}
            for f in self.db3:
                c = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
                n = dict(c.execute("select topic_id, count(*) from messages group by topic_id"))
                for i, name, typ in c.execute("select id, name, type from topics"):
                    out[name] = (typ, out.get(name, (typ, 0))[1] + n.get(i, 0))
            return out
        with AnyReader([self.path], default_typestore=TYPESTORE) as r:
            return {c.topic: (c.msgtype, c.msgcount) for c in r.connections}

    def messages(self, wanted):
        """yield (topic, deserialised message) in storage-time order"""
        if self.db3:
            for f in self.db3:
                c = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
                tmap = {i: (n, t) for i, n, t in c.execute("select id, name, type from topics") if n in wanted}
                if not tmap:
                    continue
                q = f"select topic_id, data from messages where topic_id in ({','.join('?' * len(tmap))}) order by timestamp"
                for tid, raw in c.execute(q, tuple(tmap)):
                    name, typ = tmap[tid]
                    yield name, TYPESTORE.deserialize_cdr(raw, typ)
            return
        with AnyReader([self.path], default_typestore=TYPESTORE) as r:
            con = [x for x in r.connections if x.topic in wanted]
            for x, _, raw in r.messages(connections=con):
                yield x.topic, r.deserialize(raw, x.msgtype)


def topics(path) -> dict[str, tuple[str, int]]:
    return _Bag(path).topics()


def image_array(msg) -> np.ndarray:
    h, w, enc = msg.height, msg.width, msg.encoding
    buf = np.frombuffer(msg.data, np.uint8)
    if enc in ("16UC1", "mono16"):
        return buf.view(np.uint16).reshape(h, msg.step // 2)[:, :w]
    if enc in ("32FC1",):
        return (buf.view(np.float32).reshape(h, msg.step // 4)[:, :w] * 1000).astype(np.uint16)
    ch = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}[enc]
    img = buf.reshape(h, msg.step)[:, :w * ch].reshape(h, w, ch)
    if enc == "rgb8":
        img = img[..., ::-1]
    elif enc in ("bgra8", "rgba8"):
        img = img[..., :3] if enc == "bgra8" else img[..., 2::-1]
    return np.ascontiguousarray(img)


@dataclass
class CameraInfo:
    K: np.ndarray        # 3x3
    D: np.ndarray        # distortion (plumb_bob / k1 k2 p1 p2 k3)
    width: int
    height: int


class RGBDStream:
    """Colour + depth (registered to colour) of one camera. Frames are paired by identical or nearest stamp."""

    def __init__(self, path, color_topic="/camera/camera/color/image_raw",
                 depth_topic="/camera/camera/aligned_depth_to_color/image_raw",
                 info_topic="/camera/camera/color/camera_info", depth_scale=0.001):
        self.path, self.color_topic, self.depth_topic, self.info_topic = str(path), color_topic, depth_topic, info_topic
        self.depth_scale = depth_scale          # metres per depth unit
        self.bag = _Bag(path)
        names = self.bag.topics()
        for t in (color_topic, depth_topic, info_topic):
            if t not in names:
                raise SystemExit(f"{path}: topic {t} not found (have: {sorted(names)})")
        msg = next(self.bag.messages({info_topic}))[1]
        D = np.array(msg.d, np.float64)
        self.info = CameraInfo(np.array(msg.k, np.float64).reshape(3, 3), D if D.size else np.zeros(5), msg.width, msg.height)

    def stamps(self, topic=None) -> np.ndarray:
        """header stamps of a topic (depth by default), read without decoding the images"""
        topic = topic or self.depth_topic
        return np.array([stamp_ns(m) for _, m in self.bag.messages({topic})], np.int64)

    def frames(self, every_s=0.0, t_from=None, t_to=None, color=True):
        """yield (t_ns, bgr or None, depth_m float32) — at most one frame per `every_s` seconds"""
        want = [self.depth_topic] + ([self.color_topic] if color else [])
        pending: dict[int, dict] = {}
        last = -1 << 62
        for topic, msg in self.bag.messages(set(want)):
            t = stamp_ns(msg)
            if (t_from is not None and t < t_from) or (t_to is not None and t > t_to):
                continue
            if t - last < every_s * 1e9:
                continue
            slot = pending.setdefault(t, {})
            slot["d" if topic == self.depth_topic else "c"] = msg
            if "d" in slot and (not color or "c" in slot):
                pending.pop(t)
                depth = image_array(slot["d"]).astype(np.float32) * self.depth_scale
                yield t, (image_array(slot["c"]) if color else None), depth
                last = t
                for k in [k for k in pending if k < t]:      # drop halves that never got a partner
                    pending.pop(k)


def cloud_xyz(msg) -> tuple[np.ndarray, np.ndarray | None]:
    """PointCloud2 -> (N,3) float64 xyz and per-point time offset in seconds (if the cloud has one)"""
    f = {fl.name: fl for fl in msg.fields}
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, np.uint8).reshape(n, msg.point_step)
    xyz = np.stack([buf[:, f[k].offset:f[k].offset + 4].copy().view(np.float32).ravel() for k in "xyz"], 1).astype(np.float64)
    tt = None
    for name in ("time", "t", "timestamp"):
        if name in f:
            dt = {7: np.float32, 8: np.float64, 6: np.uint32}.get(f[name].datatype)
            if dt is not None:
                size = np.dtype(dt).itemsize
                tt = buf[:, f[name].offset:f[name].offset + size].copy().view(dt).ravel().astype(np.float64)
                if dt is np.uint32:
                    tt *= 1e-9
                if tt.size and tt.max() > 1e6:          # absolute stamps -> offsets
                    tt = tt - tt.min()
            break
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
    return xyz[ok], (tt[ok] if tt is not None else None)


class LidarStream:
    def __init__(self, path, topic="/velodyne_points"):
        self.path, self.topic = str(path), topic
        self.bag = _Bag(path)
        names = self.bag.topics()
        if topic not in names:
            raise SystemExit(f"{path}: topic {topic} not found (have: {sorted(names)})")

    def scans(self, t_from=None, t_to=None):
        """yield (t_ns, xyz (N,3), per-point time offset s or None)"""
        for _, msg in self.bag.messages({self.topic}):
            t = stamp_ns(msg)
            if (t_from is not None and t < t_from) or (t_to is not None and t > t_to):
                continue
            xyz, tt = cloud_xyz(msg)
            yield t, xyz, tt
