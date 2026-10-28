"""mapping.py — LiDAR bag -> hdl_graph_slam 3D map -> 2D grid for Nav2.

hdl_graph_slam (the ROS 2 port in this repository's HDL-GRAPH-SLAM folder) runs in its own ROS domain while the bag
plays with --clock; then its save_map / dump_graph services write the point cloud map and the optimized keyframes,
and grid.py turns the keyframes into the occupancy grid. Everything of one map goes in maps/<name>/.
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import struct
import subprocess
import time
from pathlib import Path

import yaml

from .grid import make_grid

MAP_DOMAIN = os.environ.get("HUSKYROUTE_MAP_DOMAIN", "61")      # hdl + bag play; the planner uses another domain
CLOUD_TYPE = "sensor_msgs/msg/PointCloud2"


def bag_info(path):
    """topics, types, counts and duration from a rosbag2 folder's metadata.yaml"""
    p = Path(path)
    meta = p / "metadata.yaml"
    if not meta.is_file():
        raise ValueError(f"not a rosbag2 folder (no metadata.yaml): {path}")
    m = yaml.safe_load(open(meta))["rosbag2_bagfile_information"]
    topics = [{"name": t["topic_metadata"]["name"], "type": t["topic_metadata"]["type"], "count": t["message_count"]}
              for t in m["topics_with_message_count"]]
    return {"path": str(p), "storage": m.get("storage_identifier", ""), "duration_s": m["duration"]["nanoseconds"] / 1e9,
            "topics": topics, "clouds": [t["name"] for t in topics if t["type"] == CLOUD_TYPE]}


def cloud_frame(path, topic):
    """frame_id of the first message on `topic` (sqlite3 .db3 bags; None if it cannot be read)"""
    try:
        for db in sorted(Path(path).glob("*.db3")):
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            row = con.execute("SELECT m.data FROM messages m JOIN topics t ON m.topic_id = t.id WHERE t.name = ? "
                              "ORDER BY m.timestamp LIMIT 1", (topic,)).fetchone()
            con.close()
            if row:
                d = row[0]                       # CDR: 4-byte header, stamp (8), string length (4), frame_id
                n = struct.unpack_from("<I", d, 12)[0]
                return d[16:16 + n - 1].decode()
    except Exception:  # noqa: BLE001 - fall back to the launch default
        pass
    return None


def _call(srv, typ, req, env, timeout=120):
    r = subprocess.run(["ros2", "service", "call", srv, typ, req], env=env, capture_output=True, text=True, timeout=timeout)
    if "success=True" not in r.stdout:
        raise RuntimeError(f"{srv} failed:\n{r.stdout[-600:]}{r.stderr[-600:]}")


def build(bag, maps_dir, name, topic=None, rate=1.0, kf_trans=0.5, kf_angle=0.3, far=30.0, sensor_height=None,
          grid_opts=None, log=lambda frac, msg: print(msg), cancel=None):
    """run hdl_graph_slam over the bag and make maps/<name>/; returns info dict"""
    info = bag_info(bag)
    topic = topic or (info["clouds"][0] if info["clouds"] else None)
    if not topic or topic not in info["clouds"]:
        raise ValueError(f"no PointCloud2 topic {topic!r} in the bag (found: {info['clouds']})")
    frame = cloud_frame(bag, topic) or "velodyne"
    out = Path(maps_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "ROS_DOMAIN_ID": MAP_DOMAIN, "ROS_LOCALHOST_ONLY": "1"}
    qos = out / "qos_override.yaml"
    qos.write_text(f"{topic}:\n  reliability: reliable\n  durability: volatile\n  history: keep_last\n  depth: 10\n")
    log(0.01, f"bag {info['duration_s']:.0f} s, topic {topic} (frame {frame}), domain {MAP_DOMAIN}")

    hdl = subprocess.Popen(
        ["ros2", "launch", "hdl_graph_slam", "hdl_graph_slam_501.launch.py", "use_sim_time:=true",
         f"raw_points_topic:={topic}", "raw_points_qos:=reliable", f"lidar_frame_id:={frame}",
         f"graph_keyframe_delta_trans:={kf_trans}", f"graph_keyframe_delta_angle:={kf_angle}",
         f"distance_far_thresh:={far}"],
        env=env, stdout=open(out / "hdl.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)

    def stop(p):
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGINT)
                p.wait(8)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    play = None
    try:
        time.sleep(6)
        if hdl.poll() is not None:
            raise RuntimeError("hdl_graph_slam did not start:\n" + (out / "hdl.log").read_text()[-1500:])
        play = subprocess.Popen(["ros2", "bag", "play", str(bag), "--clock", "100", "--rate", str(rate),
                                 "--topics", topic, "--qos-profile-overrides-path", str(qos)],
                                env=env, stdout=open(out / "play.log", "w"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        t0, total = time.time(), info["duration_s"] / rate
        while play.poll() is None:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            if hdl.poll() is not None:
                raise RuntimeError("hdl_graph_slam stopped:\n" + (out / "hdl.log").read_text()[-1500:])
            el = time.time() - t0
            log(0.03 + 0.8 * min(el / total, 1.0), f"hdl_graph_slam: {el:.0f} / {total:.0f} s of the bag")
            time.sleep(1)
        log(0.84, "bag done; last graph optimization")
        time.sleep(5)
        _call("/hdl_graph_slam/save_map", "hdl_graph_slam/srv/SaveMap",
              f"{{utm: false, resolution: 0.05, destination: '{out / 'map.pcd'}'}}", env)
        dump = out / "dump"
        _call("/hdl_graph_slam/dump_graph", "hdl_graph_slam/srv/DumpGraph", f"{{destination: '{dump}'}}", env)
    finally:
        stop(play)
        stop(hdl)
    n_kf = len(list(dump.glob("[0-9]" * 6)))
    if n_kf < 3:
        raise RuntimeError(f"hdl_graph_slam made only {n_kf} keyframes: wrong topic/QoS, or the robot hardly moved")
    log(0.88, f"{n_kf} keyframes; making the 2D grid")
    s = make_grid(dump, str(out / name), sensor_height=sensor_height, log=lambda m: log(0.9, m), **(grid_opts or {}))
    meta = {"name": name, "bag": str(bag), "topic": topic, "frame": frame, "rate": rate, "kf_trans": kf_trans,
            "kf_angle": kf_angle, "far": far, "sensor_height": sensor_height, "grid_opts": grid_opts or {},
            "grid": s, "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(out / "info.json", "w"), indent=1, ensure_ascii=False)
    log(1.0, f"map {name}: {s['size'][0]}x{s['size'][1]} cells, driven {s['track_length']} m")
    return meta


def regrid(maps_dir, name, sensor_height=None, grid_opts=None, log=print):
    """remake the 2D grid of an existing map from its saved keyframes (no SLAM rerun)"""
    out = Path(maps_dir) / name
    meta = json.load(open(out / "info.json"))
    s = make_grid(out / "dump", str(out / name), sensor_height=sensor_height, log=log, **(grid_opts or {}))
    meta.update(sensor_height=sensor_height, grid_opts=grid_opts or {}, grid=s)
    json.dump(meta, open(out / "info.json", "w"), indent=1, ensure_ascii=False)
    return meta
