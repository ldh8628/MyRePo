#!/usr/bin/env python3
"""Summarize HDL benchmark ROS 2 bags.

The script reads sqlite3 rosbag2 storage directly for message counts. If ROS 2
Python message support is sourced, it also reports final debug counter values
and callback timing from HDL debug topics.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any


TOPICS = [
    "/velodyne_points",
    "/filtered_points",
    "/prefiltering/debug",
    "/scan_matching_odometry/debug",
    "/odom",
    "/floor_detection/floor_coeffs",
    "/hdl_graph_slam/debug/keyframe_count",
    "/hdl_graph_slam/debug/floor_constraint_count",
    "/hdl_graph_slam/debug/loop_count",
    "/hdl_graph_slam/debug/graph_vertices",
    "/hdl_graph_slam/debug/graph_edges",
    "/hdl_graph_slam/map_points",
]

INT32_DEBUG_TOPICS = {
    "/hdl_graph_slam/debug/keyframe_count": "keyframe_count",
    "/hdl_graph_slam/debug/floor_constraint_count": "floor_constraint_count",
    "/hdl_graph_slam/debug/loop_count": "loop_count",
    "/hdl_graph_slam/debug/graph_vertices": "graph_vertices",
    "/hdl_graph_slam/debug/graph_edges": "graph_edges",
}


def find_db3(bag_dir: Path) -> Path:
    if bag_dir.is_file() and bag_dir.suffix == ".db3":
        return bag_dir

    matches = sorted(bag_dir.glob("*.db3"))
    if not matches:
        raise FileNotFoundError(f"no sqlite3 rosbag file found in {bag_dir}")
    return matches[0]


def topic_metadata(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute("select id, name, type from topics").fetchall()
    return {name: {"id": topic_id, "type": msg_type} for topic_id, name, msg_type in rows}


def topic_counts(conn: sqlite3.Connection, topics: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, meta in topics.items():
        count = conn.execute(
            "select count(*) from messages where topic_id = ?",
            (meta["id"],),
        ).fetchone()[0]
        counts[name] = int(count)
    return counts


def try_import_ros() -> tuple[Any, Any] | tuple[None, None]:
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception:
        return None, None
    return deserialize_message, get_message


def read_messages(
    conn: sqlite3.Connection,
    topic_id: int,
    msg_type: Any,
    deserialize_message: Any,
) -> list[Any]:
    rows = conn.execute(
        "select data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    ).fetchall()
    return [deserialize_message(data, msg_type) for (data,) in rows]


def timing_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "avg_ms": round(statistics.fmean(values), 3),
        "max_ms": round(max(values), 3),
    }


def debug_summary(conn: sqlite3.Connection, topics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deserialize_message, get_message = try_import_ros()
    if deserialize_message is None or get_message is None:
        return {"debug_decode": "unavailable; source ROS 2 and the workspace for debug values"}

    summary: dict[str, Any] = {}
    decode_errors: dict[str, str] = {}

    for topic, key in INT32_DEBUG_TOPICS.items():
        if topic not in topics:
            continue
        try:
            msg_type = get_message(topics[topic]["type"])
            messages = read_messages(conn, topics[topic]["id"], msg_type, deserialize_message)
            if messages:
                summary[key] = int(messages[-1].data)
        except Exception as exc:
            decode_errors[topic] = str(exc)

    topic = "/prefiltering/debug"
    if topic in topics:
        try:
            msg_type = get_message(topics[topic]["type"])
            messages = read_messages(conn, topics[topic]["id"], msg_type, deserialize_message)
            if messages:
                summary["prefilter"] = {
                    "last_output_count": int(messages[-1].output_count),
                    "last_input_points": int(messages[-1].input_point_count),
                    "last_output_points": int(messages[-1].output_point_count),
                    "callback": timing_stats([float(msg.callback_time_ms) for msg in messages]),
                }
        except Exception as exc:
            decode_errors[topic] = str(exc)

    topic = "/scan_matching_odometry/debug"
    if topic in topics:
        try:
            msg_type = get_message(topics[topic]["type"])
            messages = read_messages(conn, topics[topic]["id"], msg_type, deserialize_message)
            if messages:
                registrations = [
                    float(msg.registration_time_ms)
                    for msg in messages
                    if bool(msg.registration_triggered)
                ]
                summary["scan_matching"] = {
                    "last_odom_count": int(messages[-1].odom_count),
                    "last_input_points": int(messages[-1].input_point_count),
                    "last_downsampled_points": int(messages[-1].downsampled_point_count),
                    "callback": timing_stats([float(msg.callback_time_ms) for msg in messages]),
                    "registration": timing_stats(registrations),
                    "last_matching_error": round(float(messages[-1].matching_error), 6),
                    "last_has_converged": bool(messages[-1].has_converged),
                }
        except Exception as exc:
            decode_errors[topic] = str(exc)

    if decode_errors:
        summary["decode_errors"] = decode_errors

    return summary


def summarize_bag(bag_dir: Path) -> dict[str, Any]:
    db_path = find_db3(bag_dir)
    with sqlite3.connect(db_path) as conn:
        topics = topic_metadata(conn)
        counts = topic_counts(conn, topics)
        return {
            "bag": str(bag_dir),
            "db3": str(db_path),
            "counts": {topic: counts.get(topic, 0) for topic in TOPICS},
            "debug": debug_summary(conn, topics),
        }


def print_table(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "bag",
        "raw",
        "filtered",
        "prefilter_dbg",
        "scan_dbg",
        "odom",
        "keyframes",
        "floor",
        "loops",
    ]
    print("\t".join(headers))
    for item in summaries:
        counts = item["counts"]
        debug = item["debug"]
        print(
            "\t".join(
                [
                    item["bag"],
                    str(counts.get("/velodyne_points", 0)),
                    str(counts.get("/filtered_points", 0)),
                    str(counts.get("/prefiltering/debug", 0)),
                    str(counts.get("/scan_matching_odometry/debug", 0)),
                    str(counts.get("/odom", 0)),
                    str(debug.get("keyframe_count", counts.get("/hdl_graph_slam/debug/keyframe_count", 0))),
                    str(debug.get("floor_constraint_count", counts.get("/hdl_graph_slam/debug/floor_constraint_count", 0))),
                    str(debug.get("loop_count", counts.get("/hdl_graph_slam/debug/loop_count", 0))),
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="emit full JSON summary")
    args = parser.parse_args()

    summaries = [summarize_bag(path) for path in args.bags]
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        print_table(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
