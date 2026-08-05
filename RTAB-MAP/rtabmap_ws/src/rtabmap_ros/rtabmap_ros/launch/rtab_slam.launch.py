#!/usr/bin/env python3

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _deep_merge(base, override):
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _bool_text(value):
    return "true" if _as_bool(value) else "false"


def _default_config():
    workspace = Path.home() / "rtabmap_ws"
    return {
        "dataset": "",
        "use_sim_time": True,
        "bag": {
            "path": "",
            "play": True,
            "clock": True,
            "rate": 0.5,
            "start_delay": 5.0,
        },
        "input": {
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "camera_intrinsics_json": "",
            "frame_id": "camera_color_optical_frame",
        },
        "sync": {
            "approx_sync": True,
            "approx_rgbd_sync": True,
            "max_interval_sec": 0.05,
            "queue_size": 60,
            "topic_queue_size": 60,
            "qos": 1,
        },
        "rtabmap": {
            "visual_odometry": True,
            "icp_odometry": False,
            "rgbd_sync": True,
            "rtabmap_viz": False,
            "rviz": False,
            "delete_db_on_start": True,
            "args": "",
            "odom_args": "",
            "namespace": "rtabmap",
            "map_frame_id": "map",
            "odom_topic": "odom",
            "subscribe_scan_cloud": False,
            "scan_cloud_topic": "/velodyne_points",
        },
        "output": {
            "dir": "auto",
            "database": "",
            "map_topic": "/rtabmap/cloud_map",
            "pose_topic": "/rtabmap/pose",
        },
        "publishers": {
            "map_assembler": True,
            "tf_to_pose": True,
        },
    }


def _resolve_config_path(config_value):
    path = Path(config_value).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        pkg_share = Path(get_package_share_directory("rtabmap_ros"))
        candidates.extend(
            [
                Path.cwd() / path,
                pkg_share / path,
                pkg_share / "config" / path,
                Path.home() / "rtabmap_ws" / "config" / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"config file not found: {config_value} (searched: {searched})")


def _required_string(section, key, config):
    try:
        value = config[section][key]
    except KeyError as exc:
        raise RuntimeError(f"Missing required config key: {section}.{key}") from exc
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Config key {section}.{key} must be a non-empty string")
    return value


def _load_config(config_value):
    config_path = _resolve_config_path(config_value)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"config root must be a YAML mapping: {config_path}")

    config = _deep_merge(_default_config(), loaded)
    _required_string("input", "rgb_topic", config)
    _required_string("input", "depth_topic", config)
    _required_string("input", "frame_id", config)

    camera_info_topic = str(config["input"].get("camera_info_topic", "")).strip()
    intrinsics_json = str(config["input"].get("camera_intrinsics_json", "")).strip()
    if not camera_info_topic and not intrinsics_json:
        raise RuntimeError(
            "Set either input.camera_info_topic or input.camera_intrinsics_json"
        )
    if intrinsics_json and not Path(intrinsics_json).expanduser().exists():
        raise RuntimeError(f"input.camera_intrinsics_json does not exist: {intrinsics_json}")

    if _as_bool(config["bag"]["play"]):
        bag_path_text = str(config["bag"]["path"]).strip()
        if not bag_path_text:
            raise RuntimeError("missing config value: bag.path")
        bag_path = Path(bag_path_text).expanduser()
        if not bag_path.exists():
            raise RuntimeError(f"bag.path does not exist: {bag_path}")

    max_interval_sec = float(config["sync"]["max_interval_sec"])
    if max_interval_sec < 0.0:
        raise RuntimeError("sync.max_interval_sec must be >= 0")

    return config


def _resolve_output_dir(config):
    value = str(config["output"]["dir"]).strip()
    if value.lower() not in ("", "auto"):
        return Path(value).expanduser()

    bag_path_text = str(config["bag"]["path"]).strip()
    if _as_bool(config["bag"]["play"]) and bag_path_text:
        bag_path = Path(bag_path_text).expanduser()
        resolved_bag = bag_path.resolve()
        run_name = resolved_bag.parent.name if resolved_bag.name == "bag" else resolved_bag.name
    else:
        run_name = str(config.get("dataset", "rtabmap")).strip() or "rtabmap"
    return Path.home() / "rtabmap_ws" / "output" / f"{run_name}_rtabmap"


def _launch_setup(context, *args, **kwargs):
    del args, kwargs
    config_value = LaunchConfiguration("config").perform(context)
    config = _load_config(config_value)

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    database_config = str(config["output"].get("database", "")).strip()
    database_path = (
        Path(database_config).expanduser() if database_config else output_dir / "rtabmap.db"
    )
    database_path.parent.mkdir(parents=True, exist_ok=True)

    rtabmap_launch_file = (
        Path(get_package_share_directory("rtabmap_launch"))
        / "launch"
        / "rtabmap.launch.py"
    )

    camera_info_topic = str(config["input"].get("camera_info_topic", "")).strip()
    use_json_camera_info = not camera_info_topic
    camera_info_topic_for_rtabmap = (
        "camera_info_from_json" if use_json_camera_info else camera_info_topic
    )

    rtabmap_args = str(config["rtabmap"].get("args", "")).strip()
    if _as_bool(config["rtabmap"]["delete_db_on_start"]):
        rtabmap_args = f"{rtabmap_args} --delete_db_on_start".strip()

    sync_queue = str(config["sync"]["queue_size"])
    topic_queue = str(config["sync"]["topic_queue_size"])
    max_interval = str(float(config["sync"]["max_interval_sec"]))
    use_sim_time = _as_bool(config["use_sim_time"])

    rtabmap_arguments = {
        "use_sim_time": _bool_text(use_sim_time),
        "rtabmap_viz": _bool_text(config["rtabmap"]["rtabmap_viz"]),
        "rviz": _bool_text(config["rtabmap"]["rviz"]),
        "visual_odometry": _bool_text(config["rtabmap"]["visual_odometry"]),
        "icp_odometry": _bool_text(config["rtabmap"]["icp_odometry"]),
        "stereo": "false",
        "depth": "true",
        "rgbd_sync": _bool_text(config["rtabmap"]["rgbd_sync"]),
        "subscribe_rgbd": "false",
        "rgb_topic": config["input"]["rgb_topic"],
        "depth_topic": config["input"]["depth_topic"],
        "camera_info_topic": camera_info_topic_for_rtabmap,
        "odom_topic": config["rtabmap"]["odom_topic"],
        "frame_id": config["input"]["frame_id"],
        "map_frame_id": config["rtabmap"]["map_frame_id"],
        "namespace": config["rtabmap"]["namespace"],
        "approx_sync": _bool_text(config["sync"]["approx_sync"]),
        "approx_rgbd_sync": _bool_text(config["sync"]["approx_rgbd_sync"]),
        "approx_sync_max_interval": max_interval,
        "queue_size": sync_queue,
        "topic_queue_size": topic_queue,
        "qos": str(config["sync"]["qos"]),
        "database_path": str(database_path),
        "subscribe_scan_cloud": _bool_text(config["rtabmap"]["subscribe_scan_cloud"]),
        "scan_cloud_topic": config["rtabmap"]["scan_cloud_topic"],
        "args": rtabmap_args,
        "rtabmap_args": rtabmap_args,
        "odom_args": str(config["rtabmap"].get("odom_args", "")),
    }

    actions = []

    if use_json_camera_info:
        actions.append(
            Node(
                package="rtabmap_util",
                executable="json_to_camera_info.py",
                name="json_to_camera_info",
                namespace=config["rtabmap"]["namespace"],
                output="screen",
                parameters=[
                    {
                        "json_path": config["input"]["camera_intrinsics_json"],
                        "frame_id": config["input"]["frame_id"],
                        "use_sim_time": use_sim_time,
                    }
                ],
                remappings=[
                    ("image", config["input"]["rgb_topic"]),
                    ("camera_info", camera_info_topic_for_rtabmap),
                ],
            )
        )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(rtabmap_launch_file)),
            launch_arguments=rtabmap_arguments.items(),
        )
    )

    if _as_bool(config["publishers"]["map_assembler"]):
        actions.append(
            Node(
                package="rtabmap_util",
                executable="map_assembler",
                name="map_assembler",
                namespace=config["rtabmap"]["namespace"],
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
                remappings=[
                    ("mapData", "mapData"),
                    ("cloud_map", config["output"]["map_topic"]),
                ],
            )
        )

    if _as_bool(config["publishers"]["tf_to_pose"]):
        actions.append(
            Node(
                package="rtabmap_util",
                executable="tf_to_pose.py",
                name="tf_to_pose",
                namespace=config["rtabmap"]["namespace"],
                output="screen",
                parameters=[
                    {
                        "target_frame": config["rtabmap"]["map_frame_id"],
                        "source_frame": config["input"]["frame_id"],
                        "rate": 10.0,
                        "use_sim_time": use_sim_time,
                    }
                ],
                remappings=[("pose", config["output"]["pose_topic"])],
            )
        )

    if _as_bool(config["bag"]["play"]):
        play_cmd = [
            "ros2",
            "bag",
            "play",
            str(Path(str(config["bag"]["path"])).expanduser()),
            "--rate",
            str(config["bag"]["rate"]),
        ]
        if _as_bool(config["bag"]["clock"]):
            play_cmd.append("--clock")

        bag_player = ExecuteProcess(cmd=play_cmd, output="screen")
        actions.append(
            TimerAction(period=float(config["bag"]["start_delay"]), actions=[bag_player])
        )
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=bag_player,
                    on_exit=[EmitEvent(event=Shutdown(reason="bag playback finished"))],
                )
            )
        )

    return actions


def generate_launch_description():
    default_config = (
        Path(get_package_share_directory("rtabmap_ros")) / "config" / "config.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(default_config),
                description="Path or package config name for RTAB-Map RGB-D SLAM.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
