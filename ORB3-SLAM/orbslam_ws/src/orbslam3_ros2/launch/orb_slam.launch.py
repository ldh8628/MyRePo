#!/usr/bin/env python3
"""Config-driven ORB-SLAM3 RGB-D launch."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown, matches_action
from launch.events.process import SignalProcess
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


def _workspace_root():
    pkg_share = Path(get_package_share_directory("orbslam3_ros2")).resolve()
    for candidate in (pkg_share, *pkg_share.parents):
        if (candidate / "src" / "ORB_SLAM3").exists():
            return candidate
        if candidate.name == "install":
            return candidate.parent
    return Path.cwd()


def _resolve_workspace_path(value, workspace):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return workspace / path


def _default_config():
    workspace = _workspace_root()
    return {
        "dataset": "",
        "bag": {
            "path": "",
            "play": True,
            "clock": True,
            "rate": 0.5,
            "start_delay": 12.0,
        },
        "orbslam3": {
            "vocabulary_path": str(workspace / "src" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"),
            "settings_path": "",
        },
        "topics": {
            "rgb": "/camera/camera/color/image_raw",
            "depth": "/camera/camera/aligned_depth_to_color/image_raw",
            "imu": "/camera/camera/imu",
        },
        "frames": {
            "world": "map",
            "camera": "camera_color_optical_frame",
        },
        "runtime": {
            "use_imu": False,
            "localization_mode": False,
            "sync_queue_size": 60,
            "publish_map_points": True,
            "map_points_topic": "/orbslam3/map_points",
            "map_points_publish_period_sec": 1.0,
            "save_map_points_on_shutdown": True,
        },
        "dense_map": {
            "enabled": True,
            "topic": "/orbslam3/dense_map",
            "publish_period_sec": 2.0,
            "frame_stride": 5,
            "pixel_stride": 4,
            "voxel_size": 0.03,
            "min_depth_m": 0.15,
            "max_depth_m": 6.0,
            "max_points": 2000000,
            "include_color": True,
            "save_pcd": True,
            "pcd_output_path": "orbslam3_dense_map.pcd",
            "save_ply": False,
            "ply_output_path": "orbslam3_dense_map.ply",
            # Ghosting fix: rebuild the saved dense map at shutdown using the
            # optimized (post loop-closure / BA) per-frame poses. Default ON.
            "reproject_optimized": True,
            "reproject_max_points": 80000000,
        },
        "output": {
            "dir": str(workspace / "output" / "orbslam3_config_run"),
            "map_points_path": "orbslam3_map_points.pcd",
        },
        "visualization": {
            "enabled": False,
            "python_executable": "/usr/bin/python3",
            "gt_pose_topic": "/vrpn_client_node/UWBTest/pose",
            "max_gt_match_dt": 0.2,
            "zoom_percentile": 99.0,
            "drop_zero_poses": True,
        },
    }


def _resolve_config_path(config_value):
    path = Path(config_value).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        pkg_share = Path(get_package_share_directory("orbslam3_ros2"))
        candidates.extend([
            Path.cwd() / path,
            pkg_share / path,
            pkg_share / "config" / path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"config file not found: {config_value} (searched: {searched})")


def _load_config(config_value):
    config_path = _resolve_config_path(config_value)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"config root must be a YAML mapping: {config_path}")
    return _deep_merge(_default_config(), loaded)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_path(value, base_dir):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_bag_path(cfg):
    bag_path = str(cfg["bag"]["path"]).strip()
    if not bag_path:
        return None
    return _resolve_workspace_path(bag_path, _workspace_root())


def _resolve_output_dir(cfg, bag_path):
    workspace = _workspace_root()
    value = str(cfg["output"]["dir"]).strip()
    if value.lower() not in {"", "auto"}:
        return _resolve_workspace_path(value, workspace)

    if _as_bool(cfg["bag"]["play"]) and bag_path is not None:
        resolved_bag = bag_path.resolve()
        run_name = resolved_bag.parent.name if resolved_bag.name == "bag" else resolved_bag.name
    else:
        run_name = str(cfg.get("dataset", "")).strip() or "live_d455f"
    return workspace / "output" / f"{run_name}_orbslam3"


def _validate_config(cfg):
    workspace = _workspace_root()
    if _as_bool(cfg["bag"]["play"]):
        bag_path = _resolve_bag_path(cfg)
        if bag_path is None:
            raise RuntimeError("missing config value: bag.path")
        if not bag_path.exists():
            raise RuntimeError(f"bag.path does not exist: {cfg['bag']['path']}")
    if not cfg["orbslam3"]["settings_path"]:
        raise RuntimeError("missing config value: orbslam3.settings_path")
    if not _resolve_workspace_path(cfg["orbslam3"]["settings_path"], workspace).exists():
        raise RuntimeError(f"orbslam3.settings_path does not exist: {cfg['orbslam3']['settings_path']}")
    if not _resolve_workspace_path(cfg["orbslam3"]["vocabulary_path"], workspace).exists():
        raise RuntimeError(f"orbslam3.vocabulary_path does not exist: {cfg['orbslam3']['vocabulary_path']}")


def _launch_setup(context, *args, **kwargs):
    del args, kwargs
    config_value = LaunchConfiguration("config").perform(context)
    if not config_value:
        raise RuntimeError("config launch argument is required")

    cfg = _load_config(config_value)
    _validate_config(cfg)

    bag_path = _resolve_bag_path(cfg)
    output_dir = _resolve_output_dir(cfg, bag_path)
    workspace = _workspace_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_points_path = _as_path(cfg["output"]["map_points_path"], output_dir)
    dense_map_pcd_path = _as_path(cfg["dense_map"]["pcd_output_path"], output_dir)
    dense_map_ply_path = _as_path(cfg["dense_map"]["ply_output_path"], output_dir)

    # Optional debug wrapper (e.g. gdb backtrace) injected via env var.
    node_prefix = os.environ.get("ORBSLAM_NODE_PREFIX", "").strip()

    rgbd_node = Node(
        package="orbslam3_ros2",
        executable="rgbd_node",
        name="orbslam3_rgbd_node",
        output="screen",
        cwd=str(output_dir),
        prefix=(node_prefix if node_prefix else None),
        parameters=[
            {
                "vocabulary_path": str(_resolve_workspace_path(cfg["orbslam3"]["vocabulary_path"], workspace)),
                "settings_path": str(_resolve_workspace_path(cfg["orbslam3"]["settings_path"], workspace)),
                "rgb_topic": str(cfg["topics"]["rgb"]),
                "depth_topic": str(cfg["topics"]["depth"]),
                "imu_topic": str(cfg["topics"].get("imu", "/camera/camera/imu")),
                "use_imu": _as_bool(cfg["runtime"].get("use_imu", False)),
                "localization_mode": _as_bool(cfg["runtime"].get("localization_mode", False)),
                "world_frame": str(cfg["frames"]["world"]),
                "camera_frame": str(cfg["frames"]["camera"]),
                "sync_queue_size": int(cfg["runtime"]["sync_queue_size"]),
                "publish_map_points": _as_bool(cfg["runtime"]["publish_map_points"]),
                "map_points_topic": str(cfg["runtime"]["map_points_topic"]),
                "map_points_publish_period_sec": float(cfg["runtime"]["map_points_publish_period_sec"]),
                "save_map_points_on_shutdown": _as_bool(cfg["runtime"]["save_map_points_on_shutdown"]),
                "map_points_output_path": str(map_points_path),
                "dense_map_enabled": _as_bool(cfg["dense_map"]["enabled"]),
                "dense_map_topic": str(cfg["dense_map"]["topic"]),
                "dense_map_publish_period_sec": float(cfg["dense_map"]["publish_period_sec"]),
                "dense_map_frame_stride": int(cfg["dense_map"]["frame_stride"]),
                "dense_map_pixel_stride": int(cfg["dense_map"]["pixel_stride"]),
                "dense_map_voxel_size": float(cfg["dense_map"]["voxel_size"]),
                "dense_map_min_depth_m": float(cfg["dense_map"]["min_depth_m"]),
                "dense_map_max_depth_m": float(cfg["dense_map"]["max_depth_m"]),
                "dense_map_max_points": int(cfg["dense_map"]["max_points"]),
                "dense_map_include_color": _as_bool(cfg["dense_map"]["include_color"]),
                "dense_map_save_pcd": _as_bool(cfg["dense_map"]["save_pcd"]),
                "dense_map_pcd_output_path": str(dense_map_pcd_path),
                "dense_map_save_ply": _as_bool(cfg["dense_map"]["save_ply"]),
                "dense_map_ply_output_path": str(dense_map_ply_path),
                "dense_map_reproject_optimized": _as_bool(
                    cfg["dense_map"].get("reproject_optimized", True)),
                "dense_map_reproject_max_points": int(
                    cfg["dense_map"].get("reproject_max_points", 80000000)),
            }
        ],
    )

    actions = [rgbd_node]

    if _as_bool(cfg["bag"]["play"]):
        if bag_path is None:
            raise RuntimeError("missing config value: bag.path")
        play_cmd = [
            "ros2",
            "bag",
            "play",
            str(bag_path),
            "--rate",
            str(cfg["bag"]["rate"]),
            # Large read-ahead avoids "message queue starved" bursts that delay
            # frames/IMU and expose a timing race in the ORB-SLAM3 core threads
            # (intermittent SIGSEGV in IMU-RGBD mode under load).
            "--read-ahead-queue-size",
            str(int(cfg["bag"].get("read_ahead_queue_size", 5000))),
        ]
        if _as_bool(cfg["bag"]["clock"]):
            play_cmd.append("--clock")
        bag_player = ExecuteProcess(cmd=play_cmd, output="screen")
        actions.append(TimerAction(period=float(cfg["bag"]["start_delay"]), actions=[bag_player]))
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=bag_player,
                    on_exit=[
                        EmitEvent(
                            event=SignalProcess(
                                signal_number=signal.SIGINT,
                                process_matcher=matches_action(rgbd_node),
                            )
                        )
                    ],
                )
            )
        )

    if _as_bool(cfg["visualization"]["enabled"]):
        pkg_share = get_package_share_directory("orbslam3_ros2")
        plot_script = os.path.join(pkg_share, "scripts", "plot_orbslam3_tiers_result.py")
        plot_result = ExecuteProcess(
            cmd=[
                str(cfg["visualization"]["python_executable"]),
                plot_script,
                "--trajectory",
                str(output_dir / "CameraTrajectory.txt"),
                "--bag",
                str(bag_path),
                "--output-dir",
                str(output_dir),
                "--map-points",
                str(map_points_path),
                "--dense-map-points",
                str(dense_map_pcd_path),
                "--gt-pose-topic",
                str(cfg["visualization"]["gt_pose_topic"]),
                "--max-gt-match-dt",
                str(cfg["visualization"]["max_gt_match_dt"]),
                "--zoom-percentile",
                str(cfg["visualization"]["zoom_percentile"]),
                "--drop-zero-poses",
                str(cfg["visualization"]["drop_zero_poses"]).lower(),
            ],
            cwd=str(output_dir),
            output="screen",
        )
        actions.append(
            RegisterEventHandler(
                OnProcessExit(target_action=rgbd_node, on_exit=[TimerAction(period=1.0, actions=[plot_result])])
            )
        )
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=plot_result,
                    on_exit=[EmitEvent(event=Shutdown(reason="ORB-SLAM3 visualization saved"))],
                )
            )
        )
    elif _as_bool(cfg["bag"]["play"]):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=rgbd_node,
                    on_exit=[EmitEvent(event=Shutdown(reason="ORB-SLAM3 run finished"))],
                )
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value="", description="YAML config file for ORB-SLAM3 RGB-D"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
