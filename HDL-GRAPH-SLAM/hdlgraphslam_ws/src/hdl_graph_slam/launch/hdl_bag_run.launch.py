#!/usr/bin/env python3
"""Config-driven HDL Graph SLAM bag replay and result recording."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration


def _deep_merge(base, override):
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _workspace_root():
    pkg_share = Path(get_package_share_directory("hdl_graph_slam")).resolve()
    for candidate in (pkg_share, *pkg_share.parents):
        if (candidate / "scripts" / "run_hdl_bag_benchmark.sh").exists():
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
    return {
        "bag": {
            "path": "",
            "rate": 0.5,
        },
        "hdl_graph_slam": {
            "launch_file": "hdl_graph_slam_501.launch.py",
            "raw_points_qos": "reliable",
        },
        "output": {
            "dir": "auto",
        },
    }


def _resolve_config_path(config_value):
    path = Path(config_value).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        pkg_share = Path(get_package_share_directory("hdl_graph_slam"))
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

    cfg = _deep_merge(_default_config(), loaded)
    if not cfg["bag"].get("path") and "input" in cfg and isinstance(cfg["input"], dict) and cfg["input"].get("bag_path"):
        cfg["bag"]["path"] = cfg["input"]["bag_path"]
    return cfg


def _resolve_bag_path(cfg):
    return _resolve_workspace_path(cfg["bag"]["path"], _workspace_root())


def _resolve_output_dir(cfg, bag_path):
    workspace = _workspace_root()
    value = str(cfg["output"]["dir"]).strip()
    if value.lower() not in {"", "auto"}:
        return _resolve_workspace_path(value, workspace)

    resolved_bag = bag_path.resolve()
    run_name = resolved_bag.parent.name if resolved_bag.name == "bag" else resolved_bag.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return workspace / "output" / f"{run_name}_hdl_graph_slam_{stamp}"


def _find_benchmark_script():
    workspace = _workspace_root()
    candidates = [workspace / "scripts" / "run_hdl_bag_benchmark.sh"]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "scripts" / "run_hdl_bag_benchmark.sh")
    candidates.extend([
        Path.cwd() / "scripts" / "run_hdl_bag_benchmark.sh",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"benchmark script not found: run_hdl_bag_benchmark.sh (searched: {searched})")


def _validate_config(cfg):
    if not cfg["bag"]["path"]:
        raise RuntimeError("missing config value: bag.path")
    if not _resolve_bag_path(cfg).exists():
        raise RuntimeError(f"bag.path does not exist: {cfg['bag']['path']}")
    if not cfg["hdl_graph_slam"]["launch_file"]:
        raise RuntimeError("missing config value: hdl_graph_slam.launch_file")


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
    benchmark_script = _find_benchmark_script()

    env = {}
    raw_points_qos = str(cfg["hdl_graph_slam"].get("raw_points_qos") or "").strip()
    if raw_points_qos:
        env["HDL_RAW_POINTS_QOS"] = raw_points_qos

    run_benchmark = ExecuteProcess(
        cmd=[
            "bash",
            str(benchmark_script),
            str(output_dir),
            str(bag_path),
            str(cfg["hdl_graph_slam"]["launch_file"]),
            str(cfg["bag"]["rate"]),
        ],
        cwd=str(workspace),
        additional_env=env,
        output="screen",
    )

    return [
        LogInfo(msg=f"HDL Graph SLAM output directory: {output_dir}"),
        run_benchmark,
        RegisterEventHandler(
            OnProcessExit(
                target_action=run_benchmark,
                on_exit=[EmitEvent(event=Shutdown(reason="HDL Graph SLAM bag run finished"))],
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value="", description="YAML config file for HDL bag replay"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
