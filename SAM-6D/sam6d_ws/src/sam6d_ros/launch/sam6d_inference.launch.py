import os
import subprocess
import sys
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler, TimerAction
from launch import LaunchDescription
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _detect_conda_env():
    candidates = [
        os.environ.get("SAM6D_CONDA_PREFIX"),
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        os.environ.get("SAM6D_VENV"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        env_dir = os.path.expanduser(candidate)
        python = os.path.join(env_dir, "bin", "python3")
        if os.path.isfile(python) and os.path.isdir(os.path.join(env_dir, "conda-meta")):
            return env_dir
    raise RuntimeError(
        "sam6d_ros must be launched from a conda environment. "
        "Create environment.yml, activate sam6d-runtime, then source "
        "this workspace's install/setup.bash."
    )


def _site_packages(env_dir):
    python = os.path.join(env_dir, "bin", "python3")
    return subprocess.check_output(
        [
            python,
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        text=True,
    ).strip()


def _workspace_root():
    pkg_share = Path(get_package_share_directory("sam6d_ros")).resolve()
    for candidate in (pkg_share, *pkg_share.parents):
        if (candidate / "sam6d_master" / "SAM-6D").exists():
            return candidate
        if candidate.name == "install":
            return candidate.parent
    return Path.cwd()


def _resolve_config_path(config_value):
    path = Path(config_value).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        pkg_share = Path(get_package_share_directory("sam6d_ros"))
        workspace = _workspace_root()
        candidates.extend([
            Path.cwd() / path,
            workspace / path,
            pkg_share / path,
            pkg_share / "config" / path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"config file not found: {config_value} (searched: {searched})")


def _resolve_workspace_path(value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return _workspace_root() / path


def _load_bag_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("bag") or {}


def _node_param_file(config_path):
    with Path(config_path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    if not isinstance(cfg, dict) or "bag" not in cfg:
        return str(config_path)

    cfg = {key: value for key, value in cfg.items() if key != "bag"}
    runtime_dir = _workspace_root() / "output"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_param_path = runtime_dir / "sam6d_runtime_params.yaml"
    with runtime_param_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    return str(runtime_param_path)


def _launch_setup(context, *args, **kwargs):
    del args, kwargs
    config_path = _resolve_config_path(LaunchConfiguration("config").perform(context))
    entrypoint = os.path.join(
        get_package_prefix("sam6d_ros"),
        "lib",
        "sam6d_ros",
        "sam6d_inference_node",
    )
    conda_env = _detect_conda_env()
    conda_python = os.path.join(conda_env, "bin", "python3")
    conda_site = _site_packages(conda_env)

    pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = (
        f"{conda_site}:{pythonpath}" if pythonpath else conda_site
    )

    node = (
        Node(
            executable=conda_python,
            name="sam6d_inference_node",
            arguments=[entrypoint],
            parameters=[_node_param_file(config_path)],
            output="screen",
            additional_env={
                "PYTHONPATH": new_pythonpath,
                "PATH": f"{os.path.join(conda_env, 'bin')}:{os.environ.get('PATH', '')}",
                "SAM6D_CONDA_PREFIX": conda_env,
                "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/sam6d_mpl_config"),
                "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", "/tmp/sam6d_cache"),
                "YOLO_CONFIG_DIR": os.environ.get("YOLO_CONFIG_DIR", "/tmp/sam6d_ultralytics"),
            },
        )
    )

    actions = [node]
    bag = _load_bag_config(config_path)
    if str(bag.get("play", "false")).lower() in {"1", "true", "yes", "on"}:
        bag_path = _resolve_workspace_path(bag.get("path", ""))
        if not bag_path.exists():
            raise RuntimeError(f"bag.path does not exist: {bag_path}")
        play_cmd = ["ros2", "bag", "play", str(bag_path), "--rate", str(bag.get("rate", 1.0))]
        if str(bag.get("clock", "true")).lower() in {"1", "true", "yes", "on"}:
            play_cmd.append("--clock")
        bag_player = ExecuteProcess(cmd=play_cmd, output="screen")
        actions.append(TimerAction(period=float(bag.get("start_delay", 45.0)), actions=[bag_player]))
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=bag_player,
                    on_exit=[EmitEvent(event=Shutdown(reason="SAM-6D bag replay finished"))],
                )
            )
        )

    return actions


def generate_launch_description():
    default_cfg = os.path.join(
        get_package_share_directory("sam6d_ros"), "config", "params.yaml"
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value=default_cfg,
            description="SAM-6D inference parameter YAML file",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
