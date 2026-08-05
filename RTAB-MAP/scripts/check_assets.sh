#!/usr/bin/env bash
set -euo pipefail
bag="${1:-}"
rgb="${2:-/camera/color/image_raw}"
depth="${3:-/camera/aligned_depth_to_color/image_raw}"
camera_info="${4:-/camera/color/camera_info}"
[[ -n "$bag" && -d "$bag" ]] || { echo "ERROR: pass a ROS 2 RGB-D/LiDAR bag directory" >&2; exit 1; }
info="$(ros2 bag info "$bag")"
grep -Fq "Topic: $rgb | Type: sensor_msgs/msg/Image" <<<"$info" || { echo "ERROR: RGB Image topic missing: $rgb" >&2; exit 1; }
grep -Fq "Topic: $depth | Type: sensor_msgs/msg/Image" <<<"$info" || { echo "ERROR: depth Image topic missing: $depth" >&2; exit 1; }
grep -Fq "Topic: $camera_info | Type: sensor_msgs/msg/CameraInfo" <<<"$info" || { echo "ERROR: CameraInfo topic missing: $camera_info" >&2; exit 1; }
echo "RTAB-MAP input structure OK; confirm topic names and QoS in the launch command."
