#!/usr/bin/env bash
set -euo pipefail
bag="${1:-}"
[[ -n "$bag" && -d "$bag" ]] || { echo "ERROR: pass a ROS 2 bag directory" >&2; exit 1; }
info="$(ros2 bag info "$bag")"
grep -Fq 'Topic: /velodyne_points | Type: sensor_msgs/msg/PointCloud2' <<<"$info" || { echo "ERROR: bag lacks /velodyne_points as sensor_msgs/msg/PointCloud2" >&2; exit 1; }
echo "HDL-GRAPH-SLAM input OK"
