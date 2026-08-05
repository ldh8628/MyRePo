#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vocab="$root/orbslam_ws/src/ORB_SLAM3/Vocabulary/ORBvoc.txt"
config="${1:-}"
bag="${2:-}"
[[ -f "$vocab" ]] || { echo "ERROR: vocabulary missing; run scripts/fetch_orb_vocabulary.sh" >&2; exit 1; }
[[ -n "$config" && -f "$config" ]] || { echo "ERROR: pass an ORB-SLAM3 camera settings YAML as argument 1" >&2; exit 1; }
[[ -n "$bag" && -d "$bag" ]] || { echo "ERROR: pass a ROS 2 bag directory as argument 2" >&2; exit 1; }
info="$(ros2 bag info "$bag")"
grep -Fq 'Topic: /camera/camera/color/image_raw | Type: sensor_msgs/msg/Image' <<<"$info" || { echo "ERROR: bag lacks default RGB Image topic; update no_cli_rgbd.yaml or provide a compatible bag" >&2; exit 1; }
grep -Fq 'Topic: /camera/camera/aligned_depth_to_color/image_raw | Type: sensor_msgs/msg/Image' <<<"$info" || { echo "ERROR: bag lacks default aligned-depth Image topic; update no_cli_rgbd.yaml or provide a compatible bag" >&2; exit 1; }
echo "ORB3-SLAM assets OK"
