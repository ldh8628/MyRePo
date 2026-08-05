#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 INPUT_ROS1_BAG OUTPUT_ROS2_BAG_DIR" >&2
  exit 2
fi

input_bag="$1"
output_dir="$2"

[[ -f "$input_bag" ]] || { echo "ERROR: ROS 1 bag not found: $input_bag" >&2; exit 1; }
command -v rosbags-convert >/dev/null || {
  echo "ERROR: rosbags-convert is unavailable. Create/activate environment.yml first." >&2
  exit 1
}
[[ ! -e "$output_dir" ]] || {
  echo "ERROR: destination already exists; choose a new path: $output_dir" >&2
  exit 1
}

rosbags-convert "$input_bag" --dst "$output_dir"

if command -v ros2 >/dev/null; then
  ros2 bag info "$output_dir"
else
  echo "Converted successfully. Activate ROS 2 before running ros2 bag info." >&2
fi
