#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-indoor03}"
OUTPUT_BAG="${2:-$ROOT_DIR/output/tiers/$DATASET/hdl_recorded_bag}"

if [[ -e "$OUTPUT_BAG" ]]; then
  echo "Refusing to overwrite existing bag: $OUTPUT_BAG" >&2
  echo "Pass a different output path as the second argument." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_BAG")"

set +u
if [[ "${HDL_SOURCE_OPT_ROS:-auto}" == "true" ]] || { [[ "${HDL_SOURCE_OPT_ROS:-auto}" == "auto" ]] && ! command -v ros2 >/dev/null 2>&1; }; then
  source /opt/ros/humble/setup.bash
fi
source "$ROOT_DIR/install/setup.bash"
set -u

exec ros2 bag record \
  --output "$OUTPUT_BAG" \
  /odom \
  /hdl_graph_slam/map_points \
  /filtered_points \
  /velodyne_points \
  /prefiltering/debug \
  /scan_matching_odometry/debug \
  /floor_detection/floor_coeffs \
  /hdl_graph_slam/debug/floor_constraint_count \
  /hdl_graph_slam/debug/loop_count \
  /hdl_graph_slam/debug/keyframe_count \
  /hdl_graph_slam/debug/graph_vertices \
  /hdl_graph_slam/debug/graph_edges \
  /vrpn_client_node/UWBTest/pose \
  /tf \
  /tf_static \
  /clock
