#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_POINTS_TOPIC="${HDL_RAW_POINTS_TOPIC:-/velodyne_points}"
POINTS_TOPIC="${HDL_POINTS_TOPIC:-/filtered_points}"
FILTERED_POINTS_QOS="${HDL_FILTERED_POINTS_QOS:-reliable}"
DISTANCE_FAR_THRESH="${HDL_DISTANCE_FAR_THRESH:-20.0}"
ENABLE_FLOOR_DETECTION="${HDL_ENABLE_FLOOR:-true}"
LAUNCH_FILE="${HDL_LAUNCH_FILE:-hdl_graph_slam_501.launch.py}"
CONFIG_FILE="${HDL_CONFIG_FILE:-}"
if [[ -n "${HDL_RAW_POINTS_QOS:-}" ]]; then
  RAW_POINTS_QOS="$HDL_RAW_POINTS_QOS"
elif [[ "$LAUNCH_FILE" == *_tutorial.launch.py ]]; then
  RAW_POINTS_QOS="reliable"
else
  RAW_POINTS_QOS="best_effort"
fi
PREFILTER_DOWNSAMPLE_METHOD="${HDL_PREFILTER_DOWNSAMPLE_METHOD:-}"
PREFILTER_DOWNSAMPLE_RESOLUTION="${HDL_PREFILTER_DOWNSAMPLE_RESOLUTION:-}"
PREFILTER_OUTLIER_REMOVAL_METHOD="${HDL_PREFILTER_OUTLIER_REMOVAL_METHOD:-}"
SCAN_DOWNSAMPLE_METHOD="${HDL_SCAN_DOWNSAMPLE_METHOD:-}"
SCAN_DOWNSAMPLE_RESOLUTION="${HDL_SCAN_DOWNSAMPLE_RESOLUTION:-}"
SCAN_REG_NUM_THREADS="${HDL_SCAN_REG_NUM_THREADS:-}"
SCAN_REG_MAXIMUM_ITERATIONS="${HDL_SCAN_REG_MAXIMUM_ITERATIONS:-}"
SCAN_REG_MAX_OPTIMIZER_ITERATIONS="${HDL_SCAN_REG_MAX_OPTIMIZER_ITERATIONS:-}"
SCAN_REG_CORRESPONDENCE_RANDOMNESS="${HDL_SCAN_REG_CORRESPONDENCE_RANDOMNESS:-}"

set +u
if [[ "${HDL_SOURCE_OPT_ROS:-auto}" == "true" ]] || { [[ "${HDL_SOURCE_OPT_ROS:-auto}" == "auto" ]] && ! command -v ros2 >/dev/null 2>&1; }; then
  source /opt/ros/humble/setup.bash
fi
source "$ROOT_DIR/install/setup.bash"
set -u

if [[ "$LAUNCH_FILE" == "hdl_graph_slam.launch.py" ]]; then
  LAUNCH_ARGS=()
  [[ -n "$CONFIG_FILE" ]] && LAUNCH_ARGS+=(config:="$CONFIG_FILE")
else
  LAUNCH_ARGS=(
    use_sim_time:=true
    raw_points_topic:="$RAW_POINTS_TOPIC"
    points_topic:="$POINTS_TOPIC"
    raw_points_qos:="$RAW_POINTS_QOS"
    filtered_points_qos:="$FILTERED_POINTS_QOS"
    enable_floor_detection:="$ENABLE_FLOOR_DETECTION"
  )

  if [[ "$LAUNCH_FILE" == "hdl_graph_slam_501.launch.py" ]]; then
    LAUNCH_ARGS+=(distance_far_thresh:="$DISTANCE_FAR_THRESH")
  fi

  [[ -n "$PREFILTER_DOWNSAMPLE_METHOD" ]] && LAUNCH_ARGS+=(prefilter_downsample_method:="$PREFILTER_DOWNSAMPLE_METHOD")
  [[ -n "$PREFILTER_DOWNSAMPLE_RESOLUTION" ]] && LAUNCH_ARGS+=(prefilter_downsample_resolution:="$PREFILTER_DOWNSAMPLE_RESOLUTION")
  [[ -n "$PREFILTER_OUTLIER_REMOVAL_METHOD" ]] && LAUNCH_ARGS+=(prefilter_outlier_removal_method:="$PREFILTER_OUTLIER_REMOVAL_METHOD")
  [[ -n "$SCAN_DOWNSAMPLE_METHOD" ]] && LAUNCH_ARGS+=(scan_downsample_method:="$SCAN_DOWNSAMPLE_METHOD")
  [[ -n "$SCAN_DOWNSAMPLE_RESOLUTION" ]] && LAUNCH_ARGS+=(scan_downsample_resolution:="$SCAN_DOWNSAMPLE_RESOLUTION")
  [[ -n "$SCAN_REG_NUM_THREADS" ]] && LAUNCH_ARGS+=(scan_reg_num_threads:="$SCAN_REG_NUM_THREADS")
  [[ -n "$SCAN_REG_MAXIMUM_ITERATIONS" ]] && LAUNCH_ARGS+=(scan_reg_maximum_iterations:="$SCAN_REG_MAXIMUM_ITERATIONS")
  [[ -n "$SCAN_REG_MAX_OPTIMIZER_ITERATIONS" ]] && LAUNCH_ARGS+=(scan_reg_max_optimizer_iterations:="$SCAN_REG_MAX_OPTIMIZER_ITERATIONS")
  [[ -n "$SCAN_REG_CORRESPONDENCE_RANDOMNESS" ]] && LAUNCH_ARGS+=(scan_reg_correspondence_randomness:="$SCAN_REG_CORRESPONDENCE_RANDOMNESS")
fi

exec ros2 launch hdl_graph_slam "$LAUNCH_FILE" \
  "${LAUNCH_ARGS[@]}" \
  "$@"
