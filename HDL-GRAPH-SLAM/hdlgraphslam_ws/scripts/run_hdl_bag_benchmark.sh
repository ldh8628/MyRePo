#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIR INPUT_BAG [LAUNCH_FILE] [RATE] [-- extra launch args]" >&2
  exit 2
fi

OUTPUT_DIR="$1"
INPUT_BAG="$2"
LAUNCH_FILE="${3:-hdl_graph_slam_501.launch.py}"
RATE="${4:-0.5}"
shift $(( $# >= 4 ? 4 : $# ))

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"

set +u
if [[ "${HDL_SOURCE_OPT_ROS:-auto}" == "true" ]] || { [[ "${HDL_SOURCE_OPT_ROS:-auto}" == "auto" ]] && ! command -v ros2 >/dev/null 2>&1; }; then
  source /opt/ros/humble/setup.bash
fi
source "$ROOT_DIR/install/setup.bash"
set -u

# ros2 bag play publishes recorded PointCloud2 streams reliably by default.
# Keep live driver defaults in run_hdl_graph_slam.sh, but use reliable raw input
# for benchmark replay unless the caller explicitly overrides it.
export HDL_RAW_POINTS_QOS="${HDL_RAW_POINTS_QOS:-reliable}"

LAUNCH_PID=""
RECORD_PID=""

stop_pid() {
  local pid="$1"
  local label="$2"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  kill -INT "$pid" 2>/dev/null || true
  for _ in {1..10}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 1
  done

  echo "$label did not exit after SIGINT; sending SIGTERM" >&2
  kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..5}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 1
  done

  echo "$label did not exit after SIGTERM; sending SIGKILL" >&2
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  set +e
  stop_pid "$RECORD_PID" "ros2 bag record"
  stop_pid "$LAUNCH_PID" "ros2 launch"
}
trap cleanup EXIT

(
  cd "$ROOT_DIR"
  HDL_LAUNCH_FILE="$LAUNCH_FILE" bash scripts/run_hdl_graph_slam.sh "$@"
) >"$OUTPUT_DIR/logs/launch.log" 2>&1 &
LAUNCH_PID="$!"

sleep "${HDL_BENCH_LAUNCH_WAIT:-5}"

(
  cd "$ROOT_DIR"
  bash scripts/record_tiers_hdl_bag.sh "$(basename "$OUTPUT_DIR")" "$OUTPUT_DIR/hdl_recorded_bag"
) >"$OUTPUT_DIR/logs/record.log" 2>&1 &
RECORD_PID="$!"

sleep "${HDL_BENCH_RECORD_WAIT:-3}"

ros2 bag play "$INPUT_BAG" --clock --rate "$RATE" >"$OUTPUT_DIR/logs/play.log" 2>&1

sleep "${HDL_BENCH_POST_WAIT:-5}"
cleanup
trap - EXIT

(
  cd "$ROOT_DIR"
  scripts/summarize_hdl_bag.py "$OUTPUT_DIR/hdl_recorded_bag"
) >"$OUTPUT_DIR/summary.tsv"

(
  cd "$ROOT_DIR"
  scripts/summarize_hdl_bag.py --json "$OUTPUT_DIR/hdl_recorded_bag"
) >"$OUTPUT_DIR/summary.json"

cat "$OUTPUT_DIR/summary.tsv"
