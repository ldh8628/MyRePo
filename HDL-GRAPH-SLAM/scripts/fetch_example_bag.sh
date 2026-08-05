#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${1:-$root/hdlgraphslam_ws/data}"
url="https://www.aisl.cs.tut.ac.jp/databases/hdl_graph_slam/hdl_501_filtered.bag.tar.gz"
sha256="591f3361bc07e77531fab24710d27181a56121988fea135f2a668365ecb812dc"
mkdir -p "$dest"
curl --fail --location --retry 3 "$url" --output "$dest/hdl_501_filtered.bag.tar.gz"
printf '%s  %s\n' "$sha256" "$dest/hdl_501_filtered.bag.tar.gz" | sha256sum --check
echo "Downloaded official upstream ROS 1 example. Convert it to ROS 2 before using this bundle."
