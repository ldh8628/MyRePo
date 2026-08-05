#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundles=(ORB3-SLAM HDL-GRAPH-SLAM RTAB-MAP SAM-6D)
required=(
  ORB3-SLAM/environment.yml
  ORB3-SLAM/orbslam_ws/src/ORB_SLAM3/package.xml
  ORB3-SLAM/orbslam_ws/src/ORB_SLAM3/LICENSE
  ORB3-SLAM/orbslam_ws/src/orbslam3_ros2/package.xml
  HDL-GRAPH-SLAM/environment.yml
  HDL-GRAPH-SLAM/hdlgraphslam_ws/src/hdl_graph_slam/package.xml
  HDL-GRAPH-SLAM/hdlgraphslam_ws/src/fast_gicp/package.xml
  HDL-GRAPH-SLAM/hdlgraphslam_ws/src/ndt_omp/package.xml
  HDL-GRAPH-SLAM/hdlgraphslam_ws/scripts/run_hdl_graph_slam.sh
  HDL-GRAPH-SLAM/hdlgraphslam_ws/scripts/run_hdl_bag_benchmark.sh
  HDL-GRAPH-SLAM/hdlgraphslam_ws/scripts/record_tiers_hdl_bag.sh
  HDL-GRAPH-SLAM/hdlgraphslam_ws/scripts/summarize_hdl_bag.py
  HDL-GRAPH-SLAM/scripts/convert_ros1_bag.sh
  RTAB-MAP/environment.yml
  RTAB-MAP/LICENSES/GPL-3.0.txt
  RTAB-MAP/rtabmap_ws/src/rtabmap/package.xml
  RTAB-MAP/rtabmap_ws/src/rtabmap_ros/LICENSE
  RTAB-MAP/rtabmap_ws/src/rtabmap_ros/rtabmap_launch/package.xml
  SAM-6D/environment.yml
  SAM-6D/config/runtime_example.yaml
  SAM-6D/sam6d_ws/src/sam6d_ros/package.xml
  SAM-6D/sam6d_ws/src/sam6d_ros/LICENSE
)

fail=0
for bundle in "${bundles[@]}"; do
  for f in README.md .gitignore scripts/check_assets.sh; do
    [[ -f "$root/$bundle/$f" ]] || { echo "MISSING: $bundle/$f" >&2; fail=1; }
  done
done
for f in "${required[@]}"; do
  [[ -f "$root/$f" ]] || { echo "MISSING: $f" >&2; fail=1; }
done

while IFS= read -r -d '' script; do bash -n "$script"; done < <(find "$root" \( -path '*/.git' -o -path '*/ImprovedFoundPose' \) -prune -o -type f -name '*.sh' -print0)
python3 - "$root" <<'PY'
import ast
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1])
bundles = [root / n for n in ("ORB3-SLAM", "HDL-GRAPH-SLAM", "RTAB-MAP", "SAM-6D")]
for base in bundles:
    for path in base.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Third-party upstream READMEs may refer to example images intentionally
    # omitted from this runtime bundle; validate the bundle-facing README.
    for md in [base / "README.md"]:
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            target = target.split("#", 1)[0]
            if target and not (md.parent / target).resolve().exists():
                raise SystemExit(f"BROKEN LOCAL LINK: {md}: {target}")
PY

if find "${bundles[@]/#/$root/}" -type f -size +10M -print | grep -q .; then
  echo "ERROR: file larger than 10 MiB" >&2; fail=1
fi
if find "${bundles[@]/#/$root/}" -type d \( -name build -o -name install -o -name log -o -name output -o -name _bmad-output -o -name .pytest_cache -o -name __pycache__ \) -print | grep -q .; then
  echo "ERROR: forbidden generated directory" >&2; fail=1
fi
if find "${bundles[@]/#/$root/}" -type f \( -name '*.bag' -o -name '*.db3' -o -name '*.mcap' -o -name '*.pcd' -o -name '*.ply' -o -name '*.db' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.onnx' \) -print | grep -q .; then
  echo "ERROR: forbidden input/model/output asset" >&2; fail=1
fi

allowed_png='ORB3-SLAM/docs/results/top_view_trajectory.png|HDL-GRAPH-SLAM/docs/results/map_topdown_dense_hdl_only.png|RTAB-MAP/docs/results/rtabmap_2d_top_map_trajectory_start_end.png|SAM-6D/docs/results/vis_ism.png|SAM-6D/docs/results/vis_pem.png'
while IFS= read -r png; do
  rel="${png#"$root/"}"
  grep -Eq "^(${allowed_png})$" <<<"$rel" || { echo "ERROR: unapproved PNG: $rel" >&2; fail=1; }
done < <(find "${bundles[@]/#/$root/}" -type f -iname '*.png')

if grep -RInE --exclude='README.md' '/home/(etri|ldh[^/]*)/' "${bundles[@]/#/$root/}"; then
  echo "ERROR: local home path outside README provenance" >&2; fail=1
fi

if [[ "${VERIFY_REMOTE_LINKS:-0}" == 1 ]]; then
  python3 - "$root" <<'PY'
import pathlib, re, subprocess, sys
root = pathlib.Path(sys.argv[1])
urls = set()
for name in ("ORB3-SLAM", "HDL-GRAPH-SLAM", "RTAB-MAP", "SAM-6D"):
    md = root / name / "README.md"
    urls.update(re.findall(r"https?://[^)\s]+", md.read_text(encoding="utf-8")))
for url in sorted(urls):
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--head", "--output", "/dev/null", "--max-time", "20", url], check=True)
PY
fi

(( fail == 0 )) || exit 1
echo "SLAM bundle verification passed"
