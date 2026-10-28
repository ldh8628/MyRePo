"""command line: python -m extcalib --a <bag> --a-type lidar|rgbd --b <bag> [--out result.json]

The same calculation as the web page (python -m extcalib.server), without the page."""
import argparse
import json

from . import pipeline


def main():
    ap = argparse.ArgumentParser(description="extrinsic between two sensors of one simultaneous recording")
    ap.add_argument("--a", required=True, help="bag folder of sensor a (the reference frame)")
    ap.add_argument("--a-type", choices=["lidar", "rgbd"], required=True)
    ap.add_argument("--b", required=True, help="bag folder of sensor b (RGB-D)")
    ap.add_argument("--points-topic", default="/velodyne_points", help="sensor a PointCloud2 topic (LiDAR)")
    ap.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw",
                    help="depth registered to the colour image")
    ap.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    ap.add_argument("--depth-scale", type=float, default=0.001, help="metres per depth unit")
    ap.add_argument("--out", default="extrinsic.json")
    a = ap.parse_args()
    cam = {"color_topic": a.color_topic, "depth_topic": a.depth_topic, "info_topic": a.info_topic, "depth_scale": a.depth_scale}
    sa = {"type": a.a_type, "path": a.a, **({"points_topic": a.points_topic} if a.a_type == "lidar" else cam)}
    sb = {"type": "rgbd", "path": a.b, **cam}
    res = pipeline.run(sa, sb, log=lambda stage, frac, msg: print(f"[{frac * 100:3.0f}%] {msg}", flush=True))
    pipeline.save(res, a.out)
    print(json.dumps({k: res[k] for k in ("translation_m", "rpy_deg", "tau_s", "warnings")}, ensure_ascii=False))
    print("->", a.out)


if __name__ == "__main__":
    main()
