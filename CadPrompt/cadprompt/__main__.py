"""command line: python -m cadprompt <cad file> [--name ...] [--out results/<name>]

The same calculation as the web page (python -m cadprompt.server), without the page."""
import argparse
import json

from . import pipeline


def main():
    ap = argparse.ArgumentParser(description="YOLO-World / YOLOE prompt recommendation from a CAD model")
    ap.add_argument("cad", help=".obj (with .mtl/texture next to it) / .ply / .glb / .stl")
    ap.add_argument("--name", help="object name; default: words of the file name")
    ap.add_argument("--detectors", nargs="+", default=["yolo-world", "yoloe"], choices=["yolo-world", "yoloe"])
    ap.add_argument("--yolo-world-weights", default=pipeline.DEFAULT_WEIGHTS["yolo-world"])
    ap.add_argument("--yoloe-weights", default=pipeline.DEFAULT_WEIGHTS["yoloe"])
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--tests", type=int, default=48, help="test images (renders pasted on backgrounds)")
    ap.add_argument("--thr", type=float, default=0.1, help="detector confidence that counts as found")
    ap.add_argument("--backgrounds", help="folder of photos of the place the detector will work in")
    ap.add_argument("--vocab", help="noun list file for CLIP (one per line); default LVIS + Objects365 names")
    ap.add_argument("--extra", nargs="*", default=[], help="your own candidate prompts")
    ap.add_argument("--out", help="output folder (default results/<file name>)")
    a = ap.parse_args()
    from pathlib import Path
    for k in ("cad", "backgrounds", "vocab", "out"):                 # detectors run from weights/: keep paths absolute
        if getattr(a, k):
            setattr(a, k, str(Path(getattr(a, k)).resolve()))
    res = pipeline.run(a.cad, name=a.name, detectors=a.detectors, n_views=a.views, n_tests=a.tests, thr=a.thr,
                       weights={"yolo-world": a.yolo_world_weights, "yoloe": a.yoloe_weights},
                       backgrounds=a.backgrounds, extra_prompts=a.extra, vocab_file=a.vocab,
                       log=lambda stage, frac, msg: print(f"[{frac * 100:3.0f}%] {msg}", flush=True))
    out = pipeline.save(res, a.out or Path("results").resolve() / Path(a.cad).stem)
    print(json.dumps(res["recommended"], ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
