#!/usr/bin/env python3
"""Convert FoundPose estimated-pose JSON files to BOP 2019 CSV format."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="FoundPose output root; defaults to BOP_OUTPUT_PATH.",
    )
    parser.add_argument("--object-ids", type=int, nargs="+")
    parser.add_argument("--output-csv", type=Path)
    return parser


def _matrix_values(value: Iterable[Iterable[float]]) -> str:
    return " ".join(str(number) for row in value for number in row)


def _vector_values(value: Iterable[object]) -> str:
    flattened = []
    for item in value:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return " ".join(str(number) for number in flattened)


def _validated_estimate(estimate: dict, source: Path) -> dict:
    required = ("scene_id", "img_id", "obj_id", "score", "R", "t", "time")
    missing = [key for key in required if key not in estimate]
    if missing:
        raise SystemExit(f"Missing fields {missing} in estimate from {source}")
    try:
        rotation = [float(value) for row in estimate["R"] for value in row]
        translation = [
            float(value)
            for item in estimate["t"]
            for value in (item if isinstance(item, list) else [item])
        ]
        numeric_values = [
            *rotation,
            *translation,
            float(estimate["score"]),
            *(float(value) for value in estimate["time"].values()),
        ]
    except (TypeError, ValueError, AttributeError) as exc:
        raise SystemExit(f"Malformed estimate in {source}: {exc}") from exc
    if len(rotation) != 9 or len(translation) != 3:
        raise SystemExit(f"Invalid R/t shape in estimate from {source}")
    if not all(math.isfinite(value) for value in numeric_values):
        raise SystemExit(f"Non-finite pose, score, or time in {source}")
    return estimate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root_raw = args.output_root or os.environ.get("BOP_OUTPUT_PATH")
    if output_root_raw is None:
        raise SystemExit("--output-root or BOP_OUTPUT_PATH is required")
    output_root = Path(output_root_raw).expanduser().resolve()
    inference_dir = output_root / "inference" / f"{args.dataset}_{args.version}"
    if not inference_dir.is_dir():
        raise SystemExit(f"Inference directory does not exist: {inference_dir}")

    object_ids = args.object_ids
    if object_ids is None:
        object_ids = sorted(
            int(path.name)
            for path in inference_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    if not object_ids:
        raise SystemExit(f"No object result directories found under: {inference_dir}")
    if len(object_ids) != len(set(object_ids)):
        raise SystemExit("--object-ids must not contain duplicates")

    records = []
    detection_times = {}
    stage_times = defaultdict(float)
    for object_id in object_ids:
        result_path = inference_dir / str(object_id) / "estimated-poses.json"
        if not result_path.is_file():
            raise SystemExit(f"Missing estimate file: {result_path}")
        with result_path.open(encoding="utf-8") as handle:
            estimates = json.load(handle)
        if not isinstance(estimates, list):
            raise SystemExit(f"Expected a list of estimates: {result_path}")
        for estimate in estimates:
            estimate = _validated_estimate(estimate, result_path)
            key = (estimate["scene_id"], estimate["img_id"])
            detection_times[key] = float(estimate.get("cnos_time", 0.0))
            stage_times[key] += sum(float(value) for value in estimate["time"].values())
            records.append(estimate)
    if not records:
        raise SystemExit("No pose estimates found in the selected object results")

    output_csv = args.output_csv or inference_dir / (
        f"coarse_{args.dataset}-estimated-poses.csv"
    )
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("scene_id", "im_id", "obj_id", "score", "R", "t", "time"))
        for estimate in records:
            key = (estimate["scene_id"], estimate["img_id"])
            writer.writerow(
                (
                    estimate["scene_id"],
                    estimate["img_id"],
                    estimate["obj_id"],
                    estimate["score"],
                    _matrix_values(estimate["R"]),
                    _vector_values(estimate["t"]),
                    stage_times[key] + detection_times.get(key, 0.0),
                )
            )
    print(f"Wrote {len(records)} pose estimates to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
