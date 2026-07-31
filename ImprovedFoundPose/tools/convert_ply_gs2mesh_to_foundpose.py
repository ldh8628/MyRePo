#!/usr/bin/env python3
"""Convert a triangle PLY into a centered, millimetre-scale BOP model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scale and center a reconstructed triangle mesh for FoundPose. "
            "This does not infer or correct the source coordinate-system rotation."
        )
    )
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obj-id", type=int, required=True)
    parser.add_argument("--scale-to-mm", type=float, default=1000.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _mesh_diameter(vertices: np.ndarray, block_size: int = 1024) -> float:
    """Return the exact maximum pairwise vertex distance with bounded memory."""
    max_squared = 0.0
    for start_a in range(0, len(vertices), block_size):
        block_a = vertices[start_a : start_a + block_size]
        for start_b in range(start_a, len(vertices), block_size):
            block_b = vertices[start_b : start_b + block_size]
            delta = block_a[:, None, :] - block_b[None, :, :]
            squared = np.einsum("...i,...i->...", delta, delta)
            max_squared = max(max_squared, float(np.max(squared)))
    return float(np.sqrt(max_squared))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_ply = args.source_ply.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_ply.is_file():
        raise SystemExit(f"Source PLY does not exist: {source_ply}")
    if args.obj_id <= 0:
        raise SystemExit("--obj-id must be a positive integer")
    if args.scale_to_mm <= 0:
        raise SystemExit("--scale-to-mm must be positive")

    model_path = output_dir / f"obj_{args.obj_id:06d}.ply"
    models_info_path = output_dir / "models_info.json"
    transform_path = output_dir / f"obj_{args.obj_id:06d}_transform.json"
    existing = [path for path in (model_path, transform_path) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Output exists; pass --overwrite to replace these files: "
            + ", ".join(str(path) for path in existing)
        )

    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "open3d is required for PLY conversion: python -m pip install open3d"
        ) from exc

    mesh = o3d.io.read_triangle_mesh(str(source_ply))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    if vertices.size == 0 or triangles.size == 0:
        raise SystemExit("Source PLY must contain vertices and triangle faces")
    if not np.isfinite(vertices).all():
        raise SystemExit("Source PLY contains NaN or Inf vertex coordinates")

    scaled = vertices * float(args.scale_to_mm)
    if not np.isfinite(scaled).all():
        raise SystemExit("Scaled vertices are non-finite")
    center = 0.5 * (scaled.min(axis=0) + scaled.max(axis=0))
    centered = scaled - center
    mesh.vertices = o3d.utility.Vector3dVector(centered)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise SystemExit("No valid triangle faces remain after mesh cleanup")

    final_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    try:
        convex_hull, _ = mesh.compute_convex_hull()
        diameter_vertices = np.asarray(convex_hull.vertices, dtype=np.float64)
    except RuntimeError:
        diameter_vertices = final_vertices
    bbox_min = final_vertices.min(axis=0)
    bbox_max = final_vertices.max(axis=0)
    extent = bbox_max - bbox_min
    info = {
        "diameter": _mesh_diameter(diameter_vertices),
        "min_x": float(bbox_min[0]),
        "min_y": float(bbox_min[1]),
        "min_z": float(bbox_min[2]),
        "size_x": float(extent[0]),
        "size_y": float(extent[1]),
        "size_z": float(extent[2]),
    }
    models_info = {}
    if models_info_path.is_file():
        models_info = json.loads(models_info_path.read_text(encoding="utf-8"))
    models_info[str(args.obj_id)] = info
    transform = {
        "source_ply": str(source_ply),
        "output_ply": str(model_path),
        "scale_to_mm": float(args.scale_to_mm),
        "center_subtracted_mm": center.tolist(),
        "operation": "output = source * scale_to_mm - center_subtracted_mm",
        "canonical_rotation_applied": False,
        "coordinate_limit": (
            "Only scale and bounding-box centering are applied. The caller must "
            "validate and, if necessary, rotate the source into the intended object frame."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(model_path), mesh, write_ascii=False):
        raise SystemExit(f"Failed to write converted PLY: {model_path}")
    models_info_path.write_text(
        json.dumps(models_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    transform_path.write_text(
        json.dumps(transform, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Converted model: {model_path}")
    print(f"Model metadata: {models_info_path}")
    print(f"Coordinate transform: {transform_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
