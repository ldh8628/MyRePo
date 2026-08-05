from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_entrypoint_help_without_external_dependencies() -> None:
    for path in (
        "run_pipeline.py",
        "scripts/prepare_bop_submission.py",
        "tools/convert_ply_gs2mesh_to_foundpose.py",
    ):
        completed = _run(path, "--help")
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()


def test_runner_requires_explicit_data_roots() -> None:
    completed = _run("run_pipeline.py", "infer")
    assert completed.returncode != 0
    assert "BOP_PATH is required" in completed.stderr


def test_runner_rejects_empty_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOP_PATH", "")
    monkeypatch.setenv("BOP_OUTPUT_PATH", "")
    completed = _run("run_pipeline.py", "infer")
    assert completed.returncode != 0
    assert "BOP_PATH is required" in completed.stderr


def test_submission_converter(tmp_path: Path) -> None:
    inference_dir = tmp_path / "inference" / "lmo_v1" / "1"
    inference_dir.mkdir(parents=True)
    estimate = {
        "scene_id": 1,
        "img_id": 2,
        "obj_id": 1,
        "score": 0.75,
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [[10], [20], [30]],
        "cnos_time": 0.2,
        "time": {"prep": 0.1, "pose": 0.3},
    }
    (inference_dir / "estimated-poses.json").write_text(
        json.dumps([estimate]), encoding="utf-8"
    )
    completed = _run(
        "scripts/prepare_bop_submission.py",
        "--dataset",
        "lmo",
        "--version",
        "v1",
        "--output-root",
        str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    output_path = tmp_path / "inference" / "lmo_v1" / "coarse_lmo-estimated-poses.csv"
    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["R"] == "1 0 0 0 1 0 0 0 1"
    assert rows[0]["t"] == "10 20 30"
    assert float(rows[0]["time"]) == pytest.approx(0.6)

    wrapped = _run(
        "run_pipeline.py",
        "--output-path",
        str(tmp_path),
        "prepare_bop_submission",
        "--",
        "--dataset",
        "lmo",
        "--version",
        "v1",
    )
    assert wrapped.returncode == 0, wrapped.stderr


def test_submission_converter_rejects_duplicate_ids(tmp_path: Path) -> None:
    inference_dir = tmp_path / "inference" / "lmo_v1" / "1"
    inference_dir.mkdir(parents=True)
    (inference_dir / "estimated-poses.json").write_text("[]", encoding="utf-8")
    completed = _run(
        "scripts/prepare_bop_submission.py",
        "--dataset",
        "lmo",
        "--version",
        "v1",
        "--output-root",
        str(tmp_path),
        "--object-ids",
        "1",
        "1",
    )
    assert completed.returncode != 0
    assert "must not contain duplicates" in completed.stderr


def test_submission_converter_rejects_empty_results(tmp_path: Path) -> None:
    inference_dir = tmp_path / "inference" / "lmo_v1" / "1"
    inference_dir.mkdir(parents=True)
    (inference_dir / "estimated-poses.json").write_text("[]", encoding="utf-8")
    completed = _run(
        "scripts/prepare_bop_submission.py",
        "--dataset",
        "lmo",
        "--version",
        "v1",
        "--output-root",
        str(tmp_path),
    )
    assert completed.returncode != 0
    assert "No pose estimates found" in completed.stderr
