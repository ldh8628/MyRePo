from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_release_files_exist() -> None:
    required = {
        "README.md",
        "NOTICE.md",
        "SHARE_MANIFEST.md",
        "LICENSE",
        "run_pipeline.py",
        "conda_foundpose_gpu.yaml",
        "scripts/gen_templates.py",
        "scripts/gen_repre.py",
        "scripts/infer.py",
        "scripts/prepare_bop_submission.py",
        "scripts/setup_external.sh",
        "scripts/validate_release.sh",
        "tools/convert_ply_gs2mesh_to_foundpose.py",
        "configs/gen_templates/lmo.json",
        "configs/gen_repre/lmo.json",
        "configs/infer/lmo.json",
    }
    missing = sorted(path for path in required if not (ROOT / path).is_file())
    assert not missing


def test_python_sources_parse() -> None:
    paths = [
        ROOT / "run_pipeline.py",
        *(ROOT / "scripts").glob("*.py"),
        *(ROOT / "tools").glob("*.py"),
        *(ROOT / "utils").glob("*.py"),
        *(ROOT / "tests").glob("*.py"),
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_runtime_local_imports_are_present() -> None:
    modules = {path.stem for path in (ROOT / "utils").glob("*.py")}
    missing = set()
    for path in [*(ROOT / "scripts").glob("*.py"), *(ROOT / "utils").glob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "utils":
                for alias in node.names:
                    if alias.name not in modules:
                        missing.add(alias.name)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("utils.")
            ):
                module = node.module.split(".", 1)[1]
                if module not in modules:
                    missing.add(module)
    assert not missing


def test_example_config_keys_match_runtime_option_types() -> None:
    cases = (
        ("scripts/gen_templates.py", "GenTemplatesOpts", "configs/gen_templates/lmo.json", "gen_templates_opts"),
        ("scripts/gen_repre.py", "GenRepreOpts", "configs/gen_repre/lmo.json", "gen_repre_opts"),
        ("scripts/infer.py", "InferOpts", "configs/infer/lmo.json", "infer_opts"),
    )
    for source_path, class_name, config_path, root_key in cases:
        tree = ast.parse((ROOT / source_path).read_text(encoding="utf-8"))
        option_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        fields = {
            node.target.id
            for node in option_class.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        }
        config = json.loads((ROOT / config_path).read_text(encoding="utf-8"))
        assert root_key in config
        unknown = set(config[root_key]) - fields
        assert not unknown, f"{config_path}: unknown option keys {sorted(unknown)}"


def test_no_forbidden_directories_or_large_files() -> None:
    forbidden = {
        "_bmad",
        "_bmad-output",
        "3DGS",
        "GS2Mesh",
        "research_pose",
        "select_topk",
        "outputs",
        "datasets",
    }
    violations = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_dir()
        and "external" not in path.relative_to(ROOT).parts
        and path.name in forbidden
    ]
    large = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file()
        and "external" not in path.relative_to(ROOT).parts
        and path.stat().st_size >= 100 * 1024 * 1024
    ]
    assert not violations
    assert not large


def test_no_internal_absolute_paths_in_runtime() -> None:
    pattern = re.compile(
        r"/home/(?:etri|[^/\\s]+)|/Users/[^/\\s]+|[A-Za-z]:\\\\Users\\\\"
    )
    checked = [
        ROOT / "run_pipeline.py",
        *list((ROOT / "scripts").glob("*")),
        *list((ROOT / "tools").glob("*")),
        *list((ROOT / "utils").glob("*.py")),
        *list((ROOT / "configs").rglob("*.json")),
    ]
    hits = [
        path.relative_to(ROOT)
        for path in checked
        if path.is_file()
        and path.name != "validate_release.sh"
        and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not hits
