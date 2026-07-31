#!/usr/bin/env python3
"""Portable entry point for the four supported FoundPose pipeline stages."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "gen_templates": ROOT / "scripts" / "gen_templates.py",
    "gen_repre": ROOT / "scripts" / "gen_repre.py",
    "infer": ROOT / "scripts" / "infer.py",
    "prepare_bop_submission": ROOT / "scripts" / "prepare_bop_submission.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a FoundPose stage with explicit input/output roots."
    )
    parser.add_argument(
        "--bop-path",
        type=Path,
        help="BOP dataset root. May also be supplied through BOP_PATH.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Writable output root. May also be supplied through BOP_OUTPUT_PATH.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument(
        "stage_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected stage, normally --opts-path FILE.",
    )
    return parser


def _resolve_root(cli_value: Path | None, env_name: str, option_name: str) -> Path:
    raw_value = cli_value if cli_value is not None else os.environ.get(env_name)
    if raw_value is None or not str(raw_value).strip():
        raise SystemExit(
            f"{env_name} is required; pass {option_name} or set {env_name}."
        )
    return Path(raw_value).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bop_path = None
    if args.command != "prepare_bop_submission":
        bop_path = _resolve_root(args.bop_path, "BOP_PATH", "--bop-path")
    output_path = _resolve_root(
        args.output_path, "BOP_OUTPUT_PATH", "--output-path"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    misplaced_root_options = {
        option
        for option in ("--bop-path", "--output-path")
        if option in args.stage_args
    }
    if misplaced_root_options:
        options = ", ".join(sorted(misplaced_root_options))
        raise SystemExit(
            f"Runner options must appear before the command name: {options}"
        )

    dependency_roots = (
        ROOT,
        ROOT / "external" / "bop_toolkit",
        ROOT / "external" / "dinov2",
    )
    if args.command != "prepare_bop_submission":
        assert bop_path is not None
        if not bop_path.is_dir():
            raise SystemExit(
                f"BOP input root does not exist or is not a directory: {bop_path}"
            )
        missing = [path for path in dependency_roots[1:] if not path.is_dir()]
        if missing:
            formatted = ", ".join(
                str(path.relative_to(ROOT)) for path in missing
            )
            raise SystemExit(
                f"Missing external dependencies: {formatted}. "
                "Run: bash scripts/setup_external.sh"
            )
        os.environ["BOP_PATH"] = str(bop_path)
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    for path in reversed(dependency_roots):
        sys.path.insert(0, str(path))
    os.environ["BOP_OUTPUT_PATH"] = str(output_path)

    script_path = COMMANDS[args.command]
    stage_args = args.stage_args[1:] if args.stage_args[:1] == ["--"] else args.stage_args
    sys.argv = [str(script_path), *stage_args]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
