from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_zero_shift_mask_fix_is_present() -> None:
    source = (ROOT / "utils" / "infer_pose_util.py").read_text(encoding="utf-8")
    assert "y_end = -shift_y if shift_y > 0 else None" in source
    assert "x_end = -shift_x if shift_x > 0 else None" in source
    assert "mask_modal[shift_y:y_end, shift_x:x_end]" in source
    assert "mask_modal[shift_y:-shift_y, shift_x:-shift_x]" not in source


def test_matplotlib_rgba_fallback_is_present() -> None:
    source = (ROOT / "utils" / "vis_base_util.py").read_text(encoding="utf-8")
    assert 'hasattr(fig.canvas, "tostring_rgb")' in source
    assert "fig.canvas.buffer_rgba()" in source


def test_infer_has_no_research_or_onepose_hooks() -> None:
    source = (ROOT / "scripts" / "infer.py").read_text(encoding="utf-8").lower()
    forbidden = (
        "audit_infer_paths",
        "save_topk_hypotheses",
        "projection_sanity",
        "template_candidates_json",
        "onepose",
    )
    assert not [value for value in forbidden if value in source]
    assert "require_gt: bool = true" in source


def test_gt_free_and_representation_guards_are_present() -> None:
    source = (ROOT / "scripts" / "infer.py").read_text(encoding="utf-8")
    assert "opts.debug and object_pose_m2w_gt is not None" in source
    assert "base_repre_dir, opts.repre_version" in source
    assert "targets_per_obj.get(object_lid, [])" in source


def test_pnp_failure_guard_is_present() -> None:
    source = (ROOT / "utils" / "pnp_util.py").read_text(encoding="utf-8")
    assert "if not pose_est_success:" in source
    assert "return False, None, None, None, None" in source
