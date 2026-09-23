from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_prospective_wrapper_can_reuse_completed_atomic_n10_read_only():
    source = (
        _repo_root()
        / "scripts"
        / "discovery"
        / "run_atomic_scientific_verifier_prospective_e2e.py"
    ).read_text(encoding="utf-8")

    assert '"--resume-completed-atomic-n10"' in source
    assert "args.resume_completed_atomic_n10" in source
    assert (
        'manifest["completed_atomic_n10_reused_read_only"] = True'
        in source
    )
    assert (
        'old_manifest.get("authority_mode") != "certification_only"'
        in source
    )
    assert 'old_manifest.get("source_portfolio", "")' in source
    assert 'old_manifest.get("source_query_plan", "")' in source
    assert (
        "--resume-completed-atomic-n10 requires"
        in source
    )
