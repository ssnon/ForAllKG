import json
from types import SimpleNamespace

from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _resolve_stage775_seed_root,
)
from scripts.discovery.run_scientific_portfolio_closed_loop_shadow import (
    _empty_gen1_seed_generation_report,
)


def test_empty_gen1_resolves_to_preserved_adaptive_seed(tmp_path):
    (tmp_path / "adaptive_seed").mkdir()
    (tmp_path / "closed_loop.summary.json").write_text(
        json.dumps({"status": "COMPLETE_EMPTY_GEN1"}),
        encoding="utf-8",
    )
    root, mode = _resolve_stage775_seed_root(tmp_path)
    assert root == tmp_path / "adaptive_seed"
    assert mode == "STAGE_775_GEN0_EMPTY_GEN1_FALLBACK"


def test_nonempty_stage775_keeps_normal_gen1_seed(tmp_path):
    (tmp_path / "closed_loop.summary.json").write_text(
        json.dumps({"status": "COMPLETE"}),
        encoding="utf-8",
    )
    root, mode = _resolve_stage775_seed_root(tmp_path)
    assert root == tmp_path
    assert mode == "STAGE_775_GEN1"


def test_empty_gen1_seed_report_preserves_source_identity_without_generation():
    portfolio = SimpleNamespace(
        portfolio_id="p0",
        hypotheses=[
            SimpleNamespace(hypothesis_id="h1"),
            SimpleNamespace(hypothesis_id="h2"),
        ],
    )
    report = _empty_gen1_seed_generation_report(portfolio)
    assert report["empty_gen1_fallback_seed"] is True
    assert [r["source_hypothesis_id"] for r in report["records"]] == ["h1", "h2"]
    assert all(r["generated_hypothesis_id"] is None for r in report["records"])
    assert report["generation_authority_created"] is False
