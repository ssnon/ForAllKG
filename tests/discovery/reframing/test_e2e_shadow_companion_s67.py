from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.reframing.authoritative_final_binding import (
    AuthoritativeFinalPortfolioResolution,
)
from pipeline_core.discovery.reframing.e2e_shadow_companion import (
    build_attach_command,
    build_scientific_shadow_companion_plan,
    require_completed_e2e_run,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _portfolio(path: Path, *, count: int = 1) -> Path:
    rows = [{"hypothesis_id": f"h:{i}"} for i in range(count)]
    return _write(path, {"hypotheses": rows})


def _inputs(tmp_path: Path):
    reframing = _write(tmp_path / "scientific_reframing_reasoning_portfolio.json", {})
    shadow = _write(tmp_path / "scientific_reframing_shadow.json", {})
    return reframing, shadow


def test_incomplete_e2e_fails_closed(tmp_path: Path):
    _write(tmp_path / "e2e_runner.manifest.json", {"status": "running"})
    with pytest.raises(ValueError, match="completed legacy E2E"):
        require_completed_e2e_run(tmp_path)


def test_complete_no_hypotheses_abstains_without_authority_resolution(tmp_path: Path):
    _write(
        tmp_path / "e2e_runner.manifest.json",
        {"status": "complete_no_hypotheses_after_alpha4"},
    )
    reframing, shadow = _inputs(tmp_path)
    plan = build_scientific_shadow_companion_plan(
        run_dir=tmp_path,
        reframing_portfolio_path=reframing,
        reframe_shadow_path=shadow,
    )
    assert plan.disposition == "abstained_no_authoritative_hypotheses"
    assert plan.authoritative_final is None


def test_missing_reframing_artifact_fails_closed(tmp_path: Path):
    _write(tmp_path / "e2e_runner.manifest.json", {"status": "complete"})
    _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json")
    with pytest.raises(ValueError, match="reframing reasoning portfolio"):
        build_scientific_shadow_companion_plan(
            run_dir=tmp_path,
            reframing_portfolio_path=tmp_path / "missing.json",
            reframe_shadow_path=tmp_path / "missing-shadow.json",
        )


def test_attach_command_does_not_mutate_legacy_authority_flags(tmp_path: Path, monkeypatch):
    _write(tmp_path / "e2e_runner.manifest.json", {"status": "complete"})
    _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json")
    reframing, shadow = _inputs(tmp_path)

    resolution = AuthoritativeFinalPortfolioResolution(
        run_dir=str(tmp_path.resolve()),
        portfolio_path=str((tmp_path / "novelty_refinement_a6.portfolio.json").resolve()),
        portfolio_sha256="a" * 64,
        authority_kind="filesystem_alpha6_refined",
        hypothesis_count=1,
    )
    monkeypatch.setattr(
        "pipeline_core.discovery.reframing.e2e_shadow_companion.resolve_authoritative_final_portfolio",
        lambda **_: resolution,
    )
    plan = build_scientific_shadow_companion_plan(
        run_dir=tmp_path,
        reframing_portfolio_path=reframing,
        reframe_shadow_path=shadow,
        max_syntheses=2,
    )
    command = build_attach_command(plan, model="model")
    flat = " ".join(command)
    assert "build_authoritative_integrated_scientific_shadow" in flat
    assert "--max-syntheses 2" in flat
    assert plan.production_selection_changed is False
    assert plan.canonical_graph_mutated is False
    assert plan.legacy_e2e_manifest_mutated is False


def test_optional_operator_paths_are_only_forwarded_when_present(tmp_path: Path, monkeypatch):
    _write(tmp_path / "e2e_runner.manifest.json", {"status": "complete"})
    _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json")
    reframing, shadow = _inputs(tmp_path)
    proxy = _write(tmp_path / "proxy.json", {})
    contradiction = _write(tmp_path / "contradiction.json", {})

    resolution = AuthoritativeFinalPortfolioResolution(
        run_dir=str(tmp_path.resolve()),
        portfolio_path=str((tmp_path / "novelty_refinement_a6.portfolio.json").resolve()),
        portfolio_sha256="a" * 64,
        authority_kind="filesystem_alpha6_refined",
        hypothesis_count=1,
    )
    monkeypatch.setattr(
        "pipeline_core.discovery.reframing.e2e_shadow_companion.resolve_authoritative_final_portfolio",
        lambda **_: resolution,
    )
    plan = build_scientific_shadow_companion_plan(
        run_dir=tmp_path,
        reframing_portfolio_path=reframing,
        reframe_shadow_path=shadow,
        proxy_shadow_path=proxy,
        contradiction_shadow_path=contradiction,
    )
    flat = " ".join(build_attach_command(plan))
    assert "--proxy-shadow" in flat
    assert "--contradiction-shadow" in flat
