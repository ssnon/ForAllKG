from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.reframing.authoritative_final_binding import (
    build_authoritative_shadow_commands,
    resolve_authoritative_final_portfolio,
    validate_authoritative_final_context,
)


def _portfolio(path: Path, *, context_id: str = "ctx:1", count: int = 1) -> Path:
    hypotheses = [
        {
            "schema_version": "hypothesis-card-v1",
            "hypothesis_id": f"hypothesis:{index}",
            "domain_profile_id": "sers_au_ag",
            "source_context_id": context_id,
            "source_context_sha256": "c" * 64,
            "source_report_id": "report:1",
            "source_report_sha256": "r" * 64,
            "title": f"H{index}",
            "hypothesis_statement": "X is conditionally related to Y.",
            "hypothesis_type": "context_dependency",
            "premise_statement_ids": ["p1"],
            "gap_statement_ids": ["g1"],
            "inferential_bridge": "Grounded conditional extension.",
            "predicted_observations": [
                {
                    "observation_id": f"prediction:{index}",
                    "observable": "response",
                    "expected_direction": "unspecified",
                    "rationale": "Conditional response.",
                }
            ],
            "falsification_criteria": [
                {
                    "criterion_id": f"falsifier:{index}",
                    "observable": "response",
                    "falsifying_outcome": "No conditional response.",
                }
            ],
            "assumptions": [],
            "source_paper_ids": ["paper:1"],
            "gap_paper_ids": [],
            "cross_paper_synthesis": False,
            "candidate_dependency": "none",
            "evidence_profile": {
                "premise_count": 1,
                "gap_count": 1,
                "source_paper_count": 1,
                "candidate_premise_count": 0,
                "reported_premise_count": 1,
                "synthesis_premise_count": 0,
            },
            "status": "hypothesized",
            "novelty_status": "not_assessed",
        }
        for index in range(count)
    ]
    payload = {
        "schema_version": "hypothesis-portfolio-v1",
        "portfolio_id": "portfolio:1",
        "domain_profile_id": "sers_au_ag",
        "source_context_id": context_id,
        "source_context_sha256": "c" * 64,
        "source_report_id": "report:1",
        "source_report_sha256": "r" * 64,
        "hypotheses": hypotheses,
        "abstention_reason": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    # Ensure the fixture tracks the installed contract.
    HypothesisPortfolio.model_validate_json(path.read_text(encoding="utf-8"))
    return path


def _manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_bounded_n10_has_highest_authority(tmp_path: Path):
    alpha6 = _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json")
    bounded = _portfolio(tmp_path / "novelty_refinement_a6.n10.bounded.portfolio.json")
    _manifest(
        tmp_path / "e2e_runner.manifest.json",
        {
            "status": "complete",
            "n10_bounded_continuation": {
                "enabled": True,
                "final_portfolio": str(bounded),
            },
            "post_generation_n10_authority": {
                "downstream_portfolio": str(alpha6),
            },
        },
    )
    result = resolve_authoritative_final_portfolio(run_dir=tmp_path)
    assert result.authority_kind == "manifest_bounded_n10"
    assert Path(result.portfolio_path) == bounded.resolve()


def test_manifest_post_n10_is_used_when_no_bounded_continuation(tmp_path: Path):
    n10 = _portfolio(tmp_path / "novelty_refinement_a6.n10.portfolio.json")
    _manifest(
        tmp_path / "e2e_runner.manifest.json",
        {
            "status": "complete",
            "n10_bounded_continuation": {"enabled": False},
            "post_generation_n10_authority": {"downstream_portfolio": str(n10)},
        },
    )
    result = resolve_authoritative_final_portfolio(run_dir=tmp_path)
    assert result.authority_kind == "manifest_post_n10"


def test_manifest_falls_back_only_to_alpha6_refined_not_alpha4(tmp_path: Path):
    _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json")
    _portfolio(tmp_path / "hypothesis_axis_a4.portfolio.json")
    _manifest(tmp_path / "e2e_runner.manifest.json", {"status": "complete"})
    result = resolve_authoritative_final_portfolio(run_dir=tmp_path)
    assert result.authority_kind == "manifest_alpha6_refined"


def test_filesystem_prefers_bounded_over_other_final_artifacts(tmp_path: Path):
    _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json")
    _portfolio(tmp_path / "novelty_refinement_a6.n10.portfolio.json")
    bounded = _portfolio(tmp_path / "novelty_refinement_a6.n10.bounded.portfolio.json")
    result = resolve_authoritative_final_portfolio(run_dir=tmp_path)
    assert result.authority_kind == "filesystem_bounded_n10"
    assert Path(result.portfolio_path) == bounded.resolve()


def test_filesystem_recognizes_certification_candidate_portfolio(tmp_path: Path):
    _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json")
    candidate = _portfolio(tmp_path / "novelty_refinement_a6.n10.candidate.portfolio.json")
    result = resolve_authoritative_final_portfolio(run_dir=tmp_path)
    assert result.authority_kind == "filesystem_n10_candidate"
    assert Path(result.portfolio_path) == candidate.resolve()


def test_filesystem_recognizes_hard_filter_n10_portfolio(tmp_path: Path):
    _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json")
    _portfolio(tmp_path / "novelty_refinement_a6.n10.portfolio.json")
    result = resolve_authoritative_final_portfolio(run_dir=tmp_path)
    assert result.authority_kind == "filesystem_n10_hard_filter"


def test_axis_only_run_fails_closed(tmp_path: Path):
    _portfolio(tmp_path / "hypothesis_axis_a4.portfolio.json")
    with pytest.raises(ValueError, match="not an authoritative final"):
        resolve_authoritative_final_portfolio(run_dir=tmp_path)


def test_explicit_authoritative_portfolio_is_allowed(tmp_path: Path):
    explicit = _portfolio(tmp_path / "custom_final.json", count=2)
    result = resolve_authoritative_final_portfolio(
        run_dir=tmp_path,
        explicit_portfolio=explicit,
    )
    assert result.authority_kind == "explicit"
    assert result.hypothesis_count == 2


def test_context_mismatch_fails_closed(tmp_path: Path):
    portfolio = _portfolio(tmp_path / "novelty_refinement_a6.portfolio.json", context_id="ctx:1")
    result = resolve_authoritative_final_portfolio(run_dir=tmp_path)
    context = HypothesisContext.model_construct(context_id="ctx:2")
    with pytest.raises(ValueError, match="context mismatch"):
        validate_authoritative_final_context(resolution=result, context=context)


def test_command_plan_omits_absent_optional_operators(tmp_path: Path):
    commands = build_authoritative_shadow_commands(
        context_path=tmp_path / "hypothesis.context.json",
        authoritative_portfolio_path=tmp_path / "novelty_refinement_a6.portfolio.json",
        reframing_portfolio_path=tmp_path / "scientific_reframing_reasoning_portfolio.json",
        reframe_shadow_path=tmp_path / "scientific_reframing_shadow.json",
        output_dir=tmp_path,
        proxy_shadow_path=None,
        contradiction_shadow_path=None,
        max_syntheses=2,
    )
    flat = " ".join(part for _, argv, _ in commands for part in argv)
    assert "--proxy-shadow" not in flat
    assert "--contradiction-shadow" not in flat
    assert "--max-syntheses 2" in flat


def test_command_plan_reuses_existing_cross_lane_chain(tmp_path: Path):
    commands = build_authoritative_shadow_commands(
        context_path=tmp_path / "hypothesis.context.json",
        authoritative_portfolio_path=tmp_path / "novelty_refinement_a6.portfolio.json",
        reframing_portfolio_path=tmp_path / "scientific_reframing_reasoning_portfolio.json",
        reframe_shadow_path=tmp_path / "scientific_reframing_shadow.json",
        output_dir=tmp_path,
        proxy_shadow_path=tmp_path / "proxy.json",
        contradiction_shadow_path=tmp_path / "contradiction.json",
    )
    modules = [argv[1] for _, argv, _ in commands]
    assert modules == [
        "scripts.discovery.build_cross_lane_scientific_reasoning_portfolio",
        "scripts.discovery.build_production_facing_scientific_candidate_portfolio",
        "scripts.discovery.run_cross_lane_scientific_synthesis_shadow",
        "scripts.discovery.build_integrated_scientific_hypothesis_shadow",
    ]
