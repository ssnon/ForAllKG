from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticDimensionDraft,
    HypothesisSemanticReview,
    HypothesisSemanticRunRecord,
    SEMANTIC_DIMENSIONS,
)
from pipeline_core.discovery.pre_n10_initial_semantic_gate_v1 import (
    execute_pre_n10_initial_semantic_gate_v1,
)


def _canonical_sha(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _portfolio() -> HypothesisPortfolio:
    observable = "spacing disorder increases spatial SERS intensity variance"
    card = HypothesisCard(
        hypothesis_id="hypothesis:h1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Spacing disorder and SERS variance",
        hypothesis_statement=(
            "Spacing disorder may increase spatial SERS intensity variance."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["stmt:1"],
        inferential_bridge=observable,
        predicted_observations=[
            PredictedObservation(
                observation_id="observation:1",
                observable=observable,
                expected_direction="increase",
                rationale="Tests the relation.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="criterion:1",
                observable=observable,
                falsifying_outcome=(
                    "spacing disorder does not increase "
                    "spatial SERS intensity variance"
                ),
            )
        ],
        source_paper_ids=["paper:1"],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )
    return HypothesisPortfolio(
        portfolio_id="hypothesis_portfolio:p1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        hypotheses=[card],
    )


def _review(portfolio: HypothesisPortfolio, *, fail_dimension: str | None = None):
    dimensions = []
    for dimension in SEMANTIC_DIMENSIONS:
        verdict = "fail" if dimension == fail_dimension else "pass"
        dimensions.append(
            HypothesisSemanticDimensionDraft(
                dimension=dimension,
                verdict=verdict,
                rationale="fixture semantic review",
                hypothesis_ids=(
                    ["hypothesis:h1"] if verdict == "fail" else []
                ),
            )
        )
    return HypothesisSemanticReview(
        review_id="semantic_review:1",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_portfolio_id=portfolio.portfolio_id,
        source_portfolio_sha256=_canonical_sha(portfolio),
        source_evaluator_version="fixture",
        source_hard_gate_passed=True,
        critic_prompt_version="semantic-v1",
        critic_prompt_sha256="c" * 64,
        dimensions=dimensions,
        overall_summary="fixture",
    )


def _run(
    portfolio: HypothesisPortfolio,
    *,
    accepted: bool,
    hard_gate_passed: bool,
    review_id: str | None,
    failure_stage: str,
) -> HypothesisSemanticRunRecord:
    return HypothesisSemanticRunRecord(
        run_id="semantic_run:1",
        context_id="context:1",
        context_sha256="a" * 64,
        portfolio_id=portfolio.portfolio_id,
        portfolio_sha256=_canonical_sha(portfolio),
        hard_gate_passed=hard_gate_passed,
        review_id=review_id,
        critic_prompt_version="semantic-v1",
        critic_prompt_sha256="c" * 64,
        backend="fixture",
        model="fixture",
        generated=hard_gate_passed,
        accepted=accepted,
        failure_stage=failure_stage,
    )


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _paths(tmp_path: Path, portfolio, run, review=None):
    portfolio_path = tmp_path / "portfolio.json"
    run_path = tmp_path / "semantic.run.json"
    review_path = tmp_path / "semantic.review.json"
    _write(portfolio_path, portfolio)
    _write(run_path, run)
    if review is not None:
        _write(review_path, review)
        return portfolio_path, run_path, review_path
    return portfolio_path, run_path, None


def test_initial_semantic_pass_authorizes_pre_n10_without_rerun(tmp_path: Path):
    portfolio = _portfolio()
    review = _review(portfolio)
    run = _run(
        portfolio,
        accepted=True,
        hard_gate_passed=True,
        review_id=review.review_id,
        failure_stage="none",
    )
    portfolio_path, run_path, review_path = _paths(
        tmp_path, portfolio, run, review
    )

    report, disposition = execute_pre_n10_initial_semantic_gate_v1(
        portfolio_path=portfolio_path,
        semantic_run_path=run_path,
        semantic_review_path=review_path,
        output_root=tmp_path / "gate",
    )

    assert report.status == "PRE_N10_ENTRY_AUTHORIZED"
    assert report.pre_n10_entry_authorized is True
    assert report.semantic_runtime_reexecuted is False
    assert report.semantic_critic_llm_reinvoked is False
    assert disposition is not None
    assert disposition.disposition == "PASS"


def test_initial_semantic_fail_blocks_pre_n10(tmp_path: Path):
    portfolio = _portfolio()
    review = _review(portfolio, fail_dimension="causal_strengthening")
    run = _run(
        portfolio,
        accepted=True,
        hard_gate_passed=True,
        review_id=review.review_id,
        failure_stage="none",
    )
    portfolio_path, run_path, review_path = _paths(
        tmp_path, portfolio, run, review
    )

    report, disposition = execute_pre_n10_initial_semantic_gate_v1(
        portfolio_path=portfolio_path,
        semantic_run_path=run_path,
        semantic_review_path=review_path,
        output_root=tmp_path / "gate",
    )

    assert report.status == "SEMANTIC_INTERVENTION_REQUIRED"
    assert report.pre_n10_entry_authorized is False
    assert disposition is not None
    assert disposition.disposition == "REQUIRES_SEMANTIC_INTERVENTION"
    assert "causal_strengthening" in report.semantic_failed_dimensions


def test_initial_hard_gate_failure_is_terminal_without_review(tmp_path: Path):
    portfolio = _portfolio()
    run = _run(
        portfolio,
        accepted=False,
        hard_gate_passed=False,
        review_id=None,
        failure_stage="hard_gate",
    )
    portfolio_path, run_path, review_path = _paths(tmp_path, portfolio, run)

    report, disposition = execute_pre_n10_initial_semantic_gate_v1(
        portfolio_path=portfolio_path,
        semantic_run_path=run_path,
        semantic_review_path=review_path,
        output_root=tmp_path / "gate",
    )

    assert report.status == "SEMANTIC_HARD_GATE_FAILED"
    assert report.semantic_review_valid is False
    assert report.pre_n10_entry_authorized is False
    assert disposition is None


def test_initial_review_validation_failure_blocks_without_disposition(tmp_path: Path):
    portfolio = _portfolio()
    run = _run(
        portfolio,
        accepted=False,
        hard_gate_passed=True,
        review_id=None,
        failure_stage="review_validation",
    )
    portfolio_path, run_path, review_path = _paths(tmp_path, portfolio, run)

    report, disposition = execute_pre_n10_initial_semantic_gate_v1(
        portfolio_path=portfolio_path,
        semantic_run_path=run_path,
        semantic_review_path=review_path,
        output_root=tmp_path / "gate",
    )

    assert report.status == "SEMANTIC_REVIEW_INVALID"
    assert report.pre_n10_entry_authorized is False
    assert disposition is None


def test_initial_semantic_gate_rejects_review_portfolio_mismatch(tmp_path: Path):
    portfolio = _portfolio()
    review = _review(portfolio).model_copy(
        update={"source_portfolio_id": "hypothesis_portfolio:other"}
    )
    run = _run(
        portfolio,
        accepted=True,
        hard_gate_passed=True,
        review_id=review.review_id,
        failure_stage="none",
    )
    portfolio_path, run_path, review_path = _paths(
        tmp_path, portfolio, run, review
    )

    with pytest.raises(ValueError, match="review/run portfolio ID mismatch"):
        execute_pre_n10_initial_semantic_gate_v1(
            portfolio_path=portfolio_path,
            semantic_run_path=run_path,
            semantic_review_path=review_path,
            output_root=tmp_path / "gate",
        )
