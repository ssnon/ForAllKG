from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import HypothesisNoveltyClaims, LiteratureQueryPlan, NoveltyClaim
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import build_pre_n10_scientific_contract_v1


def _card() -> HypothesisCard:
    predicted = "spacing disorder increases spatial SERS intensity variance"
    return HypothesisCard(
        hypothesis_id="hypothesis:h1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Spacing disorder and spatial SERS uniformity",
        hypothesis_statement="Spacing disorder may change spatial SERS intensity variance.",
        hypothesis_type="context_dependency",
        premise_statement_ids=["stmt:1"],
        inferential_bridge="spacing disorder changes spatial SERS intensity variance",
        predicted_observations=[
            PredictedObservation(
                observation_id="observation:1",
                observable=predicted,
                expected_direction="increase",
                rationale="Tests the proposed disorder dependence.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="criterion:1",
                observable=predicted,
                falsifying_outcome="spacing disorder changes do not alter spatial SERS intensity variance",
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


def _claim(
    *,
    claim_id: str = "claim:1",
    role: str = "NOVELTY_BEARING",
    required_bridge: str | None = None,
    predicted_observation: str | None = None,
) -> NoveltyClaim:
    bridge = "spacing disorder changes spatial SERS intensity variance" if required_bridge is None else required_bridge
    prediction = "spacing disorder increases spatial SERS intensity variance" if predicted_observation is None else predicted_observation
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role=role,
        text="spacing disorder changes spatial SERS intensity variance",
        rationale="A bounded relational claim.",
        search_concepts=["spacing disorder", "SERS uniformity"],
        search_queries=["spacing disorder SERS uniformity"],
        prior_art_identity_terms=["spacing disorder"],
        relation_nucleus_terms=["spacing disorder", "spatial SERS intensity variance", "changes"],
        required_bridge=bridge,
        predicted_observation=prediction,
        falsification_condition="spacing disorder changes do not alter spatial SERS intensity variance",
    )


def _portfolio() -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id="hypothesis_portfolio:p1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        hypotheses=[_card()],
    )


def _plan(claims: list[NoveltyClaim]) -> LiteratureQueryPlan:
    return LiteratureQueryPlan(
        plan_id="literature_query_plan:q1",
        plan_sha256="c" * 64,
        source_portfolio_id="hypothesis_portfolio:p1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:h1",
                title="Spacing disorder and spatial SERS uniformity",
                claims=claims,
            )
        ],
    )


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build(tmp_path: Path, claims: list[NoveltyClaim]):
    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "claims_queries.json"
    _write(portfolio_path, _portfolio())
    _write(plan_path, _plan(claims))
    return build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        claim_decomposition_request_count=1,
    )


def test_pre_n10_contract_ready_requires_complete_exact_source_binding(tmp_path: Path):
    report = _build(tmp_path, [_claim()])
    assert report.disposition == "READY_FOR_N10"
    assert report.ready_hypothesis_count == 1
    assert report.ready_claim_count == 1
    assert report.novelty_bearing_ready_claim_count == 1
    assert report.retrieval_performed is False
    assert report.n10_performed is False


def test_pre_n10_missing_bridge_routes_to_specification_repair(tmp_path: Path):
    report = _build(tmp_path, [_claim(required_bridge="")])
    row = report.hypotheses[0].claims[0]
    assert report.disposition == "INTERVENTION_REQUIRED"
    assert row.contract_status == "NOT_READY_FOR_N10_CONTRACT"
    assert row.router_hint == "SPECIFICATION_REPAIR_REVIEW"
    assert "missing_required_bridge" in row.binding_contract_reason_codes


def test_pre_n10_source_mismatch_routes_to_source_alignment(tmp_path: Path):
    report = _build(
        tmp_path,
        [_claim(predicted_observation="spacing disorder changes mean SERS intensity")],
    )
    row = report.hypotheses[0].claims[0]
    assert report.disposition == "INTERVENTION_REQUIRED"
    assert row.router_hint == "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    assert any(
        reason.startswith("prediction_exact_source_binding_cardinality:")
        for reason in row.source_contract_reason_codes
    )


def test_pre_n10_requires_all_claims_ready_not_only_novelty_bearing(tmp_path: Path):
    novelty = _claim()
    supporting = _claim(
        claim_id="claim:2",
        role="REQUIRED_ENABLING_RELATION",
        required_bridge="",
    ).model_copy(update={"claim_rank": 2})
    report = _build(tmp_path, [novelty, supporting])
    hypothesis = report.hypotheses[0]
    assert hypothesis.novelty_bearing_ready_claim_count == 1
    assert hypothesis.ready_claim_count == 1
    assert hypothesis.claim_count == 2
    assert hypothesis.contract_status == "REQUIRES_PRE_N10_INTERVENTION"
    assert report.disposition == "INTERVENTION_REQUIRED"
