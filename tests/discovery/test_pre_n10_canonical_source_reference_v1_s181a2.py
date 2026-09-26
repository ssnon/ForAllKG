from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    build_atomic_scientific_source_binding_bundle,
    build_atomic_scientific_source_binding_record,
)
from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
    build_pre_n10_canonical_source_reference_report_v1,
)


def _card() -> HypothesisCard:
    observable = "source observable Y"
    return HypothesisCard(
        hypothesis_id="hypothesis:1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Fixture",
        hypothesis_statement="Factor X changes response Y.",
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=["statement:1"],
        inferential_bridge="Factor X changes response Y.",
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable=observable,
                expected_direction="increase",
                rationale="fixture",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable=observable,
                falsifying_outcome="source falsifying outcome Y",
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
    predicted_observation: str = "source observable Y",
    falsification_condition: str = "source falsifying outcome Y",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Factor X changes response Y.",
        rationale="fixture",
        search_concepts=["Factor X", "response Y"],
        search_queries=["Factor X response Y"],
        prior_art_identity_terms=["Factor X"],
        relation_nucleus_terms=["Factor X", "response Y"],
        required_bridge="Factor X changes response Y.",
        predicted_observation=predicted_observation,
        falsification_condition=falsification_condition,
    )


def _portfolio(card: HypothesisCard) -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id="portfolio:1",
        domain_profile_id=card.domain_profile_id,
        source_context_id=card.source_context_id,
        source_context_sha256=card.source_context_sha256,
        source_report_id=card.source_report_id,
        source_report_sha256=card.source_report_sha256,
        hypotheses=[card],
    )


def _plan(claim: NoveltyClaim) -> LiteratureQueryPlan:
    return LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="c" * 64,
        source_portfolio_id="portfolio:1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:1",
                title="Fixture",
                claims=[claim],
            )
        ],
    )


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _bundle(
    *,
    claim: NoveltyClaim,
    plan: LiteratureQueryPlan,
    prediction_id: str | None = "prediction:1",
    falsifier_id: str | None = "falsifier:1",
):
    record = build_atomic_scientific_source_binding_record(
        hypothesis_id=claim.hypothesis_id,
        claim_local_id="atomic:1",
        claim=claim,
        binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis="Factor X changes response Y.",
            relation_endpoint_anchors=["Factor X", "response Y"],
            prediction_observation_id=prediction_id,
            falsification_criterion_id=falsifier_id,
        ),
    )
    return build_atomic_scientific_source_binding_bundle(
        source_portfolio_id="portfolio:1",
        query_plan=plan,
        records=[record],
    )


def _materialize(
    tmp_path: Path,
    *,
    claim: NoveltyClaim,
    bundle,
):
    card = _card()
    portfolio = _portfolio(card)
    plan = _plan(claim)

    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "plan.json"
    bundle_path = tmp_path / "source_binding.bundle.json"
    _write(portfolio_path, portfolio)
    _write(plan_path, plan)
    _write(bundle_path, bundle)
    return portfolio_path, plan_path, bundle_path


def test_stable_ids_remain_ready_when_exact_text_surface_does_not_match(
    tmp_path: Path,
) -> None:
    claim = _claim(
        predicted_observation="novelty surface differs from source observable",
    )
    plan = _plan(claim)
    bundle = _bundle(claim=claim, plan=plan)
    portfolio_path, plan_path, bundle_path = _materialize(
        tmp_path,
        claim=claim,
        bundle=bundle,
    )

    report = build_pre_n10_canonical_source_reference_report_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        source_binding_bundle_path=bundle_path,
    )

    row = report.rows[0]
    assert row.stable_source_reference_status == "READY"
    assert row.stable_source_reference_reason_codes == []
    assert row.exact_text_source_reference_ready is False
    assert row.prediction_exact_text_binding_count == 0
    assert row.comparison == "STABLE_READY_EXACT_NOT_READY"
    assert report.stable_ready_count == 1
    assert report.exact_text_ready_count == 0
    assert report.disagreement_count == 1
    assert report.contract_readiness_authority is False


def test_invalid_stable_id_is_visible_even_when_exact_text_would_pass(
    tmp_path: Path,
) -> None:
    claim = _claim()
    plan = _plan(claim)
    bundle = _bundle(
        claim=claim,
        plan=plan,
        prediction_id="prediction:missing",
    )
    portfolio_path, plan_path, bundle_path = _materialize(
        tmp_path,
        claim=claim,
        bundle=bundle,
    )

    report = build_pre_n10_canonical_source_reference_report_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        source_binding_bundle_path=bundle_path,
    )

    row = report.rows[0]
    assert row.stable_source_reference_status == "INVALID"
    assert any(
        reason.startswith("prediction_source_id_cardinality:0")
        for reason in row.stable_source_reference_reason_codes
    )
    assert row.exact_text_source_reference_ready is True
    assert row.comparison == "STABLE_NOT_READY_EXACT_READY"
    assert report.stable_invalid_count == 1
    assert report.exact_text_ready_count == 1
    assert report.disagreement_count == 1


def test_missing_ids_are_not_reconstructed_from_matching_text(
    tmp_path: Path,
) -> None:
    claim = _claim()
    plan = _plan(claim)
    bundle = _bundle(
        claim=claim,
        plan=plan,
        prediction_id=None,
        falsifier_id=None,
    )
    portfolio_path, plan_path, bundle_path = _materialize(
        tmp_path,
        claim=claim,
        bundle=bundle,
    )

    report = build_pre_n10_canonical_source_reference_report_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        source_binding_bundle_path=bundle_path,
    )

    row = report.rows[0]
    assert row.stable_source_reference_status == "INCOMPLETE"
    assert "missing_prediction_source_id" in (
        row.stable_source_reference_reason_codes
    )
    assert "missing_falsifier_source_id" in (
        row.stable_source_reference_reason_codes
    )
    assert row.exact_text_source_reference_ready is True
    assert row.comparison == "STABLE_NOT_READY_EXACT_READY"


def test_claim_surface_drift_after_bundle_capture_fails_closed(
    tmp_path: Path,
) -> None:
    original = _claim()
    original_plan = _plan(original)
    bundle = _bundle(
        claim=original,
        plan=original_plan,
    )

    changed = original.model_copy(
        update={"text": "Tampered claim surface."}
    )
    changed_plan = _plan(changed)
    portfolio_path, plan_path, bundle_path = _materialize(
        tmp_path,
        claim=changed,
        bundle=bundle,
    )

    with pytest.raises(
        ValueError,
        match="canonical source-reference claim SHA mismatch",
    ):
        build_pre_n10_canonical_source_reference_report_v1(
            portfolio_path=portfolio_path,
            query_plan_path=plan_path,
            source_binding_bundle_path=bundle_path,
        )
