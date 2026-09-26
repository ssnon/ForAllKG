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
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    build_pre_n10_scientific_contract_v2,
)


def _card() -> HypothesisCard:
    observable = "response Y source observable"
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
                falsifying_outcome=(
                    "response Y source falsifying outcome"
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


def _claim(
    *,
    kind: str = "mechanistic_link",
    role: str | None = "NOVELTY_BEARING",
    bridge: str = "Factor X changes response Y.",
    prediction: str = "response Y source observable",
    falsifier: str = "response Y source falsifying outcome",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind=kind,
        importance="core",
        novelty_selection_role=role,
        text="Factor X changes response Y.",
        rationale="fixture",
        search_concepts=["Factor X", "response Y"],
        search_queries=["Factor X response Y"],
        prior_art_identity_terms=["response Y"],
        relation_nucleus_terms=["Factor X", "response Y"],
        required_bridge=bridge,
        predicted_observation=prediction,
        falsification_condition=falsifier,
    )


def _portfolio() -> HypothesisPortfolio:
    card = _card()
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


def _materialize(
    tmp_path: Path,
    *,
    claim: NoveltyClaim,
    prediction_id: str | None = "prediction:1",
    falsifier_id: str | None = "falsifier:1",
):
    portfolio = _portfolio()
    plan = _plan(claim)

    record = build_atomic_scientific_source_binding_record(
        hypothesis_id=claim.hypothesis_id,
        claim_local_id="atomic:1",
        claim=claim,
        binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis="Factor X changes response Y.",
            relation_endpoint_anchors=[
                "Factor X",
                "response Y",
            ],
            prediction_observation_id=prediction_id,
            falsification_criterion_id=falsifier_id,
        ),
    )
    bundle = build_atomic_scientific_source_binding_bundle(
        source_portfolio_id=portfolio.portfolio_id,
        query_plan=plan,
        records=[record],
    )

    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "plan.json"
    bundle_path = tmp_path / "source_binding.bundle.json"
    canonical_path = tmp_path / "canonical_source.report.json"

    _write(portfolio_path, portfolio)
    _write(plan_path, plan)
    _write(bundle_path, bundle)

    canonical = build_pre_n10_canonical_source_reference_report_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        source_binding_bundle_path=bundle_path,
    )
    _write(canonical_path, canonical)

    return portfolio_path, plan_path, canonical_path


def _build(
    tmp_path: Path,
    *,
    claim: NoveltyClaim,
    prediction_id: str | None = "prediction:1",
    falsifier_id: str | None = "falsifier:1",
):
    portfolio_path, plan_path, canonical_path = _materialize(
        tmp_path,
        claim=claim,
        prediction_id=prediction_id,
        falsifier_id=falsifier_id,
    )
    return build_pre_n10_scientific_contract_v2(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        canonical_source_reference_path=canonical_path,
        claim_decomposition_request_count=1,
    )


def test_v2_stable_ready_exact_text_mismatch_is_ready(
    tmp_path: Path,
) -> None:
    report = _build(
        tmp_path,
        claim=_claim(
            prediction=(
                "response Y novelty surface differs from source observable"
            )
        ),
    )
    row = report.hypotheses[0].claims[0]

    assert row.source_reference_status == "READY"
    assert row.source_identity_comparison == (
        "STABLE_READY_EXACT_NOT_READY"
    )
    assert row.exact_text_source_reference_ready is False
    assert row.contract_status == "READY_FOR_N10_CONTRACT"
    assert row.router_hint == "PROCEED_TO_LITERAL_ENDPOINT_BINDING"
    assert report.disposition == "READY_FOR_N10"
    assert report.stable_source_ids_used_for_readiness is True
    assert report.exact_text_reconstruction_used_for_readiness is False


def test_v2_invalid_stable_id_blocks_even_when_exact_text_matches(
    tmp_path: Path,
) -> None:
    report = _build(
        tmp_path,
        claim=_claim(),
        prediction_id="prediction:missing",
    )
    row = report.hypotheses[0].claims[0]

    assert row.source_reference_status == "INVALID"
    assert row.source_identity_comparison == (
        "STABLE_NOT_READY_EXACT_READY"
    )
    assert row.exact_text_source_reference_ready is True
    assert row.contract_status == "NOT_READY_FOR_N10_CONTRACT"
    assert row.router_hint == (
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    )
    assert report.disposition == "INTERVENTION_REQUIRED"


def test_v2_missing_stable_ids_are_not_reconstructed_from_text(
    tmp_path: Path,
) -> None:
    report = _build(
        tmp_path,
        claim=_claim(),
        prediction_id=None,
        falsifier_id=None,
    )
    row = report.hypotheses[0].claims[0]

    assert row.source_reference_status == "INCOMPLETE"
    assert "missing_prediction_source_id" in (
        row.source_reference_reason_codes
    )
    assert "missing_falsifier_source_id" in (
        row.source_reference_reason_codes
    )
    assert row.exact_text_source_reference_ready is True
    assert row.contract_status == "NOT_READY_FOR_N10_CONTRACT"
    assert row.router_hint == (
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    )


def test_v2_binding_specification_failure_still_routes_to_repair(
    tmp_path: Path,
) -> None:
    report = _build(
        tmp_path,
        claim=_claim(bridge=""),
    )
    row = report.hypotheses[0].claims[0]

    assert row.source_reference_status == "READY"
    assert "missing_required_bridge" in (
        row.binding_contract_reason_codes
    )
    assert row.contract_status == "NOT_READY_FOR_N10_CONTRACT"
    assert row.router_hint == "SPECIFICATION_REPAIR_REVIEW"


def test_v2_unsupported_kind_dominates_source_alignment(
    tmp_path: Path,
) -> None:
    report = _build(
        tmp_path,
        claim=_claim(kind="composite"),
        prediction_id="prediction:missing",
    )
    row = report.hypotheses[0].claims[0]

    assert row.atomic_kind_supported is False
    assert row.source_reference_status == "INVALID"
    assert row.contract_status == "NOT_READY_FOR_N10_CONTRACT"
    assert row.router_hint == "DECOMPOSE_OR_REGENERATE_REVIEW"


def test_v2_fails_closed_if_query_plan_changes_after_source_assessment(
    tmp_path: Path,
) -> None:
    claim = _claim()
    portfolio_path, plan_path, canonical_path = _materialize(
        tmp_path,
        claim=claim,
    )

    plan = LiteratureQueryPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    changed = plan.model_copy(
        update={"plan_id": "plan:changed"}
    )
    _write(plan_path, changed)

    with pytest.raises(
        ValueError,
        match=(
            "canonical source-reference query-plan ID mismatch"
            "|canonical source-reference query-plan file changed"
        ),
    ):
        build_pre_n10_scientific_contract_v2(
            portfolio_path=portfolio_path,
            query_plan_path=plan_path,
            canonical_source_reference_path=canonical_path,
            claim_decomposition_request_count=1,
        )
