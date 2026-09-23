from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    assess_claim_binding_readiness,
    build_relational_atomic_binding_plan,
)


def _claim(
    *,
    prediction: str = "Catalyst size changes Raman intensity.",
    identity: str = "catalyst size",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="external_novelty_claim:c1",
        hypothesis_id="hypothesis:candidate",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=(
            "Catalyst size changes Raman intensity under matched conditions."
        ),
        rationale="Existing source-bounded atomic claim.",
        prior_art_identity_terms=[identity],
        relation_nucleus_terms=["catalyst size", "Raman intensity"],
        required_bridge=(
            "Catalyst size changes Raman intensity under matched conditions."
        ),
        predicted_observation=prediction,
        falsification_condition=(
            "Catalyst size does not change Raman intensity."
        ),
        search_queries=["catalyst size Raman intensity"],
    )


def _card(hypothesis_id: str, *, statement: str = "H") -> HypothesisCard:
    suffix = hypothesis_id.split(":")[-1]
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="demo",
        source_context_id="context:1",
        source_context_sha256="c" * 64,
        source_report_id="report:1",
        source_report_sha256="r" * 64,
        title="Title",
        hypothesis_statement=statement,
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        inferential_bridge=(
            "Catalyst size changes Raman intensity under matched conditions."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:" + suffix,
                observable="Raman intensity",
                expected_direction="unspecified",
                rationale="Catalyst size changes Raman intensity.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:" + suffix,
                observable="Raman intensity",
                falsifying_outcome=(
                    "Catalyst size does not change Raman intensity."
                ),
            )
        ],
        assumptions=[],
        source_paper_ids=["paper:1"],
        gap_paper_ids=[],
        cross_paper_synthesis=False,
        candidate_dependency="none",
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture_run(tmp_path: Path, *, drift: bool = False) -> Path:
    run = tmp_path / "PXX"
    candidate = _card("hypothesis:candidate")
    final = _card(
        "hypothesis:final",
        statement=("DRIFT" if drift else "H"),
    )

    final_portfolio = HypothesisPortfolio(
        portfolio_id="portfolio:final",
        domain_profile_id="demo",
        source_context_id="context:1",
        source_context_sha256="c" * 64,
        source_report_id="report:1",
        source_report_sha256="r" * 64,
        hypotheses=[final],
    )
    source_portfolio = HypothesisPortfolio(
        portfolio_id="portfolio:candidate",
        domain_profile_id="demo",
        source_context_id="context:1",
        source_context_sha256="c" * 64,
        source_report_id="report:1",
        source_report_sha256="r" * 64,
        hypotheses=[candidate],
    )
    query_plan = LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="p" * 64,
        source_portfolio_id=source_portfolio.portfolio_id,
        queries=[],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id=candidate.hypothesis_id,
                title=candidate.title,
                claims=[_claim()],
            )
        ],
    )

    final_path = run / "novelty_refinement_a6.n10.candidate.portfolio.json"
    source_path = run / "candidate.portfolio.json"
    query_path = run / "candidate.claims_queries.json"
    cert_path = run / "novelty_refinement_a6.n10.certification.json"

    _write(final_path, final_portfolio)
    _write(source_path, source_portfolio)
    _write(query_path, query_plan)
    _write(
        cert_path,
        {
            "schema_version":
                "alpha6-post-generation-novelty-certification-v2",
            "candidate_portfolio_id": final_portfolio.portfolio_id,
            "authority_mode": "certification_only",
            "candidate_portfolio_preserved": True,
            "candidate_survival_authority": False,
            "ineligible_deletes_scientific_candidate": False,
            "decisions": [
                {
                    "original_hypothesis_id": "hypothesis:original",
                    "candidate_hypothesis_id": "hypothesis:candidate",
                    "final_hypothesis_id": "hypothesis:final",
                    "alpha6_decision": "accepted_refinement",
                    "post_generation_n10_required": True,
                    "n10_selection_class": "CONDITIONAL",
                    "certification_status": "NOVELTY_UNRESOLVED",
                }
            ],
            "candidate_artifacts": [
                {
                    "candidate_id": "hypothesis:candidate",
                    "final_hypothesis_id": "hypothesis:final",
                    "candidate_final_authority_equivalent": True,
                    "source_portfolio": str(source_path),
                    "query_plan": str(query_path),
                }
            ],
        },
    )
    return run


def test_complete_claim_is_ready_for_literal_endpoint_binding() -> None:
    row = assess_claim_binding_readiness(
        claim=_claim(),
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
    )
    assert row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
    assert row.reason_codes == []
    assert row.endpoint_binding_performed is False


def test_missing_prediction_fails_closed() -> None:
    row = assess_claim_binding_readiness(
        claim=_claim(prediction=""),
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
    )
    assert row.binding_status == (
        "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
    )
    assert "missing_predicted_observation" in row.reason_codes


def test_identity_must_remain_literal_in_atomic_specification() -> None:
    row = assess_claim_binding_readiness(
        claim=_claim(identity="unseen branch identity"),
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
    )
    assert row.binding_status == (
        "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
    )
    assert any(
        code.startswith("identity_not_literal_in_")
        for code in row.reason_codes
    )


def test_binding_plan_validates_candidate_final_authority_equivalence(
    tmp_path: Path,
) -> None:
    run = _fixture_run(tmp_path)
    plan = build_relational_atomic_binding_plan(run_dir=run)
    assert plan.hypothesis_count == 1
    assert plan.ready_hypothesis_count == 1
    assert plan.binding_ready_claim_count == 1
    assert plan.novelty_bearing_binding_ready_claim_count == 1
    assert plan.endpoint_binding_performed is False
    assert plan.verifier_result_observed is False


def test_binding_plan_rejects_candidate_final_drift(
    tmp_path: Path,
) -> None:
    run = _fixture_run(tmp_path, drift=True)
    with pytest.raises(
        ValueError,
        match="candidate/final authority transfer",
    ):
        build_relational_atomic_binding_plan(run_dir=run)


def test_binding_plan_hash_is_deterministic(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)
    first = build_relational_atomic_binding_plan(run_dir=run)
    second = build_relational_atomic_binding_plan(run_dir=run)
    assert first.plan_id == second.plan_id
    assert first.plan_sha256 == second.plan_sha256
