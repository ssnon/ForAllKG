from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    build_atomic_scientific_source_binding_bundle,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    LiteratureQueryPlanner,
    NoveltyClaimDecomposer,
)


class _Backend:
    def __init__(
        self,
        *,
        prediction_id: str | None = "prediction:1",
        falsifier_id: str | None = "falsifier:1",
    ) -> None:
        self.prediction_id = prediction_id
        self.falsifier_id = falsifier_id

    def decompose(
        self,
        hypothesis: HypothesisCard,
        *,
        max_claims: int,
    ) -> NoveltyClaimDecompositionDraft:
        return NoveltyClaimDecompositionDraft(
            claims=[
                NoveltyClaimDraft(
                    local_id="atomic:1",
                    kind="mechanistic_link",
                    importance="core",
                    novelty_selection_role="NOVELTY_BEARING",
                    text="Factor X changes response Y.",
                    rationale="fixture",
                    search_concepts=["Factor X", "response Y"],
                    search_queries=["Factor X response Y"],
                    distinguishing_terms=[],
                    prior_art_identity_terms=["Factor X"],
                    relation_nucleus_terms=[
                        "Factor X",
                        "response Y",
                    ],
                    required_bridge="Factor X changes response Y.",
                    predicted_observation="response Y changes",
                    falsification_condition=(
                        "response Y does not change"
                    ),
                    semantic_fidelity_binding=(
                        NoveltyClaimSemanticFidelityBindingDraft(
                            proposition_basis=(
                                "Factor X changes response Y."
                            ),
                            relation_endpoint_anchors=[
                                "Factor X",
                                "response Y",
                            ],
                            scope_qualifier_spans=[],
                            directional_qualifier_spans=[],
                            prediction_observation_id=(
                                self.prediction_id
                            ),
                            falsification_criterion_id=(
                                self.falsifier_id
                            ),
                        )
                    ),
                )
            ]
        )


def _card() -> HypothesisCard:
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
                observable="response Y source observable",
                expected_direction="unspecified",
                rationale="fixture",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable="response Y source observable",
                falsifying_outcome="response Y does not change",
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


def test_decomposer_preserves_stable_source_ids_outside_novelty_claim():
    card = _card()
    decomposer = NoveltyClaimDecomposer(_Backend())

    claims = decomposer.decompose(card)

    assert len(claims.claims) == 1
    assert len(decomposer.atomic_source_binding_records) == 1
    record = decomposer.atomic_source_binding_records[0]
    claim = claims.claims[0]

    assert record.hypothesis_id == card.hypothesis_id
    assert record.claim_id == claim.claim_id
    assert record.claim_rank == claim.claim_rank
    assert record.claim_local_id == "atomic:1"
    assert record.prediction_observation_id == "prediction:1"
    assert record.falsification_criterion_id == "falsifier:1"
    assert record.relation_endpoint_anchors == [
        "Factor X",
        "response Y",
    ]
    assert record.exact_text_reconstruction_used_for_identity is False
    assert record.readiness_authority is False

    # NoveltyClaim remains a novelty/search surface and does not become
    # a second owner of stable source IDs.
    assert not hasattr(claim, "prediction_observation_id")
    assert not hasattr(claim, "falsification_criterion_id")


def test_bundle_is_bound_to_final_query_plan_claim_sha():
    card = _card()
    portfolio = _portfolio(card)
    decomposer = NoveltyClaimDecomposer(_Backend())
    decomposition = decomposer.decompose(card)
    plan = LiteratureQueryPlanner().build(
        portfolio,
        [decomposition],
    )

    bundle = build_atomic_scientific_source_binding_bundle(
        source_portfolio_id=portfolio.portfolio_id,
        query_plan=plan,
        records=list(decomposer.atomic_source_binding_records),
    )

    assert bundle.claim_count == 1
    assert bundle.prediction_source_id_present_count == 1
    assert bundle.falsifier_source_id_present_count == 1
    assert bundle.source_query_plan_id == plan.plan_id
    assert bundle.source_query_plan_sha256 == plan.plan_sha256
    assert bundle.exact_text_reconstruction_used_for_identity is False
    assert bundle.readiness_authority is False


def test_bundle_rejects_claim_surface_drift_after_binding_capture():
    card = _card()
    portfolio = _portfolio(card)
    decomposer = NoveltyClaimDecomposer(_Backend())
    decomposition = decomposer.decompose(card)
    claim = decomposition.claims[0]
    changed = claim.model_copy(
        update={"text": "Tampered scientific surface."}
    )
    changed_group = decomposition.model_copy(
        update={"claims": [changed]}
    )
    plan = LiteratureQueryPlanner().build(
        portfolio,
        [changed_group],
    )

    with pytest.raises(
        ValueError,
        match="source-binding/query-plan claim SHA mismatch",
    ):
        build_atomic_scientific_source_binding_bundle(
            source_portfolio_id=portfolio.portfolio_id,
            query_plan=plan,
            records=list(
                decomposer.atomic_source_binding_records
            ),
        )


def test_missing_source_ids_are_preserved_as_missing_not_reconstructed():
    card = _card()
    portfolio = _portfolio(card)
    decomposer = NoveltyClaimDecomposer(
        _Backend(
            prediction_id=None,
            falsifier_id=None,
        )
    )
    decomposition = decomposer.decompose(card)
    plan = LiteratureQueryPlanner().build(
        portfolio,
        [decomposition],
    )

    record = decomposer.atomic_source_binding_records[0]
    assert record.prediction_observation_id is None
    assert record.falsification_criterion_id is None

    bundle = build_atomic_scientific_source_binding_bundle(
        source_portfolio_id=portfolio.portfolio_id,
        query_plan=plan,
        records=list(decomposer.atomic_source_binding_records),
    )
    assert bundle.prediction_source_id_present_count == 0
    assert bundle.falsifier_source_id_present_count == 0
