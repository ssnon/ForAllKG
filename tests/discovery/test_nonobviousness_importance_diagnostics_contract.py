from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    ExternalNoveltyCard,
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    HypothesisNoveltyClaims,
    HypothesisSearchCoverage,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.nonobviousness_production_gate import (
    build_nonobviousness_fallback_gate,
)
from pipeline_core.discovery.nonobviousness_shadow import (
    compile_shadow_claim,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
    extract_novelty_residue,
)


def _planned_claim(
    *,
    importance: str = "core",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind="moderator_interaction",
        importance=importance,
        text=(
            "Factor A moderates the relation between "
            "descriptor B and outcome C."
        ),
        rationale="Test atomic role propagation.",
        search_concepts=[],
        search_queries=[
            "factor A descriptor B outcome C"
        ],
        distinguishing_terms=["factor A"],
        prior_art_identity_terms=["factor A"],
        relation_nucleus_terms=[
            "moderates descriptor B outcome C"
        ],
        required_bridge=(
            "Factor A changes the descriptor-B "
            "to outcome-C relationship."
        ),
        predicted_observation=(
            "Factor A changes the observed "
            "descriptor-B to outcome-C relationship."
        ),
        falsification_condition=(
            "The descriptor-B to outcome-C relationship "
            "is unchanged across factor-A conditions."
        ),
    )


def _plan(
    claim: NoveltyClaim,
) -> LiteratureQueryPlan:
    return LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="sha-plan",
        source_portfolio_id="portfolio:1",
        queries=[],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:1",
                title="Test hypothesis",
                claims=[claim],
            )
        ],
    )


def _report(
    *,
    importance: str = "core",
) -> ExternalNoveltyReport:
    claim_coverage = ClaimSearchCoverage(
        claim_id="claim:1",
        query_count=2,
        successful_query_count=2,
        unique_work_count=12,
        abstract_work_count=6,
        reviewed_work_count=6,
    )

    hypothesis_coverage = HypothesisSearchCoverage(
        hypothesis_id="hypothesis:1",
        query_count=2,
        successful_query_count=2,
        provider_success_count=2,
        unique_work_count=12,
        abstract_work_count=6,
        core_claim_count=(
            1 if importance == "core" else 0
        ),
        core_claims_with_minimum_abstract_coverage=(
            1 if importance == "core" else 0
        ),
        sufficient_for_absence_based_novelty=True,
    )

    review = ClaimPriorArtReview(
        hypothesis_id="hypothesis:1",
        claim_id="claim:1",
        claim_text=(
            "Factor A moderates the relation between "
            "descriptor B and outcome C."
        ),
        importance=importance,
        status="COMPONENTS_ONLY",
        matches=[],
        coverage=claim_coverage,
        interpretation="Components only.",
    )

    card = ExternalNoveltyCard(
        hypothesis_id="hypothesis:1",
        title="Test hypothesis",
        status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        claim_reviews=[review],
        coverage=hypothesis_coverage,
        interpretation="Relational gap.",
    )

    return ExternalNoveltyReport(
        report_id="report:1",
        report_sha256="sha-report",
        source_portfolio_id="portfolio:1",
        source_prior_art_packet_id="packet:1",
        searched_at_utc="2026-09-03T00:00:00+00:00",
        cards=[card],
        policy=ExternalNoveltyPolicy(),
    )


def test_core_importance_survives_query_plan_to_residue() -> None:
    claim = _planned_claim(
        importance="core"
    )

    residue = extract_novelty_residue(
        _plan(claim),
        _report(importance="core"),
    )[0].claims[0]

    assert residue.importance == "core"


def test_supporting_importance_survives_query_plan_to_residue() -> None:
    claim = _planned_claim(
        importance="supporting"
    )

    residue = extract_novelty_residue(
        _plan(claim),
        _report(importance="supporting"),
    )[0].claims[0]

    assert residue.importance == "supporting"


def test_missing_specification_diagnostics_reach_production_gate() -> None:
    claim = NoveltyResidueClaim(
        hypothesis_id="hypothesis:1",
        claim_id="claim:missing-spec",
        claim_text="Factor A moderates B to C.",
        claim_kind="moderator_interaction",
        prior_art_status="COMPONENTS_ONLY",
        disposition="RESIDUAL",
        is_residue=True,
        distinguishing_terms=("factor A",),
        prior_art_identity_terms=("factor A",),
        relation_nucleus_terms=("moderates B to C",),
        required_bridge="Factor A changes B to C.",
        predicted_observation="",
        falsification_condition="",
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
        importance="core",
    )

    decision = compile_shadow_claim(
        claim
    )

    assert (
        decision["shadow_state"]
        == "NEEDS_REFINEMENT"
    )

    specification = decision[
        "specification"
    ]

    assert set(
        specification["missing_fields"]
    ) == {
        "predicted_observation",
        "falsification_condition",
    }

    intake = {
        "schema_version":
            "nonobviousness-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            "portfolio:1",
        "hypotheses": [
            {
                "hypothesis_id":
                    "hypothesis:1",
                "external_status":
                    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                "claims": [
                    decision
                ],
            }
        ],
    }

    full = {
        "schema_version":
            "nonobviousness-full-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            "portfolio:1",
        "claims": [],
    }

    gate = (
        build_nonobviousness_fallback_gate(
            intake_shadow=intake,
            full_shadow=full,
        )
    )

    atomic = (
        gate["gates"][0]
        ["atomic_claims"][0]
    )

    assert atomic["importance"] == "core"
    assert (
        atomic["nonobviousness_outcome"]
        == "NEEDS_REFINEMENT"
    )

    reasons = set(
        atomic["reason_codes"]
    )

    assert (
        "atomic_specification_incomplete"
        in reasons
    )
    assert (
        "atomic_residue_under_specified"
        in reasons
    )
    assert (
        "missing_predicted_observation"
        in reasons
    )
    assert (
        "missing_falsification_condition"
        in reasons
    )
    assert (
        "missing_specification_field:"
        "predicted_observation"
        in reasons
    )
    assert (
        "missing_specification_field:"
        "falsification_condition"
        in reasons
    )


def test_legacy_missing_importance_fails_safe_as_core() -> None:
    intake = {
        "schema_version":
            "nonobviousness-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            "portfolio:legacy",
        "hypotheses": [
            {
                "hypothesis_id":
                    "hypothesis:legacy",
                "external_status":
                    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                "claims": [
                    {
                        "claim": {
                            "claim_id":
                                "claim:legacy",
                        },
                        "specification": {
                            "reason_codes": [
                                "atomic_residue_under_specified",
                                "missing_required_bridge",
                            ],
                            "missing_fields": [
                                "required_bridge",
                            ],
                        },
                        "shadow_state":
                            "NEEDS_REFINEMENT",
                    }
                ],
            }
        ],
    }

    full = {
        "schema_version":
            "nonobviousness-full-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            "portfolio:legacy",
        "claims": [],
    }

    gate = (
        build_nonobviousness_fallback_gate(
            intake_shadow=intake,
            full_shadow=full,
        )
    )

    atomic = (
        gate["gates"][0]
        ["atomic_claims"][0]
    )

    assert atomic["importance"] == "core"
    assert gate["gates"][0][
        "fallback_allowed"
    ] is False



def test_sanitization_diagnostics_are_preserved_in_specification() -> None:
    claim = NoveltyResidueClaim(
        hypothesis_id="hypothesis:diag",
        claim_id="claim:diag",
        claim_text="Factor A moderates B to C.",
        claim_kind="moderator_interaction",
        prior_art_status="COMPONENTS_ONLY",
        disposition="RESIDUAL",
        is_residue=True,
        distinguishing_terms=("factor A",),
        prior_art_identity_terms=("factor A",),
        relation_nucleus_terms=("moderates B to C",),
        required_bridge="Factor A changes B to C.",
        predicted_observation="",
        falsification_condition="",
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
        importance="core",
        specification_sanitization_reason_codes=(
            "predicted_observation_rejected_branch_identity",
            "falsification_condition_draft_empty",
        ),
    )

    decision = compile_shadow_claim(
        claim
    )

    assert (
        decision["shadow_state"]
        == "NEEDS_REFINEMENT"
    )

    reasons = set(
        decision[
            "specification"
        ][
            "reason_codes"
        ]
    )

    assert (
        "predicted_observation_rejected_branch_identity"
        in reasons
    )

    assert (
        "falsification_condition_draft_empty"
        in reasons
    )


def test_legacy_intake_missing_new_metadata_reconciles_fail_safe() -> None:
    from pipeline_core.discovery.nonobviousness_shadow import (
        _json_safe,
        reconcile_intake_required_bridge,
    )

    claim = NoveltyResidueClaim(
        hypothesis_id="hypothesis:legacy-metadata",
        claim_id="claim:legacy-metadata",
        claim_text="Factor A moderates B to C.",
        claim_kind="moderator_interaction",
        prior_art_status="COMPONENTS_ONLY",
        disposition="RESIDUAL",
        is_residue=True,
        distinguishing_terms=("factor A",),
        prior_art_identity_terms=("factor A",),
        relation_nucleus_terms=("moderates B to C",),
        required_bridge="Factor A changes B to C.",
        predicted_observation=(
            "Factor A changes the B-to-C relationship."
        ),
        falsification_condition=(
            "The B-to-C relationship does not change."
        ),
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
        importance="core",
        specification_sanitization_reason_codes=(
            "required_bridge_source_draft",
        ),
    )

    # Simulate an OLD JSON-SERIALIZED intake claim.
    #
    # reconcile_intake_required_bridge() compares against _json_safe(claim),
    # so the fixture must use the same JSON-native representation:
    # tuples -> lists, nested Pydantic models -> dictionaries.
    incoming = _json_safe(claim)

    assert isinstance(incoming, dict)

    incoming.pop(
        "importance",
        None,
    )
    incoming.pop(
        "specification_sanitization_reason_codes",
        None,
    )

    reconciled = (
        reconcile_intake_required_bridge(
            claim,
            intake_claim=incoming,
            specification_provenance={
                "required_bridge":
                    "QUERY_PLAN",
            },
            hypothesis=None,
        )
    )

    assert reconciled == claim
    assert reconciled.importance == "core"
