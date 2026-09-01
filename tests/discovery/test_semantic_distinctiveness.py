from __future__ import annotations

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    ExternalNoveltyCard,
    HypothesisSearchCoverage,
    PriorArtMatch,
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.scientific_distinctiveness_contracts import (
    ScientificDistinctivenessClaimSignal,
    ScientificDistinctivenessReport,
    ScientificDistinctivenessReview,
)
from pipeline_core.discovery.semantic_distinctiveness import (
    compile_semantic_distinctiveness_review,
    derive_semantic_distinctiveness_tier,
)
from pipeline_core.discovery.semantic_distinctiveness_contracts import (
    SemanticDimensionAssessment,
    SemanticDistinctivenessDraft,
)
from pipeline_core.discovery.semantic_distinctiveness_prompt import (
    SemanticDistinctivenessPromptAssembler,
)


def _fixture():
    work = PriorArtWork(
        work_id="work:known",
        title="Known lower-order relation",
        year=2020,
        abstract=(
            "Variable A changes response Y under a supplied "
            "experimental context."
        ),
        retrieval_query_ids=[
            "query:c1"
        ],
        retrieval_claim_ids=[
            "claim:c1"
        ],
    )

    packet = PriorArtPacket(
        packet_id="packet:test",
        packet_sha256="packet-sha",
        source_portfolio_id="portfolio:test",
        source_query_plan_id="plan:test",
        searched_at_utc=(
            "2026-01-01T00:00:00+00:00"
        ),
        providers_requested=[
            "fixture"
        ],
        works=[
            work
        ],
        raw_work_count=1,
        canonical_work_count=1,
    )

    match = PriorArtMatch(
        work_id="work:known",
        relationship=(
            "LOWER_ORDER_RELATION_PRIOR_ART"
        ),
        confidence=0.90,
        rationale=(
            "The work supports one lower-order relation "
            "but not the proposed interaction."
        ),
        relevance_score=0.90,
        semantic_similarity=0.85,
        lexical_coverage=0.80,
        abstract_available=True,
        title="Known lower-order relation",
        year=2020,
    )

    external_claim = ClaimPriorArtReview(
        hypothesis_id="hypothesis:test",
        claim_id="claim:c1",
        claim_text=(
            "Context B changes how A affects response Y."
        ),
        importance="core",
        status="COMPONENTS_ONLY",
        matches=[
            match
        ],
        coverage=ClaimSearchCoverage(
            claim_id="claim:c1",
            query_count=2,
            successful_query_count=2,
            unique_work_count=10,
            abstract_work_count=5,
            reviewed_work_count=5,
        ),
        reason_codes=[],
        interpretation=(
            "Lower-order relation is represented; the "
            "moderation is not directly represented."
        ),
    )

    coverage = HypothesisSearchCoverage(
        hypothesis_id="hypothesis:test",
        query_count=2,
        successful_query_count=2,
        provider_success_count=1,
        unique_work_count=10,
        abstract_work_count=5,
        core_claim_count=1,
        core_claims_with_minimum_abstract_coverage=1,
        sufficient_for_absence_based_novelty=True,
    )

    card = ExternalNoveltyCard(
        hypothesis_id="hypothesis:test",
        title="Test interaction",
        status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        claim_reviews=[
            external_claim
        ],
        coverage=coverage,
        lower_order_prior_art_work_ids=[
            "work:known"
        ],
        lower_order_supported_core_claim_ids=[
            "claim:c1"
        ],
        higher_order_relational_gap_claim_ids=[
            "claim:c1"
        ],
        lower_order_core_prior_art_work_ids=[
            "work:known"
        ],
        lower_order_core_unique_work_count=1,
        relational_gap_kind=(
            "HIGHER_ORDER_RELATIONAL_GAP"
        ),
        interpretation=(
            "Frozen external novelty fixture."
        ),
    )

    signal = ScientificDistinctivenessClaimSignal(
        hypothesis_id="hypothesis:test",
        claim_id="claim:c1",
        claim_kind="moderator_interaction",
        importance="core",
        claim_text=(
            "Context B changes how A affects response Y."
        ),
        prior_art_status="COMPONENTS_ONLY",
        query_count=2,
        successful_query_count=2,
        unique_work_count=10,
        abstract_work_count=5,
        reviewed_work_count=5,
        relationship_counts={
            "LOWER_ORDER_RELATION_PRIOR_ART":
                1
        },
        lower_order_prior_art_work_ids=[
            "work:known"
        ],
        reason_codes=[],
    )

    scientific_review = (
        ScientificDistinctivenessReview(
            hypothesis_id="hypothesis:test",
            title="Test interaction",
            external_novelty_status=(
                "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
            ),
            evidence_pattern=(
                "HIGHER_ORDER_RELATIONAL_GAP_WITH_"
                "LOWER_ORDER_PRIOR_ART"
            ),
            claim_count=1,
            core_claim_count=1,
            direct_prior_art_core_claim_count=0,
            relation_backed_core_claim_count=0,
            component_supported_core_claim_count=1,
            no_direct_match_core_claim_count=0,
            lower_order_supported_core_claim_count=1,
            direct_prior_art_core_fraction=0.0,
            relation_backed_core_fraction=0.0,
            component_supported_core_fraction=1.0,
            no_direct_match_core_fraction=0.0,
            lower_order_supported_core_fraction=1.0,
            higher_order_relational_gap_claim_count=1,
            lower_order_core_unique_work_count=1,
            directional_counterevidence_unique_work_count=0,
            search_coverage_sufficient=True,
            search_unique_work_count=10,
            search_abstract_work_count=5,
            claim_signals=[
                signal
            ],
            source_claim_ids=[
                "claim:c1"
            ],
            referenced_prior_art_work_ids=[
                "work:known"
            ],
            source_aggregate_warnings=[],
            reason_codes=[],
            interpretation=(
                "Higher-order relational gap with "
                "reviewed lower-order prior art."
            ),
        )
    )

    scientific_report = (
        ScientificDistinctivenessReport(
            report_id="scientific:test",
            report_sha256="scientific-sha",
            source_portfolio_id="portfolio:test",
            source_external_novelty_report_id=(
                "external:test"
            ),
            source_external_novelty_report_sha256=(
                "external-sha"
            ),
            source_query_plan_id="plan:test",
            source_query_plan_sha256="plan-sha",
            source_prior_art_packet_id="packet:test",
            source_prior_art_packet_sha256=(
                "packet-sha"
            ),
            source_searched_at_utc=(
                "2026-01-01T00:00:00+00:00"
            ),
            reviews=[
                scientific_review
            ],
            evidence_pattern_counts={
                (
                    "HIGHER_ORDER_RELATIONAL_GAP_"
                    "WITH_LOWER_ORDER_PRIOR_ART"
                ):
                    1
            },
            source_aggregate_warning_count=0,
        )
    )

    return (
        scientific_report,
        scientific_review,
        card,
        packet,
    )


def _assessment(
    level: str,
    *,
    claim_ids=None,
    work_ids=None,
):
    return SemanticDimensionAssessment(
        level=level,
        rationale="Fixture semantic rationale.",
        claim_ids=list(
            claim_ids or []
        ),
        work_ids=list(
            work_ids or []
        ),
    )


def _draft():
    return SemanticDistinctivenessDraft(
        hypothesis_id="hypothesis:test",

        conceptual_prior_art_density=(
            _assessment(
                "MODERATE",
                claim_ids=[
                    "claim:c1"
                ],
                work_ids=[
                    "work:known"
                ],
            )
        ),

        straightforward_reconstruction=(
            _assessment(
                "LOW",
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        mechanism_switch=(
            _assessment(
                "LOW",
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        ranking_or_regime_change=(
            _assessment(
                "MODERATE",
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        counterfactual_distinctiveness=(
            _assessment(
                "LOW",
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        evidence_role_complementarity=(
            _assessment(
                "MODERATE",
                claim_ids=[
                    "claim:c1"
                ],
                work_ids=[
                    "work:known"
                ],
            )
        ),

        confidence="MODERATE",
    )


def test_semantic_prompt_is_deterministic() -> None:
    (
        _,
        review,
        card,
        packet,
    ) = _fixture()

    assembler = (
        SemanticDistinctivenessPromptAssembler()
    )

    left = assembler.build(
        review,
        card,
        packet,
    )

    right = assembler.build(
        review,
        card,
        packet,
    )

    assert (
        left.prompt_sha256
        == right.prompt_sha256
    )

    assert left.allowed_claim_ids == (
        "claim:c1",
    )

    assert left.allowed_work_ids == (
        "work:known",
    )


def test_semantic_prompt_exposes_reviewed_evidence() -> None:
    (
        _,
        review,
        card,
        packet,
    ) = _fixture()

    prompt = (
        SemanticDistinctivenessPromptAssembler()
        .build(
            review,
            card,
            packet,
        )
    )

    assert (
        "Known lower-order relation"
        in prompt.user_prompt
    )

    assert (
        "LOWER_ORDER_RELATION_PRIOR_ART"
        in prompt.user_prompt
    )

    assert (
        "Variable A changes response Y"
        in prompt.user_prompt
    )


def test_semantic_compiler_accepts_bounded_references() -> None:
    (
        report,
        review,
        card,
        packet,
    ) = _fixture()

    prompt = (
        SemanticDistinctivenessPromptAssembler()
        .build(
            review,
            card,
            packet,
        )
    )

    result = (
        compile_semantic_distinctiveness_review(
            scientific_report=report,
            scientific_review=review,
            prompt=prompt,
            draft=_draft(),
            backend_name="fixture",
            requested_model="fixture:model",
            served_model="fixture:model",
            review_pass_index=1,
        )
    )

    assert result.overall_tier == "MODERATE"

    assert result.diagnostic_only is True
    assert result.retrieval_performed is False
    assert result.action_policy_applied is False
    assert (
        result.scientific_selection_changed
        is False
    )
    assert result.planner_changed is False
    assert result.novelty_status_changed is False

    assert result.referenced_claim_ids == [
        "claim:c1"
    ]

    assert (
        result.referenced_prior_art_work_ids
        == [
            "work:known"
        ]
    )


def test_semantic_compiler_rejects_unknown_work_reference() -> None:
    (
        report,
        review,
        card,
        packet,
    ) = _fixture()

    prompt = (
        SemanticDistinctivenessPromptAssembler()
        .build(
            review,
            card,
            packet,
        )
    )

    draft = _draft()

    draft = draft.model_copy(
        update={
            "mechanism_switch":
                SemanticDimensionAssessment(
                    level="HIGH",
                    rationale="Invalid work reference.",
                    claim_ids=[
                        "claim:c1"
                    ],
                    work_ids=[
                        "work:missing"
                    ],
                )
        }
    )

    with pytest.raises(
        ValueError,
        match="unknown work IDs",
    ):
        compile_semantic_distinctiveness_review(
            scientific_report=report,
            scientific_review=review,
            prompt=prompt,
            draft=draft,
            backend_name="fixture",
            requested_model="fixture:model",
            served_model="fixture:model",
            review_pass_index=1,
        )


def test_semantic_compiler_rejects_unknown_claim_reference() -> None:
    (
        report,
        review,
        card,
        packet,
    ) = _fixture()

    prompt = (
        SemanticDistinctivenessPromptAssembler()
        .build(
            review,
            card,
            packet,
        )
    )

    draft = _draft()

    draft = draft.model_copy(
        update={
            "ranking_or_regime_change":
                SemanticDimensionAssessment(
                    level="HIGH",
                    rationale="Invalid claim reference.",
                    claim_ids=[
                        "claim:missing"
                    ],
                    work_ids=[],
                )
        }
    )

    with pytest.raises(
        ValueError,
        match="unknown claim IDs",
    ):
        compile_semantic_distinctiveness_review(
            scientific_report=report,
            scientific_review=review,
            prompt=prompt,
            draft=draft,
            backend_name="fixture",
            requested_model="fixture:model",
            served_model="fixture:model",
            review_pass_index=1,
        )


def test_semantic_draft_schema_is_strict_structured_output_compatible() -> None:
    schema = (
        SemanticDistinctivenessDraft
        .model_json_schema()
    )

    problems = []

    def walk(
        node,
        path="$",
    ):
        if isinstance(
            node,
            dict,
        ):
            if (
                node.get("type")
                == "object"
                and isinstance(
                    node.get("properties"),
                    dict,
                )
            ):
                properties = set(
                    node["properties"]
                )

                required = set(
                    node.get(
                        "required",
                        [],
                    )
                )

                missing = (
                    properties
                    - required
                )

                if missing:
                    problems.append(
                        (
                            path,
                            "missing_required",
                            sorted(
                                missing
                            ),
                        )
                    )

                if (
                    node.get(
                        "additionalProperties"
                    )
                    is not False
                ):
                    problems.append(
                        (
                            path,
                            "additionalProperties",
                            node.get(
                                "additionalProperties",
                                "<missing>",
                            ),
                        )
                    )

            for key, value in (
                node.items()
            ):
                walk(
                    value,
                    f"{path}.{key}",
                )

        elif isinstance(
            node,
            list,
        ):
            for index, value in enumerate(
                node
            ):
                walk(
                    value,
                    f"{path}[{index}]",
                )

    walk(
        schema
    )

    assert problems == []


def test_semantic_reference_repair_prompt_is_bounded() -> None:
    (
        _,
        review,
        card,
        packet,
    ) = _fixture()

    assembler = (
        SemanticDistinctivenessPromptAssembler()
    )

    prompt = assembler.build(
        review,
        card,
        packet,
    )

    invalid = _draft().model_copy(
        update={
            "mechanism_switch":
                SemanticDimensionAssessment(
                    level="HIGH",
                    rationale=(
                        "Invalid external reference."
                    ),
                    claim_ids=[
                        "claim:c1"
                    ],
                    work_ids=[
                        "work:hallucinated"
                    ],
                )
        }
    )

    repair = (
        assembler
        .build_reference_validation_repair(
            original_prompt=
                prompt,

            previous_draft=
                invalid,

            issues=[
                (
                    "semantic dimension references "
                    "unknown work IDs: "
                    "['work:hallucinated']"
                )
            ],
        )
    )

    assert (
        repair.allowed_claim_ids
        == prompt.allowed_claim_ids
    )

    assert (
        repair.allowed_work_ids
        == prompt.allowed_work_ids
    )

    assert (
        repair.prompt_sha256
        != prompt.prompt_sha256
    )

    assert (
        "work:hallucinated"
        in repair.user_prompt
    )

    assert (
        "work_ids may contain ONLY IDs "
        "from allowed_work_ids"
        in repair.user_prompt
    )

    assert (
        "Do NOT retrieve, invent, recall"
        in repair.user_prompt
    )

    assert (
        "ALLOWED CLAIM IDS — COPY EXACTLY\n"
        "--------------------------------\n"
        "- claim:c1"
        in repair.user_prompt
    )

    assert (
        "ALLOWED WORK IDS — COPY EXACTLY\n"
        "-------------------------------\n"
        "- work:known"
        in repair.user_prompt
    )

    assert (
        "opaque provenance keys"
        in repair.user_prompt
    )


def test_semantic_compiler_records_reference_repair_provenance() -> None:
    (
        report,
        review,
        card,
        packet,
    ) = _fixture()

    prompt = (
        SemanticDistinctivenessPromptAssembler()
        .build(
            review,
            card,
            packet,
        )
    )

    result = (
        compile_semantic_distinctiveness_review(
            scientific_report=
                report,

            scientific_review=
                review,

            prompt=
                prompt,

            draft=
                _draft(),

            backend_name=
                "fixture",

            requested_model=
                "fixture:model",

            served_model=
                "fixture:model",

            review_pass_index=
                1,

            reference_contract_repair_count=
                1,

            reference_contract_repair_issues=[
                (
                    "semantic dimension references "
                    "unknown work IDs: "
                    "['work:hallucinated']"
                )
            ],
        )
    )

    assert (
        result.reference_contract_repair_count
        == 1
    )

    assert len(
        result.reference_contract_repair_issues
    ) == 1


def test_semantic_dimension_exact_duplicates_are_ordered_deduplicated() -> None:
    row = SemanticDimensionAssessment(
        level="MODERATE",
        rationale=(
            "Exact duplicate references are serialization noise."
        ),
        claim_ids=[
            "claim:c1",
            "claim:c1",
            "claim:c2",
            "claim:c1",
        ],
        work_ids=[
            "work:known",
            "work:known",
            "work:second",
            "work:known",
        ],
    )

    assert row.claim_ids == [
        "claim:c1",
        "claim:c2",
    ]

    assert row.work_ids == [
        "work:known",
        "work:second",
    ]


def _aggregation_draft(
    *,
    density="MODERATE",
    reconstruction="LOW",
    mechanism="LOW",
    ranking="LOW",
    counterfactual="LOW",
    complementarity="MODERATE",
):
    return SemanticDistinctivenessDraft(
        hypothesis_id="hypothesis:test",

        conceptual_prior_art_density=(
            _assessment(
                density,
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        straightforward_reconstruction=(
            _assessment(
                reconstruction,
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        mechanism_switch=(
            _assessment(
                mechanism,
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        ranking_or_regime_change=(
            _assessment(
                ranking,
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        counterfactual_distinctiveness=(
            _assessment(
                counterfactual,
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        evidence_role_complementarity=(
            _assessment(
                complementarity,
                claim_ids=[
                    "claim:c1"
                ],
            )
        ),

        confidence="HIGH",
    )


def test_semantic_aggregation_direct_prior_art_is_low() -> None:
    (
        _,
        review,
        _,
        _,
    ) = _fixture()

    review = review.model_copy(
        update={
            "evidence_pattern":
                "DIRECT_PRIOR_ART_SATURATED"
        }
    )

    tier, reasons = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                review,

            draft=
                _aggregation_draft(
                    ranking="HIGH",
                    complementarity="HIGH",
                ),
        )
    )

    assert tier == "LOW"

    assert (
        "DIRECT_PRIOR_ART_SATURATED"
        in reasons
    )


def test_semantic_aggregation_density_does_not_veto_primary_structure() -> None:
    (
        _,
        review,
        _,
        _,
    ) = _fixture()

    tier, reasons = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                review,

            draft=
                _aggregation_draft(
                    density="HIGH",
                    reconstruction="LOW",
                    ranking="HIGH",
                    complementarity="LOW",
                ),
        )
    )

    assert tier == "HIGH"

    assert (
        "HIGH_PRIMARY_STRUCTURAL_FEATURE"
        in reasons
    )

    assert (
        "PRIMARY_HIGH:ranking_or_regime_change"
        in reasons
    )

def test_semantic_aggregation_complementarity_alone_is_not_high() -> None:
    (
        _,
        review,
        _,
        _,
    ) = _fixture()

    tier, reasons = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                review,

            draft=
                _aggregation_draft(
                    density="MODERATE",
                    reconstruction="LOW",
                    complementarity="HIGH",
                ),
        )
    )

    assert tier == "MODERATE"

    assert (
        "NO_DECISIVE_LOW_OR_HIGH_RULE"
        in reasons
    )

    assert (
        "HIGH_PRIMARY_STRUCTURAL_FEATURE"
        not in reasons
    )


def test_semantic_aggregation_multiple_strong_features_is_high() -> None:
    (
        _,
        review,
        _,
        _,
    ) = _fixture()

    tier, reasons = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                review,

            draft=
                _aggregation_draft(
                    reconstruction="LOW",
                    ranking="HIGH",
                    counterfactual="HIGH",
                ),
        )
    )

    assert tier == "HIGH"

    assert (
        "HIGH_PRIMARY_STRUCTURAL_FEATURE"
        in reasons
    )

    assert (
        "PRIMARY_HIGH:ranking_or_regime_change"
        in reasons
    )

    assert (
        "PRIMARY_HIGH:counterfactual_distinctiveness"
        in reasons
    )


def test_semantic_aggregation_high_reconstruction_without_strong_structure_is_low() -> None:
    (
        _,
        review,
        _,
        _,
    ) = _fixture()

    tier, reasons = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                review,

            draft=
                _aggregation_draft(
                    reconstruction="HIGH",
                    mechanism="LOW",
                    ranking="LOW",
                    counterfactual="LOW",
                ),
        )
    )

    assert tier == "LOW"

    assert (
        "HIGH_STRAIGHTFORWARD_RECONSTRUCTION"
        in reasons
    )


def test_semantic_aggregation_search_coverage_limited_is_indeterminate() -> None:
    (
        _,
        review,
        _,
        _,
    ) = _fixture()

    review = review.model_copy(
        update={
            "evidence_pattern":
                "SEARCH_COVERAGE_LIMITED"
        }
    )

    tier, reasons = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                review,

            draft=
                _aggregation_draft(
                    density="LOW",
                    reconstruction="LOW",
                    ranking="HIGH",
                    counterfactual="HIGH",
                    complementarity="HIGH",
                ),
        )
    )

    assert tier == "INDETERMINATE"

    assert (
        "SEARCH_COVERAGE_LIMITED_NO_POSITIVE_LOW_EVIDENCE"
        in reasons
    )


def test_semantic_aggregation_high_reconstruction_caps_primary_structure() -> None:
    (
        _,
        review,
        _,
        _,
    ) = _fixture()

    tier, reasons = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                review,

            draft=
                _aggregation_draft(
                    density="MODERATE",
                    reconstruction="HIGH",
                    ranking="HIGH",
                    complementarity="HIGH",
                ),
        )
    )

    assert tier == "MODERATE"

    assert (
        "HIGH_STRAIGHTFORWARD_RECONSTRUCTION_CAP"
        in reasons
    )

    assert (
        "PRIMARY_HIGH:ranking_or_regime_change"
        in reasons
    )


def test_semantic_aggregation_complementarity_not_required_for_high() -> None:
    (
        _,
        review,
        _,
        _,
    ) = _fixture()

    tier, reasons = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                review,

            draft=
                _aggregation_draft(
                    density="HIGH",
                    reconstruction="MODERATE",
                    counterfactual="HIGH",
                    complementarity="LOW",
                ),
        )
    )

    assert tier == "HIGH"

    assert (
        "PRIMARY_HIGH:counterfactual_distinctiveness"
        in reasons
    )
