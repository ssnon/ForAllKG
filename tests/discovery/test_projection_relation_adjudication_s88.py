from __future__ import annotations

from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorRelationProjection,
    GroundedProjectionFactorBinding,
)
from pipeline_core.discovery.projection_relation_adjudication import (
    ProjectionRelationClaimCandidateSet,
    ProjectionRelationClaimReviewDraft,
    ProjectionRelationMatchDraft,
    ProjectionRelationWorkCandidate,
    compile_projection_relation_claim_review,
)


def _projection(
    projection_id: str,
    kind: str,
    order: int,
) -> GroundedFactorRelationProjection:
    bindings = [
        GroundedProjectionFactorBinding(
            factor_id=f"f{index}",
            group_label=f"factor-{index}",
            identity_basis_tokens=[f"factor{index}"],
            exclusive_identity_basis_tokens=[f"factor{index}"],
            exact_source_aliases=[f"factor {index}"],
        )
        for index in range(order)
    ]
    return GroundedFactorRelationProjection(
        projection_id=projection_id,
        relation_ir_id="relation:1",
        hypothesis_id="h1",
        claim_id="claim:1",
        projection_kind=kind,
        source_identity_term="synthetic identity",
        retained_factor_ids=[
            row.factor_id for row in bindings
        ],
        omitted_factor_ids=[],
        factor_order=order,
        factor_bindings=bindings,
        endpoint_terms=["endpoint A", "endpoint B"],
        scope_terms=[],
        directional_terms=[],
        canonical_search_terms=["endpoint A", "endpoint B"],
        canonical_search_query="endpoint A endpoint B",
        exact_source_query_variants=["endpoint A endpoint B"],
        exact_source_query_variant_count=1,
        relation_typing_status="READY",
        factorization_status="READY",
        eligible_for_future_typed_retrieval=True,
        reason_codes=[],
    )


BASE = _projection("p-base", "BASE_RELATION", 0)
LOWER = _projection(
    "p-lower",
    "LOWER_ORDER_FACTOR_SUBSET",
    1,
)
FULL = _projection(
    "p-full",
    "FULL_RELATION",
    2,
)
PROJECTIONS = {
    row.projection_id: row
    for row in (BASE, LOWER, FULL)
}


def _candidate(
    *,
    work_id: str = "w1",
    abstract: str | None = (
        "Endpoint A is directly associated with endpoint B "
        "under the full factor condition."
    ),
    typed_state: str = "COMPATIBLE",
    counter_modes: list[str] | None = None,
) -> ProjectionRelationWorkCandidate:
    return ProjectionRelationWorkCandidate(
        review_work_id=work_id,
        source_work_ids=[work_id],
        title="Test work",
        abstract=abstract,
        typed_compatibility_state=typed_state,
        relation_type_labels=["typed"],
        document_type_labels=["typed"],
        supporting_projection_ids=[
            "p-base",
            "p-lower",
            "p-full",
        ],
        supporting_projection_kinds=[
            "BASE_RELATION",
            "LOWER_ORDER_FACTOR_SUBSET",
            "FULL_RELATION",
        ],
        supporting_factor_orders=[0, 1, 2],
        counterevidence_projection_ids=["p-full"],
        counterevidence_modes=counter_modes or ["DECOUPLING"],
        selected_lanes=[
            "SUPPORTING_PRIOR_ART",
            "COUNTEREVIDENCE_PRIOR_ART",
        ],
        selection_score=20.0,
    )


def _set(
    candidate: ProjectionRelationWorkCandidate,
) -> ProjectionRelationClaimCandidateSet:
    return ProjectionRelationClaimCandidateSet(
        hypothesis_id="h1",
        claim_id="claim:1",
        relation_ir_id="relation:1",
        claim_text="Full typed relation.",
        endpoint_terms=["endpoint A", "endpoint B"],
        identity_terms=["synthetic identity"],
        projection_ids=["p-base", "p-lower", "p-full"],
        full_projection_ids=["p-full"],
        lower_order_projection_ids=["p-base", "p-lower"],
        candidates=[candidate],
        candidate_count=1,
        abstract_candidate_count=int(bool(candidate.abstract)),
        typed_identity_excluded_work_count=0,
        source_unique_work_count=1,
        max_review_works=20,
    )


def _compile(
    candidate: ProjectionRelationWorkCandidate,
    match: ProjectionRelationMatchDraft,
):
    return compile_projection_relation_claim_review(
        candidate_set=_set(candidate),
        draft=ProjectionRelationClaimReviewDraft(
            matches=[match],
            interpretation="bounded review",
        ),
        projection_by_id=PROJECTIONS,
    )


def test_direct_prior_art_requires_full_projection_and_exact_abstract_span():
    candidate = _candidate()
    candidate.endpoint_supported_count = 2
    candidate.all_endpoints_supported = True
    candidate.relation_anchor_tier = "PAIR_OR_MULTI_ENDPOINT_ANCHORED"

    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="DIRECT_PRIOR_ART",
            confidence=0.9,
            basis_projection_ids=["p-full"],
            evidence_span=(
                "Endpoint A is directly associated with endpoint B"
            ),
            rationale="same full relation",
        ),
    )

    assert review.direct_prior_art_work_ids == ["w1"]
    assert review.matches[0].relationship == "DIRECT_PRIOR_ART"
    assert review.relation_state == "DIRECT_RELATION_FOUND"


def test_direct_prior_art_requires_pair_or_multi_endpoint_anchor():
    candidate = _candidate()
    candidate.endpoint_supported_count = 1
    candidate.all_endpoints_supported = False
    candidate.relation_anchor_tier = "PARTIAL_ENDPOINT_ANCHORED"

    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="DIRECT_PRIOR_ART",
            confidence=0.9,
            basis_projection_ids=["p-full"],
            evidence_span=(
                "Endpoint A is directly associated with endpoint B"
            ),
            rationale=(
                "model proposed direct prior art despite only one "
                "deterministically supported endpoint"
            ),
        ),
    )

    assert review.direct_prior_art_work_ids == []
    assert review.matches[0].relationship == "PARTIAL_PRIOR_ART"
    assert (
        "direct_prior_art_requires_pair_or_multi_endpoint_anchor"
        in review.matches[0].deterministic_reason_codes
    )


def test_strong_relation_without_exact_abstract_span_fails_closed():
    candidate = _candidate()
    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="DIRECT_PRIOR_ART",
            confidence=0.9,
            basis_projection_ids=["p-full"],
            evidence_span="paraphrased relation not in abstract",
            rationale="unsupported paraphrase",
        ),
    )

    assert review.matches[0].relationship == "INSUFFICIENT_METADATA"
    assert (
        "strong_relation_downgraded_without_exact_abstract_span"
        in review.matches[0].deterministic_reason_codes
    )


def test_typed_identity_conflict_blocks_strong_relation():
    candidate = _candidate(
        typed_state="TYPE_IDENTITY_CONFLICT"
    )
    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="DIRECT_PRIOR_ART",
            confidence=0.9,
            basis_projection_ids=["p-full"],
            evidence_span=(
                "Endpoint A is directly associated with endpoint B"
            ),
            rationale="wrong scientific identity",
        ),
    )

    assert review.matches[0].relationship == "UNRELATED"
    assert (
        "strong_relation_downgraded_for_typed_identity_conflict"
        in review.matches[0].deterministic_reason_codes
    )


def test_lower_order_requires_lower_order_projection_basis():
    candidate = _candidate()
    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="LOWER_ORDER_RELATION_PRIOR_ART",
            confidence=0.85,
            basis_projection_ids=["p-full"],
            evidence_span=(
                "Endpoint A is directly associated with endpoint B"
            ),
            rationale="model cited only full projection",
        ),
    )

    assert review.matches[0].relationship == "COMPONENT_ONLY"
    assert (
        "lower_order_prior_art_requires_lower_order_projection_basis"
        in review.matches[0].deterministic_reason_codes
    )


def test_directional_counterevidence_requires_retrieved_counter_mode():
    candidate = _candidate(
        counter_modes=["DECOUPLING"]
    )
    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="DIRECTIONAL_COUNTEREVIDENCE",
            confidence=0.8,
            basis_projection_ids=["p-full"],
            counterevidence_modes=["OPPOSITE_DIRECTION"],
            evidence_span=(
                "Endpoint A is directly associated with endpoint B"
            ),
            rationale="mode was not retrieved for this work",
        ),
    )

    assert review.matches[0].relationship == "COMPONENT_ONLY"
    assert (
        "directional_counterevidence_requires_retrieved_counterevidence_provenance"
        in review.matches[0].deterministic_reason_codes
    )
    assert (
        "unretrieved_counterevidence_mode_dropped"
        in review.matches[0].deterministic_reason_codes
    )


def test_scope_mismatched_conflict_becomes_contextual_conflict():
    candidate = _candidate(
        typed_state="DOMAIN_OR_SCOPE_MISMATCH"
    )
    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="CONFLICTING_PRIOR_ART",
            confidence=0.85,
            basis_projection_ids=["p-full"],
            counterevidence_modes=["DECOUPLING"],
            evidence_span=(
                "Endpoint A is directly associated with endpoint B"
            ),
            rationale="opposing neighboring-scope result",
        ),
    )

    assert review.matches[0].relationship == "CONTEXTUAL_CONFLICT"
    assert review.contextual_conflict_work_ids == ["w1"]
    assert (
        "conflicting_prior_art_downgraded_for_scope_mismatch"
        in review.matches[0].deterministic_reason_codes
    )



def test_candidate_selection_hard_excludes_ambiguous_document_identity():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _balanced_select,
    )

    bad = _candidate(
        work_id="cultural-hotspot",
        typed_state="AMBIGUOUS_DOCUMENT_IDENTITY",
    )
    good = _candidate(
        work_id="sers-paper",
        typed_state="DOMAIN_COMPATIBLE",
    )
    good.endpoint_lexical_coverage = 0.5

    selected = _balanced_select(
        [bad, good],
        max_review_works=2,
    )

    assert [
        row.review_work_id
        for row in selected
    ] == ["sers-paper"]


def test_candidate_selection_prefers_abstract_backed_record_over_title_only_score():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _balanced_select,
    )

    abstracted = _candidate(
        work_id="abstracted",
        typed_state="DOMAIN_COMPATIBLE",
        abstract="A real abstract.",
    )
    abstracted.selection_score = 10.0

    title_only = _candidate(
        work_id="title-only",
        typed_state="DOMAIN_COMPATIBLE",
        abstract=None,
    )
    title_only.selection_score = 100.0

    selected = _balanced_select(
        [title_only, abstracted],
        max_review_works=1,
    )

    assert selected[0].review_work_id == "abstracted"


def test_candidate_compatibility_rejects_untyped_cultural_hotspot_collision():
    from domains.registry import get_domain_profile
    from pipeline_core.discovery.projection_relation_adjudication import (
        _candidate_compatibility,
    )
    from pipeline_core.discovery.scientific_relation_ir import (
        ScientificConceptIR,
        ScientificRelationIR,
    )

    def concept(cid, role, text, index):
        return ScientificConceptIR(
            concept_id=cid,
            role=role,
            source_field=role.lower(),
            source_index=index,
            surface_text=text,
            normalized_text=text.casefold(),
            lexical_tokens=text.casefold().split(),
            type_labels=(
                ["electromagnetic_plasmonic_hotspot"]
                if "hotspot" in text
                else []
            ),
            ambiguity_labels=[],
            literal_in_claim_text=True,
            literal_in_required_bridge=(
                None if role == "OBSERVABLE" else True
            ),
        )

    relation = ScientificRelationIR(
        relation_ir_id="relation:sers",
        hypothesis_id="h1",
        claim_id="claim:sers",
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text=(
            "hotspot population stability is associated with "
            "calibration transfer error"
        ),
        endpoint_concepts=[
            concept(
                "e1",
                "RELATION_ENDPOINT",
                "hotspot population stability",
                0,
            ),
            concept(
                "e2",
                "RELATION_ENDPOINT",
                "calibration transfer error",
                1,
            ),
        ],
        identity_concepts=[
            concept(
                "i1",
                "BRANCH_IDENTITY",
                "plasmonic hotspot accessibility",
                0,
            ),
        ],
        observable_concept=concept(
            "o1",
            "OBSERVABLE",
            "calibration transfer error",
            0,
        ),
        relation_type_labels=[
            "electromagnetic_plasmonic_hotspot"
        ],
        relation_domain_labels=[],
        relation_scope_features=[
            "electromagnetic_enhancement"
        ],
        typing_status="READY",
        reason_codes=[],
    )

    state, *_rest = _candidate_compatibility(
        relation=relation,
        document=(
            "Australian cultural and creative activity: "
            "population and hotspot analysis."
        ),
        domain_profile=get_domain_profile("sers_au_ag"),
    )

    assert state == "AMBIGUOUS_DOCUMENT_IDENTITY"



def test_endpoint_discriminative_coverage_removes_shared_generic_tokens():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _endpoint_discriminative_evidence,
        _relation_anchor_state,
    )
    from pipeline_core.discovery.scientific_relation_ir import (
        ScientificConceptIR,
        ScientificRelationIR,
    )

    def concept(cid, role, text, index):
        return ScientificConceptIR(
            concept_id=cid,
            role=role,
            source_field=role.lower(),
            source_index=index,
            surface_text=text,
            normalized_text=text.casefold(),
            lexical_tokens=text.casefold().split(),
            type_labels=[],
            ambiguity_labels=[],
            literal_in_claim_text=True,
            literal_in_required_bridge=(
                None if role == "OBSERVABLE" else True
            ),
        )

    relation = ScientificRelationIR(
        relation_ir_id="relation:h2",
        hypothesis_id="h2",
        claim_id="claim:h2",
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text="test",
        endpoint_concepts=[
            concept(
                "e1",
                "RELATION_ENDPOINT",
                "spatial heterogeneity of plasmonically active environments",
                0,
            ),
            concept(
                "e2",
                "RELATION_ENDPOINT",
                "spatial heterogeneity of Raman reporter oxidation",
                1,
            ),
        ],
        identity_concepts=[],
        observable_concept=concept(
            "o1",
            "OBSERVABLE",
            "oxidation",
            0,
        ),
        relation_type_labels=[],
        relation_domain_labels=[],
        relation_scope_features=[],
        typing_status="READY",
        reason_codes=[],
    )

    coverages, matches, required = _endpoint_discriminative_evidence(
        relation,
        (
            "High-resolution Raman imaging reveals spatial heterogeneity "
            "of heme oxidation in red blood cells."
        ),
    )

    assert coverages[0] == 0.0
    assert coverages[1] > 0.0
    assert matches[0] == 0
    assert matches[1] >= required[1]

    count, all_supported, tier = _relation_anchor_state(
        matches,
        required,
    )
    assert count == 1
    assert all_supported is False
    assert tier == "PARTIAL_ENDPOINT_ANCHORED"


def test_balanced_selection_prioritizes_pair_anchored_relation_candidate():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _balanced_select,
    )

    pair = _candidate(
        work_id="pair",
        typed_state="DOMAIN_COMPATIBLE",
        abstract="abstract",
    )
    pair.selection_score = 10.0
    pair.endpoint_supported_count = 2
    pair.all_endpoints_supported = True
    pair.relation_anchor_tier = "PAIR_OR_MULTI_ENDPOINT_ANCHORED"

    broad = _candidate(
        work_id="broad",
        typed_state="DOMAIN_COMPATIBLE",
        abstract="abstract",
    )
    broad.selection_score = 100.0
    broad.endpoint_supported_count = 1
    broad.all_endpoints_supported = False
    broad.relation_anchor_tier = "PARTIAL_ENDPOINT_ANCHORED"

    selected = _balanced_select(
        [broad, pair],
        max_review_works=1,
    )

    assert selected[0].review_work_id == "pair"


def test_relation_anchor_state_requires_each_endpoint_to_be_supported():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _relation_anchor_state,
    )

    count, all_supported, tier = _relation_anchor_state(
        [1, 0],
        [2, 2],
    )
    assert count == 0
    assert all_supported is False
    assert tier == "CONTEXT_ONLY"

    count, all_supported, tier = _relation_anchor_state(
        [2, 0],
        [2, 2],
    )
    assert count == 1
    assert all_supported is False
    assert tier == "PARTIAL_ENDPOINT_ANCHORED"

    count, all_supported, tier = _relation_anchor_state(
        [2, 2],
        [2, 2],
    )
    assert count == 2
    assert all_supported is True
    assert tier == "PAIR_OR_MULTI_ENDPOINT_ANCHORED"



def test_single_token_overlap_does_not_strongly_support_multi_token_endpoint():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _relation_anchor_state,
    )

    count, all_supported, tier = _relation_anchor_state(
        [1, 1],
        [2, 2],
    )

    assert count == 0
    assert all_supported is False
    assert tier == "CONTEXT_ONLY"


def test_in_domain_partial_candidate_outranks_neighboring_pair_candidate():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _balanced_select,
    )

    in_domain = _candidate(
        work_id="in-domain",
        typed_state="DOMAIN_COMPATIBLE",
        abstract="abstract",
    )
    in_domain.selection_score = 20.0
    in_domain.endpoint_supported_count = 1
    in_domain.relation_anchor_tier = "PARTIAL_ENDPOINT_ANCHORED"

    neighboring = _candidate(
        work_id="neighboring",
        typed_state="NEIGHBORING_SCOPE",
        abstract="abstract",
    )
    neighboring.selection_score = 100.0
    neighboring.endpoint_supported_count = 2
    neighboring.relation_anchor_tier = "PAIR_OR_MULTI_ENDPOINT_ANCHORED"

    selected = _balanced_select(
        [neighboring, in_domain],
        max_review_works=1,
    )

    assert selected[0].review_work_id == "in-domain"



def test_semantic_second_pass_counterevidence_provenance_allows_directional_review():
    candidate = _candidate(
        counter_modes=[],
    )
    candidate.second_pass_roles = [
        "SEMANTIC_COUNTEREVIDENCE"
    ]

    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="DIRECTIONAL_COUNTEREVIDENCE",
            confidence=0.82,
            basis_projection_ids=["p-base"],
            counterevidence_modes=[],
            second_pass_roles=[
                "SEMANTIC_COUNTEREVIDENCE"
            ],
            evidence_span=(
                "Endpoint A is directly associated with endpoint B"
            ),
            rationale=(
                "abstract-backed directional challenge retrieved "
                "through semantic counterevidence search"
            ),
        ),
    )

    assert review.matches[0].relationship == (
        "DIRECTIONAL_COUNTEREVIDENCE"
    )
    assert review.matches[0].second_pass_roles == [
        "SEMANTIC_COUNTEREVIDENCE"
    ]
    assert review.directional_counterevidence_work_ids == [
        "w1"
    ]


def test_hallucinated_semantic_counterevidence_role_is_dropped_fail_closed():
    candidate = _candidate(
        counter_modes=[],
    )
    candidate.second_pass_roles = [
        "SEMANTIC_ENDPOINT_PAIR"
    ]

    review = _compile(
        candidate,
        ProjectionRelationMatchDraft(
            work_id="w1",
            relationship="DIRECTIONAL_COUNTEREVIDENCE",
            confidence=0.82,
            basis_projection_ids=["p-base"],
            counterevidence_modes=[],
            second_pass_roles=[
                "SEMANTIC_COUNTEREVIDENCE"
            ],
            evidence_span=(
                "Endpoint A is directly associated with endpoint B"
            ),
            rationale=(
                "model invented counterevidence provenance"
            ),
        ),
    )

    assert review.matches[0].relationship == "COMPONENT_ONLY"
    assert review.matches[0].second_pass_roles == []
    assert (
        "unretrieved_second_pass_role_dropped"
        in review.matches[0].deterministic_reason_codes
    )
    assert (
        "directional_counterevidence_requires_retrieved_counterevidence_provenance"
        in review.matches[0].deterministic_reason_codes
    )



def test_adjudication_coverage_distinguishes_presented_from_classified_records():
    first = _candidate(
        work_id="w1",
        typed_state="DOMAIN_COMPATIBLE",
    )
    second = _candidate(
        work_id="w2",
        typed_state="DOMAIN_COMPATIBLE",
    )

    candidate_set = ProjectionRelationClaimCandidateSet(
        hypothesis_id="h1",
        claim_id="claim:1",
        relation_ir_id="relation:1",
        claim_text="Full typed relation.",
        endpoint_terms=["endpoint A", "endpoint B"],
        identity_terms=["synthetic identity"],
        projection_ids=["p-base", "p-lower", "p-full"],
        full_projection_ids=["p-full"],
        lower_order_projection_ids=["p-base", "p-lower"],
        candidates=[first, second],
        candidate_count=2,
        abstract_candidate_count=2,
        typed_identity_excluded_work_count=0,
        source_unique_work_count=2,
        max_review_works=20,
    )

    review = compile_projection_relation_claim_review(
        candidate_set=candidate_set,
        draft=ProjectionRelationClaimReviewDraft(
            matches=[
                ProjectionRelationMatchDraft(
                    work_id="w1",
                    relationship="COMPONENT_ONLY",
                    confidence=0.8,
                    rationale="one returned record",
                )
            ],
            interpretation=(
                "The adjudicator omitted the second presented work."
            ),
        ),
        projection_by_id=PROJECTIONS,
    )

    assert review.presented_work_count == 2
    assert review.classified_work_count == 1
    assert review.reviewed_work_count == 1
    assert review.unclassified_work_ids == ["w2"]


def test_exhaustive_followup_reasks_only_omitted_presented_work_ids():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _review_claim_with_exhaustive_followup,
    )

    candidates = [
        _candidate(work_id=work_id, typed_state="DOMAIN_COMPATIBLE")
        for work_id in ("w1", "w2", "w3")
    ]
    candidate_set = ProjectionRelationClaimCandidateSet(
        hypothesis_id="h1",
        claim_id="claim:1",
        relation_ir_id="relation:1",
        claim_text="Full typed relation.",
        endpoint_terms=["endpoint A", "endpoint B"],
        identity_terms=["synthetic identity"],
        projection_ids=["p-base", "p-lower", "p-full"],
        full_projection_ids=["p-full"],
        lower_order_projection_ids=["p-base", "p-lower"],
        candidates=candidates,
        candidate_count=3,
        abstract_candidate_count=3,
        typed_identity_excluded_work_count=0,
        source_unique_work_count=3,
        max_review_works=20,
    )

    class OnePerRoundBackend:
        backend_name = "test"
        model_name = "test"

        def __init__(self):
            self.presented_ids = []

        def review(self, *, candidate_set, projection_by_id):
            del projection_by_id
            self.presented_ids.append(
                [row.review_work_id for row in candidate_set.candidates]
            )
            work_id = candidate_set.candidates[0].review_work_id
            return ProjectionRelationClaimReviewDraft(
                matches=[
                    ProjectionRelationMatchDraft(
                        work_id=work_id,
                        relationship="UNRELATED",
                        confidence=0.8,
                        rationale="bounded test classification",
                    )
                ],
                interpretation="bounded round",
            )

    backend = OnePerRoundBackend()
    draft, calls = _review_claim_with_exhaustive_followup(
        candidate_set=candidate_set,
        projection_by_id=PROJECTIONS,
        backend=backend,
        max_exhaustive_rounds=3,
    )

    assert calls == 3
    assert backend.presented_ids == [
        ["w1", "w2", "w3"],
        ["w2", "w3"],
        ["w3"],
    ]

    review = compile_projection_relation_claim_review(
        candidate_set=candidate_set,
        draft=draft,
        projection_by_id=PROJECTIONS,
    )
    assert review.classified_work_count == 3
    assert review.unclassified_work_ids == []


def test_exhaustive_followup_stays_fail_closed_after_round_budget():
    from pipeline_core.discovery.projection_relation_adjudication import (
        _review_claim_with_exhaustive_followup,
    )

    candidates = [
        _candidate(work_id=work_id, typed_state="DOMAIN_COMPATIBLE")
        for work_id in ("w1", "w2", "w3", "w4")
    ]
    candidate_set = ProjectionRelationClaimCandidateSet(
        hypothesis_id="h1",
        claim_id="claim:1",
        relation_ir_id="relation:1",
        claim_text="Full typed relation.",
        endpoint_terms=["endpoint A", "endpoint B"],
        identity_terms=["synthetic identity"],
        projection_ids=["p-base", "p-lower", "p-full"],
        full_projection_ids=["p-full"],
        lower_order_projection_ids=["p-base", "p-lower"],
        candidates=candidates,
        candidate_count=4,
        abstract_candidate_count=4,
        typed_identity_excluded_work_count=0,
        source_unique_work_count=4,
        max_review_works=20,
    )

    class OnePerRoundBackend:
        backend_name = "test"
        model_name = "test"

        def review(self, *, candidate_set, projection_by_id):
            del projection_by_id
            work_id = candidate_set.candidates[0].review_work_id
            return ProjectionRelationClaimReviewDraft(
                matches=[
                    ProjectionRelationMatchDraft(
                        work_id=work_id,
                        relationship="UNRELATED",
                        confidence=0.8,
                        rationale="bounded test classification",
                    )
                ],
                interpretation="bounded round",
            )

    draft, calls = _review_claim_with_exhaustive_followup(
        candidate_set=candidate_set,
        projection_by_id=PROJECTIONS,
        backend=OnePerRoundBackend(),
        max_exhaustive_rounds=2,
    )
    assert calls == 2

    review = compile_projection_relation_claim_review(
        candidate_set=candidate_set,
        draft=draft,
        projection_by_id=PROJECTIONS,
    )
    assert review.classified_work_count == 2
    assert review.unclassified_work_ids == ["w3", "w4"]
