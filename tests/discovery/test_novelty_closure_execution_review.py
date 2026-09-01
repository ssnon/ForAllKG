from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
    RankedPriorArtWork,
)
from pipeline_core.discovery.novelty_closure_execution import (
    build_closure_execution_plan,
)
from pipeline_core.discovery.novelty_closure_planner import (
    build_closure_retrieval_plan,
)
from pipeline_core.discovery.novelty_closure_review import (
    ClosureEvidenceMatchDraft,
    ClosureSlotReviewDraft,
    compile_closure_slot_review,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
)


def source_claim():
    return NoveltyResidueClaim(
        hypothesis_id="hypothesis:test",
        claim_id="claim:test",
        claim_text=(
            "A critical laser power Pc separates two distinct "
            "regimes of the spacing-to-SERS relationship."
        ),
        claim_kind="distinctive_prediction",
        prior_art_status="NO_DIRECT_MATCH_FOUND",
        disposition="RESIDUAL",
        is_residue=True,

        distinguishing_terms=(
            "critical laser power",
            "two regimes",
        ),
        prior_art_identity_terms=(
            "laser power",
        ),
        relation_nucleus_terms=(
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ),

        required_bridge=(
            "Laser power drives a transition at Pc that changes "
            "how interparticle spacing maps to measured SERS "
            "enhancement."
        ),
        predicted_observation=(
            "Two spacing-to-SERS regimes appear below and "
            "above Pc."
        ),
        falsification_condition=(
            "No reproducible power-dependent regime boundary "
            "is observed."
        ),

        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
    )


def execution_plan():
    closure = build_closure_retrieval_plan(
        source_claim()
    )

    return build_closure_execution_plan(
        source_portfolio_id="portfolio:test",
        closure_plan=closure,
    )


def candidate_set(
    target_id,
    *,
    abstract_available=True,
):
    return ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:test",
        claim_id=target_id,
        ranked_works=[
            RankedPriorArtWork(
                work_id="work:1",
                relevance_score=0.9,
                semantic_similarity=0.9,
                lexical_coverage=0.8,
                reaction_domain_relevance=0.9,
                catalyst_scope_relevance=0.9,
                abstract_available=(
                    abstract_available
                ),
            )
        ],
    )


def packet(
    plan,
    target,
    *,
    abstract="Explicit relation in abstract.",
    success=True,
):
    query = next(
        row
        for row in plan.queries
        if row.claim_id == target.target_id
    )

    return PriorArtPacket(
        packet_id="packet:test",
        packet_sha256="sha:test",
        source_portfolio_id=(
            plan.source_portfolio_id
        ),
        source_query_plan_id=(
            plan.plan_id
        ),
        searched_at_utc=(
            "2026-09-01T00:00:00+00:00"
        ),
        providers_requested=["fake"],
        works=[
            PriorArtWork(
                work_id="work:1",
                title="Test paper",
                abstract=abstract,
                providers=["fake"],
                retrieval_query_ids=[
                    query.query_id
                ],
                retrieval_claim_ids=[
                    target.target_id
                ],
            )
        ],
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="fake",
                success=success,
                result_count=(
                    1 if success else 0
                ),
            )
        ],
        raw_work_count=1,
        canonical_work_count=1,
    )


def positive_draft():
    return ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id="work:1",
                relationship=(
                    "ESTABLISHES_SLOT"
                ),
                confidence=0.9,
                rationale=(
                    "The abstract explicitly establishes "
                    "the slot relation."
                ),
            )
        ],
        interpretation=(
            "Bounded slot review."
        ),
    )


def test_execution_plan_has_four_unique_unassessed_targets():
    plan = execution_plan()

    assert len(plan.targets) == 4
    assert len(plan.queries) == 4

    assert len(
        {
            row.target_id
            for row in plan.targets
        }
    ) == 4

    assert all(
        row.evidence_status
        == "UNASSESSED"
        for row in plan.targets
    )


def test_query_claim_ids_are_target_ids():
    plan = execution_plan()

    target_ids = {
        row.target_id
        for row in plan.targets
    }

    assert {
        row.claim_id
        for row in plan.queries
    } == target_ids


def test_abstract_backed_positive_review_establishes_slot():
    plan = execution_plan()
    target = plan.targets[0]

    result = compile_closure_slot_review(
        target=target,
        draft=positive_draft(),
        candidates=candidate_set(
            target.target_id
        ),
        packet=packet(
            plan,
            target,
        ),
        plan=plan,
    )

    assert (
        result.evidence_state
        == "ESTABLISHED"
    )

    assert result.positive_work_ids == [
        "work:1"
    ]


def test_title_only_cannot_establish_slot():
    plan = execution_plan()
    target = plan.targets[0]

    result = compile_closure_slot_review(
        target=target,
        draft=positive_draft(),
        candidates=candidate_set(
            target.target_id,
            abstract_available=False,
        ),
        packet=packet(
            plan,
            target,
            abstract=None,
        ),
        plan=plan,
    )

    assert (
        result.evidence_state
        == "UNASSESSED"
    )

    assert (
        result.matches[0].relationship
        == "TITLE_ONLY_NEIGHBOR"
    )

    assert result.negative_coverage_sufficient is False
    assert (
        "insufficient_abstract_coverage_for_negative_closure"
        in result.reason_codes
    )


def test_failed_provider_query_is_unassessed_not_not_found():
    plan = execution_plan()
    target = plan.targets[0]

    result = compile_closure_slot_review(
        target=target,
        draft=ClosureSlotReviewDraft(
            matches=[],
            interpretation=(
                "Provider failed."
            ),
        ),
        candidates=ClaimPriorArtCandidateSet(
            hypothesis_id=(
                "hypothesis:test"
            ),
            claim_id=target.target_id,
            ranked_works=[],
        ),
        packet=PriorArtPacket(
            packet_id="packet:test",
            packet_sha256="sha:test",
            source_portfolio_id=(
                plan.source_portfolio_id
            ),
            source_query_plan_id=(
                plan.plan_id
            ),
            searched_at_utc=(
                "2026-09-01T00:00:00+00:00"
            ),
            providers_requested=["fake"],
            works=[],
            executions=[
                QueryExecution(
                    query_id=next(
                        q.query_id
                        for q in plan.queries
                        if (
                            q.claim_id
                            == target.target_id
                        )
                    ),
                    provider="fake",
                    success=False,
                    error="test failure",
                )
            ],
        ),
        plan=plan,
    )

    assert (
        result.evidence_state
        == "UNASSESSED"
    )


def test_component_only_factor_evidence_does_not_establish_slot():
    plan = execution_plan()

    target = next(
        row
        for row in plan.targets
        if (
            row.slot
            == "DISTINGUISHING_FACTOR_EFFECT"
        )
    )

    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id="work:1",
                relationship="COMPONENT_ONLY",
                confidence=0.95,
                rationale=(
                    "Factor and outcome are mentioned but "
                    "no lower-order relation is established."
                ),
            )
        ],
        interpretation="Bounded factor review.",
    )

    result = compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidate_set(
            target.target_id
        ),
        packet=packet(
            plan,
            target,
        ),
        plan=plan,
    )

    assert (
        result.evidence_state
        == "UNASSESSED"
    )

    assert result.negative_coverage_sufficient is False
    assert (
        "insufficient_identity_anchored_material_coverage_for_negative_closure"
        in result.reason_codes
    )


def test_partial_slot_relation_does_not_establish_full_slot():
    plan = execution_plan()

    target = next(
        row
        for row in plan.targets
        if row.slot == "FULL_RELATION"
    )

    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id="work:1",
                relationship="PARTIAL_SLOT_RELATION",
                confidence=0.95,
                rationale=(
                    "The abstract establishes only a substantial "
                    "subset of the residual relation, not the full "
                    "threshold/regime proposition."
                ),
            )
        ],
        interpretation=(
            "Partial neighboring evidence exists, but the "
            "full residual relation is not established."
        ),
    )

    result = compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidate_set(
            target.target_id
        ),
        packet=packet(
            plan,
            target,
        ),
        plan=plan,
    )

    assert result.evidence_state == "UNASSESSED"

    assert (
        result.matches[0].relationship
        == "PARTIAL_SLOT_RELATION"
    )

    assert result.positive_work_ids == []

    assert result.negative_coverage_sufficient is False
    assert (
        "insufficient_identity_anchored_material_coverage_for_negative_closure"
        in result.reason_codes
    )


def test_partial_bridge_relation_does_not_close_bridge_slot():
    plan = execution_plan()

    target = next(
        row
        for row in plan.targets
        if row.slot == "BRIDGE_RELATION"
    )

    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id="work:1",
                relationship="PARTIAL_SLOT_RELATION",
                confidence=0.95,
                rationale=(
                    "Only part of the proposed bridge is supported."
                ),
            )
        ],
        interpretation=(
            "The complete bridge relation remains unestablished."
        ),
    )

    result = compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidate_set(
            target.target_id
        ),
        packet=packet(
            plan,
            target,
        ),
        plan=plan,
    )

    assert result.evidence_state == "UNASSESSED"
    assert result.positive_work_ids == []

    assert result.negative_coverage_sufficient is False
    assert (
        "insufficient_identity_anchored_material_coverage_for_negative_closure"
        in result.reason_codes
    )


def test_negative_closure_requires_minimum_abstract_coverage():
    plan = execution_plan()
    target = plan.targets[0]

    query = next(
        row
        for row in plan.queries
        if row.claim_id == target.target_id
    )

    limited_packet = PriorArtPacket(
        packet_id="packet:limited",
        packet_sha256="sha:limited",
        source_portfolio_id=plan.source_portfolio_id,
        source_query_plan_id=plan.plan_id,
        searched_at_utc="2026-09-01T00:00:00+00:00",
        providers_requested=["fake"],
        works=[
            PriorArtWork(
                work_id="work:1",
                title="Neighboring paper",
                abstract=(
                    "Relevant variables are discussed, but the "
                    "required slot relation is not established."
                ),
                providers=["fake"],
                retrieval_query_ids=[query.query_id],
                retrieval_claim_ids=[target.target_id],
            ),
        ],
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="fake",
                success=True,
                result_count=1,
            )
        ],
        raw_work_count=1,
        canonical_work_count=1,
    )

    result = compile_closure_slot_review(
        target=target,
        draft=ClosureSlotReviewDraft(
            matches=[
                ClosureEvidenceMatchDraft(
                    work_id="work:1",
                    relationship="COMPONENT_ONLY",
                    confidence=0.9,
                    rationale=(
                        "The required relation is not established."
                    ),
                )
            ],
            interpretation="Limited bounded evidence.",
        ),
        candidates=candidate_set(target.target_id),
        packet=limited_packet,
        plan=plan,
    )

    assert result.evidence_state == "UNASSESSED"
    assert result.negative_coverage_sufficient is False
    assert (
        "insufficient_abstract_coverage_for_negative_closure"
        in result.reason_codes
    )


def test_sufficient_abstract_coverage_can_support_bounded_not_found():
    plan = execution_plan()
    target = plan.targets[0]

    query = next(
        row
        for row in plan.queries
        if row.claim_id == target.target_id
    )

    works = [
        PriorArtWork(
            work_id=f"work:{i}",
            title=f"Neighboring paper {i}",
            abstract=(
                "The paper concerns neighboring variables but "
                "does not establish the required relation."
            ),
            providers=["fake"],
            retrieval_query_ids=[query.query_id],
            retrieval_claim_ids=[target.target_id],
        )
        for i in range(1, 4)
    ]

    sufficient_packet = PriorArtPacket(
        packet_id="packet:sufficient",
        packet_sha256="sha:sufficient",
        source_portfolio_id=plan.source_portfolio_id,
        source_query_plan_id=plan.plan_id,
        searched_at_utc="2026-09-01T00:00:00+00:00",
        providers_requested=["fake"],
        works=works,
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="fake",
                success=True,
                result_count=3,
            )
        ],
        raw_work_count=3,
        canonical_work_count=3,
    )

    candidates = ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:test",
        claim_id=target.target_id,
        ranked_works=[
            RankedPriorArtWork(
                work_id=f"work:{i}",
                relevance_score=0.8,
                semantic_similarity=0.8,
                lexical_coverage=0.7,
                reaction_domain_relevance=0.8,
                catalyst_scope_relevance=0.8,
                abstract_available=True,
            )
            for i in range(1, 4)
        ],
    )

    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id=f"work:{i}",
                relationship="COMPONENT_ONLY",
                confidence=0.9,
                rationale="No required slot relation.",
            )
            for i in range(1, 4)
        ],
        interpretation="Sufficient bounded negative coverage.",
    )

    result = compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidates,
        packet=sufficient_packet,
        plan=plan,
    )

    assert result.evidence_state == "NOT_FOUND"
    assert result.negative_coverage_sufficient is True


def test_query_and_provider_success_metrics_are_not_conflated():
    plan = execution_plan()
    target = plan.targets[0]

    query = next(
        row
        for row in plan.queries
        if row.claim_id == target.target_id
    )

    multi_provider_packet = PriorArtPacket(
        packet_id="packet:providers",
        packet_sha256="sha:providers",
        source_portfolio_id=plan.source_portfolio_id,
        source_query_plan_id=plan.plan_id,
        searched_at_utc="2026-09-01T00:00:00+00:00",
        providers_requested=["p1", "p2"],
        works=[
            PriorArtWork(
                work_id="work:1",
                title="Explicit relation",
                abstract=(
                    "The abstract explicitly establishes "
                    "the required relation."
                ),
                providers=["p1", "p2"],
                retrieval_query_ids=[query.query_id],
                retrieval_claim_ids=[target.target_id],
            )
        ],
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="p1",
                success=True,
                result_count=1,
            ),
            QueryExecution(
                query_id=query.query_id,
                provider="p2",
                success=True,
                result_count=1,
            ),
        ],
        raw_work_count=1,
        canonical_work_count=1,
    )

    result = compile_closure_slot_review(
        target=target,
        draft=positive_draft(),
        candidates=candidate_set(target.target_id),
        packet=multi_provider_packet,
        plan=plan,
    )

    assert result.query_count == 1
    assert result.successful_query_count == 1
    assert result.provider_execution_count == 2
    assert result.successful_provider_execution_count == 2


def test_unrelated_abstracts_do_not_satisfy_negative_coverage():
    plan = execution_plan()
    target = plan.targets[0]

    query = next(
        row
        for row in plan.queries
        if row.claim_id == target.target_id
    )

    works = [
        PriorArtWork(
            work_id=f"work:irrelevant:{i}",
            title=f"Unrelated abstract paper {i}",
            abstract=(
                "This abstract contains scientific text "
                "but does not materially bear on the closure target."
            ),
            providers=["fake"],
            retrieval_query_ids=[query.query_id],
            retrieval_claim_ids=[target.target_id],
        )
        for i in range(1, 5)
    ]

    packet_value = PriorArtPacket(
        packet_id="packet:irrelevant",
        packet_sha256="sha:irrelevant",
        source_portfolio_id=plan.source_portfolio_id,
        source_query_plan_id=plan.plan_id,
        searched_at_utc="2026-09-01T00:00:00+00:00",
        providers_requested=["fake"],
        works=works,
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="fake",
                success=True,
                result_count=4,
            )
        ],
        raw_work_count=4,
        canonical_work_count=4,
    )

    candidate_value = ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:test",
        claim_id=target.target_id,
        ranked_works=[
            RankedPriorArtWork(
                work_id=f"work:irrelevant:{i}",
                relevance_score=0.7,
                semantic_similarity=0.7,
                lexical_coverage=0.5,
                reaction_domain_relevance=0.5,
                catalyst_scope_relevance=0.5,
                abstract_available=True,
            )
            for i in range(1, 5)
        ],
    )

    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id=f"work:irrelevant:{i}",
                relationship="UNRELATED",
                confidence=0.95,
                rationale="Not materially related.",
            )
            for i in range(1, 5)
        ],
        interpretation="Irrelevant abstract set.",
    )

    result = compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidate_value,
        packet=packet_value,
        plan=plan,
    )

    assert result.abstract_candidate_count == 4
    assert result.material_abstract_review_count == 0
    assert result.negative_coverage_sufficient is False
    assert result.evidence_state == "UNASSESSED"


def test_three_material_component_abstracts_support_bounded_negative():
    plan = execution_plan()
    target = plan.targets[0]

    query = next(
        row
        for row in plan.queries
        if row.claim_id == target.target_id
    )

    works = [
        PriorArtWork(
            work_id=f"work:material:{i}",
            title=f"Material neighboring paper {i}",
            abstract=(
                "This abstract explicitly discusses target-related "
                "components but does not establish the required relation."
            ),
            providers=["fake"],
            retrieval_query_ids=[query.query_id],
            retrieval_claim_ids=[target.target_id],
        )
        for i in range(1, 4)
    ]

    packet_value = PriorArtPacket(
        packet_id="packet:material",
        packet_sha256="sha:material",
        source_portfolio_id=plan.source_portfolio_id,
        source_query_plan_id=plan.plan_id,
        searched_at_utc="2026-09-01T00:00:00+00:00",
        providers_requested=["fake"],
        works=works,
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="fake",
                success=True,
                result_count=3,
            )
        ],
        raw_work_count=3,
        canonical_work_count=3,
    )

    candidate_value = ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:test",
        claim_id=target.target_id,
        ranked_works=[
            RankedPriorArtWork(
                work_id=f"work:material:{i}",
                relevance_score=0.8,
                semantic_similarity=0.8,
                lexical_coverage=0.7,
                reaction_domain_relevance=0.8,
                catalyst_scope_relevance=0.8,
                abstract_available=True,
            )
            for i in range(1, 4)
        ],
    )

    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id=f"work:material:{i}",
                relationship="COMPONENT_ONLY",
                confidence=0.9,
                rationale=(
                    "Material neighboring evidence, but not "
                    "the required slot relation."
                ),
            )
            for i in range(1, 4)
        ],
        interpretation="Material bounded negative evidence.",
    )

    result = compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidate_value,
        packet=packet_value,
        plan=plan,
    )

    assert result.material_abstract_review_count == 3
    assert result.negative_coverage_sufficient is True
    assert result.evidence_state == "NOT_FOUND"


def test_factor_material_abstracts_without_identity_do_not_close_negative():
    plan = execution_plan()

    target = next(
        row
        for row in plan.targets
        if row.slot == "DISTINGUISHING_FACTOR_EFFECT"
    )

    query = next(
        row
        for row in plan.queries
        if row.claim_id == target.target_id
    )

    works = [
        PriorArtWork(
            work_id=f"work:no-anchor:{i}",
            title=f"Neighboring factor-context paper {i}",
            abstract=(
                "Interparticle spacing and SERS enhancement are "
                "discussed, but the atomic distinguishing factor "
                "is not present in this abstract."
            ),
            providers=["fake"],
            retrieval_query_ids=[query.query_id],
            retrieval_claim_ids=[target.target_id],
        )
        for i in range(1, 4)
    ]

    packet_value = PriorArtPacket(
        packet_id="packet:no-anchor",
        packet_sha256="sha:no-anchor",
        source_portfolio_id=plan.source_portfolio_id,
        source_query_plan_id=plan.plan_id,
        searched_at_utc="2026-09-01T00:00:00+00:00",
        providers_requested=["fake"],
        works=works,
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="fake",
                success=True,
                result_count=3,
            )
        ],
        raw_work_count=3,
        canonical_work_count=3,
    )

    candidates = ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:test",
        claim_id=target.target_id,
        ranked_works=[
            RankedPriorArtWork(
                work_id=f"work:no-anchor:{i}",
                relevance_score=0.8,
                semantic_similarity=0.8,
                lexical_coverage=0.7,
                reaction_domain_relevance=0.8,
                catalyst_scope_relevance=0.8,
                abstract_available=True,
            )
            for i in range(1, 4)
        ],
    )

    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id=f"work:no-anchor:{i}",
                relationship="COMPONENT_ONLY",
                confidence=0.9,
                rationale="Neighboring components only.",
            )
            for i in range(1, 4)
        ],
        interpretation="No atomic factor anchor.",
    )

    result = compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidates,
        packet=packet_value,
        plan=plan,
    )

    assert result.material_abstract_review_count == 3
    assert (
        result.negative_eligible_material_abstract_review_count
        == 0
    )
    assert result.negative_coverage_sufficient is False
    assert result.evidence_state == "UNASSESSED"


def test_factor_anchor_backed_material_abstracts_can_support_negative():
    plan = execution_plan()

    target = next(
        row
        for row in plan.targets
        if row.slot == "DISTINGUISHING_FACTOR_EFFECT"
    )

    query = next(
        row
        for row in plan.queries
        if row.claim_id == target.target_id
    )

    works = [
        PriorArtWork(
            work_id=f"work:anchor:{i}",
            title=f"Laser-power neighboring paper {i}",
            abstract=(
                "Laser power was explicitly reported in experiments "
                "on interparticle spacing and SERS, but the abstract "
                "does not establish the required lower-order relation."
            ),
            providers=["fake"],
            retrieval_query_ids=[query.query_id],
            retrieval_claim_ids=[target.target_id],
        )
        for i in range(1, 4)
    ]

    packet_value = PriorArtPacket(
        packet_id="packet:anchor",
        packet_sha256="sha:anchor",
        source_portfolio_id=plan.source_portfolio_id,
        source_query_plan_id=plan.plan_id,
        searched_at_utc="2026-09-01T00:00:00+00:00",
        providers_requested=["fake"],
        works=works,
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="fake",
                success=True,
                result_count=3,
            )
        ],
        raw_work_count=3,
        canonical_work_count=3,
    )

    candidates = ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:test",
        claim_id=target.target_id,
        ranked_works=[
            RankedPriorArtWork(
                work_id=f"work:anchor:{i}",
                relevance_score=0.8,
                semantic_similarity=0.8,
                lexical_coverage=0.7,
                reaction_domain_relevance=0.8,
                catalyst_scope_relevance=0.8,
                abstract_available=True,
            )
            for i in range(1, 4)
        ],
    )

    draft = ClosureSlotReviewDraft(
        matches=[
            ClosureEvidenceMatchDraft(
                work_id=f"work:anchor:{i}",
                relationship="COMPONENT_ONLY",
                confidence=0.9,
                rationale=(
                    "Laser power is explicitly present in the "
                    "relevant context, but the slot relation is absent."
                ),
            )
            for i in range(1, 4)
        ],
        interpretation="Identity-anchored bounded evidence.",
    )

    result = compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidates,
        packet=packet_value,
        plan=plan,
    )

    assert result.material_abstract_review_count == 3
    assert (
        result.negative_eligible_material_abstract_review_count
        == 3
    )
    assert result.negative_coverage_sufficient is True
    assert result.evidence_state == "NOT_FOUND"
