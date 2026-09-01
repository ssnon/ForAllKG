from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    PriorArtPacket,
    PriorArtWork,
    RankedPriorArtWork,
)
from pipeline_core.discovery.novelty_closure_execution import (
    ExecutableClosureTarget,
)
from pipeline_core.discovery.novelty_closure_llm import (
    _CLOSURE_REVIEW_SYSTEM,
    build_closure_review_user_prompt,
)


def packet(
    *,
    abstract=(
        "Laser power and interparticle spacing are both "
        "discussed in the experiment."
    ),
):
    return PriorArtPacket(
        packet_id="packet:test",
        packet_sha256="sha:test",
        source_portfolio_id=(
            "portfolio:test"
        ),
        source_query_plan_id=(
            "plan:test"
        ),
        searched_at_utc=(
            "2026-09-01T00:00:00+00:00"
        ),
        providers_requested=["fake"],
        works=[
            PriorArtWork(
                work_id="work:exact",
                title="Test paper",
                abstract=abstract,
                providers=["fake"],
            )
        ],
    )


def candidates():
    return ClaimPriorArtCandidateSet(
        hypothesis_id=(
            "hypothesis:test"
        ),
        claim_id=(
            "closure_target:test"
        ),
        ranked_works=[
            RankedPriorArtWork(
                work_id="work:exact",
                relevance_score=0.9,
                semantic_similarity=0.9,
                lexical_coverage=0.8,
                abstract_available=True,
            )
        ],
    )


def factor_target():
    return ExecutableClosureTarget(
        target_id=(
            "closure_target:test"
        ),
        slot=(
            "DISTINGUISHING_FACTOR_EFFECT"
        ),
        source_claim_id=(
            "claim:test"
        ),
        target_basis=(
            "IDENTITY_PLUS_RELATION_CONTEXT"
        ),
        search_terms=(
            "laser power",
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ),
        search_query=(
            "laser power interparticle spacing "
            "SERS enhancement dependence"
        ),
        source_text=(
            "identity=laser power; "
            "relation_context="
            "interparticle spacing | "
            "SERS enhancement | dependence"
        ),
    )


def bridge_target():
    return ExecutableClosureTarget(
        target_id=(
            "closure_target:test"
        ),
        slot="BRIDGE_RELATION",
        source_claim_id=(
            "claim:test"
        ),
        target_basis=(
            "EXTRACTIVE_REQUIRED_BRIDGE"
        ),
        search_terms=(),
        search_query=(
            "Laser power drives a transition "
            "that changes spacing to SERS."
        ),
        source_text=(
            "Laser power drives a transition "
            "that changes spacing to SERS."
        ),
    )


def full_target():
    return ExecutableClosureTarget(
        target_id=(
            "closure_target:test"
        ),
        slot="FULL_RELATION",
        source_claim_id=(
            "claim:test"
        ),
        target_basis=(
            "FULL_RESIDUAL_CLAIM"
        ),
        search_terms=(),
        search_query=(
            "A critical laser power separates "
            "two spacing to SERS regimes."
        ),
        source_text=(
            "A critical laser power separates "
            "two spacing to SERS regimes."
        ),
    )


def test_factor_contract_explicitly_blocks_comention_inference():
    system = (
        _CLOSURE_REVIEW_SYSTEM
    )

    assert (
        "broad retrieval target, NOT a "
        "predefined scientific proposition"
        in system
    )

    assert (
        "Mere mention of the factor and "
        "contextual variables"
        in system
    )

    assert (
        "Do NOT invent which mechanism "
        "the factor acts through"
        in system
    )


def test_full_contract_requires_actual_higher_order_relation():
    system = (
        _CLOSURE_REVIEW_SYSTEM
    )

    assert (
        "the full residual relation itself"
        in system
    )

    assert (
        "threshold/regime structure itself"
        in system
    )

    assert (
        "must NOT be promoted to full establishment"
        in system
    )


def test_partial_relation_is_explicitly_not_establishment():
    assert (
        "PARTIAL_SLOT_RELATION is not "
        "equivalent to ESTABLISHES_SLOT"
        in _CLOSURE_REVIEW_SYSTEM
    )


def test_factor_prompt_preserves_retrieval_target_as_provenance():
    prompt = (
        build_closure_review_user_prompt(
            target=factor_target(),
            candidates=candidates(),
            packet=packet(),
        )
    )

    assert (
        "slot: DISTINGUISHING_FACTOR_EFFECT"
        in prompt
    )

    assert (
        "target_basis: "
        "IDENTITY_PLUS_RELATION_CONTEXT"
        in prompt
    )

    assert (
        "identity=laser power"
        in prompt
    )

    assert (
        "search_query is retrieval intent, "
        "not scientific evidence"
        in prompt
    )


def test_bridge_prompt_preserves_exact_source_material():
    target = bridge_target()

    prompt = (
        build_closure_review_user_prompt(
            target=target,
            candidates=candidates(),
            packet=packet(),
        )
    )

    assert (
        target.source_text
        in prompt
    )


def test_full_prompt_preserves_residual_claim_text():
    target = full_target()

    prompt = (
        build_closure_review_user_prompt(
            target=target,
            candidates=candidates(),
            packet=packet(),
        )
    )

    assert (
        target.source_text
        in prompt
    )


def test_allowed_work_id_is_exactly_exposed():
    prompt = (
        build_closure_review_user_prompt(
            target=factor_target(),
            candidates=candidates(),
            packet=packet(),
        )
    )

    assert (
        "ALLOWED_WORK_IDS"
        in prompt
    )

    assert (
        "work:exact"
        in prompt
    )


def test_missing_candidate_work_is_not_invented():
    bad_candidates = (
        ClaimPriorArtCandidateSet(
            hypothesis_id=(
                "hypothesis:test"
            ),
            claim_id=(
                "closure_target:test"
            ),
            ranked_works=[
                RankedPriorArtWork(
                    work_id=(
                        "work:not-in-packet"
                    ),
                    relevance_score=0.9,
                    semantic_similarity=0.9,
                    lexical_coverage=0.8,
                    abstract_available=True,
                )
            ],
        )
    )

    prompt = (
        build_closure_review_user_prompt(
            target=factor_target(),
            candidates=bad_candidates,
            packet=packet(),
        )
    )

    assert (
        "work:not-in-packet"
        not in prompt
    )

    assert (
        "ALLOWED_WORK_IDS\n"
        "================\n"
        "NONE"
        in prompt
    )
