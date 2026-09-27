from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.pre_n10_exploratory_novelty_v1 import (
    build_exploratory_query_plan,
    select_ready_novelty_claim_ids,
)


def _claim(
    claim_id: str,
    *,
    role: str,
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:test",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role=role,
        text="A changes B.",
        rationale="test",
        search_concepts=["A B"],
        search_queries=["A changes B"],
        prior_art_identity_terms=["A"],
        relation_nucleus_terms=["changes"],
        required_bridge="A changes B.",
        predicted_observation="B changes.",
        falsification_condition="B does not change.",
    )


def test_exploratory_selection_keeps_only_strict_ready_novelty_claims():
    ready_novelty = SimpleNamespace(
        claim_id="claim:ready",
        contract_status="READY_FOR_N10_CONTRACT",
        novelty_selection_role="NOVELTY_BEARING",
    )
    incomplete_testing = SimpleNamespace(
        claim_id="claim:testing-gap",
        contract_status="NOT_READY_FOR_N10_CONTRACT",
        novelty_selection_role="TESTING_PREDICTION",
    )
    incomplete_novelty = SimpleNamespace(
        claim_id="claim:novelty-gap",
        contract_status="NOT_READY_FOR_N10_CONTRACT",
        novelty_selection_role="NOVELTY_BEARING",
    )
    contract = SimpleNamespace(
        claims=[
            ready_novelty,
            incomplete_testing,
            incomplete_novelty,
        ]
    )

    assert select_ready_novelty_claim_ids(contract) == [
        "claim:ready"
    ]


def test_exploratory_query_plan_excludes_incomplete_supporting_claims():
    ready = _claim(
        "claim:ready",
        role="NOVELTY_BEARING",
    )
    testing = _claim(
        "claim:testing",
        role="TESTING_PREDICTION",
    )
    source = LiteratureQueryPlan(
        plan_id="literature_query_plan:source",
        plan_sha256="0" * 64,
        source_portfolio_id="portfolio:test",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:test",
                title="test",
                claims=[ready, testing],
            )
        ],
        queries=[
            LiteratureQuery(
                query_id="query:ready",
                hypothesis_id="hypothesis:test",
                claim_id="claim:ready",
                query_kind="claim_primary",
                query_text="ready query",
            ),
            LiteratureQuery(
                query_id="query:testing",
                hypothesis_id="hypothesis:test",
                claim_id="claim:testing",
                query_kind="claim_primary",
                query_text="testing query",
            ),
            LiteratureQuery(
                query_id="query:composite",
                hypothesis_id="hypothesis:test",
                claim_id=None,
                query_kind="hypothesis_composite",
                query_text="whole hypothesis query",
            ),
        ],
    )

    result = build_exploratory_query_plan(
        source_plan=source,
        hypothesis_id="hypothesis:test",
        selected_claim_ids=["claim:ready"],
    )

    assert result.source_portfolio_id == source.source_portfolio_id
    assert [
        claim.claim_id
        for claim in result.claims[0].claims
    ] == ["claim:ready"]
    assert [
        query.claim_id
        for query in result.queries
    ] == ["claim:ready"]
    assert result.plan_id.startswith(
        "literature_query_plan:pre_n10_exploratory:"
    )


def test_supporting_testing_gap_does_not_block_ready_core_search():
    contract = SimpleNamespace(
        claims=[
            SimpleNamespace(
                claim_id="claim:core-1",
                contract_status="READY_FOR_N10_CONTRACT",
                novelty_selection_role="NOVELTY_BEARING",
            ),
            SimpleNamespace(
                claim_id="claim:core-2",
                contract_status="READY_FOR_N10_CONTRACT",
                novelty_selection_role="NOVELTY_BEARING",
            ),
            SimpleNamespace(
                claim_id="claim:testing",
                contract_status="NOT_READY_FOR_N10_CONTRACT",
                novelty_selection_role="TESTING_PREDICTION",
            ),
        ]
    )

    assert select_ready_novelty_claim_ids(contract) == [
        "claim:core-1",
        "claim:core-2",
    ]
