from __future__ import annotations

from pipeline_core.discovery.diagnostic_prior_art_retrieval import (
    build_diagnostic_query_plan,
)
from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
)


def _claim(
    *,
    claim_id: str,
    diagnostic_kind: str,
    diagnostic_query: str | None,
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:test",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text=(
            "A moderator changes how X "
            "relates to Y."
        ),
        rationale="test",
        search_queries=[
            "ordinary full relation",
        ],
        diagnostic_query_kind=(
            diagnostic_kind
        ),
        diagnostic_search_query=(
            "human readable diagnostic"
            if diagnostic_kind != "NONE"
            else None
        ),
        diagnostic_execution_query=(
            diagnostic_query
        ),
        diagnostic_relation_terms=[
            "X",
            "Y",
            "relationship",
        ]
        if diagnostic_kind != "NONE"
        else [],
    )


def _base_plan() -> LiteratureQueryPlan:
    diagnostic = _claim(
        claim_id=(
            "external_novelty_claim:"
            "diagnostic"
        ),
        diagnostic_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_query=(
            "generic dimer X Y "
            "relationship dependence"
        ),
    )

    ordinary = _claim(
        claim_id=(
            "external_novelty_claim:"
            "ordinary"
        ),
        diagnostic_kind="NONE",
        diagnostic_query=None,
    )

    group = HypothesisNoveltyClaims(
        hypothesis_id=(
            "hypothesis:test"
        ),
        title="Test",
        claims=[
            diagnostic,
            ordinary,
        ],
    )

    first_pass = LiteratureQuery(
        query_id=(
            "literature_query:"
            "ordinary"
        ),
        hypothesis_id=(
            "hypothesis:test"
        ),
        claim_id=diagnostic.claim_id,
        query_kind="claim_primary",
        query_text=(
            "ordinary full relation"
        ),
    )

    return LiteratureQueryPlan(
        plan_id=(
            "literature_query_plan:"
            "base"
        ),
        plan_sha256="base-sha",
        source_portfolio_id=(
            "hypothesis_portfolio:test"
        ),
        queries=[
            first_pass,
        ],
        claims=[
            group,
        ],
    )


def test_diagnostic_plan_materializes_only_canonical_diagnostic_queries() -> None:
    base = _base_plan()

    diagnostic = (
        build_diagnostic_query_plan(
            base
        )
    )

    assert len(
        diagnostic.queries
    ) == 1

    query = (
        diagnostic.queries[0]
    )

    assert (
        query.query_kind
        == "claim_diagnostic"
    )

    assert (
        query.claim_id
        == (
            "external_novelty_claim:"
            "diagnostic"
        )
    )

    assert (
        query.query_text
        == (
            "generic dimer X Y "
            "relationship dependence"
        )
    )

    # First-pass plan remains untouched.
    assert [
        row.query_kind
        for row in base.queries
    ] == [
        "claim_primary",
    ]


def test_diagnostic_plan_is_deterministic() -> None:
    base = _base_plan()

    left = (
        build_diagnostic_query_plan(
            base
        )
    )

    right = (
        build_diagnostic_query_plan(
            base
        )
    )

    assert (
        left.model_dump(mode="json")
        == right.model_dump(mode="json")
    )


def test_diagnostic_plan_does_not_invent_missing_execution_query() -> None:
    base = _base_plan()

    claim = (
        base.claims[0].claims[0]
    )

    missing = claim.model_copy(
        update={
            "diagnostic_execution_query":
                None,
        }
    )

    group = (
        base.claims[0].model_copy(
            update={
                "claims": [
                    missing,
                    base.claims[0]
                    .claims[1],
                ]
            }
        )
    )

    changed = base.model_copy(
        update={
            "claims": [
                group,
            ]
        }
    )

    diagnostic = (
        build_diagnostic_query_plan(
            changed
        )
    )

    assert (
        diagnostic.queries
        == []
    )
