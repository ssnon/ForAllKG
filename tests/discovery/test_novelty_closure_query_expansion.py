from tests.discovery.test_novelty_closure_execution_review import (
    execution_plan,
)

from pipeline_core.discovery.novelty_closure_execution import (
    _source_query_tokens,
    expand_closure_query_plan_source_preserving,
)


def _tokens(value: str) -> set[str]:
    return {
        row.casefold()
        for row in _source_query_tokens(
            value
        )
    }


def test_source_preserving_expansion_is_bounded_and_deterministic():
    plan = execution_plan()

    first = expand_closure_query_plan_source_preserving(
        plan=plan,
        max_queries_per_target=3,
    )

    second = expand_closure_query_plan_source_preserving(
        plan=plan,
        max_queries_per_target=3,
    )

    assert first.plan_id == second.plan_id
    assert first.plan_sha256 == second.plan_sha256

    counts = {
        target.slot: len(
            [
                row
                for row in first.queries
                if row.claim_id
                == target.target_id
            ]
        )
        for target in first.targets
    }

    assert counts == {
        "BASE_RELATION": 2,
        "DISTINGUISHING_FACTOR_EFFECT": 3,
        "BRIDGE_RELATION": 3,
        "FULL_RELATION": 3,
    }

    assert len(first.queries) == 11

    assert len(
        {
            row.query_id
            for row in first.queries
        }
    ) == len(first.queries)


def test_expansion_preserves_primary_query_and_target_identity():
    plan = execution_plan()

    expanded = expand_closure_query_plan_source_preserving(
        plan=plan
    )

    assert [
        row.target_id
        for row in expanded.targets
    ] == [
        row.target_id
        for row in plan.targets
    ]

    for target in plan.targets:
        queries = [
            row.query_text
            for row in expanded.queries
            if row.claim_id
            == target.target_id
        ]

        assert queries[0] == target.search_query


def test_every_variant_uses_only_source_preserved_vocabulary():
    plan = execution_plan()

    expanded = expand_closure_query_plan_source_preserving(
        plan=plan
    )

    target_by_id = {
        row.target_id: row
        for row in expanded.targets
    }

    for query in expanded.queries:
        target = target_by_id[
            query.claim_id
        ]

        source_tokens = _tokens(
            target.source_text
        )

        query_tokens = _tokens(
            query.query_text
        )

        assert query_tokens <= source_tokens, (
            target.slot,
            query.query_text,
            query_tokens - source_tokens,
        )


def test_expanding_an_expanded_plan_is_idempotent():
    plan = execution_plan()

    once = expand_closure_query_plan_source_preserving(
        plan=plan
    )

    twice = expand_closure_query_plan_source_preserving(
        plan=once
    )

    assert once.plan_id == twice.plan_id
    assert once.plan_sha256 == twice.plan_sha256

    assert [
        row.model_dump(mode="json")
        for row in once.queries
    ] == [
        row.model_dump(mode="json")
        for row in twice.queries
    ]
