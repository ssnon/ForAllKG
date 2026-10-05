from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
)
from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _query_plan_hypothesis_ids,
    _subset_query_plan_for_hypotheses,
    _subset_topology_completion_payload,
)


def _plan() -> LiteratureQueryPlan:
    claims = [
        HypothesisNoveltyClaims(
            hypothesis_id="h1",
            title="one",
            claims=[],
        ),
        HypothesisNoveltyClaims(
            hypothesis_id="h2",
            title="two",
            claims=[],
        ),
        HypothesisNoveltyClaims(
            hypothesis_id="h3",
            title="three",
            claims=[],
        ),
    ]
    queries = [
        LiteratureQuery(
            query_id="q1",
            hypothesis_id="h1",
            claim_id=None,
            query_kind="hypothesis_composite",
            query_text="one",
        ),
        LiteratureQuery(
            query_id="q2",
            hypothesis_id="h2",
            claim_id=None,
            query_kind="hypothesis_composite",
            query_text="two",
        ),
        LiteratureQuery(
            query_id="q3",
            hypothesis_id="h3",
            claim_id=None,
            query_kind="hypothesis_composite",
            query_text="three",
        ),
    ]
    return LiteratureQueryPlan(
        plan_id="literature_query_plan:source",
        plan_sha256="source-sha",
        source_portfolio_id="portfolio:seed",
        queries=queries,
        claims=claims,
        policy_version="external-novelty-query-policy-v2-domain-recall",
    )


def test_query_plan_projection_is_exact_hypothesis_subset():
    source = _plan()
    projected = _subset_query_plan_for_hypotheses(
        source_plan=source,
        active_hypothesis_ids={"h1", "h3"},
    )

    assert _query_plan_hypothesis_ids(projected) == {"h1", "h3"}
    assert [x.hypothesis_id for x in projected.queries] == ["h1", "h3"]
    assert projected.source_portfolio_id == source.source_portfolio_id
    assert projected.plan_id != source.plan_id
    assert projected.plan_sha256 != source.plan_sha256

    source_groups = {
        row.hypothesis_id: row.model_dump(mode="json")
        for row in source.claims
    }
    for row in projected.claims:
        assert row.model_dump(mode="json") == source_groups[row.hypothesis_id]


def test_query_plan_projection_rejects_non_subset():
    source = _plan()
    try:
        _subset_query_plan_for_hypotheses(
            source_plan=source,
            active_hypothesis_ids={"h1", "missing"},
        )
    except ValueError as exc:
        assert "missing from source query plan" in str(exc)
    else:
        raise AssertionError("expected non-subset projection to fail")


def test_topology_completion_projection_filters_by_hypothesis():
    payload = {
        "schema_version": "source-bound-topology-completion-shadow-v1",
        "diagnostic_only": True,
        "production_authority": False,
        "enabled": True,
        "records": [
            {"hypothesis_id": "h1", "claim_id": "c1"},
            {"hypothesis_id": "h2", "claim_id": "c2"},
            {"hypothesis_id": "h3", "claim_id": "c3"},
        ],
    }
    projected = _subset_topology_completion_payload(
        payload=payload,
        active_hypothesis_ids={"h2"},
    )
    assert projected["records"] == [
        {"hypothesis_id": "h2", "claim_id": "c2"}
    ]
    assert projected["diagnostic_only"] is True
    assert projected["production_authority"] is False
