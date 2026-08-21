from __future__ import annotations

from types import SimpleNamespace

import pipeline_core.corpus.graph.graph_validation_legacy_relation_compat as compat
from pipeline_core.corpus.graph.legacy_dac_relation_policy import (
    LegacyRelationEndpointPolicy,
)


def _issue(**kwargs):
    return kwargs


def test_renderer_consumes_relation_policy_data(
    monkeypatch,
):
    policy = LegacyRelationEndpointPolicy(
        relation="CUSTOM_RELATION",
        source_types=frozenset({"Alpha"}),
        target_types=frozenset({"Beta"}),
    )

    monkeypatch.setattr(
        compat,
        "LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION",
        {
            "CUSTOM_RELATION": policy,
        },
    )

    graph = SimpleNamespace(
        entities=[
            SimpleNamespace(
                id="source",
                type="Alpha",
            ),
            SimpleNamespace(
                id="target",
                type="Beta",
            ),
        ],
        edges=[
            SimpleNamespace(
                source="source",
                target="target",
                relation="CUSTOM_RELATION",
            ),
        ],
    )

    issues = []

    compat.append_legacy_dac_relation_compat_issues(
        graph=graph,
        node_ids={"source", "target"},
        by_id={
            "source": [
                (
                    "entities",
                    graph.entities[0],
                )
            ],
            "target": [
                (
                    "entities",
                    graph.entities[1],
                )
            ],
        },
        claim_ids=set(),
        observation_ids=set(),
        mechanism_ids=set(),
        issues=issues,
        entity_types_fn=lambda value: {
            node.id: node.type
            for node in value.entities
        },
        relation_type_issue_fn=_issue,
    )

    assert issues == []


def test_renderer_uses_policy_expected_types_for_mismatch(
    monkeypatch,
):
    policy = LegacyRelationEndpointPolicy(
        relation="CUSTOM_RELATION",
        source_types=frozenset({"Alpha"}),
        target_types=frozenset({"Beta"}),
    )

    monkeypatch.setattr(
        compat,
        "LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION",
        {
            "CUSTOM_RELATION": policy,
        },
    )

    graph = SimpleNamespace(
        entities=[
            SimpleNamespace(
                id="source",
                type="Wrong",
            ),
            SimpleNamespace(
                id="target",
                type="Beta",
            ),
        ],
        edges=[
            SimpleNamespace(
                source="source",
                target="target",
                relation="CUSTOM_RELATION",
            ),
        ],
    )

    issues = []

    compat.append_legacy_dac_relation_compat_issues(
        graph=graph,
        node_ids={"source", "target"},
        by_id={
            "source": [
                (
                    "entities",
                    graph.entities[0],
                )
            ],
            "target": [
                (
                    "entities",
                    graph.entities[1],
                )
            ],
        },
        claim_ids=set(),
        observation_ids=set(),
        mechanism_ids=set(),
        issues=issues,
        entity_types_fn=lambda value: {
            node.id: node.type
            for node in value.entities
        },
        relation_type_issue_fn=_issue,
    )

    assert len(issues) == 1

    issue = issues[0]

    assert issue["source"] is True
    assert issue["relation"] == "CUSTOM_RELATION"
    assert issue["endpoint_id"] == "source"
    assert issue["expected"] == frozenset(
        {"Alpha"}
    )
    assert issue["actual"] == "Wrong"
