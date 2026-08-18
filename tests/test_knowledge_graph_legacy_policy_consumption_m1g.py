from __future__ import annotations

from types import SimpleNamespace

import pipeline_core.knowledge_graph_legacy_relation_compat as compat
from pipeline_core.legacy_dac_relation_policy import (
    LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION,
    LegacyRelationEndpointPolicy,
)


def _graph(
    *,
    source_type: str,
    target_type: str,
):
    source = SimpleNamespace(
        id="source",
        type=source_type,
    )

    target = SimpleNamespace(
        id="target",
        type=target_type,
    )

    return SimpleNamespace(
        entities=[source, target],
        experiments=[],
        calculations=[],
        measurements=[],
        measurement_groups=[],
        observation_claims=[],
        mechanism_claims=[],
        edges=[
            SimpleNamespace(
                source="source",
                relation="HAS_METAL",
                target="target",
            )
        ],
    )


def test_strict_renderer_consumes_policy_source_types(
    monkeypatch,
):
    policy = LegacyRelationEndpointPolicy(
        relation="HAS_METAL",
        source_types=frozenset({"Support"}),
        target_types=frozenset({"Metal"}),
    )

    policies = dict(
        LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION
    )

    policies["HAS_METAL"] = policy

    monkeypatch.setattr(
        compat,
        "LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION",
        policies,
    )

    graph = _graph(
        source_type="Support",
        target_type="Metal",
    )

    assert (
        compat.validate_legacy_relation_semantics_compat(
            graph
        )
        is graph
    )


def test_strict_renderer_consumes_policy_target_types(
    monkeypatch,
):
    policy = LegacyRelationEndpointPolicy(
        relation="HAS_METAL",
        source_types=frozenset({"Catalyst"}),
        target_types=frozenset({"Support"}),
    )

    policies = dict(
        LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION
    )

    policies["HAS_METAL"] = policy

    monkeypatch.setattr(
        compat,
        "LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION",
        policies,
    )

    graph = _graph(
        source_type="Catalyst",
        target_type="Support",
    )

    assert (
        compat.validate_legacy_relation_semantics_compat(
            graph
        )
        is graph
    )
