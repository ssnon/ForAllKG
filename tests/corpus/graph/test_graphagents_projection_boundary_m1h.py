from __future__ import annotations

import networkx as nx

import pipeline_core.corpus.graph.graphagents_adapter as adapter

from domains.dac_her.profile import (
    DAC_HER_PROFILE,
)
from domains.sers.profile import (
    SERS_AU_AG_PROFILE,
)


def _rule_signatures(semantics):
    return {
        (
            rule.relation,
            rule.direction,
        )
        for rule in semantics.backtrace_rules
    }


def test_default_projection_semantics_is_historical_dac_compatibility():
    semantics = adapter._resolve_projection_semantics(
        None
    )

    assert (
        semantics.semantics_id
        == "dac_her_legacy_projection_v1"
    )


def test_legacy_default_matches_active_dac_projection_structure_except_identity():
    legacy = adapter._resolve_projection_semantics(
        None
    )

    active = DAC_HER_PROFILE.projection

    assert active is not None

    assert (
        legacy.semantics_id
        != active.semantics_id
    )

    assert (
        legacy.mechanism_node_types
        == active.mechanism_node_types
    )

    assert (
        legacy.origin_node_types
        == active.origin_node_types
    )

    assert (
        _rule_signatures(legacy)
        == _rule_signatures(active)
    )

    assert (
        legacy.max_backtrace_depth
        == active.max_backtrace_depth
    )


def test_explicit_sers_projection_semantics_bypasses_legacy_default():
    semantics = SERS_AU_AG_PROFILE.projection

    assert semantics is not None

    assert (
        adapter._resolve_projection_semantics(
            semantics
        )
        is semantics
    )

    assert (
        semantics.semantics_id
        == "sers_au_ag_projection_v2_alpha4b2c3"
    )


def test_dac_and_sers_projection_policies_are_meaningfully_distinct():
    legacy = adapter._resolve_projection_semantics(
        None
    )

    sers = SERS_AU_AG_PROFILE.projection

    assert sers is not None

    legacy_rules = _rule_signatures(
        legacy
    )
    sers_rules = _rule_signatures(
        sers
    )

    assert (
        "MEASURED_FOR",
        "outgoing",
    ) not in legacy_rules

    assert (
        "MEASURED_FOR",
        "outgoing",
    ) in sers_rules

    assert (
        "PlasmonicSubstrate"
        not in legacy.mechanism_node_types
    )

    assert (
        "PlasmonicSubstrate"
        in sers.mechanism_node_types
    )


def test_build_projection_uses_explicit_domain_semantics_for_filtering():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "dac_node",
        type="Catalyst",
        label="DAC node",
    )

    graph.add_node(
        "sers_node",
        type="PlasmonicSubstrate",
        label="SERS node",
    )

    legacy_projection, _, _ = (
        adapter.build_graphagents_projection(
            graph,
            mode="mechanism",
        )
    )

    sers_semantics = (
        SERS_AU_AG_PROFILE.projection
    )

    assert sers_semantics is not None

    sers_projection, _, _ = (
        adapter.build_graphagents_projection(
            graph,
            mode="mechanism",
            projection_semantics=sers_semantics,
        )
    )

    assert set(
        legacy_projection.nodes
    ) == {"dac_node"}

    assert set(
        sers_projection.nodes
    ) == {"sers_node"}

    assert (
        legacy_projection.graph[
            "projection_semantics_id"
        ]
        == "dac_her_legacy_projection_v1"
    )

    assert (
        sers_projection.graph[
            "projection_semantics_id"
        ]
        == "sers_au_ag_projection_v2_alpha4b2c3"
    )
