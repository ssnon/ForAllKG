from __future__ import annotations

import networkx as nx

import pipeline_core.graphagents_adapter as adapter

from domains.registry import (
    get_domain_profile,
)


def test_builtin_projection_coverage_is_explicitly_characterized():
    dac = get_domain_profile("dac_her")
    broad = get_domain_profile(
        "catalysis_mechanism"
    )
    sers = get_domain_profile("sers_au_ag")

    assert dac.projection is not None
    assert sers.projection is not None

    # Historical/current boundary:
    # broad catalysis currently relies on the shared
    # projection engine's compatibility fallback.
    assert broad.projection is None


def test_catalysis_mechanism_currently_resolves_to_legacy_dac_projection():
    broad = get_domain_profile(
        "catalysis_mechanism"
    )

    semantics = (
        adapter._resolve_projection_semantics(
            broad.projection
        )
    )

    assert (
        semantics.semantics_id
        == "dac_her_legacy_projection_v1"
    )


def test_catalysis_mechanism_fallback_affects_current_projection_filtering():
    broad = get_domain_profile(
        "catalysis_mechanism"
    )

    graph = nx.MultiDiGraph()

    graph.add_node(
        "catalyst",
        type="Catalyst",
        label="Catalyst",
    )

    graph.add_node(
        "active_site",
        type="ActiveSite",
        label="Active site",
    )

    projection, _, _ = (
        adapter.build_graphagents_projection(
            graph,
            mode="mechanism",
            projection_semantics=(
                broad.projection
            ),
        )
    )

    # Legacy DAC fallback retains Catalyst but does
    # not know the broad-domain ActiveSite type.
    assert set(
        projection.nodes
    ) == {"catalyst"}

    assert (
        projection.graph[
            "projection_semantics_id"
        ]
        == "dac_her_legacy_projection_v1"
    )
