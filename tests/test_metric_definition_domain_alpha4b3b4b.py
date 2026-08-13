from __future__ import annotations

import networkx as nx
import pytest

from dac_her.metric_definition_domain import (
    MetricDefinitionContext,
    MetricDefinitionDomainAdapter,
)


def _context(**overrides):
    values = dict(
        context_id="metricdef:1",
        domain_profile_id="sers_au_ag",
        metric_definition_semantics_id="sem",
        paper_id="P1",
        measurement_id="m1",
        observable_key="sers_enhancement_factor",
        definition_status="unknown",
        definition_family="reported_ef_unspecified",
        aggregation_scope="unspecified",
        normalization_basis="unspecified",
        reference_basis="unspecified",
        source_measurement_ids=("m1",),
        source_node_ids=("m1",),
    )
    values.update(overrides)
    return MetricDefinitionContext(**values)


def test_unknown_definition_must_use_unspecified_family():
    with pytest.raises(ValueError):
        _context(definition_family="molecule_normalized_intensity_ratio")


def test_unknown_definition_cannot_carry_formula():
    with pytest.raises(ValueError):
        _context(formula_text="EF = (...)")


def test_measurement_must_be_typed_source():
    with pytest.raises(ValueError):
        _context(
            source_measurement_ids=("other",),
            source_node_ids=("m1", "other"),
        )


def test_adapter_rejects_duplicate_measurement_contexts():
    graph = nx.MultiDiGraph()
    adapter = MetricDefinitionDomainAdapter(
        adapter_id="sers_au_ag",
        domain_profile_id="sers_au_ag",
        semantics_id="sem",
        supported_observable_keys=frozenset({"sers_enhancement_factor"}),
        definition_families=frozenset({"reported_ef_unspecified"}),
        aggregation_scopes=frozenset({"unspecified"}),
        normalization_bases=frozenset({"unspecified"}),
        reference_bases=frozenset({"unspecified"}),
        extract_contexts_fn=lambda _g, _p: [_context(), _context(context_id="metricdef:2")],
    )
    with pytest.raises(ValueError):
        adapter.extract_contexts(graph, "P1")
