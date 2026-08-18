"""Characterization tests for the graph/Bridge extraction compatibility slice."""

from __future__ import annotations

import dac_her.bridge_draft_schema as legacy_bridge_draft
import dac_her.bridge_schemas as legacy_bridge
import dac_her.graph_io as legacy_graph_io
import dac_her.graph_validation as legacy_graph_validation
import dac_her.node_references as legacy_node_references
import dac_her.discovery_semantics as legacy_discovery
import dac_her.validation as legacy_validation
import dac_her.validation_issues as legacy_issues
import pipeline_core.bridge_draft_schema as core_bridge_draft
import pipeline_core.bridge_schemas as core_bridge
import pipeline_core.graph_io as core_graph_io
import pipeline_core.graph_validation as core_graph_validation
import pipeline_core.node_references as core_node_references
import pipeline_core.discovery_semantics as core_discovery
import pipeline_core.validation as core_validation
import pipeline_core.validation_issues as core_issues


def test_legacy_graph_bridge_modules_reexport_core_implementations():
    symbol_pairs = (
        (legacy_bridge_draft.BridgeChunkDraft, core_bridge_draft.BridgeChunkDraft),
        (legacy_bridge_draft.BridgeCandidateRepair, core_bridge_draft.BridgeCandidateRepair),
        (legacy_bridge.BridgeConcept, core_bridge.BridgeConcept),
        (legacy_bridge.BridgeChunkGraph, core_bridge.BridgeChunkGraph),
        (legacy_graph_io.knowledge_graph_to_networkx, core_graph_io.knowledge_graph_to_networkx),
        (legacy_graph_io.save_graphml, core_graph_io.save_graphml),
        (legacy_graph_validation.collect_graph_issues, core_graph_validation.collect_graph_issues),
        (
            legacy_node_references.remap_node_reference_attributes,
            core_node_references.remap_node_reference_attributes,
        ),
        (legacy_discovery.normalized_node_type, core_discovery.normalized_node_type),
        (legacy_discovery.is_mechanism_node, core_discovery.is_mechanism_node),
        (legacy_discovery.is_generic_entity_node, core_discovery.is_generic_entity_node),
        (legacy_validation.validate_graph_provenance, core_validation.validate_graph_provenance),
        (legacy_issues.ValidationIssue, core_issues.ValidationIssue),
        (legacy_issues.ValidationReport, core_issues.ValidationReport),
        (legacy_issues.issue, core_issues.issue),
    )

    for legacy_symbol, core_symbol in symbol_pairs:
        assert legacy_symbol is core_symbol


def test_node_reference_remapping_preserves_graphml_foreign_keys():
    payload = {
        "type": "Measurement",
        "subject_id": "entity-1",
        "group_id": "group-1",
        "unrelated": "keep",
    }
    assert legacy_node_references.remap_node_reference_attributes(
        payload,
        {"entity-1": "paper:entity-1", "group-1": "paper:group-1"},
    ) == {
        "type": "Measurement",
        "subject_id": "paper:entity-1",
        "group_id": "paper:group-1",
        "unrelated": "keep",
    }
