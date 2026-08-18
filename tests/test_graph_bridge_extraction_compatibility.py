"""Characterization tests for the graph/Bridge extraction compatibility slice."""

from __future__ import annotations

import dac_her.bridge_draft_schema as legacy_bridge_draft
import dac_her.bridge_schemas as legacy_bridge
import dac_her.graph_io as legacy_graph_io
import dac_her.graph_validation as legacy_graph_validation
import dac_her.validation as legacy_validation
import dac_her.validation_issues as legacy_issues
import pipeline_core.bridge_draft_schema as core_bridge_draft
import pipeline_core.bridge_schemas as core_bridge
import pipeline_core.graph_io as core_graph_io
import pipeline_core.graph_validation as core_graph_validation
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
        (legacy_validation.validate_graph_provenance, core_validation.validate_graph_provenance),
        (legacy_issues.ValidationIssue, core_issues.ValidationIssue),
        (legacy_issues.ValidationReport, core_issues.ValidationReport),
        (legacy_issues.issue, core_issues.issue),
    )

    for legacy_symbol, core_symbol in symbol_pairs:
        assert legacy_symbol is core_symbol
