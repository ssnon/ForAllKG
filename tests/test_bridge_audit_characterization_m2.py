from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from dac_her.bridge_audit import (
    audit_bridge_graph,
    write_bridge_audit,
)
from dac_her.bridge_policy import (
    BRIDGE_POLICY_VERSION,
)


def _ready_graph(
    *,
    anchor_status: str = "resolved",
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    graph.graph.update(
        {
            "bridge_policy_version":
                BRIDGE_POLICY_VERSION,
            "bridge_prompt_version":
                "test-prompt-v1",
            "bridge_run_id":
                "bridge-run:test",
        }
    )

    graph.add_node(
        "pattern-1",
        type="BridgeConcept",
        label="metal identity varies with adsorption",
        retention_lane="accepted_pattern",
        concept_type="relational_pattern",
        evidence_scope="paper",
        pattern_subject="metal identity",
        pattern_relation="VARIES_WITH",
        pattern_object="adsorption energy",
        relation_strength="correlational",
        pattern_support_mode="explicit_single_span",
        qualifiers_json="[]",
    )

    graph.add_node(
        "frontier-1",
        type="BridgeConcept",
        label="unresolved local frontier",
        retention_lane="paper_local_frontier",
        concept_type="frontier",
        evidence_scope="paper",
        pattern_support_mode="",
    )

    graph.add_edge(
        "pattern-1",
        "frontier-1",
        relation="BRIDGES",
        anchor_resolution_status=anchor_status,
    )

    return graph


def test_audit_bridge_graph_characterizes_ready_report():
    graph = _ready_graph()

    report, tables = audit_bridge_graph(
        graph
    )

    assert report[
        "policy_version"
    ] == BRIDGE_POLICY_VERSION

    assert report["nodes"] == 2
    assert report["edges"] == 1
    assert report["bridge_concepts"] == 2
    assert report["patterns"] == 1
    assert report["frontier_concepts"] == 1

    assert report["retention_lanes"] == {
        "accepted_pattern": 1,
        "paper_local_frontier": 1,
    }

    assert report["relations"] == {
        "BRIDGES": 1,
    }

    assert report[
        "anchor_resolution_statuses"
    ] == {
        "resolved": 1,
    }

    assert report["pattern_issues"] == 0
    assert report["duplicate_label_groups"] == 0
    assert report["ready_for_projection"] is True

    assert report[
        "prompt_version"
    ] == "test-prompt-v1"

    assert report[
        "bridge_run_id"
    ] == "bridge-run:test"

    assert len(tables["concepts"]) == 2
    assert tables["pattern_issues"] == []
    assert tables["duplicate_candidates"] == []
    assert tables["rejections"] == []


def test_unresolved_anchor_blocks_projection():
    graph = _ready_graph(
        anchor_status="unresolved_in_canonical"
    )

    report, _ = audit_bridge_graph(
        graph
    )

    assert report[
        "anchor_resolution_statuses"
    ] == {
        "unresolved_in_canonical": 1,
    }

    assert report[
        "ready_for_projection"
    ] is False


def test_missing_pattern_fields_are_reported():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "pattern-1",
        type="BridgeConcept",
        label="incomplete pattern",
        retention_lane="accepted_pattern",
        pattern_subject="metal identity",
        pattern_relation="VARIES_WITH",
        pattern_object="",
        relation_strength="",
        pattern_support_mode="",
        qualifiers_json="[]",
    )

    report, tables = audit_bridge_graph(
        graph
    )

    assert report["pattern_issues"] == 1
    assert report[
        "ready_for_projection"
    ] is False

    issue = tables["pattern_issues"][0]

    assert issue[
        "issue"
    ] == (
        "accepted pattern is missing "
        "required fields"
    )

    assert issue["missing_fields"] == [
        "pattern_object",
        "relation_strength",
        "pattern_support_mode",
    ]


def test_rejection_reason_counts_accept_lists_and_json_strings():
    graph = nx.MultiDiGraph()

    rejection_rows = [
        {
            "reason_codes": [
                "UNSUPPORTED_RELATION",
                "RELATION_CUE_MISMATCH",
            ],
        },
        {
            "reason_codes":
                '["UNSUPPORTED_RELATION"]',
        },
    ]

    report, tables = audit_bridge_graph(
        graph,
        rejection_rows=rejection_rows,
    )

    assert report[
        "rejected_candidates"
    ] == 2

    assert report[
        "rejection_reasons"
    ] == {
        "UNSUPPORTED_RELATION": 2,
        "RELATION_CUE_MISMATCH": 1,
    }

    assert tables[
        "rejections"
    ] == rejection_rows


def test_write_bridge_audit_materializes_expected_artifacts(
    tmp_path: Path,
):
    graph = _ready_graph()

    report = write_bridge_audit(
        graph,
        output_dir=tmp_path,
    )

    expected = {
        "bridge_audit.json",
        "bridge_audit_concepts.csv",
        "bridge_pattern_issues.csv",
        "bridge_duplicate_candidates.csv",
        "bridge_rejections_audit.csv",
    }

    assert {
        path.name
        for path in tmp_path.iterdir()
    } == expected

    written = json.loads(
        (
            tmp_path
            / "bridge_audit.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert written == report
    assert report[
        "ready_for_projection"
    ] is True

    assert (
        tmp_path
        / "bridge_audit_concepts.csv"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        tmp_path
        / "bridge_pattern_issues.csv"
    ).read_text(
        encoding="utf-8"
    ) == ""

    assert (
        tmp_path
        / "bridge_duplicate_candidates.csv"
    ).read_text(
        encoding="utf-8"
    ) == ""

    assert (
        tmp_path
        / "bridge_rejections_audit.csv"
    ).read_text(
        encoding="utf-8"
    ) == ""
