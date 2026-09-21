from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.corpus.semantic_ir.grounded_scope import (
    select_grounded_context_scope,
)
from pipeline_core.corpus.semantic_ir.multi_root import (
    load_semantic_ir_from_roots,
)
from pipeline_core.corpus.semantic_ir.schema import SemanticIRBundle


def _packet() -> dict:
    return {
        "schema_version": "graph-explorer-input-v1",
        "packet_id": "packet:1",
        "packet_sha256": "sha:1",
        "task": {"task_id": "task:1"},
        "evidence_catalog": {
            "nodes": {
                "paper::P1::a": {"node_id": "paper::P1::a"},
                "paper::P1::b": {"node_id": "paper::P1::b"},
                "paper::P2::c": {"node_id": "paper::P2::c"},
                "paper::UNUSED::x": {"node_id": "paper::UNUSED::x"},
            },
            "edges": {
                "e1": {
                    "scientific_source": "paper::P1::a",
                    "scientific_target": "paper::P1::b",
                    "supporting_node_ids": ["paper::P1::a"],
                    "source_paper_ids": ["P1"],
                },
                "e2": {
                    "scientific_source": "paper::P2::c",
                    "scientific_target": "paper::UNUSED::x",
                    "supporting_node_ids": [],
                    "source_paper_ids": ["P2"],
                },
            },
        },
    }


def _context() -> dict:
    return {
        "schema_version": "hypothesis-context-v1",
        "source_packet_id": "packet:1",
        "source_packet_sha256": "sha:1",
        "task_id": "task:1",
        "evidence_statements": [
            {
                "statement_id": "s1",
                "eligible_as_premise": True,
                "eligible_as_gap": False,
                "scientific_support_node_ids": ["paper::P1::a"],
                "scientific_support_edge_ids": ["e1"],
                "paper_ids": ["P1"],
            },
            {
                "statement_id": "s2",
                "eligible_as_premise": False,
                "eligible_as_gap": True,
                "scientific_support_node_ids": ["paper::P2::c"],
                "scientific_support_edge_ids": [],
                "paper_ids": ["P2"],
            },
            {
                "statement_id": "s3",
                "eligible_as_premise": False,
                "eligible_as_gap": False,
                "scientific_support_node_ids": ["paper::UNUSED::x"],
                "scientific_support_edge_ids": ["e2"],
                "paper_ids": ["UNUSED"],
            },
        ],
    }


def test_grounded_scope_excludes_wholesale_packet_catalog():
    selection, reduced = select_grounded_context_scope(
        packet_payload=_packet(),
        context_payload=_context(),
        scope_mode="premise_and_gap",
    )

    assert selection.selected_statement_ids == ["s1", "s2"]
    assert selection.selected_support_edge_ids == ["e1"]
    assert selection.selected_support_node_ids == [
        "paper::P1::a",
        "paper::P1::b",
        "paper::P2::c",
    ]
    assert "paper::UNUSED::x" not in reduced["evidence_catalog"]["nodes"]
    assert reduced["paths"] == []
    assert reduced["direct_concept_hits"] == []


def test_premise_only_scope_excludes_gap_statement():
    selection, _ = select_grounded_context_scope(
        packet_payload=_packet(),
        context_payload=_context(),
        scope_mode="premise_only",
    )

    assert selection.selected_statement_ids == ["s1"]
    assert selection.selected_gap_statement_ids == []
    assert selection.selected_support_node_ids == [
        "paper::P1::a",
        "paper::P1::b",
    ]


def test_context_packet_lineage_mismatch_fails_closed():
    context = _context()
    context["source_packet_sha256"] = "wrong"
    with pytest.raises(ValueError, match="source_packet_sha256 mismatch"):
        select_grounded_context_scope(
            packet_payload=_packet(),
            context_payload=context,
        )


def test_multi_root_duplicate_fails_closed(monkeypatch):
    def fake_discover(root, *, duplicate_paper_policy):
        return SimpleNamespace(
            attempts=[
                SimpleNamespace(
                    paper_id="P1",
                    attempt_id="a",
                    active_chunks_path=f"{root}/active_chunks.json",
                    attempt_directory=f"{root}/attempt",
                )
            ]
        )

    monkeypatch.setattr(
        "pipeline_core.corpus.semantic_ir.multi_root.discover_extraction_attempts",
        fake_discover,
    )

    with pytest.raises(ValueError, match="multiple extraction roots"):
        load_semantic_ir_from_roots(["/r1", "/r2"], requested_paper_ids={"P1"})


def test_multi_root_loads_only_requested_papers(monkeypatch):
    def fake_discover(root, *, duplicate_paper_policy):
        paper = "P1" if str(root).endswith("r1") else "P2"
        return SimpleNamespace(
            attempts=[
                SimpleNamespace(
                    paper_id=paper,
                    attempt_id="a",
                    active_chunks_path=f"{root}/{paper}/active_chunks.json",
                    attempt_directory=f"{root}/{paper}/attempt",
                )
            ]
        )

    def fake_load(attempt_directory):
        paper = "P1" if "/P1/" in str(attempt_directory) else "P2"
        return SimpleNamespace(
            bundle=SemanticIRBundle(
                bundle_id=f"bundle:{paper}",
                paper_id=paper,
                records=[],
                source_chunks=[],
            )
        )

    monkeypatch.setattr(
        "pipeline_core.corpus.semantic_ir.multi_root.discover_extraction_attempts",
        fake_discover,
    )
    monkeypatch.setattr(
        "pipeline_core.corpus.semantic_ir.multi_root.load_existing_extraction_semantic_ir",
        fake_load,
    )

    result, bundles = load_semantic_ir_from_roots(
        ["/r1", "/r2"],
        requested_paper_ids={"P2", "MISSING"},
    )

    assert sorted(bundles) == ["P2"]
    assert result.available_paper_ids == ["P2"]
    assert result.missing_paper_ids == ["MISSING"]
    assert result.requested_paper_ids == ["MISSING", "P2"]
