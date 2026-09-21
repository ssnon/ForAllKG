from __future__ import annotations

import json
from pathlib import Path

from scripts.discovery.materialize_operationalization_evidence_lane import (
    _paper_status_map,
    _projection_complete,
    _validated_existing_evidence,
)


def test_paper_status_map_preserves_manifest_status():
    manifest = {
        "papers": [
            {
                "paper_id": "p1",
                "extraction_quality_status": "complete",
            },
            {
                "paper_id": "p2",
                "extraction_quality_status": "partial_critical",
            },
        ]
    }
    assert _paper_status_map(manifest) == {
        "p1": "complete",
        "p2": "partial_critical",
    }


def test_projection_complete_requires_all_four_bundle_files(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()

    for name in (
        "graph.graphml",
        "node_text.jsonl",
        "edge_evidence.jsonl",
    ):
        (root / name).write_text("", encoding="utf-8")

    assert _projection_complete(root) is False

    (root / "summary.json").write_text("{}", encoding="utf-8")
    assert _projection_complete(root) is True


def test_existing_evidence_reuse_requires_same_canonical(tmp_path):
    canonical = tmp_path / "paper.graphml"
    canonical.write_text("<graphml/>", encoding="utf-8")

    root = tmp_path / "evidence"
    root.mkdir()
    for name in (
        "graph.graphml",
        "node_text.jsonl",
        "edge_evidence.jsonl",
    ):
        (root / name).write_text("", encoding="utf-8")

    (root / "summary.json").write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "mode": "evidence",
                "canonical_graphml": str(canonical),
            }
        ),
        encoding="utf-8",
    )

    assert _validated_existing_evidence(
        root=root,
        paper_id="p1",
        canonical=canonical,
    ) is True


def test_existing_evidence_wrong_mode_is_not_reused(tmp_path):
    canonical = tmp_path / "paper.graphml"
    canonical.write_text("<graphml/>", encoding="utf-8")

    root = tmp_path / "evidence"
    root.mkdir()
    for name in (
        "graph.graphml",
        "node_text.jsonl",
        "edge_evidence.jsonl",
    ):
        (root / name).write_text("", encoding="utf-8")

    (root / "summary.json").write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "mode": "mechanism",
                "canonical_graphml": str(canonical),
            }
        ),
        encoding="utf-8",
    )

    assert _validated_existing_evidence(
        root=root,
        paper_id="p1",
        canonical=canonical,
    ) is False

def test_materializer_uses_module_child_cli_invocation():
    import inspect
    import scripts.discovery.materialize_operationalization_evidence_lane as lane

    source = inspect.getsource(lane.main)

    assert '"-m"' in source
    assert '"scripts.corpus.build_graphagents_projection"' in source
    assert '"scripts.corpus.build_corpus_graph"' in source
    assert "str(projection_script)" not in source
    assert "str(corpus_script)" not in source

