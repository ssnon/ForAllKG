from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import yaml

from dac_her.config import get_paper_config, paper_config_fingerprint_payload
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.incremental_reconcile import IncrementalCorpusReconciler, ReconcileOptions
from dac_her.run_state import document_source_fingerprints


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    paper_id = "P1"
    package = tmp_path / "pkg"
    package.mkdir()
    md = package / "main.md"
    md.write_text("# paper\nbody\n", encoding="utf-8")
    config = tmp_path / "papers.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "papers": {
                    paper_id: {
                        "enabled": True,
                        "documents": [
                            {
                                "document_id": "main",
                                "role": "main",
                                "markdown_path": str(md),
                                "selection": {"mode": "whole_document"},
                            }
                        ],
                        "resolution_file": None,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.json"
    _write_json(
        frozen,
        {
            "schema_version": "graphagentsdac-frozen-corpus-v01",
            "corpus_id": "c1",
            "documents": [{"paper_id": paper_id}],
        },
    )
    return config, frozen, md, paper_id


def _reconciler(
    tmp_path: Path, *, freshness: str = "semantic"
) -> tuple[IncrementalCorpusReconciler, Path, str]:
    config, frozen, md, paper_id = _fixture(tmp_path)
    rec = IncrementalCorpusReconciler(
        project_root=tmp_path,
        papers_yaml=config,
        frozen_manifest=frozen,
        corpus_id="c1",
        options=ReconcileOptions(
            mode="evidence",
            kg_data_root="data_dac",
            freshness=freshness,
            extract_model="test-model",
        ),
    )
    return rec, md, paper_id


def _matching_run_meta(rec: IncrementalCorpusReconciler, paper_id: str, run_id: str) -> dict:
    paper = get_paper_config(rec.papers_yaml, project_root=rec.root, paper_id=paper_id)
    current = rec._current_contract(paper_id)
    semantic = current["semantic"]
    return {
        "run_id": run_id,
        "run_fingerprint": run_id + "-fingerprint",
        "paper": paper_config_fingerprint_payload(paper),
        "document_sources": document_source_fingerprints(paper),
        "model": semantic["model"],
        "provider": semantic["provider"],
        "prompt_version": semantic["prompt_version"],
        "prompt_sha256": semantic["prompt_sha256"],
        "schema_sha256": semantic["schema_sha256"],
        "vocabularies": semantic["vocabularies"],
        "policy": asdict(ExtractionPolicy()),
        # Deliberately operational; semantic freshness must not care.
        "domain_profile_id": "legacy-profile-id",
        "data_root": "/some/old/location",
        "chunking_sha256": current["full"]["chunking_sha256"],
        "implementation_files": [],
    }


def _write_complete_run(
    rec: IncrementalCorpusReconciler,
    paper_id: str,
    run_id: str,
    *,
    mutate_meta=None,
) -> Path:
    run_dir = rec.paper_root(paper_id) / "runs" / run_id
    meta = _matching_run_meta(rec, paper_id, run_id)
    if mutate_meta:
        mutate_meta(meta)
    _write_json(run_dir / "run.json", meta)
    _write_json(
        run_dir / "active_chunks.json",
        {
            "run_id": run_id,
            "graph_materialization_status": "complete",
            "chunks": [{"chunk_id": "c"}],
        },
    )
    return run_dir


def test_missing_artifacts_are_pending(tmp_path: Path) -> None:
    rec, _, paper_id = _reconciler(tmp_path)
    strict = rec.strict_state(paper_id)
    assert not strict.stage.valid
    graph = rec.strict_graph_state(paper_id, strict)
    assert not graph.valid


def test_semantic_freshness_recognizes_matching_run_and_ignores_operational_fields(
    tmp_path: Path,
) -> None:
    rec, _, paper_id = _reconciler(tmp_path)
    run_id = "abc123"
    _write_complete_run(rec, paper_id, run_id)
    _write_json(rec.paper_root(paper_id) / "latest_run.json", {"paper_id": paper_id, "run_id": run_id})

    strict = rec.strict_state(paper_id)
    assert strict.stage.valid
    assert "semantic contract matches" in strict.stage.reason

    graph = nx.MultiDiGraph(run_id=run_id, domain_profile_id="dac_her")
    graph.add_node("x")
    graph_path = rec.paper_root(paper_id) / f"{paper_id}.graphml"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, graph_path)
    assert rec.strict_graph_state(paper_id, strict).valid


def test_semantic_freshness_invalidates_schema_or_vocabulary_change(tmp_path: Path) -> None:
    rec, _, paper_id = _reconciler(tmp_path)
    run_id = "old"
    _write_complete_run(
        rec,
        paper_id,
        run_id,
        mutate_meta=lambda meta: meta.__setitem__("schema_sha256", "old-schema"),
    )
    _write_json(rec.paper_root(paper_id) / "latest_run.json", {"paper_id": paper_id, "run_id": run_id})
    state = rec.strict_state(paper_id)
    assert not state.stage.valid
    assert "schema_sha256" in state.stage.reason


def test_source_freshness_reuses_run_across_schema_change(tmp_path: Path) -> None:
    rec, _, paper_id = _reconciler(tmp_path, freshness="source")
    run_id = "old"
    _write_complete_run(
        rec,
        paper_id,
        run_id,
        mutate_meta=lambda meta: meta.__setitem__("schema_sha256", "old-schema"),
    )
    _write_json(rec.paper_root(paper_id) / "latest_run.json", {"paper_id": paper_id, "run_id": run_id})
    state = rec.strict_state(paper_id)
    assert state.stage.valid
    assert "source contract matches" in state.stage.reason


def test_source_change_invalidates_strict_run(tmp_path: Path) -> None:
    rec, md, paper_id = _reconciler(tmp_path)
    run_id = "abc123"
    _write_complete_run(rec, paper_id, run_id)
    _write_json(rec.paper_root(paper_id) / "latest_run.json", {"paper_id": paper_id, "run_id": run_id})
    assert rec.strict_state(paper_id).stage.valid
    md.write_text("# paper\nchanged\n", encoding="utf-8")
    state = rec.strict_state(paper_id)
    assert not state.stage.valid
    assert "document_sources" in state.stage.reason


def test_incomplete_latest_pointer_falls_back_to_compatible_complete_run(tmp_path: Path) -> None:
    rec, _, paper_id = _reconciler(tmp_path)
    old_id = "complete-old"
    _write_complete_run(rec, paper_id, old_id)

    new_id = "interrupted-new"
    new_dir = rec.paper_root(paper_id) / "runs" / new_id
    _write_json(new_dir / "run.json", _matching_run_meta(rec, paper_id, new_id))
    _write_json(rec.paper_root(paper_id) / "latest_run.json", {"paper_id": paper_id, "run_id": new_id})

    state = rec.strict_state(paper_id)
    assert state.stage.valid
    assert state.run_id == old_id
    assert "recovered compatible usable strict run" in state.stage.reason

    rec._repair_broken_latest_pointer_before_attempt(paper_id)
    pointer = json.loads((rec.paper_root(paper_id) / "latest_run.json").read_text())
    assert pointer["run_id"] == old_id
    assert pointer["recovered_by"] == "incremental_reconciler"


def test_downstream_commands_pin_selected_strict_run(tmp_path: Path) -> None:
    rec, _, paper_id = _reconciler(tmp_path)
    command = rec._paper_command(paper_id, "strict_graph", run_id="run42")
    assert command[command.index("--run-id") + 1] == "run42"
    bridge = rec._paper_command(paper_id, "bridge", run_id="run42")
    assert bridge[bridge.index("--run-id") + 1] == "run42"


def test_corpus_hash_change_is_detected(tmp_path: Path) -> None:
    rec, _, paper_id = _reconciler(tmp_path)
    projection_root = rec.paper_root(paper_id) / "graphagents" / "evidence"
    projection_root.mkdir(parents=True)
    for name, content in {
        "graph.graphml": "g",
        "node_text.jsonl": "n",
        "edge_evidence.jsonl": "e",
        "summary.json": "{}",
    }.items():
        (projection_root / name).write_text(content, encoding="utf-8")
    hashes = rec._projection_hashes(paper_id)
    assert hashes is not None

    corpus_root = tmp_path / "data_dac" / "corpus" / "c1" / "evidence"
    corpus_root.mkdir(parents=True)
    (corpus_root / "graph.graphml").write_text("corpus", encoding="utf-8")
    _write_json(
        corpus_root / "manifest.json",
        {
            "paper_ids": [paper_id],
            "passes_structural_gate": True,
            "papers": [{"paper_id": paper_id, "sha256": hashes}],
        },
    )
    _write_json(corpus_root / "audit.json", {"passes_structural_gate": True})
    assert rec.corpus_state().valid

    (projection_root / "node_text.jsonl").write_text("changed", encoding="utf-8")
    state = rec.corpus_state()
    assert not state.valid
    assert "projection changed" in state.reason
