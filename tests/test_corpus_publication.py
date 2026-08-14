from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dac_her.corpus_publication import (
    CorpusPublicationError,
    CorpusPublicationOptions,
    StrictBridgeCorpusPublisher,
    build_paper_lifecycle,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_lifecycle_fixture(root: Path) -> dict[str, Path]:
    m3 = root / "m3"
    m4 = root / "m4"
    m45 = root / "m4_5"
    outcomes = root / "outcomes.jsonl"
    selected = m3 / "selected_works.jsonl"

    _write_jsonl(
        selected,
        [
            {
                "work_id": "w1",
                "title": "Paper 1",
                "doi": "10.1/one",
                "matched_axes": ["nanogap", "hotspot"],
                "primary_quota_axis": "nanogap",
                "total_score": 9.0,
            },
            {
                "work_id": "w2",
                "title": "Paper 2",
                "doi": "10.1/two",
                "matched_axes": ["shell"],
                "primary_quota_axis": "shell",
                "total_score": 8.0,
            },
            {
                "work_id": "w3",
                "title": "Paper 3",
                "doi": None,
                "matched_axes": ["nanogap"],
                "primary_quota_axis": "nanogap",
                "total_score": 7.0,
            },
            {
                "work_id": "w4",
                "title": "Paper 4",
                "doi": None,
                "matched_axes": ["simulation"],
                "primary_quota_axis": "simulation",
                "total_score": 6.0,
            },
        ],
    )
    _write_jsonl(
        m3 / "artifacts.jsonl",
        [
            {"work_id": "w1", "role": "main", "status": "downloaded", "sha256": "a"},
            {"work_id": "w2", "role": "main", "status": "downloaded", "sha256": "b"},
            {"work_id": "w3", "role": "main", "status": "download_failed", "error": "403"},
            {"work_id": "w4", "role": "main", "status": "not_attempted"},
        ],
    )
    _write_jsonl(
        m4 / "paper_map.jsonl",
        [
            {"work_id": "w1", "paper_id": "p1"},
            {"work_id": "w2", "paper_id": "p2"},
            {"work_id": "w3", "paper_id": "p3"},
        ],
    )
    _write_jsonl(
        m4 / "paper_materialization_records.jsonl",
        [
            {
                "work_id": "w1",
                "paper_id": "p1",
                "main_document_status": "materialized",
                "extraction_ready": True,
            },
            {
                "work_id": "w2",
                "paper_id": "p2",
                "main_document_status": "materialized",
                "extraction_ready": True,
            },
            {
                "work_id": "w3",
                "paper_id": "p3",
                "main_document_status": "failed",
                "extraction_ready": False,
            },
        ],
    )
    _write_jsonl(
        m45 / "pre_extraction_gate_assessments.jsonl",
        [
            {
                "work_id": "w1",
                "paper_id": "p1",
                "auto_extraction_allowed": True,
                "identity": {"status": "verified", "method": "doi_exact", "reasons": []},
                "suitability": {"status": "suitable", "suitable_axes": ["nanogap"], "reasons": []},
            },
            {
                "work_id": "w2",
                "paper_id": "p2",
                "auto_extraction_allowed": False,
                "identity": {"status": "verified", "method": "doi_exact", "reasons": []},
                "suitability": {
                    "status": "unsuitable",
                    "suitable_axes": [],
                    "reasons": ["selected_axis_absent_from_fulltext"],
                },
            },
        ],
    )
    _write_jsonl(
        outcomes,
        [
            {
                "paper_id": "p1",
                "strict_status": "STRICT_USABLE",
                "bridge_status": "BRIDGE_USEFUL",
                "projection_status": "PROJECTION_USABLE",
                "corpus_eligible": True,
                "canonical_graph_sha256": "strict-sha",
                "bridge_graph_sha256": "bridge-sha",
                "projection_sha256": "projection-sha",
            }
        ],
    )
    return {
        "selected": selected,
        "m3": m3,
        "m4": m4,
        "m45": m45,
        "outcomes": outcomes,
    }


def test_lifecycle_accounts_every_selected_work_and_exposes_attrition(tmp_path: Path):
    paths = _base_lifecycle_fixture(tmp_path)
    lifecycle, summary = build_paper_lifecycle(
        selected_works_path=paths["selected"],
        m3_dir=paths["m3"],
        m4_dir=paths["m4"],
        m4_5_dir=paths["m45"],
        outcomes_path=paths["outcomes"],
    )

    assert [row["work_id"] for row in lifecycle] == ["w1", "w2", "w3", "w4"]
    by_work = {row["work_id"]: row for row in lifecycle}
    assert by_work["w1"]["terminal_stage"] == "corpus"
    assert by_work["w2"]["terminal_stage"] == "m4_5"
    assert by_work["w2"]["m4_5_suitability_status"] == "unsuitable"
    assert by_work["w3"]["terminal_stage"] == "m4"
    assert by_work["w4"]["terminal_stage"] == "m3"
    assert by_work["w4"]["paper_id"] is None

    assert summary["selected_work_count"] == 4
    assert summary["m3_main_downloaded_count"] == 2
    assert summary["m4_extraction_ready_count"] == 2
    assert summary["m4_5_auto_extraction_ready_count"] == 1
    assert summary["strict_usable_count"] == 1
    assert summary["bridge_useful_count"] == 1
    assert summary["corpus_eligible_count"] == 1
    assert summary["selected_work_accounting_complete"] is True
    assert summary["selected_matched_axis_counts"] == {
        "hotspot": 1,
        "nanogap": 2,
        "shell": 1,
        "simulation": 1,
    }
    assert summary["corpus_eligible_matched_axis_counts"] == {
        "hotspot": 1,
        "nanogap": 1,
    }


def _publication_fixture(root: Path, *, corpus_eligible: bool = True) -> dict[str, Path]:
    paths = _base_lifecycle_fixture(root)
    if not corpus_eligible:
        _write_jsonl(
            paths["outcomes"],
            [
                {
                    "paper_id": "p1",
                    "strict_status": "STRICT_USABLE",
                    "bridge_status": "BRIDGE_USEFUL",
                    "projection_status": "PROJECTION_ERROR",
                    "corpus_eligible": False,
                }
            ],
        )

    data_root = root / "data_sers"
    mode_root = data_root / "corpus" / "pilot" / "mechanism"
    mode_root.mkdir(parents=True, exist_ok=True)
    (mode_root / "graph.graphml").write_text("<graphml>corpus</graphml>\n", encoding="utf-8")
    (mode_root / "node_text.jsonl").write_text('{"node_id":"n1"}\n', encoding="utf-8")
    _write_json(
        mode_root / "manifest.json",
        {"passes_structural_gate": True, "paper_ids": ["p1"]},
    )
    _write_json(mode_root / "audit.json", {"passes_structural_gate": True})

    strict_run = data_root / "pipeline_runs" / "pilot" / "strict_bridge"
    strict_run.mkdir(parents=True, exist_ok=True)
    (strict_run / "paper_outcomes.jsonl").write_bytes(paths["outcomes"].read_bytes())
    paths["data_root"] = data_root
    paths["mode_root"] = mode_root
    paths["strict_outcomes"] = strict_run / "paper_outcomes.jsonl"
    return paths


def test_publish_rejects_bridge_target_when_traversal_ready_count_is_lower(tmp_path: Path):
    paths = _publication_fixture(tmp_path, corpus_eligible=False)
    publisher = StrictBridgeCorpusPublisher(
        project_root=tmp_path,
        corpus_id="pilot",
        domain_profile="sers_au_ag",
        data_root=paths["data_root"],
        selected_works_path=paths["selected"],
        m3_dir=paths["m3"],
        m4_dir=paths["m4"],
        m4_5_dir=paths["m45"],
        outcomes_path=paths["strict_outcomes"],
        options=CorpusPublicationOptions(
            mode="mechanism",
            target_count=1,
            target_status="BRIDGE_USEFUL",
            build_node_index=False,
        ),
    )
    with pytest.raises(CorpusPublicationError, match="Traversal-ready corpus"):
        publisher.run()


def test_publish_builds_bound_navigation_and_node_index_then_resumes(tmp_path: Path):
    paths = _publication_fixture(tmp_path, corpus_eligible=True)
    calls: list[str] = []

    def fake_runner(command: list[str], label: str) -> bool:
        calls.append(label)
        mode_root = paths["mode_root"]
        nav_root = mode_root / "navigation"
        nav_root.mkdir(parents=True, exist_ok=True)
        if label == "navigation":
            nav_graph = nav_root / "graph.graphml"
            nav_graph.write_text("<graphml>navigation</graphml>\n", encoding="utf-8")
            _write_json(
                nav_root / "summary.json",
                {
                    "source_graph_sha256": _sha(mode_root / "graph.graphml"),
                    "graphml_sha256": _sha(nav_graph),
                },
            )
            return True
        if label == "node_index":
            index_root = nav_root / "node_index"
            index_root.mkdir(parents=True, exist_ok=True)
            _write_json(
                index_root / "manifest.json",
                {
                    "model_name": "test-model",
                    "include_alignment_hubs": False,
                    "navigation_graph_sha256": _sha(nav_root / "graph.graphml"),
                    "node_text_sha256": _sha(mode_root / "node_text.jsonl"),
                },
            )
            return True
        raise AssertionError(label)

    options = CorpusPublicationOptions(
        mode="mechanism",
        target_count=1,
        target_status="BRIDGE_USEFUL",
        build_node_index=True,
        embedding_model="test-model",
    )
    publisher = StrictBridgeCorpusPublisher(
        project_root=tmp_path,
        corpus_id="pilot",
        domain_profile="sers_au_ag",
        data_root=paths["data_root"],
        selected_works_path=paths["selected"],
        m3_dir=paths["m3"],
        m4_dir=paths["m4"],
        m4_5_dir=paths["m45"],
        outcomes_path=paths["strict_outcomes"],
        options=options,
        command_runner=fake_runner,
    )
    result = publisher.run()
    assert result["status"] == "published"
    assert calls == ["navigation", "node_index"]
    assert result["integrity_chain"]["chain_valid"] is True
    assert Path(result["paper_lifecycle"]).is_file()
    assert Path(result["funnel_summary"]).is_file()
    assert Path(result["node_index_manifest"]).is_file()

    def fail_if_called(command: list[str], label: str) -> bool:
        raise AssertionError(f"unexpected rebuild: {label}: {command}")

    resumed = StrictBridgeCorpusPublisher(
        project_root=tmp_path,
        corpus_id="pilot",
        domain_profile="sers_au_ag",
        data_root=paths["data_root"],
        selected_works_path=paths["selected"],
        m3_dir=paths["m3"],
        m4_dir=paths["m4"],
        m4_5_dir=paths["m45"],
        outcomes_path=paths["strict_outcomes"],
        options=options,
        command_runner=fail_if_called,
    ).run()
    assert resumed["status"] == "already_current"
    assert resumed["command_records"] == []


def test_publish_rebuilds_navigation_if_corpus_graph_changes(tmp_path: Path):
    paths = _publication_fixture(tmp_path, corpus_eligible=True)
    calls: list[str] = []

    def runner(command: list[str], label: str) -> bool:
        calls.append(label)
        mode_root = paths["mode_root"]
        nav_root = mode_root / "navigation"
        nav_root.mkdir(parents=True, exist_ok=True)
        if label == "navigation":
            nav = nav_root / "graph.graphml"
            nav.write_text(f"nav-{len(calls)}\n", encoding="utf-8")
            _write_json(
                nav_root / "summary.json",
                {
                    "source_graph_sha256": _sha(mode_root / "graph.graphml"),
                    "graphml_sha256": _sha(nav),
                },
            )
        return True

    options = CorpusPublicationOptions(
        mode="mechanism",
        target_count=1,
        target_status="CORPUS_ELIGIBLE",
        build_node_index=False,
    )
    kwargs = dict(
        project_root=tmp_path,
        corpus_id="pilot",
        domain_profile="sers_au_ag",
        data_root=paths["data_root"],
        selected_works_path=paths["selected"],
        m3_dir=paths["m3"],
        m4_dir=paths["m4"],
        m4_5_dir=paths["m45"],
        outcomes_path=paths["strict_outcomes"],
        options=options,
        command_runner=runner,
    )
    StrictBridgeCorpusPublisher(**kwargs).run()
    assert calls == ["navigation"]

    (paths["mode_root"] / "graph.graphml").write_text(
        "<graphml>corpus changed</graphml>\n", encoding="utf-8"
    )
    StrictBridgeCorpusPublisher(**kwargs).run()
    assert calls == ["navigation", "navigation"]


def test_publish_rejects_stale_corpus_membership(tmp_path: Path):
    paths = _publication_fixture(tmp_path, corpus_eligible=True)
    _write_json(
        paths["mode_root"] / "manifest.json",
        {"passes_structural_gate": True, "paper_ids": ["stale_paper"]},
    )
    publisher = StrictBridgeCorpusPublisher(
        project_root=tmp_path,
        corpus_id="pilot",
        domain_profile="sers_au_ag",
        data_root=paths["data_root"],
        selected_works_path=paths["selected"],
        m3_dir=paths["m3"],
        m4_dir=paths["m4"],
        m4_5_dir=paths["m45"],
        outcomes_path=paths["strict_outcomes"],
        options=CorpusPublicationOptions(
            mode="mechanism",
            target_count=1,
            target_status="CORPUS_ELIGIBLE",
            build_node_index=False,
        ),
    )
    with pytest.raises(CorpusPublicationError, match="paper_ids do not match"):
        publisher.run()


def test_publish_allows_canonical_m4_superset_records(tmp_path: Path):
    paths = _publication_fixture(tmp_path)
    # Canonical M4/M4.5 directories may retain records from an older/broader
    # acquisition selection.  They must not be mistaken for current-corpus
    # contamination when the active selected set and final corpus agree.
    with (paths["m4"] / "paper_map.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"work_id": "historical-w", "paper_id": "historical-p"}) + "\n")
    with (paths["m4"] / "paper_materialization_records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "work_id": "historical-w",
            "paper_id": "historical-p",
            "main_document_status": "materialized",
            "extraction_ready": False,
        }) + "\n")
    with (paths["m45"] / "pre_extraction_gate_assessments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "work_id": "historical-w",
            "paper_id": "historical-p",
            "auto_extraction_allowed": False,
            "identity": {"status": "verified"},
            "suitability": {"status": "unsuitable"},
        }) + "\n")

    commands: list[tuple[str, list[str]]] = []

    def fake_runner(command: list[str], label: str) -> bool:
        commands.append((label, command))
        if label == "navigation":
            nav = paths["mode_root"] / "navigation"
            nav.mkdir(parents=True, exist_ok=True)
            (nav / "graph.graphml").write_text("<graphml>nav</graphml>\n", encoding="utf-8")
            _write_json(
                nav / "summary.json",
                {
                    "source_graph_sha256": _sha(paths["mode_root"] / "graph.graphml"),
                    "graphml_sha256": _sha(nav / "graph.graphml"),
                },
            )
        return True

    publisher = StrictBridgeCorpusPublisher(
        project_root=tmp_path,
        corpus_id="pilot",
        domain_profile="sers_au_ag",
        data_root=paths["data_root"],
        selected_works_path=paths["selected"],
        m3_dir=paths["m3"],
        m4_dir=paths["m4"],
        m4_5_dir=paths["m45"],
        outcomes_path=paths["strict_outcomes"],
        options=CorpusPublicationOptions(
            mode="mechanism",
            target_count=1,
            target_status="BRIDGE_USEFUL",
            build_node_index=False,
            resume=False,
        ),
        command_runner=fake_runner,
    )
    result = publisher.run()
    assert result["status"] == "published"
    funnel = json.loads((paths["mode_root"] / "publication" / "funnel_summary.json").read_text(encoding="utf-8"))
    assert funnel["allowed_superset_record_counts"] == {
        "m4_5_gate": 1,
        "m4_materialization": 1,
        "m4_paper_map": 1,
    }
    assert funnel["unexpected_non_lifecycle_record_counts"] == {
        "m3_main_artifact": 0,
        "strict_bridge_outcome": 0,
    }


def test_publish_still_rejects_active_m3_records_outside_selected_lifecycle(tmp_path: Path):
    paths = _publication_fixture(tmp_path)
    with (paths["m3"] / "artifacts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "work_id": "unexpected-active-w",
            "role": "main",
            "status": "downloaded",
            "sha256": "unexpected",
        }) + "\n")
    publisher = StrictBridgeCorpusPublisher(
        project_root=tmp_path,
        corpus_id="pilot",
        domain_profile="sers_au_ag",
        data_root=paths["data_root"],
        selected_works_path=paths["selected"],
        m3_dir=paths["m3"],
        m4_dir=paths["m4"],
        m4_5_dir=paths["m45"],
        outcomes_path=paths["strict_outcomes"],
        options=CorpusPublicationOptions(
            mode="mechanism",
            target_count=1,
            target_status="BRIDGE_USEFUL",
            build_node_index=False,
        ),
    )
    with pytest.raises(CorpusPublicationError, match="Active-run artifacts"):
        publisher.run()
