from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import yaml

from pipeline_core.strict_bridge_corpus_pipeline import (
    StrictBridgeCorpusPipeline,
    StrictBridgePipelineOptions,
    _sha256_file,
    load_strict_ready_paper_ids,
    select_strict_ready_paper_ids,
)


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "papers": {
                    "paper_a": {"documents": []},
                    "paper_b": {"documents": []},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _pipeline(tmp_path: Path, *, mode: str = "mechanism") -> StrictBridgeCorpusPipeline:
    config = tmp_path / "papers.yaml"
    _write_config(config)
    return StrictBridgeCorpusPipeline(
        project_root=tmp_path,
        config=config,
        corpus_id="pilot",
        domain_profile="sers_au_ag",
        data_root=tmp_path / "data_sers",
        options=StrictBridgePipelineOptions(mode=mode, heartbeat_seconds=0),
        requested_paper_ids=["paper_a"],
    )


def _write_latest_strict_run(pipeline: StrictBridgeCorpusPipeline) -> dict[str, str]:
    paper_root = pipeline._paper_root("paper_a")
    run_dir = paper_root / "runs" / "run_1" / "attempts" / "attempt_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    active = {
        "graph_materialization_status": "complete",
        "run_id": "run_1",
        "run_fingerprint": "fp_1",
        "attempt_id": "attempt_1",
    }
    (run_dir / "active_chunks.json").write_text(json.dumps(active), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(active), encoding="utf-8")
    pointer = {
        "run_directory": str((paper_root / "runs" / "run_1").resolve()),
        "attempt_directory": str(run_dir.resolve()),
        **active,
    }
    paper_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "latest_run.json").write_text(json.dumps(pointer), encoding="utf-8")
    return {
        "run_id": "run_1",
        "run_fingerprint": "fp_1",
        "attempt_id": "attempt_1",
        "run_directory": str(run_dir.resolve()),
    }


def test_load_and_select_strict_ready_papers(tmp_path: Path) -> None:
    config = tmp_path / "papers.yaml"
    _write_config(config)
    assert load_strict_ready_paper_ids(config) == ["paper_a", "paper_b"]
    assert select_strict_ready_paper_ids(
        ["paper_a", "paper_b"], requested=["paper_b", "paper_b", "paper_a"], limit=1
    ) == ["paper_b"]


def test_paper_graph_freshness_is_bound_to_strict_identity(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    identity = _write_latest_strict_run(pipeline)
    graph = nx.MultiDiGraph()
    graph.graph.update(
        {
            "run_id": identity["run_id"],
            "run_fingerprint": identity["run_fingerprint"],
            "source_extraction_attempt_id": identity["attempt_id"],
        }
    )
    graph.add_node("n1", type="Material")
    nx.write_graphml(graph, pipeline._canonical_path("paper_a"))
    assert pipeline._paper_graph_matches_latest_extraction("paper_a") is True

    graph.graph["run_fingerprint"] = "stale"
    nx.write_graphml(graph, pipeline._canonical_path("paper_a"))
    assert pipeline._paper_graph_matches_latest_extraction("paper_a") is False


def test_bridge_freshness_checks_policy_run_and_canonical_sha(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    identity = _write_latest_strict_run(pipeline)

    canonical = nx.MultiDiGraph()
    canonical.add_node("m1", type="Material")
    pipeline._canonical_path("paper_a").parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(canonical, pipeline._canonical_path("paper_a"))
    canonical_sha = _sha256_file(pipeline._canonical_path("paper_a"))

    policy_dir = Path(identity["run_directory"]) / "bridge_policy_runs" / "policy_1"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_payload = {
        "strict_run_directory": identity["run_directory"],
        "canonical_graph_sha256": canonical_sha,
        "bridge_extraction_id": "extract_1",
        "bridge_policy_run_id": "policy_1",
        "bridge_policy_run_fingerprint": "policy_fp_1",
        "domain_profile_id": "sers_au_ag",
    }
    (policy_dir / "run.json").write_text(json.dumps(policy_payload), encoding="utf-8")
    (Path(identity["run_directory"]) / "latest_bridge_policy_run.json").write_text(
        json.dumps(
            {
                "bridge_policy_run_directory": str(policy_dir.resolve()),
                "bridge_extraction_id": "extract_1",
                "bridge_policy_run_id": "policy_1",
                "bridge_policy_run_fingerprint": "policy_fp_1",
            }
        ),
        encoding="utf-8",
    )

    bridge = nx.MultiDiGraph()
    bridge.graph.update(
        {
            "bridge_extraction_id": "extract_1",
            "bridge_policy_run_id": "policy_1",
            "bridge_policy_run_fingerprint": "policy_fp_1",
            "domain_profile_id": "sers_au_ag",
        }
    )
    bridge.add_node("b1", type="BridgeConcept")
    nx.write_graphml(bridge, pipeline._bridge_path("paper_a"))

    binding = pipeline._current_bridge_binding("paper_a")
    assert binding is not None
    assert binding["canonical_sha256"] == canonical_sha
    assert binding["bridge_policy_run_id"] == "policy_1"

    policy_payload["canonical_graph_sha256"] = "stale"
    (policy_dir / "run.json").write_text(json.dumps(policy_payload), encoding="utf-8")
    assert pipeline._current_bridge_binding("paper_a") is None


def test_bridge_status_distinguishes_empty_and_useful(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline._bridge_path("paper_a").parent.mkdir(parents=True, exist_ok=True)

    empty = nx.MultiDiGraph()
    empty.add_node("m1", type="Material")
    nx.write_graphml(empty, pipeline._bridge_path("paper_a"))
    status, counts = pipeline._bridge_status("paper_a")
    assert status == "BRIDGE_EMPTY"
    assert counts["bridge_concepts"] == 0

    useful = nx.MultiDiGraph()
    useful.add_node("b1", type="BridgeConcept")
    useful.add_node("m1", type="Material")
    useful.add_edge("b1", "m1", relation="ANCHORS_TO")
    nx.write_graphml(useful, pipeline._bridge_path("paper_a"))
    status, counts = pipeline._bridge_status("paper_a")
    assert status == "BRIDGE_USEFUL"
    assert counts == {"bridge_concepts": 1, "bridge_edges": 1}


def test_commands_propagate_domain_and_data_root(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    command = pipeline._paper_command("paper_a", "bridge")
    assert "--domain-profile" in command
    assert "sers_au_ag" in command
    assert "--data-root" in command
    assert str((tmp_path / "data_sers").resolve()) in command

    corpus = pipeline._corpus_command(["paper_a"])
    assert "--paper-ids" in corpus
    assert corpus[-1] == "paper_a"


def test_run_isolates_failed_paper_and_builds_corpus_from_usable_subset(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "papers.yaml"
    _write_config(config)
    pipeline = StrictBridgeCorpusPipeline(
        project_root=tmp_path,
        config=config,
        corpus_id="pilot_tolerant",
        domain_profile="sers_au_ag",
        data_root=tmp_path / "data_sers",
        options=StrictBridgePipelineOptions(heartbeat_seconds=0),
    )

    def fake_extract(paper_id: str):
        if paper_id == "paper_b":
            return False, "run", {"status": "rejected"}
        return True, "run", {
            "status": "complete",
            "run_id": "run_a",
            "run_fingerprint": "fp_a",
            "attempt_id": "attempt_a",
        }

    monkeypatch.setattr(pipeline, "_run_extract", fake_extract)
    monkeypatch.setattr(pipeline, "_run_paper_graph", lambda paper_id: (True, "run"))
    monkeypatch.setattr(pipeline, "_run_bridge", lambda paper_id: (True, "run"))
    monkeypatch.setattr(
        pipeline,
        "_bridge_status",
        lambda paper_id: ("BRIDGE_USEFUL", {"bridge_concepts": 1, "bridge_edges": 1}),
    )
    monkeypatch.setattr(pipeline, "_current_bridge_binding", lambda paper_id: {})
    monkeypatch.setattr(pipeline, "_run_projection", lambda paper_id: (True, "run"))

    captured: dict[str, list[str]] = {}

    def fake_corpus(paper_ids):
        captured["paper_ids"] = list(paper_ids)
        return True, "run"

    monkeypatch.setattr(pipeline, "_run_corpus", fake_corpus)
    summary = pipeline.run()

    assert summary["status"] == "passed_with_paper_skips"
    assert summary["usable_paper_ids"] == ["paper_a"]
    assert captured["paper_ids"] == ["paper_a"]

    outcomes = [
        json.loads(line)
        for line in Path(summary["paper_outcomes"]).read_text(encoding="utf-8").splitlines()
    ]
    assert outcomes[0]["strict_status"] == "STRICT_USABLE"
    assert outcomes[1]["strict_status"] == "STRICT_REJECTED"


def test_extract_resume_does_not_require_orchestrator_state(tmp_path: Path, monkeypatch) -> None:
    pipeline = _pipeline(tmp_path)
    identity = _write_latest_strict_run(pipeline)
    assert pipeline.state.get("papers", {}) == {}

    monkeypatch.setattr(
        pipeline,
        "_strict_run_matches_current_inputs",
        lambda paper_id, current: paper_id == "paper_a" and current == pipeline._latest_extraction_identity("paper_a"),
    )

    assert pipeline._extract_resume_safe("paper_a") is True


def test_extract_resume_rejects_stale_strict_run_metadata(tmp_path: Path, monkeypatch) -> None:
    pipeline = _pipeline(tmp_path)
    _write_latest_strict_run(pipeline)
    monkeypatch.setattr(
        pipeline,
        "_strict_run_matches_current_inputs",
        lambda paper_id, current: False,
    )
    assert pipeline._extract_resume_safe("paper_a") is False


def test_dry_run_summary_is_not_reported_as_usable(tmp_path: Path, monkeypatch) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.options = StrictBridgePipelineOptions(
        mode="mechanism",
        heartbeat_seconds=0,
        dry_run=True,
        skip_corpus=True,
    )
    monkeypatch.setattr(pipeline, "_run_extract", lambda paper_id: (True, "dry_run", None))
    monkeypatch.setattr(pipeline, "_run_paper_graph", lambda paper_id: (True, "dry_run"))
    monkeypatch.setattr(pipeline, "_run_bridge", lambda paper_id: (True, "dry_run"))
    monkeypatch.setattr(pipeline, "_run_projection", lambda paper_id: (True, "dry_run"))

    summary = pipeline.run()
    assert summary["status"] == "dry_run"
    assert summary["planned_paper_count"] == 1
    assert summary["usable_paper_count"] is None
    assert summary["usable_paper_ids"] == []
