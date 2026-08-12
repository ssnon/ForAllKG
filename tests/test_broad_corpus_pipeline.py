from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.broad_corpus_pipeline import (
    BroadCorpusPilotPipeline,
    BroadPilotOptions,
    select_broad_paper_ids,
)


def _papers_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "papers.yaml"
    path.write_text(
        yaml.safe_dump({
            "version": 3,
            "papers": {
                "broad_A": {"enabled": True, "documents": []},
                "broad_B": {"enabled": True, "documents": []},
                "broad_C": {"enabled": True, "documents": []},
            },
        }, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_broad_paper_selection_preserves_order_and_limit():
    available = ["A", "B", "C"]
    assert select_broad_paper_ids(
        available,
        requested=["C", "A", "C"],
    ) == ["C", "A"]
    assert select_broad_paper_ids(available, limit=2) == ["A", "B"]


def test_broad_pipeline_uses_domain_specific_bridge_free_projection(tmp_path: Path):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="broad-smoke",
        options=BroadPilotOptions(
            data_root="data_broad",
            dry_run=True,
        ),
        paper_limit=2,
    )

    extract = pipeline.paper_command("broad_A", "extract")
    graph = pipeline.paper_command("broad_A", "paper_graph")
    projection = pipeline.paper_command("broad_A", "projection")
    corpus = pipeline.corpus_command()

    assert extract[1:3] == ["-m", "scripts.extract_paper"]
    assert "--domain-profile" in extract
    assert extract[extract.index("--domain-profile") + 1] == "catalysis_mechanism"
    assert graph[1:3] == ["-m", "scripts.build_paper_graph"]
    assert projection[1:3] == ["-m", "scripts.build_broad_projection"]
    assert "extract_bridge_graph" not in " ".join(projection)
    assert corpus[1:3] == ["-m", "scripts.build_corpus_graph"]
    assert "--no-pattern-alignment" in corpus
    assert corpus[corpus.index("--paper-ids") + 1 :] == ["broad_A", "broad_B"]



def _write_cached_extraction(
    pipeline: BroadCorpusPilotPipeline,
    paper_id: str,
    status: str,
) -> None:
    paper_root = pipeline._paper_root(paper_id)
    run_dir = paper_root / "runs" / "cached-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "active_chunks.json").write_text(
        __import__("json").dumps({
            "paper_id": paper_id,
            "graph_materialization_status": status,
            "quality": {"graph_materialization_status": status},
        }),
        encoding="utf-8",
    )
    paper_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "latest_run.json").write_text(
        __import__("json").dumps({
            "paper_id": paper_id,
            "run_directory": str(run_dir),
        }),
        encoding="utf-8",
    )


def test_cached_rejected_extraction_is_skipped_without_retry(tmp_path: Path, monkeypatch):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="cached-rejected",
        options=BroadPilotOptions(data_root="data_broad"),
        requested_paper_ids=["broad_A"],
    )
    _write_cached_extraction(pipeline, "broad_A", "rejected")

    called = []
    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda command, label: called.append((command, label)) or True,
    )

    assert pipeline._run_paper_stage("broad_A", "extract") is False
    assert called == []
    assert pipeline.records[-1]["status"] == "cached_rejected"


def test_cached_complete_extraction_resume_skips(tmp_path: Path, monkeypatch):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="cached-complete",
        options=BroadPilotOptions(data_root="data_broad"),
        requested_paper_ids=["broad_A"],
    )
    _write_cached_extraction(pipeline, "broad_A", "complete")

    called = []
    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda command, label: called.append((command, label)) or True,
    )

    assert pipeline._run_paper_stage("broad_A", "extract") is True
    assert called == []
    assert pipeline.records[-1]["status"] == "resume_skip"


def test_paper_failure_is_excluded_and_corpus_uses_successful_subset(
    tmp_path: Path,
    monkeypatch,
):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="skip-one",
        options=BroadPilotOptions(data_root="data_broad"),
    )

    def fake_stage(paper_id: str, stage: str) -> bool:
        return not (paper_id == "broad_B" and stage == "extract")

    global_calls = []

    def fake_run(command, *, label):
        global_calls.append((label, command))
        return True

    monkeypatch.setattr(pipeline, "_run_paper_stage", fake_stage)
    monkeypatch.setattr(pipeline, "_run", fake_run)

    manifest_path = pipeline.run()
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["status"] == "complete_with_skips"
    assert payload["usable_paper_ids"] == ["broad_A", "broad_C"]
    assert payload["skipped_paper_count"] == 1
    assert payload["skipped_papers"][0]["paper_id"] == "broad_B"

    corpus_command = next(command for label, command in global_calls if label == "corpus_graph")
    included = corpus_command[corpus_command.index("--paper-ids") + 1 :]
    assert included == ["broad_A", "broad_C"]


def test_fail_fast_remains_available(tmp_path: Path, monkeypatch):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="fail-fast",
        options=BroadPilotOptions(
            data_root="data_broad",
            continue_on_error=False,
        ),
        requested_paper_ids=["broad_A", "broad_B"],
    )
    monkeypatch.setattr(pipeline, "_run_paper_stage", lambda paper_id, stage: False)

    import pytest
    from dac_her.broad_corpus_pipeline import BroadCorpusPipelineError

    with pytest.raises(BroadCorpusPipelineError):
        pipeline.run()


def test_pipeline_runs_extraction_diagnostics_for_all_requested_papers(
    tmp_path: Path,
    monkeypatch,
):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="diagnostics-all",
        options=BroadPilotOptions(data_root="data_broad"),
    )

    def fake_stage(paper_id: str, stage: str) -> bool:
        return not (paper_id == "broad_B" and stage == "extract")

    calls = []

    def fake_run(command, *, label):
        calls.append((label, command))
        return True

    monkeypatch.setattr(pipeline, "_run_paper_stage", fake_stage)
    monkeypatch.setattr(pipeline, "_run", fake_run)
    pipeline.run()

    diagnostic_command = next(
        command for label, command in calls if label == "extraction_diagnostics"
    )
    included = diagnostic_command[diagnostic_command.index("--paper-ids") + 1 :]
    assert included == ["broad_A", "broad_B", "broad_C"]


def test_diagnostics_failure_does_not_block_usable_corpus(tmp_path: Path, monkeypatch):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="diagnostics-best-effort",
        options=BroadPilotOptions(data_root="data_broad"),
        requested_paper_ids=["broad_A"],
    )
    monkeypatch.setattr(
        pipeline,
        "_run_paper_stage",
        lambda paper_id, stage: True,
    )

    def fake_run(command, *, label):
        return label != "extraction_diagnostics"

    monkeypatch.setattr(pipeline, "_run", fake_run)
    manifest = pipeline.run()
    payload = __import__("json").loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"


def _papers_yaml_with_abstract(tmp_path: Path, *, token_words: int = 10) -> Path:
    package = tmp_path / "packages" / "broad_A"
    package.mkdir(parents=True, exist_ok=True)
    (package / "main.md").write_text(
        "# Demo\n\n## Abstract\n\n" + ("mechanism " * token_words),
        encoding="utf-8",
    )
    path = tmp_path / "papers_with_abstract.yaml"
    path.write_text(
        yaml.safe_dump({
            "version": 3,
            "papers": {
                "broad_A": {
                    "enabled": True,
                    "documents": [{
                        "document_id": "abstract",
                        "role": "main",
                        "package_dir": str(package),
                        "markdown_file": "main.md",
                    }],
                },
            },
        }, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_abstract_preflight_flags_fulltext_like_package(tmp_path: Path):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml_with_abstract(tmp_path, token_words=200),
        corpus_id="preflight",
        options=BroadPilotOptions(
            data_root="data_broad",
            max_abstract_source_tokens=50,
        ),
    )
    result = pipeline._abstract_source_preflight("broad_A")
    assert result["checked"] is True
    assert result["outlier"] is True
    assert result["source_tokens_estimated"] > 50


def _write_extraction_identity(
    pipeline: BroadCorpusPilotPipeline,
    paper_id: str,
    *,
    run_id: str = "r-new",
    fingerprint: str = "fp-new",
) -> None:
    paper_root = pipeline._paper_root(paper_id)
    run_dir = paper_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "active_chunks.json").write_text(
        __import__("json").dumps({
            "paper_id": paper_id,
            "run_id": run_id,
            "run_fingerprint": fingerprint,
            "graph_materialization_status": "complete",
            "quality": {"graph_materialization_status": "complete"},
        }),
        encoding="utf-8",
    )
    (paper_root / "latest_run.json").write_text(
        __import__("json").dumps({
            "paper_id": paper_id,
            "run_id": run_id,
            "run_fingerprint": fingerprint,
            "run_directory": str(run_dir),
        }),
        encoding="utf-8",
    )


def test_stale_paper_graph_is_rebuilt_instead_of_resume_skipped(tmp_path: Path, monkeypatch):
    import networkx as nx

    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="stale-graph",
        options=BroadPilotOptions(data_root="data_broad"),
        requested_paper_ids=["broad_A"],
    )
    _write_extraction_identity(pipeline, "broad_A")
    graph_path = pipeline._expected_output("broad_A", "paper_graph")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(
        nx.MultiDiGraph(run_id="r-old", run_fingerprint="fp-old"),
        graph_path,
    )

    calls = []
    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda command, label: calls.append((command, label)) or True,
    )
    assert pipeline._run_paper_stage("broad_A", "paper_graph") is True
    assert calls and calls[0][1] == "broad_A:paper_graph"
    assert any(row.get("status") == "cache_invalidated" for row in pipeline.records)


def test_force_extract_invalidates_existing_graph_and_projection(tmp_path: Path, monkeypatch):
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=_papers_yaml(tmp_path),
        corpus_id="force-chain",
        options=BroadPilotOptions(
            data_root="data_broad",
            force_extract=True,
        ),
        requested_paper_ids=["broad_A"],
    )
    # Existing downstream artifacts must not be resume-skipped after a forced
    # extraction, even if the deterministic extraction run_id is unchanged.
    for stage in ("paper_graph", "projection"):
        path = pipeline._expected_output("broad_A", stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda command, label: calls.append(label) or True,
    )
    assert pipeline._run_paper_stage("broad_A", "extract") is True
    assert pipeline._run_paper_stage("broad_A", "paper_graph") is True
    assert pipeline._run_paper_stage("broad_A", "projection") is True
    assert calls == [
        "broad_A:extract",
        "broad_A:paper_graph",
        "broad_A:projection",
    ]
