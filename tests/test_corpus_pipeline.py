from __future__ import annotations

import json
from pathlib import Path

import yaml

from dac_her.corpus_pipeline import FrozenCorpusPipeline, PipelineOptions, select_paper_ids


def _make_inputs(root: Path) -> tuple[Path, Path]:
    config = root / "papers.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "papers": {
                    "P1": {"enabled": True, "documents": []},
                    "P2": {"enabled": True, "documents": []},
                    "P3": {"enabled": True, "documents": []},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    frozen = root / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "schema_version": "graphagentsdac-frozen-corpus-v01",
                "corpus_id": "demo",
                "documents": [],
            }
        ),
        encoding="utf-8",
    )
    return config, frozen


def test_paper_selection_preserves_requested_order_and_limit():
    available = ["P1", "P2", "P3"]
    assert select_paper_ids(available, requested=["P3", "P1", "P3"]) == ["P3", "P1"]
    assert select_paper_ids(available, limit=2) == ["P1", "P2"]


def test_pipeline_commands_match_current_repo_contract(tmp_path: Path):
    papers, frozen = _make_inputs(tmp_path)
    runner = FrozenCorpusPipeline(
        project_root=tmp_path,
        papers_yaml=papers,
        frozen_manifest=frozen,
        corpus_id="smoke",
        selected_paper_ids=["P2", "P1"],
        options=PipelineOptions(mode="evidence", dry_run=True, skip_node_index=True),
    )

    extract = runner._paper_command("P2", "extract")
    graph = runner._paper_command("P2", "paper_graph")
    projection = runner._paper_command("P2", "projection")
    corpus = runner._global_command("corpus_graph")

    assert extract[1:3] == ["-m", "scripts.extract_paper"]
    assert "--config" in extract
    assert graph[1:3] == ["-m", "scripts.build_paper_graph"]
    assert projection[1:3] == ["-m", "scripts.build_graphagents_projection"]
    assert projection[-2:] == ["--mode", "evidence"]
    assert corpus[1:3] == ["-m", "scripts.build_corpus_graph"]
    assert corpus[corpus.index("--paper-ids") + 1 :] == ["P2", "P1"]


def test_mechanism_pipeline_adds_bridge_stage(tmp_path: Path):
    papers, frozen = _make_inputs(tmp_path)
    runner = FrozenCorpusPipeline(
        project_root=tmp_path,
        papers_yaml=papers,
        frozen_manifest=frozen,
        corpus_id="smoke-mech",
        selected_paper_ids=["P1"],
        options=PipelineOptions(mode="mechanism", dry_run=True, skip_node_index=True),
    )
    bridge = runner._paper_command("P1", "bridge")
    assert bridge[1:3] == ["-m", "scripts.extract_bridge_graph"]
    assert "--config" in bridge
