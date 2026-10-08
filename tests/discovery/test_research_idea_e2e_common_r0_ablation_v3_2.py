from __future__ import annotations

from pathlib import Path

from scripts.discovery.run_research_idea_e2e_common_r0_ablation_v3_2 import (
    _common_complete,
    _common_paths,
    _common_paths_from_root,
)


def test_common_r0_paths_point_to_one_none_realization_branch(tmp_path: Path):
    paths = _common_paths(tmp_path)
    assert paths["root"] == tmp_path / "common_r0"
    assert paths["p0_execution"] == tmp_path / "common_r0" / "p0.execution.json"
    assert (
        paths["lifecycle"]
        == tmp_path
        / "common_r0"
        / "arms"
        / "none"
        / "case_root"
        / "scientific_portfolio_shadow"
        / "sis_v3_0.g2_realization_lifecycle.json"
    )
    assert paths["portfolio"].name == "sis_v3_0.g2_materialized_portfolio.json"
    assert paths["parallel"].name == "sis_v3_0.g2_parallel_partial_search.json"


def test_common_complete_requires_frozen_policy_inputs(tmp_path: Path):
    paths = _common_paths(tmp_path)
    assert _common_complete(paths) is False
    for key, path in paths.items():
        if key == "root":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    assert _common_complete(paths) is True


def test_existing_common_r0_source_can_be_addressed_directly(tmp_path: Path):
    paths = _common_paths_from_root(tmp_path / "existing_sis_v3_2_e2e")
    assert paths["root"] == tmp_path / "existing_sis_v3_2_e2e"
    assert paths["p0_seed"] == tmp_path / "existing_sis_v3_2_e2e" / "p0.seed_report.json"
    assert "arms/none/case_root" in str(paths["lifecycle"])
