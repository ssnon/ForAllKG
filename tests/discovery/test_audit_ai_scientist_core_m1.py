from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.discovery.audit_ai_scientist_core_m1 import (
    FREEZE_FILES, REQUIRED, _local_references, _module_catalog,
    build_dependency_inventory, freeze_and_audit,
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts/discovery").mkdir(parents=True)
    (repo / "pipeline_core/discovery").mkdir(parents=True)
    (repo / "run_full_current_ai_scientist_v1.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    for entry in (
        "run_dac_discovery_e2e", "run_research_idea_e2e_search_v3_2",
        "run_standard_portfolio_verification_shadow", "run_research_idea_e2e_search_v3_4",
    ):
        (repo / "scripts/discovery" / f"{entry}.py").write_text("x=1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    for key, rel in FREEZE_FILES.items():
        if key not in REQUIRED:
            continue
        p = case / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"key": key, "final_hypothesis_count": 7}) + "\n", encoding="utf-8")
    return case


def test_static_reachability_tracks_import_and_literal_stage(tmp_path):
    repo = _repo(tmp_path)
    helper = repo / "pipeline_core/discovery/helper.py"
    helper.write_text("value=1\n", encoding="utf-8")
    old = repo / "scripts/discovery/run_dac_discovery_e2e.py"
    old.write_text('from pipeline_core.discovery import helper\nmodule = "scripts.discovery.run_standard_portfolio_verification_shadow"\n')
    new = repo / "scripts/discovery/run_research_idea_e2e_search_v3_4.py"
    new.write_text("from pipeline_core.discovery import helper\n")
    dep = build_dependency_inventory(repo)
    row = {r["module"]: r for r in dep["inventory"]}
    assert row["pipeline_core.discovery.helper"]["classification"] == "SHARED_STATIC_POSSIBLE"
    assert row["scripts.discovery.run_standard_portfolio_verification_shadow"]["old_possible"]
    assert not row["scripts.discovery.run_standard_portfolio_verification_shadow"]["v3_4_possible"]


def test_snapshot_is_immutable_and_fails_closed(tmp_path):
    repo = _repo(tmp_path)
    case = _case(tmp_path)
    output = tmp_path / "freeze"
    manifest = freeze_and_audit(repo, case, output)
    assert manifest["no_network_or_llm_called"]
    assert (output / "AUDIT.md").exists()
    source = case / FREEZE_FILES["v3_4_final"]
    snap = output / "frozen_artifacts/v3_4_final.json"
    assert source.read_bytes() == snap.read_bytes()
    before = snap.read_bytes()
    source.write_text("{}\n", encoding="utf-8")
    assert snap.read_bytes() == before
    with pytest.raises(FileExistsError):
        freeze_and_audit(repo, case, output)


def test_missing_required_is_error_and_no_directory_created(tmp_path):
    repo = _repo(tmp_path)
    case = _case(tmp_path)
    (case / FREEZE_FILES["v3_4_final"]).unlink()
    output = tmp_path / "should_not_exist"
    with pytest.raises(FileNotFoundError, match="Required frozen QA artifacts missing"):
        freeze_and_audit(repo, case, output)
    assert not output.exists()


def test_unreached_is_not_unnecessary(tmp_path):
    repo = _repo(tmp_path)
    (repo / "pipeline_core/discovery/dynamic_only.py").write_text("x=1\n", encoding="utf-8")
    dep = build_dependency_inventory(repo)
    row = next(row for row in dep["inventory"] if row["module"] == "pipeline_core.discovery.dynamic_only")
    assert row["classification"] == "NOT_REACHED_BY_STATIC_DISCOVERY"
    assert dep["unreached_is_not_safe_to_delete"]
