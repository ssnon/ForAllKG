from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.discovery.audit_p0_scientific_lineage_m2a import (
    NEEDED, analyze, main, read_json, resolve_files,
)


def fixture():
    f1={"idea_id":"frontier:A","source_kind":"KG_AXIS","source_lineage":[{"source_kind":"KG_AXIS","source_object_id":"kg:A"}]}
    f2={"idea_id":"frontier:B","source_kind":"HIGHER_ORDER","source_lineage":[{"source_kind":"HIGHER_ORDER","source_object_id":"ho:B"}]}
    e1={"evolution_id":"evolution:X","operator_id":"CROSS_SOURCE_BRIDGE","parent_idea_ids":["frontier:A","frontier:B"],"lineage_refs":[{"lineage_kind":"FRONTIER_IDEA","source_object_id":"frontier:A"},{"lineage_kind":"FRONTIER_IDEA","source_object_id":"frontier:B"}]}
    c1={"candidate_id":"candidate:A","source_object_id":"frontier:A","origin":"FRONTIER","scientific_intent":"Affects band ratio","conceptual_family_signature":"F1"}
    c2={"candidate_id":"candidate:X","source_object_id":"evolution:X","origin":"EVOLUTION","scientific_intent":"Mediated band ratio","conceptual_family_signature":"F2","external_literature_lineage":True}
    selected=[{"candidate_id":"candidate:A","source_object_id":"frontier:A","origin":"FRONTIER","assigned_profile":"MECHANISM"},{"candidate_id":"candidate:X","source_object_id":"evolution:X","origin":"EVOLUTION","assigned_profile":"NOVELTY"}]
    return {"frontier":{"population_id":"frontier:pop","ideas":[f1,f2]},
            "stage7_idea_evolution":{"report_id":"evolution:report","source_population_id":"frontier:pop","ideas":[e1]},
            "stage7_candidate_pool":{"pool_id":"pool:1","source_population_id":"frontier:pop","source_evolution_report_id":"evolution:report","candidates":[c1,c2]},
            "stage7_selection":{"selection_id":"selected:1","source_pool_id":"pool:1","retained_candidate_ids":["candidate:A","candidate:X"],"entries":selected},
            "frozen_p0_execution":{"source_plan_id":"selected:1","g4_population_nodes":[{"idea_id":"research:A","origin_kind":"FRONTIER","source_object_id":"frontier:A","kernel":{"canonical_intent":"A"}},{"idea_id":"research:X","origin_kind":"EVOLUTION","source_object_id":"evolution:X","kernel":{"canonical_intent":"X"}}]}}


def test_cross_source_evolution_and_direct_frontier_attribution():
    data = analyze(fixture(), graph={"classification_counts":{"UNREACHED":100},"dynamic_sites":{"scripts.discovery.run_dac_discovery_e2e":["1:run_stage", "2:run_stage"]}})
    c = data["counts"]
    assert c["raw_frontier_count"] == 2
    assert c["raw_evolution_count"] == 1
    assert c["selected_p0_by_origin"] == {"EVOLUTION":1,"FRONTIER":1}
    assert c["selected_p0_by_attribution_role"]["EVOLUTION_CROSS_SOURCE"] == 1
    assert data["selected_p0"][1]["source_frontier_parent_ids"] == ["frontier:A","frontier:B"]
    assert c["selected_p0_upstream_source_kind_incidence"] == {"HIGHER_ORDER":1,"KG_AXIS":2}
    assert data["counterfactual_input_coverage_only"]["not_a_counterfactual_selection_or_quality_outcome"]


def test_mismatch_selection_p0_is_failure():
    data=fixture()
    data["frozen_p0_execution"]["g4_population_nodes"][1]["source_object_id"]="evolution:other"
    with pytest.raises(ValueError, match="selected source not represented"):
        analyze(data)


def test_sha_guard_rejects_tampered_frozen_source(tmp_path: Path):
    frozen=tmp_path/"frozen_artifacts"
    frozen.mkdir()
    manifest={"artifacts":[]}
    data=fixture()
    for key in NEEDED:
        file=frozen/f"{key}.json"
        content=json.dumps(data[key]).encode()
        file.write_bytes(content)
        manifest["artifacts"].append({"key":key,"frozen_relpath":f"frozen_artifacts/{key}.json","source_relpath":f"{key}.json","source_sha256":hashlib.sha256(content).hexdigest()})
    found,proof=resolve_files(tmp_path,manifest,None)
    assert len(found) == len(NEEDED) == len(proof)
    (frozen/"stage7_selection.json").write_text("{}")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        resolve_files(tmp_path,manifest,None)


def test_preflight_does_not_fake_attribution(tmp_path: Path):
    data=fixture()
    manifest={"git_head":"deadbeef","artifacts":[{"key":k,"present":True} for k in data]}
    (tmp_path/"freeze.manifest.json").write_text(json.dumps(manifest))
    out=tmp_path/"result"
    assert main(["--freeze-dir",str(tmp_path),"--output-dir",str(out),"--preflight-only"]) == 0
    report=read_json(out/"lineage.attribution.json")
    assert report["status"] == "MANIFEST_ONLY_PREFLIGHT"
    assert report["selected_p0"] == []
    with pytest.raises(FileExistsError):
        main(["--freeze-dir",str(tmp_path),"--output-dir",str(out),"--preflight-only"])


def test_full_offline_cli_and_no_overwrite(tmp_path: Path):
    data=fixture()
    frozen=tmp_path/"frozen_artifacts"
    frozen.mkdir()
    manifest={"git_head":"abcdef","artifacts":[]}
    for key,value in data.items():
        path=frozen/(key+".json")
        content=json.dumps(value).encode()
        path.write_bytes(content)
        manifest["artifacts"].append({"key":key,"frozen_relpath":f"frozen_artifacts/{key}.json","source_relpath":f"{key}.json","source_sha256":hashlib.sha256(content).hexdigest()})
    (tmp_path/"freeze.manifest.json").write_text(json.dumps(manifest))
    out=tmp_path/"out"
    assert main(["--freeze-dir",str(tmp_path),"--output-dir",str(out)]) == 0
    assert (out/"M2A_REPORT.md").exists()
    assert (out/"p0.lineage.csv").exists()
    report=read_json(out/"lineage.attribution.json")
    assert report["status"]=="ATTRIBUTION_COMPLETE"
    assert report["counts"]["selected_p0_count"]==2
