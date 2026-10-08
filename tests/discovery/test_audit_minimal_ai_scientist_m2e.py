"""Deterministic, free-of-charge M2-E audit tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[2] / "scripts" / "discovery" / "audit_minimal_ai_scientist_m2e.py"
spec = importlib.util.spec_from_file_location("m2e_audit", MODULE)
m2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2e)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    case = tmp_path / "case"
    p0 = tmp_path / "m2c" / "p0.execution.json"
    summary = tmp_path / "m2c" / "direct_seed.summary.json"
    manifest = tmp_path / "m1" / "freeze.manifest.json"
    src = "xyz"
    h = {"hypothesis_id": "hypothesis:abc", "title": "Different science", "hypothesis_statement": "Competing explanations", "inferential_bridge": "Mechanism hypothesis", "predicted_observations": [{"observable": "ratio", "expected_direction": "shift"}], "falsification_criteria": [{"falsifying_outcome": "no shift"}], "source_paper_ids": ["p1"], "premise_statement_ids": ["s1"], "gap_statement_ids": [], "novelty_status": "not_assessed", "status": "hypothesized"}
    portfolio = {"schema_version": "hypothesis-portfolio-v1", "source_context_id": "c1", "source_context_sha256": src, "hypotheses": [h]}
    expected=[]
    for arm,paths in m2e.ARM_PATHS.items():
        x=save(case / paths["portfolio"], portfolio)
        save(case / paths["summary"], {"final_hypothesis_count": 1, "continuation_generation_llm_calls": 0 if arm=="V31" else 3, "continuation_realization_llm_calls": 10})
        expected.append({"key": paths["freeze_key"], "present": True, "source_sha256": x})
    save(p0, {"report_id": "seed-exec"})
    save(summary, {"old_p0_exact_parity_checked": True, "old_selection_parity_checked": True, "early_hypothesis_materialization_executed": False, "early_materialization_llm_calls": 0, "selected_research_idea_count": 8, "seed_sha256": "abc", "p0_execution_report_id": "seed-exec", "p0_execution": str(p0)})
    save(manifest, {"artifacts": expected, "comparison": {"v3_1": {"source_p0_seed_sha256": "abc"}, "v3_4": {"source_p0_report_id": "seed-exec"}}})
    return case,manifest,summary


def test_offline_outputs_and_no_scientific_certification(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    out=tmp_path/"out"
    result=m2e.analyze(case,manifest,summary,out,"fixed-blind")
    assert result["m2c_exact_p0_parity"]["passed"]
    assert result["arms"]["V31"]["metrics"]["hypothesis_count"]==1
    assert result["arms"]["V34"]["generation_llm_calls"]==3
    assert result["limitations"]["semantic_quality_or_causal_effect_proven"] is False
    assert set(x.name for x in out.iterdir()) == {"M2E_REPORT.md", "m2e.metrics.json", "hypothesis_inventory.csv", "review_packet_blinded.md", "review_key_KEEP_PRIVATE.json", "review_scores_template.csv"}
    packet=(out/"review_packet_blinded.md").read_text()
    assert "V31" not in packet and "V34" not in packet
    assert "hypothesis:abc" not in packet
    assert "Competing explanations" in packet
    assert len(json.loads((out/"review_key_KEEP_PRIVATE.json").read_text())["mapping"])==2


def test_refuse_tampered_portfolio(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    p=case/m2e.ARM_PATHS["V34"]["portfolio"]
    p.write_text(p.read_text().replace("Different science","Manipulated"))
    with pytest.raises(ValueError,match="SHA mismatch"):
        m2e.analyze(case,manifest,summary,tmp_path/"out","key")


def test_refuse_context_mismatch_even_if_manifest_updated(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    p=case/m2e.ARM_PATHS["V34"]["portfolio"]
    obj=json.loads(p.read_text());obj["source_context_id"]="different"
    digest=save(p,obj)
    manifest_data=json.loads(manifest.read_text())
    for artifact in manifest_data["artifacts"]:
        if artifact["key"]=="v3_4_final": artifact["source_sha256"]=digest
    save(manifest,manifest_data)
    with pytest.raises(ValueError,match="different source contexts"):
        m2e.analyze(case,manifest,summary,tmp_path/"out","key")


def test_refuse_m2c_parity_not_confirmed(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    obj=json.loads(summary.read_text());obj["old_p0_exact_parity_checked"]=False
    save(summary,obj)
    with pytest.raises(ValueError,match="exact parity"):
        m2e.analyze(case,manifest,summary,tmp_path/"out","key")


def test_refuse_m2c_early_materialization(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    obj=json.loads(summary.read_text());obj["early_hypothesis_materialization_executed"]=True
    save(summary,obj)
    with pytest.raises(ValueError,match="must not have executed"):
        m2e.analyze(case,manifest,summary,tmp_path/"out","key")


def test_no_overwrite(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    out=tmp_path/"out";out.mkdir()
    with pytest.raises(FileExistsError):
        m2e.analyze(case,manifest,summary,out,"key")


def test_output_outside_source(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    with pytest.raises(ValueError,match="outside frozen case"):
        m2e.analyze(case,manifest,summary,case/"report","key")


def test_duplicate_hypothesis_id_rejected(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    p=case/m2e.ARM_PATHS["V34"]["portfolio"]
    obj=json.loads(p.read_text());obj["hypotheses"].append(obj["hypotheses"][0]);newsha=save(p,obj)
    mm=json.loads(manifest.read_text());
    for a in mm["artifacts"]:
        if a["key"]=="v3_4_final":a["source_sha256"]=newsha
    save(manifest,mm)
    with pytest.raises(ValueError,match="duplicate hypothesis IDs"):
        m2e.analyze(case,manifest,summary,tmp_path/"out","key")


def test_expected_m2c_report_id_matches(tmp_path):
    case,manifest,summary=fixture(tmp_path)
    obj=json.loads(summary.read_text());obj["p0_execution_report_id"]="WRONG";save(summary,obj)
    with pytest.raises(ValueError,match="summary P0 report ID mismatch"):
        m2e.analyze(case,manifest,summary,tmp_path/"out","key")
