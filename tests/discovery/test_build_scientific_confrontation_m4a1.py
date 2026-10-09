"""M4-A1 offline contract regression. These tests never call a model or network."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.discovery import build_scientific_confrontation_m4a1 as m4  # noqa: E402


def save(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def trajectories(tmp_path):
    rows = []
    for i in range(1, 22):
        rows.append({
            "blind_id": f"B{i:02}", "arm": "M3A_NEW" if i > 14 else "V31_FROZEN",
            "hypothesis_id": f"hypothesis:{i:02}",
            "p0_root_idea_id": f"research_idea:root{i:02}",
            "terminal_idea_id": f"research_idea:term{i:02}",
            "lineage_validation": "PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY",
            "steps_earliest_to_latest": [{"idea_id": f"research_idea:term{i:02}", "kernel": {"canonical_intent": f"Question {i}"}}],
            "premise_statement_ids": [f"stmt:{i}"],
            "card_predictions_UNREVIEWED": [{"observable": "SERS intensity"}],
            "card_falsification_criteria_UNREVIEWED": [{"falsifying_outcome": "an unexpected shift"}],
        })
    obj = {"status": "SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW",
           "original_data_mutated": False, "authoritative_science_judgment": False,
           "rows": rows}
    return save(tmp_path / "trajectories.json", obj)


def case():
    return {"case_id": "air-exposure", "hypothesis_refs": ["B12"],
            "mechanisms": [
                {"role": "A", "explanation": "molecular reorientation", "predicted_pattern": "orientation changes during exposure"},
                {"role": "B", "explanation": "field evolution", "predicted_pattern": "field descriptors change under matched orientation"}],
            "intervention": "Vary air exposure time in a controlled series",
            "pair_relationship": "COMPETING_PARTIAL",
            "matched_controls": ["excitation wavelength", "surface morphology"],
            "observables": [
                {"name": "independent orientation", "role": "INDEPENDENT_DISCRIMINATOR", "measurement_method": "orientation-sensitive assay TBD", "independent_of_sers_outcome": "UNKNOWN", "independence_basis": ""},
                {"name": "field distribution", "role": "INDEPENDENT_DISCRIMINATOR", "measurement_method": "electromagnetic proxy TBD", "independent_of_sers_outcome": "UNKNOWN", "independence_basis": ""},
                {"name": "SERS band ratios", "role": "TARGET_OUTCOME", "measurement_method": "polarization SERS", "independent_of_sers_outcome": "NO", "independence_basis": ""},
            ],
            "decision_rule": "Compare orientation and field proxies across time; no automatic winner",
            "inconclusive_rule": "Both change or neither proxy is independently measurable",
            "falsifier": {"target_role": "A", "outcome": "No orientation change under sufficient power", "scope": "MECHANISM_IN_SPECIFIED_CONDITIONS", "necessity_review": "UNREVIEWED", "alternatives_review": "UNREVIEWED"},
            "reviewer": "", "review_source": ""}


def spec(case_obj):
    return {"schema_version": "m4a1-case-spec-v1", "authority": "AI_OR_HUMAN_DRAFT_NOT_CERTIFIED", "cases": [case_obj]}


def run(monkeypatch, tmp_path, cs=None):
    src = trajectories(tmp_path)
    monkeypatch.setattr(m4, "_critique", lambda *_args, **_kwargs: ["MEASUREMENT_INDEPENDENCE_UNRESOLVED"])
    cases = save(tmp_path / "cases.json", spec(cs or case()))
    return m4.execute(trajectories=src, portfolios=[], cases=cases, output_dir=tmp_path / "out")


def test_default_template_no_claim(monkeypatch, tmp_path):
    src = trajectories(tmp_path)
    result = m4.execute(trajectories=src, portfolios=[], cases=None, output_dir=tmp_path / "out", select_refs=["B12", "B01"])
    assert result["status"] == "TEMPLATE_READY_UNREVIEWED"
    template = json.loads((tmp_path / "out" / "M4A1_CASE_SPEC_TEMPLATE_PRIVATE.json").read_text())
    assert [c["hypothesis_refs"] for c in template["cases"]] == [["B12"], ["B01"]]
    assert not result["human_or_expert_science_certification"]


def test_case_status_no_false_science_authority(monkeypatch, tmp_path):
    result = run(monkeypatch, tmp_path)
    c = result["cases"][0]
    assert c["status"] == "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED"
    assert "MEASUREMENT_INDEPENDENCE_NOT_ESTABLISHED" in c["issues"]
    assert "CIRCULAR_OUTCOME_DERIVED_MEASUREMENT" not in c["issues"]
    assert c["existing_experiment_critic_diagnostic_codes"] == ["MEASUREMENT_INDEPENDENCE_UNRESOLVED"]
    assert c["confrontation_validated"] is False


def test_genuine_independent_observation_still_not_certified(monkeypatch, tmp_path):
    c = case()
    for o in c["observables"]:
        o["independent_of_sers_outcome"] = "YES"
        o["independence_basis"] = "A proposed physically separate sensor; not externally audited"
    c["falsifier"]["necessity_review"] = "SUPPORTED"
    c["falsifier"]["alternatives_review"] = "CONTROLLED"
    result = run(monkeypatch, tmp_path, c)["cases"][0]
    assert result["status"] == "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED"
    assert "INDEPENDENT_EXPERT_REVIEW_ABSENT" in result["issues"]
    assert not result["falsification_logic_certified"]


def test_whole_research_question_falsifier_scope_blocked(monkeypatch, tmp_path):
    c = case()
    c["falsifier"]["scope"] = "WHOLE_RESEARCH_QUESTION"
    result = run(monkeypatch, tmp_path, c)["cases"][0]
    assert "INVALID_WHOLE_QUESTION_FALSIFIER_SCOPE" in result["issues"]


def test_unsupported_necessary_falsifier_blocked(monkeypatch, tmp_path):
    c = case()
    c["falsifier"]["necessity_review"] = "UNSUPPORTED"
    assert "FALSIFIER_NOT_NECESSARY_FOR_TARGET" in run(monkeypatch, tmp_path, c)["cases"][0]["issues"]


def test_identical_predictions_fail_closed(monkeypatch, tmp_path):
    c = case()
    c["mechanisms"][1]["predicted_pattern"] = c["mechanisms"][0]["predicted_pattern"]
    with pytest.raises(ValueError, match="identical predictions"):
        run(monkeypatch, tmp_path, c)
    assert not (tmp_path / "out").exists()


def test_missing_measurement_method_rejected(monkeypatch, tmp_path):
    c = case()
    c["observables"][0]["measurement_method"] = ""
    with pytest.raises(ValueError, match="measurement method absent"):
        run(monkeypatch, tmp_path, c)


def test_unsupported_independence_claim_without_basis_rejected(monkeypatch, tmp_path):
    c = case()
    c["observables"][0]["independent_of_sers_outcome"] = "YES"
    with pytest.raises(ValueError, match="without witness"):
        run(monkeypatch, tmp_path, c)


def test_unknown_hypothesis_id_rejected(monkeypatch, tmp_path):
    c = case()
    c["hypothesis_refs"] = ["B99"]
    with pytest.raises(ValueError, match="reference not present"):
        run(monkeypatch, tmp_path, c)


def test_duplicate_blind_id_rejected(tmp_path):
    src = trajectories(tmp_path)
    obj = json.loads(src.read_text())
    obj["rows"][1]["blind_id"] = obj["rows"][0]["blind_id"]
    save(src, obj)
    with pytest.raises(ValueError, match="duplicate blind ID"):
        m4.source_index(trajectories=src, portfolios=[], expected_sha=None)


def test_invalid_trajectory_status_rejected(tmp_path):
    src = trajectories(tmp_path)
    obj = json.loads(src.read_text())
    obj["status"] = "NOT_VERIFIED"
    save(src, obj)
    with pytest.raises(ValueError, match="status invalid"):
        m4.source_index(trajectories=src, portfolios=[], expected_sha=None)


def test_sha_pinning_rejects_byte_change(tmp_path):
    src = trajectories(tmp_path)
    pin = m4.digest(src)
    src.write_text(src.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        m4.source_index(trajectories=src, portfolios=[], expected_sha=pin)


def test_report_not_overwritten(monkeypatch, tmp_path):
    run(monkeypatch, tmp_path)
    cpath = tmp_path / "cases.json"
    with pytest.raises(ValueError, match="already exists"):
        m4.execute(trajectories=tmp_path / "trajectories.json", portfolios=[], cases=cpath, output_dir=tmp_path / "out")


def test_case_spec_duplicate_ids_fail_closed(monkeypatch, tmp_path):
    src = trajectories(tmp_path)
    cases = save(tmp_path / "cases.json", spec(case()))
    data = json.loads(cases.read_text())
    data["cases"].append(copy.deepcopy(data["cases"][0]))
    save(cases, data)
    with pytest.raises(ValueError, match="duplicate case ID"):
        m4.execute(trajectories=src, portfolios=[], cases=cases, output_dir=tmp_path / "out")


def test_source_original_bytes_unchanged(monkeypatch, tmp_path):
    src = trajectories(tmp_path)
    source_sha = m4.digest(src)
    monkeypatch.setattr(m4, "_critique", lambda *_a, **_k: [])
    specpath = save(tmp_path / "cases.json", spec(case()))
    m4.execute(trajectories=src, portfolios=[], cases=specpath, output_dir=tmp_path / "out")
    assert m4.digest(src) == source_sha


def test_invalid_authority_rejected(monkeypatch, tmp_path):
    src = trajectories(tmp_path)
    s = spec(case());s["authority"] = "CERTIFIED"
    specpath = save(tmp_path / "cases.json", s)
    with pytest.raises(ValueError, match="no science authority"):
        m4.execute(trajectories=src, portfolios=[], cases=specpath, output_dir=tmp_path / "out")


def test_nonexclusive_adjacent_program_not_claimed_distinct(monkeypatch, tmp_path):
    c = case()
    c["pair_relationship"] = "ADJACENT_NOT_EXCLUSIVE"
    result = run(monkeypatch, tmp_path, c)["cases"][0]
    assert "NONEXCLUSIVE_MECHANISMS_REQUIRE_JOINT_OR_ADDITIVE_TEST" in result["issues"]
    assert result["confrontation_validated"] is False


def test_circular_discriminator_not_treated_as_independent(monkeypatch, tmp_path):
    c = case()
    c["observables"][0]["independent_of_sers_outcome"] = "NO"
    result = run(monkeypatch, tmp_path, c)["cases"][0]
    assert "CIRCULAR_OUTCOME_DERIVED_MEASUREMENT" in result["issues"]
    assert result["status"] == "BLOCKED_LOGIC_OR_MEASUREMENT"


def test_existing_legacy_critic_adapter_uses_real_source_context(tmp_path):
    pytest.importorskip("pipeline_core.discovery.higher_order_experiment_critic")
    src = trajectories(tmp_path)
    cases = save(tmp_path / "cases.json", spec(case()))
    result = m4.execute(trajectories=src, portfolios=[], cases=cases,
                        output_dir=tmp_path / "out", source_context_id="hypothesis_context:fixture")
    codes = result["cases"][0]["existing_experiment_critic_diagnostic_codes"]
    assert "MEASUREMENT_INDEPENDENCE_UNRESOLVED" in codes
    assert "LEGACY_BASIS_ALIGNMENT_NOT_APPLICABLE_NO_GRAPH_RELATION" in codes
    assert "BASIS_ALIGNMENT_UNRESOLVED" not in codes


def test_portfolio_input_validates_with_existing_hypothesis_model(tmp_path):
    pytest.importorskip("pipeline_core.discovery.hypothesis_contracts")
    context = "hypothesis_context:test"
    obj = {
        "schema_version": "hypothesis-portfolio-v1", "portfolio_id": "portfolio:test",
        "domain_profile_id": "test", "source_context_id": context,
        "source_context_sha256": "context-sha", "source_report_id": "report:test",
        "source_report_sha256": "report-sha", "abstention_reason": None,
        "hypotheses": [{
            "schema_version": "hypothesis-card-v1", "hypothesis_id": "hypothesis:test",
            "domain_profile_id": "test", "source_context_id": context,
            "source_context_sha256": "context-sha", "source_report_id": "report:test",
            "source_report_sha256": "report-sha", "title": "Test", "hypothesis_statement": "Test mechanism comparison",
            "hypothesis_type": "mechanistic_extension", "premise_statement_ids": ["stmt:1"],
            "inferential_bridge": "It may connect two independently testable factors",
            "predicted_observations": [{"observation_id": "observation:1", "observable": "SERS ratio", "expected_direction": "shift", "rationale": "test"}],
            "falsification_criteria": [{"criterion_id": "falsifier:1", "observable": "SERS ratio", "falsifying_outcome": "No shift"}],
            "evidence_profile": {"premise_count": 1, "gap_count": 0, "source_paper_count": 1,
                                 "candidate_premise_count": 0, "reported_premise_count": 1, "synthesis_premise_count": 0},
        }],
    }
    src = save(tmp_path / "portfolio.json", obj)
    rows, inventory = m4.source_index(trajectories=None, portfolios=[f"TEST={src}"], expected_sha=None)
    assert len(rows) == 1
    assert "TEST:hypothesis:test" in rows
    assert inventory["source_kind"] == "HYPOTHESIS_PORTFOLIOS"
