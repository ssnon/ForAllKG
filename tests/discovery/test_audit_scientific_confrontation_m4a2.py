"""Synthetic fixtures only: tests contain no PRIVATE hypothesis or review data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.discovery.audit_scientific_confrontation_m4a2 import (
    evaluate, index_cases, load, run, sha256,
)


def sample_case(cid="synthetic:1", relationship="COMPETING_PARTIAL"):
    return {
        "case_id": cid, "hypothesis_refs": ["example:H1"],
        "status": "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED",
        "confrontation_validated": False,
        "experiment_physically_feasible_certified": False,
        "falsification_logic_certified": False,
        "novelty_or_truth_authority": False,
        "issues": ["MEASUREMENT_INDEPENDENCE_NOT_ESTABLISHED"],
        "existing_experiment_critic_diagnostic_codes": [],
        "input_hypotheses": [{"ref": "example:H1", "arm": "ARM", "hypothesis_id": "h:1", "scientific_intent": "test mechanism"}],
        "proposed_mechanisms_UNREVIEWED": [
            {"role": "A", "explanation": "cause A", "predicted_pattern": "outcome X"},
            {"role": "B", "explanation": "cause B", "predicted_pattern": "outcome Y"},
        ],
        "proposed_experiment_UNREVIEWED": {
            "pair_relationship": relationship,
            "intervention": "vary molecule concentration",
            "matched_controls": ["excitation"],
            "decision_rule": "compare patterns", "inconclusive_rule": "if missing controls",
            "observables": [
                {"name": "SERS ratios", "role": "TARGET_OUTCOME", "measurement_method": "SERS", "independent_of_sers_outcome": "NO", "independence_basis": ""},
                {"name": "independent state", "role": "INDEPENDENT_DISCRIMINATOR", "measurement_method": "separate method TBD", "independent_of_sers_outcome": "UNKNOWN", "independence_basis": ""},
            ],
            "falsifier": {"target_role": "A", "scope": "MECHANISM_IN_SPECIFIED_CONDITIONS", "outcome": "No effect", "necessity_review": "UNREVIEWED", "alternatives_review": "UNREVIEWED"},
        },
    }


def sample_report():
    return {"schema_version": "m4a1-scientific-confrontation-v1",
            "status": "DRAFT_CONFRONTATIONS_NEED_REVIEW", "case_count": 1,
            "case_status_counts": {"REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED": 1},
            "cases": [sample_case()], "input_case_spec_sha256": "a" * 64,
            "external_evidence_validated": False, "human_or_expert_science_certification": False,
            "hypothesis_cards_modified": False, "production_selection_changed": False,
            "source_files_mutated": False, "llm_or_network_calls": 0,
            "source": {"source_count": 21}}


def write_report(tmp_path: Path, data=None) -> Path:
    path = tmp_path / "m4a1.json"
    path.write_text(json.dumps(sample_report() if data is None else data), encoding="utf-8")
    return path


def test_valid_fixture_returns_non_authoritative_status(tmp_path):
    p = write_report(tmp_path)
    out = tmp_path / "result"
    r = run(input_report=p, expected_sha=sha256(p), output_dir=out)
    assert r["status"] == "REVIEW_REQUIRED_NO_SCIENCE_CERTIFICATION"
    assert r["scientific_truth_or_novelty_certified"] is False
    assert r["deletion_or_production_authority"] is False
    assert (out / "M4A2_READINESS_PRIVATE.json").is_file()
    assert (out / "M4A2_EXPERT_REVIEW_TEMPLATE_PRIVATE.json").is_file()
    assert (out / "M4A2_REPORT_PRIVATE.md").is_file()
    assert json.loads((out / "M4A2_EXPERT_REVIEW_TEMPLATE_PRIVATE.json").read_text())["reviews"][0]["prediction_separation"] == "UNREVIEWED"


def test_target_outcome_not_penalized_for_own_dependence():
    row = evaluate(sample_case())
    assert not any(x["state"] == "FAILED" for x in row["structural_review_checks"])
    assert row["readiness"] == "NOT_READY_EMPIRICAL_ADJUDICATION"


def test_nonexclusive_requires_joint_incremental_model():
    row = evaluate(sample_case(relationship="ADJACENT_NOT_EXCLUSIVE"))
    assert "JOINT_INCREMENTAL_MODEL" in [x["check_id"] for x in row["structural_review_checks"]]


def test_competing_partial_requires_mixture_review():
    row = evaluate(sample_case())
    assert "MIXTURE_OR_INTERACTION" in [x["check_id"] for x in row["structural_review_checks"]]


def test_independent_discriminator_derived_from_target_fails():
    row = sample_case()
    row["proposed_experiment_UNREVIEWED"]["observables"][1]["independent_of_sers_outcome"] = "NO"
    assert evaluate(row)["readiness"] == "BLOCKED_DRAFT_INVALID"


def test_whole_question_scope_fails():
    row = sample_case()
    row["proposed_experiment_UNREVIEWED"]["falsifier"]["scope"] = "WHOLE_RESEARCH_QUESTION"
    assert evaluate(row)["readiness"] == "BLOCKED_DRAFT_INVALID"


def test_unsupported_falsifier_necessity_fails():
    row = sample_case()
    row["proposed_experiment_UNREVIEWED"]["falsifier"]["necessity_review"] = "UNSUPPORTED"
    assert evaluate(row)["readiness"] == "BLOCKED_DRAFT_INVALID"


def test_equal_predictions_flag_as_failure():
    row = sample_case()
    row["proposed_mechanisms_UNREVIEWED"][1]["predicted_pattern"] = "outcome X"
    assert evaluate(row)["readiness"] == "BLOCKED_DRAFT_INVALID"


def test_yes_independence_is_not_automatically_certified():
    row = sample_case()
    ind = row["proposed_experiment_UNREVIEWED"]["observables"][1]
    ind["independent_of_sers_outcome"] = "YES"
    ind["independence_basis"] = "some claimed calibration"
    result = evaluate(row)
    assert result["independent_measurement_certified"] is False
    assert result["readiness"] == "NOT_READY_EMPIRICAL_ADJUDICATION"


@pytest.mark.parametrize("field", ["external_evidence_validated", "human_or_expert_science_certification",
                                   "hypothesis_cards_modified", "production_selection_changed", "source_files_mutated"])
def test_source_authority_flags_rejected(field):
    p = sample_report(); p[field] = True
    with pytest.raises(ValueError, match="non-authorizing"):
        index_cases(p)


def test_digest_change_aborts_without_output(tmp_path):
    p = write_report(tmp_path); out = tmp_path / "result"
    prior = sha256(p)
    p.write_text(p.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        run(input_report=p, expected_sha=prior, output_dir=out)
    assert not out.exists()


def test_mismatched_count_rejected():
    p = sample_report(); p["case_count"] = 2
    with pytest.raises(ValueError, match="case_count"):
        index_cases(p)


def test_duplicate_case_rejected():
    p = sample_report(); p["cases"].append(copy.deepcopy(p["cases"][0])); p["case_count"] = 2
    p["case_status_counts"]["REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        index_cases(p)


def test_missing_discriminator_rejected():
    p = sample_report(); p["cases"][0]["proposed_experiment_UNREVIEWED"]["observables"][1]["role"] = "CONTROL"
    with pytest.raises(ValueError, match="no discriminator"):
        index_cases(p)


def test_yes_without_witness_rejected():
    p = sample_report(); p["cases"][0]["proposed_experiment_UNREVIEWED"]["observables"][1]["independent_of_sers_outcome"] = "YES"
    with pytest.raises(ValueError, match="unjustified independence"):
        index_cases(p)


def test_rejects_overwrite(tmp_path):
    p = write_report(tmp_path); out = tmp_path / "result"; out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run(input_report=p, expected_sha=sha256(p), output_dir=out)


def test_invalid_sha_required(tmp_path):
    p = write_report(tmp_path)
    with pytest.raises(ValueError, match="required"):
        run(input_report=p, expected_sha="", output_dir=tmp_path / "out")


def test_source_preserved(tmp_path):
    p = write_report(tmp_path); original = p.read_bytes()
    run(input_report=p, expected_sha=sha256(p), output_dir=tmp_path / "result")
    assert p.read_bytes() == original


def test_status_counter_validation():
    p = sample_report(); p["case_status_counts"] = {"REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED": 0}
    with pytest.raises(ValueError, match="counts"):
        index_cases(p)
