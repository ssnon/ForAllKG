"""M4-C0 non-authority and replay policy tests, no model calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.discovery.empirical_revision_policy_m4c0 import (
    SCENARIOS, build_policy_replay, render_report, validated_cases,
)
from scripts.discovery.replay_empirical_revision_policy_m4c0 import run


def _source(relation="COMPETING_PARTIAL"):
    return {
        "schema_version": "m4a1-scientific-confrontation-v1",
        "status": "DRAFT_CONFRONTATIONS_NEED_REVIEW",
        "case_count": 1,
        "external_evidence_validated": False,
        "human_or_expert_science_certification": False,
        "hypothesis_cards_modified": False,
        "source_files_mutated": False,
        "production_selection_changed": False,
        "llm_or_network_calls": 0,
        "cases": [{
            "case_id": "CASE_1", "status": "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED",
            "confrontation_validated": False,
            "experiment_physically_feasible_certified": False,
            "falsification_logic_certified": False,
            "novelty_or_truth_authority": False,
            "hypothesis_refs": ["B12"],
            "input_hypotheses": [{"ref": "B12", "hypothesis_id": "hypothesis:1", "terminal_idea_id": "research_idea:1", "scientific_intent": "contrast"}],
            "proposed_experiment_UNREVIEWED": {"pair_relationship": relation},
            "proposed_mechanisms_UNREVIEWED": [{"role": "A"}, {"role": "B"}],
        }],
    }


def _replay(source=None):
    return build_policy_replay(_source() if source is None else source, source_sha256="a"*64)


def test_scenarios_and_count():
    result = _replay()
    assert result["policy_row_count"] == len(SCENARIOS) == 5
    assert {x["synthetic_scenario"] for x in result["cases"]} == set(SCENARIOS)


def test_preserves_original_ids_and_no_scientific_claims():
    result = _replay()
    assert all(x["original_terminal_idea_ids_PRESERVED"] == ["research_idea:1"] for x in result["cases"])
    assert all(not x["new_research_idea_created"] and not x["hypothesis_falsified"] for x in result["cases"])
    assert not result["scientific_learning_demonstrated"]
    assert not result["real_measurements_consumed"]
    assert not result["production_selection_changed"]
    assert not result["scientific_truth_or_falsification_authority"]


def test_neither_prompts_missing_mechanism_not_model_rejection():
    row = next(x for x in _replay()["cases"] if x["synthetic_scenario"] == "NEITHER_COMPATIBLE")
    assert "SEARCH_MISSING_MECHANISM" in row["suggested_review_actions_NOT_EXECUTED"]
    assert "FAILURE_OF_TWO_MODELS_NOT_EXHAUSTIVE" in row["scientific_cautions"]


def test_non_identifiable_no_adjudication():
    row = next(x for x in _replay()["cases"] if x["synthetic_scenario"] == "NOT_IDENTIFIABLE")
    assert "NO_ADJUDICATION_FROM_UNIDENTIFIABLE_OBSERVATION" in row["scientific_cautions"]


def test_nonexclusive_requires_additive_test_for_every_scenario():
    source = _source("ADJACENT_NOT_EXCLUSIVE")
    for row in _replay(source)["cases"]:
        assert "EVALUATE_JOINT_OR_ADDITIVE_EXPLANATORY_POWER" in row["suggested_review_actions_NOT_EXECUTED"]
        assert "NONEXCLUSIVE_MECHANISMS_NO_AUTOMATIC_EXCLUSION" in row["scientific_cautions"]


def test_A_and_B_only_never_prove_mechanism():
    rows = {x["synthetic_scenario"]: x for x in _replay()["cases"]}
    assert "A_NOT_PROVEN" in rows["A_ONLY_IN_SCOPE"]["scientific_cautions"]
    assert "B_NOT_PROVEN" in rows["B_ONLY_IN_SCOPE"]["scientific_cautions"]


def test_source_input_not_mutated():
    source = _source()
    before = copy.deepcopy(source)
    _replay(source)
    assert source == before


@pytest.mark.parametrize("field,value", [
    ("schema_version", "other"), ("status", "CERTIFIED"),
    ("external_evidence_validated", True), ("human_or_expert_science_certification", True),
    ("hypothesis_cards_modified", True), ("source_files_mutated", True),
    ("production_selection_changed", True), ("llm_or_network_calls", 1),
    ("case_count", 2),
])
def test_invalid_header_fail_closed(field, value):
    source = _source()
    source[field] = value
    with pytest.raises(ValueError, match="M4C0_INTEGRITY_FAILURE"):
        validated_cases(source)


@pytest.mark.parametrize("field", [
    "confrontation_validated", "experiment_physically_feasible_certified", "falsification_logic_certified", "novelty_or_truth_authority"
])
def test_authority_flags_fail_closed(field):
    source = _source()
    source["cases"][0][field] = True
    with pytest.raises(ValueError):
        _replay(source)


def test_unknown_pair_relation_fails():
    with pytest.raises(ValueError, match="unknown pair relationship"):
        _replay(_source("EXCLUSIVE"))


def test_duplicate_case_id_fails():
    source = _source()
    source["cases"].append(copy.deepcopy(source["cases"][0]))
    source["case_count"] = 2
    with pytest.raises(ValueError, match="duplicate case id"):
        _replay(source)


def test_missing_parent_id_fails():
    source = _source()
    source["cases"][0]["input_hypotheses"][0]["terminal_idea_id"] = ""
    with pytest.raises(ValueError):
        _replay(source)


def test_unknown_ref_fails():
    source = _source()
    source["cases"][0]["hypothesis_refs"] = ["B01"]
    with pytest.raises(ValueError):
        _replay(source)


def test_bad_source_hash_fails():
    with pytest.raises(ValueError):
        build_policy_replay(_source(), source_sha256="bad")


def test_determinism():
    a, b = _replay(), _replay()
    assert a == b
    assert a["report_id"].startswith("m4c0_synthetic_policy:")


def test_render_report_no_promotion():
    text = render_report(_replay())
    assert "not empirical science" in text
    assert "No deletion" in text


def test_cli_run_and_refuse_existing_dir(tmp_path):
    inp = tmp_path / "input.json"
    raw = json.dumps(_source()).encode()
    inp.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    result = run(confrontation=inp, expected_sha256=digest, output_dir=tmp_path / "out")
    assert result["policy_row_count"] == 5
    assert (tmp_path / "out/M4C0_REPORT_PRIVATE.md").exists()
    with pytest.raises(FileExistsError):
        run(confrontation=inp, expected_sha256=digest, output_dir=tmp_path / "out")


def test_cli_sha_mismatch_prevents_dir_creation(tmp_path):
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps(_source()))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run(confrontation=inp, expected_sha256="0"*64, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cli_invalid_source_prevents_dir_creation(tmp_path):
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"schema_version": "incorrect"}))
    with pytest.raises(ValueError, match="M4C0_INTEGRITY_FAILURE"):
        run(confrontation=inp, expected_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(), output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()
