from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.discovery.empirical_revision_policy_m4c0 import build_policy_replay
from pipeline_core.discovery.empirical_evidence_event_m4c1 import (
    _safe_source, audit_event_template, make_event_template, validate_source,
)
from scripts.discovery.build_empirical_evidence_event_m4c1 import run


def frozen_inputs(tmp_path: Path, *, exclusive: bool = False):
    relationships = "COMPETING_PARTIAL" if exclusive else "ADJACENT_NOT_EXCLUSIVE"
    report = {
        "schema_version": "m4a1-scientific-confrontation-v1",
        "status": "DRAFT_CONFRONTATIONS_NEED_REVIEW",
        "external_evidence_validated": False, "human_or_expert_science_certification": False,
        "hypothesis_cards_modified": False, "source_files_mutated": False,
        "production_selection_changed": False, "llm_or_network_calls": 0,
        "case_count": 1,
        "cases": [{
            "case_id": "QA_TEST", "status": "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED",
            "hypothesis_refs": ["B03", "B06"],
            "input_hypotheses": [
                {"ref": "B03", "hypothesis_id": "hypothesis:aa", "terminal_idea_id": "research_idea:a", "scientific_intent": "test a"},
                {"ref": "B06", "hypothesis_id": "hypothesis:bb", "terminal_idea_id": "research_idea:b", "scientific_intent": "test b"},
            ],
            "confrontation_validated": False, "experiment_physically_feasible_certified": False,
            "falsification_logic_certified": False, "novelty_or_truth_authority": False,
            "proposed_experiment_UNREVIEWED": {
                "pair_relationship": relationships,
                "observables": [
                    {"name": "SERS ratio", "role": "TARGET_OUTCOME"},
                    {"name": "independent orientation", "role": "INDEPENDENT_DISCRIMINATOR"},
                ]},
            "proposed_mechanisms_UNREVIEWED": [{"role": "A"}, {"role": "B"}],
        }],
    }
    p = tmp_path / "source_m4a1.json"
    p.write_text(json.dumps(report, sort_keys=True))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    replay = build_policy_replay(report, source_sha256=sha)
    c = tmp_path / "source_m4c0.json"
    c.write_text(json.dumps(replay, sort_keys=True))
    sha0 = hashlib.sha256(c.read_bytes()).hexdigest()
    return p, c, report, replay, sha, sha0


def inputs(tmp_path):
    p, c, a1, c0, s1, s0 = frozen_inputs(tmp_path)
    cases = validate_source(a1, c0, s1)
    template = make_event_template(cases, m4a1_sha=s1, m4c0_sha=s0)
    root = tmp_path / "evidence"
    root.mkdir()
    return cases, template, root, (p, c, s1, s0)


def audit(cases, template, root, ids):
    p, c, s1, s0 = ids
    return audit_event_template(template, cases, root=root, m4a1_sha=s1, m4c0_sha=s0)


def complete_row(row, *, path="observations.csv", sha="", klass="PRIMARY_INSTRUMENT"):
    row.update({
        "source_class": klass, "source_relative_path": path, "source_sha256": sha,
        "value_locator": "column:intensity_ratio", "sample_id": "sample-001",
        "condition_id": "condition-A", "time_window": "t0-t1",
        "measurement_method": "instrument log", "units": "dimensionless ratio",
        "uncertainty_description": "replicate variability recorded separately",
        "confound_controls": "temperature and excitation fixed",
        "independence_basis": "different physical acquisition protocol",
        "origin_citation": "local measurement log run-01", "reviewer_notes": "unreviewed provenance assertion",
    })
    return row


def test_template_has_only_unreviewed_rows(tmp_path):
    cases, template, root, ids = inputs(tmp_path)
    assert len(template["events"]) == 2
    assert all(e["source_class"] == "UNCLASSIFIED" for e in template["events"])
    assert template["is_verified_scientific_evidence"] is False
    assert all(e["source_relative_path"] == "" for e in template["events"])


def test_empty_template_abstains(tmp_path):
    cases, template, root, ids = inputs(tmp_path)
    result = audit(cases, template, root, ids)
    assert result["status"] == "EVIDENCE_EVENT_INTAKE_INCOMPLETE"
    assert result["file_sha256_match_count"] == 0
    assert result["new_idea_generated"] is False
    assert result["independence_certified"] is False
    assert result["scientific_truth_or_falsification_authority"] is False
    assert result["cases"][0]["original_terminal_idea_ids_PRESERVED"] == ["research_idea:a", "research_idea:b"]


def test_file_bytes_match_but_no_science_authority(tmp_path):
    cases, template, root, ids = inputs(tmp_path)
    file = root / "observations.csv"
    file.write_bytes(b"intensity_ratio,orientation\n1.5,42\n")
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    for row in template["events"]:
        complete_row(row, sha=sha)
    result = audit(cases, template, root, ids)
    assert result["status"] == "SOURCE_METADATA_ONLY_EXPERT_REVIEW_REQUIRED"
    assert result["file_sha256_match_count"] == 2
    assert result["real_measurement_values_read"] is False
    assert all(x["metadata_complete_as_declared"] for x in result["events"])
    assert all(x["physical_measurement_independence_certified"] is False for x in result["events"])


@pytest.mark.parametrize("failure", [
    "SOURCE_PATH_NOT_PROVIDED", "SOURCE_FILE_NOT_FOUND", "SOURCE_SHA256_MISSING_OR_INVALID",
    "SOURCE_SHA256_MISMATCH", "MISSING_VALUE_LOCATOR", "MISSING_SAMPLE_ID",
    "MISSING_CONDITION_ID", "MISSING_TIME_WINDOW", "MISSING_MEASUREMENT_METHOD",
    "MISSING_UNITS", "MISSING_UNCERTAINTY_DESCRIPTION", "MISSING_CONFOUND_CONTROLS",
    "MISSING_ORIGIN_CITATION", "INDEPENDENCE_BASIS_NOT_DESCRIBED",
    "REVIEWER_NOTE_UNRECORDED", "NOT_PRIMARY_EMPIRICAL_MEASUREMENT",
    "SOURCE_CLASS_UNCLASSIFIED",
])
def test_fail_closed_metadata_issues(tmp_path, failure):
    cases, template, root, ids = inputs(tmp_path)
    file = root / "observations.csv"
    file.write_text("hello")
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    row = complete_row(template["events"][1], sha=sha)
    if failure == "SOURCE_PATH_NOT_PROVIDED": row["source_relative_path"] = ""
    if failure == "SOURCE_FILE_NOT_FOUND": row["source_relative_path"] = "missing.csv"
    if failure == "SOURCE_SHA256_MISSING_OR_INVALID": row["source_sha256"] = "invalid"
    if failure == "SOURCE_SHA256_MISMATCH": row["source_sha256"] = "0"*64
    fields = {
        "MISSING_VALUE_LOCATOR": "value_locator", "MISSING_SAMPLE_ID": "sample_id",
        "MISSING_CONDITION_ID": "condition_id", "MISSING_TIME_WINDOW": "time_window",
        "MISSING_MEASUREMENT_METHOD": "measurement_method", "MISSING_UNITS": "units",
        "MISSING_UNCERTAINTY_DESCRIPTION": "uncertainty_description",
        "MISSING_CONFOUND_CONTROLS": "confound_controls", "MISSING_ORIGIN_CITATION": "origin_citation",
        "INDEPENDENCE_BASIS_NOT_DESCRIBED": "independence_basis",
        "REVIEWER_NOTE_UNRECORDED": "reviewer_notes",
    }
    if failure in fields: row[fields[failure]] = "unknown"
    if failure == "NOT_PRIMARY_EMPIRICAL_MEASUREMENT": row["source_class"] = "KG_DERIVED"
    if failure == "SOURCE_CLASS_UNCLASSIFIED": row["source_class"] = "UNCLASSIFIED"
    result = audit(cases, template, root, ids)
    assert failure in result["events"][1]["issues"]
    assert result["scientific_truth_or_falsification_authority"] is False


@pytest.mark.parametrize("unsafe", ["../secret.txt", "/etc/passwd", "../../../../etc/passwd", "~/secret", "C:\\outside.txt"])
def test_root_escape_is_rejected(tmp_path, unsafe):
    root = tmp_path / "ev"; root.mkdir()
    path, issue = _safe_source(root, unsafe)
    assert path is None and issue == "SOURCE_PATH_OUTSIDE_ALLOWED_ROOT"


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "ev"; root.mkdir()
    external = tmp_path / "secret.txt"; external.write_text("abc")
    (root / "link.txt").symlink_to(external)
    path, issue = _safe_source(root, "link.txt")
    assert path is None and issue == "SOURCE_PATH_OUTSIDE_ALLOWED_ROOT" or issue == "SYMLINK_SOURCE_DISALLOWED"


@pytest.mark.parametrize("mutation", ["role", "foreign", "duplicate", "schema", "sha", "claim", "key"])
def test_template_contract_rejects_mutations(tmp_path, mutation):
    cases, template, root, ids = inputs(tmp_path)
    if mutation == "role": template["events"][0]["observable_role"] = "INDEPENDENT_DISCRIMINATOR"
    elif mutation == "foreign": template["events"][0]["case_id"] = "OTHER"
    elif mutation == "duplicate": template["events"][1]["event_id"] = template["events"][0]["event_id"]
    elif mutation == "schema": template["schema_version"] = "arbitrary"
    elif mutation == "sha": template["source_m4a1_sha256"] = "0"*64
    elif mutation == "claim": template["is_verified_scientific_evidence"] = True
    elif mutation == "key": template["events"][0]["new_authority"] = True
    with pytest.raises(ValueError, match="M4C1_INTEGRITY_FAILURE"):
        audit(cases, template, root, ids)


@pytest.mark.parametrize("mutation", ["c0_sha", "c0_claim", "c0_lineage", "a1_status"])
def test_source_integrity_rejects_mutations(tmp_path, mutation):
    p,c,a1,c0,s1,s0 = frozen_inputs(tmp_path)
    if mutation == "c0_sha": c0["input_m4a1_sha256"] = "0"*64
    elif mutation == "c0_claim": c0["scientific_learning_demonstrated"] = True
    elif mutation == "c0_lineage": c0["cases"][0]["original_terminal_idea_ids_PRESERVED"] = ["fake"]
    elif mutation == "a1_status": a1["status"] = "CERTIFIED"
    with pytest.raises(ValueError): validate_source(a1,c0,s1)


def test_run_writes_new_output_and_never_overwrites(tmp_path):
    p,c,a1,c0,s1,s0 = frozen_inputs(tmp_path)
    dest = tmp_path / "output"
    result = run(m4a1=p, expected_m4a1_sha256=s1, m4c0=c,
                 expected_m4c0_sha256=s0, output_dir=dest)
    assert result["expected_event_count"] == 2
    assert (dest / "M4C1_EVENT_TEMPLATE_PRIVATE.json").exists()
    assert (dest / "M4C1_EVENT_AUDIT_PRIVATE.json").exists()
    assert (dest / "M4C1_REPORT_PRIVATE.md").exists()
    with pytest.raises(FileExistsError):
        run(m4a1=p, expected_m4a1_sha256=s1, m4c0=c,
            expected_m4c0_sha256=s0, output_dir=dest)


def test_run_rejects_wrong_source_sha_before_creation(tmp_path):
    p,c,a1,c0,s1,s0 = frozen_inputs(tmp_path)
    dest = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run(m4a1=p, expected_m4a1_sha256="0"*64, m4c0=c,
            expected_m4c0_sha256=s0, output_dir=dest)
    assert not dest.exists()


def test_run_audits_submitted_events(tmp_path):
    p,c,a1,c0,s1,s0 = frozen_inputs(tmp_path)
    cases = validate_source(a1, c0, s1)
    template = make_event_template(cases, m4a1_sha=s1, m4c0_sha=s0)
    root = tmp_path / "data"; root.mkdir()
    evidence = root / "measurements.csv"; evidence.write_text("value\n3\n")
    for row in template["events"]:
        complete_row(row, path="measurements.csv", sha=hashlib.sha256(evidence.read_bytes()).hexdigest())
    events = tmp_path / "completed.json"; events.write_text(json.dumps(template))
    result = run(m4a1=p, expected_m4a1_sha256=s1, m4c0=c,
                 expected_m4c0_sha256=s0, output_dir=tmp_path / "result", events=events, evidence_root=root)
    assert result["file_sha256_match_count"] == 2
    assert result["original_research_ideas_modified"] is False


def test_run_rejects_events_without_root(tmp_path):
    p,c,a1,c0,s1,s0 = frozen_inputs(tmp_path)
    events = tmp_path / "completed.json"; events.write_text("{}")
    with pytest.raises(ValueError, match="evidence-root"):
        run(m4a1=p, expected_m4a1_sha256=s1, m4c0=c,
            expected_m4c0_sha256=s0, output_dir=tmp_path / "result", events=events)


def test_omitted_observable_not_promoted(tmp_path):
    cases, template, root, ids = inputs(tmp_path)
    template["events"] = template["events"][:1]
    result = audit(cases, template, root, ids)
    assert len(result["missing_observable_keys"]) == 1
    assert result["status"] == "EVIDENCE_EVENT_INTAKE_INCOMPLETE"


def test_nonexclusive_pair_is_not_automatically_falsified(tmp_path):
    p,c,a1,c0,s1,s0 = frozen_inputs(tmp_path, exclusive=False)
    cases = validate_source(a1,c0,s1)
    template = make_event_template(cases,m4a1_sha=s1,m4c0_sha=s0)
    result = audit_event_template(template,cases,root=tmp_path,m4a1_sha=s1,m4c0_sha=s0)
    assert all(x["claim_adjudication_authorized"] is False for x in result["cases"])
    assert result["production_selection_changed"] is False
