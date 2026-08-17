from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.verify_sers_r0_manual_scientific_adjudication_v1 import (
    H1,
    H2,
    H3,
    C_H1_A,
    C_H1_B,
    C_H3_A,
    C_H3_B,
    validate_adjudication,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_v1"
    / "adjudication.json"
)


def _load() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_frozen_adjudication_is_self_consistent() -> None:
    assert validate_adjudication(_load()) == []


def test_expected_claim_statuses() -> None:
    data = _load()
    statuses = {
        (row["hypothesis_id"], row["claim_id"]): row["status"]
        for row in data["claim_assessments"]
    }
    assert statuses[(H1, C_H1_A)] == "DIRECT_PRIOR_ART"
    assert statuses[(H1, C_H1_B)] == "DIRECT_PRIOR_ART"
    assert statuses[(H3, C_H3_A)] == "PARTIAL_PRIOR_ART"
    assert statuses[(H3, C_H3_B)] == "COMPONENTS_ONLY"


def test_h1_direct_coverage_does_not_authorize_r1() -> None:
    data = _load()
    outcome = next(row for row in data["r0_outcomes"] if row["hypothesis_id"] == H1)
    assert outcome["source_action"] == "targeted_search_then_refine"
    assert outcome["evidence_state"] == "directly_covered"
    assert outcome["route"] == "pass_original_to_r2"
    assert outcome["r1_authorized"] is False
    assert outcome["max_refinements_authorized"] == 0


def test_h2_keep_is_preserved() -> None:
    data = _load()
    outcome = next(row for row in data["r0_outcomes"] if row["hypothesis_id"] == H2)
    assert outcome["source_action"] == "keep"
    assert outcome["route"] == "pass_through_frozen"
    assert outcome["r1_authorized"] is False


def test_h3_targeted_search_only_barrier_is_preserved() -> None:
    data = _load()
    outcome = next(row for row in data["r0_outcomes"] if row["hypothesis_id"] == H3)
    assert outcome["source_action"] == "targeted_search_only"
    assert outcome["evidence_state"] == "relational_gap_remains"
    assert outcome["route"] == "pass_original_to_r2"
    assert outcome["r1_authorized"] is False
    assert outcome["max_refinements_authorized"] == 0


def test_reviewer_found_external_prior_art_is_separate_from_t1() -> None:
    data = _load()
    source = next(
        row
        for row in data["primary_source_records"]
        if row["source_id"] == "manual_prior_art:wu2017_core_satellite"
    )
    assert source["origin"] == "reviewer_found_external_prior_art_not_in_frozen_h3_packet"
    assert data["epistemic_guards"]["external_lookup_not_part_of_frozen_t1"] is True
    assert data["epistemic_guards"]["reviewer_found_external_prior_art_kept_separate"] is True
    assert data["epistemic_guards"]["frozen_t1_modified"] is False


def test_no_absence_or_positive_premise_inference() -> None:
    guards = _load()["epistemic_guards"]
    assert guards["literature_absence_claimed"] is False
    assert guards["external_prior_art_used_as_positive_hypothesis_premise"] is False


def test_llm_scientific_review_is_explicit_but_router_remains_deterministic() -> None:
    reviewer = _load()["reviewer"]
    assert reviewer["scientific_reviewer_llm_used"] is True
    assert reviewer["human_scientist_reviewer_present"] is False
    assert reviewer["deterministic_r0_router_llm_calls"] == 0


def test_stage_boundary_is_stop() -> None:
    boundary = _load()["stage_boundary"]
    assert boundary["r0_scientific_adjudication_complete"] is True
    assert boundary["r1_authorized_for_any_hypothesis"] is False
    assert boundary["r2_started"] is False
    assert boundary["fresh_reserve_c_authorized"] is False
    assert boundary["stop_after_freeze"] is True


def test_hash_tamper_is_detected() -> None:
    data = _load()
    tampered = copy.deepcopy(data)
    tampered["claim_assessments"][0]["status"] = "PARTIAL_PRIOR_ART"
    issues = validate_adjudication(tampered)
    assert issues
    assert any(
        "adjudication_sha256 mismatch" in issue
        or "claim status mismatch" in issue
        for issue in issues
    )


def test_r1_tamper_is_detected() -> None:
    data = _load()
    tampered = copy.deepcopy(data)
    tampered["r0_outcomes"][0]["r1_authorized"] = True
    issues = validate_adjudication(tampered)
    assert any("R1 unexpectedly authorized" in issue for issue in issues)


def test_reserve_c_tamper_is_detected() -> None:
    data = _load()
    tampered = copy.deepcopy(data)
    tampered["epistemic_guards"]["fresh_reserve_c_consumed"] = True
    issues = validate_adjudication(tampered)
    assert any("fresh_reserve_c_consumed" in issue for issue in issues)
