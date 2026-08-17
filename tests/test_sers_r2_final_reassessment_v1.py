from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "dac_her/sers_r2_final_reassessment_spec_v1.json"
RUNNER = ROOT / "scripts/run_sers_r2_final_reassessment_v1.py"
FREEZER = ROOT / "scripts/freeze_sers_r2_final_reassessment_v1.py"
H1 = "direction_aware_trend_hypothesis:ad13dac8334238124899"
H2 = "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de"
H3 = "direction_aware_trend_hypothesis:1cf889e57332402d88c9"


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def test_spec_hash_and_id_are_deterministic() -> None:
    spec = load_spec()
    payload = dict(spec)
    spec_id = payload.pop("spec_id")
    spec_sha = payload.pop("spec_sha256")
    digest = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    assert spec_sha == digest
    assert spec_id == "sers_r2_final_reassessment_spec_v1:" + digest[:20]


def test_exact_three_hypotheses_and_no_id_specific_runtime_rules() -> None:
    spec = load_spec()
    rows = {row["hypothesis_id"]: row for row in spec["hypothesis_decisions"]}
    assert set(rows) == {H1, H2, H3}
    source = RUNNER.read_text(encoding="utf-8")
    assert "if hypothesis_id == H1" not in source
    assert "if hypothesis_id == H2" not in source
    assert "if hypothesis_id == H3" not in source


def test_h1_is_bounded_extension_not_high_novelty_core() -> None:
    rows = {row["hypothesis_id"]: row for row in load_spec()["hypothesis_decisions"]}
    row = rows[H1]
    assert row["candidate_disposition"] == "KEEP_BOUNDED_EXTENSION"
    assert row["r2_classification"] == "BOUNDED_LITERATURE_SUPPORTED_EXTENSION"
    assert row["hypothesis_rewrite_performed"] is False
    assert row["residual_question_is_new_hypothesis"] is False


def test_h2_is_rejected_as_formulated_not_rewritten() -> None:
    rows = {row["hypothesis_id"]: row for row in load_spec()["hypothesis_decisions"]}
    row = rows[H2]
    assert row["candidate_disposition"] == "REJECT_AS_FORMULATED"
    assert row["r2_classification"] == "KNOWN_MODE_MATCHING_UNSUPPORTED_MONOTONIC_DIRECTION"
    assert row["hypothesis_rewrite_performed"] is False
    assert row["residual_question_is_new_hypothesis"] is True


def test_h3_is_primary_relational_gap_candidate() -> None:
    spec = load_spec()
    rows = {row["hypothesis_id"]: row for row in spec["hypothesis_decisions"]}
    row = rows[H3]
    assert row["candidate_disposition"] == "KEEP_RELATIONAL_GAP_CANDIDATE"
    assert row["r2_classification"] == "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
    assert row["novelty_priority"] == 1
    assert spec["portfolio_decision"]["primary_remaining_candidate_hypothesis_id"] == H3


def test_no_r1_no_rewrite_no_reserve_c_no_auto_next() -> None:
    spec = load_spec()
    p = spec["portfolio_decision"]
    assert p["r1_executed"] is False
    assert p["hypothesis_rewrites"] == 0
    assert p["i0_started"] is False
    assert p["fresh_reserve_c_consumed"] is False
    assert p["fresh_reserve_c_authorized"] is False
    assert p["automatic_next_stage_authorized"] is False
    assert p["stop_after_r2_freeze"] is True


def test_external_prior_art_is_not_positive_premise_and_no_absence_claim() -> None:
    rules = load_spec()["source_rules"]
    assert rules["external_prior_art_can_be_positive_premise"] is False
    assert rules["literature_absence_claimed"] is False


def test_runtime_has_no_network_or_llm_dependency() -> None:
    source = RUNNER.read_text(encoding="utf-8").lower()
    forbidden = ["import requests", "import httpx", "from openai", "import openai", "urllib.request"]
    for token in forbidden:
        assert token not in source
    reviewer = load_spec()["reviewer"]
    assert reviewer["scientific_reviewer_llm_used"] is True
    assert reviewer["runtime_llm_calls"] == 0
    assert reviewer["runtime_network_calls"] == 0


def test_runner_closes_ignored_evaluation_lineage_gap() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"git", "cat-file", "-e"' in source
    assert "R0 source commit does not contain adjudication payload" in source
    assert "required tracked input absent from HEAD" in source


def test_freezer_requires_r2_report_to_be_tracked() -> None:
    source = FREEZER.read_text(encoding="utf-8")
    assert "critical file is not tracked in source commit" in source
    assert "REPORT_PATH" in source
    assert "COMPLETE_PATH" in source


def test_primary_source_dois_are_locked() -> None:
    dois = {row["doi"] for row in load_spec()["primary_source_records"]}
    assert {
        "10.1038/s41598-017-10262-9",
        "10.1021/acsami.0c17929",
        "10.1021/acs.jpcc.0c07701",
        "10.1021/jp5073395",
        "10.1039/B903533H",
        "10.1038/s41598-017-13577-9",
    } == dois
