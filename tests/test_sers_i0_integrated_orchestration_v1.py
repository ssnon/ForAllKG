from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_sers_i0_integrated_orchestration_v1 import (
    ALLOWED_DISPOSITIONS,
    build_handoff,
    canonical,
    validate_r2_portfolio,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "dac_her/sers_i0_integrated_orchestration_spec_v1.json"
RUNNER = ROOT / "scripts/run_sers_i0_integrated_orchestration_v1.py"
FREEZER = ROOT / "scripts/freeze_sers_i0_integrated_orchestration_v1.py"
FREEZE_VERIFIER = ROOT / "scripts/verify_sers_i0_integrated_orchestration_freeze_v1.py"


def load_spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def mock_report() -> dict:
    spec = load_spec()
    return {
        "report_id": spec["source_contract"]["required_r2_report_id"],
        "report_sha256": spec["source_contract"]["required_r2_report_sha256"],
        "hypothesis_decisions": [
            {
                "hypothesis_id": "hypothesis:a",
                "candidate_disposition": "KEEP_BOUNDED_EXTENSION",
                "hypothesis_rewrite_performed": False,
            },
            {
                "hypothesis_id": "hypothesis:b",
                "candidate_disposition": "REJECT_AS_FORMULATED",
                "hypothesis_rewrite_performed": False,
            },
            {
                "hypothesis_id": "hypothesis:c",
                "candidate_disposition": "KEEP_RELATIONAL_GAP_CANDIDATE",
                "hypothesis_rewrite_performed": False,
            },
        ],
        "portfolio_decision": {
            "primary_remaining_candidate_hypothesis_id": "hypothesis:c",
            "secondary_bounded_extension_hypothesis_id": "hypothesis:a",
            "rejected_as_formulated_hypothesis_ids": ["hypothesis:b"],
            "r1_executed": False,
            "hypothesis_rewrites": 0,
            "i0_started": False,
            "fresh_reserve_c_consumed": False,
            "fresh_reserve_c_authorized": False,
            "automatic_next_stage_authorized": False,
            "ranking_is_scientific_review_not_empirical_validation": True,
            "stop_after_r2_freeze": True,
        },
        "epistemic_guards": {
            "r1_executed": False,
            "hypothesis_rewrite_called": False,
            "external_prior_art_used_as_positive_premise": False,
            "literature_absence_claimed": False,
            "runtime_llm_calls": 0,
            "runtime_network_calls": 0,
            "i0_started": False,
            "fresh_reserve_c_consumed": False,
            "fresh_reserve_c_authorized": False,
            "automatic_next_stage_authorized": False,
            "stop_after_r2": True,
        },
    }


def mock_context() -> dict:
    spec = load_spec()
    report = mock_report()
    return {
        "spec": spec,
        "i0_code_commit": "code-commit",
        "r2_report": report,
        "r2_manifest": {
            "freeze_id": spec["source_contract"]["required_r2_freeze_id"],
            "manifest_sha256": spec["source_contract"]["required_r2_manifest_sha256"],
            "source_r2_report_commit": spec["source_contract"]["required_r2_report_commit"],
        },
        "r0_manifest": {
            "freeze_id": spec["source_contract"]["required_r0_freeze_id"],
            "manifest_sha256": spec["source_contract"]["required_r0_manifest_sha256"],
            "source_adjudication_commit": spec["source_contract"][
                "required_r0_source_adjudication_commit"
            ],
        },
        "t1_manifest": {
            "v2_run_id": spec["source_contract"]["required_t1_run_id"],
            "freeze_id": spec["source_contract"]["required_t1_freeze_id"],
            "manifest_sha256": spec["source_contract"]["required_t1_manifest_sha256"],
        },
        "t0_manifest": {
            "freeze_id": spec["source_contract"]["required_t0_freeze_id"],
        },
        "gap_plan": {
            "plan_id": spec["source_contract"]["required_gap_plan_id"],
            "plan_sha256": spec["source_contract"]["required_gap_plan_sha256"],
        },
    }


def test_spec_hash_and_id_are_deterministic() -> None:
    spec = load_spec()
    payload = dict(spec)
    spec_id = payload.pop("spec_id")
    spec_sha = payload.pop("spec_sha256")
    digest = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    assert spec_sha == digest
    assert spec_id == "sers_i0_integrated_orchestration_spec_v1:" + digest[:20]


def test_exact_frozen_r2_boundary_is_locked() -> None:
    source = load_spec()["source_contract"]
    assert source["required_r2_freeze_commit"] == (
        "5aef1b4373b2ac37e5c62fb3e2c41def326a3a34"
    )
    assert source["required_r2_freeze_id"] == (
        "sers_r2_final_reassessment_freeze_v1:aa2f75aa46fb82284db0"
    )
    assert source["required_r2_manifest_sha256"] == (
        "aa2f75aa46fb82284db04c5031b40227b4b2bdd12f4044ccdb8acfa84a1d3831"
    )
    assert source["required_r2_report_id"] == (
        "sers_r2_final_reassessment_report_v1:e9a9502cbfaa7566d457"
    )


def test_i0_runtime_has_no_hypothesis_id_specific_rules() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "direction_aware_trend_hypothesis:" not in source
    assert "H1 =" not in source
    assert "H2 =" not in source
    assert "H3 =" not in source


def test_i0_runtime_has_no_network_or_llm_client_dependency() -> None:
    source = RUNNER.read_text(encoding="utf-8").lower()
    for token in [
        "import requests",
        "import httpx",
        "from openai",
        "import openai",
        "urllib.request",
        "priorartranker(",
        "claimpriorartcompiler(",
    ]:
        assert token not in source


def test_validate_r2_portfolio_accepts_generic_frozen_roles() -> None:
    validate_r2_portfolio(mock_report())
    assert ALLOWED_DISPOSITIONS == {
        "KEEP_BOUNDED_EXTENSION",
        "REJECT_AS_FORMULATED",
        "KEEP_RELATIONAL_GAP_CANDIDATE",
    }


def test_validate_r2_portfolio_rejects_primary_role_drift() -> None:
    report = mock_report()
    report["portfolio_decision"]["primary_remaining_candidate_hypothesis_id"] = (
        "hypothesis:a"
    )
    report["portfolio_decision"]["secondary_bounded_extension_hypothesis_id"] = (
        "hypothesis:c"
    )
    with pytest.raises(ValueError, match="primary candidate disposition mismatch"):
        validate_r2_portfolio(report)


def test_validate_r2_portfolio_rejects_rejected_set_drift() -> None:
    report = mock_report()
    report["portfolio_decision"]["rejected_as_formulated_hypothesis_ids"] = []
    with pytest.raises(ValueError, match="rejected set does not match"):
        validate_r2_portfolio(report)


def test_validate_r2_portfolio_rejects_hypothesis_rewrite() -> None:
    report = mock_report()
    report["hypothesis_decisions"][0]["hypothesis_rewrite_performed"] = True
    with pytest.raises(ValueError, match="records hypothesis rewrite"):
        validate_r2_portfolio(report)


def test_build_handoff_preserves_r2_decisions_exactly() -> None:
    ctx = mock_context()
    handoff = build_handoff(ctx)
    assert canonical(handoff["frozen_r2_hypothesis_decisions"]) == canonical(
        ctx["r2_report"]["hypothesis_decisions"]
    )
    assert canonical(handoff["frozen_r2_portfolio_decision"]) == canonical(
        ctx["r2_report"]["portfolio_decision"]
    )


def test_build_handoff_does_not_authorize_reserve_c() -> None:
    handoff = build_handoff(mock_context())
    reserve = handoff["reserve_c_boundary"]
    assert reserve["readiness_assessed"] is False
    assert reserve["authorized"] is False
    assert reserve["consumed"] is False
    assert reserve["marker_write_allowed"] is False
    assert reserve["holdout_execution_allowed"] is False
    assert handoff["stop_after_i0"] is True


def test_build_handoff_is_orchestration_only() -> None:
    handoff = build_handoff(mock_context())
    guards = handoff["orchestration_guards"]
    assert guards["scientific_reassessment_performed"] is False
    assert guards["new_scientific_judgment_performed"] is False
    assert guards["new_retrieval_performed"] is False
    assert guards["runtime_network_calls"] == 0
    assert guards["runtime_llm_calls"] == 0
    assert handoff["epistemic_usage"] == (
        "frozen_decision_handoff_only_not_new_scientific_evidence"
    )


def test_freezer_and_verifier_require_git_reproducible_outputs() -> None:
    freezer = FREEZER.read_text(encoding="utf-8")
    verifier = FREEZE_VERIFIER.read_text(encoding="utf-8")
    assert "critical file is not tracked in source commit" in freezer
    assert "HANDOFF_PATH" in freezer
    assert "COMPLETE_PATH" in freezer
    assert "source handoff commit missing tracked artifact" in verifier
