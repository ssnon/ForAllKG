from __future__ import annotations

import json
import subprocess
import sys

from campaigns.sers_alpha4_epoch.post_t1.cli.run_sers_i0_integrated_orchestration_v1 import (
    COMPLETE_PATH,
    HANDOFF_PATH,
    ROOT,
    build_handoff,
    canonical,
    is_ancestor,
    sha256_bytes,
    validate_inputs,
    validate_r2_portfolio,
)


def main() -> int:
    try:
        ctx = validate_inputs(
            require_output_absent=False,
            enforce_execution_workspace=False,
        )
    except Exception as exc:
        print("SERS I0 integrated orchestration verification: FAIL")
        print(" - input validation:", exc)
        return 2

    issues: list[str] = []
    if not HANDOFF_PATH.is_file():
        issues.append("I0 handoff missing")
    if not COMPLETE_PATH.is_file():
        issues.append("I0 complete marker missing")
    if issues:
        print("SERS I0 integrated orchestration verification: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    handoff = json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))
    marker = json.loads(COMPLETE_PATH.read_text(encoding="utf-8"))
    expected = build_handoff(ctx)
    if canonical(handoff) != canonical(expected):
        issues.append("I0 handoff differs from deterministic reconstruction")

    payload = dict(handoff)
    orchestration_id = payload.pop("orchestration_id", None)
    orchestration_sha = payload.pop("orchestration_sha256", None)
    recomputed = sha256_bytes(canonical(payload).encode("utf-8"))
    if orchestration_sha != recomputed:
        issues.append("I0 orchestration SHA mismatch")
    if orchestration_id != "sers_i0_integrated_orchestration_v1:" + recomputed[:20]:
        issues.append("I0 orchestration ID mismatch")

    if marker.get("complete") is not True or marker.get("stop") is not True:
        issues.append("I0 complete/STOP marker invalid")
    if marker.get("input_lineage_verified") is not True:
        issues.append("I0 input-lineage marker false")
    if marker.get("orchestration_id") != orchestration_id:
        issues.append("I0 marker orchestration ID mismatch")
    if marker.get("orchestration_sha256") != orchestration_sha:
        issues.append("I0 marker orchestration SHA mismatch")

    for key in [
        "scientific_reassessment_performed",
        "r1_executed",
        "fresh_reserve_c_readiness_assessed",
        "fresh_reserve_c_authorized",
        "fresh_reserve_c_consumed",
        "automatic_next_stage_authorized",
    ]:
        if marker.get(key) is not False:
            issues.append(f"I0 marker guard changed:{key}")

    guards = handoff.get("orchestration_guards", {})
    for key in [
        "scientific_reassessment_performed",
        "new_scientific_judgment_performed",
        "new_retrieval_performed",
        "ranker_called",
        "claim_reviewer_called",
        "hypothesis_rewrite_called",
        "r1_executed",
        "external_prior_art_used_as_positive_premise",
        "literature_absence_claimed",
        "automatic_next_stage_authorized",
    ]:
        if guards.get(key) is not False:
            issues.append(f"I0 orchestration guard changed:{key}")
    if guards.get("runtime_network_calls") != 0:
        issues.append("I0 runtime network calls changed")
    if guards.get("runtime_llm_calls") != 0:
        issues.append("I0 runtime LLM calls changed")

    reserve = handoff.get("reserve_c_boundary", {})
    for key in [
        "readiness_assessed",
        "authorized",
        "consumed",
        "marker_write_allowed",
        "holdout_execution_allowed",
    ]:
        if reserve.get(key) is not False:
            issues.append(f"I0 Reserve-C guard changed:{key}")
    if reserve.get("authorization_required") != (
        "explicit_separate_guarded_one_shot_authorization"
    ):
        issues.append("I0 Reserve-C authorization boundary changed")

    if handoff.get("stop_after_i0") is not True:
        issues.append("I0 STOP boundary missing")
    if handoff.get("epistemic_usage") != (
        "frozen_decision_handoff_only_not_new_scientific_evidence"
    ):
        issues.append("I0 epistemic usage changed")

    ledger = handoff.get("stage_ledger", {})
    expected_ledger = {
        "g0_g2": "FROZEN_INPUT_BOUND",
        "t0": "FROZEN_INPUT_BOUND",
        "t1": "FROZEN_VERIFIED",
        "r0": "FROZEN_VERIFIED",
        "r1": "NOT_EXECUTED_NOT_AUTHORIZED",
        "r2": "FROZEN_VERIFIED",
        "i0": "COMPLETE",
        "fresh_reserve_c": "UNTOUCHED_UNAUTHORIZED",
    }
    if ledger != expected_ledger:
        issues.append("I0 stage ledger changed")

    if canonical(handoff.get("frozen_r2_hypothesis_decisions")) != canonical(
        ctx["r2_report"]["hypothesis_decisions"]
    ):
        issues.append("I0 changed frozen R2 hypothesis decisions")
    if canonical(handoff.get("frozen_r2_portfolio_decision")) != canonical(
        ctx["r2_report"]["portfolio_decision"]
    ):
        issues.append("I0 changed frozen R2 portfolio decision")
    try:
        validate_r2_portfolio(
            {
                **ctx["r2_report"],
                "hypothesis_decisions": handoff["frozen_r2_hypothesis_decisions"],
                "portfolio_decision": handoff["frozen_r2_portfolio_decision"],
            }
        )
    except Exception as exc:
        issues.append(f"I0 copied R2 portfolio is invalid:{exc}")

    source_code_commit = handoff.get("source_lineage", {}).get("source_i0_code_commit")
    if not isinstance(source_code_commit, str):
        issues.append("I0 source code commit missing")
    elif not is_ancestor(source_code_commit, "HEAD"):
        issues.append("I0 source code commit not ancestor of HEAD")

    if issues:
        print("SERS I0 integrated orchestration verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Runtime network calls:", 0)
        print("Runtime LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        print("Fresh Reserve C authorized:", False)
        return 2

    print("SERS I0 integrated orchestration verification: PASS")
    print("Orchestration ID:", orchestration_id)
    print("Orchestration SHA256:", orchestration_sha)
    print("Source R2 freeze ID:", handoff["source_lineage"]["source_r2_freeze_id"])
    print(
        "Primary remaining candidate:",
        handoff["frozen_r2_portfolio_decision"][
            "primary_remaining_candidate_hypothesis_id"
        ],
    )
    print("Scientific reassessment performed:", False)
    print("Runtime network calls:", 0)
    print("Runtime LLM calls:", 0)
    print("R1 executed:", False)
    print("Fresh Reserve C readiness assessed:", False)
    print("Fresh Reserve C authorized:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
