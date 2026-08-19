from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_recovery_v2 import (
    ROOT,
    SPEC_ROOT,
    V1_RUN_ROOT,
    V2_RUN_ROOT,
    build_v2_report,
    load_frozen_context,
    recover_v1_gap1_audit,
)
from campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_validation_v2 import (
    audit_live_gap_outcome,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    if not V2_RUN_ROOT.is_dir():
        print("T1 v2 offline verification: FAIL")
        print(" - v2 run root missing")
        return 2

    failure_check = subprocess.run(
        [
            sys.executable,
            "-m",
            "campaigns.sers_novelty_gap.scripts.verify_sers_targeted_retrieval_t1_v1_failure_evidence",
        ],
        cwd=ROOT,
        text=True,
    )
    if failure_check.returncode != 0:
        return 2

    fatal = V2_RUN_ROOT / "FATAL_ERROR.json"
    if fatal.exists():
        print("T1 v2 offline verification: FAIL")
        print(" - v2 FATAL_ERROR.json exists")
        print(fatal.read_text(encoding="utf-8"))
        return 2

    (
        spec,
        base_plan,
        base_packet,
        gap_plan,
        provider_plan,
    ) = load_frozen_context(SPEC_ROOT)

    issues: list[str] = []
    recovered = recover_v1_gap1_audit(
        spec_root=SPEC_ROOT,
        v1_run_root=V1_RUN_ROOT,
    )
    saved_recovered = json.loads(
        (V2_RUN_ROOT / "gap_01/gap_audit.json")
        .read_text(encoding="utf-8")
    )
    if _canonical(recovered) != _canonical(saved_recovered):
        issues.append("recovered gap_01 audit mismatch")

    expected_skip = {
        "gap_id": gap_plan.gaps[1].gap_id,
        "hypothesis_id": gap_plan.gaps[1].hypothesis_id,
        "action": gap_plan.gaps[1].action,
        "targeted_query_count": 0,
        "provider_calls": 0,
        "network_retrieval_authorized": False,
        "reason": "KEEP_ACTION_ZERO_TARGETED_QUERIES",
    }
    saved_skip = json.loads(
        (V2_RUN_ROOT / "gap_02/SKIPPED.json")
        .read_text(encoding="utf-8")
    )
    if _canonical(expected_skip) != _canonical(saved_skip):
        issues.append("gap_02 skip artifact mismatch")

    gap3_root = V2_RUN_ROOT / "gap_03"
    augmented = LiteratureQueryPlan.model_validate_json(
        (gap3_root / "augmented_plan.json").read_text(encoding="utf-8")
    )
    delta_plan = LiteratureQueryPlan.model_validate_json(
        (gap3_root / "delta_plan.json").read_text(encoding="utf-8")
    )
    delta_packet = PriorArtPacket.model_validate_json(
        (gap3_root / "delta_packet.json").read_text(encoding="utf-8")
    )
    merged_packet = PriorArtPacket.model_validate_json(
        (gap3_root / "merged_packet.json").read_text(encoding="utf-8")
    )
    gap3_audit = audit_live_gap_outcome(
        base_plan=base_plan,
        base_packet=base_packet,
        gap=gap_plan.gaps[2],
        provider_plan=provider_plan,
        augmented_plan=augmented,
        delta_plan=delta_plan,
        delta_packet=delta_packet,
        merged_packet=merged_packet,
    )
    saved_gap3 = json.loads(
        (gap3_root / "gap_audit.json").read_text(encoding="utf-8")
    )
    if _canonical(gap3_audit) != _canonical(saved_gap3):
        issues.append("gap_03 audit mismatch")

    report = build_v2_report(
        gap_plan_id=gap_plan.plan_id,
        provider_plan=provider_plan,
        gap_audits=[recovered, gap3_audit],
        skipped_gaps=[expected_skip],
        total_targeted_query_count=int(
            spec["targeted_query_count"]
        ),
        recovered_v1_provider_execution_count=
            recovered["observed_execution_count"],
        v2_new_provider_execution_count=
            gap3_audit["observed_execution_count"],
    )
    saved_report = json.loads(
        (V2_RUN_ROOT / "t1_live_report.json")
        .read_text(encoding="utf-8")
    )
    if _canonical(report) != _canonical(saved_report):
        issues.append("global v2 report mismatch")

    marker = json.loads(
        (V2_RUN_ROOT / "LIVE_ATTEMPT_CONSUMED.json")
        .read_text(encoding="utf-8")
    )
    if marker.get("v1_gap1_retrieval_replayed") is not False:
        issues.append("v2 marker indicates v1 network replay")
    if marker.get("fresh_reserve_c_consumed") is not False:
        issues.append("v2 marker records Reserve C consumption")

    if issues:
        print("T1 v2 offline verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls during verification:", 0)
        print("LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("T1 v2 offline verification: PASS")
    print("Run ID:", report["run_id"])
    print("Outcome:", report["outcome"])
    print("Provider mode:", report["provider_mode"])
    print("Providers:", report["providers"])
    print(
        "Recovered V1 provider executions:",
        report["recovered_v1_provider_execution_count"],
    )
    print(
        "V2 new provider executions:",
        report["v2_new_provider_execution_count"],
    )
    print("V1 gap_01 network replayed:", False)
    print(
        "Provider executions:",
        report["successful_provider_execution_count"],
        "success /",
        report["failed_provider_execution_count"],
        "failed",
    )
    print(
        "Delta works:",
        report["delta_canonical_work_count"],
        "canonical /",
        report["delta_abstract_work_count"],
        "with abstract",
    )
    print(
        "Every targeted query operational:",
        report["every_targeted_query_operational"],
    )
    print("All structural checks pass:", report[
        "all_structural_checks_pass"
    ])
    print("Scientific novelty reassessed:", False)
    print("Ranker called:", False)
    print("Claim reviewer called:", False)
    print("LLM calls:", 0)
    print("Hypothesis rewrite called:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
