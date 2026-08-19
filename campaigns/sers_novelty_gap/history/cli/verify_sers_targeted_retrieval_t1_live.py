from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.literature_provider_plan import LiteratureProviderPlan
from dac_her.novelty_refinement_contracts import NoveltyGapPlan
from campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_validation import (
    aggregate_t1_report,
    audit_live_gap_outcome,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPEC_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_spec_v1"
)
DEFAULT_RUN_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1"
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    args = parser.parse_args()

    report_path = args.run_root / "t1_live_report.json"
    if not report_path.is_file():
        print("T1 offline verification: FAIL")
        print(" - t1_live_report.json missing")
        print("Network calls during verification:", 0)
        return 2

    base_plan = LiteratureQueryPlan.model_validate_json(
        (args.spec_root / "base_query_plan.json")
        .read_text(encoding="utf-8")
    )
    base_packet = PriorArtPacket.model_validate_json(
        (args.spec_root / "base_prior_art_packet.json")
        .read_text(encoding="utf-8")
    )
    gap_plan = NoveltyGapPlan.model_validate_json(
        (args.spec_root / "novelty_gap_plan.json")
        .read_text(encoding="utf-8")
    )
    provider_plan = LiteratureProviderPlan.model_validate_json(
        (args.spec_root / "provider_plan.json")
        .read_text(encoding="utf-8")
    )
    spec = json.loads(
        (args.spec_root / "t1_spec.json")
        .read_text(encoding="utf-8")
    )

    gap_audits = []
    skipped = []
    issues = []
    for index, gap in enumerate(gap_plan.gaps, start=1):
        gap_root = args.run_root / f"gap_{index:02d}"
        if not gap.targeted_queries:
            skip_path = gap_root / "SKIPPED.json"
            if not skip_path.is_file():
                issues.append(
                    f"missing skip evidence for gap {index}"
                )
            else:
                skipped.append(
                    json.loads(
                        skip_path.read_text(encoding="utf-8")
                    )
                )
            continue

        required = {
            "augmented": gap_root / "augmented_plan.json",
            "delta_plan": gap_root / "delta_plan.json",
            "delta_packet": gap_root / "delta_packet.json",
            "merged": gap_root / "merged_packet.json",
            "audit": gap_root / "gap_audit.json",
        }
        missing = [
            name for name, path in required.items()
            if not path.is_file()
        ]
        if missing:
            issues.append(
                f"gap {index} missing artifacts: {missing}"
            )
            continue

        augmented = LiteratureQueryPlan.model_validate_json(
            required["augmented"].read_text(encoding="utf-8")
        )
        delta_plan = LiteratureQueryPlan.model_validate_json(
            required["delta_plan"].read_text(encoding="utf-8")
        )
        delta_packet = PriorArtPacket.model_validate_json(
            required["delta_packet"].read_text(encoding="utf-8")
        )
        merged = PriorArtPacket.model_validate_json(
            required["merged"].read_text(encoding="utf-8")
        )
        stored_audit = json.loads(
            required["audit"].read_text(encoding="utf-8")
        )
        recomputed = audit_live_gap_outcome(
            base_plan=base_plan,
            base_packet=base_packet,
            gap=gap,
            provider_plan=provider_plan,
            augmented_plan=augmented,
            delta_plan=delta_plan,
            delta_packet=delta_packet,
            merged_packet=merged,
        )
        if _canonical(stored_audit) != _canonical(recomputed):
            issues.append(f"gap {index} audit drift")
        gap_audits.append(recomputed)

    recomputed_report = aggregate_t1_report(
        gap_plan_id=gap_plan.plan_id,
        provider_plan=provider_plan,
        gap_audits=gap_audits,
        skipped_gaps=skipped,
        total_targeted_query_count=int(
            spec["targeted_query_count"]
        ),
    )
    stored_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    if _canonical(stored_report) != _canonical(recomputed_report):
        issues.append("global T1 report drift")

    if issues:
        print("T1 offline verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls during verification:", 0)
        print("LLM calls during verification:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("T1 offline verification: PASS")
    print("Run ID:", stored_report["run_id"])
    print("Outcome:", stored_report["outcome"])
    print("Provider mode:", stored_report["provider_mode"])
    print("Providers:", stored_report["providers"])
    print(
        "Targeted queries:",
        stored_report["total_targeted_query_count"],
    )
    print(
        "Provider executions:",
        stored_report["successful_provider_execution_count"],
        "success /",
        stored_report["failed_provider_execution_count"],
        "failed",
    )
    print(
        "Delta works:",
        stored_report["delta_canonical_work_count"],
        "canonical /",
        stored_report["delta_abstract_work_count"],
        "with abstract",
    )
    print(
        "Every targeted query operational:",
        stored_report["every_targeted_query_operational"],
    )
    print("Network calls during verification:", 0)
    print("Ranker called:", False)
    print("Claim reviewer called:", False)
    print("LLM calls during verification:", 0)
    print("Hypothesis rewrite called:", False)
    print("Fresh Reserve C consumed:", False)
    print("Scientific novelty reassessed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
