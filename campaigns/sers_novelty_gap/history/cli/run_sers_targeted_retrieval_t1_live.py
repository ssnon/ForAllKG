from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.literature_provider_plan import (
    LiteratureProviderPlan,
    build_literature_providers,
)
from dac_her.literature_retrieval import LiteratureRetriever
from dac_her.novelty_refinement_contracts import NoveltyGapPlan
from dac_her.sers_targeted_retrieval_t1_live_guard import (
    validate_t1_pre_network_guard,
)
from dac_her.sers_targeted_retrieval_t1_live_validation import (
    aggregate_t1_report,
    audit_live_gap_outcome,
)
from dac_her.targeted_novelty_retrieval import (
    TargetedNoveltyRetriever,
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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load(spec_root: Path):
    spec = json.loads(
        (spec_root / "t1_spec.json").read_text(encoding="utf-8")
    )
    base_plan = LiteratureQueryPlan.model_validate_json(
        (spec_root / "base_query_plan.json").read_text(encoding="utf-8")
    )
    base_packet = PriorArtPacket.model_validate_json(
        (spec_root / "base_prior_art_packet.json").read_text(
            encoding="utf-8"
        )
    )
    gap_plan = NoveltyGapPlan.model_validate_json(
        (spec_root / "novelty_gap_plan.json").read_text(encoding="utf-8")
    )
    provider_plan = LiteratureProviderPlan.model_validate_json(
        (spec_root / "provider_plan.json").read_text(encoding="utf-8")
    )
    return spec, base_plan, base_packet, gap_plan, provider_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--confirm-one-shot-t1-live",
        action="store_true",
    )
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
    if not args.run or not args.confirm_one_shot_t1_live:
        parser.error(
            "--run and --confirm-one-shot-t1-live are required"
        )
    if args.run_root.exists():
        print("T1 live run: FAIL")
        print(" - run root already exists:", args.run_root)
        print(" - v1 is one-shot; do not delete it to rerun.")
        return 2

    spec, base_plan, base_packet, gap_plan, provider_plan = _load(
        args.spec_root
    )

    # Guard the complete frozen execution boundary before any run-root
    # creation or provider/network activity.
    try:
        guard = validate_t1_pre_network_guard(
            root=ROOT,
            spec_root=args.spec_root,
            spec=spec,
            base_plan=base_plan,
            base_packet=base_packet,
            gap_plan=gap_plan,
            provider_plan=provider_plan,
        )
        providers = build_literature_providers(provider_plan)
    except Exception as exc:
        print("T1 live pre-network guard: FAIL")
        print("Exception type:", type(exc).__name__)
        print("Reason:", str(exc))
        print("No live attempt marker was written.")
        print("Network calls before failure:", 0)
        print("LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("T1 live pre-network guard: PASS")
    print("Source git HEAD:", guard["source_git_head"])
    print("Spec ID:", guard["spec_id"])
    print("Provider mode:", guard["provider_mode"])
    print("Providers:", guard["providers"])
    print("Network calls during guard:", 0)

    retriever = LiteratureRetriever(
        providers,
        results_per_query=int(spec["results_per_query"]),
    )
    targeted = TargetedNoveltyRetriever(retriever)

    args.run_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        args.run_root / "LIVE_ATTEMPT_CONSUMED.json",
        {
            "schema_version":
                "sers-targeted-retrieval-t1-live-attempt-v1",
            "spec_id": spec["spec_id"],
            "provider_plan_id": provider_plan.plan_id,
            "source_git_head": guard["source_git_head"],
            "source_git_branch": guard["source_git_branch"],
            "spec_sha256": guard["spec_sha256"],
            "guarded_preflight_passed": True,
            "attempt_started_at_utc":
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
            "one_shot": True,
            "rerun_authorized": False,
            "fresh_reserve_c_consumed": False,
        },
    )

    gap_audits = []
    skipped_gaps = []
    try:
        for index, gap in enumerate(gap_plan.gaps, start=1):
            gap_root = args.run_root / f"gap_{index:02d}"
            gap_root.mkdir(parents=True, exist_ok=False)

            if not gap.targeted_queries:
                if gap.action != "keep":
                    raise RuntimeError(
                        "non-keep gap unexpectedly has zero targeted queries"
                    )
                skip = {
                    "gap_id": gap.gap_id,
                    "hypothesis_id": gap.hypothesis_id,
                    "action": gap.action,
                    "targeted_query_count": 0,
                    "provider_calls": 0,
                    "network_retrieval_authorized": False,
                    "reason": "KEEP_ACTION_ZERO_TARGETED_QUERIES",
                }
                _atomic_json(gap_root / "SKIPPED.json", skip)
                skipped_gaps.append(skip)
                continue

            outcome = targeted.retrieve(
                base_plan,
                base_packet,
                gap,
            )
            _atomic_json(
                gap_root / "augmented_plan.json",
                outcome.augmented_plan.model_dump(mode="json"),
            )
            _atomic_json(
                gap_root / "delta_plan.json",
                outcome.delta_plan.model_dump(mode="json"),
            )
            _atomic_json(
                gap_root / "delta_packet.json",
                outcome.delta_packet.model_dump(mode="json"),
            )
            _atomic_json(
                gap_root / "merged_packet.json",
                outcome.merged_packet.model_dump(mode="json"),
            )

            audit = audit_live_gap_outcome(
                base_plan=base_plan,
                base_packet=base_packet,
                gap=gap,
                provider_plan=provider_plan,
                augmented_plan=outcome.augmented_plan,
                delta_plan=outcome.delta_plan,
                delta_packet=outcome.delta_packet,
                merged_packet=outcome.merged_packet,
            )
            _atomic_json(gap_root / "gap_audit.json", audit)
            gap_audits.append(audit)

            print(
                f"[T1] gap {index}/{len(gap_plan.gaps)} "
                f"{gap.hypothesis_id}"
            )
            print("  action:", gap.action)
            print("  queries:", audit["delta_query_count"])
            print(
                "  executions:",
                audit["successful_execution_count"],
                "success /",
                audit["failed_execution_count"],
                "failed",
            )
            print(
                "  delta works:",
                audit["delta_canonical_work_count"],
                "canonical /",
                audit["delta_abstract_work_count"],
                "with abstract",
            )
            print("  outcome:", audit["outcome"])

        report = aggregate_t1_report(
            gap_plan_id=gap_plan.plan_id,
            provider_plan=provider_plan,
            gap_audits=gap_audits,
            skipped_gaps=skipped_gaps,
            total_targeted_query_count=int(
                spec["targeted_query_count"]
            ),
        )
        _atomic_json(args.run_root / "t1_live_report.json", report)
        marker = (
            "MECHANICAL_PASS.json"
            if (
                report["all_structural_checks_pass"]
                and report["every_targeted_query_operational"]
            )
            else "MECHANICAL_INCOMPLETE_OR_FAIL.json"
        )
        _atomic_json(
            args.run_root / marker,
            {
                "run_id": report["run_id"],
                "outcome": report["outcome"],
            },
        )

        print()
        print("SERS T1 live targeted retrieval")
        print("Run ID:", report["run_id"])
        print("Outcome:", report["outcome"])
        print("Provider mode:", report["provider_mode"])
        print("Providers:", report["providers"])
        print(
            "Targeted queries:",
            report["total_targeted_query_count"],
        )
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
        print("Ranker called:", False)
        print("Claim reviewer called:", False)
        print("LLM calls:", 0)
        print("Hypothesis rewrite called:", False)
        print("Fresh Reserve C consumed:", False)
        print("Scientific novelty reassessed:", False)
        print("Automatic next stage authorized:", False)
        return (
            0
            if (
                report["all_structural_checks_pass"]
                and report["every_targeted_query_operational"]
            )
            else 2
        )
    except Exception as exc:
        _atomic_json(
            args.run_root / "FATAL_ERROR.json",
            {
                "stage": "T1_LIVE_TARGETED_RETRIEVAL",
                "exception_type": type(exc).__name__,
                "message":
                    "Fatal harness/runtime error; v1 attempt remains "
                    "consumed and must not be silently rerun.",
                "fresh_reserve_c_consumed": False,
            },
        )
        print("T1 live run: FATAL")
        print("Exception type:", type(exc).__name__)
        print(
            "The v1 attempt remains consumed. Preserve the run root; "
            "fix via a new harness version rather than deleting evidence."
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
