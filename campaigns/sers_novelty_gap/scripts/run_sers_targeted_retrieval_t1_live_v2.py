from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.literature_provider_plan import build_literature_providers
from dac_her.literature_retrieval import LiteratureRetriever
from campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_recovery_v2 import (
    ROOT,
    SPEC_ROOT,
    V1_FAILURE_MANIFEST,
    V1_RUN_ROOT,
    V2_RUN_ROOT,
    build_v2_report,
    load_frozen_context,
    recover_v1_gap1_audit,
)
from campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_validation_v2 import (
    audit_live_gap_outcome,
)
from dac_her.targeted_novelty_retrieval import TargetedNoveltyRetriever
from campaigns.sers_novelty_gap.scripts.preflight_sers_targeted_retrieval_t1_live_v2 import (
    validate_v2_preflight,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--confirm-one-shot-t1-live-v2",
        action="store_true",
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=SPEC_ROOT,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=V2_RUN_ROOT,
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    if not args.confirm_one_shot_t1_live_v2:
        parser.error("--confirm-one-shot-t1-live-v2 is required")

    if args.run_root.exists():
        print("T1 v2 live run: REFUSED")
        print(" - v2 run root already exists:", args.run_root)
        return 2

    try:
        state = validate_v2_preflight(
            root=ROOT,
            spec_root=args.spec_root,
            run_root=args.run_root,
        )
        provider_plan = state["provider_plan"]
        providers = build_literature_providers(provider_plan)
    except Exception as exc:
        print("T1 v2 live pre-network guard: FAIL")
        print("Exception type:", type(exc).__name__)
        print("Reason:", str(exc))
        print("No v2 live attempt marker was written.")
        print("Network calls before failure:", 0)
        print("LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    spec = state["spec"]
    base_plan = state["base_plan"]
    base_packet = state["base_packet"]
    gap_plan = state["gap_plan"]
    guard = state["guard"]
    recovered = state["recovered"]
    targeted = TargetedNoveltyRetriever(
        LiteratureRetriever(
            providers,
            results_per_query=int(spec["results_per_query"]),
        )
    )

    failure_manifest = json.loads(
        V1_FAILURE_MANIFEST.read_text(encoding="utf-8")
    )
    args.run_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        args.run_root / "LIVE_ATTEMPT_CONSUMED.json",
        {
            "schema_version":
                "sers-targeted-retrieval-t1-live-attempt-v2",
            "spec_id": spec["spec_id"],
            "provider_plan_id": provider_plan.plan_id,
            "source_git_head": guard["source_git_head"],
            "source_git_branch": guard["source_git_branch"],
            "v1_failure_freeze_id":
                failure_manifest["failure_freeze_id"],
            "v1_gap1_retrieval_replayed": False,
            "v1_recovered_gap_id": recovered["gap_id"],
            "remaining_live_gap_id": gap_plan.gaps[2].gap_id,
            "attempt_started_at_utc":
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
            "one_shot": True,
            "rerun_authorized": False,
            "fresh_reserve_c_consumed": False,
        },
    )

    gap_audits: list[dict[str, Any]] = []
    skipped_gaps: list[dict[str, Any]] = []
    try:
        # Gap 01: recover exact already-retrieved v1 evidence OFFLINE.
        gap1_root = args.run_root / "gap_01"
        gap1_root.mkdir(parents=True, exist_ok=False)
        recovered_audit = recover_v1_gap1_audit(
            spec_root=args.spec_root,
            v1_run_root=V1_RUN_ROOT,
        )
        _atomic_json(gap1_root / "gap_audit.json", recovered_audit)
        _atomic_json(
            gap1_root / "RECOVERED_FROM_V1.json",
            {
                "source_v1_run_root":
                    str(V1_RUN_ROOT.relative_to(ROOT)),
                "source_v1_gap_root":
                    str((V1_RUN_ROOT / "gap_01").relative_to(ROOT)),
                "source_failure_freeze_id":
                    failure_manifest["failure_freeze_id"],
                "gap_id": recovered_audit["gap_id"],
                "hypothesis_id":
                    recovered_audit["hypothesis_id"],
                "network_replayed": False,
                "provider_executions_reused":
                    recovered_audit["observed_execution_count"],
                "audit_sha256":
                    recovered_audit["audit_sha256"],
            },
        )
        gap_audits.append(recovered_audit)

        # Gap 02: frozen keep action, no network.
        gap2 = gap_plan.gaps[1]
        gap2_root = args.run_root / "gap_02"
        gap2_root.mkdir(parents=True, exist_ok=False)
        skip = {
            "gap_id": gap2.gap_id,
            "hypothesis_id": gap2.hypothesis_id,
            "action": gap2.action,
            "targeted_query_count": 0,
            "provider_calls": 0,
            "network_retrieval_authorized": False,
            "reason": "KEEP_ACTION_ZERO_TARGETED_QUERIES",
        }
        _atomic_json(gap2_root / "SKIPPED.json", skip)
        skipped_gaps.append(skip)

        # Gap 03: the only new live retrieval in v2.
        gap3 = gap_plan.gaps[2]
        gap3_root = args.run_root / "gap_03"
        gap3_root.mkdir(parents=True, exist_ok=False)
        outcome = targeted.retrieve(
            base_plan,
            base_packet,
            gap3,
        )
        _atomic_json(
            gap3_root / "augmented_plan.json",
            outcome.augmented_plan.model_dump(mode="json"),
        )
        _atomic_json(
            gap3_root / "delta_plan.json",
            outcome.delta_plan.model_dump(mode="json"),
        )
        _atomic_json(
            gap3_root / "delta_packet.json",
            outcome.delta_packet.model_dump(mode="json"),
        )
        _atomic_json(
            gap3_root / "merged_packet.json",
            outcome.merged_packet.model_dump(mode="json"),
        )
        gap3_audit = audit_live_gap_outcome(
            base_plan=base_plan,
            base_packet=base_packet,
            gap=gap3,
            provider_plan=provider_plan,
            augmented_plan=outcome.augmented_plan,
            delta_plan=outcome.delta_plan,
            delta_packet=outcome.delta_packet,
            merged_packet=outcome.merged_packet,
        )
        _atomic_json(gap3_root / "gap_audit.json", gap3_audit)
        gap_audits.append(gap3_audit)

        report = build_v2_report(
            gap_plan_id=gap_plan.plan_id,
            provider_plan=provider_plan,
            gap_audits=gap_audits,
            skipped_gaps=skipped_gaps,
            total_targeted_query_count=int(
                spec["targeted_query_count"]
            ),
            recovered_v1_provider_execution_count=
                recovered_audit["observed_execution_count"],
            v2_new_provider_execution_count=
                gap3_audit["observed_execution_count"],
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

        print("SERS T1 live targeted retrieval v2")
        print("Run ID:", report["run_id"])
        print("Outcome:", report["outcome"])
        print("Provider mode:", report["provider_mode"])
        print("Providers:", report["providers"])
        print(
            "V1 recovered executions:",
            report["recovered_v1_provider_execution_count"],
        )
        print(
            "V2 new executions:",
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
                "schema_version":
                    "sers-targeted-retrieval-t1-live-fatal-v2",
                "stage":
                    "T1_LIVE_TARGETED_RETRIEVAL_V2",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "v1_gap1_network_replayed": False,
                "fresh_reserve_c_consumed": False,
                "message":
                    "Fatal v2 harness/runtime error; preserve both "
                    "v1 and v2 run roots. Do not silently rerun v2.",
            },
        )
        print("T1 v2 live run: FATAL")
        print("Exception type:", type(exc).__name__)
        print("Exception message:", str(exc))
        print("The v2 attempt remains consumed.")
        print("V1 gap_01 network replayed:", False)
        print("Fresh Reserve C consumed:", False)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
