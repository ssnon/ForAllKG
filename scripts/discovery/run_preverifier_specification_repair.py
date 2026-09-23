from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from pipeline_core.discovery.preverifier_specification_repair_executor import (
    InstructorSpecificationRepairBackend,
    build_execution_report,
    execute_case_repair,
)
from pipeline_core.discovery.preverifier_specification_repair_policy import (
    SpecificationRepairCampaignPlan,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return head, bool(tracked.strip())


def _require_ancestor(older: str, newer: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "frozen repair-policy HEAD is not an ancestor of executor HEAD"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen P06-P10 zero-scientific-delta specification "
            "repair plan exactly once per eligible claim. This stage performs "
            "no literature retrieval and no verifier execution."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not plan_path.is_file():
        raise ValueError("missing repair plan: " + str(plan_path))
    if output_path.exists():
        raise ValueError(
            "specification repair execution report is write-once; "
            "use a fresh output path"
        )

    plan = SpecificationRepairCampaignPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    head, dirty = _git_state()
    if dirty:
        raise ValueError(
            "specification repair execution requires a clean tracked worktree"
        )
    _require_ancestor(plan.repair_policy_repository_head_sha, head)

    print("Pre-verifier specification repair")
    print("Plan:", plan.plan_id)
    print("Frozen policy HEAD:", plan.repair_policy_repository_head_sha)
    print("Executor HEAD:", head)
    print("Automatic cases:", plan.automatic_case_ids)
    print("Excluded cases:", plan.excluded_case_ids)
    print("Planned claims:", plan.automatic_claim_repair_count)
    print("Max repair attempts/claim: 1")
    print("Literature retrieval: false")
    print("Verifier execution: false")

    if args.dry_run:
        print("LLM calls performed: 0")
        print("Would execute repair generation: true")
        print("Would execute semantic delta audit: true")
        return 0

    backend = InstructorSpecificationRepairBackend(
        repair_model=plan.repair_model,
        audit_model=plan.audit_model,
        api_key_env=plan.api_key_env,
        base_url=plan.base_url,
        temperature=plan.temperature,
        parse_retries=plan.parse_retries,
        timeout_seconds=plan.timeout_seconds,
        telemetry_path=args.telemetry,
        telemetry_context={
            "run_id": plan.plan_id,
        },
    )

    cases = []
    for case_plan in plan.cases:
        print()
        print("=" * 88)
        print(case_plan.case_id, case_plan.status)
        print("=" * 88)
        case_result = execute_case_repair(
            case_plan=case_plan,
            backend=backend,
        )
        cases.append(case_result)
        if case_result.excluded_reason:
            print("Excluded:", case_result.excluded_reason)
        for row in case_result.claim_results:
            print(
                row.claim_id,
                "|",
                row.repair_action,
                "|",
                row.status,
            )
            if row.deterministic_reason_codes:
                print(
                    "  reasons:",
                    row.deterministic_reason_codes,
                )
            if row.semantic_audit is not None:
                print(
                    "  zero scientific delta:",
                    row.semantic_audit.zero_scientific_delta,
                )

    report = build_execution_report(
        plan=plan,
        plan_file_sha256=_sha256_file(plan_path),
        cases=cases,
    )
    write_json_exclusive(output_path, report)

    print()
    print("Specification repair execution complete")
    print("Report:", report.report_id)
    print("Planned claims:", report.planned_claim_count)
    print("Materialized R1 claims:", report.materialized_claim_count)
    print("Claim statuses:", report.claim_status_counts)
    print("Generation LLM calls:", report.generation_llm_call_count)
    print("Audit LLM calls:", report.audit_llm_call_count)
    print("Original prospective results preserved: true")
    print("Literature retrieval performed: false")
    print("Verifier result observed: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
