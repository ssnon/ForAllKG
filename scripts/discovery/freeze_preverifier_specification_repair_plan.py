from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from pipeline_core.discovery.preverifier_specification_diagnostics import (
    PreVerifierSpecificationDiagnosticReport,
)
from pipeline_core.discovery.preverifier_specification_repair_policy import (
    build_specification_repair_campaign_plan,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the P06-P10 zero-scientific-delta specification-repair "
            "policy and exact eligible claim population before any repair "
            "generation is executed."
        )
    )
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--repair-model",
        default="openai/gpt-5.6-luna",
    )
    parser.add_argument(
        "--audit-model",
        default="openai/gpt-5.6-luna",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    args = parser.parse_args()

    diagnostic_path = args.diagnostic.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not diagnostic_path.is_file():
        raise ValueError("missing diagnostic report: " + str(diagnostic_path))
    if output_path.exists():
        raise ValueError(
            "specification repair campaign plan is write-once; "
            "use a fresh output path"
        )

    diagnostic = (
        PreVerifierSpecificationDiagnosticReport.model_validate_json(
            diagnostic_path.read_text(encoding="utf-8")
        )
    )
    head, dirty = _git_state()

    plan = build_specification_repair_campaign_plan(
        diagnostic=diagnostic,
        diagnostic_file_sha256=_sha256_file(diagnostic_path),
        repair_policy_repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
        campaign_root=args.campaign_root,
        repair_model=args.repair_model,
        audit_model=args.audit_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    write_json_exclusive(output_path, plan)

    print("Pre-verifier specification repair plan frozen")
    print("Plan:", plan.plan_id)
    print("Policy HEAD:", plan.repair_policy_repository_head_sha)
    print("Automatic cases:", plan.automatic_case_ids)
    print("Excluded cases:", plan.excluded_case_ids)
    print("Automatic claim repairs:", plan.automatic_claim_repair_count)
    print("Repair action counts:", plan.repair_action_counts)
    print()
    for case in plan.cases:
        print(
            case.case_id,
            "|",
            case.status,
            "|",
            case.source_primary_diagnostic_class,
        )
        if case.excluded_reason:
            print("  excluded:", case.excluded_reason)
        for claim in case.claim_repairs:
            print(
                " ",
                claim.claim_id,
                "|",
                claim.repair_action,
                "| editable=",
                claim.editable_fields,
                "| fill_only=",
                claim.fill_only_fields,
            )

    print()
    print("Claim text mutation allowed: false")
    print("New scientific concepts allowed: false")
    print("Literature retrieval during repair allowed: false")
    print("Max repair attempts per claim: 1")
    print("Semantic delta audit required: true")
    print("Strict binding re-entry required: true")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
