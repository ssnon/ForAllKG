from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRoutedCampaignFreeze,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    RoutedExecutionSettings,
    build_prospective_routed_execution_plan,
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
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return head, dirty


def _require_ancestor(older: str, newer: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "routed campaign freeze HEAD is not an ancestor "
            "of the routed execution-plan HEAD"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze identical P11-P15 routed execution semantics before "
            "the first fresh case is generated."
        )
    )
    parser.add_argument("--campaign-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.campaign_freeze.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not source_path.is_file():
        raise ValueError(
            "missing routed campaign freeze: " + str(source_path)
        )
    if output_path.exists():
        raise ValueError("routed execution plan is write-once")

    source = ProspectiveRoutedCampaignFreeze.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    head, dirty = _git_state()
    _require_ancestor(source.repository_head_sha, head)

    plan = build_prospective_routed_execution_plan(
        source_freeze=source,
        source_freeze_file_sha256=_sha256_file(source_path),
        execution_plan_repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
        settings=RoutedExecutionSettings(),
    )
    write_json_exclusive(output_path, plan)

    print("Prospective routed execution plan frozen")
    print("Plan:", plan.plan_id)
    print("Source freeze:", plan.source_campaign_freeze_id)
    print("Source freeze HEAD:", plan.source_campaign_freeze_repository_head_sha)
    print("Execution-plan HEAD:", plan.execution_plan_repository_head_sha)
    print("Cases:", plan.case_ids)
    print()
    print("Pre-route gate:", plan.settings.pre_route_gate)
    print("Post-route gate:", plan.settings.post_route_gate)
    print("Post-route selection:", plan.settings.post_route_selection_rule)
    print("Projection preflight before verifier: true")
    print()
    for protocol in plan.route_protocols:
        print(protocol.route_action)
        print("  primary:", protocol.primary_operation)
        print("  primary attempts:", protocol.primary_max_attempts)
        print(
            "  primary LLM/audit max:",
            protocol.primary_llm_calls_per_unit_max,
            "/",
            protocol.primary_audit_calls_per_unit_max,
        )
        if protocol.fallback_operation:
            print("  fallback:", protocol.fallback_operation)
            print("  fallback attempts:", protocol.fallback_max_attempts)
    print()
    print("Post-case adaptation allowed: false")
    print("Case replacement allowed: false")
    print("Second route attempt after failed post-gate: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
