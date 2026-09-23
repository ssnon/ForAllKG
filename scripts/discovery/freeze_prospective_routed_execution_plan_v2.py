from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    ProspectiveRegenerationDownstreamV2Freeze,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze_v2 import (
    ProspectiveRoutedCampaignFreezeV2,
)
from pipeline_core.discovery.prospective_routed_execution_plan_v2 import (
    RoutedExecutionSettingsV2,
    build_prospective_routed_execution_plan_v2,
    sha256_file,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


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


def _require_ancestor(older: str, newer: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(label + " HEAD is not an ancestor of current HEAD")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze P16-P20 routed execution-plan v2. Regeneration fallback "
            "is represented only by regeneration-unit-v2 and downstream-v2 "
            "lineage contracts; no regeneration full-E2E argv is generated."
        )
    )
    parser.add_argument("--campaign-freeze", required=True, type=Path)
    parser.add_argument(
        "--regeneration-unit-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--regeneration-downstream-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    campaign_path = args.campaign_freeze.expanduser().resolve()
    unit_path = args.regeneration_unit_freeze.expanduser().resolve()
    downstream_path = (
        args.regeneration_downstream_freeze.expanduser().resolve()
    )
    output_path = args.output.expanduser().resolve()

    for label, path in (
        ("campaign freeze", campaign_path),
        ("regeneration-unit freeze", unit_path),
        ("regeneration-downstream freeze", downstream_path),
    ):
        if not path.is_file():
            raise ValueError("missing " + label + ": " + str(path))
    if output_path.exists():
        raise ValueError("routed execution-plan v2 is write-once")

    campaign = ProspectiveRoutedCampaignFreezeV2.model_validate_json(
        campaign_path.read_text(encoding="utf-8")
    )
    unit = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        unit_path.read_text(encoding="utf-8")
    )
    downstream = ProspectiveRegenerationDownstreamV2Freeze.model_validate_json(
        downstream_path.read_text(encoding="utf-8")
    )

    head, dirty = _git_state()
    _require_ancestor(
        campaign.repository_head_sha,
        head,
        "campaign freeze",
    )
    _require_ancestor(
        unit.repository_head_sha,
        head,
        "regeneration-unit freeze",
    )
    _require_ancestor(
        downstream.repository_head_sha,
        head,
        "regeneration-downstream freeze",
    )

    plan = build_prospective_routed_execution_plan_v2(
        source_freeze=campaign,
        source_freeze_file_sha256=sha256_file(campaign_path),
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256=sha256_file(unit_path),
        regeneration_downstream_freeze=downstream,
        regeneration_downstream_freeze_file_sha256=sha256_file(
            downstream_path
        ),
        execution_plan_repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
        settings=RoutedExecutionSettingsV2(),
    )
    write_json_exclusive(output_path, plan)

    print("Prospective routed execution-plan v2 frozen")
    print("Plan:", plan.plan_id)
    print("Campaign:", plan.source_campaign_freeze_id)
    print("Repository HEAD:", plan.execution_plan_repository_head_sha)
    print("Cases:", plan.case_ids)
    print("Regeneration unit:", plan.source_regeneration_unit_freeze_id)
    print(
        "Regeneration downstream:",
        plan.source_regeneration_downstream_freeze_id,
    )
    print()
    print("Initial full E2E is regeneration: false")
    print("Regeneration full E2E argv present: false")
    print(
        "Regeneration operation:",
        plan.settings.regeneration_operation,
    )
    print(
        "Regeneration downstream:",
        plan.settings.regeneration_downstream_operation,
    )
    print()
    for protocol in plan.route_protocols:
        print(protocol.route_action)
        print("  primary:", protocol.primary_operation)
        if protocol.fallback_operation:
            print("  fallback:", protocol.fallback_operation)
            print(
                "  structured generation calls max:",
                protocol.fallback_structured_generation_calls_per_hypothesis_max,
            )
            print(
                "  repair calls max:",
                protocol.fallback_repair_calls_per_hypothesis_max,
            )
            print("  full E2E fallback argv allowed: false")
            print(
                "  downstream:",
                protocol.fallback_downstream_operation,
            )
    print()
    print("Later-case adaptation: false")
    print("Case replacement: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
