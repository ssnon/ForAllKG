from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    ProspectiveAtomicAdmissibilityFreezeV5,
)
from pipeline_core.discovery.prospective_atomic_admissibility_execution_plan_v5 import (
    build_prospective_atomic_admissibility_execution_plan_v5,
    sha256_file,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
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
            "Freeze exact P29-P33 initial/downstream argv before any P29-P33 "
            "result is observed. This plan does not grant execution authority; "
            "a separate comparison-collector contract must be frozen first."
        )
    )
    parser.add_argument("--campaign-freeze", required=True, type=Path)
    parser.add_argument(
        "--regeneration-unit-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args()

    campaign_path = args.campaign_freeze.expanduser().resolve()
    unit_path = args.regeneration_unit_freeze.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    for label, path in (
        ("campaign freeze", campaign_path),
        ("regeneration-unit freeze", unit_path),
    ):
        if not path.is_file():
            raise ValueError("missing " + label + ": " + str(path))
    if output_path.exists():
        raise ValueError(
            "prospective atomic admissibility execution-plan v5 is write-once"
        )

    campaign = ProspectiveAtomicAdmissibilityFreezeV5.model_validate_json(
        campaign_path.read_text(encoding="utf-8")
    )
    unit = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        unit_path.read_text(encoding="utf-8")
    )

    head, dirty = _git_state()
    _require_ancestor(
        campaign.repository_head_sha,
        head,
        "P29-P33 campaign freeze",
    )
    _require_ancestor(
        unit.repository_head_sha,
        head,
        "regeneration-unit freeze",
    )

    settings = ProspectiveAuthorityExecutionSettingsV3(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    plan = build_prospective_atomic_admissibility_execution_plan_v5(
        campaign_freeze=campaign,
        campaign_freeze_file_sha256=sha256_file(campaign_path),
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_path=unit_path,
        regeneration_unit_freeze_file_sha256=sha256_file(unit_path),
        execution_plan_repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
        settings=settings,
    )
    write_json_exclusive(output_path, plan)

    print("Prospective atomic admissibility execution-plan v5 frozen")
    print("Plan:", plan.plan_id)
    print("Repository HEAD:", plan.execution_plan_repository_head_sha)
    print("Cases:", plan.case_ids)
    print("Goal:", plan.prospective_goal)
    print("Comparison collector required before execution: true")
    print("Execution authority granted by this plan: false")
    print("P29-P33 outcomes observed before plan freeze: false")
    print("Later-case adaptation: false")
    print("Result-conditioned route changes: false")
    print("Second regeneration: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
