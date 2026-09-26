from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_v7 import (
    ProspectiveAtomicAdmissibilityFreezeV7,
    build_prospective_atomic_admissibility_execution_plan_v7,
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


def git_state() -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
    )
    return head, dirty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-freeze", required=True, type=Path)
    parser.add_argument("--regeneration-unit-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args()

    campaign_path = args.campaign_freeze.expanduser().resolve()
    unit_path = args.regeneration_unit_freeze.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("v7 execution plan is write-once")

    campaign = ProspectiveAtomicAdmissibilityFreezeV7.model_validate_json(
        campaign_path.read_text(encoding="utf-8")
    )
    unit = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        unit_path.read_text(encoding="utf-8")
    )
    head, dirty = git_state()

    plan = build_prospective_atomic_admissibility_execution_plan_v7(
        campaign_freeze=campaign,
        campaign_freeze_file_sha256=sha256_file(campaign_path),
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_path=unit_path,
        regeneration_unit_freeze_file_sha256=sha256_file(unit_path),
        execution_plan_repository_head_sha=head,
        repository_worktree_dirty=dirty,
        settings=ProspectiveAuthorityExecutionSettingsV3(
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        ),
    )
    write_json_exclusive(output, plan)
    print("P39-P43 v7 execution plan frozen")
    print("Plan:", plan.plan_id)
    print("HEAD:", plan.execution_plan_repository_head_sha)
    print("Cases:", plan.case_ids)
    print("Full worktree clean: true")
    print("Execution authority granted by plan: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
