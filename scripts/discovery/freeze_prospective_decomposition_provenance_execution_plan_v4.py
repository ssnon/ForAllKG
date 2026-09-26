from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    ProspectiveDecompositionProvenanceFreezeV4,
)
from pipeline_core.discovery.prospective_decomposition_provenance_execution_plan_v4 import (
    build_prospective_decomposition_provenance_execution_plan_v4,
    sha256_file,
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
            "Freeze the P26-P28 v4 engineering diagnostic execution plan "
            "before any P26-P28 initial, semantic, routing, regeneration, "
            "source-ID coverage, or exact-source rebind result is observed."
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
            "prospective decomposition provenance execution-plan v4 "
            "is write-once"
        )

    campaign = ProspectiveDecompositionProvenanceFreezeV4.model_validate_json(
        campaign_path.read_text(encoding="utf-8")
    )
    unit = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        unit_path.read_text(encoding="utf-8")
    )

    head, dirty = _git_state()
    _require_ancestor(campaign.repository_head_sha, head, "campaign freeze")
    _require_ancestor(unit.repository_head_sha, head, "regeneration-unit freeze")

    settings = ProspectiveAuthorityExecutionSettingsV3(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    plan = build_prospective_decomposition_provenance_execution_plan_v4(
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

    print("Prospective decomposition provenance execution-plan v4 frozen")
    print("Plan:", plan.plan_id)
    print("Repository HEAD:", plan.execution_plan_repository_head_sha)
    print("Cases:", plan.case_ids)
    print("Campaign:", plan.source_campaign_freeze_id)
    print("Regeneration unit:", plan.source_regeneration_unit_freeze_id)
    print("Diagnostic goal:", plan.diagnostic_goal)
    print("Scientific validation authority: false")
    print()
    for case in plan.cases:
        print(case.case_id)
        print("  initial:", " ".join(case.initial_e2e_argv))
        print(
            "  downstream base:",
            " ".join(case.downstream_campaign_argv_base),
        )
        print(
            "  sidecar glob:",
            str(
                Path(case.regeneration_reentry_root)
                / case.regeneration_decomposition_sidecar_relative_glob
            ),
        )
        print(
            "  sidecar schema:",
            case.regeneration_decomposition_sidecar_schema,
        )
    print()
    print("Initial scientific outputs observed before plan freeze: false")
    print("Source-ID coverage observed before plan freeze: false")
    print("Exact-source rebind recovery observed before plan freeze: false")
    print("Later-case adaptation: false")
    print("Failed/terminal-case replacement: false")
    print("Result-conditioned route changes: false")
    print("Second regeneration: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
