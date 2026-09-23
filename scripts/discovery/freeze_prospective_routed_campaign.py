from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRoutedCampaignSpec,
    build_prospective_routed_campaign_freeze,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze fresh P11-P15 source tasks and the pre-verifier "
            "repair/regeneration router policy before generation."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not spec_path.is_file():
        raise ValueError("missing routed campaign spec: " + str(spec_path))
    if output_path.exists():
        raise ValueError("routed campaign freeze is write-once")

    spec = ProspectiveRoutedCampaignSpec.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    head, dirty = _git_state()
    frozen = build_prospective_routed_campaign_freeze(
        spec=spec,
        source_spec_sha256=_sha256_file(spec_path),
        repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
    )
    write_json_exclusive(output_path, frozen)

    print("Prospective routed campaign frozen")
    print("Freeze:", frozen.freeze_id)
    print("Repository HEAD:", frozen.repository_head_sha)
    print("Cases:", frozen.case_ids)
    print("Router schema:", frozen.router_policy.schema_version)
    print()
    print(
        "PROCEED ->",
        frozen.router_policy.proceed_route,
    )
    print(
        "SPECIFICATION_REPAIR_REVIEW ->",
        frozen.router_policy.specification_repair_route,
    )
    print(
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW ->",
        frozen.router_policy.source_contract_route,
    )
    print(
        "DECOMPOSE_OR_REGENERATE_REVIEW ->",
        frozen.router_policy.decompose_route,
    )
    print()
    for task in frozen.tasks:
        print(
            task.case_id,
            "|",
            task.relation_family,
            "|",
            task.source,
            "->",
            task.target,
        )
    print()
    print("Later-case adaptation: false")
    print("Failed-case replacement: false")
    print("Router policy editable after freeze: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
