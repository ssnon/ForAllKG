from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_authority_campaign_freeze_v3 import (
    ProspectiveAuthorityCampaignFreezeV3,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    ProspectiveDecompositionProvenanceSpecV4,
    build_prospective_decomposition_provenance_freeze_v4,
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
            "Freeze P26-P28 v4 engineering diagnostic tasks before any v4 "
            "initial hypothesis, semantic, routing, regeneration, or source-ID "
            "coverage output is observed."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--infrastructure-freeze", required=True, type=Path)
    parser.add_argument("--regeneration-unit-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    infrastructure_path = args.infrastructure_freeze.expanduser().resolve()
    unit_path = args.regeneration_unit_freeze.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    for label, path in (
        ("spec", spec_path),
        ("P21-P25 v3 infrastructure freeze", infrastructure_path),
        ("regeneration-unit freeze", unit_path),
    ):
        if not path.is_file():
            raise ValueError("missing " + label + ": " + str(path))
    if output_path.exists():
        raise ValueError("prospective decomposition provenance v4 freeze is write-once")

    spec = ProspectiveDecompositionProvenanceSpecV4.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    infrastructure = ProspectiveAuthorityCampaignFreezeV3.model_validate_json(
        infrastructure_path.read_text(encoding="utf-8")
    )
    unit = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        unit_path.read_text(encoding="utf-8")
    )

    head, dirty = _git_state()
    _require_ancestor(
        infrastructure.repository_head_sha,
        head,
        "P21-P25 v3 infrastructure freeze",
    )
    _require_ancestor(
        unit.repository_head_sha,
        head,
        "regeneration-unit freeze",
    )

    frozen = build_prospective_decomposition_provenance_freeze_v4(
        spec=spec,
        source_spec_sha256=sha256_file(spec_path),
        infrastructure_freeze=infrastructure,
        infrastructure_freeze_file_sha256=sha256_file(infrastructure_path),
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256=sha256_file(unit_path),
        repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
    )
    write_json_exclusive(output_path, frozen)

    print("Prospective decomposition provenance campaign v4 frozen")
    print("Freeze:", frozen.freeze_id)
    print("Repository HEAD:", frozen.repository_head_sha)
    print("Cases:", frozen.case_ids)
    print("Diagnostic goal:", frozen.diagnostic_goal)
    print(
        "Required sidecar:",
        frozen.required_regeneration_decomposition_sidecar_filename,
    )
    print(
        "Required sidecar schema:",
        frozen.required_regeneration_decomposition_sidecar_schema,
    )
    print()
    for task in frozen.tasks:
        print(
            task.case_id,
            "|",
            task.design_axis,
            "|",
            task.relation_family,
            "|",
            task.source,
            "->",
            task.target,
        )
    print()
    print("Scientific validation authority: false")
    print("Later-case adaptation: false")
    print("Failed/terminal-case replacement: false")
    print("Second regeneration: false")
    print("Result-conditioned route changes: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
