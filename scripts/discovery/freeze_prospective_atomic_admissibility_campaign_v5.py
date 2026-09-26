from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    ProspectiveAtomicAdmissibilitySpecV5,
    build_prospective_atomic_admissibility_freeze_v5,
    sha256_file,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    ProspectiveDecompositionProvenanceFreezeV4,
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
            "Freeze P29-P33 atomic-admissibility source tasks before any "
            "P29-P33 generation, semantic, legacy V_pre, neutral Pre-N10, "
            "source-reference, semantic-fidelity, or comparison outcome exists."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--infrastructure-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--regeneration-unit-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    infra_path = args.infrastructure_freeze.expanduser().resolve()
    unit_path = args.regeneration_unit_freeze.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    for label, path in (
        ("spec", spec_path),
        ("infrastructure freeze", infra_path),
        ("regeneration-unit freeze", unit_path),
    ):
        if not path.is_file():
            raise ValueError("missing " + label + ": " + str(path))
    if output_path.exists():
        raise ValueError(
            "prospective atomic admissibility v5 freeze is write-once"
        )

    spec = ProspectiveAtomicAdmissibilitySpecV5.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    infrastructure = (
        ProspectiveDecompositionProvenanceFreezeV4.model_validate_json(
            infra_path.read_text(encoding="utf-8")
        )
    )
    unit = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        unit_path.read_text(encoding="utf-8")
    )

    head, dirty = _git_state()
    _require_ancestor(
        infrastructure.repository_head_sha,
        head,
        "P26-P28 v4 infrastructure freeze",
    )
    _require_ancestor(
        unit.repository_head_sha,
        head,
        "regeneration-unit freeze",
    )

    frozen = build_prospective_atomic_admissibility_freeze_v5(
        spec=spec,
        source_spec_sha256=sha256_file(spec_path),
        infrastructure_freeze=infrastructure,
        infrastructure_freeze_file_sha256=sha256_file(infra_path),
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256=sha256_file(unit_path),
        repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
    )
    write_json_exclusive(output_path, frozen)

    print("Prospective atomic admissibility v5 campaign frozen")
    print("Freeze:", frozen.freeze_id)
    print("Repository HEAD:", frozen.repository_head_sha)
    print("Cases:", frozen.case_ids)
    print("Goal:", frozen.prospective_goal)
    print("P29-P33 outputs observed before freeze: false")
    print("Later-case adaptation: false")
    print("Failed-case replacement: false")
    print("Scientific validation authority: false")
    print("Production selection authority: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
