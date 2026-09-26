from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_v6 import (
    ProspectiveAtomicAdmissibilityFreezeV6,
)
from pipeline_core.discovery.prospective_atomic_admissibility_v7 import (
    ProspectiveAtomicAdmissibilitySpecV7,
    build_prospective_atomic_admissibility_freeze_v7,
    sha256_file,
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
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--infrastructure-freeze", required=True, type=Path)
    parser.add_argument("--regeneration-unit-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    infra_path = args.infrastructure_freeze.expanduser().resolve()
    unit_path = args.regeneration_unit_freeze.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("v7 campaign freeze is write-once")

    spec = ProspectiveAtomicAdmissibilitySpecV7.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    infra = ProspectiveAtomicAdmissibilityFreezeV6.model_validate_json(
        infra_path.read_text(encoding="utf-8")
    )
    unit = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        unit_path.read_text(encoding="utf-8")
    )
    head, dirty = git_state()

    frozen = build_prospective_atomic_admissibility_freeze_v7(
        spec=spec,
        source_spec_sha256=sha256_file(spec_path),
        infrastructure_freeze=infra,
        infrastructure_freeze_file_sha256=sha256_file(infra_path),
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256=sha256_file(unit_path),
        repository_head_sha=head,
        repository_worktree_dirty=dirty,
    )
    write_json_exclusive(output, frozen)
    print("P39-P43 v7 campaign frozen")
    print("Freeze:", frozen.freeze_id)
    print("HEAD:", frozen.repository_head_sha)
    print("Cases:", frozen.case_ids)
    print("Full worktree clean: true")
    print("Outputs observed before freeze: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
