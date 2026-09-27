from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_v7 import (
    ProspectiveAtomicAdmissibilityFreezeV7,
)
from pipeline_core.discovery.prospective_canonical_validation_v8 import (
    build_p44_p48_spec_from_v7_infrastructure,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infrastructure-freeze", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.infrastructure_freeze.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise ValueError("missing P39-P43 v7 freeze: " + str(source_path))
    if output.exists():
        raise ValueError("P44-P48 v8 spec is write-once")

    source = ProspectiveAtomicAdmissibilityFreezeV7.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    spec = build_p44_p48_spec_from_v7_infrastructure(
        infrastructure_freeze=source,
        campaign_root=args.campaign_root,
    )
    write_json_exclusive(output, spec)

    print("P44-P48 canonical validation v8 spec materialized")
    print("Prior scientific outcomes consumed: false")
    print("Prospective goal:", spec.prospective_goal)
    for task in spec.tasks:
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
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
