from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_canonical_validation_v8 import (
    ProspectiveCanonicalValidationFreezeV8,
)
from pipeline_core.discovery.prospective_canonical_validation_v9 import (
    build_p49_p53_spec_from_v8_infrastructure,
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
        raise ValueError("missing P44-P48 v8 freeze: " + str(source_path))
    if output.exists():
        raise ValueError("P49-P53 v9 spec is write-once")

    source = ProspectiveCanonicalValidationFreezeV8.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    spec = build_p49_p53_spec_from_v8_infrastructure(
        infrastructure_freeze=source,
        campaign_root=args.campaign_root,
    )
    write_json_exclusive(output, spec)

    print("P49-P53 canonical validation v9 spec materialized")
    print("Prior scientific outcomes consumed: false")
    print("Prior v8 execution outputs consumed: false")
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
