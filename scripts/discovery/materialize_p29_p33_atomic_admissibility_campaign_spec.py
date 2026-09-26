from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    build_p29_p33_spec_from_v4_infrastructure,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    ProspectiveDecompositionProvenanceFreezeV4,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize predeclared P29-P33 atomic-admissibility source tasks. "
            "P26-P28 v4 contributes infrastructure/model policy and relation-"
            "family names for nonoverlap only; no v4 outcome artifacts are read."
        )
    )
    parser.add_argument(
        "--infrastructure-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.infrastructure_freeze.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(
            "missing P26-P28 v4 infrastructure freeze: " + str(source_path)
        )
    if output_path.exists():
        raise ValueError(
            "P29-P33 atomic admissibility campaign spec is write-once"
        )

    source = ProspectiveDecompositionProvenanceFreezeV4.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=args.campaign_root,
    )
    write_json_exclusive(output_path, spec)

    print("P29-P33 atomic admissibility v5 spec materialized")
    print("Campaign:", spec.campaign_name)
    print("Root:", spec.campaign_root)
    print("Infrastructure source:", spec.infrastructure_source_freeze_id)
    print("Prior v4 scientific outcomes consumed: false")
    print("Prior v4 task definitions used only for nonoverlap guard: true")
    print("Expected P29-P33 outcomes predeclared: false")
    print()
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
    print()
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
