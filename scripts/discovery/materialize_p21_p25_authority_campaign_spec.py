from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_authority_campaign_freeze_v3 import (
    build_p21_p25_spec_from_v2_infrastructure,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze_v2 import (
    ProspectiveRoutedCampaignFreezeV2,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the predeclared P21-P25 v3 scientific-authority "
            "source-task spec. P16-P20 contributes infrastructure fields only; "
            "its task definitions are consulted solely to reject relation-family "
            "reuse, never to consume scientific outcomes."
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
            "missing P16-P20 infrastructure freeze: " + str(source_path)
        )
    if output_path.exists():
        raise ValueError("P21-P25 authority campaign spec is write-once")

    source = ProspectiveRoutedCampaignFreezeV2.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    spec = build_p21_p25_spec_from_v2_infrastructure(
        infrastructure_freeze=source,
        campaign_root=args.campaign_root,
    )
    write_json_exclusive(output_path, spec)

    print("P21-P25 authority campaign v3 spec materialized")
    print("Campaign:", spec.campaign_name)
    print("Root:", spec.campaign_root)
    print("Infrastructure source:", spec.infrastructure_source_freeze_id)
    print("Initial cut point:", spec.initial_generation_cutpoint)
    print("Downstream campaign:", spec.downstream_campaign_schema)
    print("Prior scientific outputs used to define tasks: false")
    print("Prior task definitions used only for nonoverlap guard: true")
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
