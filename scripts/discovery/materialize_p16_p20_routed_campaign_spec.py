from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRoutedCampaignFreeze,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze_v2 import (
    build_p16_p20_spec_from_infrastructure_freeze,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the predeclared P16-P20 routed source-task spec. "
            "The prior P11-P15 freeze is used only to inherit infrastructure "
            "settings, never scientific outcomes."
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
            "missing infrastructure campaign freeze: " + str(source_path)
        )
    if output_path.exists():
        raise ValueError("P16-P20 routed campaign spec is write-once")

    source = ProspectiveRoutedCampaignFreeze.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=source,
        campaign_root=args.campaign_root,
    )
    write_json_exclusive(output_path, spec)

    print("P16-P20 routed campaign spec materialized")
    print("Campaign:", spec.campaign_name)
    print("Root:", spec.campaign_root)
    print("Infrastructure source:", spec.infrastructure_source_freeze_id)
    print("Prior scientific outcomes used to define tasks: false")
    print()
    for task in spec.tasks:
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
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
