from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_authority_campaign_freeze_v3 import (
    ProspectiveAuthorityCampaignFreezeV3,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    build_p26_p28_spec_from_v3_infrastructure,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the P26-P28 v4 engineering diagnostic source-task "
            "spec before any v4 execution. P21-P25 contributes infrastructure, "
            "model policy, and relation-family nonoverlap only; no v3 outcome "
            "artifact is consumed by this builder."
        )
    )
    parser.add_argument("--infrastructure-freeze", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.infrastructure_freeze.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(
            "missing P21-P25 v3 infrastructure freeze: " + str(source_path)
        )
    if output_path.exists():
        raise ValueError("P26-P28 v4 diagnostic spec is write-once")

    source = ProspectiveAuthorityCampaignFreezeV3.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    spec = build_p26_p28_spec_from_v3_infrastructure(
        infrastructure_freeze=source,
        campaign_root=args.campaign_root,
    )
    write_json_exclusive(output_path, spec)

    print("P26-P28 decomposition provenance diagnostic v4 spec materialized")
    print("Campaign:", spec.campaign_name)
    print("Root:", spec.campaign_root)
    print("Infrastructure source:", spec.infrastructure_source_freeze_id)
    print("Diagnostic goal:", spec.diagnostic_goal)
    print("Scientific validation authority: false")
    print("Prior v3 outcome artifacts consumed by builder: false")
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
