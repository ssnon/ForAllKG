from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.direct_higher_order_topology import compose_direct_higher_order_topologies
from pipeline_core.discovery.direct_task_relation_backbone import DirectTaskRelationBackboneView
from scripts.discovery.run_higher_order_shadow_lane import (
    _candidate_modifier_components_from_canonical_root,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-backbones", required=True, type=Path)
    parser.add_argument("--modifier-audit", required=True, type=Path)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    backbone_payload = _json(args.direct_backbones)
    modifier_payload = _json(args.modifier_audit)

    backbones = tuple(
        DirectTaskRelationBackboneView.model_validate(row)
        for row in backbone_payload.get("backbones", [])
    )

    candidate_components, candidate_replay = _candidate_modifier_components_from_canonical_root(
        canonical_root=args.canonical_root,
        domain_profile=args.domain_profile,
    )

    eligible_records = list(modifier_payload.get("eligible_modifier_records", []))

    topologies = compose_direct_higher_order_topologies(
        backbones=backbones,
        modifier_components=candidate_components,
        eligible_modifier_records=eligible_records,
        audit_source=str(args.modifier_audit),
    )

    out = {
        "schema_version": "direct-higher-order-topology-shadow-report-v1",
        "direct_backbone_source": str(args.direct_backbones),
        "modifier_audit_source": str(args.modifier_audit),
        "canonical_root": str(args.canonical_root),
        "direct_backbone_count": len(backbones),
        "candidate_modifier_component_count": len(candidate_components),
        "eligible_modifier_record_count": len(eligible_records),
        "topology_count": len(topologies),
        "candidate_replay": candidate_replay,
        "topologies": [x.model_dump(mode="json") for x in topologies],
        "authority": {
            "shadow_only": True,
            "topology_is_interaction_evidence": False,
            "interaction_claim_authorized": False,
            "endpoint_equivalence_authorized": False,
            "scientific_identity_asserted": False,
            "novelty_authority_created": False,
            "positive_premise_authority_created": False,
            "production_selection_authority": False,
            "existing_task_backbone_types_changed": False,
            "existing_higher_order_consumers_changed": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== S202 DIRECT HIGHER-ORDER TOPOLOGY SHADOW ===")
    print("direct backbones:", len(backbones))
    print("eligible modifier records:", len(eligible_records))
    print("materialized topologies:", len(topologies))

    for row in topologies:
        b = row.backbone.component
        m = row.modifier_component
        rb = row.role_binding
        print(f"\nbackbone: {b.subject} --{b.relation}--> {b.object}")
        print(" anchor role:", rb.anchor_role)
        print(" backbone role text:", rb.backbone_role_text)
        print(f" modifier relation: {m.subject} --{m.relation}--> {m.object}")
        print(" modifier C:", rb.modifier_text)
        print(" role match:", rb.match_mode, rb.compatibility_score)

    print(
        "\nStructural opportunities only; "
        "no interaction evidence or production authority created."
    )
    print("artifact:", args.output)
    print("S202_SHADOW_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
