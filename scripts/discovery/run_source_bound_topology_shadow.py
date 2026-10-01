from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.source_bound_topology_shadow import (
    build_sentinel_audit,
    build_topology_report,
    discover_source_binding_bundle,
    load_json,
)


def _write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--abl", required=True, type=Path)
    p.add_argument(
        "--query-plan",
        default="B.claims_queries.json",
    )
    p.add_argument(
        "--saturation-report",
        default="B.lower_order_saturation_v13_v12prompt.report.json",
    )
    p.add_argument(
        "--saturation-reviews",
        default="B.lower_order_saturation_v13_v12prompt.claim_reviews.json",
    )
    p.add_argument(
        "--prior-art-packet",
        default="B.lower_order_saturation_v13.prior_art.json",
    )
    p.add_argument(
        "--external-report",
        default="B.report.json",
    )
    p.add_argument(
        "--output",
        default="B.source_bound_topology_residual_v2.report.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    abl = args.abl.expanduser().resolve()

    query_plan = load_json(abl / args.query_plan)
    saturation_report = load_json(abl / args.saturation_report)
    saturation_reviews = load_json(abl / args.saturation_reviews)
    prior_art_packet = load_json(abl / args.prior_art_packet)
    external_report = load_json(abl / args.external_report)

    bundle_path, bundle = discover_source_binding_bundle(
        abl,
        source_query_plan_id=str(query_plan.get("plan_id") or ""),
    )

    topology = build_topology_report(
        query_plan=query_plan,
        saturation_report=saturation_report,
        external_report=external_report,
        source_binding_bundle=bundle,
    )
    sentinel = build_sentinel_audit(
        prior_art_packet=prior_art_packet,
        saturation_claim_reviews=saturation_reviews,
    )

    payload = {
        "schema_version": "lower-order-topology-milestone-v2",
        "shadow_only": True,
        "source_binding_bundle_path": (
            str(bundle_path) if bundle_path is not None else None
        ),
        "sentinel_audit": sentinel,
        "topology_residual": topology,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }

    output = abl / args.output
    _write(output, payload)

    print("===== SOURCE-BOUND TOPOLOGY / RESIDUAL MILESTONE V2 =====")
    print("Source binding bundle:", bundle_path or "NOT FOUND")
    print(
        "Generic semiconductor-bilayer false positives:",
        sentinel["generic_semiconductor_bilayer_relation_backed_count"],
    )
    for name, row in sentinel["known_case_presence"].items():
        print(
            "SENTINEL",
            name,
            "| retrieved=", row["retrieved"],
            "| backed_any=", row["relation_backed_any_claim"],
            "|", row.get("title"),
        )

    topo = topology
    print("Topology states:", topo["topology_state_counts"])
    print("Closure states:", topo["component_closure_counts"])
    print("Residual states:", topo["residual_state_counts"])
    print()

    for row in topo["composites"]:
        print(
            "COMPOSITE",
            row["claim_id"],
            "| hypothesis=", row["hypothesis_id"],
            "| full=", row["full_relation_status"],
            "| topology=", row["topology_state"],
            "| components=",
            f'{row["relation_backed_component_count"]}/{row["component_count"]}',
            "| closure=", row["component_closure"],
            "| residual=", row["residual_state"],
        )
        if row["source_bound_recovered_component_claim_ids"]:
            print(
                "  recovered:",
                ", ".join(row["source_bound_recovered_component_claim_ids"]),
            )

    print()
    print("Report:", output)
    print("SHADOW_ONLY=True")
    print("NOVELTY_AUTHORITY_CREATED=False")
    print("N9_AUTHORITY_CREATED=False")
    print("N10_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
