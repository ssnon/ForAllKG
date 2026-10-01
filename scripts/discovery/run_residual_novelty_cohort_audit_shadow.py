
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation", required=True, type=Path)
    parser.add_argument("--topology-completion", required=True, type=Path)
    parser.add_argument("--source-binding", required=True, type=Path)
    parser.add_argument("--topology-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    aggregation = load(args.aggregation)
    completion = load(args.topology_completion)
    bundle = load(args.source_binding)
    topology = load(args.topology_report)

    failures: list[str] = []
    warnings: list[str] = []

    for row in aggregation.get("composites", []):
        if row.get("aggregation_disposition") == "RESIDUAL_CANDIDATE_SHADOW":
            if row.get("full_relation_status") in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
                failures.append(
                    "residual_candidate_has_full_relation_prior_art:"
                    + str(row.get("claim_id"))
                )
            if row.get("topology_state") != "EXPLICIT":
                failures.append(
                    "residual_candidate_without_explicit_topology:"
                    + str(row.get("claim_id"))
                )
            if len(row.get("aggregated_relation_backed_component_claim_ids", [])) != len(
                row.get("aggregated_component_claim_ids", [])
            ):
                failures.append(
                    "residual_candidate_without_full_component_saturation:"
                    + str(row.get("claim_id"))
                )

        if (
            row.get("topology_state") == "NO_COMPONENT_TOPOLOGY"
            and row.get("full_relation_status") not in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}
            and row.get("aggregation_disposition") != "HOLD_FOR_TOPOLOGY"
        ):
            failures.append(
                "no_topology_not_held:" + str(row.get("claim_id"))
            )

    completion_records = completion.get("records", [])
    if not completion_records:
        warnings.append("topology_completion_records_empty")
    if int(bundle.get("claim_count") or 0) <= 0:
        failures.append("source_binding_bundle_empty")

    topology_payload = topology.get("topology_residual", topology)
    if not topology_payload.get("composites"):
        failures.append("topology_composite_population_empty")

    report = {
        "schema_version": "residual-novelty-cohort-audit-shadow-v1",
        "shadow_only": True,
        "pass": not failures,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "source_binding_claim_count": bundle.get("claim_count"),
        "topology_completion_record_count": len(completion_records),
        "composite_count": len(aggregation.get("composites", [])),
        "residual_candidate_count": sum(
            row.get("aggregation_disposition") == "RESIDUAL_CANDIDATE_SHADOW"
            for row in aggregation.get("composites", [])
        ),
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== RESIDUAL NOVELTY COHORT AUDIT SHADOW =====")
    print("PASS:", report["pass"])
    print("Failures:", report["failure_count"])
    print("Warnings:", report["warning_count"])
    print("Source-binding claims:", report["source_binding_claim_count"])
    print("Topology records:", report["topology_completion_record_count"])
    print("Composites:", report["composite_count"])
    print("Residual candidates:", report["residual_candidate_count"])
    for item in failures:
        print("FAIL:", item)
    for item in warnings:
        print("WARN:", item)
    print("Report:", args.output)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
