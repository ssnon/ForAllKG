from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_audit_report(
    *,
    aggregation: dict[str, Any],
    completion: dict[str, Any],
    bundle: dict[str, Any],
    topology: dict[str, Any],
    query_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    for row in aggregation.get("composites", []):
        if row.get("aggregation_disposition") == "RESIDUAL_CANDIDATE_SHADOW":
            if row.get("full_relation_status") in {
                "DIRECT_PRIOR_ART",
                "PARTIAL_PRIOR_ART",
            }:
                failures.append(
                    "residual_candidate_has_full_relation_prior_art:"
                    + str(row.get("claim_id"))
                )

            if row.get("topology_state") != "EXPLICIT":
                failures.append(
                    "residual_candidate_without_explicit_topology:"
                    + str(row.get("claim_id"))
                )

            if len(
                row.get(
                    "aggregated_relation_backed_component_claim_ids",
                    [],
                )
            ) != len(
                row.get(
                    "aggregated_component_claim_ids",
                    [],
                )
            ):
                failures.append(
                    "residual_candidate_without_full_component_saturation:"
                    + str(row.get("claim_id"))
                )

        if (
            row.get("topology_state") == "NO_COMPONENT_TOPOLOGY"
            and row.get("full_relation_status")
            not in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}
            and row.get("aggregation_disposition") != "HOLD_FOR_TOPOLOGY"
        ):
            failures.append(
                "no_topology_not_held:"
                + str(row.get("claim_id"))
            )

    completion_records = completion.get("records", [])

    if int(bundle.get("claim_count") or 0) <= 0:
        failures.append("source_binding_bundle_empty")

    topology_payload = topology.get(
        "topology_residual",
        topology,
    )
    topology_composites = list(
        topology_payload.get("composites", [])
    )

    observed_composite_claim_ids = sorted(
        {
            str(row.get("claim_id") or "")
            for row in topology_composites
            if str(row.get("claim_id") or "")
        }
    )

    expected_composite_claim_ids: list[str] | None

    if isinstance(query_plan, dict):
        expected_composite_claim_ids = sorted(
            {
                str(claim.get("claim_id") or "")
                for group in query_plan.get("claims", [])
                for claim in group.get("claims", [])
                if (
                    str(claim.get("kind") or "") == "composite"
                    and str(claim.get("claim_id") or "")
                )
            }
        )
    else:
        expected_composite_claim_ids = None

    if expected_composite_claim_ids is None:
        topology_applicability = "UNKNOWN_NO_QUERY_PLAN"

        # Without the query plan we cannot prove that an empty topology
        # population is legitimately non-applicable. Preserve the old
        # conservative fail-closed behavior.
        if not topology_composites:
            failures.append(
                "topology_applicability_unknown_no_query_plan"
            )

        if not completion_records:
            warnings.append(
                "topology_completion_records_empty"
            )

    elif expected_composite_claim_ids:
        topology_applicability = "APPLICABLE"

        if not completion_records:
            warnings.append(
                "topology_completion_records_empty"
            )

        if not topology_composites:
            failures.append(
                "topology_composite_population_empty"
            )

        missing = sorted(
            set(expected_composite_claim_ids)
            - set(observed_composite_claim_ids)
        )
        unexpected = sorted(
            set(observed_composite_claim_ids)
            - set(expected_composite_claim_ids)
        )

        if missing:
            failures.append(
                "topology_missing_expected_composites:"
                + ",".join(missing)
            )

        if unexpected:
            failures.append(
                "topology_has_unplanned_composites:"
                + ",".join(unexpected)
            )

    else:
        topology_applicability = (
            "NOT_APPLICABLE_NO_PLANNED_COMPOSITES"
        )

        # Empty topology is the correct result when decomposition emitted
        # no kind=composite claim.
        warnings.append(
            "topology_not_applicable_no_composite_claims"
        )

        if topology_composites:
            failures.append(
                "topology_has_composites_without_planned_composite_claims"
            )

    report = {
        "schema_version": "residual-novelty-cohort-audit-shadow-v1",
        "shadow_only": True,
        "pass": not failures,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "source_binding_claim_count": bundle.get("claim_count"),
        "topology_completion_record_count": len(
            completion_records
        ),
        "expected_composite_claim_count": (
            None
            if expected_composite_claim_ids is None
            else len(expected_composite_claim_ids)
        ),
        "expected_composite_claim_ids": (
            []
            if expected_composite_claim_ids is None
            else expected_composite_claim_ids
        ),
        "observed_topology_composite_claim_ids": (
            observed_composite_claim_ids
        ),
        "topology_applicability": topology_applicability,
        "topology_applicable": (
            None
            if expected_composite_claim_ids is None
            else bool(expected_composite_claim_ids)
        ),
        "topology_composite_count": len(
            topology_composites
        ),
        "composite_count": len(
            aggregation.get("composites", [])
        ),
        "residual_candidate_count": sum(
            row.get("aggregation_disposition")
            == "RESIDUAL_CANDIDATE_SHADOW"
            for row in aggregation.get("composites", [])
        ),
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregation",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--query-plan",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--topology-completion",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-binding",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--topology-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    report = build_audit_report(
        aggregation=load(args.aggregation),
        completion=load(args.topology_completion),
        bundle=load(args.source_binding),
        topology=load(args.topology_report),
        query_plan=(
            load(args.query_plan)
            if args.query_plan is not None
            else None
        ),
    )

    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "===== RESIDUAL NOVELTY COHORT AUDIT SHADOW ====="
    )
    print("PASS:", report["pass"])
    print("Failures:", report["failure_count"])
    print("Warnings:", report["warning_count"])
    print(
        "Topology applicability:",
        report["topology_applicability"],
    )
    print(
        "Expected composites:",
        report["expected_composite_claim_count"],
    )
    print(
        "Observed composites:",
        report["topology_composite_count"],
    )
    print(
        "Source-binding claims:",
        report["source_binding_claim_count"],
    )
    print(
        "Residual candidates:",
        report["residual_candidate_count"],
    )

    for item in report["failures"]:
        print("FAIL:", item)

    for item in report["warnings"]:
        print("WARN:", item)

    print("Report:", args.output)

    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
