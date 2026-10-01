
from __future__ import annotations

from typing import Any


def _map_by_claim(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("claim_id")): row
        for row in rows
        if row.get("claim_id")
    }


def build_authority_readiness(
    *,
    aggregation_a: dict[str, Any],
    aggregation_b: dict[str, Any],
    stability: dict[str, Any],
    cohort_audit_a: dict[str, Any],
    cohort_audit_b: dict[str, Any],
    candidate_memory: dict[str, Any],
) -> dict[str, Any]:
    a = _map_by_claim(aggregation_a.get("composites", []))
    b = _map_by_claim(aggregation_b.get("composites", []))
    stable = _map_by_claim(stability.get("composites", []))

    rows: list[dict[str, Any]] = []

    cohort_ok = bool(
        cohort_audit_a.get("pass")
        and cohort_audit_b.get("pass")
    )

    for claim_id in sorted(set(a) | set(b)):
        ra = a.get(claim_id, {})
        rb = b.get(claim_id, {})
        st = stable.get(claim_id, {})

        candidate_in_both = (
            ra.get("aggregation_disposition")
            == "RESIDUAL_CANDIDATE_SHADOW"
            and rb.get("aggregation_disposition")
            == "RESIDUAL_CANDIDATE_SHADOW"
        )
        explicit_topology = (
            ra.get("topology_state") == "EXPLICIT"
            and rb.get("topology_state") == "EXPLICIT"
        )
        all_components_backed = (
            bool(
                ra.get(
                    "aggregated_component_claim_ids",
                    [],
                )
            )
            and len(
                ra.get(
                    "aggregated_relation_backed_component_claim_ids",
                    [],
                )
            )
            == len(
                ra.get(
                    "aggregated_component_claim_ids",
                    [],
                )
            )
            and len(
                rb.get(
                    "aggregated_relation_backed_component_claim_ids",
                    [],
                )
            )
            == len(
                rb.get(
                    "aggregated_component_claim_ids",
                    [],
                )
            )
        )
        unresolved_fulltext = bool(
            ra.get(
                "fulltext_unavailable_component_claim_ids",
                [],
            )
            or rb.get(
                "fulltext_unavailable_component_claim_ids",
                [],
            )
        )
        stable_disposition = bool(st.get("stable"))

        ready = all(
            [
                cohort_ok,
                candidate_in_both,
                explicit_topology,
                all_components_backed,
                not unresolved_fulltext,
                stable_disposition,
                stability.get(
                    "authority_relevant_stable",
                    False,
                ),
            ]
        )

        if ready:
            state = "AUTHORITY_READY_CANDIDATE_SHADOW"
        elif not candidate_in_both:
            state = "NOT_RESIDUAL_CANDIDATE"
        elif not stable_disposition:
            state = "HOLD_UNSTABLE_REVIEW"
        elif unresolved_fulltext:
            state = "HOLD_UNRESOLVED_FULLTEXT"
        elif not all_components_backed:
            state = "HOLD_INCOMPLETE_BASE"
        elif not explicit_topology:
            state = "HOLD_TOPOLOGY"
        elif not cohort_ok:
            state = "HOLD_COHORT_AUDIT"
        else:
            state = "HOLD_AUTHORITY_RELEVANT_INSTABILITY"

        rows.append(
            {
                "claim_id": claim_id,
                "hypothesis_id": (
                    ra.get("hypothesis_id")
                    or rb.get("hypothesis_id")
                ),
                "state": state,
                "authority_ready_candidate_shadow": ready,
                "candidate_in_both_runs": candidate_in_both,
                "explicit_topology": explicit_topology,
                "all_components_backed": all_components_backed,
                "unresolved_fulltext": unresolved_fulltext,
                "stable_disposition": stable_disposition,
            }
        )

    return {
        "schema_version": "residual-authority-readiness-shadow-v1",
        "shadow_only": True,
        "candidate_memory_is_evidence": False,
        "candidate_memory_reexposed_claim_count": (
            candidate_memory.get(
                "reexposed_current_claim_count",
                0,
            )
        ),
        "cohort_audits_pass": cohort_ok,
        "review_stability_pass": stability.get(
            "authority_relevant_stable",
            False,
        ),
        "authority_ready_candidate_count": sum(
            row["authority_ready_candidate_shadow"]
            for row in rows
        ),
        "rows": rows,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
