
from __future__ import annotations

from typing import Any

BACKED = {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}


def _review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        rows = payload.get("claim_reviews")
    return rows if isinstance(rows, list) else []


def _metadata_work_ids(payload: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for review in _review_rows(payload):
        cid = str(review.get("claim_id") or "")
        if not cid:
            continue
        out[cid] = {
            str(match.get("work_id"))
            for match in review.get("matches", [])
            if match.get("relationship") in BACKED and match.get("work_id")
        }
    return out


def aggregate_residual_novelty(
    *,
    topology_report: dict[str, Any],
    claim_reviews: dict[str, Any],
    fulltext_escalation: dict[str, Any],
) -> dict[str, Any]:
    metadata = _metadata_work_ids(claim_reviews)
    fulltext = {
        str(row.get("claim_id")): row
        for row in fulltext_escalation.get("targets", [])
        if row.get("claim_id")
    }
    topology = topology_report.get("topology_residual", topology_report)
    output: list[dict[str, Any]] = []

    for row in topology.get("composites", []):
        components = [
            str(x)
            for x in row.get("effective_component_claim_ids_shadow", [])
            if str(x)
        ]
        backed_components: list[str] = []
        fulltext_components: list[str] = []
        unavailable_components: list[str] = []
        work_sets: list[set[str]] = []

        for cid in components:
            works = set(metadata.get(cid, set()))
            ft = fulltext.get(cid)
            if ft:
                ft_ids = {
                    str(x)
                    for x in ft.get("fulltext_relation_backed_work_ids", [])
                    if str(x)
                }
                if ft_ids:
                    fulltext_components.append(cid)
                works |= ft_ids
                if not works and ft.get("status") == "FULLTEXT_UNAVAILABLE":
                    unavailable_components.append(cid)
            if works:
                backed_components.append(cid)
                work_sets.append(works)

        all_backed = bool(components) and len(backed_components) == len(components)
        if all_backed:
            shared = set.intersection(*work_sets) if work_sets else set()
            closure = "SAME_WORK" if shared else "DISTRIBUTED"
        elif backed_components:
            shared = set()
            closure = "PARTIAL"
        elif components:
            shared = set()
            closure = "NONE"
        else:
            shared = set()
            closure = "NOT_APPLICABLE"

        full_relation = str(row.get("full_relation_status") or "")
        topology_state = str(row.get("topology_state") or "")

        if full_relation in BACKED:
            residual_state = "NO_RESIDUAL_FULL_RELATION_BACKED"
            disposition = "NO_RESIDUAL"
        elif topology_state == "NO_COMPONENT_TOPOLOGY":
            residual_state = "NOT_ASSESSABLE_NO_COMPONENT_TOPOLOGY"
            disposition = "HOLD_FOR_TOPOLOGY"
        elif all_backed:
            residual_state = (
                "HIGHER_ORDER_RESIDUAL_SAME_WORK_KNOWN_BASE"
                if closure == "SAME_WORK"
                else "HIGHER_ORDER_RESIDUAL_DISTRIBUTED_KNOWN_BASE"
            )
            disposition = "RESIDUAL_CANDIDATE_SHADOW"
        elif backed_components:
            residual_state = "PARTIAL_BASE_SATURATION"
            disposition = "HOLD_FOR_BASE_EVIDENCE"
        else:
            residual_state = "UNSATURATED_BASE"
            disposition = "HOLD_FOR_BASE_EVIDENCE"

        evidence_depth = (
            "FULLTEXT_ESCALATED"
            if fulltext_components
            else ("METADATA_ONLY" if components else "NOT_APPLICABLE")
        )

        output.append(
            {
                **row,
                "aggregated_component_claim_ids": components,
                "aggregated_relation_backed_component_claim_ids": backed_components,
                "fulltext_escalated_component_claim_ids": fulltext_components,
                "fulltext_unavailable_component_claim_ids": unavailable_components,
                "aggregated_component_closure": closure,
                "aggregated_same_work_ids": sorted(shared),
                "evidence_depth": evidence_depth,
                "aggregated_residual_state": residual_state,
                "aggregation_disposition": disposition,
            }
        )

    state_counts: dict[str, int] = {}
    disposition_counts: dict[str, int] = {}
    for row in output:
        state_counts[row["aggregated_residual_state"]] = (
            state_counts.get(row["aggregated_residual_state"], 0) + 1
        )
        disposition_counts[row["aggregation_disposition"]] = (
            disposition_counts.get(row["aggregation_disposition"], 0) + 1
        )

    return {
        "schema_version": "residual-novelty-aggregation-shadow-v1",
        "shadow_only": True,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
        "residual_state_counts": dict(sorted(state_counts.items())),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "composites": output,
    }
