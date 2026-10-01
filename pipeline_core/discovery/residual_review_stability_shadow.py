
from __future__ import annotations

from typing import Any


BACKED = {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}


def _target_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("claim_id")): row
        for row in report.get("targets", [])
        if row.get("claim_id")
    }


def _work_relations(
    target: dict[str, Any],
) -> dict[str, str]:
    return {
        str(row.get("work_id")): str(
            row.get("relationship") or ""
        )
        for row in target.get("reviews", [])
        if row.get("work_id")
    }


def _composite_map(
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("claim_id")): row
        for row in report.get("composites", [])
        if row.get("claim_id")
    }


def compare_residual_runs(
    *,
    fulltext_a: dict[str, Any],
    fulltext_b: dict[str, Any],
    aggregation_a: dict[str, Any],
    aggregation_b: dict[str, Any],
) -> dict[str, Any]:
    target_a = _target_map(fulltext_a)
    target_b = _target_map(fulltext_b)

    target_rows: list[dict[str, Any]] = []
    authority_relevant_relation_instability = 0

    for claim_id in sorted(set(target_a) | set(target_b)):
        a = target_a.get(claim_id, {})
        b = target_b.get(claim_id, {})
        ra = _work_relations(a)
        rb = _work_relations(b)

        work_rows = []
        for work_id in sorted(set(ra) | set(rb)):
            relation_a = ra.get(work_id)
            relation_b = rb.get(work_id)
            backed_a = relation_a in BACKED
            backed_b = relation_b in BACKED

            authority_relevant_stable = (
                backed_a == backed_b
                and (
                    not backed_a
                    or relation_a == relation_b
                )
            )
            if not authority_relevant_stable:
                authority_relevant_relation_instability += 1

            work_rows.append(
                {
                    "work_id": work_id,
                    "relationship_a": relation_a,
                    "relationship_b": relation_b,
                    "backed_a": backed_a,
                    "backed_b": backed_b,
                    "authority_relevant_stable": (
                        authority_relevant_stable
                    ),
                }
            )

        target_rows.append(
            {
                "claim_id": claim_id,
                "status_a": a.get("status"),
                "status_b": b.get("status"),
                "backed_work_ids_a": sorted(
                    a.get(
                        "fulltext_relation_backed_work_ids",
                        [],
                    )
                ),
                "backed_work_ids_b": sorted(
                    b.get(
                        "fulltext_relation_backed_work_ids",
                        [],
                    )
                ),
                "backed_set_stable": set(
                    a.get(
                        "fulltext_relation_backed_work_ids",
                        [],
                    )
                )
                == set(
                    b.get(
                        "fulltext_relation_backed_work_ids",
                        [],
                    )
                ),
                "work_reviews": work_rows,
            }
        )

    comp_a = _composite_map(aggregation_a)
    comp_b = _composite_map(aggregation_b)

    composite_rows: list[dict[str, Any]] = []
    disposition_instability = 0

    for claim_id in sorted(set(comp_a) | set(comp_b)):
        a = comp_a.get(claim_id, {})
        b = comp_b.get(claim_id, {})
        stable = (
            a.get("aggregation_disposition")
            == b.get("aggregation_disposition")
            and a.get("aggregated_residual_state")
            == b.get("aggregated_residual_state")
            and set(
                a.get(
                    "aggregated_relation_backed_component_claim_ids",
                    [],
                )
            )
            == set(
                b.get(
                    "aggregated_relation_backed_component_claim_ids",
                    [],
                )
            )
        )
        if not stable:
            disposition_instability += 1

        composite_rows.append(
            {
                "claim_id": claim_id,
                "hypothesis_id": (
                    a.get("hypothesis_id")
                    or b.get("hypothesis_id")
                ),
                "disposition_a": a.get(
                    "aggregation_disposition"
                ),
                "disposition_b": b.get(
                    "aggregation_disposition"
                ),
                "residual_state_a": a.get(
                    "aggregated_residual_state"
                ),
                "residual_state_b": b.get(
                    "aggregated_residual_state"
                ),
                "stable": stable,
            }
        )

    authority_relevant_stable = (
        authority_relevant_relation_instability == 0
        and disposition_instability == 0
    )

    return {
        "schema_version": "residual-review-stability-shadow-v1",
        "shadow_only": True,
        "authority_relevant_stable": (
            authority_relevant_stable
        ),
        "authority_relevant_relation_instability_count": (
            authority_relevant_relation_instability
        ),
        "aggregation_disposition_instability_count": (
            disposition_instability
        ),
        "targets": target_rows,
        "composites": composite_rows,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
