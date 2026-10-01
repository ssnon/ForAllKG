
from __future__ import annotations

from typing import Any


BACKED = {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}


def _doi_key(value: object) -> str:
    text = str(value or "").strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "doi:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.rstrip(" .")


def audit_candidate_selection(
    *,
    selection_plan: dict[str, Any],
    packet: dict[str, Any],
    sentinel: dict[str, Any],
) -> dict[str, Any]:
    works = {
        str(work.get("work_id")): work
        for work in packet.get("works", [])
        if work.get("work_id")
    }

    failures: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for target in selection_plan.get("targets", []):
        selected = target.get("selected", [])
        work_ids = [
            str(row.get("work_id") or "")
            for row in selected
            if row.get("work_id")
        ]
        if len(work_ids) != len(set(work_ids)):
            failures.append(
                "duplicate_selected_work:"
                + str(target.get("claim_id"))
            )
        if len(work_ids) > int(target.get("total_cap") or 0):
            failures.append(
                "selection_exceeds_cap:"
                + str(target.get("claim_id"))
            )

        rows.append(
            {
                "claim_id": target.get("claim_id"),
                "selected_count": len(work_ids),
                "unique_selected_count": len(set(work_ids)),
                "total_cap": target.get("total_cap"),
                "tier_available_counts": target.get(
                    "tier_available_counts",
                    {},
                ),
                "tier_selected_counts": target.get(
                    "tier_selected_counts",
                    {},
                ),
            }
        )

    sentinel_claim = str(sentinel.get("claim_id") or "")
    sentinel_doi = _doi_key(sentinel.get("doi"))
    sentinel_relationship = str(
        sentinel.get("relationship") or ""
    )

    selected_sentinel = False
    for target in selection_plan.get("targets", []):
        if str(target.get("claim_id") or "") != sentinel_claim:
            continue
        for selected in target.get("selected", []):
            work = works.get(str(selected.get("work_id") or ""), {})
            if _doi_key(work.get("doi")) == sentinel_doi:
                selected_sentinel = True
                break

    sentinel_decisive = sentinel_relationship in {
        "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART",
        "NOT_RELATION_BACKED",
    }

    if sentinel_relationship in BACKED and not selected_sentinel:
        failures.append(
            "positive_diagnostic_sentinel_missed_by_selector"
        )
    elif sentinel_relationship == "INSUFFICIENT_FULLTEXT_EVIDENCE":
        failures.append(
            "diagnostic_sentinel_inconclusive"
        )
    elif not sentinel_decisive:
        failures.append(
            "diagnostic_sentinel_not_decisive:"
            + str(sentinel.get("status"))
        )

    if sentinel_relationship == "NOT_RELATION_BACKED":
        warnings.append(
            "diagnostic_sentinel_negative_for_exact_current_claim"
        )

    return {
        "schema_version": (
            "candidate-selection-completeness-audit-shadow-v1"
        ),
        "shadow_only": True,
        "pass": not failures,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "sentinel_claim_id": sentinel_claim,
        "sentinel_doi": sentinel.get("doi"),
        "sentinel_relationship": sentinel_relationship,
        "sentinel_selected_by_tiered_plan": selected_sentinel,
        "sentinel_decisive": sentinel_decisive,
        "targets": rows,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
