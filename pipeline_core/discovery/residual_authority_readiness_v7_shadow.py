
from __future__ import annotations

from typing import Any

from pipeline_core.discovery.residual_authority_readiness_shadow import (
    build_authority_readiness,
)


def build_authority_readiness_v7(
    *,
    aggregation_a: dict[str, Any],
    aggregation_b: dict[str, Any],
    stability: dict[str, Any],
    cohort_audit_a: dict[str, Any],
    cohort_audit_b: dict[str, Any],
    candidate_memory: dict[str, Any],
    selection_audit: dict[str, Any],
) -> dict[str, Any]:
    base = build_authority_readiness(
        aggregation_a=aggregation_a,
        aggregation_b=aggregation_b,
        stability=stability,
        cohort_audit_a=cohort_audit_a,
        cohort_audit_b=cohort_audit_b,
        candidate_memory=candidate_memory,
    )

    selection_pass = bool(selection_audit.get("pass"))
    rows = []

    for row in base.get("rows", []):
        updated = dict(row)
        base_ready = bool(
            row.get("authority_ready_candidate_shadow")
        )

        if base_ready and selection_pass:
            state = "AUTHORITY_READY_CANDIDATE_SHADOW"
            ready = True
        elif base_ready and not selection_pass:
            state = "HOLD_SELECTION_COMPLETENESS"
            ready = False
        else:
            state = str(row.get("state") or "")
            ready = False

        updated["base_authority_ready_candidate_shadow"] = (
            base_ready
        )
        updated["candidate_selection_completeness_pass"] = (
            selection_pass
        )
        updated["state"] = state
        updated["authority_ready_candidate_shadow"] = ready
        rows.append(updated)

    return {
        "schema_version": (
            "residual-authority-readiness-shadow-v2"
        ),
        "shadow_only": True,
        "candidate_memory_is_evidence": False,
        "candidate_selection_completeness_pass": selection_pass,
        "candidate_selection_sentinel_relationship": (
            selection_audit.get("sentinel_relationship")
        ),
        "cohort_audits_pass": base.get(
            "cohort_audits_pass",
            False,
        ),
        "review_stability_pass": base.get(
            "review_stability_pass",
            False,
        ),
        "base_authority_ready_candidate_count": sum(
            bool(row.get(
                "base_authority_ready_candidate_shadow"
            ))
            for row in rows
        ),
        "authority_ready_candidate_count": sum(
            bool(row.get(
                "authority_ready_candidate_shadow"
            ))
            for row in rows
        ),
        "rows": rows,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
