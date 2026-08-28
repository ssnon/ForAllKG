from __future__ import annotations

from typing import Any


_GATE_SCHEMA = "scientific-novelty-fallback-gate-v1"
_SOURCE_SCHEMA = "scientific-novelty-action-shadow-batch-v1"


def build_scientific_novelty_fallback_gate(
    action_batch: dict[str, Any],
) -> dict[str, Any]:
    """Promote frozen N1 decisions to Alpha6 original-fallback authority."""

    if action_batch.get("schema_version") != _SOURCE_SCHEMA:
        raise ValueError(
            "unexpected scientific novelty action batch schema"
        )

    decisions = action_batch.get("decisions")

    if not isinstance(decisions, list):
        raise ValueError(
            "scientific novelty action batch decisions must be a list"
        )

    gates = []
    seen = set()

    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError(
                "scientific novelty action decision row must be an object"
            )

        hypothesis_id = str(
            row.get("hypothesis_id") or ""
        ).strip()

        decision = row.get("decision")

        if not hypothesis_id:
            raise ValueError(
                "scientific novelty action row is missing hypothesis_id"
            )

        if hypothesis_id in seen:
            raise ValueError(
                f"duplicate scientific novelty hypothesis_id: {hypothesis_id}"
            )

        if not isinstance(decision, dict):
            raise ValueError(
                f"scientific novelty action decision missing for {hypothesis_id}"
            )

        selection_class = str(
            decision.get("selection_class") or ""
        )

        action = str(
            decision.get("action") or ""
        )

        if selection_class not in {
            "ELIGIBLE",
            "CONDITIONAL",
            "INELIGIBLE",
        }:
            raise ValueError(
                f"invalid selection_class for {hypothesis_id}: "
                f"{selection_class}"
            )

        fallback_allowed = (
            selection_class
            in {
                "ELIGIBLE",
                "CONDITIONAL",
            }
        )

        gates.append(
            {
                "hypothesis_id": hypothesis_id,
                "fallback_allowed": fallback_allowed,
                "selection_class": selection_class,
                "action": action,
                "reason_codes": list(
                    decision.get("reason_codes") or []
                ),
                "semantic_stable": decision.get(
                    "semantic_stable"
                ),
                "stable_semantic_tier": decision.get(
                    "stable_semantic_tier"
                ),
                "external_status": decision.get(
                    "external_status"
                ),
            }
        )

        seen.add(hypothesis_id)

    return {
        "schema_version": _GATE_SCHEMA,
        "source_action_batch_schema": _SOURCE_SCHEMA,
        "source_external_report_id": action_batch.get(
            "source_external_report_id"
        ),
        "gate_count": len(gates),
        "gates": gates,
        "production_authority": True,
        "authority_scope": "alpha6_original_fallback_only",
        "action_policy_applied": True,
    }
