from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.direct_higher_order_shadow_bundle import (
    DirectHigherOrderShadowBundle,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DirectHigherOrderAlpha6TriggerRecommendation(StrictModel):
    schema_version: Literal[
        "direct-higher-order-alpha6-trigger-recommendation-v1"
    ] = "direct-higher-order-alpha6-trigger-recommendation-v1"

    arm_index: int = Field(ge=1)
    direct_topology_id: str
    modifier_text: str

    external_statuses: list[str] = Field(default_factory=list)
    first_gap_level: str | None = None

    recommendation: Literal[
        "KEEP",
        "REAXIS_CANDIDATE",
        "HOLD_UNRESOLVED",
        "NO_RECOMMENDATION",
    ]

    reason_codes: list[str] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    alpha6_trigger_authority: Literal[False] = False
    candidate_survival_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


def recommend_direct_higher_order_alpha6_triggers(
    bundle: DirectHigherOrderShadowBundle,
) -> tuple[
    DirectHigherOrderAlpha6TriggerRecommendation,
    ...,
]:
    rows = []

    for arm in bundle.arms:
        external_statuses = list(
            arm.external_novelty.card_statuses
        )
        first_gap = (
            arm.conceptual_knownness.first_gap_level
            if arm.conceptual_knownness.status == "AVAILABLE"
            else None
        )

        relational_gap = any(
            status
            in {
                "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                "SEARCH_BOUNDED_RELATIONAL_GAP",
                "HIGHER_ORDER_RELATIONAL_GAP",
            }
            for status in external_statuses
        )

        if arm.external_novelty.status != "COMPLETED":
            recommendation = "HOLD_UNRESOLVED"
            reasons = [
                "external_novelty_not_completed",
            ]

        elif first_gap is None:
            recommendation = "HOLD_UNRESOLVED"
            reasons = [
                "conceptual_knownness_unavailable",
            ]

        elif relational_gap and first_gap == "L3_EXACT":
            recommendation = "REAXIS_CANDIDATE"
            reasons = [
                "structural_relational_gap",
                "conceptual_gap_begins_only_at_exact_level",
                "possible_shallow_local_extension",
            ]

        elif relational_gap and first_gap in {
            "L1_BROAD",
            "L2_INTERMEDIATE",
        }:
            recommendation = "KEEP"
            reasons = [
                "structural_relational_gap",
                "conceptual_gap_survives_abstraction",
                "deeper_relational_novelty_candidate",
            ]

        elif first_gap == "L3_EXACT":
            recommendation = "REAXIS_CANDIDATE"
            reasons = [
                "conceptual_gap_begins_only_at_exact_level",
                "possible_shallow_local_extension",
            ]

        else:
            recommendation = "NO_RECOMMENDATION"
            reasons = [
                "no_supported_trigger_pattern",
            ]

        rows.append(
            DirectHigherOrderAlpha6TriggerRecommendation(
                arm_index=arm.arm_index,
                direct_topology_id=arm.direct_topology_id,
                modifier_text=arm.modifier_text,
                external_statuses=external_statuses,
                first_gap_level=first_gap,
                recommendation=recommendation,
                reason_codes=reasons,
            )
        )

    return tuple(rows)


__all__ = [
    "DirectHigherOrderAlpha6TriggerRecommendation",
    "recommend_direct_higher_order_alpha6_triggers",
]
