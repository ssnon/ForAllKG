from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.higher_order_hypothesis_batch import (
    HigherOrderShadowBatchOutcome,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class HigherOrderExternalShadowArmPlan(StrictModel):
    schema_version: Literal[
        "higher-order-external-shadow-arm-plan-v1"
    ] = "higher-order-external-shadow-arm-plan-v1"

    arm_index: int = Field(ge=1)

    higher_order_context_id: str
    higher_order_topology_id: str
    modifier_component_id: str
    modifier_text: str

    materialization_id: str
    authorization_id: str

    hypothesis_portfolio_id: str
    hypothesis_id: str
    source_hypothesis_context_id: str
    source_hypothesis_context_sha256: str

    external_novelty_required: Literal[True] = True
    pre_review_coverage_shadow_required: Literal[True] = True
    downstream_gate_shadow_required: Literal[True] = True

    query_plan_mode: Literal[
        "fresh_required"
    ] = "fresh_required"
    prior_art_retrieval_mode: Literal[
        "fresh_required"
    ] = "fresh_required"

    reuse_query_plan_authorized: Literal[False] = False
    reuse_prior_art_authorized: Literal[False] = False

    shadow_only: Literal[True] = True
    novelty_authority: Literal[False] = False
    selection_class_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_arm(
        self,
    ) -> "HigherOrderExternalShadowArmPlan":
        required = (
            self.higher_order_context_id,
            self.higher_order_topology_id,
            self.modifier_component_id,
            self.modifier_text,
            self.materialization_id,
            self.authorization_id,
            self.hypothesis_portfolio_id,
            self.hypothesis_id,
            self.source_hypothesis_context_id,
            self.source_hypothesis_context_sha256,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order external shadow arm requires complete lineage"
            )

        return self


class HigherOrderExternalShadowBatchPlan(StrictModel):
    schema_version: Literal[
        "higher-order-external-shadow-batch-plan-v1"
    ] = "higher-order-external-shadow-batch-plan-v1"

    plan_id: str
    source_generation_batch_run_id: str

    source_selected_context_count: int = Field(ge=1)
    source_proposed_count: int = Field(ge=0)
    planned_external_arm_count: int = Field(ge=1)
    skipped_nonproposed_count: int = Field(ge=0)

    arms: list[
        HigherOrderExternalShadowArmPlan
    ] = Field(min_length=1)

    execution_policy: Literal[
        "independent_fresh_external_shadow_per_proposed_arm"
    ] = "independent_fresh_external_shadow_per_proposed_arm"

    fresh_query_plans_required: Literal[True] = True
    fresh_prior_art_retrieval_required: Literal[True] = True

    shared_query_plan_reuse_authorized: Literal[False] = False
    shared_prior_art_reuse_authorized: Literal[False] = False

    external_novelty_shadow_only: Literal[True] = True
    pre_review_coverage_shadow_required: Literal[True] = True
    downstream_gate_shadow_required: Literal[True] = True

    scientific_quality_ranking_performed: Literal[False] = False
    novelty_authority: Literal[False] = False
    selection_class_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    legacy_portfolio_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_batch(
        self,
    ) -> "HigherOrderExternalShadowBatchPlan":
        if not self.plan_id.strip():
            raise ValueError("plan_id is required")
        if not self.source_generation_batch_run_id.strip():
            raise ValueError(
                "source_generation_batch_run_id is required"
            )

        if self.planned_external_arm_count != len(self.arms):
            raise ValueError(
                "planned_external_arm_count must match arms"
            )
        if self.source_proposed_count != len(self.arms):
            raise ValueError(
                "all and only proposed generation arms must be planned"
            )
        if (
            self.skipped_nonproposed_count
            != self.source_selected_context_count
            - self.source_proposed_count
        ):
            raise ValueError(
                "skipped_nonproposed_count mismatch"
            )

        arm_indices = [row.arm_index for row in self.arms]
        if arm_indices != sorted(arm_indices):
            raise ValueError(
                "external shadow arms must preserve generation arm order"
            )
        if len(arm_indices) != len(set(arm_indices)):
            raise ValueError(
                "duplicate external shadow arm index"
            )

        portfolio_ids = [
            row.hypothesis_portfolio_id
            for row in self.arms
        ]
        if len(portfolio_ids) != len(set(portfolio_ids)):
            raise ValueError(
                "duplicate hypothesis portfolio in external shadow plan"
            )

        hypothesis_ids = [
            row.hypothesis_id
            for row in self.arms
        ]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError(
                "duplicate hypothesis ID in external shadow plan"
            )

        return self


def build_higher_order_external_shadow_batch_plan(
    batch: HigherOrderShadowBatchOutcome,
) -> HigherOrderExternalShadowBatchPlan:
    """
    Convert a completed higher-order generation batch into an authority-neutral
    external-review execution plan.

    The plan intentionally forbids historical query-plan/prior-art reuse.
    Production-generated hypotheses have their own portfolio/hypothesis IDs and
    scientific wording, so each proposed arm requires a fresh decomposition,
    fresh query plan, and fresh retrieval when it is actually executed.
    """

    if not batch.arms:
        raise ValueError(
            "higher-order external shadow planning requires a non-empty "
            "generation batch"
        )

    arm_plans: list[
        HigherOrderExternalShadowArmPlan
    ] = []

    for index, arm in enumerate(
        batch.arms,
        start=1,
    ):
        run = arm.run_outcome

        if run.status != "proposed":
            continue

        portfolio = run.portfolio_for_downstream
        if portfolio is None:
            raise ValueError(
                "proposed higher-order arm lacks downstream portfolio"
            )
        if len(portfolio.hypotheses) != 1:
            raise ValueError(
                "proposed higher-order external shadow arm must contain "
                "exactly one hypothesis"
            )

        hypothesis = portfolio.hypotheses[0]

        if (
            portfolio.source_context_id
            != arm.projection.context.context_id
        ):
            raise ValueError(
                "portfolio/source projection context mismatch"
            )
        if (
            portfolio.source_context_sha256
            != arm.projection.context.context_sha256
        ):
            raise ValueError(
                "portfolio/source projection context SHA mismatch"
            )

        arm_plans.append(
            HigherOrderExternalShadowArmPlan(
                arm_index=index,
                higher_order_context_id=(
                    arm.higher_order_context.context_id
                ),
                higher_order_topology_id=(
                    arm.higher_order_context
                    .higher_order_topology_id
                ),
                modifier_component_id=(
                    arm.higher_order_context
                    .lineage.modifier_component_id
                ),
                modifier_text=(
                    arm.higher_order_context
                    .lineage.modifier_text
                ),
                materialization_id=(
                    arm.projection.materialization.materialization_id
                ),
                authorization_id=(
                    arm.authorization.authorization_id
                ),
                hypothesis_portfolio_id=(
                    portfolio.portfolio_id
                ),
                hypothesis_id=(
                    hypothesis.hypothesis_id
                ),
                source_hypothesis_context_id=(
                    portfolio.source_context_id
                ),
                source_hypothesis_context_sha256=(
                    portfolio.source_context_sha256
                ),
            )
        )

    if not arm_plans:
        raise ValueError(
            "higher-order generation batch contains zero proposed arms "
            "eligible for external shadow planning"
        )

    plan_id = _stable_id(
        "higher_order_external_shadow_batch_plan",
        batch.record.batch_run_id,
        *[
            row.hypothesis_portfolio_id
            for row in arm_plans
        ],
        *[
            row.hypothesis_id
            for row in arm_plans
        ],
    )

    return HigherOrderExternalShadowBatchPlan(
        plan_id=plan_id,
        source_generation_batch_run_id=(
            batch.record.batch_run_id
        ),
        source_selected_context_count=(
            batch.record.selected_context_count
        ),
        source_proposed_count=len(arm_plans),
        planned_external_arm_count=len(arm_plans),
        skipped_nonproposed_count=(
            batch.record.selected_context_count
            - len(arm_plans)
        ),
        arms=arm_plans,
    )
