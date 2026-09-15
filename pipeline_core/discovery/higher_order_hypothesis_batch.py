from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.higher_order_hypothesis_context import (
    HigherOrderHypothesisContextProjection,
    materialize_higher_order_hypothesis_context,
)
from pipeline_core.discovery.higher_order_hypothesis_runtime import (
    HigherOrderShadowGenerationAuthorization,
    HigherOrderShadowHypothesisRunOutcome,
    HigherOrderShadowHypothesisRuntime,
    authorize_higher_order_shadow_generation,
)
from pipeline_core.discovery.higher_order_shadow_synthesis import (
    select_shadow_synthesis_contexts,
)
from pipeline_core.discovery.higher_order_synthesis_context import (
    HigherOrderSynthesisContext,
)
from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class HigherOrderShadowBatchArmRecord(StrictModel):
    schema_version: Literal[
        "higher-order-shadow-batch-arm-record-v1"
    ] = "higher-order-shadow-batch-arm-record-v1"

    arm_index: int = Field(ge=1)
    higher_order_context_id: str
    higher_order_topology_id: str
    modifier_component_id: str
    modifier_text: str

    materialization_id: str
    authorization_id: str
    derived_hypothesis_context_id: str
    derived_hypothesis_context_sha256: str

    status: Literal[
        "proposed",
        "abstained",
        "canonical_rejected",
        "shadow_contract_rejected",
    ]
    shadow_contract_passed: bool
    canonical_runtime_accepted: bool

    portfolio_id: str | None = None
    hypothesis_ids: list[str] = Field(default_factory=list)

    canonical_failure_stage: str
    canonical_compile_issue_codes: list[str] = Field(
        default_factory=list
    )
    canonical_validation_issue_codes: list[str] = Field(
        default_factory=list
    )

    shadow_failure_code: str | None = None
    shadow_failure_message: str | None = None

    @model_validator(mode="after")
    def validate_arm(
        self,
    ) -> "HigherOrderShadowBatchArmRecord":
        required = (
            self.higher_order_context_id,
            self.higher_order_topology_id,
            self.modifier_component_id,
            self.modifier_text,
            self.materialization_id,
            self.authorization_id,
            self.derived_hypothesis_context_id,
            self.derived_hypothesis_context_sha256,
            self.canonical_failure_stage,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order shadow batch arm requires complete lineage"
            )

        if self.status == "proposed":
            if not self.shadow_contract_passed:
                raise ValueError(
                    "proposed batch arm must pass shadow contract"
                )
            if not self.canonical_runtime_accepted:
                raise ValueError(
                    "proposed batch arm must be canonically accepted"
                )
            if not self.portfolio_id or len(self.hypothesis_ids) != 1:
                raise ValueError(
                    "proposed batch arm requires exactly one hypothesis"
                )
        elif self.status == "abstained":
            if not self.shadow_contract_passed:
                raise ValueError(
                    "abstained batch arm must pass shadow contract"
                )
            if not self.canonical_runtime_accepted:
                raise ValueError(
                    "abstained batch arm must be canonically accepted"
                )
            if not self.portfolio_id:
                raise ValueError(
                    "abstained batch arm still requires canonical portfolio"
                )
            if self.hypothesis_ids:
                raise ValueError(
                    "abstained batch arm cannot contain hypotheses"
                )
        else:
            if self.shadow_contract_passed:
                raise ValueError(
                    "rejected batch arm cannot pass shadow contract"
                )

        return self


class HigherOrderShadowBatchRunRecord(StrictModel):
    """
    Batch-level audit artifact.

    Selection here means deterministic coverage/dedup only, never scientific
    quality ranking and never production selection.
    """

    schema_version: Literal[
        "higher-order-shadow-batch-run-record-v1"
    ] = "higher-order-shadow-batch-run-record-v1"

    batch_run_id: str

    source_hypothesis_context_id: str
    source_hypothesis_context_sha256: str

    input_context_count: int = Field(ge=1)
    selected_context_count: int = Field(ge=1)
    unique_modifier_count: int = Field(ge=1)
    duplicate_or_over_quota_context_count: int = Field(ge=0)

    max_contexts: int = Field(ge=1)

    selection_policy: Literal[
        "deterministic_modifier_coverage_no_quality_ranking"
    ] = "deterministic_modifier_coverage_no_quality_ranking"

    selected_higher_order_context_ids: list[str] = Field(
        min_length=1
    )
    selected_modifier_component_ids: list[str] = Field(
        min_length=1
    )

    proposed_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    canonical_rejected_count: int = Field(ge=0)
    shadow_contract_rejected_count: int = Field(ge=0)

    arms: list[HigherOrderShadowBatchArmRecord] = Field(
        min_length=1
    )

    shadow_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    novelty_authority: Literal[False] = False
    external_novelty_review_performed: Literal[False] = False
    n10_review_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    legacy_portfolio_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_batch(
        self,
    ) -> "HigherOrderShadowBatchRunRecord":
        required = (
            self.batch_run_id,
            self.source_hypothesis_context_id,
            self.source_hypothesis_context_sha256,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order shadow batch requires complete source lineage"
            )

        if self.selected_context_count != len(self.arms):
            raise ValueError(
                "selected_context_count must match arm count"
            )
        if (
            self.selected_context_count
            != len(self.selected_higher_order_context_ids)
        ):
            raise ValueError(
                "selected context ID count mismatch"
            )
        if (
            self.selected_context_count
            != len(self.selected_modifier_component_ids)
        ):
            raise ValueError(
                "selected modifier ID count mismatch"
            )

        if (
            len(self.selected_higher_order_context_ids)
            != len(set(self.selected_higher_order_context_ids))
        ):
            raise ValueError(
                "duplicate selected higher-order context ID"
            )
        if (
            len(self.selected_modifier_component_ids)
            != len(set(self.selected_modifier_component_ids))
        ):
            raise ValueError(
                "batch selection retained duplicate modifier component"
            )

        expected_counts = {
            "proposed_count": sum(
                row.status == "proposed"
                for row in self.arms
            ),
            "abstained_count": sum(
                row.status == "abstained"
                for row in self.arms
            ),
            "canonical_rejected_count": sum(
                row.status == "canonical_rejected"
                for row in self.arms
            ),
            "shadow_contract_rejected_count": sum(
                row.status == "shadow_contract_rejected"
                for row in self.arms
            ),
        }
        for name, expected in expected_counts.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} does not match arm statuses"
                )

        if (
            self.duplicate_or_over_quota_context_count
            != self.input_context_count - self.selected_context_count
        ):
            raise ValueError(
                "batch dropped-context count mismatch"
            )

        return self


@dataclass(frozen=True)
class HigherOrderShadowBatchArmOutcome:
    higher_order_context: HigherOrderSynthesisContext
    projection: HigherOrderHypothesisContextProjection
    authorization: HigherOrderShadowGenerationAuthorization
    run_outcome: HigherOrderShadowHypothesisRunOutcome

    @property
    def portfolio_for_downstream(
        self,
    ) -> HypothesisPortfolio | None:
        return self.run_outcome.portfolio_for_downstream


@dataclass(frozen=True)
class HigherOrderShadowBatchOutcome:
    arms: tuple[HigherOrderShadowBatchArmOutcome, ...]
    record: HigherOrderShadowBatchRunRecord

    @property
    def proposed_arms(
        self,
    ) -> tuple[HigherOrderShadowBatchArmOutcome, ...]:
        return tuple(
            arm
            for arm in self.arms
            if arm.run_outcome.proposed
        )

    @property
    def portfolios_for_downstream(
        self,
    ) -> tuple[HypothesisPortfolio, ...]:
        rows = []
        for arm in self.proposed_arms:
            portfolio = arm.portfolio_for_downstream
            if portfolio is not None:
                rows.append(portfolio)
        return tuple(rows)


class HigherOrderShadowBatchRuntime:
    """
    Execute the automatic higher-order pool as independent shadow arms.

    The batch reuses the existing deterministic one-context-per-modifier
    coverage selector. It performs no scientific quality ranking and does not
    merge, replace, reorder, or mutate any legacy/production hypothesis
    portfolio.
    """

    def __init__(
        self,
        draft_backend: HypothesisDraftBackend,
        *,
        compiler: HypothesisCompiler | None = None,
        validator: HypothesisValidator | None = None,
        max_repairs: int = 1,
    ) -> None:
        if max_repairs not in {0, 1}:
            raise ValueError(
                "higher-order shadow batch supports max_repairs "
                "of 0 or 1 only"
            )
        self.draft_backend = draft_backend
        self.compiler = compiler or HypothesisCompiler()
        self.validator = validator or HypothesisValidator()
        self.max_repairs = int(max_repairs)

    def _validate_input_contexts(
        self,
        contexts: Sequence[HigherOrderSynthesisContext],
    ) -> None:
        if not contexts:
            raise ValueError(
                "higher-order shadow batch requires at least one context"
            )

        context_ids = [row.context_id for row in contexts]
        if len(context_ids) != len(set(context_ids)):
            raise ValueError(
                "higher-order shadow batch input contains duplicate "
                "context_id"
            )

        requested_pairs = {
            (
                row.requested_source,
                row.requested_target,
            )
            for row in contexts
        }
        if len(requested_pairs) != 1:
            raise ValueError(
                "higher-order shadow batch contexts must share one "
                "requested source-target relation"
            )

    def _arm_record(
        self,
        *,
        index: int,
        arm: HigherOrderShadowBatchArmOutcome,
    ) -> HigherOrderShadowBatchArmRecord:
        run = arm.run_outcome
        canonical = run.canonical_outcome
        portfolio = canonical.accepted_portfolio

        hypothesis_ids = (
            [
                row.hypothesis_id
                for row in portfolio.hypotheses
            ]
            if portfolio is not None
            else []
        )

        validation_codes = (
            [
                row.code
                for row in canonical.validation.issues
            ]
            if canonical.validation is not None
            else []
        )

        return HigherOrderShadowBatchArmRecord(
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
            authorization_id=arm.authorization.authorization_id,
            derived_hypothesis_context_id=(
                arm.projection.context.context_id
            ),
            derived_hypothesis_context_sha256=(
                arm.projection.context.context_sha256
            ),
            status=run.status,
            shadow_contract_passed=run.shadow_contract_passed,
            canonical_runtime_accepted=canonical.accepted,
            portfolio_id=(
                portfolio.portfolio_id
                if portfolio is not None
                else None
            ),
            hypothesis_ids=hypothesis_ids,
            canonical_failure_stage=(
                canonical.run_record.failure_stage
            ),
            canonical_compile_issue_codes=[
                row.code
                for row in canonical.compile_issues
            ],
            canonical_validation_issue_codes=validation_codes,
            shadow_failure_code=run.shadow_failure_code,
            shadow_failure_message=run.shadow_failure_message,
        )

    def run(
        self,
        *,
        source_context: HypothesisContext,
        contexts: Sequence[HigherOrderSynthesisContext],
        max_contexts: int,
    ) -> HigherOrderShadowBatchOutcome:
        if max_contexts < 1:
            raise ValueError(
                "max_contexts must be at least 1"
            )

        self._validate_input_contexts(contexts)

        selected = select_shadow_synthesis_contexts(
            contexts=contexts,
            max_contexts=max_contexts,
        )
        if not selected:
            raise ValueError(
                "higher-order shadow batch selected zero contexts"
            )

        arm_outcomes: list[
            HigherOrderShadowBatchArmOutcome
        ] = []

        single_runtime = HigherOrderShadowHypothesisRuntime(
            self.draft_backend,
            compiler=self.compiler,
            validator=self.validator,
            max_repairs=self.max_repairs,
        )

        for higher_order_context in selected:
            projection = materialize_higher_order_hypothesis_context(
                source_context=source_context,
                higher_order_context=higher_order_context,
            )
            authorization = authorize_higher_order_shadow_generation(
                projection=projection,
                higher_order_context=higher_order_context,
            )
            run_outcome = single_runtime.run(
                projection=projection,
                higher_order_context=higher_order_context,
                authorization=authorization,
            )
            arm_outcomes.append(
                HigherOrderShadowBatchArmOutcome(
                    higher_order_context=higher_order_context,
                    projection=projection,
                    authorization=authorization,
                    run_outcome=run_outcome,
                )
            )

        arm_records = [
            self._arm_record(
                index=index,
                arm=arm,
            )
            for index, arm in enumerate(
                arm_outcomes,
                start=1,
            )
        ]

        selected_context_ids = [
            row.context_id
            for row in selected
        ]
        selected_modifier_ids = [
            row.lineage.modifier_component_id
            for row in selected
        ]

        batch_run_id = _stable_id(
            "higher_order_shadow_batch_run",
            source_context.context_id,
            source_context.context_sha256,
            max_contexts,
            *selected_context_ids,
            *[
                row.canonical_outcome.run_record.run_id
                for row in (
                    arm.run_outcome
                    for arm in arm_outcomes
                )
            ],
        )

        record = HigherOrderShadowBatchRunRecord(
            batch_run_id=batch_run_id,
            source_hypothesis_context_id=source_context.context_id,
            source_hypothesis_context_sha256=(
                source_context.context_sha256
            ),
            input_context_count=len(contexts),
            selected_context_count=len(selected),
            unique_modifier_count=len(
                {
                    row.lineage.modifier_component_id
                    for row in contexts
                }
            ),
            duplicate_or_over_quota_context_count=(
                len(contexts) - len(selected)
            ),
            max_contexts=max_contexts,
            selected_higher_order_context_ids=(
                selected_context_ids
            ),
            selected_modifier_component_ids=(
                selected_modifier_ids
            ),
            proposed_count=sum(
                row.status == "proposed"
                for row in arm_records
            ),
            abstained_count=sum(
                row.status == "abstained"
                for row in arm_records
            ),
            canonical_rejected_count=sum(
                row.status == "canonical_rejected"
                for row in arm_records
            ),
            shadow_contract_rejected_count=sum(
                row.status == "shadow_contract_rejected"
                for row in arm_records
            ),
            arms=arm_records,
        )

        return HigherOrderShadowBatchOutcome(
            arms=tuple(arm_outcomes),
            record=record,
        )
