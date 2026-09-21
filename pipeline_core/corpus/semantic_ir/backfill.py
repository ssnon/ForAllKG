from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.capability import CapabilityRequirement
from pipeline_core.corpus.semantic_ir.schema import SemanticObjectRef


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackfillTarget(StrictModel):
    """One source chunk selected for a future targeted enrichment pass."""

    source_chunk_ref: SemanticObjectRef
    missing_capabilities: list[str] = Field(min_length=1)
    reasons: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chunk_target(self) -> "BackfillTarget":
        if self.source_chunk_ref.object_kind != "chunk":
            raise ValueError("backfill targets must reference source chunks")
        return self


class BackfillPlan(StrictModel):
    """Planning artifact only; constructing it performs no enrichment call."""

    schema_version: Literal[
        "semantic-backfill-plan-v1"
    ] = "semantic-backfill-plan-v1"

    plan_id: str = Field(min_length=1)
    requirements: list[CapabilityRequirement] = Field(default_factory=list)
    targets: list[BackfillTarget] = Field(default_factory=list)
    target_count: int = Field(ge=0)

    execution_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    selection_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_target_count(self) -> "BackfillPlan":
        if self.target_count != len(self.targets):
            raise ValueError("target_count must equal len(targets)")
        return self


def build_backfill_plan(
    *,
    plan_id: str,
    requirements: list[CapabilityRequirement],
    targets: list[BackfillTarget],
) -> BackfillPlan:
    """Deduplicate chunk targets without executing any extraction work."""

    merged: dict[
        tuple[str, str | None, str, str],
        BackfillTarget,
    ] = {}

    for target in targets:
        key = target.source_chunk_ref.identity_key()
        previous = merged.get(key)
        if previous is None:
            merged[key] = target
            continue

        merged[key] = BackfillTarget(
            source_chunk_ref=previous.source_chunk_ref,
            missing_capabilities=sorted(
                set(previous.missing_capabilities)
                | set(target.missing_capabilities)
            ),
            reasons=list(
                dict.fromkeys(previous.reasons + target.reasons)
            ),
        )

    ordered = sorted(
        merged.values(),
        key=lambda item: item.source_chunk_ref.identity_key(),
    )
    return BackfillPlan(
        plan_id=plan_id,
        requirements=requirements,
        targets=ordered,
        target_count=len(ordered),
    )
