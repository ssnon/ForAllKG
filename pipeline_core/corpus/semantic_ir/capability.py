from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CapabilityState = Literal[
    "absent",
    "partial",
    "complete",
    "unknown",
]
CapabilityMinimumState = Literal["partial", "complete"]
CapabilityObservedState = Literal[
    "absent",
    "partial",
    "complete",
    "unknown",
    "undeclared",
]


class CapabilityRecord(StrictModel):
    name: str = Field(min_length=1)
    state: CapabilityState
    detail: str = ""
    assessed_from: list[str] = Field(default_factory=list)


class CapabilityManifest(StrictModel):
    """Audit-only declaration of what a reusable extraction can support."""

    schema_version: Literal[
        "semantic-capability-manifest-v1"
    ] = "semantic-capability-manifest-v1"

    manifest_id: str = Field(min_length=1)
    scope_kind: Literal["paper", "corpus", "task"]
    scope_id: str = Field(min_length=1)
    capabilities: dict[str, CapabilityRecord] = Field(default_factory=dict)

    audit_only: Literal[True] = True
    negative_evidence_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_capability_keys(self) -> "CapabilityManifest":
        for key, record in self.capabilities.items():
            if key != record.name:
                raise ValueError(
                    "capability mapping key must equal CapabilityRecord.name"
                )
        return self


class CapabilityRequirement(StrictModel):
    capability: str = Field(min_length=1)
    minimum_state: CapabilityMinimumState = "complete"
    reason: str = Field(min_length=1)


class CapabilityRequirementAssessment(StrictModel):
    capability: str
    minimum_state: CapabilityMinimumState
    observed_state: CapabilityObservedState
    satisfied: bool
    requires_backfill: bool
    reason: str

    negative_evidence_inferred: Literal[False] = False


class CapabilityRequirementCheck(StrictModel):
    schema_version: Literal[
        "semantic-capability-requirement-check-v1"
    ] = "semantic-capability-requirement-check-v1"

    manifest_id: str
    requirement_count: int = Field(ge=0)
    satisfied_count: int = Field(ge=0)
    backfill_required_count: int = Field(ge=0)
    assessments: list[CapabilityRequirementAssessment] = Field(
        default_factory=list
    )

    negative_evidence_inferred: Literal[False] = False
    scientific_selection_changed: Literal[False] = False


def evaluate_capability_requirements(
    *,
    manifest: CapabilityManifest,
    requirements: list[CapabilityRequirement],
) -> CapabilityRequirementCheck:
    rank = {
        "undeclared": 0,
        "unknown": 0,
        "absent": 0,
        "partial": 1,
        "complete": 2,
    }
    required_rank = {
        "partial": 1,
        "complete": 2,
    }

    assessments: list[CapabilityRequirementAssessment] = []
    for requirement in requirements:
        record = manifest.capabilities.get(requirement.capability)
        observed: CapabilityObservedState = (
            record.state if record is not None else "undeclared"
        )
        satisfied = rank[observed] >= required_rank[requirement.minimum_state]
        assessments.append(
            CapabilityRequirementAssessment(
                capability=requirement.capability,
                minimum_state=requirement.minimum_state,
                observed_state=observed,
                satisfied=satisfied,
                requires_backfill=not satisfied,
                reason=requirement.reason,
            )
        )

    satisfied_count = sum(row.satisfied for row in assessments)
    return CapabilityRequirementCheck(
        manifest_id=manifest.manifest_id,
        requirement_count=len(assessments),
        satisfied_count=satisfied_count,
        backfill_required_count=(len(assessments) - satisfied_count),
        assessments=assessments,
    )
