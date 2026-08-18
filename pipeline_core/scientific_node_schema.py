from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.measurement_schema import Condition


# ============================================================
# Shared scientific node contracts
# ============================================================

EntityType = str
CalculationType = str
ObservationClaimType = str
MechanismClaimType = str

MechanismBasis = Literal[
    "experimental",
    "computational",
    "mixed",
]


class EntityNode(BaseModel):
    """
    Ordinary scientific entities.

    Measurements, experiments, calculations, and
    mechanism claims should not be placed here.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique canonical graph node identifier."
        ),
    )

    type: EntityType

    label: str = Field(
        ...,
        description=(
            "Human-readable canonical node label."
        ),
    )

    description: str | None = Field(
        ...,
        description=(
            "Brief source-grounded description. "
            "Use null when unnecessary."
        ),
    )


class CalculationNode(BaseModel):
    """
    Computational procedure such as DFT, PDOS,
    adsorption-energy calculation, or FPMD.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique calculation identifier."
        ),
    )

    name: str = Field(
        ...,
        description=(
            "Concise human-readable calculation name."
        ),
    )

    calculation_type: CalculationType

    conditions: list[Condition] = Field(
        ...,
        description=(
            "Explicit computational settings or "
            "coverage conditions. Use an empty list "
            "when none are stated."
        ),
    )

    method_details: str | None = Field(
        ...,
        description=(
            "Brief method description, such as DFT "
            "model or functional. Use null when the "
            "chunk does not provide this information."
        ),
    )


class ObservationClaimNode(BaseModel):
    """
    Measurements or calculations summarized into a
    directly evidence-supported scientific conclusion.

    This node must not contain a causal explanation.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique observation-claim identifier."
        ),
    )

    claim_type: ObservationClaimType

    statement: str = Field(
        ...,
        description=(
            "A concise conclusion directly supported "
            "by reported measurements, calculations, "
            "or characterization results."
        ),
    )

    basis: MechanismBasis = Field(
        ...,
        description=(
            "Whether the observation is based on "
            "experimental, computational, or mixed "
            "evidence."
        ),
    )

    description: str | None = Field(
        ...,
        description=(
            "Additional source-grounded clarification. "
            "Use null when unnecessary."
        ),
    )

class MechanismClaimNode(BaseModel):
    """
    A causal, mechanistic, or explanatory interpretation.

    Directly reported numerical comparisons belong in
    ObservationClaimNode, not here.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique mechanism-claim identifier."
        ),
    )

    claim_type: MechanismClaimType

    statement: str = Field(
        ...,
        description=(
            "Concise faithful statement of the authors' "
            "causal or mechanistic interpretation."
        ),
    )

    basis: MechanismBasis

    description: str | None = Field(
        ...,
        description=(
            "Additional clarification. "
            "Use null when unnecessary."
        ),
    )


# ============================================================
# Edge and graph models
# ============================================================
