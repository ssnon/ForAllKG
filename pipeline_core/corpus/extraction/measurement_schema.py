from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# Shared value object
# ============================================================

class Condition(BaseModel):
    """
    One structured experimental or computational condition.

    Examples:
    - electrolyte = 0.5 M H2SO4
    - scan rate = 2 mV s^-1
    - current density = 10 mA cm^-2
    - overpotential = 0.05 V
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    name: str = Field(
        ...,
        description=(
            "Condition name, such as electrolyte, "
            "scan rate, overpotential, pH, duration, "
            "temperature, or current density."
        ),
    )

    value_numeric: float | None = Field(
        ...,
        description=(
            "Numeric value when available. "
            "Use null for textual values."
        ),
    )

    value_text: str | None = Field(
        ...,
        description=(
            "Textual value when a numeric value "
            "is inappropriate, such as room temperature "
            "or N2-saturated."
        ),
    )

    unit: str | None = Field(
        ...,
        description=(
            "Unit associated with value_numeric. "
            "Use null when not applicable."
        ),
    )

    reference: str | None = Field(
        ...,
        description=(
            "Reference scale or qualifier, such as "
            "versus RHE. Use null when unavailable."
        ),
    )

    @model_validator(mode="after")
    def validate_value(self) -> "Condition":
        if (
            self.value_numeric is None
            and self.value_text is None
        ):
            raise ValueError(
                "Condition must contain either "
                "value_numeric or value_text."
            )

        return self


# ============================================================
# Measurement node models
# ============================================================

class MeasurementNode(BaseModel):
    """
    One measured or calculated result.

    Each catalyst-specific value should normally be
    represented by a separate MeasurementNode.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique measurement identifier."
        ),
    )

    metric_id: str = Field(
        ...,
        description=(
            "Registry-backed generic metric ID. Encode analyte, orbital, site, "
            "component, or isotope as structured Condition objects rather "
            "than creating element-specific metric IDs. Use "
            "unregistered_<slug> only when no generic metric fits."
        ),
    )

    metric: str = Field(
        ...,
        description=(
            "Preferred human-readable metric label, such as mass activity, "
            "Tafel slope, potential drop, adsorption energy, or Delta G H."
        ),
    )

    subject_id: str = Field(
        ...,
        description=(
            "Exactly one scientific Entity ID whose property this result "
            "describes. The graph must contain a matching MEASURED_FOR edge."
        ),
    )

    source_expression: str = Field(
        ...,
        description=(
            "Concise source expression containing this single result and its "
            "immediate qualifiers. Do not combine multiple subjects or "
            "conditions into one expression."
        ),
    )

    group_id: str | None = Field(
        ...,
        description=(
            "MeasurementGroup ID when this value belongs to a comparison, "
            "series, before/after pair, or condition sweep. Otherwise null."
        ),
    )

    value_numeric: float | None = Field(
        ...,
        description=(
            "Numeric result when available."
        ),
    )

    value_text: str | None = Field(
        ...,
        description=(
            "Textual result when the result is "
            "qualitative or comparative."
        ),
    )

    unit: str | None = Field(
        ...,
        description=(
            "Unit of value_numeric. Use null when "
            "dimensionless or unavailable."
        ),
    )

    uncertainty: str | None = Field(
        ...,
        description=(
            "Reported uncertainty or range, preserving "
            "the source notation. Use null when absent."
        ),
    )

    qualifier: str | None = Field(
        ...,
        description=(
            "Qualifier such as approximately, only, "
            "54-fold greater, or comparable. "
            "Use null when unnecessary."
        ),
    )

    basis: str | None = Field(
        ...,
        description=(
            "Normalization or measurement basis, such as "
            "per metal mass, geometric area, or Pt mass."
        ),
    )

    conditions: list[Condition] = Field(
        ...,
        description=(
            "Conditions and metric parameters specific to this result, such as "
            "electrolyte, overpotential, H coverage, analyte, orbital, "
            "site, component, or isotope."
        ),
    )

    description: str | None = Field(
        ...,
        description=(
            "Brief source-grounded explanation."
        ),
    )

    @model_validator(mode="after")
    def validate_value(self) -> "MeasurementNode":
        populated = sum(
            value is not None
            for value in (self.value_numeric, self.value_text)
        )
        if populated != 1:
            raise ValueError(
                "Measurement must contain exactly one of value_numeric or "
                "value_text. Preserve the source wording separately in "
                "source_expression."
            )
        if not self.subject_id.strip():
            raise ValueError("Measurement subject_id must not be blank.")
        if not self.metric_id.strip():
            raise ValueError("Measurement metric_id must not be blank.")
        if not self.source_expression.strip():
            raise ValueError("Measurement source_expression must not be blank.")
        return self


MeasurementGroupType = Literal[
    "comparison",
    "series",
    "before_after",
    "condition_sweep",
]


class MeasurementGroupNode(BaseModel):
    """Non-destructive container preserving that scalar results were reported together."""

    model_config = ConfigDict(extra="forbid")

    id: str
    group_type: MeasurementGroupType
    label: str
    member_measurement_ids: list[str]
    description: str | None

    @model_validator(mode="after")
    def validate_members(self) -> "MeasurementGroupNode":
        if len(self.member_measurement_ids) < 2:
            raise ValueError(
                "MeasurementGroup must contain at least two scalar measurements."
            )
        if len(self.member_measurement_ids) != len(set(self.member_measurement_ids)):
            raise ValueError("MeasurementGroup contains duplicate member IDs.")
        return self
