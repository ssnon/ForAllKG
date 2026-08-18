from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.measurement_schema import Condition


# ============================================================
# Shared experiment wire contract
# ============================================================

ExperimentFamily = Literal[
    "electrochemistry",
    "microscopy",
    "spectroscopy",
    "diffraction",
    "composition_analysis",
    "surface_area_analysis",
    "thermal_analysis",
    "synthesis",
    "stability_test",
    "other",
]

# Registry-backed method identifier. The YAML vocabulary, rather than this
# Python type alias, controls which method IDs are registered.
ExperimentType = str


class ExperimentNode(BaseModel):
    """
    Experimental or characterization setup.

    Results from the experiment belong in
    MeasurementNode objects.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique experiment identifier."
        ),
    )

    name: str = Field(
        ...,
        description=(
            "Concise human-readable experiment name."
        ),
    )

    experiment_type: ExperimentType = Field(
        ...,
        description=(
            "Registry-backed method ID, such as xps, haadf_stem, "
            "or electrochemical_impedance_spectroscopy. Use an "
            "unregistered_<slug> ID only when no registry method fits."
        ),
    )

    experiment_family: ExperimentFamily = Field(
        ...,
        description="Broad experiment family from the controlled taxonomy.",
    )

    method_label: str = Field(
        ...,
        description="Preferred human-readable method label.",
    )

    raw_method_name: str | None = Field(
        ...,
        description=(
            "Source wording when it differs from the preferred label. "
            "Use null when unnecessary."
        ),
    )

    conditions: list[Condition] = Field(
        ...,
        description=(
            "All explicitly reported experimental "
            "conditions. Use an empty list when none "
            "are stated."
        ),
    )

    description: str | None = Field(
        ...,
        description=(
            "Brief description of what was performed."
        ),
    )


    @model_validator(mode="before")
    @classmethod
    def backfill_legacy_registry_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            method_id = str(value.get("experiment_type", "other"))
            family_map = {
                "cyclic_voltammetry": "electrochemistry",
                "linear_sweep_voltammetry": "electrochemistry",
                "tafel_analysis": "electrochemistry",
                "accelerated_degradation_test": "stability_test",
                "extended_electrolysis": "stability_test",
                "chronoamperometry": "electrochemistry",
                "chronopotentiometry": "electrochemistry",
                "haadf_stem": "microscopy",
                "tem": "microscopy",
                "xanes": "spectroscopy",
                "exafs": "spectroscopy",
                "xas": "spectroscopy",
                "icp_oes": "composition_analysis",
            }
            value.setdefault("experiment_family", family_map.get(method_id, "other"))
            value.setdefault("method_label", value.get("name") or method_id)
            value.setdefault("raw_method_name", None)
        return value
