from __future__ import annotations

from typing import Any


# Historical pre-registry compatibility mapping.
#
# This table is migration behavior for already-existing DAC-era extraction
# payloads. It is not the authoritative experiment-method vocabulary for
# current domain extraction.
LEGACY_EXPERIMENT_FAMILY_BY_METHOD_ID: dict[str, str] = {
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


def backfill_legacy_experiment_registry_fields(
    value: Any,
) -> Any:
    """Preserve historical ExperimentNode registry-field backfill behavior."""

    if not isinstance(value, dict):
        return value

    updated = dict(value)

    method_id = str(
        updated.get("experiment_type", "other")
    )

    updated.setdefault(
        "experiment_family",
        LEGACY_EXPERIMENT_FAMILY_BY_METHOD_ID.get(
            method_id,
            "other",
        ),
    )
    updated.setdefault(
        "method_label",
        updated.get("name") or method_id,
    )
    updated.setdefault(
        "raw_method_name",
        None,
    )

    return updated
