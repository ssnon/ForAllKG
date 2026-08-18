from __future__ import annotations

import dac_her.schemas as legacy
import pipeline_core.measurement_schema as core


def test_legacy_measurement_schema_symbols_are_core_objects():
    assert legacy.Condition is core.Condition
    assert legacy.MeasurementNode is core.MeasurementNode
    assert legacy.MeasurementGroupType is core.MeasurementGroupType
    assert legacy.MeasurementGroupNode is core.MeasurementGroupNode


def test_measurement_models_are_owned_by_pipeline_core():
    assert core.Condition.__module__ == "pipeline_core.measurement_schema"
    assert core.MeasurementNode.__module__ == "pipeline_core.measurement_schema"
    assert (
        core.MeasurementGroupNode.__module__
        == "pipeline_core.measurement_schema"
    )
