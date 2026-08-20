from __future__ import annotations

import pipeline_core.corpus.schemas as legacy
import pipeline_core.experiment_schema as core


def test_legacy_experiment_schema_symbols_are_core_objects():
    assert legacy.ExperimentFamily is core.ExperimentFamily
    assert legacy.ExperimentType is core.ExperimentType
    assert legacy.ExperimentNode is core.ExperimentNode


def test_experiment_node_is_owned_by_pipeline_core():
    assert core.ExperimentNode.__module__ == (
        "pipeline_core.experiment_schema"
    )


def test_legacy_experiment_backfill_survives_core_ownership():
    node = legacy.ExperimentNode(
        id="exp",
        name="HAADF-STEM",
        experiment_type="haadf_stem",
        conditions=[],
        description=None,
    )

    assert node.experiment_family == "microscopy"
    assert node.method_label == "HAADF-STEM"
    assert node.raw_method_name is None


def test_explicit_registry_fields_still_override_backfill():
    node = core.ExperimentNode(
        id="exp",
        name="XANES",
        experiment_type="xanes",
        experiment_family="diffraction",
        method_label="Explicit method label",
        raw_method_name="Explicit raw wording",
        conditions=[],
        description=None,
    )

    assert node.experiment_family == "diffraction"
    assert node.method_label == "Explicit method label"
    assert node.raw_method_name == "Explicit raw wording"
