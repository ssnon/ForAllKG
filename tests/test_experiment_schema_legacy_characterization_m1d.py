from __future__ import annotations

import pytest

from pipeline_core.corpus.schemas import ExperimentNode


@pytest.mark.parametrize(
    ("method_id", "expected_family"),
    [
        ("cyclic_voltammetry", "electrochemistry"),
        ("linear_sweep_voltammetry", "electrochemistry"),
        ("tafel_analysis", "electrochemistry"),
        ("accelerated_degradation_test", "stability_test"),
        ("extended_electrolysis", "stability_test"),
        ("chronoamperometry", "electrochemistry"),
        ("chronopotentiometry", "electrochemistry"),
        ("haadf_stem", "microscopy"),
        ("tem", "microscopy"),
        ("xanes", "spectroscopy"),
        ("exafs", "spectroscopy"),
        ("xas", "spectroscopy"),
        ("icp_oes", "composition_analysis"),
    ],
)
def test_legacy_method_family_backfill_is_frozen(
    method_id,
    expected_family,
):
    node = ExperimentNode(
        id="exp",
        name="Legacy method",
        experiment_type=method_id,
        conditions=[],
        description=None,
    )

    assert node.experiment_family == expected_family
    assert node.method_label == "Legacy method"
    assert node.raw_method_name is None


def test_unknown_method_defaults_to_other():
    node = ExperimentNode(
        id="exp",
        name="Future method",
        experiment_type="future_domain_method",
        conditions=[],
        description=None,
    )

    assert node.experiment_family == "other"
    assert node.method_label == "Future method"
    assert node.raw_method_name is None


def test_explicit_registry_fields_override_legacy_backfill():
    node = ExperimentNode(
        id="exp",
        name="HAADF-STEM",
        experiment_type="haadf_stem",
        experiment_family="spectroscopy",
        method_label="Explicit label",
        raw_method_name="Original wording",
        conditions=[],
        description=None,
    )

    assert node.experiment_family == "spectroscopy"
    assert node.method_label == "Explicit label"
    assert node.raw_method_name == "Original wording"


def test_missing_method_label_falls_back_to_method_id_when_name_empty():
    node = ExperimentNode(
        id="exp",
        name="",
        experiment_type="xanes",
        conditions=[],
        description=None,
    )

    assert node.method_label == "xanes"
    assert node.experiment_family == "spectroscopy"
