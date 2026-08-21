from __future__ import annotations

from pipeline_core.corpus.extraction.experiment_legacy_compat import (
    LEGACY_EXPERIMENT_FAMILY_BY_METHOD_ID,
    backfill_legacy_experiment_registry_fields,
)
from pipeline_core.corpus.extraction.experiment_schema import ExperimentNode


def test_legacy_mapping_is_explicitly_isolated():
    assert (
        LEGACY_EXPERIMENT_FAMILY_BY_METHOD_ID["haadf_stem"]
        == "microscopy"
    )
    assert (
        LEGACY_EXPERIMENT_FAMILY_BY_METHOD_ID["xanes"]
        == "spectroscopy"
    )


def test_compat_helper_preserves_unknown_fallback():
    result = backfill_legacy_experiment_registry_fields(
        {
            "id": "exp",
            "name": "Future method",
            "experiment_type": "future_method",
        }
    )

    assert result["experiment_family"] == "other"
    assert result["method_label"] == "Future method"
    assert result["raw_method_name"] is None


def test_compat_helper_preserves_explicit_values():
    result = backfill_legacy_experiment_registry_fields(
        {
            "id": "exp",
            "name": "XANES",
            "experiment_type": "xanes",
            "experiment_family": "diffraction",
            "method_label": "Explicit",
            "raw_method_name": "Source wording",
        }
    )

    assert result["experiment_family"] == "diffraction"
    assert result["method_label"] == "Explicit"
    assert result["raw_method_name"] == "Source wording"


def test_experiment_node_still_applies_compat_helper():
    node = ExperimentNode(
        id="exp",
        name="HAADF-STEM",
        experiment_type="haadf_stem",
        conditions=[],
        description=None,
    )

    assert node.experiment_family == "microscopy"
    assert node.method_label == "HAADF-STEM"
    assert node.raw_method_name is None
