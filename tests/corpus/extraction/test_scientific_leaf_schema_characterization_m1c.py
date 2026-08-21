from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.corpus.schemas import (
    CalculationNode,
    EntityNode,
    ExperimentNode,
    MechanismClaimNode,
    ObservationClaimNode,
)


def test_entity_type_is_domain_extensible_string():
    dac = EntityNode(
        id="dac",
        type="Catalyst",
        label="DAC catalyst",
        description=None,
    )
    sers = EntityNode(
        id="sers",
        type="PlasmonicSubstrate",
        label="SERS substrate",
        description=None,
    )

    assert dac.type == "Catalyst"
    assert sers.type == "PlasmonicSubstrate"


def test_calculation_type_is_domain_extensible_string():
    node = CalculationNode(
        id="calc",
        name="Electromagnetic simulation",
        calculation_type="fdtd",
        conditions=[],
        method_details="Finite-difference time-domain simulation.",
    )

    assert node.calculation_type == "fdtd"


@pytest.mark.parametrize(
    "claim_type",
    [
        "performance_comparison",
        "sers_intensity_observation",
    ],
)
def test_observation_claim_type_is_domain_extensible_string(claim_type):
    node = ObservationClaimNode(
        id="obs",
        claim_type=claim_type,
        statement="A directly evidence-supported observation.",
        basis="experimental",
        description=None,
    )

    assert node.claim_type == claim_type


@pytest.mark.parametrize(
    "claim_type",
    [
        "electronic_structure",
        "plasmonic_coupling",
    ],
)
def test_mechanism_claim_type_is_domain_extensible_string(claim_type):
    node = MechanismClaimNode(
        id="mech",
        claim_type=claim_type,
        statement="A source-grounded mechanistic interpretation.",
        basis="mixed",
        description=None,
    )

    assert node.claim_type == claim_type


@pytest.mark.parametrize(
    "model",
    [ObservationClaimNode, MechanismClaimNode],
)
def test_claim_basis_remains_controlled(model):
    with pytest.raises(ValidationError):
        model(
            id="claim",
            claim_type="example",
            statement="Example statement.",
            basis="speculative",
            description=None,
        )


def test_experiment_legacy_backfill_is_explicitly_preserved_as_holdout():
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


def test_unknown_experiment_method_legacy_backfill_defaults_to_other():
    node = ExperimentNode(
        id="exp",
        name="Custom method",
        experiment_type="custom_future_method",
        conditions=[],
        description=None,
    )

    assert node.experiment_family == "other"
    assert node.method_label == "Custom method"
    assert node.raw_method_name is None
