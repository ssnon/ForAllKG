from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.candidate_decision import CandidateDecisionEngine
from dac_her.experimental_runtime import ExperimentalRealizabilityRuntime
from pipeline_core.feasibility_contracts import FeasibilityIntake
from dac_her.physics_runtime import PhysicsFeasibilityRuntime
from dac_her.scope_compiler import HypothesisScopeCompiler
from dac_her.validation_specification import ValidationSpecificationCompiler


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "feasibility_v271_real_intake.json"
)


def _run_real_feasibility_slice():
    intake = FeasibilityIntake.model_validate_json(
        FIXTURE.read_text(encoding="utf-8")
    )
    scopes = HypothesisScopeCompiler().compile_intake(intake)
    specs = ValidationSpecificationCompiler().compile_intake(intake, scopes)
    physics = PhysicsFeasibilityRuntime().run_intake(
        intake,
        scopes,
        specs,
    )
    experimental = ExperimentalRealizabilityRuntime().run_intake(
        intake,
        physics,
        scopes,
        specs,
    )
    decisions = CandidateDecisionEngine().decide(
        intake,
        scopes,
        specs,
        physics,
        experimental,
    )
    return (
        intake,
        {row.hypothesis_id: row for row in scopes},
        {row.hypothesis_id: row for row in specs},
        {row.hypothesis_id: row for row in physics},
        {row.hypothesis_id: row for row in experimental},
        {row.hypothesis_id: row for row in decisions.cards},
    )


def test_real_h1_dac_material_family_regression():
    _, scopes, specs, physics, experimental, decisions = (
        _run_real_feasibility_slice()
    )
    hid = "hypothesis:dd869c34bca71796a46e"

    assert scopes[hid].catalyst_class == "dual_atom"
    assert scopes[hid].hypothesis_level == "material_family"
    assert "pair_stability" in specs[hid].required_physics_checks
    assert "isolated_site_stability" in specs[hid].not_applicable_physics_checks
    assert "pair_stability" not in physics[hid].not_applicable_checks
    assert "atomic_pair_selective_synthesis" in (
        experimental[hid].required_synthesis_capabilities
    )
    assert decisions[hid].final_disposition == "requires_validation_design"


def test_real_h2_controlled_geometry_regression():
    _, scopes, specs, _, experimental, decisions = (
        _run_real_feasibility_slice()
    )
    hid = "hypothesis:b4d2ea70f6bf7f262361"

    assert scopes[hid].catalyst_class == "dual_atom"
    assert scopes[hid].hypothesis_level == "comparative_study"
    assert "nitrogen_coordination_number" in specs[hid].controlled_variables
    assert "local_coordination_geometry" in specs[hid].varied_variables
    assert "atomic_pair_selective_synthesis" in (
        experimental[hid].required_synthesis_capabilities
    )
    assert decisions[hid].final_disposition == "requires_validation_design"


def test_real_h3_sac_context_extension_never_receives_dac_checks():
    _, scopes, specs, physics, experimental, decisions = (
        _run_real_feasibility_slice()
    )
    hid = "hypothesis:bb5cc23b0145f1220881"

    assert scopes[hid].catalyst_class == "single_atom"
    assert scopes[hid].hypothesis_level == "context_extension"
    assert set(scopes[hid].environments) == {"acidic", "alkaline"}

    assert "pair_stability" in specs[hid].not_applicable_physics_checks
    assert "pair_stability" not in specs[hid].required_physics_checks
    assert "isolated_site_stability" in specs[hid].required_physics_checks
    assert "pair_stability" in physics[hid].not_applicable_checks

    assert "atomic_pair_selective_synthesis" in (
        specs[hid].not_applicable_experimental_capabilities
    )
    assert "atomic_pair_selective_synthesis" not in (
        experimental[hid].required_synthesis_capabilities
    )
    assert "isolated_single_atom_synthesis" in (
        experimental[hid].required_synthesis_capabilities
    )
    assert (
        "dac_structural_assignment_requires_complementary_evidence"
        not in experimental[hid].dominant_uncertainties
    )
    assert (
        "sac_isolation_assignment_requires_complementary_evidence"
        in experimental[hid].dominant_uncertainties
    )
    assert decisions[hid].final_disposition == "requires_validation_design"
