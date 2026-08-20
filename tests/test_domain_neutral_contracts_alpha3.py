from dac_her.candidate_contracts import CandidateDecisionCard
from pipeline_core.experimental_contracts import (
    ExperimentalCheckResult,
    ExperimentalRealizabilityReport,
    ExperimentalRequirement,
)
from pipeline_core.physics_contracts import PhysicsCheckRequest
from pipeline_core.scope_contracts import ScientificScope
from pipeline_core.validation_contracts import ValidationSpecification


def test_legacy_v02_scope_populates_generic_mirrors():
    scope = ScientificScope(
        scope_id="scope:1",
        hypothesis_id="hypothesis:1",
        catalyst_class="dual_atom",
        hypothesis_level="candidate_specific",
        reaction="HER",
        environments=["acidic"],
        metals=["Pt", "Ru"],
        coordination_variables=["local_coordination_geometry"],
        independent_variables=["local_coordination_geometry"],
        dependent_observables=["HER_activity"],
        requires_candidate_concretization=False,
        scope_confidence="high",
        catalyst_class_rationale="dual-atom target",
        hypothesis_level_rationale="explicit candidate",
    )
    assert scope.schema_version == "scientific-scope-v02"
    assert scope.system_class == "dual_atom"
    assert scope.process == "HER"
    assert scope.components == ["Pt", "Ru"]
    assert scope.structural_variables == ["local_coordination_geometry"]


def test_sers_like_scope_is_not_forced_into_catalyst_or_reaction_vocab():
    scope = ScientificScope(
        schema_version="scientific-scope-v03",
        scope_id="scope:sers",
        hypothesis_id="hypothesis:sers",
        system_class="bimetallic_plasmonic_substrate",
        scientific_domain="SERS",
        process="surface_enhanced_raman_scattering",
        components=["Au", "Ag"],
        structural_variables=["nanogap", "surface_composition"],
        hypothesis_level="comparative_study",
        independent_variables=["nanogap"],
        dependent_observables=["enhancement_factor"],
        requires_candidate_concretization=True,
        scope_confidence="high",
        hypothesis_level_rationale="compare Au-Ag designs",
        system_class_rationale="plasmonic substrate",
    )
    assert scope.system_class == "bimetallic_plasmonic_substrate"
    assert scope.catalyst_class == "unknown"
    assert scope.reaction == "unknown"


def test_validation_checks_are_domain_extensible_and_sync_legacy_alias():
    spec = ValidationSpecification(
        schema_version="validation-specification-v03",
        specification_id="spec:sers",
        hypothesis_id="hypothesis:sers",
        source_scope_id="scope:sers",
        scientific_domain="SERS",
        validation_strategy="electromagnetic_and_spectroscopic_validation",
        requires_candidate_concretization=True,
        required_scientific_checks=[
            "local_field_enhancement",
            "lspr_alignment",
            "nanogap_stability",
        ],
    )
    assert spec.required_physics_checks == spec.required_scientific_checks


def test_physics_check_identifier_accepts_future_domain_check():
    row = PhysicsCheckRequest(
        request_id="request:1",
        hypothesis_id="hypothesis:sers",
        source_scope_id="scope:sers",
        source_validation_specification_id="spec:sers",
        scientific_domain="SERS",
        backend_class="electromagnetic_simulation",
        check_type="local_field_enhancement",
        reason="SERS enhancement requires near-field validation",
    )
    assert row.check_type == "local_field_enhancement"


def test_experimental_contract_accepts_optical_category_and_sers_check():
    requirement = ExperimentalRequirement(
        requirement_id="req:1",
        category="optical_characterization",
        capability="raman_mapping",
        necessity="required",
        rationale="map substrate uniformity",
        scientific_domain="SERS",
    )
    check = ExperimentalCheckResult(
        check_id="check:1",
        hypothesis_id="hypothesis:sers",
        check_type="sers_performance_testability",
        status="conditional",
        rationale="requires Raman mapping",
        scientific_domain="SERS",
        requirements=[requirement],
    )
    assert check.check_type == "sers_performance_testability"
    assert check.requirements[0].category == "optical_characterization"


def test_legacy_electrochemical_tests_are_mirrored_to_generic_performance_tests():
    report = ExperimentalRealizabilityReport(
        report_id="report:1",
        source_intake_id="intake:1",
        source_intake_sha256="sha",
        source_physics_report_id="physics:1",
        source_scope_id="scope:1",
        source_validation_specification_id="spec:1",
        hypothesis_id="hypothesis:1",
        disposition="conditionally_plausible",
        required_characterization=["atomic_resolution_microscopy"],
        required_electrochemical_tests=["her_polarization_and_kinetic_testing"],
    )
    assert report.required_performance_tests == [
        "her_polarization_and_kinetic_testing"
    ]
    assert "atomic_resolution_microscopy" in report.required_measurement_capabilities


def test_candidate_decision_contract_has_generic_scope_fields():
    card = CandidateDecisionCard(
        decision_id="decision:1",
        hypothesis_id="hypothesis:sers",
        hypothesis_statement="Au-Ag nanogap controls SERS",
        source_intake_id="intake:1",
        source_scope_id="scope:sers",
        source_validation_specification_id="spec:sers",
        source_physics_report_id="physics:sers",
        source_experimental_report_id="experimental:sers",
        system_class="bimetallic_plasmonic_substrate",
        scientific_domain="SERS",
        process="surface_enhanced_raman_scattering",
        hypothesis_level="comparative_study",
        validation_strategy="electromagnetic_and_spectroscopic_validation",
        requires_candidate_concretization=True,
        semantic_status="pass",
        physics_disposition="requires_computation",
        experimental_disposition="conditionally_plausible",
        final_disposition="requires_validation_design",
        synthesis_complexity="moderate",
        characterization_complexity="high",
        relative_cost_burden="high",
        relative_effort_burden="high",
        required_performance_tests=["sers_enhancement_factor_mapping"],
    )
    assert card.catalyst_class == "unknown"
    assert card.required_performance_tests == ["sers_enhancement_factor_mapping"]
