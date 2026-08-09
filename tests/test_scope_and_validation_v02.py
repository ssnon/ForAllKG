from __future__ import annotations

from dac_her.feasibility_contracts import (
    FeasibilityHypothesis,
    FeasibilityIntake,
    FeasibilityPrediction,
)
from dac_her.scope_compiler import HypothesisScopeCompiler
from dac_her.validation_specification import ValidationSpecificationCompiler


def _intake(*hypotheses: FeasibilityHypothesis) -> FeasibilityIntake:
    return FeasibilityIntake(
        intake_id="intake",
        intake_sha256="isha",
        source_context_id="ctx",
        source_context_sha256="csha",
        source_portfolio_id="portfolio",
        source_portfolio_sha256="psha",
        source_semantic_review_id="review",
        task_id="task",
        question="q",
        corpus_id="corpus",
        hypotheses=list(hypotheses),
    )


def test_scope_compiler_does_not_let_dac_premise_override_sac_target():
    h = FeasibilityHypothesis(
        hypothesis_id="h:sac",
        title="Context extension to single-atom catalysts",
        statement=(
            "The influence of nitrogen coordination may extend to nitrogen-coordinated "
            "single-atom catalysts and differ between acidic and alkaline environments."
        ),
        hypothesis_type="context_dependency",
        inferential_bridge="A DAC paper supplies one supporting relationship.",
        source_paper_ids=["Kiwook_6", "Kiwook_9"],
        candidate_dependency="none",
        predictions=[
            FeasibilityPrediction(
                observation_id="p",
                observable="HER activity across acidic and alkaline environments",
                expected_direction="qualitative_change",
                rationale="environment-dependent response",
            )
        ],
        semantic_gate_status="eligible_with_warnings",
    )
    intake = _intake(h)
    scope = HypothesisScopeCompiler().compile_intake(intake)[0]
    spec = ValidationSpecificationCompiler().compile_intake(intake, [scope])[0]

    assert scope.catalyst_class == "single_atom"
    assert scope.hypothesis_level == "context_extension"
    assert set(scope.environments) == {"acidic", "alkaline"}
    assert "pair_stability" in spec.not_applicable_physics_checks
    assert "isolated_site_stability" in spec.required_physics_checks
    assert "atomic_pair_selective_synthesis" in spec.not_applicable_experimental_capabilities
    assert spec.validation_strategy == "context_comparison"


def test_dac_family_and_controlled_geometry_are_distinguished():
    family = FeasibilityHypothesis(
        hypothesis_id="h:family",
        title="Optimum-like coordination regime",
        statement=(
            "Within TM2@Nx-Gr dual-atom catalysts, HER activity may vary non-monotonically "
            "with nitrogen coordination number and local geometry."
        ),
        hypothesis_type="mechanistic_extension",
        inferential_bridge="Mechanistic proposal.",
        source_paper_ids=["Kiwook_9"],
        candidate_dependency="none",
        semantic_gate_status="eligible",
    )
    geometry = FeasibilityHypothesis(
        hypothesis_id="h:geometry",
        title="Geometry mediation",
        statement=(
            "At a given nitrogen coordination number in TM2@Nx-Gr dual-atom catalysts, "
            "changing local nitrogen coordination geometry may alter HER activity."
        ),
        hypothesis_type="descriptor_mediation",
        inferential_bridge="Controlled comparison proposal.",
        source_paper_ids=["Kiwook_9"],
        candidate_dependency="none",
        semantic_gate_status="eligible",
    )
    intake = _intake(family, geometry)
    scopes = HypothesisScopeCompiler().compile_intake(intake)
    specs = ValidationSpecificationCompiler().compile_intake(intake, scopes)
    by_id = {row.hypothesis_id: row for row in scopes}
    spec_by_id = {row.hypothesis_id: row for row in specs}

    assert by_id["h:family"].catalyst_class == "dual_atom"
    assert by_id["h:family"].hypothesis_level == "material_family"
    assert "pair_stability" in spec_by_id["h:family"].required_physics_checks

    assert by_id["h:geometry"].hypothesis_level == "comparative_study"
    assert "nitrogen_coordination_number" in spec_by_id["h:geometry"].controlled_variables
    assert "local_coordination_geometry" in spec_by_id["h:geometry"].varied_variables
