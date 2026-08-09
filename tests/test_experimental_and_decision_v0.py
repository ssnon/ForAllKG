from __future__ import annotations

from dac_her.candidate_decision import CandidateDecisionEngine
from dac_her.experimental_runtime import ExperimentalRealizabilityRuntime
from dac_her.feasibility_contracts import FeasibilityHypothesis, FeasibilityIntake
from dac_her.physics_runtime import PhysicsFeasibilityRuntime
from dac_her.scope_compiler import HypothesisScopeCompiler
from dac_her.validation_specification import ValidationSpecificationCompiler


def make_intake(statement: str, hypothesis_type: str) -> FeasibilityIntake:
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
        hypotheses=[
            FeasibilityHypothesis(
                hypothesis_id="h:1",
                title="validation target",
                statement=statement,
                hypothesis_type=hypothesis_type,
                inferential_bridge="Proposed bridge.",
                source_paper_ids=["Kiwook_1"],
                candidate_dependency="none",
                semantic_gate_status="eligible",
            )
        ],
    )


def _run(intake: FeasibilityIntake):
    scopes = HypothesisScopeCompiler().compile_intake(intake)
    specs = ValidationSpecificationCompiler().compile_intake(intake, scopes)
    physics = PhysicsFeasibilityRuntime().run_intake(intake, scopes, specs)
    experimental = ExperimentalRealizabilityRuntime().run_intake(
        intake, physics, scopes, specs
    )
    decision = CandidateDecisionEngine().decide(
        intake, scopes, specs, physics, experimental
    )
    return scopes[0], specs[0], physics[0], experimental[0], decision.cards[0]


def test_dac_requires_atomic_pair_synthesis():
    _, _, _, experimental, _ = _run(
        make_intake(
            "A Fe-Ru dual-atom site may facilitate water dissociation and improve HER activity.",
            "mechanistic_extension",
        )
    )
    assert "atomic_pair_selective_synthesis" in experimental.required_synthesis_capabilities
    assert "isolated_single_atom_synthesis" not in experimental.required_synthesis_capabilities
    assert experimental.precedent_status == "not_assessed"


def test_sac_does_not_receive_dac_synthesis_template_and_requires_validation_design():
    scope, spec, physics, experimental, card = _run(
        make_intake(
            "The influence of nitrogen coordination may extend to nitrogen-coordinated single-atom catalysts, but the response may differ between acidic and alkaline environments.",
            "context_dependency",
        )
    )
    assert scope.catalyst_class == "single_atom"
    assert "isolated_single_atom_synthesis" in experimental.required_synthesis_capabilities
    assert "atomic_pair_selective_synthesis" not in experimental.required_synthesis_capabilities
    assert "atomic_pair_selective_synthesis" in experimental.not_applicable_capabilities
    assert "pair_stability" in physics.not_applicable_checks
    assert spec.validation_strategy == "context_comparison"
    assert card.final_disposition == "requires_validation_design"
    assert card.requires_candidate_concretization is True
