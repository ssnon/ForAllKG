from __future__ import annotations

from pipeline_core.discovery.feasibility.feasibility_contracts import (
    FeasibilityHypothesis,
    FeasibilityIntake,
    FeasibilityPremise,
    FeasibilityPrediction,
)
from pipeline_core.discovery.feasibility.physics_runtime import PhysicsFeasibilityRuntime
from pipeline_core.discovery.feasibility.scope_compiler import HypothesisScopeCompiler
from pipeline_core.runtime.validation_specification import ValidationSpecificationCompiler


def make_intake(statement: str, hypothesis_type: str = "mechanistic_extension") -> FeasibilityIntake:
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
                title="Water activation hypothesis",
                statement=statement,
                hypothesis_type=hypothesis_type,
                inferential_bridge="Proposed mechanistic bridge.",
                premises=[
                    FeasibilityPremise(
                        statement_id="s:1",
                        text="The study discusses water dissociation and hydrogen adsorption.",
                        epistemic_role="reported",
                        claim_kind="mechanism",
                        paper_ids=["Kiwook_3"],
                    )
                ],
                source_paper_ids=["Kiwook_3"],
                candidate_dependency="none",
                predictions=[
                    FeasibilityPrediction(
                        observation_id="p:1",
                        observable="water dissociation barrier",
                        expected_direction="decrease",
                        rationale="tests proposed water activation",
                    )
                ],
                semantic_gate_status="eligible",
            )
        ],
    )


def test_dac_physics_requires_pair_stability_not_isolated_site_stability():
    intake = make_intake(
        "A dual-atom site may facilitate water dissociation in alkaline HER."
    )
    scopes = HypothesisScopeCompiler().compile_intake(intake)
    specs = ValidationSpecificationCompiler().compile_intake(intake, scopes)
    report = PhysicsFeasibilityRuntime().run_intake(intake, scopes, specs)[0]

    assert report.disposition == "requires_computation"
    assert "pair_stability" in {row.check_type for row in report.checks}
    assert "isolated_site_stability" not in {row.check_type for row in report.checks}
    assert "isolated_site_stability" in report.not_applicable_checks
    assert any(row.check_type == "water_dissociation" for row in report.checks)


def test_sac_physics_requires_isolated_site_stability_not_pair_stability():
    intake = make_intake(
        "A nitrogen-coordinated single-atom catalyst may show environment-dependent HER activity.",
        hypothesis_type="context_dependency",
    )
    scopes = HypothesisScopeCompiler().compile_intake(intake)
    specs = ValidationSpecificationCompiler().compile_intake(intake, scopes)
    report = PhysicsFeasibilityRuntime().run_intake(intake, scopes, specs)[0]
    checks = {row.check_type for row in report.checks}

    assert "isolated_site_stability" in checks
    assert "pair_stability" not in checks
    assert "pair_stability" in report.not_applicable_checks
