from __future__ import annotations

import hashlib

from dac_her.candidate_contracts import CandidateDecisionCard, CandidateDecisionPortfolio
from pipeline_core.experimental_contracts import ExperimentalRealizabilityReport
from pipeline_core.feasibility_contracts import FeasibilityIntake
from pipeline_core.physics_contracts import PhysicsFeasibilityReport
from pipeline_core.scope_contracts import ScientificScope
from pipeline_core.validation_contracts import ValidationSpecification


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


class CandidateDecisionEngine:
    """Deterministic v0.2 triage with validation-design awareness."""

    def decide(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope],
        specifications: list[ValidationSpecification],
        physics_reports: list[PhysicsFeasibilityReport],
        experimental_reports: list[ExperimentalRealizabilityReport],
    ) -> CandidateDecisionPortfolio:
        scope_by_id = {row.hypothesis_id: row for row in scopes}
        spec_by_id = {row.hypothesis_id: row for row in specifications}
        physics_by_id = {row.hypothesis_id: row for row in physics_reports}
        experimental_by_id = {row.hypothesis_id: row for row in experimental_reports}

        cards: list[CandidateDecisionCard] = []
        for hypothesis in intake.hypotheses:
            scope = scope_by_id.get(hypothesis.hypothesis_id)
            specification = spec_by_id.get(hypothesis.hypothesis_id)
            physics = physics_by_id.get(hypothesis.hypothesis_id)
            experimental = experimental_by_id.get(hypothesis.hypothesis_id)
            if any(row is None for row in (scope, specification, physics, experimental)):
                raise ValueError(
                    f"Missing downstream scope/spec/report(s) for {hypothesis.hypothesis_id}"
                )
            assert scope is not None
            assert specification is not None
            assert physics is not None
            assert experimental is not None

            final = self._final_disposition(
                semantic_status=hypothesis.semantic_gate_status,
                specification=specification,
                physics=physics,
                experimental=experimental,
            )
            uncertainties = sorted({
                *(f"semantic:{dimension}" for dimension in hypothesis.semantic_warning_dimensions),
                *(f"semantic_fail:{dimension}" for dimension in hypothesis.semantic_fail_dimensions),
                *(f"scope:{warning}" for warning in scope.scope_warnings),
                *(f"physics:{check}" for check in physics.unresolved_checks),
                *(f"experimental:{item}" for item in experimental.dominant_uncertainties),
            })

            cards.append(
                CandidateDecisionCard(
                    decision_id=_stable_id(
                        "candidate_decision",
                        intake.intake_sha256,
                        hypothesis.hypothesis_id,
                        scope.scope_id,
                        specification.specification_id,
                        physics.report_id,
                        experimental.report_id,
                        final,
                    ),
                    hypothesis_id=hypothesis.hypothesis_id,
                    hypothesis_statement=hypothesis.statement,
                    source_intake_id=intake.intake_id,
                    source_scope_id=scope.scope_id,
                    source_validation_specification_id=specification.specification_id,
                    source_physics_report_id=physics.report_id,
                    source_experimental_report_id=experimental.report_id,
                    system_class=scope.system_class,
                    scientific_domain=scope.scientific_domain,
                    process=scope.process,
                    catalyst_class=scope.catalyst_class,  # legacy compatibility
                    hypothesis_level=scope.hypothesis_level,
                    validation_strategy=specification.validation_strategy,
                    requires_candidate_concretization=specification.requires_candidate_concretization,
                    semantic_status=hypothesis.semantic_gate_status,
                    physics_disposition=physics.disposition,
                    experimental_disposition=experimental.disposition,
                    final_disposition=final,  # type: ignore[arg-type]
                    key_uncertainties=uncertainties,
                    required_computations=list(physics.next_required_computations),
                    candidate_concretization_requirements=list(
                        specification.candidate_concretization_requirements
                    ),
                    required_comparisons=list(specification.required_comparisons),
                    not_applicable_physics_checks=list(
                        specification.not_applicable_physics_checks
                    ),
                    required_synthesis_capabilities=list(
                        experimental.required_synthesis_capabilities
                    ),
                    required_characterization=list(experimental.required_characterization),
                    required_performance_tests=list(
                        experimental.required_performance_tests
                    ),
                    required_electrochemical_tests=list(
                        experimental.required_electrochemical_tests
                    ),
                    not_applicable_experimental_capabilities=list(
                        specification.not_applicable_experimental_capabilities
                    ),
                    synthesis_complexity=experimental.synthesis_complexity,
                    characterization_complexity=experimental.characterization_complexity,
                    relative_cost_burden=experimental.relative_cost_burden,
                    relative_effort_burden=experimental.relative_effort_burden,
                )
            )

        return CandidateDecisionPortfolio(
            decision_portfolio_id=_stable_id(
                "candidate_decision_portfolio",
                intake.intake_sha256,
                *(row.decision_id for row in cards),
                intake.abstention_reason or "",
            ),
            source_intake_id=intake.intake_id,
            cards=cards,
            abstention_reason=intake.abstention_reason,
        )

    @staticmethod
    def _final_disposition(
        *,
        semantic_status: str,
        specification: ValidationSpecification,
        physics: PhysicsFeasibilityReport,
        experimental: ExperimentalRealizabilityReport,
    ) -> str:
        if semantic_status == "human_review_required":
            return "human_review_required"
        if physics.disposition == "human_review_required":
            return "human_review_required"
        if physics.disposition == "physically_implausible":
            return "rejected_physical"
        if experimental.disposition == "human_review_required":
            return "human_review_required"
        if experimental.disposition == "experimentally_implausible":
            return "rejected_experimental"

        # Relationship/family/context hypotheses must first be converted into
        # concrete comparison systems. This prevents an abstract hypothesis from
        # being incorrectly treated as an immediately runnable DFT candidate.
        if specification.requires_candidate_concretization:
            return "requires_validation_design"

        if physics.disposition == "requires_computation":
            return "ready_for_high_fidelity_computation"
        if physics.disposition == "insufficient_information":
            return "insufficient_information"

        if experimental.precedent_status == "no_precedent_found":
            return "low_precedent_candidate"
        if experimental.disposition == "high_complexity":
            return "high_complexity_candidate"
        if experimental.disposition == "insufficient_information":
            return "insufficient_information"
        if experimental.disposition == "conditionally_plausible":
            return "conditional_candidate"
        return "ready_for_experimental_validation"
