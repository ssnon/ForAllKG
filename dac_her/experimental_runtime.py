from __future__ import annotations

import hashlib
from collections.abc import Iterable

from pipeline_core.experimental_contracts import (
    ComplexityLevel,
    CostBurden,
    EffortBurden,
    ExperimentalCheckResult,
    ExperimentalRealizabilityReport,
    ExperimentalRequirement,
)
from pipeline_core.experimental_rules import GenericExperimentalRequirementPlanner
from pipeline_core.feasibility_contracts import FeasibilityHypothesis, FeasibilityIntake
from pipeline_core.physics_contracts import PhysicsFeasibilityReport
from pipeline_core.scope_compiler import HypothesisScopeCompiler
from pipeline_core.scope_contracts import ScientificScope
from pipeline_core.validation_contracts import ValidationSpecification
from pipeline_core.validation_specification import ValidationSpecificationCompiler


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _requirements_by_category(
    requirements: Iterable[ExperimentalRequirement],
    category: str,
) -> list[str]:
    return sorted({row.capability for row in requirements if row.category == category})


def _has(requirements: Iterable[ExperimentalRequirement], capability: str) -> bool:
    return any(row.capability == capability for row in requirements)


class ExperimentalRealizabilityRuntime:
    """Scope-aware, laboratory-agnostic experimental feasibility evaluator."""

    def __init__(
        self,
        *,
        planner: GenericExperimentalRequirementPlanner | None = None,
        scope_compiler: HypothesisScopeCompiler | None = None,
        specification_compiler: ValidationSpecificationCompiler | None = None,
    ) -> None:
        self.planner = planner or GenericExperimentalRequirementPlanner()
        self.scope_compiler = scope_compiler or HypothesisScopeCompiler()
        self.specification_compiler = (
            specification_compiler or ValidationSpecificationCompiler()
        )

    def run_intake(
        self,
        intake: FeasibilityIntake,
        physics_reports: list[PhysicsFeasibilityReport],
        scopes: list[ScientificScope] | None = None,
        specifications: list[ValidationSpecification] | None = None,
    ) -> list[ExperimentalRealizabilityReport]:
        scopes = scopes or self.scope_compiler.compile_intake(intake)
        specifications = specifications or self.specification_compiler.compile_intake(
            intake, scopes
        )
        physics_by_hypothesis = {row.hypothesis_id: row for row in physics_reports}
        scope_by_hypothesis = {row.hypothesis_id: row for row in scopes}
        spec_by_hypothesis = {row.hypothesis_id: row for row in specifications}

        reports: list[ExperimentalRealizabilityReport] = []
        for hypothesis in intake.hypotheses:
            physics = physics_by_hypothesis.get(hypothesis.hypothesis_id)
            if physics is None:
                raise ValueError(
                    f"Missing physics report for hypothesis {hypothesis.hypothesis_id}"
                )
            reports.append(
                self.run_hypothesis(
                    intake,
                    hypothesis,
                    physics,
                    scope_by_hypothesis[hypothesis.hypothesis_id],
                    spec_by_hypothesis[hypothesis.hypothesis_id],
                )
            )
        return reports

    def run_hypothesis(
        self,
        intake: FeasibilityIntake,
        hypothesis: FeasibilityHypothesis,
        physics: PhysicsFeasibilityReport,
        scope: ScientificScope,
        specification: ValidationSpecification,
    ) -> ExperimentalRealizabilityReport:
        if hypothesis.semantic_gate_status == "human_review_required" or (
            physics.disposition == "human_review_required"
        ):
            return ExperimentalRealizabilityReport(
                report_id=_stable_id(
                    "experimental_report",
                    intake.intake_sha256,
                    hypothesis.hypothesis_id,
                    scope.scope_id,
                    "human_review_required",
                ),
                source_intake_id=intake.intake_id,
                source_intake_sha256=intake.intake_sha256,
                source_physics_report_id=physics.report_id,
                source_scope_id=scope.scope_id,
                source_validation_specification_id=specification.specification_id,
                hypothesis_id=hypothesis.hypothesis_id,
                scientific_domain=scope.scientific_domain,
                disposition="human_review_required",
                synthesis_feasibility="unknown",
                structural_verifiability="unknown",
                active_site_verifiability="unknown",
                performance_testability="unknown",
                precedent_status="not_assessed",
                not_applicable_capabilities=specification.not_applicable_experimental_capabilities,
                synthesis_complexity="unknown",
                characterization_complexity="unknown",
                relative_cost_burden="unknown",
                relative_effort_burden="unknown",
                dominant_uncertainties=["upstream_semantic_or_physics_review_required"],
            )

        requirements = self.planner.plan(hypothesis, scope, specification)
        checks: list[ExperimentalCheckResult] = []

        def add_check(
            check_type: str,
            status: str,
            rationale: str,
            *,
            precedent_status: str | None = None,
            check_requirements: list[ExperimentalRequirement] | None = None,
            complexity: str | None = None,
            cost_burden: str | None = None,
            effort_burden: str | None = None,
        ) -> None:
            checks.append(
                ExperimentalCheckResult(
                    check_id=_stable_id(
                        "experimental_check",
                        hypothesis.hypothesis_id,
                        scope.scope_id,
                        check_type,
                        status,
                    ),
                    hypothesis_id=hypothesis.hypothesis_id,
                    check_type=check_type,
                    scientific_domain=scope.scientific_domain,
                    status=status,  # type: ignore[arg-type]
                    rationale=rationale,
                    precedent_status=precedent_status,  # type: ignore[arg-type]
                    requirements=list(check_requirements or []),
                    complexity=complexity,  # type: ignore[arg-type]
                    cost_burden=cost_burden,  # type: ignore[arg-type]
                    effort_burden=effort_burden,  # type: ignore[arg-type]
                )
            )

        add_check(
            "synthesis_precedent",
            "unknown",
            (
                "No dedicated synthesis-precedent retrieval backend is configured. "
                "Source papers supporting the hypothesis are not automatically treated as precedent for the target validation system."
            ),
            precedent_status="not_assessed",
        )
        add_check(
            "synthesis_route_plausibility",
            "unknown",
            "No explicit target-system-specific synthesis route has been retrieved or validated in v0.2.",
        )
        add_check(
            "precursor_availability",
            "unknown",
            "Target-system-specific precursor identities and availability are not yet represented by a dedicated backend.",
        )

        if scope.catalyst_class == "dual_atom":
            synthesis_complexity: ComplexityLevel = "high"
            synthesis_status = "conditional"
            synthesis_rationale = (
                "The target is DAC-scoped and requires selective paired-site formation while suppressing SAC, cluster, and nanoparticle alternatives."
            )
        elif scope.catalyst_class == "single_atom":
            synthesis_complexity = "high"
            synthesis_status = "conditional"
            synthesis_rationale = (
                "The target is SAC-scoped and requires stabilization of isolated sites while suppressing pair, cluster, and nanoparticle formation."
            )
        elif scope.catalyst_class == "mixed_atomic_site":
            synthesis_complexity = "very_high"
            synthesis_status = "conditional"
            synthesis_rationale = (
                "The target compares multiple site nuclearities, requiring explicit site-state control and discrimination."
            )
        elif _requirements_by_category(requirements, "synthesis"):
            synthesis_complexity = "moderate"
            synthesis_status = "conditional"
            synthesis_rationale = (
                "A generic atomic-site validation route can be described, but route-specific synthesizability remains unresolved."
            )
        else:
            synthesis_complexity = "unknown"
            synthesis_status = "unknown"
            synthesis_rationale = (
                "No synthesis-specific requirement could be inferred from the target scientific scope."
            )

        add_check(
            "synthesis_complexity",
            synthesis_status,
            synthesis_rationale,
            check_requirements=[row for row in requirements if row.category == "synthesis"],
            complexity=synthesis_complexity,
            effort_burden=(
                "very_high" if synthesis_complexity == "very_high"
                else "high" if synthesis_complexity == "high"
                else "moderate" if synthesis_complexity == "moderate"
                else "unknown"
            ),
        )

        structural_requirements = [
            row
            for row in requirements
            if row.category == "characterization"
            and row.capability in {
                "atomic_resolution_microscopy",
                "xray_absorption_coordination_analysis",
            }
        ]
        if structural_requirements:
            structural_status = "conditional"
            if scope.catalyst_class == "dual_atom":
                structural_rationale = (
                    "The DAC structure is testable in principle with complementary atomic-resolution and coordination-sensitive evidence, but unambiguous paired-site assignment remains conditional."
                )
            elif scope.catalyst_class == "single_atom":
                structural_rationale = (
                    "The SAC structure is testable in principle with complementary atomic-resolution and coordination-sensitive evidence, but isolation from paired/cluster alternatives must be demonstrated."
                )
            else:
                structural_rationale = (
                    "The atomic-site structure is experimentally addressable with complementary microscopy and coordination-sensitive characterization."
                )
        else:
            structural_status = "unknown"
            structural_rationale = (
                "No structure-specific verification requirement was inferred from the target scope."
            )
        add_check(
            "structural_verifiability",
            structural_status,
            structural_rationale,
            check_requirements=structural_requirements,
        )

        active_requirements = [
            row
            for row in requirements
            if row.category == "characterization"
            and row.capability in {
                "mechanism_sensitive_characterization",
                "operando_or_in_situ_validation",
            }
        ]
        if active_requirements:
            active_status = "conditional"
            active_rationale = (
                "The mechanistic/active-site claim is experimentally addressable in principle, but attribution remains conditional because ex-situ structure alone cannot establish the working active site."
            )
        else:
            active_status = "unknown"
            active_rationale = (
                "The current target scope does not provide enough detail to specify an active-site attribution experiment."
            )
        add_check(
            "active_site_verifiability",
            active_status,
            active_rationale,
            check_requirements=active_requirements,
        )

        performance_requirements = [
            row for row in requirements if row.category == "electrochemistry"
        ]
        if performance_requirements:
            performance_status = "pass"
            performance_rationale = (
                "The HER response is directly testable with standard electrochemical performance and durability measurements under explicit conditions."
            )
        else:
            performance_status = "unknown"
            performance_rationale = (
                "No directly testable HER performance observable was inferred from the target scope."
            )
        add_check(
            "performance_testability",
            performance_status,
            performance_rationale,
            check_requirements=performance_requirements,
        )

        characterization_count = len(
            _requirements_by_category(requirements, "characterization")
        )
        operando_needed = _has(requirements, "operando_or_in_situ_validation")
        if operando_needed and characterization_count >= 3:
            characterization_complexity: ComplexityLevel = "very_high"
        elif characterization_count >= 2:
            characterization_complexity = "high"
        elif characterization_count == 1:
            characterization_complexity = "moderate"
        else:
            characterization_complexity = "unknown"

        if synthesis_complexity == "very_high" or characterization_complexity == "very_high":
            relative_cost: CostBurden = "very_high"
            relative_effort: EffortBurden = "very_high"
        elif synthesis_complexity == "high" or characterization_complexity == "high":
            relative_cost = "high"
            relative_effort = "high"
        elif synthesis_complexity == "moderate" or characterization_complexity == "moderate":
            relative_cost = "moderate"
            relative_effort = "moderate"
        else:
            relative_cost = "unknown"
            relative_effort = "unknown"

        add_check(
            "relative_cost_burden",
            "conditional" if relative_cost != "unknown" else "unknown",
            "Relative burden is inferred from generic synthesis/characterization classes, not laboratory-specific monetary cost.",
            cost_burden=relative_cost,
        )
        add_check(
            "relative_effort_burden",
            "conditional" if relative_effort != "unknown" else "unknown",
            "Relative effort is a generic validation-complexity estimate and does not use local staffing, queues, schedules, or instrument ownership.",
            effort_burden=relative_effort,
        )
        add_check(
            "safety",
            "unknown",
            "Route-specific reagents and process conditions are not yet available, so safety cannot be responsibly inferred.",
        )

        uncertainties: list[str] = [
            "synthesis_precedent_not_assessed",
            "precursor_availability_not_assessed",
            "route_specific_safety_not_assessed",
        ]
        if synthesis_status in {"unknown", "conditional"}:
            uncertainties.append("target_specific_synthesis_route_not_validated")
        if structural_status != "pass":
            if scope.catalyst_class == "dual_atom":
                uncertainties.append("dac_structural_assignment_requires_complementary_evidence")
            elif scope.catalyst_class == "single_atom":
                uncertainties.append("sac_isolation_assignment_requires_complementary_evidence")
            else:
                uncertainties.append("atomic_site_structural_assignment_requires_complementary_evidence")
        if active_status != "pass":
            uncertainties.append("working_active_site_attribution_remains_conditional")
        if specification.requires_candidate_concretization:
            uncertainties.append("validation_systems_require_concretization")

        critical_statuses = {
            synthesis_status,
            structural_status,
            active_status,
            performance_status,
        }
        if "fail" in critical_statuses:
            disposition = "experimentally_implausible"
        elif synthesis_complexity == "very_high" or characterization_complexity == "very_high":
            disposition = "high_complexity"
        elif all(status == "unknown" for status in critical_statuses):
            disposition = "insufficient_information"
        elif "conditional" in critical_statuses or "unknown" in critical_statuses:
            disposition = "conditionally_plausible"
        else:
            disposition = "experimentally_plausible"

        return ExperimentalRealizabilityReport(
            report_id=_stable_id(
                "experimental_report",
                intake.intake_sha256,
                physics.report_id,
                scope.scope_id,
                specification.specification_id,
                hypothesis.hypothesis_id,
                disposition,
                synthesis_complexity,
                characterization_complexity,
            ),
            source_intake_id=intake.intake_id,
            source_intake_sha256=intake.intake_sha256,
            source_physics_report_id=physics.report_id,
            source_scope_id=scope.scope_id,
            source_validation_specification_id=specification.specification_id,
            hypothesis_id=hypothesis.hypothesis_id,
            scientific_domain=scope.scientific_domain,
            disposition=disposition,  # type: ignore[arg-type]
            checks=checks,
            synthesis_feasibility=synthesis_status,  # type: ignore[arg-type]
            structural_verifiability=structural_status,  # type: ignore[arg-type]
            active_site_verifiability=active_status,  # type: ignore[arg-type]
            performance_testability=performance_status,  # type: ignore[arg-type]
            precedent_status="not_assessed",
            required_synthesis_capabilities=_requirements_by_category(requirements, "synthesis"),
            required_characterization=_requirements_by_category(requirements, "characterization"),
            required_performance_tests=_requirements_by_category(requirements, "electrochemistry"),
            required_electrochemical_tests=_requirements_by_category(requirements, "electrochemistry"),
            not_applicable_capabilities=specification.not_applicable_experimental_capabilities,
            synthesis_complexity=synthesis_complexity,
            characterization_complexity=characterization_complexity,
            relative_cost_burden=relative_cost,
            relative_effort_burden=relative_effort,
            dominant_uncertainties=sorted(set(uncertainties)),
        )
