from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Protocol

from pipeline_core.feasibility_contracts import FeasibilityHypothesis, FeasibilityIntake
from pipeline_core.physics_contracts import (
    PhysicsCheckRequest,
    PhysicsCheckResult,
    PhysicsFeasibilityReport,
)
from dac_her.physics_rules import PhysicsCheckPlanner
from dac_her.scope_compiler import HypothesisScopeCompiler
from pipeline_core.scope_contracts import ScientificScope
from pipeline_core.validation_contracts import ValidationSpecification
from dac_her.validation_specification import ValidationSpecificationCompiler


class PhysicsTool(Protocol):
    tool_name: str
    tool_version: str

    def supports(self, request: PhysicsCheckRequest) -> bool: ...

    def run(
        self,
        request: PhysicsCheckRequest,
        hypothesis: FeasibilityHypothesis,
    ) -> PhysicsCheckResult: ...


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


class EvidenceSignalPhysicsTool:
    tool_name = "evidence_signal"
    tool_version = "physics-evidence-signal-v02"

    def supports(self, request: PhysicsCheckRequest) -> bool:
        return bool(request.relevant_terms)

    def run(
        self,
        request: PhysicsCheckRequest,
        hypothesis: FeasibilityHypothesis,
    ) -> PhysicsCheckResult:
        terms = [term.lower() for term in request.relevant_terms]
        matching = [
            row
            for row in hypothesis.premises
            if any(term in row.text.lower() for term in terms)
        ]
        if not matching:
            raise ValueError("EvidenceSignalPhysicsTool called without matching premise")

        return PhysicsCheckResult(
            check_id=_stable_id(
                "physics_check",
                hypothesis.hypothesis_id,
                request.check_type,
                self.tool_version,
            ),
            request_id=request.request_id,
            hypothesis_id=hypothesis.hypothesis_id,
            check_type=request.check_type,
            scientific_domain=request.scientific_domain,
            backend_class=self.tool_name,
            status="conditional",
            basis="reported_evidence",
            rationale=(
                "Supplied positive premise(s) mention this scope-applicable physical "
                "concept, but literature wording alone is not a definitive feasibility pass."
            ),
            source_statement_ids=[row.statement_id for row in matching],
            source_paper_ids=sorted({p for row in matching for p in row.paper_ids}),
        )


class DeferredComputationPhysicsTool:
    tool_name = "deferred_computation"
    tool_version = "physics-deferred-v02"

    COMPUTATION_CHECKS = {
        "pair_stability",
        "isolated_site_stability",
        "aggregation_risk",
        "thermodynamic_stability",
        "operating_state_stability",
        "hydrogen_adsorption",
        "water_dissociation",
        "oh_binding",
        "reaction_pathway",
        "electronic_structure",
    }

    def supports(self, request: PhysicsCheckRequest) -> bool:
        return True

    def run(
        self,
        request: PhysicsCheckRequest,
        hypothesis: FeasibilityHypothesis,
    ) -> PhysicsCheckResult:
        needs_computation = request.check_type in self.COMPUTATION_CHECKS
        return PhysicsCheckResult(
            check_id=_stable_id(
                "physics_check",
                hypothesis.hypothesis_id,
                request.check_type,
                self.tool_version,
            ),
            request_id=request.request_id,
            hypothesis_id=hypothesis.hypothesis_id,
            check_type=request.check_type,
            scientific_domain=request.scientific_domain,
            backend_class=self.tool_name,
            status="requires_computation" if needs_computation else "unknown",
            basis="unavailable",
            rationale=(
                "No configured deterministic/computational backend can currently "
                f"resolve {request.check_type}; preserve uncertainty instead of inferring feasibility."
            ),
        )


class PhysicsFeasibilityRuntime:
    def __init__(
        self,
        *,
        planner: PhysicsCheckPlanner | None = None,
        tools: Iterable[PhysicsTool] | None = None,
        scope_compiler: HypothesisScopeCompiler | None = None,
        specification_compiler: ValidationSpecificationCompiler | None = None,
    ) -> None:
        self.planner = planner or PhysicsCheckPlanner()
        self.tools = list(
            tools
            or [EvidenceSignalPhysicsTool(), DeferredComputationPhysicsTool()]
        )
        self.scope_compiler = scope_compiler or HypothesisScopeCompiler()
        self.specification_compiler = (
            specification_compiler or ValidationSpecificationCompiler()
        )

    def run_intake(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope] | None = None,
        specifications: list[ValidationSpecification] | None = None,
    ) -> list[PhysicsFeasibilityReport]:
        scopes = scopes or self.scope_compiler.compile_intake(intake)
        specifications = specifications or self.specification_compiler.compile_intake(
            intake, scopes
        )
        scope_by_id = {row.hypothesis_id: row for row in scopes}
        spec_by_id = {row.hypothesis_id: row for row in specifications}
        return [
            self.run_hypothesis(
                intake,
                hypothesis,
                scope_by_id[hypothesis.hypothesis_id],
                spec_by_id[hypothesis.hypothesis_id],
            )
            for hypothesis in intake.hypotheses
        ]

    def run_hypothesis(
        self,
        intake: FeasibilityIntake,
        hypothesis: FeasibilityHypothesis,
        scope: ScientificScope,
        specification: ValidationSpecification,
    ) -> PhysicsFeasibilityReport:
        if hypothesis.semantic_gate_status == "human_review_required":
            return PhysicsFeasibilityReport(
                report_id=_stable_id(
                    "physics_report",
                    intake.intake_sha256,
                    hypothesis.hypothesis_id,
                    scope.scope_id,
                    specification.specification_id,
                ),
                source_intake_id=intake.intake_id,
                source_intake_sha256=intake.intake_sha256,
                source_scope_id=scope.scope_id,
                source_validation_specification_id=specification.specification_id,
                hypothesis_id=hypothesis.hypothesis_id,
                scientific_domain=scope.scientific_domain,
                disposition="human_review_required",
                confidence="low",
                checks=[],
                unresolved_checks=[],
                not_applicable_checks=specification.not_applicable_physics_checks,  # type: ignore[arg-type]
                next_required_computations=[],
            )

        results: list[PhysicsCheckResult] = []
        for request in self.planner.plan(hypothesis, scope, specification):
            result = None
            for tool in self.tools:
                if not tool.supports(request):
                    continue
                try:
                    candidate = tool.run(request, hypothesis)
                except ValueError:
                    continue
                result = candidate
                break
            if result is None:
                raise RuntimeError(f"No physics tool handled request {request.request_id}")
            results.append(result)

        statuses = {row.status for row in results}
        blocking = sorted({row.check_type for row in results if row.status == "fail"})
        unresolved = sorted({
            row.check_type
            for row in results
            if row.status in {"unknown", "requires_computation"}
        })
        next_computations = [
            f"Resolve {row.check_type} for {hypothesis.hypothesis_id} with a configured "
            "calculation/database tool after any required validation-system concretization."
            for row in results
            if row.status == "requires_computation"
        ]

        if "fail" in statuses:
            disposition = "physically_implausible"
        elif "requires_computation" in statuses:
            disposition = "requires_computation"
        elif "unknown" in statuses:
            disposition = "insufficient_information"
        elif statuses == {"pass"}:
            disposition = "physically_supported"
        else:
            disposition = "conditionally_supported"

        high_quality = sum(
            row.basis in {"deterministic_rule", "computed_value"} for row in results
        )
        if results and high_quality == len(results):
            confidence = "high"
        elif any(
            row.basis in {"reported_evidence", "surrogate_prediction"}
            for row in results
        ):
            confidence = "medium"
        else:
            confidence = "low"

        return PhysicsFeasibilityReport(
            report_id=_stable_id(
                "physics_report",
                intake.intake_sha256,
                hypothesis.hypothesis_id,
                scope.scope_id,
                specification.specification_id,
                disposition,
            ),
            source_intake_id=intake.intake_id,
            source_intake_sha256=intake.intake_sha256,
            source_scope_id=scope.scope_id,
            source_validation_specification_id=specification.specification_id,
            hypothesis_id=hypothesis.hypothesis_id,
            scientific_domain=scope.scientific_domain,
            disposition=disposition,  # type: ignore[arg-type]
            confidence=confidence,  # type: ignore[arg-type]
            checks=results,
            blocking_checks=blocking,
            unresolved_checks=unresolved,
            not_applicable_checks=specification.not_applicable_physics_checks,  # type: ignore[arg-type]
            next_required_computations=next_computations,
        )
