from __future__ import annotations

import hashlib
import json

from domains.sers.classical_em_evidence_contracts import (
    SERSClassicalEMPhysicsEvidence,
    SERSClassicalEMPhysicsEvidenceBundle,
)
from domains.sers.classical_em_readiness_contracts import (
    SERSClassicalEMReadinessAssessment,
    SERSClassicalEMReadinessBundle,
    SERSEMModelFormSnapshot,
)
from domains.sers.validation_design_contracts import (
    SERSValidationDesign,
    SERSValidationDesignBundle,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


_GATE_VERSION = "sers-classical-em-readiness-gate-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _route_target_partition(
    route: SERSValidationRoute,
    evidence: SERSClassicalEMPhysicsEvidence,
) -> tuple[list[str], list[str]]:
    """Report only observables directly represented by the current evidence.

    v0 EM evidence contains scattering spectra/resonance locations and numerical
    convergence metadata. It does not contain spatial near-field maps, hotspot
    coverage, or local-field enhancement values. Unknown target labels fail
    closed as unobserved rather than being inferred from a spectrum.
    """

    observed: list[str] = []
    unobserved: list[str] = []
    has_spectrum = bool(evidence.spectrum_snapshots)

    for observable in route.target_observables:
        normalized = observable.strip().lower()
        if normalized == "substrate electromagnetic spectral response" and has_spectrum:
            observed.append(observable)
        else:
            unobserved.append(observable)
    return _unique(observed), _unique(unobserved)


def _find_route(
    plans: SERSHypothesisValidationPlanBundle,
    evidence: SERSClassicalEMPhysicsEvidence,
) -> SERSValidationRoute:
    plan = next(
        (row for row in plans.plans if row.plan_id == evidence.source_validation_plan_id),
        None,
    )
    if plan is None:
        raise ValueError(
            "classical-EM evidence references missing validation plan "
            f"{evidence.source_validation_plan_id}"
        )
    if plan.hypothesis_id != evidence.hypothesis_id:
        raise ValueError("validation plan/evidence hypothesis mismatch")
    route = next(
        (row for row in plan.routes if row.route_id == evidence.source_validation_route_id),
        None,
    )
    if route is None:
        raise ValueError(
            "classical-EM evidence references missing validation route "
            f"{evidence.source_validation_route_id}"
        )
    if route.route_kind != "classical_em":
        raise ValueError("classical-EM evidence must descend from a classical_em route")
    if set(route.source_fdtd_subclaim_ids) != set(evidence.source_fdtd_subclaim_ids):
        raise ValueError("validation route/FDTD subclaim lineage mismatch")
    return route


def _find_design(
    designs: SERSValidationDesignBundle,
    evidence: SERSClassicalEMPhysicsEvidence,
) -> tuple[SERSValidationDesign, object]:
    candidates = [
        row
        for row in designs.designs
        if row.hypothesis_id == evidence.hypothesis_id
        and row.source_spec_id == evidence.source_simulation_spec_id
    ]
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one validation design for evidence lineage; "
            f"found {len(candidates)}"
        )
    design = candidates[0]
    geometry = next(
        (
            row
            for row in design.geometry_candidates
            if row.candidate_id == evidence.geometry_candidate_id
        ),
        None,
    )
    if geometry is None:
        raise ValueError(
            "evidence geometry_candidate_id absent from validation design: "
            f"{evidence.geometry_candidate_id}"
        )
    if evidence.surrounding_medium not in design.surrounding_media:
        raise ValueError(
            "evidence surrounding medium absent from validation design: "
            f"{evidence.surrounding_medium}"
        )
    return design, geometry


def _numerical_actions(evidence: SERSClassicalEMPhysicsEvidence) -> tuple[list[str], list[str]]:
    immediate: list[str] = []
    deferred: list[str] = []
    for action in evidence.next_required_actions:
        if action == "resolve_spectral_boundary":
            immediate.append("resolve_spectral_boundary")
        elif action == "complete_polarization_pair":
            immediate.append("complete_polarization_pair")
        elif action == "add_resolution_point":
            immediate.append("add_resolution_point")
        elif action == "establish_convergence_acceptance_policy":
            immediate.append("establish_convergence_acceptance_policy")
        elif action == "improve_spatial_resolution_or_solver_fidelity":
            deferred.append("defer_high_cost_spatial_fidelity_escalation")
    return _unique(immediate), _unique(deferred)


class SERSClassicalEMReadinessGate:
    """Assess whether routed EM evidence is ready for scientific component review.

    This gate is deliberately stricter than the numerical evidence assembler. It
    distinguishes numerical adequacy, model-form adequacy, and evidence scope.
    v0 does not issue support/contradiction, experiment-promotion, or self-feedback
    authority.
    """

    gate_version = _GATE_VERSION

    def assess(
        self,
        evidence_bundle: SERSClassicalEMPhysicsEvidenceBundle,
        validation_plans: SERSHypothesisValidationPlanBundle,
        validation_designs: SERSValidationDesignBundle,
    ) -> SERSClassicalEMReadinessBundle:
        assessments: list[SERSClassicalEMReadinessAssessment] = []

        for evidence in evidence_bundle.evidence:
            route = _find_route(validation_plans, evidence)
            design, geometry = _find_design(validation_designs, evidence)

            observed_targets, unobserved_targets = _route_target_partition(route, evidence)
            scope_status = (
                "route_targets_observed"
                if not unobserved_targets
                else "route_targets_partially_observed"
            )

            numerical_readiness = (
                "review_candidate"
                if not evidence.numerical_blockers
                else "repair_required"
            )

            model_form = SERSEMModelFormSnapshot(
                validation_design_id=design.design_id,
                source_label=design.source_label,
                operator_note=design.operator_note,
                geometry_candidate_id=geometry.candidate_id,
                geometry_candidate_label=geometry.label,
                geometry_candidate_rationale=geometry.rationale,
                nanorod_length_nm=geometry.nanorod_length_nm,
                nanorod_diameter_nm=geometry.nanorod_diameter_nm,
                shell_thickness_nm=geometry.shell_thickness_nm,
                surrounding_medium=evidence.surrounding_medium,
                source_baseline_wavelength_nm=(
                    design.source_constraints.baseline_wavelength_nm
                ),
                source_reported_resonance_wavelengths_nm=list(
                    design.source_constraints.reported_resonance_wavelengths_nm
                ),
            )

            blockers: list[str] = []
            if numerical_readiness == "repair_required":
                blockers.append("numerical_evidence_not_ready")
            blockers.append("operator_concretized_model_unvalidated")
            blockers.append("positive_control_not_assessed")
            if unobserved_targets:
                blockers.append("route_target_observable_unobserved")

            immediate, deferred = _numerical_actions(evidence)

            next_actions: list[str] = list(immediate)
            next_actions.extend([
                "establish_positive_control_model_validation",
                "review_operator_concretized_model",
            ])
            if unobserved_targets:
                next_actions.append("expand_classical_em_observables")
            next_actions.extend(deferred)

            if numerical_readiness == "repair_required":
                overall = "numerical_repair_required"
            elif model_form.positive_control_status != "not_assessed":
                # Reserved for a future model-validation artifact; v0 never reaches
                # this branch because no positive-control evidence is accepted yet.
                overall = "evidence_scope_limited" if unobserved_targets else "ready_for_component_review"
            else:
                overall = "model_validation_required"

            assessments.append(SERSClassicalEMReadinessAssessment(
                assessment_id=_stable_id(
                    "sers_classical_em_readiness",
                    evidence.evidence_id,
                    route.route_id,
                    design.design_id,
                    self.gate_version,
                ),
                hypothesis_id=evidence.hypothesis_id,
                source_evidence_id=evidence.evidence_id,
                source_validation_plan_id=evidence.source_validation_plan_id,
                source_validation_route_id=evidence.source_validation_route_id,
                source_validation_design_id=design.design_id,
                source_simulation_spec_id=evidence.source_simulation_spec_id,
                route_target_observables=list(route.target_observables),
                observed_route_target_observables=observed_targets,
                unobserved_route_target_observables=unobserved_targets,
                evidence_scope_status=scope_status,
                numerical_evidence_state=evidence.numerical_evidence_state,
                numerical_blockers=list(evidence.numerical_blockers),
                numerical_readiness=numerical_readiness,
                model_form=model_form,
                readiness_blockers=_unique(blockers),
                immediate_numerical_repairs=immediate,
                deferred_high_cost_actions=deferred,
                next_research_actions=_unique(next_actions),
                overall_readiness=overall,
            ))

        return SERSClassicalEMReadinessBundle(
            bundle_id=_stable_id(
                "sers_classical_em_readiness_bundle",
                evidence_bundle.bundle_id,
                validation_plans.bundle_id,
                validation_designs.bundle_id,
                *(row.assessment_id for row in assessments),
                self.gate_version,
            ),
            source_evidence_bundle_id=evidence_bundle.bundle_id,
            source_validation_plan_bundle_id=validation_plans.bundle_id,
            source_validation_design_bundle_id=validation_designs.bundle_id,
            assessments=assessments,
            assessment_count=len(assessments),
        )
