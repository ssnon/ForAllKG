from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from pipeline_core.discovery.feasibility.experimental_contracts import (
    ExperimentalCheckResult,
    ExperimentalRealizabilityReport,
    ExperimentalRequirement,
)
from pipeline_core.discovery.feasibility.feasibility_contracts import (
    FeasibilityHypothesis,
    FeasibilityIntake,
)
from pipeline_core.discovery.feasibility.physics_contracts import (
    PhysicsCheckResult,
    PhysicsFeasibilityReport,
)
from pipeline_core.discovery.feasibility.scope_contracts import ScientificScope
from pipeline_core.runtime.validation_contracts import ValidationSpecification


_ADAPTER_VERSION = "sers-feasibility-adapter-v0"


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _text(hypothesis: FeasibilityHypothesis) -> str:
    return " ".join(
        [
            hypothesis.title,
            hypothesis.statement,
            hypothesis.inferential_bridge,
            *hypothesis.assumptions,
            *(row.observable for row in hypothesis.predictions),
            *(row.rationale for row in hypothesis.predictions),
            *(row.observable for row in hypothesis.falsifiers),
            *(row.falsifying_outcome for row in hypothesis.falsifiers),
        ]
    ).lower()


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


_CLASSICAL_EM = (
    r"\blspr\b",
    r"plasmon(?:ic)?",
    r"electromagnetic",
    r"near[- ]field",
    r"hot\s*spot|hotspot",
    r"substrate[- ](?:only\s+)?resonance",
    r"optical\s+resonance",
    r"scattering\s+(?:spectrum|cross[- ]section)",
    r"extinction\s+spectrum",
    r"nanogap\s+coupling",
)
_MOLECULAR = (
    r"analyte[- ]specific",
    r"molecular\s+(?:resonance|absorption|response)",
    r"resonance\s+raman",
    r"raman\s+cross[- ]section",
    r"dye\s+absorption",
    r"methylene\s+blue",
    r"rhodamine",
    r"excitation[- ]wavelength\s+depend",
    r"wavelength[- ]depend(?:ent|ence)",
)
_SURFACE_CHEMISTRY = (
    r"chemical\s+enhancement",
    r"charge\s+transfer",
    r"adsor(?:b|ption|bed)",
    r"surface\s+(?:association|affinity|binding|chemistry)",
    r"chemical\s+association",
    r"binding\s+(?:energy|affinity)",
    r"molecular\s+orientation",
)
_FABRICATION = (
    r"synthes(?:is|ized|ize)",
    r"fabricat(?:e|ed|ion)",
    r"anneal(?:ing|ed)?",
    r"deposition",
    r"reproducib",
    r"uniformity",
    r"stability",
    r"aging",
    r"\brsd\b",
)
_INTEGRATED_SERS = (
    r"\bsers\s+(?:intensity|signal|response|enhancement\s+factor|ef)\b",
    r"raman\s+(?:intensity|signal)",
    r"limit\s+of\s+detection",
    r"\blod\b",
    r"spot[- ]to[- ]spot",
    r"batch[- ]to[- ]batch",
)


def _route_kinds(hypothesis: FeasibilityHypothesis) -> list[str]:
    text = _text(hypothesis)
    rows: list[str] = []
    if _matches(text, _CLASSICAL_EM):
        rows.append("classical_em")
    if _matches(text, _MOLECULAR):
        rows.append("molecular_spectroscopy")
    if _matches(text, _SURFACE_CHEMISTRY):
        rows.append("surface_chemistry")
    if _matches(text, _FABRICATION):
        rows.append("fabrication_process")
    if _matches(text, _INTEGRATED_SERS):
        rows.append("integrated_sers_experiment")
    if not rows:
        rows.append("unresolved_scientific_route")
    return rows


def _components(text: str) -> list[str]:
    rows: list[str] = []
    if re.search(r"\bau\b|gold", text, flags=re.IGNORECASE):
        rows.append("Au")
    if re.search(r"\bag\b|silver", text, flags=re.IGNORECASE):
        rows.append("Ag")
    if re.search(r"methylene\s+blue", text, flags=re.IGNORECASE):
        rows.append("methylene_blue")
    if re.search(r"rhodamine", text, flags=re.IGNORECASE):
        rows.append("rhodamine_reporter")
    return rows


def _independent_variables(text: str) -> list[str]:
    rows: list[str] = []
    candidates = (
        ("excitation_wavelength", r"excitation\s+wavelength|wavelength[- ]depend"),
        ("interparticle_gap", r"nanogap|interparticle\s+(?:gap|spacing|distance)"),
        ("shell_thickness", r"shell\s+thickness"),
        ("nanorod_aspect_ratio", r"aspect\s+ratio"),
        ("particle_geometry", r"nanorod|nanocube|nanohole|dimple|geometry"),
        ("surface_chemistry", r"adsor|binding|association|charge\s+transfer|chemical\s+enhancement"),
        ("fabrication_conditions", r"synthes|fabricat|anneal|deposition|temperature\s+treatment"),
    )
    for label, pattern in candidates:
        if re.search(pattern, text, flags=re.IGNORECASE):
            rows.append(label)
    return rows


def _structural_variables(text: str) -> list[str]:
    rows: list[str] = []
    candidates = (
        ("core_shell_architecture", r"core[- ]?shell|au\s*@\s*ag|ag\s*@\s*au"),
        ("nanorod_geometry", r"nanorod"),
        ("nanogap_geometry", r"nanogap|interparticle\s+(?:gap|spacing|distance)"),
        ("support_or_substrate_geometry", r"substrate|support|film|dimple|nanohole"),
        ("shell_thickness", r"shell\s+thickness"),
        ("aspect_ratio", r"aspect\s+ratio"),
    )
    for label, pattern in candidates:
        if re.search(pattern, text, flags=re.IGNORECASE):
            rows.append(label)
    return rows


def _system_class(text: str) -> str:
    if re.search(r"core[- ]?shell", text, flags=re.IGNORECASE) and re.search(
        r"nanorod", text, flags=re.IGNORECASE
    ):
        return "au_ag_core_shell_nanorod_sers"
    if re.search(r"nanogap", text, flags=re.IGNORECASE):
        return "au_ag_nanogap_sers"
    if re.search(r"au\s*@\s*ag|ag\s*@\s*au", text, flags=re.IGNORECASE):
        return "au_ag_core_shell_sers"
    return "au_ag_plasmonic_sers_hypothesis"


def _requirement(
    hypothesis_id: str,
    *,
    category: str,
    capability: str,
    rationale: str,
) -> ExperimentalRequirement:
    return ExperimentalRequirement(
        requirement_id=_stable_id(
            "sers_experimental_requirement",
            hypothesis_id,
            category,
            capability,
            _ADAPTER_VERSION,
        ),
        category=category,
        capability=capability,
        necessity="required",
        rationale=rationale,
        scientific_domain="sers_au_ag",
    )


@dataclass
class SERSFeasibilityAdapter:
    """Safe canonical feasibility adapter for SERS scientific hypotheses.

    The adapter deliberately does *not* turn a numerical FDTD problem into a
    scientific rejection.  It compiles the canonical feasibility artifacts and
    records which validator families remain unresolved.  The richer SERS
    multi-validator lifecycle remains the scientific-detail layer and can be
    projected alongside these canonical artifacts without changing authority.
    """

    adapter_id: str = "sers_au_ag"
    domain_profile_id: str = "sers_au_ag"
    adapter_version: str = _ADAPTER_VERSION

    def compile_scopes(self, intake: FeasibilityIntake) -> list[ScientificScope]:
        rows: list[ScientificScope] = []
        for hypothesis in intake.hypotheses:
            text = _text(hypothesis)
            route_kinds = _route_kinds(hypothesis)
            warnings = [
                "sers_scientific_hypothesis_requires_validator_specific_concretization"
            ]
            if "integrated_sers_experiment" in route_kinds:
                warnings.append("integrated_sers_outcome_requires_controlled_experiment")
            if "classical_em" in route_kinds:
                warnings.append(
                    "classical_em_result_must_not_be_used_as_whole_hypothesis_verdict"
                )
            if "unresolved_scientific_route" in route_kinds:
                warnings.append("validator_route_requires_interpretation")

            rows.append(
                ScientificScope(
                    schema_version="scientific-scope-v03",
                    scope_id=_stable_id(
                        "scientific_scope",
                        intake.intake_sha256,
                        hypothesis.hypothesis_id,
                        self.adapter_version,
                    ),
                    hypothesis_id=hypothesis.hypothesis_id,
                    catalyst_class="unknown",
                    hypothesis_level=hypothesis.hypothesis_type,
                    reaction="unknown",
                    system_class=_system_class(text),
                    scientific_domain="sers_au_ag",
                    process="surface_enhanced_raman_scattering",
                    components=_components(text),
                    structural_variables=_structural_variables(text),
                    independent_variables=_independent_variables(text),
                    dependent_observables=[
                        row.observable for row in hypothesis.predictions
                    ],
                    requires_candidate_concretization=True,
                    scope_confidence=(
                        "low"
                        if route_kinds == ["unresolved_scientific_route"]
                        else "medium"
                    ),
                    scope_warnings=warnings,
                    catalyst_class_rationale=(
                        "Catalyst vocabulary is not applicable to the SERS domain."
                    ),
                    hypothesis_level_rationale=(
                        "Preserve the scientific hypothesis type while requiring "
                        "validator-specific test/design concretization before execution."
                    ),
                    system_class_rationale=(
                        "SERS hypotheses are scoped as plasmonic substrate/analyte "
                        "systems rather than HER catalyst classes."
                    ),
                )
            )
        return rows

    def compile_specifications(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope],
    ) -> list[ValidationSpecification]:
        scope_by_id = {row.hypothesis_id: row for row in scopes}
        rows: list[ValidationSpecification] = []
        for hypothesis in intake.hypotheses:
            scope = scope_by_id[hypothesis.hypothesis_id]
            route_kinds = _route_kinds(hypothesis)

            concretization = [
                "decompose the scientific hypothesis into validator-specific claims before execution"
            ]
            comparisons: list[str] = []
            required_experimental: list[str] = []
            secondary: list[str] = []

            if "classical_em" in route_kinds:
                concretization.append(
                    "define source-resolved or explicitly labeled-assumption geometry, dielectric environment, illumination, and polarization for classical-EM validation"
                )
                comparisons.append(
                    "compare the proposed EM mechanism against an explicit geometry/material/optical-condition control"
                )
                secondary.extend(
                    [
                        "substrate electromagnetic spectral response",
                        "electromagnetic near-field response",
                    ]
                )
            if "molecular_spectroscopy" in route_kinds:
                concretization.append(
                    "define the analyte-specific molecular spectroscopy observable and excitation comparison"
                )
                required_experimental.append(
                    "excitation_dependent_molecular_spectroscopy_or_independent_validation_evidence"
                )
            if "surface_chemistry" in route_kinds:
                concretization.append(
                    "define the molecule-surface interaction observable and chemistry-aware comparator"
                )
                required_experimental.append(
                    "surface_chemistry_or_binding_characterization"
                )
            if "fabrication_process" in route_kinds:
                concretization.append(
                    "define fabrication realization and structural characterization criteria"
                )
                required_experimental.append(
                    "substrate_fabrication_and_structural_characterization"
                )
            if "integrated_sers_experiment" in route_kinds:
                concretization.append(
                    "define the controlled integrated SERS comparison, analyte conditions, excitation conditions, and measurement provenance"
                )
                comparisons.append(
                    "define an explicit baseline/control for the claimed integrated SERS contrast"
                )
                required_experimental.append("controlled_integrated_sers_measurement")

            if "unresolved_scientific_route" in route_kinds:
                concretization.append(
                    "resolve the validator family before launching computation or experiment"
                )

            success_patterns = [
                f"{row.observable}: expected_direction={row.expected_direction}"
                for row in hypothesis.predictions
            ]
            falsification_patterns = [
                f"{row.observable}: {row.falsifying_outcome}"
                for row in hypothesis.falsifiers
            ]

            rows.append(
                ValidationSpecification(
                    schema_version="validation-specification-v03",
                    specification_id=_stable_id(
                        "validation_specification",
                        hypothesis.hypothesis_id,
                        scope.scope_id,
                        tuple(route_kinds),
                        self.adapter_version,
                    ),
                    hypothesis_id=hypothesis.hypothesis_id,
                    source_scope_id=scope.scope_id,
                    scientific_domain="sers_au_ag",
                    validation_strategy="multi_validator",
                    requires_candidate_concretization=True,
                    controlled_variables=(
                        ["non_focal_acquisition_and_analyte_conditions"]
                        if "integrated_sers_experiment" in route_kinds
                        else []
                    ),
                    varied_variables=list(scope.independent_variables),
                    primary_observables=list(scope.dependent_observables),
                    secondary_observables=list(dict.fromkeys(secondary)),
                    required_comparisons=list(dict.fromkeys(comparisons)),
                    candidate_concretization_requirements=list(
                        dict.fromkeys(concretization)
                    ),
                    required_scientific_checks=route_kinds,
                    required_experimental_capabilities=list(
                        dict.fromkeys(required_experimental)
                    ),
                    success_patterns=success_patterns,
                    falsification_patterns=falsification_patterns,
                    next_actions=[
                        "route the hypothesis through the SERS multi-validator validation lifecycle",
                        "preserve numerical/model/evidence uncertainty instead of converting it into scientific rejection",
                        "use integrated SERS experiment evidence for whole-outcome assessment when the routed hypothesis requires it",
                    ],
                )
            )
        return rows

    def run_physics(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope],
        specifications: list[ValidationSpecification],
    ) -> list[PhysicsFeasibilityReport]:
        hypothesis_by_id = {row.hypothesis_id: row for row in intake.hypotheses}
        scope_by_id = {row.hypothesis_id: row for row in scopes}
        spec_by_id = {row.hypothesis_id: row for row in specifications}
        reports: list[PhysicsFeasibilityReport] = []

        for hypothesis_id, hypothesis in hypothesis_by_id.items():
            scope = scope_by_id[hypothesis_id]
            spec = spec_by_id[hypothesis_id]

            if hypothesis.semantic_gate_status == "human_review_required":
                reports.append(
                    PhysicsFeasibilityReport(
                        schema_version="physics-feasibility-report-v03",
                        report_id=_stable_id(
                            "physics_report",
                            intake.intake_sha256,
                            hypothesis_id,
                            "human_review_required",
                            self.adapter_version,
                        ),
                        source_intake_id=intake.intake_id,
                        source_intake_sha256=intake.intake_sha256,
                        source_scope_id=scope.scope_id,
                        source_validation_specification_id=spec.specification_id,
                        hypothesis_id=hypothesis_id,
                        scientific_domain="sers_au_ag",
                        disposition="human_review_required",
                        confidence="low",
                    )
                )
                continue

            checks: list[PhysicsCheckResult] = []
            for check_type in spec.required_scientific_checks:
                if check_type == "classical_em":
                    status = "requires_computation"
                    backend = "sers_classical_em_validation_lifecycle"
                    rationale = (
                        "Classical-EM evidence requires an explicit routed computational "
                        "model and readiness assessment. Numerical/model limitations must "
                        "remain uncertainty and never become an automatic scientific fail."
                    )
                else:
                    status = "unknown"
                    backend = "sers_multi_validator_validation_lifecycle"
                    rationale = (
                        "This scientific component is outside a classical-EM-only solver "
                        "and remains unresolved until its routed validator supplies "
                        "reviewable evidence."
                    )
                checks.append(
                    PhysicsCheckResult(
                        check_id=_stable_id(
                            "physics_check",
                            hypothesis_id,
                            check_type,
                            self.adapter_version,
                        ),
                        request_id=_stable_id(
                            "physics_request",
                            hypothesis_id,
                            check_type,
                            spec.specification_id,
                        ),
                        hypothesis_id=hypothesis_id,
                        check_type=check_type,
                        status=status,
                        basis="unavailable",
                        rationale=rationale,
                        scientific_domain="sers_au_ag",
                        backend_class=backend,
                    )
                )

            unresolved = sorted({row.check_type for row in checks})
            has_computation = any(row.status == "requires_computation" for row in checks)
            disposition = (
                "requires_computation" if has_computation else "insufficient_information"
            )
            next_computations = [
                (
                    "Resolve the routed classical_em claim through the SERS "
                    "validation lifecycle; do not promote underresolved/model-uncertain "
                    "FDTD output to hypothesis rejection or support."
                )
                for row in checks
                if row.check_type == "classical_em"
            ]

            reports.append(
                PhysicsFeasibilityReport(
                    schema_version="physics-feasibility-report-v03",
                    report_id=_stable_id(
                        "physics_report",
                        intake.intake_sha256,
                        hypothesis_id,
                        spec.specification_id,
                        disposition,
                        self.adapter_version,
                    ),
                    source_intake_id=intake.intake_id,
                    source_intake_sha256=intake.intake_sha256,
                    source_scope_id=scope.scope_id,
                    source_validation_specification_id=spec.specification_id,
                    hypothesis_id=hypothesis_id,
                    scientific_domain="sers_au_ag",
                    disposition=disposition,
                    confidence="low",
                    checks=checks,
                    blocking_checks=[],
                    unresolved_checks=unresolved,
                    not_applicable_checks=[],
                    next_required_computations=next_computations,
                )
            )

        return reports

    def run_experimental(
        self,
        intake: FeasibilityIntake,
        physics_reports: list[PhysicsFeasibilityReport],
        scopes: list[ScientificScope],
        specifications: list[ValidationSpecification],
    ) -> list[ExperimentalRealizabilityReport]:
        hypothesis_by_id = {row.hypothesis_id: row for row in intake.hypotheses}
        physics_by_id = {row.hypothesis_id: row for row in physics_reports}
        scope_by_id = {row.hypothesis_id: row for row in scopes}
        spec_by_id = {row.hypothesis_id: row for row in specifications}
        reports: list[ExperimentalRealizabilityReport] = []

        for hypothesis_id, hypothesis in hypothesis_by_id.items():
            physics = physics_by_id[hypothesis_id]
            scope = scope_by_id[hypothesis_id]
            spec = spec_by_id[hypothesis_id]
            route_kinds = list(spec.required_scientific_checks)

            if hypothesis.semantic_gate_status == "human_review_required":
                disposition = "human_review_required"
            elif "integrated_sers_experiment" in route_kinds:
                disposition = "conditionally_plausible"
            else:
                disposition = "insufficient_information"

            requirements: list[ExperimentalRequirement] = []
            checks: list[ExperimentalCheckResult] = []
            required_synthesis: list[str] = []
            required_characterization: list[str] = []
            required_performance: list[str] = []
            dominant_uncertainties = [
                "validator_specific_concretization",
            ]

            if "classical_em" in route_kinds:
                required_characterization.extend(
                    [
                        "substrate morphology and geometry characterization",
                        "substrate optical response characterization",
                    ]
                )
                dominant_uncertainties.extend(
                    [
                        "classical_em_model_form",
                        "classical_em_numerical_readiness",
                    ]
                )

            if "molecular_spectroscopy" in route_kinds:
                capability = "excitation-dependent molecular spectroscopy or independent validation evidence"
                requirements.append(
                    _requirement(
                        hypothesis_id,
                        category="molecular_spectroscopy",
                        capability=capability,
                        rationale=(
                            "Resolve analyte-specific spectral behavior separately from "
                            "substrate electromagnetic response."
                        ),
                    )
                )
                checks.append(
                    ExperimentalCheckResult(
                        check_id=_stable_id(
                            "experimental_check",
                            hypothesis_id,
                            "molecular_spectroscopy_validation",
                            self.adapter_version,
                        ),
                        hypothesis_id=hypothesis_id,
                        check_type="molecular_spectroscopy_validation",
                        status="conditional",
                        rationale=(
                            "The route is experimentally/literature testable but no "
                            "canonical result is inferred at adapter compilation time."
                        ),
                        scientific_domain="sers_au_ag",
                        precedent_status="not_assessed",
                        requirements=[requirements[-1]],
                    )
                )

            if "surface_chemistry" in route_kinds:
                capability = "surface chemistry or molecule-surface binding characterization"
                requirements.append(
                    _requirement(
                        hypothesis_id,
                        category="surface_chemistry",
                        capability=capability,
                        rationale=(
                            "Adsorption, association, orientation, and chemical enhancement "
                            "require chemistry-aware validation."
                        ),
                    )
                )
                checks.append(
                    ExperimentalCheckResult(
                        check_id=_stable_id(
                            "experimental_check",
                            hypothesis_id,
                            "surface_chemistry_validation",
                            self.adapter_version,
                        ),
                        hypothesis_id=hypothesis_id,
                        check_type="surface_chemistry_validation",
                        status="unknown",
                        rationale=(
                            "Surface-chemistry realizability is unresolved until a "
                            "chemistry-aware validator supplies evidence."
                        ),
                        scientific_domain="sers_au_ag",
                        precedent_status="not_assessed",
                        requirements=[requirements[-1]],
                    )
                )
                dominant_uncertainties.append("molecule_surface_interaction")

            if "fabrication_process" in route_kinds:
                capability = "substrate fabrication and structural characterization"
                required_synthesis.append(
                    "fabricate or obtain the concretized plasmonic substrate"
                )
                requirements.append(
                    _requirement(
                        hypothesis_id,
                        category="fabrication_process",
                        capability=capability,
                        rationale=(
                            "The fabrication/process claim must be realized and "
                            "characterized rather than inferred from Maxwell simulation."
                        ),
                    )
                )
                checks.append(
                    ExperimentalCheckResult(
                        check_id=_stable_id(
                            "experimental_check",
                            hypothesis_id,
                            "fabrication_realizability",
                            self.adapter_version,
                        ),
                        hypothesis_id=hypothesis_id,
                        check_type="fabrication_realizability",
                        status="unknown",
                        rationale=(
                            "Fabrication realizability and variability are not yet "
                            "established by canonical evidence."
                        ),
                        scientific_domain="sers_au_ag",
                        precedent_status="not_assessed",
                        requirements=[requirements[-1]],
                    )
                )
                dominant_uncertainties.append("fabrication_realization")

            if "integrated_sers_experiment" in route_kinds:
                capability = "controlled integrated SERS measurement"
                if not required_synthesis:
                    required_synthesis.append(
                        "fabricate or obtain the concretized plasmonic substrate"
                    )
                required_performance.append(
                    "controlled SERS response measurement under the predeclared comparison"
                )
                requirements.append(
                    _requirement(
                        hypothesis_id,
                        category="integrated_sers_experiment",
                        capability=capability,
                        rationale=(
                            "Whole integrated SERS outcomes require a controlled experiment; "
                            "classical-EM evidence alone is insufficient."
                        ),
                    )
                )
                checks.append(
                    ExperimentalCheckResult(
                        check_id=_stable_id(
                            "experimental_check",
                            hypothesis_id,
                            "integrated_sers_outcome_testability",
                            self.adapter_version,
                        ),
                        hypothesis_id=hypothesis_id,
                        check_type="integrated_sers_outcome_testability",
                        status="conditional",
                        rationale=(
                            "The outcome is testable after comparison/control/protocol "
                            "concretization; the adapter does not generate or execute a protocol."
                        ),
                        scientific_domain="sers_au_ag",
                        precedent_status="not_assessed",
                        requirements=[requirements[-1]],
                    )
                )
                dominant_uncertainties.extend(
                    [
                        "integrated_experiment_protocol",
                        "measurement_controls_and_replication",
                    ]
                )

            reports.append(
                ExperimentalRealizabilityReport(
                    schema_version="experimental-realizability-report-v03",
                    report_id=_stable_id(
                        "experimental_report",
                        intake.intake_sha256,
                        hypothesis_id,
                        physics.report_id,
                        disposition,
                        self.adapter_version,
                    ),
                    source_intake_id=intake.intake_id,
                    source_intake_sha256=intake.intake_sha256,
                    source_physics_report_id=physics.report_id,
                    source_scope_id=scope.scope_id,
                    source_validation_specification_id=spec.specification_id,
                    hypothesis_id=hypothesis_id,
                    scientific_domain="sers_au_ag",
                    disposition=disposition,
                    checks=checks,
                    synthesis_feasibility=(
                        "unknown" if required_synthesis else "unknown"
                    ),
                    structural_verifiability=(
                        "conditional"
                        if "classical_em" in route_kinds
                        or "fabrication_process" in route_kinds
                        else "unknown"
                    ),
                    active_site_verifiability="unknown",
                    performance_testability=(
                        "conditional"
                        if "integrated_sers_experiment" in route_kinds
                        else "unknown"
                    ),
                    precedent_status="not_assessed",
                    required_synthesis_capabilities=required_synthesis,
                    required_characterization=list(
                        dict.fromkeys(required_characterization)
                    ),
                    required_performance_tests=required_performance,
                    required_electrochemical_tests=[],
                    not_applicable_capabilities=[
                        "electrochemical_performance_testing"
                    ],
                    synthesis_complexity="unknown",
                    characterization_complexity="unknown",
                    relative_cost_burden="unknown",
                    relative_effort_burden="unknown",
                    dominant_uncertainties=list(dict.fromkeys(dominant_uncertainties)),
                )
            )

        return reports


SERS_FEASIBILITY_ADAPTER = SERSFeasibilityAdapter()
