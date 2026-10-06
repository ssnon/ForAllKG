from __future__ import annotations

import hashlib
import itertools
import json

from domains.sers.simulation_contracts import (
    SERSSimulationCompilationBundle,
    SERSSimulationSpec,
    SERSSimulationValidationBundle,
)
from domains.sers.validation_design_contracts import (
    SERSFDTDSimulationCase,
    SERSFDTDSimulationCaseBundle,
    SERSValidationDesign,
    SERSValidationDesignBundle,
    SERSValidationDesignOverride,
    SERSValidationDesignOverrideBundle,
    SERSValidationDesignRequest,
    SERSValidationDesignRequestBundle,
    SERSValidationDesignRequirement,
    SERSSourceSpectralConstraint,
)


_DESIGNER_VERSION = "sers-validation-designer-v0"
_MAX_CASES = 512


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _report_by_spec(validation: SERSSimulationValidationBundle):
    return {row.source_spec_id: row for row in validation.reports}


def _source_constraints(spec: SERSSimulationSpec) -> SERSSourceSpectralConstraint:
    return SERSSourceSpectralConstraint(
        baseline_wavelength_nm=spec.baseline_wavelength_nm,
        reported_resonance_wavelengths_nm=list(
            spec.reported_resonance_wavelengths_nm
        ),
    )


class SERSValidationDesignPlanner:
    """Lift low-level missing parameters into a multi-case validation design.

    v0 deliberately supports the first real use case only:
    core-shell nanorod + wavelength sweep. A different scientific geometry or
    study axis is blocked rather than silently coerced into this design.
    """

    designer_version = _DESIGNER_VERSION

    def plan(
        self,
        compilation: SERSSimulationCompilationBundle,
        validation: SERSSimulationValidationBundle,
    ) -> SERSValidationDesignRequestBundle:
        if validation.source_compilation_bundle_id != compilation.bundle_id:
            raise ValueError("validation/compilation bundle mismatch")

        reports = _report_by_spec(validation)
        requests: list[SERSValidationDesignRequest] = []

        for spec in compilation.specs:
            report = reports.get(spec.spec_id)
            if report is None:
                raise ValueError(
                    f"missing validation report for source spec {spec.spec_id}"
                )

            request = self._plan_one(spec, report)
            requests.append(request)

        return SERSValidationDesignRequestBundle(
            bundle_id=_stable_id(
                "sers_validation_design_request_bundle",
                compilation.bundle_id,
                validation.bundle_id,
                *(row.request_id for row in requests),
                self.designer_version,
            ),
            source_compilation_bundle_id=compilation.bundle_id,
            source_validation_bundle_id=validation.bundle_id,
            requests=requests,
        )

    def _plan_one(self, spec, report) -> SERSValidationDesignRequest:
        base = dict(
            request_id=_stable_id(
                "sers_validation_design_request",
                spec.spec_id,
                report.report_id,
                self.designer_version,
            ),
            hypothesis_id=spec.hypothesis_id,
            source_spec_id=spec.spec_id,
            source_validation_report_id=report.report_id,
            geometry_family=spec.geometry_family,
            study_kind=spec.study_kind,
            source_constraints=_source_constraints(spec),
        )

        if report.disposition == "runnable_shadow":
            return SERSValidationDesignRequest(
                **base,
                status="not_required",
                focal_axes=list(spec.study_axes),
                uncertainty_axes=[],
            )

        if report.disposition != "requires_concretization":
            return SERSValidationDesignRequest(
                **base,
                status="blocked",
                focal_axes=list(spec.study_axes),
                uncertainty_axes=[],
                blocked_reason=(
                    "Only requires_concretization specs are eligible for the "
                    "multi-case validation-design lane in v0; got "
                    f"{report.disposition!r}."
                ),
            )

        if not (
            spec.geometry_family == "core_shell_nanorod"
            and spec.study_kind == "wavelength_sweep"
        ):
            return SERSValidationDesignRequest(
                **base,
                status="blocked",
                focal_axes=list(spec.study_axes),
                uncertainty_axes=[],
                blocked_reason=(
                    "ValidationDesign v0 supports core_shell_nanorod + "
                    "wavelength_sweep only. Extend the design contract for "
                    "other scientific studies rather than coercing them."
                ),
            )

        issue_fields = {row.field for row in report.issues}
        requirements: list[SERSValidationDesignRequirement] = []

        geometry_fields = {
            "nanorod_length_nm",
            "nanorod_diameter_nm",
            "shell_thickness_nm",
        }
        if issue_fields & geometry_fields:
            requirements.append(
                SERSValidationDesignRequirement(
                    field="geometry_candidates",
                    rationale=(
                        "The target source does not uniquely specify all "
                        "core-shell nanorod dimensions. Supply one or more "
                        "correlated geometry candidates as explicit validation-"
                        "design assumptions rather than source facts."
                    ),
                    source_fields=sorted(issue_fields & geometry_fields),
                    unit="nm",
                )
            )

        if "sweep.values" in issue_fields:
            requirements.append(
                SERSValidationDesignRequirement(
                    field="wavelength_sweep_nm",
                    rationale=(
                        "The hypothesis asserts wavelength dependence, so the "
                        "validation unit is a sweep rather than one excitation "
                        "point. Source-derived spectral landmarks must remain "
                        "inside the explicit sweep."
                    ),
                    source_fields=["sweep.values"],
                    unit="nm",
                )
            )

        if "polarization" in issue_fields:
            requirements.append(
                SERSValidationDesignRequirement(
                    field="polarizations",
                    rationale=(
                        "Nanorod optical response is anisotropic. Treat "
                        "polarization as an uncertainty/design axis when the "
                        "source hypothesis does not fix it."
                    ),
                    source_fields=["polarization"],
                    allowed_values=[
                        "parallel_to_nanorod_long_axis",
                        "perpendicular_to_nanorod_long_axis",
                    ],
                )
            )

        if "surrounding_medium" in issue_fields:
            requirements.append(
                SERSValidationDesignRequirement(
                    field="surrounding_media",
                    rationale=(
                        "The optical environment is not source-resolved. "
                        "Supply one or more explicitly labelled screening "
                        "media instead of silently assuming one."
                    ),
                    source_fields=["surrounding_medium"],
                    allowed_values=["air", "water"],
                )
            )

        unhandled = issue_fields - geometry_fields - {
            "sweep.values",
            "polarization",
            "surrounding_medium",
        }
        if unhandled:
            return SERSValidationDesignRequest(
                **base,
                status="blocked",
                focal_axes=["wavelength"],
                uncertainty_axes=[
                    "geometry_candidate",
                    "polarization",
                    "surrounding_medium",
                ],
                blocked_reason=(
                    "ValidationDesign v0 cannot safely absorb validation "
                    "issues: " + ", ".join(sorted(unhandled))
                ),
            )

        if not requirements:
            return SERSValidationDesignRequest(
                **base,
                status="blocked",
                focal_axes=["wavelength"],
                uncertainty_axes=[],
                blocked_reason=(
                    "No design-level concretization requirement was derived."
                ),
            )

        return SERSValidationDesignRequest(
            **base,
            status="ready_for_design_input",
            focal_axes=["wavelength"],
            uncertainty_axes=[
                "geometry_candidate",
                "polarization",
                "surrounding_medium",
            ],
            requirements=requirements,
        )


class SERSValidationDesignBuilder:
    """Apply explicit validation-design choices and expand deterministic cases."""

    designer_version = _DESIGNER_VERSION

    def build(
        self,
        compilation: SERSSimulationCompilationBundle,
        requests: SERSValidationDesignRequestBundle,
        overrides: SERSValidationDesignOverrideBundle,
    ) -> tuple[SERSValidationDesignBundle, SERSFDTDSimulationCaseBundle]:
        if requests.source_compilation_bundle_id != compilation.bundle_id:
            raise ValueError("request/compilation bundle mismatch")

        specs = {row.spec_id: row for row in compilation.specs}
        reqs = {
            (row.hypothesis_id, row.source_spec_id): row
            for row in requests.requests
        }

        designs: list[SERSValidationDesign] = []
        cases: list[SERSFDTDSimulationCase] = []

        for override in overrides.overrides:
            key = (override.hypothesis_id, override.source_spec_id)
            request = reqs.get(key)
            if request is None:
                raise ValueError(
                    "validation-design override has no matching request: "
                    f"{key}"
                )
            if request.status != "ready_for_design_input":
                raise ValueError(
                    "validation-design override target is not ready_for_design_input"
                )
            spec = specs.get(override.source_spec_id)
            if spec is None:
                raise ValueError("source spec missing from compilation bundle")

            design = self._build_one(spec, request, override)
            designs.append(design)
            cases.extend(self._expand_cases(design))

        if len(cases) > _MAX_CASES:
            raise ValueError(
                f"validation design expands to {len(cases)} cases; "
                f"v0 safety limit is {_MAX_CASES}. Reduce explicit design axes."
            )

        design_bundle = SERSValidationDesignBundle(
            bundle_id=_stable_id(
                "sers_validation_design_bundle",
                requests.bundle_id,
                *(row.design_id for row in designs),
                self.designer_version,
            ),
            source_request_bundle_id=requests.bundle_id,
            designs=designs,
        )
        case_bundle = SERSFDTDSimulationCaseBundle(
            bundle_id=_stable_id(
                "sers_fdtd_case_bundle",
                design_bundle.bundle_id,
                *(row.case_id for row in cases),
            ),
            source_validation_design_bundle_id=design_bundle.bundle_id,
            cases=cases,
        )
        return design_bundle, case_bundle

    def _build_one(
        self,
        spec: SERSSimulationSpec,
        request: SERSValidationDesignRequest,
        override: SERSValidationDesignOverride,
    ) -> SERSValidationDesign:
        if spec.geometry_family != "core_shell_nanorod":
            raise ValueError("ValidationDesign v0 requires core_shell_nanorod")
        if spec.core_material is None or spec.shell_material is None:
            raise ValueError("source spec lacks explicit core/shell materials")

        required = {row.field for row in request.requirements}
        provided = {
            "geometry_candidates": bool(override.geometry_candidates),
            "wavelength_sweep_nm": bool(override.wavelength_sweep_nm),
            "polarizations": bool(override.polarizations),
            "surrounding_media": bool(override.surrounding_media),
        }
        missing = sorted(
            field for field in required if not provided.get(field, False)
        )
        if missing:
            raise ValueError(
                "validation design is incomplete; missing explicit axes: "
                + ", ".join(missing)
            )

        wrong_polarizations = sorted(
            value
            for value in override.polarizations
            if value not in {
                "parallel_to_nanorod_long_axis",
                "perpendicular_to_nanorod_long_axis",
            }
        )
        if wrong_polarizations:
            raise ValueError(
                "nanorod design received incompatible polarization(s): "
                + ", ".join(wrong_polarizations)
            )

        wavelengths = sorted(override.wavelength_sweep_nm)
        constraints = request.source_constraints
        mandatory = set(constraints.reported_resonance_wavelengths_nm)
        if constraints.baseline_wavelength_nm is not None:
            mandatory.add(constraints.baseline_wavelength_nm)

        missing_landmarks = sorted(mandatory - set(wavelengths))
        if missing_landmarks:
            raise ValueError(
                "wavelength sweep must preserve source-derived spectral "
                "landmarks: " + ", ".join(f"{v:g}" for v in missing_landmarks)
            )

        baseline = constraints.baseline_wavelength_nm
        if baseline is not None:
            if not any(value < baseline for value in wavelengths) or not any(
                value > baseline for value in wavelengths
            ):
                raise ValueError(
                    "wavelength sweep must bracket the source-derived baseline"
                )

        design_id = _stable_id(
            "sers_validation_design",
            request.request_id,
            override.model_dump(mode="json"),
            constraints.model_dump(mode="json"),
            self.designer_version,
        )

        return SERSValidationDesign(
            design_id=design_id,
            request_id=request.request_id,
            hypothesis_id=spec.hypothesis_id,
            source_spec_id=spec.spec_id,
            source_label=override.source_label,
            operator_note=override.operator_note,
            geometry_family="core_shell_nanorod",
            core_material=spec.core_material,
            shell_material=spec.shell_material,
            focal_axes=["wavelength"],
            uncertainty_axes=[
                "geometry_candidate",
                "polarization",
                "surrounding_medium",
            ],
            source_constraints=constraints,
            geometry_candidates=list(override.geometry_candidates),
            wavelength_sweep_nm=wavelengths,
            polarizations=list(override.polarizations),
            surrounding_media=list(override.surrounding_media),
        )

    def _expand_cases(
        self,
        design: SERSValidationDesign,
    ) -> list[SERSFDTDSimulationCase]:
        rows: list[SERSFDTDSimulationCase] = []
        for geometry, wavelength, polarization, medium in itertools.product(
            design.geometry_candidates,
            design.wavelength_sweep_nm,
            design.polarizations,
            design.surrounding_media,
        ):
            case_id = _stable_id(
                "sers_fdtd_case",
                design.design_id,
                geometry.candidate_id,
                wavelength,
                polarization,
                medium,
            )
            rows.append(
                SERSFDTDSimulationCase(
                    case_id=case_id,
                    design_id=design.design_id,
                    hypothesis_id=design.hypothesis_id,
                    source_spec_id=design.source_spec_id,
                    geometry_candidate_id=geometry.candidate_id,
                    geometry_family="core_shell_nanorod",
                    core_material=design.core_material,
                    shell_material=design.shell_material,
                    nanorod_length_nm=geometry.nanorod_length_nm,
                    nanorod_diameter_nm=geometry.nanorod_diameter_nm,
                    shell_thickness_nm=geometry.shell_thickness_nm,
                    excitation_wavelength_nm=wavelength,
                    polarization=polarization,
                    surrounding_medium=medium,
                    baseline_wavelength_nm=(
                        design.source_constraints.baseline_wavelength_nm
                    ),
                    reported_resonance_wavelengths_nm=list(
                        design.source_constraints.reported_resonance_wavelengths_nm
                    ),
                )
            )
        return rows
