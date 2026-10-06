from __future__ import annotations

import hashlib
import json

from domains.sers.simulation_contracts import (
    SERSSimulationCompilationBundle,
    SERSSimulationSpec,
    SERSSimulationValidationBundle,
    SERSSimulationValidationIssue,
    SERSSimulationValidationReport,
)


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


def _issue(
    *,
    code: str,
    field: str,
    message: str,
    severity: str = "error",
) -> SERSSimulationValidationIssue:
    return SERSSimulationValidationIssue(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        field=field,
        message=message,
    )


def _require_positive(
    issues: list[SERSSimulationValidationIssue],
    *,
    field: str,
    value: float | None,
) -> None:
    if value is None:
        issues.append(
            _issue(
                code="MISSING_NUMERIC_PARAMETER",
                field=field,
                message=(
                    f"{field} is required for a runnable shadow simulation "
                    "and was not invented by the compiler."
                ),
            )
        )
    elif value <= 0:
        issues.append(
            _issue(
                code="INVALID_NUMERIC_PARAMETER",
                field=field,
                message=f"{field} must be > 0.",
            )
        )


class SERSDeterministicSimulationValidator:
    """Fail-closed deterministic validator for SERS simulation intent.

    Validation determines whether the FDTD-addressable subclaim is concrete
    enough to hand to a later solver backend. A partial/not-applicable result is
    never converted into a scientific rejection of the original hypothesis.
    """

    def validate_bundle(
        self,
        bundle: SERSSimulationCompilationBundle,
    ) -> SERSSimulationValidationBundle:
        reports = [self.validate(spec) for spec in bundle.specs]
        return SERSSimulationValidationBundle(
            bundle_id=_stable_id(
                "sers_simulation_validation_bundle",
                bundle.bundle_id,
                *(row.report_id for row in reports),
            ),
            source_compilation_bundle_id=bundle.bundle_id,
            reports=reports,
        )

    def validate(
        self,
        spec: SERSSimulationSpec,
    ) -> SERSSimulationValidationReport:
        issues: list[SERSSimulationValidationIssue] = []

        if spec.domain_profile_id != "sers_au_ag":
            issues.append(
                _issue(
                    code="INVALID_DOMAIN_PROFILE",
                    field="domain_profile_id",
                    message=(
                        "SERS FDTD v0.2 accepts only the sers_au_ag "
                        "domain profile."
                    ),
                )
            )

        # Applicability is an upstream scientific-routing decision. Do not
        # manufacture geometry errors for a hypothesis that is deliberately not
        # routed to FDTD.
        if spec.fdtd_applicability == "not_applicable":
            issues.append(
                _issue(
                    code="FDTD_NOT_APPLICABLE",
                    field="fdtd_applicability",
                    severity="warning",
                    message=(
                        "The hypothesis has no FDTD-computable subclaim. "
                        "Preserve it for a non-FDTD validation route."
                    ),
                )
            )
            disposition = "not_applicable"
            return self._report(spec, disposition, issues)

        if spec.fdtd_applicability == "requires_interpretation":
            issues.append(
                _issue(
                    code="FDTD_APPLICABILITY_REQUIRES_INTERPRETATION",
                    field="fdtd_applicability",
                    severity="warning",
                    message=(
                        "Deterministic routing could not isolate a sufficiently "
                        "explicit electromagnetic subclaim."
                    ),
                )
            )
            disposition = "requires_interpretation"
            return self._report(spec, disposition, issues)

        if not spec.fdtd_subclaim_ids:
            issues.append(
                _issue(
                    code="INVALID_FDTD_SUBCLAIM_BINDING",
                    field="fdtd_subclaim_ids",
                    message=(
                        "direct/partial FDTD applicability requires at least "
                        "one FDTD-computable subclaim."
                    ),
                )
            )

        if spec.fdtd_applicability == "partial" and not spec.non_fdtd_subclaim_ids:
            issues.append(
                _issue(
                    code="INVALID_PARTIAL_APPLICABILITY",
                    field="non_fdtd_subclaim_ids",
                    message=(
                        "partial FDTD applicability must retain at least one "
                        "non-FDTD subclaim."
                    ),
                )
            )

        if spec.unsupported_features:
            for feature in sorted(set(spec.unsupported_features)):
                issues.append(
                    _issue(
                        code="UNSUPPORTED_FEATURE",
                        field="unsupported_features",
                        message=(
                            f"v0.1 does not yet represent feature {feature!r}; "
                            "preserve the hypothesis and extend the geometry "
                            "contract instead of approximating it silently."
                        ),
                    )
                )

        if spec.geometry_family is None:
            issues.append(
                _issue(
                    code="MISSING_GEOMETRY_FAMILY",
                    field="geometry_family",
                    message=(
                        "An explicit supported geometry is required. v0.1 "
                        "represents sphere_dimer and core_shell_nanorod."
                    ),
                )
            )

        if spec.geometry_family == "sphere_dimer":
            if spec.left_material is None or spec.right_material is None:
                issues.append(
                    _issue(
                        code="MISSING_MATERIAL_PAIR",
                        field="left_material/right_material",
                        message=(
                            "Both dimer materials must be explicit in the "
                            "compiled simulation intent."
                        ),
                    )
                )
            _require_positive(
                issues,
                field="particle_radius_nm",
                value=spec.particle_radius_nm,
            )

            if spec.study_kind == "single_candidate":
                _require_positive(
                    issues,
                    field="gap_nm",
                    value=spec.gap_nm,
                )

        elif spec.geometry_family == "core_shell_nanorod":
            if spec.core_material is None or spec.shell_material is None:
                issues.append(
                    _issue(
                        code="MISSING_CORE_SHELL_MATERIALS",
                        field="core_material/shell_material",
                        message=(
                            "Core and shell material identities must be explicit; "
                            "Au@Ag is interpreted as Au core / Ag shell."
                        ),
                    )
                )
            _require_positive(
                issues,
                field="nanorod_length_nm",
                value=spec.nanorod_length_nm,
            )
            _require_positive(
                issues,
                field="nanorod_diameter_nm",
                value=spec.nanorod_diameter_nm,
            )
            _require_positive(
                issues,
                field="shell_thickness_nm",
                value=spec.shell_thickness_nm,
            )
            if (
                spec.nanorod_length_nm is not None
                and spec.nanorod_diameter_nm is not None
                and spec.nanorod_length_nm > 0
                and spec.nanorod_diameter_nm > 0
                and spec.nanorod_length_nm <= spec.nanorod_diameter_nm
            ):
                issues.append(
                    _issue(
                        code="INVALID_NANOROD_ASPECT",
                        field="nanorod_length_nm/nanorod_diameter_nm",
                        message=(
                            "core-shell nanorod core length must exceed core "
                            "diameter. In v0.2 these fields denote core "
                            "end-to-end length and core diameter before shell "
                            "thickness is added."
                        ),
                    )
                )

        if spec.baseline_wavelength_nm is not None and spec.baseline_wavelength_nm <= 0:
            issues.append(
                _issue(
                    code="INVALID_NUMERIC_PARAMETER",
                    field="baseline_wavelength_nm",
                    message="baseline_wavelength_nm must be > 0.",
                )
            )

        if spec.study_kind == "single_candidate":
            _require_positive(
                issues,
                field="excitation_wavelength_nm",
                value=spec.excitation_wavelength_nm,
            )
        elif spec.study_kind in {"parameter_sweep", "wavelength_sweep"}:
            expected_parameter = (
                "excitation_wavelength_nm"
                if spec.study_kind == "wavelength_sweep"
                else "gap_nm"
            )
            if spec.sweep is None:
                issues.append(
                    _issue(
                        code="MISSING_SWEEP_DEFINITION",
                        field="sweep",
                        message=(
                            f"{spec.study_kind} requires an explicit sweep "
                            "definition."
                        ),
                    )
                )
            else:
                if spec.sweep.parameter != expected_parameter:
                    issues.append(
                        _issue(
                            code="INVALID_SWEEP_PARAMETER",
                            field="sweep.parameter",
                            message=(
                                f"{spec.study_kind} requires sweep.parameter="
                                f"{expected_parameter!r}."
                            ),
                        )
                    )
                minimum_points = (
                    3 if spec.study_kind == "wavelength_sweep" else 2
                )
                if len(spec.sweep.values) < minimum_points:
                    issues.append(
                        _issue(
                            code="MISSING_SWEEP_VALUES",
                            field="sweep.values",
                            message=(
                                f"At least {minimum_points} explicit sweep "
                                "values are required for this study. v0.2 "
                                "never invents sweep bounds."
                            ),
                        )
                    )
                else:
                    if any(value <= 0 for value in spec.sweep.values):
                        issues.append(
                            _issue(
                                code="INVALID_SWEEP_VALUE",
                                field="sweep.values",
                                message="All sweep values must be > 0.",
                            )
                        )
                    if len(spec.sweep.values) != len(set(spec.sweep.values)):
                        issues.append(
                            _issue(
                                code="DUPLICATE_SWEEP_VALUE",
                                field="sweep.values",
                                message="Sweep values must be unique.",
                            )
                        )
                    if (
                        spec.study_kind == "wavelength_sweep"
                        and spec.baseline_wavelength_nm is not None
                    ):
                        baseline = spec.baseline_wavelength_nm
                        values = spec.sweep.values
                        if not any(abs(value - baseline) <= 1e-9 for value in values):
                            issues.append(
                                _issue(
                                    code="MISSING_BASELINE_SWEEP_POINT",
                                    field="sweep.values",
                                    message=(
                                        "The explicit wavelength sweep must "
                                        "include baseline_wavelength_nm so the "
                                        "substrate-side comparison anchor is "
                                        "actually evaluated."
                                    ),
                                )
                            )
                        if not (
                            any(value < baseline for value in values)
                            and any(value > baseline for value in values)
                        ):
                            issues.append(
                                _issue(
                                    code="MISSING_BASELINE_BRACKETING",
                                    field="sweep.values",
                                    message=(
                                        "A wavelength-shift study must include "
                                        "at least one wavelength below and one "
                                        "above the baseline comparison point."
                                    ),
                                )
                            )

        if spec.polarization is None:
            issues.append(
                _issue(
                    code="MISSING_EXCITATION_CONDITION",
                    field="polarization",
                    message=(
                        "Polarization is required because plasmonic response "
                        "depends on field orientation; v0.2 does not assume it."
                    ),
                )
            )
        elif (
            spec.geometry_family == "sphere_dimer"
            and "dimer_axis" not in spec.polarization
        ):
            issues.append(
                _issue(
                    code="INVALID_POLARIZATION_GEOMETRY",
                    field="polarization",
                    message=(
                        "sphere_dimer requires a dimer-axis polarization "
                        "definition."
                    ),
                )
            )
        elif (
            spec.geometry_family == "core_shell_nanorod"
            and "nanorod_long_axis" not in spec.polarization
        ):
            issues.append(
                _issue(
                    code="INVALID_POLARIZATION_GEOMETRY",
                    field="polarization",
                    message=(
                        "core_shell_nanorod requires a nanorod-long-axis "
                        "polarization definition."
                    ),
                )
            )

        if spec.surrounding_medium is None:
            issues.append(
                _issue(
                    code="MISSING_ENVIRONMENT",
                    field="surrounding_medium",
                    message=(
                        "The surrounding optical medium must be explicit; "
                        "v0.2 does not silently assume air or water."
                    ),
                )
            )

        codes = {row.code for row in issues}
        if "UNSUPPORTED_FEATURE" in codes:
            disposition = "unsupported"
        elif any(code.startswith("INVALID_") for code in codes):
            disposition = "invalid"
        elif any(row.severity == "error" for row in issues):
            disposition = "requires_concretization"
        else:
            disposition = "runnable_shadow"

        return self._report(spec, disposition, issues)

    @staticmethod
    def _report(
        spec: SERSSimulationSpec,
        disposition: str,
        issues: list[SERSSimulationValidationIssue],
    ) -> SERSSimulationValidationReport:
        return SERSSimulationValidationReport(
            report_id=_stable_id(
                "sers_simulation_validation",
                spec.spec_id,
                disposition,
                *(row.model_dump(mode="json") for row in issues),
            ),
            source_spec_id=spec.spec_id,
            hypothesis_id=spec.hypothesis_id,
            disposition=disposition,  # type: ignore[arg-type]
            issues=issues,
        )
