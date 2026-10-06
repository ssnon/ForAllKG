from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from domains.sers.simulation_concretization_contracts import (
    SERSConcretizationOverride,
    SERSConcretizationOverrideBundle,
    SERSConcretizationRecord,
    SERSConcretizationRequest,
    SERSConcretizationRequestBundle,
    SERSConcretizationRequirement,
    SERSSimulationConcretizationBundle,
)
from domains.sers.simulation_contracts import (
    SERSParameterProvenance,
    SERSSimulationCompilationBundle,
    SERSSimulationSpec,
    SERSSimulationValidationBundle,
    SERSSimulationValidationReport,
    SERSSweepDefinition,
)


_CONCRETIZER_VERSION = "sers-simulation-concretizer-v0"


_FILLABLE_FIELDS = {
    "particle_radius_nm",
    "gap_nm",
    "nanorod_length_nm",
    "nanorod_diameter_nm",
    "shell_thickness_nm",
    "excitation_wavelength_nm",
    "polarization",
    "surrounding_medium",
    "sweep.values",
}


_ALLOWED_VALUES = {
    "polarization": [
        "parallel_to_dimer_axis",
        "perpendicular_to_dimer_axis",
        "parallel_to_nanorod_long_axis",
        "perpendicular_to_nanorod_long_axis",
    ],
    "surrounding_medium": ["air", "water"],
}


_UNITS = {
    "particle_radius_nm": "nm",
    "gap_nm": "nm",
    "nanorod_length_nm": "nm",
    "nanorod_diameter_nm": "nm",
    "shell_thickness_nm": "nm",
    "excitation_wavelength_nm": "nm",
    "sweep.values": "nm when sweep.parameter=excitation_wavelength_nm; "
    "nm when sweep.parameter=gap_nm",
}


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


def _requirement_from_issue(
    issue,
) -> SERSConcretizationRequirement | None:
    field = issue.field
    if field not in _FILLABLE_FIELDS:
        return None
    return SERSConcretizationRequirement(
        field=field,
        reason_code=issue.code,
        rationale=issue.message,
        unit=_UNITS.get(field),
        allowed_values=list(_ALLOWED_VALUES.get(field, [])),
    )


def _blocked_reason(report: SERSSimulationValidationReport) -> str:
    if report.disposition == "unsupported":
        return (
            "The simulation intent requires a geometry/feature that this "
            "FDTD contract does not represent. Extend the simulation backend "
            "rather than filling parameters manually."
        )
    if report.disposition == "not_applicable":
        return (
            "The hypothesis is not routed to FDTD; use the preserved non-FDTD "
            "validation route."
        )
    if report.disposition == "requires_interpretation":
        return (
            "Scientific routing is unresolved; interpretation must be fixed "
            "before numerical concretization."
        )
    if report.disposition == "invalid":
        return (
            "The compiled simulation intent is internally invalid; repair the "
            "upstream contract rather than overriding it."
        )
    return "The validation report is not eligible for operator concretization."


class SERSConcretizationPlanner:
    """Translate validation gaps into an explicit operator-input contract."""

    def plan(
        self,
        compilation: SERSSimulationCompilationBundle,
        validation: SERSSimulationValidationBundle,
    ) -> SERSConcretizationRequestBundle:
        if validation.source_compilation_bundle_id != compilation.bundle_id:
            raise ValueError("validation/compilation bundle mismatch")

        spec_by_id = {row.spec_id: row for row in compilation.specs}
        reports = validation.reports
        requests: list[SERSConcretizationRequest] = []

        for report in reports:
            spec = spec_by_id.get(report.source_spec_id)
            if spec is None:
                raise ValueError(
                    "validation report references unknown simulation spec: "
                    f"{report.source_spec_id}"
                )
            if spec.hypothesis_id != report.hypothesis_id:
                raise ValueError("validation/spec hypothesis mismatch")

            if report.disposition == "runnable_shadow":
                status = "not_required"
                requirements = []
                blocked_reason = None
            elif report.disposition == "requires_concretization":
                requirements = [
                    requirement
                    for issue in report.issues
                    if (requirement := _requirement_from_issue(issue))
                    is not None
                ]
                if not requirements:
                    status = "blocked_nonconcretizable"
                    blocked_reason = (
                        "The validator requires concretization, but none of the "
                        "reported gaps are fillable by the v0 operator contract."
                    )
                else:
                    status = "ready_for_operator_input"
                    blocked_reason = None
            else:
                status = "blocked_nonconcretizable"
                requirements = []
                blocked_reason = _blocked_reason(report)

            requests.append(
                SERSConcretizationRequest(
                    request_id=_stable_id(
                        "sers_concretization_request",
                        spec.spec_id,
                        report.report_id,
                        status,
                        *(row.model_dump(mode="json") for row in requirements),
                    ),
                    hypothesis_id=spec.hypothesis_id,
                    source_spec_id=spec.spec_id,
                    source_validation_report_id=report.report_id,
                    status=status,  # type: ignore[arg-type]
                    geometry_family=spec.geometry_family,
                    study_kind=spec.study_kind,
                    baseline_wavelength_nm=spec.baseline_wavelength_nm,
                    requirements=requirements,
                    blocked_reason=blocked_reason,
                )
            )

        return SERSConcretizationRequestBundle(
            bundle_id=_stable_id(
                "sers_concretization_request_bundle",
                compilation.bundle_id,
                validation.bundle_id,
                *(row.request_id for row in requests),
            ),
            source_compilation_bundle_id=compilation.bundle_id,
            source_validation_bundle_id=validation.bundle_id,
            requests=requests,
        )


class SERSOperatorConcretizer:
    """Apply explicit operator values without rewriting hypothesis-derived data."""

    concretizer_version = _CONCRETIZER_VERSION

    def apply(
        self,
        compilation: SERSSimulationCompilationBundle,
        requests: SERSConcretizationRequestBundle,
        overrides: SERSConcretizationOverrideBundle,
    ) -> SERSSimulationConcretizationBundle:
        if requests.source_compilation_bundle_id != compilation.bundle_id:
            raise ValueError("concretization request/compilation mismatch")

        spec_by_id = {row.spec_id: row for row in compilation.specs}
        request_by_key = {
            (row.hypothesis_id, row.source_spec_id): row
            for row in requests.requests
        }
        override_by_key = {
            (row.hypothesis_id, row.source_spec_id): row
            for row in overrides.overrides
        }

        unknown = sorted(set(override_by_key) - set(request_by_key))
        if unknown:
            raise ValueError(
                "override references unknown concretization request target: "
                + repr(unknown)
            )

        output_specs: list[SERSSimulationSpec] = []
        records: list[SERSConcretizationRecord] = []

        for request in requests.requests:
            spec = spec_by_id.get(request.source_spec_id)
            if spec is None:
                raise ValueError(
                    f"concretization request references unknown spec {request.source_spec_id}"
                )

            override = override_by_key.get(
                (request.hypothesis_id, request.source_spec_id)
            )

            if request.status != "ready_for_operator_input":
                if override is not None:
                    raise ValueError(
                        "override supplied for non-concretizable request: "
                        f"{request.request_id} status={request.status}"
                    )
                output_specs.append(spec)
                continue

            if override is None:
                output_specs.append(spec)
                continue

            concretized, applied_fields = self._apply_one(
                spec,
                request,
                override,
            )
            output_specs.append(concretized)
            records.append(
                SERSConcretizationRecord(
                    record_id=_stable_id(
                        "sers_concretization_record",
                        request.request_id,
                        override.model_dump(mode="json"),
                        concretized.spec_id,
                        self.concretizer_version,
                    ),
                    request_id=request.request_id,
                    hypothesis_id=spec.hypothesis_id,
                    source_spec_id=spec.spec_id,
                    output_spec_id=concretized.spec_id,
                    source_label=override.source_label,
                    applied_fields=applied_fields,
                    operator_note=override.operator_note,
                )
            )

        return SERSSimulationConcretizationBundle(
            bundle_id=_stable_id(
                "sers_concretization_bundle",
                requests.bundle_id,
                *(row.spec_id for row in output_specs),
                *(row.record_id for row in records),
                self.concretizer_version,
            ),
            source_request_bundle_id=requests.bundle_id,
            records=records,
            specs=output_specs,
        )

    def _apply_one(
        self,
        spec: SERSSimulationSpec,
        request: SERSConcretizationRequest,
        override: SERSConcretizationOverride,
    ) -> tuple[SERSSimulationSpec, list[str]]:
        if override.hypothesis_id != spec.hypothesis_id:
            raise ValueError("override/spec hypothesis mismatch")
        if override.source_spec_id != spec.spec_id:
            raise ValueError("override/source spec mismatch")

        requested = {row.field for row in request.requirements}
        update = deepcopy(spec.model_dump(mode="python"))
        provenance = list(spec.parameter_provenance)
        applied_fields: list[str] = []

        scalar_fields = (
            "particle_radius_nm",
            "gap_nm",
            "nanorod_length_nm",
            "nanorod_diameter_nm",
            "shell_thickness_nm",
            "excitation_wavelength_nm",
            "polarization",
            "surrounding_medium",
        )

        for field in scalar_fields:
            value = getattr(override, field)
            if value is None:
                continue
            if field not in requested:
                raise ValueError(
                    f"operator override for {field!r} was not requested by "
                    "deterministic validation"
                )
            if getattr(spec, field) is not None:
                raise ValueError(
                    f"operator override may not replace existing {field!r}"
                )
            update[field] = value
            applied_fields.append(field)
            provenance.append(
                SERSParameterProvenance(
                    parameter=field,
                    source_kind="operator_override",
                    source_field="concretization_override",
                    source_text=(
                        f"{override.source_label}: {field}={value}"
                    ),
                    derivation=None,
                )
            )

        if override.sweep_values is not None:
            field = "sweep.values"
            if not override.sweep_values:
                raise ValueError(
                    "operator sweep_values must contain explicit values; "
                    "an empty template is not a concretization"
                )
            if field not in requested:
                raise ValueError(
                    "operator sweep_values were not requested by deterministic "
                    "validation"
                )
            if spec.sweep is None:
                raise ValueError("cannot fill sweep.values without sweep definition")
            if spec.sweep.values:
                raise ValueError(
                    "operator override may not replace existing sweep.values"
                )
            update["sweep"] = SERSSweepDefinition(
                parameter=spec.sweep.parameter,
                values=list(override.sweep_values),
            ).model_dump(mode="python")
            applied_fields.append(field)
            provenance.append(
                SERSParameterProvenance(
                    parameter=field,
                    source_kind="operator_override",
                    source_field="concretization_override",
                    source_text=(
                        f"{override.source_label}: sweep_values="
                        + json.dumps(override.sweep_values)
                    ),
                    derivation=None,
                )
            )

        if not applied_fields:
            raise ValueError("concretization override did not fill any requested field")

        update["parameter_provenance"] = [
            row.model_dump(mode="python")
            for row in provenance
        ]
        update["spec_id"] = _stable_id(
            "sers_simulation_spec",
            spec.spec_id,
            override.model_dump(mode="json"),
            self.concretizer_version,
        )

        return SERSSimulationSpec.model_validate(update), sorted(applied_fields)
