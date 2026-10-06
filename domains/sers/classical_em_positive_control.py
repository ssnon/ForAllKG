from __future__ import annotations

import hashlib
import json

from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlQualification,
    SERSEMPositiveControlQualificationBundle,
    SERSEMPositiveControlQualificationIssue,
    SERSEMPositiveControlReference,
    SERSEMPositiveControlReferenceBundle,
)


_QUALIFIER_VERSION = "sers-em-positive-control-qualifier-v0"
_BACKEND_PROFILE = "meep_core_shell_nanorod_scattering_v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _issue(blocker: str, source_field: str, message: str):
    return SERSEMPositiveControlQualificationIssue(
        blocker=blocker,  # type: ignore[arg-type]
        source_field=source_field,
        message=message,
    )


class SERSClassicalEMPositiveControlQualifier:
    """Fail-closed compatibility gate for physical positive-control references.

    The current Meep backend represents a homogeneous-water/air, isolated,
    uniform-shell, spherocylinder-like Au/Ag nanorod under a normal-incidence
    plane wave with one explicit Ex/Ey polarization, and reports scattering
    cross section. A literature reference that differs on those axes is not
    silently treated as a validation target for this model.
    """

    qualifier_version = _QUALIFIER_VERSION
    backend_profile = _BACKEND_PROFILE

    def qualify(
        self,
        references: SERSEMPositiveControlReferenceBundle,
    ) -> SERSEMPositiveControlQualificationBundle:
        rows = [self._qualify_one(row) for row in references.references]
        return SERSEMPositiveControlQualificationBundle(
            bundle_id=_stable_id(
                "sers_em_positive_control_qualification_bundle",
                references.model_dump(mode="json"),
                self.backend_profile,
                self.qualifier_version,
            ),
            qualifications=rows,
            qualification_count=len(rows),
        )

    def _qualify_one(
        self,
        ref: SERSEMPositiveControlReference,
    ) -> SERSEMPositiveControlQualification:
        issues: list[SERSEMPositiveControlQualificationIssue] = []

        shape_ok = ref.particle_shape == "spherocylinder_like"
        shell_ok = (
            ref.shell_geometry == "uniform_thickness"
            and ref.uniform_shell_thickness_nm is not None
            and ref.core_length_nm is not None
            and ref.core_diameter_nm is not None
        )
        geometry_ok = shape_ok and shell_ok

        if not shape_ok:
            issues.append(_issue(
                "particle_shape_mismatch",
                "particle_shape",
                "Current backend models a spherocylinder-like nanorod; faceted, cuboid, or uncertain particle shapes require an explicit model extension.",
            ))
        if not shell_ok:
            issues.append(_issue(
                "shell_geometry_mismatch",
                "shell_geometry",
                "Current backend requires source-resolved core dimensions plus one uniform shell thickness; anisotropic/faceted/unknown shells are not coerced into a uniform shell.",
            ))

        environment_ok = ref.environment in {
            "homogeneous_water",
            "homogeneous_air",
        }
        if not environment_ok:
            issues.append(_issue(
                "environment_mismatch",
                "environment",
                "Current backend has a homogeneous background only; substrate-supported or otherwise heterogeneous reference environments require a model extension.",
            ))

        illumination_ok = ref.illumination == "normal_incidence_plane_wave"
        if not illumination_ok:
            issues.append(_issue(
                "illumination_mismatch",
                "illumination",
                "Current backend uses one normal-incidence plane wave; high-NA dark-field or ensemble illumination is not silently approximated.",
            ))

        polarization_ok = ref.polarization in {
            "parallel_to_nanorod_long_axis",
            "perpendicular_to_nanorod_long_axis",
        }
        if not polarization_ok:
            issues.append(_issue(
                "polarization_mismatch",
                "polarization",
                "Current backend validates one explicit longitudinal/transverse polarization at a time; unpolarized or unreported polarization requires an explicit averaging/measurement model.",
            ))

        observable_ok = ref.observable == "single_particle_scattering_spectrum"
        if not observable_ok:
            issues.append(_issue(
                "observable_mismatch",
                "observable",
                "Current backend reports single-particle scattering cross section; extinction or ensemble spectra require a distinct observable/ensemble model.",
            ))

        peak_ok = ref.reference_peak_wavelength_nm is not None
        if not peak_ok:
            issues.append(_issue(
                "reference_peak_missing",
                "reference_peak_wavelength_nm",
                "A source-resolved scattering peak wavelength is required before spectral positive-control comparison can be planned.",
            ))

        blockers = list(dict.fromkeys(row.blocker for row in issues))
        insufficient_only = bool(blockers) and set(blockers) <= {
            "reference_peak_missing",
        }
        if not blockers:
            status = "qualified_for_current_backend"
        elif insufficient_only:
            status = "insufficient_reference_detail"
        else:
            status = "requires_model_extension"

        return SERSEMPositiveControlQualification(
            qualification_id=_stable_id(
                "sers_em_positive_control_qualification",
                ref.control_id,
                ref.model_dump(mode="json"),
                self.backend_profile,
                self.qualifier_version,
            ),
            control_id=ref.control_id,
            source_label=ref.source_label,
            source_locator=ref.source_locator,
            status=status,  # type: ignore[arg-type]
            blockers=blockers,
            issues=issues,
            backend_geometry_representable=geometry_ok,
            backend_environment_representable=environment_ok,
            backend_illumination_representable=illumination_ok,
            backend_polarization_representable=polarization_ok,
            backend_observable_representable=observable_ok,
            spectral_reference_available=peak_ok,
            positive_control_execution_candidate=(status == "qualified_for_current_backend"),
        )
