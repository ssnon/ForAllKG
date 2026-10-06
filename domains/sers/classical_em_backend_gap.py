from __future__ import annotations

import hashlib
import json

from domains.sers.classical_em_backend_gap_contracts import (
    SERSEMBackendGapAssessment,
    SERSEMBackendGapAssessmentBundle,
)
from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlQualification,
    SERSEMPositiveControlQualificationBundle,
)


_ASSESSOR_VERSION = "sers-em-backend-gap-assessor-v0"

_GAP_META = {
    "particle_shape_mismatch": (
        "backend_capability_gap",
        "particle_shape",
        "faceted_or_generalized_particle_geometry",
    ),
    "shell_geometry_mismatch": (
        "backend_capability_gap",
        "shell_geometry",
        "nonuniform_or_faceted_shell_geometry",
    ),
    "environment_mismatch": (
        "backend_capability_gap",
        "environment",
        "substrate_and_heterogeneous_environment_model",
    ),
    "illumination_mismatch": (
        "backend_capability_gap",
        "illumination",
        "dark_field_or_high_na_illumination_model",
    ),
    "polarization_mismatch": (
        "backend_capability_gap",
        "polarization",
        "polarization_aggregation_or_measurement_model",
    ),
    "observable_mismatch": (
        "backend_capability_gap",
        "observable",
        "additional_optical_observable_model",
    ),
    "reference_peak_missing": (
        "reference_detail_gap",
        "reference_detail",
        "curate_source_resolved_reference_peak",
    ),
}


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _status_counts(rows: list[SERSEMPositiveControlQualification]) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        values[row.status] = values.get(row.status, 0) + 1
    return values


class SERSClassicalEMBackendGapAssessor:
    """Aggregate qualified positive-control mismatches without authorizing fixes.

    Repetition within one source is deliberately separated from cross-source
    recurrence. A backend extension becomes a *candidate* only after at least
    two independent source locators exhibit the same capability gap.
    """

    assessor_version = _ASSESSOR_VERSION

    def assess(
        self,
        qualifications: SERSEMPositiveControlQualificationBundle,
    ) -> SERSEMBackendGapAssessmentBundle:
        rows = list(qualifications.qualifications)
        source_locators = sorted({row.source_locator for row in rows})
        status_counts = _status_counts(rows)

        by_blocker: dict[str, list[SERSEMPositiveControlQualification]] = {}
        for row in rows:
            for blocker in row.blockers:
                by_blocker.setdefault(blocker, []).append(row)

        gaps: list[SERSEMBackendGapAssessment] = []
        for blocker in sorted(by_blocker):
            affected = by_blocker[blocker]
            gap_kind, model_axis, extension_target = _GAP_META[blocker]
            affected_controls = sorted({row.control_id for row in affected})
            affected_sources = sorted({row.source_locator for row in affected})
            unlock = sorted(
                row.control_id
                for row in affected
                if set(row.blockers) == {blocker}
            )

            if gap_kind == "reference_detail_gap":
                priority = "reference_curation_required"
                interpretation = (
                    "This is missing source detail, not evidence that the numerical "
                    "backend requires a new capability. Curate the reference or choose "
                    "another control."
                )
            elif len(affected_sources) < 2:
                priority = "insufficient_independent_source_support"
                interpretation = (
                    "The mismatch recurs in one or more controls but is supported by "
                    "fewer than two independent source locators. Do not prioritize a "
                    "backend extension from within-source repetition alone."
                )
            else:
                priority = "cross_source_gap_candidate"
                if unlock:
                    interpretation = (
                        "This backend gap recurs across independent sources and fixing "
                        "this axis alone would unlock at least one blocked control for "
                        "positive-control execution planning."
                    )
                else:
                    interpretation = (
                        "This backend gap recurs across independent sources, but every "
                        "affected control remains blocked on additional axes. Treat it "
                        "as a joint model-design candidate rather than an isolated fix."
                    )

            gaps.append(SERSEMBackendGapAssessment(
                blocker=blocker,  # type: ignore[arg-type]
                gap_kind=gap_kind,  # type: ignore[arg-type]
                model_axis=model_axis,  # type: ignore[arg-type]
                extension_target=extension_target,
                affected_control_ids=affected_controls,
                affected_source_locators=affected_sources,
                affected_control_count=len(affected_controls),
                independent_source_count=len(affected_sources),
                single_axis_unlock_control_ids=unlock,
                single_axis_unlock_count=len(unlock),
                priority_status=priority,  # type: ignore[arg-type]
                interpretation=interpretation,
            ))

        model_extension_rows = [
            row for row in rows if row.status == "requires_model_extension"
        ]
        if model_extension_rows:
            shared = set(model_extension_rows[0].blockers)
            for row in model_extension_rows[1:]:
                shared &= set(row.blockers)
            shared_blockers = sorted(shared)
        else:
            shared_blockers = []

        cross_source = [
            row for row in gaps
            if row.gap_kind == "backend_capability_gap"
            and row.priority_status == "cross_source_gap_candidate"
        ]
        targeted = [row for row in cross_source if row.single_axis_unlock_count > 0]
        backend_gaps = [row for row in gaps if row.gap_kind == "backend_capability_gap"]

        if targeted:
            decision = "targeted_extension_candidate_available"
            actions = [
                "review_targeted_cross_source_extension_candidate",
                "preserve_positive_control_source_diversity",
                "predeclare_model_validation_criterion_before_execution",
            ]
        elif cross_source:
            decision = "cross_source_gaps_require_joint_design"
            actions = [
                "design_minimal_joint_backend_extension",
                "avoid_single_axis_patch_that_unlocks_no_control",
                "predeclare_model_validation_criterion_before_execution",
            ]
        elif backend_gaps:
            decision = "collect_more_independent_controls"
            actions = [
                "collect_additional_independent_positive_controls",
                "prefer_controls_closer_to_current_backend_profile",
                "do_not_extend_backend_from_single_source_repetition",
            ]
        else:
            decision = "no_backend_extension_indicated"
            actions = []
            if any(row.gap_kind == "reference_detail_gap" for row in gaps):
                actions.append("curate_missing_positive_control_reference_details")
            if status_counts.get("qualified_for_current_backend", 0):
                actions.append("plan_qualified_positive_control_execution")
            if not actions:
                actions.append("collect_positive_control_references")

        return SERSEMBackendGapAssessmentBundle(
            bundle_id=_stable_id(
                "sers_em_backend_gap_assessment_bundle",
                qualifications.bundle_id,
                qualifications.model_dump(mode="json"),
                self.assessor_version,
            ),
            source_qualification_bundle_id=qualifications.bundle_id,
            backend_profile=qualifications.backend_profile,
            control_count=len(rows),
            independent_source_count=len(source_locators),
            qualified_control_count=status_counts.get("qualified_for_current_backend", 0),
            model_extension_control_count=status_counts.get("requires_model_extension", 0),
            insufficient_detail_control_count=status_counts.get("insufficient_reference_detail", 0),
            gap_assessments=gaps,
            shared_blockers_across_model_extension_controls=shared_blockers,
            extension_decision=decision,  # type: ignore[arg-type]
            recommended_next_actions=actions,
        )
