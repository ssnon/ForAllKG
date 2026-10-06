from __future__ import annotations

import hashlib
import itertools
import json

from domains.sers.classical_em_backend_gap_contracts import (
    SERSEMBackendGapAssessmentBundle,
)
from domains.sers.classical_em_extension_design_contracts import (
    SERSEMJointExtensionCandidate,
    SERSEMJointExtensionDesignBundle,
)
from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlQualification,
    SERSEMPositiveControlQualificationBundle,
)


_DESIGNER_VERSION = "sers-em-joint-extension-designer-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _capability_rows(
    rows: list[SERSEMPositiveControlQualification],
) -> list[SERSEMPositiveControlQualification]:
    return [row for row in rows if row.status == "requires_model_extension"]


def _is_dominated(candidate: dict, other: dict) -> bool:
    """Pareto dominance: no larger extension, no worse unlock coverage."""
    no_more_axes = other["axis_count"] <= candidate["axis_count"]
    no_fewer_controls = set(other["unlocked_control_ids"]) >= set(
        candidate["unlocked_control_ids"]
    )
    no_fewer_sources = set(other["unlocked_source_locators"]) >= set(
        candidate["unlocked_source_locators"]
    )
    strictly_better = (
        other["axis_count"] < candidate["axis_count"]
        or set(other["unlocked_control_ids"]) > set(candidate["unlocked_control_ids"])
        or set(other["unlocked_source_locators"]) > set(
            candidate["unlocked_source_locators"]
        )
    )
    return no_more_axes and no_fewer_controls and no_fewer_sources and strictly_better


class SERSClassicalEMJointExtensionDesigner:
    """Design, but never authorize, minimal joint backend extensions.

    v0 enumerates subsets of the backend-capability blockers already identified
    by S3.4b. A control is considered *unlockable* only if every backend blocker
    on that control is covered by the candidate extension. The returned set is a
    Pareto frontier balancing fewer changed model axes against more unlocked
    controls and independent sources.
    """

    designer_version = _DESIGNER_VERSION

    def design(
        self,
        qualifications: SERSEMPositiveControlQualificationBundle,
        gaps: SERSEMBackendGapAssessmentBundle,
    ) -> SERSEMJointExtensionDesignBundle:
        if gaps.source_qualification_bundle_id != qualifications.bundle_id:
            raise ValueError("backend-gap/qualification lineage mismatch")
        if gaps.backend_profile != qualifications.backend_profile:
            raise ValueError("backend-gap/qualification backend profile mismatch")
        if gaps.extension_decision != "cross_source_gaps_require_joint_design":
            raise ValueError(
                "joint extension design requires "
                "extension_decision='cross_source_gaps_require_joint_design'"
            )

        rows = _capability_rows(list(qualifications.qualifications))
        by_control = {row.control_id: row for row in rows}
        if len(by_control) != len(rows):
            raise ValueError("model-extension control ids must be unique")

        gap_by_blocker = {
            row.blocker: row
            for row in gaps.gap_assessments
            if row.gap_kind == "backend_capability_gap"
        }
        present_blockers = sorted(
            {
                blocker
                for row in rows
                for blocker in row.blockers
                if blocker in gap_by_blocker
            }
        )

        source_count = len({row.source_locator for row in rows})
        if not rows or not present_blockers:
            return SERSEMJointExtensionDesignBundle(
                bundle_id=_stable_id(
                    "sers_em_joint_extension_design_bundle",
                    qualifications.bundle_id,
                    gaps.bundle_id,
                    self.designer_version,
                ),
                source_qualification_bundle_id=qualifications.bundle_id,
                source_backend_gap_bundle_id=gaps.bundle_id,
                backend_profile=qualifications.backend_profile,
                model_extension_control_count=len(rows),
                independent_source_count=source_count,
                capability_blocker_count=len(present_blockers),
                candidates=[],
                candidate_count=0,
                design_decision="no_joint_extension_candidate",
                recommended_next_actions=[
                    "collect_or_curate_positive_controls_before_extension_design"
                ],
            )

        raw_candidates: list[dict] = []
        # The v0 blocker vocabulary is deliberately small. Enumerating all
        # subsets keeps the algorithm deterministic and makes joint-unlock
        # semantics auditable instead of relying on a heuristic search.
        for size in range(1, len(present_blockers) + 1):
            for combo in itertools.combinations(present_blockers, size):
                extension = set(combo)
                unlocked = [
                    row
                    for row in rows
                    if set(row.blockers) <= extension
                ]
                if not unlocked:
                    continue
                unlocked_ids = sorted(row.control_id for row in unlocked)
                unlocked_sources = sorted({row.source_locator for row in unlocked})
                remaining = sorted(set(by_control) - set(unlocked_ids))
                raw_candidates.append({
                    "blockers": list(combo),
                    "axis_count": size,
                    "unlocked_control_ids": unlocked_ids,
                    "unlocked_source_locators": unlocked_sources,
                    "remaining_blocked_control_ids": remaining,
                })

        frontier = [
            row
            for row in raw_candidates
            if not any(
                _is_dominated(row, other)
                for other in raw_candidates
                if other is not row
            )
        ]
        frontier.sort(
            key=lambda row: (
                row["axis_count"],
                -len(row["unlocked_source_locators"]),
                -len(row["unlocked_control_ids"]),
                row["blockers"],
            )
        )

        candidates: list[SERSEMJointExtensionCandidate] = []
        all_control_ids = set(by_control)
        for row in frontier:
            blocker_rows = [gap_by_blocker[value] for value in row["blockers"]]
            model_axes = [item.model_axis for item in blocker_rows]
            targets = [item.extension_target for item in blocker_rows]
            candidate_id = _stable_id(
                "sers_em_joint_extension_candidate",
                qualifications.bundle_id,
                gaps.bundle_id,
                row,
                self.designer_version,
            )
            candidates.append(SERSEMJointExtensionCandidate(
                candidate_id=candidate_id,
                blockers_addressed=row["blockers"],
                model_axes=model_axes,
                extension_targets=targets,
                axis_count=row["axis_count"],
                unlocked_control_ids=row["unlocked_control_ids"],
                unlocked_source_locators=row["unlocked_source_locators"],
                unlocked_control_count=len(row["unlocked_control_ids"]),
                unlocked_independent_source_count=len(row["unlocked_source_locators"]),
                remaining_blocked_control_ids=row["remaining_blocked_control_ids"],
                remaining_blocked_control_count=len(row["remaining_blocked_control_ids"]),
                unlocks_cross_source_controls=(
                    len(row["unlocked_source_locators"]) >= 2
                ),
                unlocks_all_model_extension_controls=(
                    set(row["unlocked_control_ids"]) == all_control_ids
                ),
            ))

        any_axes = min((row.axis_count for row in candidates), default=None)
        cross_axes = min(
            (
                row.axis_count
                for row in candidates
                if row.unlocks_cross_source_controls
            ),
            default=None,
        )
        full_axes = min(
            (
                row.axis_count
                for row in candidates
                if row.unlocks_all_model_extension_controls
            ),
            default=None,
        )

        actions = [
            "review_joint_extension_pareto_frontier",
            "compare_backend_extension_cost_against_canonical_benchmark_search",
            "predeclare_positive_control_acceptance_criterion_before_execution",
            "do_not_implement_backend_extension_from_this_artifact_alone",
        ]

        return SERSEMJointExtensionDesignBundle(
            bundle_id=_stable_id(
                "sers_em_joint_extension_design_bundle",
                qualifications.bundle_id,
                gaps.bundle_id,
                *(row.candidate_id for row in candidates),
                self.designer_version,
            ),
            source_qualification_bundle_id=qualifications.bundle_id,
            source_backend_gap_bundle_id=gaps.bundle_id,
            backend_profile=qualifications.backend_profile,
            model_extension_control_count=len(rows),
            independent_source_count=source_count,
            capability_blocker_count=len(present_blockers),
            candidates=candidates,
            candidate_count=len(candidates),
            minimum_axis_count_for_any_unlock=any_axes,
            minimum_axis_count_for_cross_source_unlock=cross_axes,
            minimum_axis_count_for_full_coverage=full_axes,
            design_decision=(
                "review_joint_extension_frontier"
                if candidates
                else "no_joint_extension_candidate"
            ),
            recommended_next_actions=actions if candidates else [
                "collect_or_curate_positive_controls_before_extension_design"
            ],
        )
