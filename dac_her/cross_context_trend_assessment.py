from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable, Literal

from dac_her.cross_context_trend import (
    CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    CrossContextTrendAssessment,
    PairwiseTrendContrast,
    TrendContextProfile,
    audit_cross_context_trends,
    classify_direction_relation,
    classify_evidence_kind_relation,
    classify_shape_relation,
    stable_cross_context_assessment_id,
    stable_pairwise_trend_contrast_id,
)
from dac_her.trend_precision import PaperLocalTrendResult


CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID = (
    "cross_context_trend_assessment_v1_alpha4c3c"
)

PairRole = Literal[
    "repeated",
    "reversal",
    "context_specific",
    "unresolved",
]

_PAIR_ROLES = frozenset({
    "repeated",
    "reversal",
    "context_specific",
    "unresolved",
})


def _profile_sort_key(
    profile: TrendContextProfile,
) -> tuple[str, str, str]:
    return (
        profile.paper_id,
        profile.local_result_id,
        profile.context_profile_id,
    )


def _compare_context_dimensions(
    left: TrendContextProfile,
    right: TrendContextProfile,
) -> dict[str, tuple[str, ...] | str]:
    left_map = left.dimension_map
    right_map = right.dimension_map
    if set(left_map) != set(right_map):
        raise ValueError(
            "TrendContextProfile dimension contracts differ for "
            f"{left.local_result_id!r} and "
            f"{right.local_result_id!r}."
        )

    matched: list[str] = []
    mismatched: list[str] = []
    unknown: list[str] = []
    ambiguous: list[str] = []
    varied: list[str] = []
    not_applicable: list[str] = []

    for name in sorted(left_map):
        left_dimension = left_map[name]
        right_dimension = right_map[name]
        statuses = {
            left_dimension.status,
            right_dimension.status,
        }

        if "varied_control" in statuses:
            if statuses != {"varied_control"}:
                raise ValueError(
                    "Same-relation profiles disagree on varied_control "
                    f"mask for dimension {name!r}: "
                    f"{left_dimension.status!r} vs "
                    f"{right_dimension.status!r}."
                )
            varied.append(name)
            continue

        if statuses == {"not_applicable"}:
            not_applicable.append(name)
            continue

        # Applicability disagreement is not a known same-scale context
        # difference. Preserve it as ambiguity rather than inventing a
        # mismatch value.
        if "not_applicable" in statuses:
            ambiguous.append(name)
            continue

        if "ambiguous" in statuses:
            ambiguous.append(name)
            continue

        if "unknown" in statuses:
            unknown.append(name)
            continue

        if statuses != {"known"}:
            raise ValueError(
                "Unsupported context-status pair for dimension "
                f"{name!r}: {sorted(statuses)!r}."
            )

        if (
            left_dimension.normalized_value
            == right_dimension.normalized_value
        ):
            matched.append(name)
        else:
            mismatched.append(name)

    if mismatched:
        context_relation = "context_different"
    elif unknown or ambiguous:
        context_relation = (
            "context_partially_known"
            if matched
            else "context_unknown"
        )
    elif matched:
        context_relation = "same_context"
    else:
        # Only varied-control and/or not-applicable dimensions provide no
        # positive evidence that the background contexts are the same.
        context_relation = "context_unknown"

    return {
        "context_relation": context_relation,
        "matched_dimensions": tuple(matched),
        "mismatched_dimensions": tuple(mismatched),
        "unknown_dimensions": tuple(unknown),
        "ambiguous_dimensions": tuple(ambiguous),
        "varied_control_dimensions": tuple(varied),
        "not_applicable_dimensions": tuple(not_applicable),
    }


def classify_pair_role(
    contrast: PairwiseTrendContrast,
) -> PairRole:
    if contrast.direction_relation == "opposite_direction":
        return "reversal"

    if contrast.direction_relation in {
        "same_direction",
        "same_non_monotonic",
    }:
        return "repeated"

    if (
        contrast.direction_relation
        in {
            "monotonic_vs_non_monotonic",
            "unchanged_contrast",
        }
        and contrast.context_relation == "context_different"
    ):
        return "context_specific"

    return "unresolved"


def build_pairwise_trend_contrast(
    left: TrendContextProfile,
    right: TrendContextProfile,
    *,
    assessment_semantics_id: str = (
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID
    ),
) -> PairwiseTrendContrast:
    if left.relation_id != right.relation_id:
        raise ValueError(
            "Pairwise contrast requires one TrendRelation ID."
        )
    if left.domain_profile_id != right.domain_profile_id:
        raise ValueError(
            "Pairwise contrast cannot cross domain profiles."
        )
    if left.context_semantics_id != right.context_semantics_id:
        raise ValueError(
            "Pairwise contrast cannot mix context semantics."
        )
    if left.paper_id == right.paper_id:
        raise ValueError(
            "Pairwise cross-context contrast is cross-paper only."
        )

    # Canonicalize left/right ordering so IDs and serialized output are stable
    # regardless of caller order.
    ordered = sorted(
        (left, right),
        key=_profile_sort_key,
    )
    left, right = ordered[0], ordered[1]

    direction_relation = classify_direction_relation(
        left.direction,
        right.direction,
    )
    shape_relation = classify_shape_relation(
        left.shape,
        right.shape,
    )
    evidence_kind_relation = (
        classify_evidence_kind_relation(
            left.evidence_kinds,
            right.evidence_kinds,
        )
    )
    context = _compare_context_dimensions(
        left,
        right,
    )

    differentiating_dimensions: tuple[str, ...] = ()
    if (
        direction_relation
        in {
            "opposite_direction",
            "monotonic_vs_non_monotonic",
            "unchanged_contrast",
        }
        and context["mismatched_dimensions"]
    ):
        differentiating_dimensions = tuple(
            context["mismatched_dimensions"]
        )

    reason_codes = [
        f"direction:{direction_relation}",
        f"context:{context['context_relation']}",
    ]
    if shape_relation == "different_shape":
        reason_codes.append("shape_differs")
    if context["mismatched_dimensions"]:
        reason_codes.append("known_context_mismatch")
    if context["unknown_dimensions"]:
        reason_codes.append("context_unknown_dimensions")
    if context["ambiguous_dimensions"]:
        reason_codes.append("context_ambiguous_dimensions")

    return PairwiseTrendContrast(
        contrast_id=stable_pairwise_trend_contrast_id(
            assessment_semantics_id=
                assessment_semantics_id,
            relation_id=left.relation_id,
            left_result_id=left.local_result_id,
            right_result_id=right.local_result_id,
        ),
        contract_semantics_id=
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        assessment_semantics_id=
            assessment_semantics_id,
        relation_id=left.relation_id,
        left_context_profile_id=
            left.context_profile_id,
        right_context_profile_id=
            right.context_profile_id,
        left_result_id=left.local_result_id,
        right_result_id=right.local_result_id,
        left_paper_id=left.paper_id,
        right_paper_id=right.paper_id,
        direction_relation=direction_relation,
        shape_relation=shape_relation,
        evidence_kind_relation=
            evidence_kind_relation,
        context_relation=str(
            context["context_relation"]
        ),
        matched_dimensions=tuple(
            context["matched_dimensions"]
        ),
        mismatched_dimensions=tuple(
            context["mismatched_dimensions"]
        ),
        unknown_dimensions=tuple(
            context["unknown_dimensions"]
        ),
        ambiguous_dimensions=tuple(
            context["ambiguous_dimensions"]
        ),
        varied_control_dimensions=tuple(
            context["varied_control_dimensions"]
        ),
        not_applicable_dimensions=tuple(
            context["not_applicable_dimensions"]
        ),
        differentiating_dimensions=
            differentiating_dimensions,
        reason_codes=tuple(reason_codes),
    )


def _group_profiles(
    profiles: Iterable[TrendContextProfile],
) -> dict[str, list[TrendContextProfile]]:
    grouped: dict[str, list[TrendContextProfile]] = defaultdict(list)
    seen_profile_ids: set[str] = set()
    seen_result_ids: set[str] = set()

    for profile in profiles:
        if profile.context_profile_id in seen_profile_ids:
            raise ValueError(
                "Duplicate TrendContextProfile ID: "
                f"{profile.context_profile_id!r}."
            )
        if profile.local_result_id in seen_result_ids:
            raise ValueError(
                "Multiple TrendContextProfile rows reference one "
                f"local result: {profile.local_result_id!r}."
            )
        seen_profile_ids.add(profile.context_profile_id)
        seen_result_ids.add(profile.local_result_id)
        grouped[profile.relation_id].append(profile)

    for rows in grouped.values():
        rows.sort(key=_profile_sort_key)
        first = rows[0]
        for row in rows[1:]:
            for field_name in (
                "domain_profile_id",
                "relation_id",
                "independent_variable_key",
                "dependent_observable_key",
                "control_family",
                "observable_semantics",
                "context_semantics_id",
            ):
                if getattr(row, field_name) != getattr(
                    first,
                    field_name,
                ):
                    raise ValueError(
                        "One relation group contains incompatible "
                        f"{field_name}: {first.local_result_id!r} vs "
                        f"{row.local_result_id!r}."
                    )
    return dict(grouped)


def build_complete_cross_paper_contrasts(
    profiles: Iterable[TrendContextProfile],
    *,
    assessment_semantics_id: str = (
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID
    ),
) -> list[PairwiseTrendContrast]:
    grouped = _group_profiles(profiles)
    output: list[PairwiseTrendContrast] = []

    for relation_id in sorted(grouped):
        rows = grouped[relation_id]
        for left, right in combinations(rows, 2):
            if left.paper_id == right.paper_id:
                continue
            output.append(
                build_pairwise_trend_contrast(
                    left,
                    right,
                    assessment_semantics_id=
                        assessment_semantics_id,
                )
            )

    return sorted(
        output,
        key=lambda row: (
            row.relation_id,
            row.left_paper_id,
            row.right_paper_id,
            row.left_result_id,
            row.right_result_id,
            row.contrast_id,
        ),
    )


def _assessment_status(
    contrasts: Iterable[PairwiseTrendContrast],
) -> str:
    roles = {
        classify_pair_role(contrast)
        for contrast in contrasts
    }
    # Priority is deliberately non-majoritarian.
    if "reversal" in roles:
        return "reversed"
    if "context_specific" in roles:
        return "context_specific"
    if "repeated" in roles:
        return "repeated"
    return "insufficient"


def _direction_buckets(
    profiles: Iterable[TrendContextProfile],
) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {
        "positive_result_ids": [],
        "negative_result_ids": [],
        "non_monotonic_result_ids": [],
        "unchanged_result_ids": [],
        "unspecified_result_ids": [],
    }
    mapping = {
        "positive": "positive_result_ids",
        "negative": "negative_result_ids",
        "non_monotonic": "non_monotonic_result_ids",
        "unchanged": "unchanged_result_ids",
    }
    for profile in profiles:
        bucket = mapping.get(
            profile.direction,
            "unspecified_result_ids",
        )
        buckets[bucket].append(
            profile.local_result_id
        )
    return {
        key: tuple(sorted(values))
        for key, values in buckets.items()
    }


def _evidence_kind_buckets(
    profiles: Iterable[TrendContextProfile],
) -> dict[str, tuple[str, ...]]:
    fields = {
        "experimental_numeric":
            "experimental_numeric_result_ids",
        "calculated_numeric":
            "calculated_numeric_result_ids",
        "reported_claim":
            "reported_claim_result_ids",
    }
    output: dict[str, tuple[str, ...]] = {}
    rows = list(profiles)
    for evidence_kind, field_name in fields.items():
        output[field_name] = tuple(sorted(
            profile.local_result_id
            for profile in rows
            if evidence_kind in profile.evidence_kinds
        ))
    return output


def build_relation_assessment(
    profiles: Iterable[TrendContextProfile],
    contrasts: Iterable[PairwiseTrendContrast],
    *,
    assessment_semantics_id: str = (
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID
    ),
) -> CrossContextTrendAssessment:
    profile_rows = sorted(
        list(profiles),
        key=_profile_sort_key,
    )
    if not profile_rows:
        raise ValueError(
            "Relation assessment requires at least one profile."
        )

    first = profile_rows[0]
    for row in profile_rows[1:]:
        if row.relation_id != first.relation_id:
            raise ValueError(
                "Relation assessment cannot cross relation IDs."
            )
        if row.domain_profile_id != first.domain_profile_id:
            raise ValueError(
                "Relation assessment cannot cross domains."
            )

    contrast_rows = sorted(
        list(contrasts),
        key=lambda row: row.contrast_id,
    )
    member_result_ids = {
        row.local_result_id
        for row in profile_rows
    }
    for contrast in contrast_rows:
        if contrast.relation_id != first.relation_id:
            raise ValueError(
                "Assessment contrast crosses relation IDs."
            )
        if not {
            contrast.left_result_id,
            contrast.right_result_id,
        }.issubset(member_result_ids):
            raise ValueError(
                "Assessment contrast references a result outside "
                "the relation group."
            )

    repeated: list[str] = []
    reversal: list[str] = []
    context_specific: list[str] = []
    unresolved: list[str] = []
    for contrast in contrast_rows:
        role = classify_pair_role(contrast)
        if role == "repeated":
            repeated.append(contrast.contrast_id)
        elif role == "reversal":
            reversal.append(contrast.contrast_id)
        elif role == "context_specific":
            context_specific.append(contrast.contrast_id)
        elif role == "unresolved":
            unresolved.append(contrast.contrast_id)
        else:
            raise ValueError(
                f"Unknown pair role: {role!r}."
            )

    status = _assessment_status(
        contrast_rows
    )

    differentiating_dimensions = tuple(sorted({
        dimension
        for contrast in contrast_rows
        if classify_pair_role(contrast)
        in {"reversal", "context_specific"}
        for dimension in contrast.differentiating_dimensions
    }))
    unresolved_dimensions = tuple(sorted({
        dimension
        for contrast in contrast_rows
        for dimension in (
            *contrast.unknown_dimensions,
            *contrast.ambiguous_dimensions,
        )
    }))

    paper_ids = tuple(sorted({
        row.paper_id
        for row in profile_rows
    }))
    reason_codes: list[str] = []
    if status == "reversed":
        reason_codes.append(
            "strict_direction_reversal_present"
        )
    elif status == "context_specific":
        reason_codes.append(
            "context_linked_trend_character_change"
        )
    elif status == "repeated":
        reason_codes.append(
            "cross_paper_directional_repetition"
        )
    else:
        if len(paper_ids) < 2:
            reason_codes.append(
                "fewer_than_two_papers"
            )
            if len(profile_rows) > 1:
                reason_codes.append(
                    "same_paper_support_not_replication"
                )
        elif not contrast_rows:
            reason_codes.append(
                "no_cross_paper_pair"
            )
        else:
            reason_codes.append(
                "no_resolved_cross_paper_direction_relation"
            )

    direction_buckets = _direction_buckets(
        profile_rows
    )
    evidence_buckets = _evidence_kind_buckets(
        profile_rows
    )
    pairwise_ids = tuple(
        row.contrast_id
        for row in contrast_rows
    )
    local_result_ids = tuple(sorted(
        member_result_ids
    ))

    return CrossContextTrendAssessment(
        assessment_id=
            stable_cross_context_assessment_id(
                assessment_semantics_id=
                    assessment_semantics_id,
                relation_id=first.relation_id,
                member_result_ids=
                    local_result_ids,
            ),
        domain_profile_id=first.domain_profile_id,
        contract_semantics_id=
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        assessment_semantics_id=
            assessment_semantics_id,
        relation_id=first.relation_id,
        independent_variable_key=
            first.independent_variable_key,
        dependent_observable_key=
            first.dependent_observable_key,
        control_family=first.control_family,
        observable_semantics=
            first.observable_semantics,
        member_result_ids=local_result_ids,
        paper_ids=paper_ids,
        pairwise_contrast_ids=
            pairwise_ids,
        status=status,
        **direction_buckets,
        **evidence_buckets,
        repeated_pair_ids=tuple(repeated),
        reversal_pair_ids=tuple(reversal),
        context_specific_pair_ids=
            tuple(context_specific),
        unresolved_pair_ids=tuple(unresolved),
        differentiating_dimensions=
            differentiating_dimensions,
        unresolved_dimensions=
            unresolved_dimensions,
        reason_codes=tuple(reason_codes),
    )


def build_deterministic_cross_context_assessments(
    profiles: Iterable[TrendContextProfile],
    *,
    assessment_semantics_id: str = (
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID
    ),
) -> tuple[
    list[PairwiseTrendContrast],
    list[CrossContextTrendAssessment],
]:
    profile_rows = list(profiles)
    grouped = _group_profiles(profile_rows)
    contrasts = build_complete_cross_paper_contrasts(
        profile_rows,
        assessment_semantics_id=
            assessment_semantics_id,
    )
    contrasts_by_relation: dict[
        str,
        list[PairwiseTrendContrast],
    ] = defaultdict(list)
    for contrast in contrasts:
        contrasts_by_relation[
            contrast.relation_id
        ].append(contrast)

    assessments = [
        build_relation_assessment(
            grouped[relation_id],
            contrasts_by_relation.get(
                relation_id,
                [],
            ),
            assessment_semantics_id=
                assessment_semantics_id,
        )
        for relation_id in sorted(grouped)
    ]
    return contrasts, assessments


@dataclass(frozen=True)
class DeterministicCrossContextAssessmentAudit:
    contract_semantics_id: str
    assessment_semantics_id: str
    relation_count: int
    local_result_count: int
    context_profile_count: int
    expected_cross_paper_pair_count: int
    pairwise_contrast_count: int
    assessment_count: int
    status_counts: dict[str, int]
    pair_role_counts: dict[str, int]
    base_structural_gate: bool
    issues: tuple[str, ...]
    structural_gate: bool

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = list(self.issues)
        return row


def audit_deterministic_cross_context_assessments(
    *,
    local_results: list[PaperLocalTrendResult],
    profiles: list[TrendContextProfile],
    contrasts: list[PairwiseTrendContrast],
    assessments: list[CrossContextTrendAssessment],
    assessment_semantics_id: str = (
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID
    ),
) -> DeterministicCrossContextAssessmentAudit:
    base = audit_cross_context_trends(
        local_results=local_results,
        profiles=profiles,
        contrasts=contrasts,
        assessments=assessments,
    )
    issues: list[str] = list(base.issues)

    grouped = _group_profiles(profiles)
    expected_contrasts = (
        build_complete_cross_paper_contrasts(
            profiles,
            assessment_semantics_id=
                assessment_semantics_id,
        )
    )
    expected_contrast_by_id = {
        row.contrast_id: row
        for row in expected_contrasts
    }
    actual_contrast_by_id = {
        row.contrast_id: row
        for row in contrasts
    }

    if (
        set(actual_contrast_by_id)
        != set(expected_contrast_by_id)
    ):
        issues.append(
            "cross_paper_pair_completeness_mismatch"
        )
    for contrast_id in sorted(
        set(actual_contrast_by_id)
        & set(expected_contrast_by_id)
    ):
        if (
            actual_contrast_by_id[contrast_id]
            != expected_contrast_by_id[contrast_id]
        ):
            issues.append(
                "deterministic_pairwise_contrast_mismatch:"
                f"{contrast_id}"
            )

    assessment_by_relation = {
        row.relation_id: row
        for row in assessments
    }
    if len(assessment_by_relation) != len(assessments):
        issues.append(
            "multiple_assessments_for_one_relation"
        )
    if set(assessment_by_relation) != set(grouped):
        issues.append(
            "relation_assessment_coverage_mismatch"
        )

    expected_contrasts_by_relation: dict[
        str,
        list[PairwiseTrendContrast],
    ] = defaultdict(list)
    for contrast in expected_contrasts:
        expected_contrasts_by_relation[
            contrast.relation_id
        ].append(contrast)

    for relation_id in sorted(
        set(grouped)
        & set(assessment_by_relation)
    ):
        expected = build_relation_assessment(
            grouped[relation_id],
            expected_contrasts_by_relation.get(
                relation_id,
                [],
            ),
            assessment_semantics_id=
                assessment_semantics_id,
        )
        observed = assessment_by_relation[
            relation_id
        ]
        if observed != expected:
            issues.append(
                "deterministic_relation_assessment_mismatch:"
                f"{relation_id}"
            )

    role_counts = Counter(
        classify_pair_role(contrast)
        for contrast in contrasts
    )
    status_counts = Counter(
        assessment.status
        for assessment in assessments
    )

    unique_issues = tuple(sorted(set(issues)))
    return DeterministicCrossContextAssessmentAudit(
        contract_semantics_id=
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        assessment_semantics_id=
            assessment_semantics_id,
        relation_count=len(grouped),
        local_result_count=len(local_results),
        context_profile_count=len(profiles),
        expected_cross_paper_pair_count=
            len(expected_contrasts),
        pairwise_contrast_count=len(contrasts),
        assessment_count=len(assessments),
        status_counts=dict(
            sorted(status_counts.items())
        ),
        pair_role_counts=dict(
            sorted(role_counts.items())
        ),
        base_structural_gate=
            base.structural_gate,
        issues=unique_issues,
        structural_gate=not unique_issues,
    )
