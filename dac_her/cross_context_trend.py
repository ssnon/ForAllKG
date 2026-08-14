from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping

import networkx as nx

from dac_her.trend_precision import (
    PaperLocalTrendResult,
    TREND_EVIDENCE_KINDS,
    TrendEvidenceKind,
    TrendResultLane,
)


CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID = (
    "cross_context_trend_contract_v1_alpha4c3a"
)

TrendContextDimensionStatus = Literal[
    "known",
    "unknown",
    "ambiguous",
    "varied_control",
    "not_applicable",
]
TrendDirectionRelation = Literal[
    "same_direction",
    "opposite_direction",
    "same_non_monotonic",
    "monotonic_vs_non_monotonic",
    "unchanged_contrast",
    "unresolved",
]
TrendShapeRelation = Literal[
    "same_shape",
    "different_shape",
    "unresolved",
]
TrendEvidenceKindRelation = Literal[
    "same_kind",
    "cross_kind",
    "mixed_kind",
    "unresolved",
]
TrendContextRelation = Literal[
    "same_context",
    "context_different",
    "context_partially_known",
    "context_unknown",
]
CrossContextTrendStatus = Literal[
    "repeated",
    "context_specific",
    "reversed",
    "insufficient",
]

TREND_CONTEXT_DIMENSION_STATUSES = frozenset({
    "known",
    "unknown",
    "ambiguous",
    "varied_control",
    "not_applicable",
})
TREND_DIRECTION_RELATIONS = frozenset({
    "same_direction",
    "opposite_direction",
    "same_non_monotonic",
    "monotonic_vs_non_monotonic",
    "unchanged_contrast",
    "unresolved",
})
TREND_SHAPE_RELATIONS = frozenset({
    "same_shape",
    "different_shape",
    "unresolved",
})
TREND_EVIDENCE_KIND_RELATIONS = frozenset({
    "same_kind",
    "cross_kind",
    "mixed_kind",
    "unresolved",
})
TREND_CONTEXT_RELATIONS = frozenset({
    "same_context",
    "context_different",
    "context_partially_known",
    "context_unknown",
})
CROSS_CONTEXT_TREND_STATUSES = frozenset({
    "repeated",
    "context_specific",
    "reversed",
    "insufficient",
})

_DIRECTION_BUCKETS = (
    "positive_result_ids",
    "negative_result_ids",
    "non_monotonic_result_ids",
    "unchanged_result_ids",
    "unspecified_result_ids",
)
_PAIR_ROLE_BUCKETS = (
    "repeated_pair_ids",
    "reversal_pair_ids",
    "context_specific_pair_ids",
    "unresolved_pair_ids",
)
_CONTRAST_DIMENSION_BUCKETS = (
    "matched_dimensions",
    "mismatched_dimensions",
    "unknown_dimensions",
    "ambiguous_dimensions",
    "varied_control_dimensions",
    "not_applicable_dimensions",
)


def _require_nonempty(name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty.")
    return text


def _tuple_unique(name: str, values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if any(not value.strip() for value in normalized):
        raise ValueError(f"{name} must not contain empty values.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} values must be unique.")
    return normalized


def stable_trend_relation_id(
    *,
    independent_variable_key: str,
    dependent_observable_key: str,
    control_family: str,
    observable_semantics: str,
) -> str:
    payload = "|".join((
        _require_nonempty("independent_variable_key", independent_variable_key),
        _require_nonempty("dependent_observable_key", dependent_observable_key),
        _require_nonempty("control_family", control_family),
        _require_nonempty("observable_semantics", observable_semantics),
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"trend-relation:{digest}"


def stable_trend_context_profile_id(
    *,
    context_semantics_id: str,
    local_result_id: str,
) -> str:
    payload = "|".join((
        _require_nonempty("context_semantics_id", context_semantics_id),
        _require_nonempty("local_result_id", local_result_id),
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"trend-context:{digest}"


def stable_pairwise_trend_contrast_id(
    *,
    assessment_semantics_id: str,
    relation_id: str,
    left_result_id: str,
    right_result_id: str,
) -> str:
    left = _require_nonempty("left_result_id", left_result_id)
    right = _require_nonempty("right_result_id", right_result_id)
    if left == right:
        raise ValueError(
            "Pairwise trend contrast requires distinct result IDs."
        )
    ordered = tuple(sorted((left, right)))
    payload = "|".join((
        _require_nonempty(
            "assessment_semantics_id",
            assessment_semantics_id,
        ),
        _require_nonempty("relation_id", relation_id),
        *ordered,
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"trend-contrast:{digest}"


def stable_cross_context_assessment_id(
    *,
    assessment_semantics_id: str,
    relation_id: str,
    member_result_ids: Iterable[str],
) -> str:
    members = tuple(sorted(set(map(str, member_result_ids))))
    if not members:
        raise ValueError(
            "Cross-context assessment requires at least one member result."
        )
    if any(not value.strip() for value in members):
        raise ValueError(
            "Cross-context assessment member IDs must not be empty."
        )
    payload = "|".join((
        _require_nonempty(
            "assessment_semantics_id",
            assessment_semantics_id,
        ),
        _require_nonempty("relation_id", relation_id),
        *members,
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"cross-context-trend:{digest}"


def classify_direction_relation(
    left_direction: str,
    right_direction: str,
) -> TrendDirectionRelation:
    left = str(left_direction).strip()
    right = str(right_direction).strip()
    if not left or not right:
        return "unresolved"
    if left == right:
        if left == "non_monotonic":
            return "same_non_monotonic"
        if left == "unspecified":
            return "unresolved"
        return "same_direction"
    if {left, right} == {"positive", "negative"}:
        return "opposite_direction"
    if (
        "non_monotonic" in {left, right}
        and ({left, right} & {"positive", "negative"})
    ):
        return "monotonic_vs_non_monotonic"
    if "unchanged" in {left, right}:
        return "unchanged_contrast"
    return "unresolved"


def classify_shape_relation(
    left_shape: str,
    right_shape: str,
) -> TrendShapeRelation:
    left = str(left_shape).strip()
    right = str(right_shape).strip()
    if not left or not right or "unspecified" in {left, right}:
        return "unresolved"
    if left == right:
        return "same_shape"
    return "different_shape"


def classify_evidence_kind_relation(
    left_kinds: Iterable[str],
    right_kinds: Iterable[str],
) -> TrendEvidenceKindRelation:
    left = frozenset(map(str, left_kinds))
    right = frozenset(map(str, right_kinds))
    if not left or not right:
        return "unresolved"
    if (
        not left.issubset(TREND_EVIDENCE_KINDS)
        or not right.issubset(TREND_EVIDENCE_KINDS)
    ):
        return "unresolved"
    if len(left) > 1 or len(right) > 1:
        return "mixed_kind"
    if left == right:
        return "same_kind"
    return "cross_kind"


@dataclass(frozen=True)
class TrendContextDimension:
    name: str
    status: TrendContextDimensionStatus
    normalized_value: str = ""
    source_values: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    provenance_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("TrendContextDimension name", self.name)
        if self.status not in TREND_CONTEXT_DIMENSION_STATUSES:
            raise ValueError(
                "Unknown TrendContextDimension status: "
                f"{self.status!r}."
            )
        _tuple_unique("source_values", self.source_values)
        _tuple_unique("source_node_ids", self.source_node_ids)
        _tuple_unique("provenance_scopes", self.provenance_scopes)

        normalized = str(self.normalized_value)
        if self.status == "known":
            if not normalized.strip():
                raise ValueError(
                    f"Known context dimension {self.name!r} "
                    "requires normalized_value."
                )
            if not self.source_node_ids and not self.provenance_scopes:
                raise ValueError(
                    f"Known context dimension {self.name!r} requires "
                    "source_node_ids or provenance_scopes."
                )
        elif self.status in {"unknown", "varied_control", "not_applicable"}:
            if normalized:
                raise ValueError(
                    f"{self.status} context dimension {self.name!r} "
                    "cannot carry normalized_value."
                )

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["source_values"] = list(self.source_values)
        row["source_node_ids"] = list(self.source_node_ids)
        row["provenance_scopes"] = list(self.provenance_scopes)
        return row


@dataclass(frozen=True)
class TrendContextProfile:
    context_profile_id: str
    domain_profile_id: str
    contract_semantics_id: str
    context_semantics_id: str
    local_result_id: str
    paper_id: str
    relation_id: str
    independent_variable_key: str
    dependent_observable_key: str
    control_family: str
    observable_semantics: str
    result_lane: TrendResultLane
    direction: str
    shape: str
    evidence_kinds: tuple[TrendEvidenceKind, ...]
    member_trend_ids: tuple[str, ...]
    dimensions: tuple[TrendContextDimension, ...]
    source_comparison_context_ids: tuple[str, ...] = ()
    source_method_context_ids: tuple[str, ...] = ()
    source_claim_ids: tuple[str, ...] = ()
    source_measurement_ids: tuple[str, ...] = ()
    source_measurement_result_ids: tuple[str, ...] = ()
    source_calculation_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("context_profile_id", self.context_profile_id),
            ("domain_profile_id", self.domain_profile_id),
            ("contract_semantics_id", self.contract_semantics_id),
            ("context_semantics_id", self.context_semantics_id),
            ("local_result_id", self.local_result_id),
            ("paper_id", self.paper_id),
            ("relation_id", self.relation_id),
            ("independent_variable_key", self.independent_variable_key),
            ("dependent_observable_key", self.dependent_observable_key),
            ("control_family", self.control_family),
            ("observable_semantics", self.observable_semantics),
            ("direction", self.direction),
            ("shape", self.shape),
        ):
            _require_nonempty(f"TrendContextProfile {name}", value)

        if (
            self.contract_semantics_id
            != CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID
        ):
            raise ValueError(
                "TrendContextProfile contract semantics mismatch."
            )
        if self.result_lane not in {"numeric", "claim"}:
            raise ValueError(
                "Unknown TrendContextProfile result lane: "
                f"{self.result_lane!r}."
            )
        if not self.evidence_kinds:
            raise ValueError(
                "TrendContextProfile requires evidence_kinds."
            )
        if not set(self.evidence_kinds).issubset(TREND_EVIDENCE_KINDS):
            raise ValueError(
                "TrendContextProfile has unknown evidence kind."
            )
        _tuple_unique(
            "TrendContextProfile evidence_kinds",
            self.evidence_kinds,
        )
        _tuple_unique(
            "TrendContextProfile member_trend_ids",
            self.member_trend_ids,
        )
        if not self.member_trend_ids:
            raise ValueError(
                "TrendContextProfile requires member_trend_ids."
            )
        if not self.dimensions:
            raise ValueError(
                "TrendContextProfile requires explicit context "
                "dimensions; unknown must be represented explicitly."
            )
        names = [item.name for item in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError(
                "TrendContextProfile dimensions must be unique."
            )

        expected_relation = stable_trend_relation_id(
            independent_variable_key=self.independent_variable_key,
            dependent_observable_key=self.dependent_observable_key,
            control_family=self.control_family,
            observable_semantics=self.observable_semantics,
        )
        if self.relation_id != expected_relation:
            raise ValueError(
                "TrendContextProfile relation_id does not match "
                "relation semantics."
            )
        expected_profile = stable_trend_context_profile_id(
            context_semantics_id=self.context_semantics_id,
            local_result_id=self.local_result_id,
        )
        if self.context_profile_id != expected_profile:
            raise ValueError(
                "TrendContextProfile context_profile_id is not stable."
            )

        for field_name in (
            "source_comparison_context_ids",
            "source_method_context_ids",
            "source_claim_ids",
            "source_measurement_ids",
            "source_measurement_result_ids",
            "source_calculation_ids",
            "source_node_ids",
        ):
            _tuple_unique(
                f"TrendContextProfile {field_name}",
                getattr(self, field_name),
            )

    @property
    def dimension_map(self) -> dict[str, TrendContextDimension]:
        return {item.name: item for item in self.dimensions}

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["evidence_kinds"] = list(self.evidence_kinds)
        row["member_trend_ids"] = list(self.member_trend_ids)
        row["dimensions"] = [
            item.to_dict() for item in self.dimensions
        ]
        for key in (
            "source_comparison_context_ids",
            "source_method_context_ids",
            "source_claim_ids",
            "source_measurement_ids",
            "source_measurement_result_ids",
            "source_calculation_ids",
            "source_node_ids",
        ):
            row[key] = list(row[key])
        return row


@dataclass(frozen=True)
class PairwiseTrendContrast:
    contrast_id: str
    contract_semantics_id: str
    assessment_semantics_id: str
    relation_id: str
    left_context_profile_id: str
    right_context_profile_id: str
    left_result_id: str
    right_result_id: str
    left_paper_id: str
    right_paper_id: str
    direction_relation: TrendDirectionRelation
    shape_relation: TrendShapeRelation
    evidence_kind_relation: TrendEvidenceKindRelation
    context_relation: TrendContextRelation
    matched_dimensions: tuple[str, ...] = ()
    mismatched_dimensions: tuple[str, ...] = ()
    unknown_dimensions: tuple[str, ...] = ()
    ambiguous_dimensions: tuple[str, ...] = ()
    varied_control_dimensions: tuple[str, ...] = ()
    not_applicable_dimensions: tuple[str, ...] = ()
    differentiating_dimensions: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("contrast_id", self.contrast_id),
            ("contract_semantics_id", self.contract_semantics_id),
            ("assessment_semantics_id", self.assessment_semantics_id),
            ("relation_id", self.relation_id),
            ("left_context_profile_id", self.left_context_profile_id),
            ("right_context_profile_id", self.right_context_profile_id),
            ("left_result_id", self.left_result_id),
            ("right_result_id", self.right_result_id),
            ("left_paper_id", self.left_paper_id),
            ("right_paper_id", self.right_paper_id),
        ):
            _require_nonempty(f"PairwiseTrendContrast {name}", value)
        if (
            self.contract_semantics_id
            != CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID
        ):
            raise ValueError(
                "PairwiseTrendContrast contract semantics mismatch."
            )
        if self.left_result_id == self.right_result_id:
            raise ValueError(
                "PairwiseTrendContrast requires distinct result IDs."
            )
        if self.left_paper_id == self.right_paper_id:
            raise ValueError(
                "PairwiseTrendContrast is cross-paper only."
            )
        if self.left_context_profile_id == self.right_context_profile_id:
            raise ValueError(
                "PairwiseTrendContrast requires distinct profiles."
            )
        if self.direction_relation not in TREND_DIRECTION_RELATIONS:
            raise ValueError(
                "Unknown direction_relation: "
                f"{self.direction_relation!r}."
            )
        if self.shape_relation not in TREND_SHAPE_RELATIONS:
            raise ValueError(
                "Unknown shape_relation: "
                f"{self.shape_relation!r}."
            )
        if (
            self.evidence_kind_relation
            not in TREND_EVIDENCE_KIND_RELATIONS
        ):
            raise ValueError(
                "Unknown evidence_kind_relation: "
                f"{self.evidence_kind_relation!r}."
            )
        if self.context_relation not in TREND_CONTEXT_RELATIONS:
            raise ValueError(
                "Unknown context_relation: "
                f"{self.context_relation!r}."
            )

        bucket_sets: dict[str, set[str]] = {}
        for field_name in _CONTRAST_DIMENSION_BUCKETS:
            values = _tuple_unique(
                f"PairwiseTrendContrast {field_name}",
                getattr(self, field_name),
            )
            bucket_sets[field_name] = set(values)

        names = list(bucket_sets)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1:]:
                overlap = (
                    bucket_sets[left_name] & bucket_sets[right_name]
                )
                if overlap:
                    raise ValueError(
                        "PairwiseTrendContrast context dimension "
                        "buckets must be disjoint: "
                        f"{left_name} vs {right_name}: "
                        f"{sorted(overlap)!r}."
                    )

        _tuple_unique(
            "PairwiseTrendContrast differentiating_dimensions",
            self.differentiating_dimensions,
        )
        if not set(self.differentiating_dimensions).issubset(
            set(self.mismatched_dimensions)
        ):
            raise ValueError(
                "differentiating_dimensions must be a subset of "
                "mismatched_dimensions."
            )
        _tuple_unique(
            "PairwiseTrendContrast reason_codes",
            self.reason_codes,
        )

        if (
            self.context_relation == "context_different"
            and not self.mismatched_dimensions
        ):
            raise ValueError(
                "context_different requires at least one "
                "mismatched dimension."
            )

        expected_id = stable_pairwise_trend_contrast_id(
            assessment_semantics_id=self.assessment_semantics_id,
            relation_id=self.relation_id,
            left_result_id=self.left_result_id,
            right_result_id=self.right_result_id,
        )
        if self.contrast_id != expected_id:
            raise ValueError(
                "PairwiseTrendContrast contrast_id is not stable."
            )

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in (
            *_CONTRAST_DIMENSION_BUCKETS,
            "differentiating_dimensions",
            "reason_codes",
        ):
            row[key] = list(row[key])
        return row


@dataclass(frozen=True)
class CrossContextTrendAssessment:
    assessment_id: str
    domain_profile_id: str
    contract_semantics_id: str
    assessment_semantics_id: str
    relation_id: str
    independent_variable_key: str
    dependent_observable_key: str
    control_family: str
    observable_semantics: str
    member_result_ids: tuple[str, ...]
    paper_ids: tuple[str, ...]
    pairwise_contrast_ids: tuple[str, ...]
    status: CrossContextTrendStatus
    positive_result_ids: tuple[str, ...] = ()
    negative_result_ids: tuple[str, ...] = ()
    non_monotonic_result_ids: tuple[str, ...] = ()
    unchanged_result_ids: tuple[str, ...] = ()
    unspecified_result_ids: tuple[str, ...] = ()
    experimental_numeric_result_ids: tuple[str, ...] = ()
    calculated_numeric_result_ids: tuple[str, ...] = ()
    reported_claim_result_ids: tuple[str, ...] = ()
    repeated_pair_ids: tuple[str, ...] = ()
    reversal_pair_ids: tuple[str, ...] = ()
    context_specific_pair_ids: tuple[str, ...] = ()
    unresolved_pair_ids: tuple[str, ...] = ()
    differentiating_dimensions: tuple[str, ...] = ()
    unresolved_dimensions: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("domain_profile_id", self.domain_profile_id),
            ("contract_semantics_id", self.contract_semantics_id),
            ("assessment_semantics_id", self.assessment_semantics_id),
            ("relation_id", self.relation_id),
            ("independent_variable_key", self.independent_variable_key),
            ("dependent_observable_key", self.dependent_observable_key),
            ("control_family", self.control_family),
            ("observable_semantics", self.observable_semantics),
        ):
            _require_nonempty(
                f"CrossContextTrendAssessment {name}",
                value,
            )
        if (
            self.contract_semantics_id
            != CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID
        ):
            raise ValueError(
                "CrossContextTrendAssessment contract semantics "
                "mismatch."
            )
        if self.status not in CROSS_CONTEXT_TREND_STATUSES:
            raise ValueError(
                "Unknown cross-context trend status: "
                f"{self.status!r}."
            )

        members = _tuple_unique(
            "member_result_ids",
            self.member_result_ids,
        )
        if not members:
            raise ValueError(
                "CrossContextTrendAssessment requires member results."
            )
        papers = _tuple_unique("paper_ids", self.paper_ids)
        if not papers:
            raise ValueError(
                "CrossContextTrendAssessment requires paper_ids."
            )
        pair_ids = _tuple_unique(
            "pairwise_contrast_ids",
            self.pairwise_contrast_ids,
        )

        direction_sets: dict[str, set[str]] = {}
        for field_name in _DIRECTION_BUCKETS:
            values = _tuple_unique(
                field_name,
                getattr(self, field_name),
            )
            if not set(values).issubset(set(members)):
                raise ValueError(
                    f"{field_name} must be a subset of "
                    "member_result_ids."
                )
            direction_sets[field_name] = set(values)
        direction_names = list(direction_sets)
        for index, left_name in enumerate(direction_names):
            for right_name in direction_names[index + 1:]:
                overlap = (
                    direction_sets[left_name]
                    & direction_sets[right_name]
                )
                if overlap:
                    raise ValueError(
                        "Direction result buckets must be disjoint: "
                        f"{sorted(overlap)!r}."
                    )

        for field_name in (
            "experimental_numeric_result_ids",
            "calculated_numeric_result_ids",
            "reported_claim_result_ids",
        ):
            values = _tuple_unique(
                field_name,
                getattr(self, field_name),
            )
            if not set(values).issubset(set(members)):
                raise ValueError(
                    f"{field_name} must be a subset of "
                    "member_result_ids."
                )

        pair_sets: dict[str, set[str]] = {}
        for field_name in _PAIR_ROLE_BUCKETS:
            values = _tuple_unique(
                field_name,
                getattr(self, field_name),
            )
            if not set(values).issubset(set(pair_ids)):
                raise ValueError(
                    f"{field_name} must be a subset of "
                    "pairwise_contrast_ids."
                )
            pair_sets[field_name] = set(values)

        pair_names = list(pair_sets)
        for index, left_name in enumerate(pair_names):
            for right_name in pair_names[index + 1:]:
                overlap = (
                    pair_sets[left_name] & pair_sets[right_name]
                )
                if overlap:
                    raise ValueError(
                        "Pair role buckets must be disjoint: "
                        f"{sorted(overlap)!r}."
                    )
        if set().union(*pair_sets.values()) != set(pair_ids):
            raise ValueError(
                "Pair role buckets must exactly cover "
                "pairwise_contrast_ids."
            )

        if self.status == "reversed":
            if not self.reversal_pair_ids:
                raise ValueError(
                    "reversed assessment requires reversal_pair_ids."
                )
        elif self.reversal_pair_ids:
            raise ValueError(
                "Any reversal pair requires status='reversed'; "
                "majority vote is forbidden."
            )

        if self.status == "repeated":
            if not self.repeated_pair_ids:
                raise ValueError(
                    "repeated assessment requires repeated_pair_ids."
                )
            if len(self.paper_ids) < 2:
                raise ValueError(
                    "repeated assessment requires at least two papers."
                )
        if self.status == "context_specific":
            if not self.context_specific_pair_ids:
                raise ValueError(
                    "context_specific assessment requires "
                    "context_specific_pair_ids."
                )
        if self.status == "insufficient":
            if self.repeated_pair_ids or self.context_specific_pair_ids:
                raise ValueError(
                    "insufficient assessment cannot contain resolved "
                    "repeated/context-specific pairs."
                )

        expected_relation = stable_trend_relation_id(
            independent_variable_key=self.independent_variable_key,
            dependent_observable_key=self.dependent_observable_key,
            control_family=self.control_family,
            observable_semantics=self.observable_semantics,
        )
        if self.relation_id != expected_relation:
            raise ValueError(
                "CrossContextTrendAssessment relation_id mismatch."
            )
        expected_assessment = stable_cross_context_assessment_id(
            assessment_semantics_id=self.assessment_semantics_id,
            relation_id=self.relation_id,
            member_result_ids=self.member_result_ids,
        )
        if self.assessment_id != expected_assessment:
            raise ValueError(
                "CrossContextTrendAssessment assessment_id "
                "is not stable."
            )

        _tuple_unique(
            "differentiating_dimensions",
            self.differentiating_dimensions,
        )
        _tuple_unique(
            "unresolved_dimensions",
            self.unresolved_dimensions,
        )
        _tuple_unique("reason_codes", self.reason_codes)
        if not self.reason_codes:
            raise ValueError(
                "CrossContextTrendAssessment requires reason_codes."
            )

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in (
            "member_result_ids",
            "paper_ids",
            "pairwise_contrast_ids",
            *_DIRECTION_BUCKETS,
            "experimental_numeric_result_ids",
            "calculated_numeric_result_ids",
            "reported_claim_result_ids",
            *_PAIR_ROLE_BUCKETS,
            "differentiating_dimensions",
            "unresolved_dimensions",
            "reason_codes",
        ):
            row[key] = list(row[key])
        return row


@dataclass(frozen=True)
class CrossContextTrendSource:
    local_results: tuple[PaperLocalTrendResult, ...]
    comparison_context_rows: tuple[Mapping[str, Any], ...] = ()
    method_context_rows: tuple[Mapping[str, Any], ...] = ()
    graphs_by_paper: Mapping[str, nx.Graph] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.local_results:
            raise ValueError(
                "CrossContextTrendSource requires local_results."
            )
        result_ids = [
            result.result_id for result in self.local_results
        ]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError(
                "CrossContextTrendSource local result IDs "
                "must be unique."
            )

    @property
    def paper_ids(self) -> tuple[str, ...]:
        return tuple(sorted({
            result.paper_id
            for result in self.local_results
        }))

    @property
    def available_inputs(self) -> frozenset[str]:
        values = {"paper_local_trend_results"}
        if self.comparison_context_rows:
            values.add("comparison_context")
        if self.method_context_rows:
            values.add("method_context")
        if self.graphs_by_paper:
            values.add("canonical_graph")
        return frozenset(values)


ProjectTrendContexts = Callable[
    [CrossContextTrendSource],
    list[TrendContextProfile],
]


@dataclass(frozen=True)
class CrossContextTrendAdapter:
    adapter_id: str
    domain_profile_id: str
    context_semantics_id: str
    context_dimensions: tuple[str, ...]
    required_inputs: frozenset[str]
    project_contexts_fn: ProjectTrendContexts = field(repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_id", self.adapter_id),
            ("domain_profile_id", self.domain_profile_id),
            ("context_semantics_id", self.context_semantics_id),
        ):
            _require_nonempty(
                f"CrossContextTrendAdapter {name}",
                value,
            )
        if not self.context_dimensions:
            raise ValueError(
                "CrossContextTrendAdapter context_dimensions "
                "must not be empty."
            )
        _tuple_unique(
            "CrossContextTrendAdapter context_dimensions",
            self.context_dimensions,
        )
        if any(not str(value).strip() for value in self.required_inputs):
            raise ValueError(
                "CrossContextTrendAdapter required_inputs "
                "must not contain empty values."
            )
        if "paper_local_trend_results" not in self.required_inputs:
            raise ValueError(
                "CrossContextTrendAdapter must require "
                "paper_local_trend_results."
            )

    def project_contexts(
        self,
        source: CrossContextTrendSource,
    ) -> list[TrendContextProfile]:
        missing = self.required_inputs - source.available_inputs
        if missing:
            raise ValueError(
                "Cross-context trend adapter is missing required "
                f"source inputs: {sorted(missing)!r}."
            )
        profiles = self.project_contexts_fn(source)

        expected_dimensions = set(self.context_dimensions)
        seen_result_ids: set[str] = set()
        for profile in profiles:
            if profile.domain_profile_id != self.domain_profile_id:
                raise ValueError(
                    "Cross-context profile/domain mismatch: "
                    f"{profile.domain_profile_id!r} != "
                    f"{self.domain_profile_id!r}."
                )
            if profile.context_semantics_id != self.context_semantics_id:
                raise ValueError(
                    "Cross-context profile/context semantics "
                    "mismatch."
                )
            found_dimensions = {
                item.name for item in profile.dimensions
            }
            if found_dimensions != expected_dimensions:
                raise ValueError(
                    "Cross-context profile dimension contract "
                    "mismatch: "
                    f"expected={sorted(expected_dimensions)!r}, "
                    f"found={sorted(found_dimensions)!r}."
                )
            if profile.local_result_id in seen_result_ids:
                raise ValueError(
                    "Cross-context adapter emitted multiple profiles "
                    "for one local result: "
                    f"{profile.local_result_id!r}."
                )
            seen_result_ids.add(profile.local_result_id)

        expected_result_ids = {
            result.result_id
            for result in source.local_results
        }
        if seen_result_ids != expected_result_ids:
            raise ValueError(
                "Cross-context adapter must project exactly one "
                "profile per PaperLocalTrendResult."
            )
        return profiles


@dataclass(frozen=True)
class CrossContextTrendAudit:
    contract_semantics_id: str
    local_result_count: int
    context_profile_count: int
    pairwise_contrast_count: int
    assessment_count: int
    status_counts: dict[str, int]
    direction_relation_counts: dict[str, int]
    context_relation_counts: dict[str, int]
    issues: tuple[str, ...]
    structural_gate: bool

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = list(self.issues)
        return row


def _expected_direction_bucket(direction: str) -> str:
    return {
        "positive": "positive_result_ids",
        "negative": "negative_result_ids",
        "non_monotonic": "non_monotonic_result_ids",
        "unchanged": "unchanged_result_ids",
    }.get(direction, "unspecified_result_ids")


def audit_cross_context_trends(
    *,
    local_results: list[PaperLocalTrendResult],
    profiles: list[TrendContextProfile],
    contrasts: list[PairwiseTrendContrast],
    assessments: list[CrossContextTrendAssessment],
) -> CrossContextTrendAudit:
    issues: list[str] = []

    result_by_id = {
        result.result_id: result
        for result in local_results
    }
    if len(result_by_id) != len(local_results):
        issues.append("duplicate_local_result_id")

    profile_by_result: dict[str, TrendContextProfile] = {}
    profile_by_id: dict[str, TrendContextProfile] = {}
    for profile in profiles:
        if profile.context_profile_id in profile_by_id:
            issues.append(
                f"duplicate_context_profile_id:{profile.context_profile_id}"
            )
        profile_by_id[profile.context_profile_id] = profile

        if profile.local_result_id in profile_by_result:
            issues.append(
                "multiple_context_profiles_for_local_result:"
                f"{profile.local_result_id}"
            )
        profile_by_result[profile.local_result_id] = profile

        result = result_by_id.get(profile.local_result_id)
        if result is None:
            issues.append(
                "context_profile_unknown_local_result:"
                f"{profile.local_result_id}"
            )
            continue

        for field_name in (
            "paper_id",
            "independent_variable_key",
            "dependent_observable_key",
            "control_family",
            "observable_semantics",
            "result_lane",
            "direction",
            "shape",
        ):
            if getattr(profile, field_name) != getattr(
                result,
                field_name,
            ):
                issues.append(
                    "context_profile_local_result_mismatch:"
                    f"{profile.local_result_id}:{field_name}"
                )
        if set(profile.evidence_kinds) != set(result.evidence_kinds):
            issues.append(
                "context_profile_evidence_kind_mismatch:"
                f"{profile.local_result_id}"
            )
        if set(profile.member_trend_ids) != set(result.member_trend_ids):
            issues.append(
                "context_profile_member_trend_mismatch:"
                f"{profile.local_result_id}"
            )

    if set(profile_by_result) != set(result_by_id):
        issues.append("context_profile_coverage_mismatch")

    contrast_by_id: dict[str, PairwiseTrendContrast] = {}
    for contrast in contrasts:
        if contrast.contrast_id in contrast_by_id:
            issues.append(
                f"duplicate_pairwise_contrast_id:{contrast.contrast_id}"
            )
        contrast_by_id[contrast.contrast_id] = contrast

        left = profile_by_id.get(contrast.left_context_profile_id)
        right = profile_by_id.get(contrast.right_context_profile_id)
        if left is None or right is None:
            issues.append(
                f"contrast_missing_context_profile:{contrast.contrast_id}"
            )
            continue

        if left.local_result_id != contrast.left_result_id:
            issues.append(
                f"contrast_left_result_mismatch:{contrast.contrast_id}"
            )
        if right.local_result_id != contrast.right_result_id:
            issues.append(
                f"contrast_right_result_mismatch:{contrast.contrast_id}"
            )
        if left.paper_id != contrast.left_paper_id:
            issues.append(
                f"contrast_left_paper_mismatch:{contrast.contrast_id}"
            )
        if right.paper_id != contrast.right_paper_id:
            issues.append(
                f"contrast_right_paper_mismatch:{contrast.contrast_id}"
            )
        if (
            left.relation_id != contrast.relation_id
            or right.relation_id != contrast.relation_id
        ):
            issues.append(
                f"cross_relation_pairwise_contrast:{contrast.contrast_id}"
            )

        expected_direction = classify_direction_relation(
            left.direction,
            right.direction,
        )
        if contrast.direction_relation != expected_direction:
            issues.append(
                "contrast_direction_relation_mismatch:"
                f"{contrast.contrast_id}"
            )
        expected_shape = classify_shape_relation(
            left.shape,
            right.shape,
        )
        if contrast.shape_relation != expected_shape:
            issues.append(
                f"contrast_shape_relation_mismatch:{contrast.contrast_id}"
            )
        expected_kind = classify_evidence_kind_relation(
            left.evidence_kinds,
            right.evidence_kinds,
        )
        if contrast.evidence_kind_relation != expected_kind:
            issues.append(
                "contrast_evidence_kind_relation_mismatch:"
                f"{contrast.contrast_id}"
            )

    assessment_by_id: dict[str, CrossContextTrendAssessment] = {}
    result_memberships: list[str] = []
    pair_memberships: list[str] = []

    for assessment in assessments:
        if assessment.assessment_id in assessment_by_id:
            issues.append(
                "duplicate_cross_context_assessment_id:"
                f"{assessment.assessment_id}"
            )
        assessment_by_id[assessment.assessment_id] = assessment

        member_results: list[PaperLocalTrendResult] = []
        for result_id in assessment.member_result_ids:
            result = result_by_id.get(result_id)
            if result is None:
                issues.append(
                    f"assessment_unknown_result:"
                    f"{assessment.assessment_id}:{result_id}"
                )
                continue
            member_results.append(result)
            result_memberships.append(result_id)

            profile = profile_by_result.get(result_id)
            if (
                profile is None
                or profile.relation_id != assessment.relation_id
            ):
                issues.append(
                    f"assessment_relation_mismatch:"
                    f"{assessment.assessment_id}:{result_id}"
                )

        observed_papers = {
            result.paper_id
            for result in member_results
        }
        if observed_papers != set(assessment.paper_ids):
            issues.append(
                "assessment_paper_membership_mismatch:"
                f"{assessment.assessment_id}"
            )

        for contrast_id in assessment.pairwise_contrast_ids:
            contrast = contrast_by_id.get(contrast_id)
            if contrast is None:
                issues.append(
                    f"assessment_unknown_contrast:"
                    f"{assessment.assessment_id}:{contrast_id}"
                )
                continue
            pair_memberships.append(contrast_id)
            if contrast.relation_id != assessment.relation_id:
                issues.append(
                    f"assessment_contrast_relation_mismatch:"
                    f"{assessment.assessment_id}:{contrast_id}"
                )
            if not {
                contrast.left_result_id,
                contrast.right_result_id,
            }.issubset(set(assessment.member_result_ids)):
                issues.append(
                    f"assessment_contrast_member_mismatch:"
                    f"{assessment.assessment_id}:{contrast_id}"
                )

        expected_direction_sets = {
            field_name: set()
            for field_name in _DIRECTION_BUCKETS
        }
        for result in member_results:
            expected_direction_sets[
                _expected_direction_bucket(result.direction)
            ].add(result.result_id)
        for field_name, expected in expected_direction_sets.items():
            if set(getattr(assessment, field_name)) != expected:
                issues.append(
                    "assessment_direction_bucket_mismatch:"
                    f"{assessment.assessment_id}:{field_name}"
                )

        kind_field = {
            "experimental_numeric":
                "experimental_numeric_result_ids",
            "calculated_numeric":
                "calculated_numeric_result_ids",
            "reported_claim":
                "reported_claim_result_ids",
        }
        for kind, field_name in kind_field.items():
            expected = {
                result.result_id
                for result in member_results
                if kind in result.evidence_kinds
            }
            if set(getattr(assessment, field_name)) != expected:
                issues.append(
                    "assessment_evidence_kind_bucket_mismatch:"
                    f"{assessment.assessment_id}:{field_name}"
                )

        reversal_ids = set(assessment.reversal_pair_ids)
        actual_reversal_ids = {
            contrast_id
            for contrast_id in assessment.pairwise_contrast_ids
            if (
                contrast_id in contrast_by_id
                and contrast_by_id[
                    contrast_id
                ].direction_relation == "opposite_direction"
            )
        }
        if reversal_ids != actual_reversal_ids:
            issues.append(
                f"assessment_reversal_pair_mismatch:"
                f"{assessment.assessment_id}"
            )

        if actual_reversal_ids and assessment.status != "reversed":
            issues.append(
                f"assessment_status_must_be_reversed:"
                f"{assessment.assessment_id}"
            )
        if (
            assessment.status == "repeated"
            and len(observed_papers) < 2
        ):
            issues.append(
                f"same_paper_repetition_forbidden:"
                f"{assessment.assessment_id}"
            )

    if sorted(result_memberships) != sorted(result_by_id):
        issues.append("assessment_local_result_coverage_mismatch")
    if len(result_memberships) != len(set(result_memberships)):
        issues.append(
            "local_result_assigned_to_multiple_assessments"
        )

    if sorted(pair_memberships) != sorted(contrast_by_id):
        issues.append("assessment_contrast_coverage_mismatch")
    if len(pair_memberships) != len(set(pair_memberships)):
        issues.append(
            "pairwise_contrast_assigned_to_multiple_assessments"
        )

    status_counts = Counter(
        assessment.status for assessment in assessments
    )
    direction_counts = Counter(
        contrast.direction_relation for contrast in contrasts
    )
    context_counts = Counter(
        contrast.context_relation for contrast in contrasts
    )

    unique_issues = tuple(sorted(set(issues)))
    return CrossContextTrendAudit(
        contract_semantics_id=
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        local_result_count=len(local_results),
        context_profile_count=len(profiles),
        pairwise_contrast_count=len(contrasts),
        assessment_count=len(assessments),
        status_counts=dict(sorted(status_counts.items())),
        direction_relation_counts=dict(
            sorted(direction_counts.items())
        ),
        context_relation_counts=dict(
            sorted(context_counts.items())
        ),
        issues=unique_issues,
        structural_gate=not unique_issues,
    )
