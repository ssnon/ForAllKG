from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping

import networkx as nx


TrendEvidenceKind = Literal[
    "experimental_numeric",
    "calculated_numeric",
    "reported_claim",
]
TrendResultLane = Literal["numeric", "claim"]

TREND_EVIDENCE_KINDS = frozenset({
    "experimental_numeric",
    "calculated_numeric",
    "reported_claim",
})
TREND_RESULT_LANES = frozenset({"numeric", "claim"})


@dataclass(frozen=True)
class TrendEvidenceAnnotation:
    trend_id: str
    paper_id: str
    precision_semantics_id: str
    evidence_kind: TrendEvidenceKind
    classification_basis: str
    control_family: str
    observable_semantics: str
    trend_subject_ids: tuple[str, ...] = ()
    reference_subject_ids: tuple[str, ...] = ()
    source_control_value_text: str = ""
    canonical_control_value_numeric: float | None = None
    canonical_control_unit: str = ""
    normalization_transform: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("trend_id", self.trend_id),
            ("paper_id", self.paper_id),
            ("precision_semantics_id", self.precision_semantics_id),
            ("classification_basis", self.classification_basis),
            ("control_family", self.control_family),
            ("observable_semantics", self.observable_semantics),
        ):
            if not str(value).strip():
                raise ValueError(
                    f"TrendEvidenceAnnotation {name} must not be empty."
                )
        if self.evidence_kind not in TREND_EVIDENCE_KINDS:
            raise ValueError(
                "Unknown TrendEvidenceAnnotation evidence_kind: "
                f"{self.evidence_kind!r}."
            )
        if (
            self.canonical_control_value_numeric is not None
            and not isinstance(self.canonical_control_value_numeric, (int, float))
        ):
            raise ValueError(
                "canonical_control_value_numeric must be numeric when populated."
            )

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["trend_subject_ids"] = list(self.trend_subject_ids)
        row["reference_subject_ids"] = list(self.reference_subject_ids)
        return row


@dataclass(frozen=True)
class PaperLocalTrendResult:
    result_id: str
    paper_id: str
    domain_profile_id: str
    trend_semantics_id: str
    precision_semantics_id: str
    result_lane: TrendResultLane
    independent_variable_key: str
    dependent_observable_key: str
    direction: str
    shape: str
    control_family: str
    observable_semantics: str
    member_trend_ids: tuple[str, ...]
    evidence_kinds: tuple[str, ...]
    trend_subject_ids: tuple[str, ...] = ()
    reference_subject_ids: tuple[str, ...] = ()
    source_claim_ids: tuple[str, ...] = ()
    source_measurement_ids: tuple[str, ...] = ()
    source_measurement_result_ids: tuple[str, ...] = ()
    source_calculation_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    support_mention_count: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("paper_id", self.paper_id),
            ("domain_profile_id", self.domain_profile_id),
            ("trend_semantics_id", self.trend_semantics_id),
            ("precision_semantics_id", self.precision_semantics_id),
            ("independent_variable_key", self.independent_variable_key),
            ("dependent_observable_key", self.dependent_observable_key),
            ("direction", self.direction),
            ("shape", self.shape),
            ("control_family", self.control_family),
            ("observable_semantics", self.observable_semantics),
        ):
            if not str(value).strip():
                raise ValueError(f"PaperLocalTrendResult {name} must not be empty.")
        if self.result_lane not in TREND_RESULT_LANES:
            raise ValueError(
                f"Unknown PaperLocalTrendResult result_lane: {self.result_lane!r}."
            )
        if not self.member_trend_ids:
            raise ValueError(
                "PaperLocalTrendResult requires at least one member trend."
            )
        if len(set(self.member_trend_ids)) != len(self.member_trend_ids):
            raise ValueError(
                "PaperLocalTrendResult member_trend_ids must be unique."
            )
        if self.support_mention_count != len(self.member_trend_ids):
            raise ValueError(
                "support_mention_count must equal member_trend_ids length."
            )
        if not set(self.evidence_kinds).issubset(TREND_EVIDENCE_KINDS):
            raise ValueError("PaperLocalTrendResult has unknown evidence kind.")

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in (
            "member_trend_ids",
            "evidence_kinds",
            "trend_subject_ids",
            "reference_subject_ids",
            "source_claim_ids",
            "source_measurement_ids",
            "source_measurement_result_ids",
            "source_calculation_ids",
            "source_node_ids",
        ):
            row[key] = list(row[key])
        return row


def stable_local_trend_result_id(
    *,
    paper_id: str,
    result_lane: str,
    independent_variable_key: str,
    dependent_observable_key: str,
    direction: str,
    shape: str,
    member_trend_ids: Iterable[str],
) -> str:
    members = tuple(sorted(set(map(str, member_trend_ids))))
    if not members:
        raise ValueError(
            "stable local trend result IDs require at least one member."
        )
    payload = "|".join((
        paper_id,
        result_lane,
        independent_variable_key,
        dependent_observable_key,
        direction,
        shape,
        *members,
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"paper-local-trend:{digest}"


AnnotateTrendEvidence = Callable[
    [Mapping[str, Any], nx.Graph],
    TrendEvidenceAnnotation,
]
ConsolidateTrendEvidence = Callable[
    [
        list[Mapping[str, Any]],
        list[TrendEvidenceAnnotation],
        dict[str, nx.Graph],
    ],
    list[PaperLocalTrendResult],
]


@dataclass(frozen=True)
class TrendPrecisionAdapter:
    adapter_id: str
    domain_profile_id: str
    trend_semantics_id: str
    precision_semantics_id: str
    annotate_fn: AnnotateTrendEvidence = field(repr=False)
    consolidate_fn: ConsolidateTrendEvidence = field(repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_id", self.adapter_id),
            ("domain_profile_id", self.domain_profile_id),
            ("trend_semantics_id", self.trend_semantics_id),
            ("precision_semantics_id", self.precision_semantics_id),
        ):
            if not value.strip():
                raise ValueError(f"TrendPrecisionAdapter {name} must not be empty.")

    def annotate(
        self,
        row: Mapping[str, Any],
        graph: nx.Graph,
    ) -> TrendEvidenceAnnotation:
        annotation = self.annotate_fn(row, graph)
        if annotation.precision_semantics_id != self.precision_semantics_id:
            raise ValueError("Trend precision annotation semantics mismatch.")
        return annotation

    def consolidate(
        self,
        rows: list[Mapping[str, Any]],
        annotations: list[TrendEvidenceAnnotation],
        graphs: dict[str, nx.Graph],
    ) -> list[PaperLocalTrendResult]:
        results = self.consolidate_fn(rows, annotations, graphs)
        for result in results:
            if result.precision_semantics_id != self.precision_semantics_id:
                raise ValueError("Paper-local trend result semantics mismatch.")
        return results


@dataclass(frozen=True)
class TrendPrecisionAudit:
    domain_profile_id: str
    trend_semantics_id: str
    precision_semantics_id: str
    evidence_count: int
    annotation_count: int
    local_result_count: int
    evidence_kind_counts: dict[str, int]
    result_lane_counts: dict[str, int]
    control_family_counts: dict[str, int]
    observable_semantics_counts: dict[str, int]
    duplicate_claim_mentions_collapsed: int
    claim_result_count_with_multiple_mentions: int
    issues: tuple[str, ...]
    structural_gate: bool

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = list(self.issues)
        return row


def audit_trend_precision(
    *,
    evidence_rows: list[Mapping[str, Any]],
    annotations: list[TrendEvidenceAnnotation],
    results: list[PaperLocalTrendResult],
    adapter: TrendPrecisionAdapter,
) -> TrendPrecisionAudit:
    issues: list[str] = []
    evidence_ids = [
        str(row.get("trend_id", "")).strip() for row in evidence_rows
    ]
    if any(not value for value in evidence_ids):
        issues.append("evidence_missing_trend_id")
    if len(evidence_ids) != len(set(evidence_ids)):
        issues.append("duplicate_evidence_trend_id")

    annotation_ids = [row.trend_id for row in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        issues.append("duplicate_annotation_trend_id")
    if set(annotation_ids) != set(evidence_ids):
        issues.append("annotation_coverage_mismatch")

    annotation_by_id = {row.trend_id: row for row in annotations}
    evidence_by_id = {
        str(row.get("trend_id", "")): row for row in evidence_rows
    }

    memberships: list[str] = []
    result_ids: set[str] = set()
    for result in results:
        if result.result_id in result_ids:
            issues.append(f"duplicate_local_result_id:{result.result_id}")
        result_ids.add(result.result_id)
        memberships.extend(result.member_trend_ids)

        for trend_id in result.member_trend_ids:
            if trend_id not in evidence_by_id:
                issues.append(
                    f"missing_local_result_member:{result.result_id}:{trend_id}"
                )
                continue
            evidence = evidence_by_id[trend_id]
            annotation = annotation_by_id.get(trend_id)
            if annotation is None:
                continue
            if str(evidence.get("paper_id", "")) != result.paper_id:
                issues.append(
                    f"cross_paper_local_result:{result.result_id}:{trend_id}"
                )
            if annotation.control_family != result.control_family:
                issues.append(
                    "local_result_control_family_mismatch:"
                    f"{result.result_id}:{trend_id}"
                )
            if annotation.observable_semantics != result.observable_semantics:
                issues.append(
                    "local_result_observable_semantics_mismatch:"
                    f"{result.result_id}:{trend_id}"
                )
            expected_lane = (
                "numeric"
                if str(evidence.get("evidence_basis", "")).startswith(
                    "controlled_numeric"
                )
                else "claim"
            )
            if expected_lane != result.result_lane:
                issues.append(
                    f"local_result_lane_mismatch:{result.result_id}:{trend_id}"
                )

    if sorted(memberships) != sorted(evidence_ids):
        issues.append("local_result_membership_coverage_mismatch")
    if len(memberships) != len(set(memberships)):
        issues.append("trend_evidence_assigned_to_multiple_local_results")

    for annotation in annotations:
        evidence = evidence_by_id.get(annotation.trend_id, {})
        basis = str(evidence.get("evidence_basis", ""))
        if basis.startswith("controlled_numeric"):
            if annotation.evidence_kind not in {
                "experimental_numeric", "calculated_numeric"
            }:
                issues.append(
                    f"numeric_evidence_kind_invalid:{annotation.trend_id}"
                )
        elif annotation.evidence_kind != "reported_claim":
            issues.append(f"claim_evidence_kind_invalid:{annotation.trend_id}")
        if (
            str(evidence.get("dependent_observable_key", ""))
            == "relative_sers_intensity_ratio"
            and annotation.observable_semantics
            == "formal_sers_enhancement_factor"
        ):
            issues.append(
                f"relative_ratio_promoted_to_formal_ef:{annotation.trend_id}"
            )

    evidence_kind_counts = Counter(row.evidence_kind for row in annotations)
    result_lane_counts = Counter(row.result_lane for row in results)
    control_family_counts = Counter(row.control_family for row in annotations)
    observable_semantics_counts = Counter(
        row.observable_semantics for row in annotations
    )
    duplicate_claim_mentions_collapsed = sum(
        max(0, len(row.member_trend_ids) - 1)
        for row in results
        if row.result_lane == "claim"
    )
    multi_claim = sum(
        row.result_lane == "claim" and len(row.member_trend_ids) > 1
        for row in results
    )

    return TrendPrecisionAudit(
        domain_profile_id=adapter.domain_profile_id,
        trend_semantics_id=adapter.trend_semantics_id,
        precision_semantics_id=adapter.precision_semantics_id,
        evidence_count=len(evidence_rows),
        annotation_count=len(annotations),
        local_result_count=len(results),
        evidence_kind_counts=dict(sorted(evidence_kind_counts.items())),
        result_lane_counts=dict(sorted(result_lane_counts.items())),
        control_family_counts=dict(sorted(control_family_counts.items())),
        observable_semantics_counts=dict(
            sorted(observable_semantics_counts.items())
        ),
        duplicate_claim_mentions_collapsed=duplicate_claim_mentions_collapsed,
        claim_result_count_with_multiple_mentions=multi_claim,
        issues=tuple(sorted(set(issues))),
        structural_gate=not issues,
    )
