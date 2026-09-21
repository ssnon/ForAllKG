from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_co_measurement_resolver import (
    CoMeasurementContextCandidate,
    CoMeasurementWitnessResolutionSet,
)
from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirementSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CoMeasurementCriticIssueCode = Literal[
    "TARGET_METRIC_IDENTITY_UNRESOLVED",
    "ANCHOR_METRIC_IDENTITY_UNRESOLVED",
    "EXPLICIT_MEASUREMENT_DERIVATION_PRESENT",
    "DERIVATION_LANGUAGE_CUE_UNRESOLVED",
    "SHARED_CALCULATION_DEPENDENCY_UNRESOLVED",
]


CoMeasurementContextSignal = Literal[
    "SHARED_EXPERIMENT_CONTEXT",
    "SAME_SUBJECT_SEPARATE_PROVIDER_CONTEXT",
    "SHARED_MEASUREMENT_GROUP_CONTEXT",
]


class CoMeasurementCandidateCritique(StrictModel):
    requirement_id: str
    output_experiment_id: str
    source_paper_id: str

    target_candidate_id: str
    anchor_candidate_id: str
    target_measurement_node_id: str
    anchor_measurement_node_id: str

    target_metric_id: str
    anchor_metric_id: str

    target_metric_identity_coverage: float
    anchor_metric_identity_coverage: float
    target_matched_tokens: list[str] = Field(default_factory=list)
    target_unmatched_tokens: list[str] = Field(default_factory=list)
    anchor_matched_tokens: list[str] = Field(default_factory=list)
    anchor_unmatched_tokens: list[str] = Field(default_factory=list)

    direct_derivation_present: bool
    derivation_language_cue_present: bool

    target_provider_types: list[str] = Field(default_factory=list)
    anchor_provider_types: list[str] = Field(default_factory=list)
    shared_provider_types: list[str] = Field(default_factory=list)

    same_subject_id: bool
    shared_provider_present: bool
    shared_measurement_group_present: bool

    issues: list[CoMeasurementCriticIssueCode] = Field(
        default_factory=list
    )
    context_signals: list[CoMeasurementContextSignal] = Field(
        default_factory=list
    )

    exact_metric_identity_on_both_sides: bool
    eligible_for_independence_followup: bool

    candidate_is_independence_witness: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    independence_certification_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class CoMeasurementRequirementCritique(StrictModel):
    requirement_id: str
    output_experiment_id: str

    candidate_count: int = Field(ge=0)
    target_metric_identity_exact_count: int = Field(ge=0)
    anchor_metric_identity_exact_count: int = Field(ge=0)
    both_metric_identity_exact_count: int = Field(ge=0)

    explicit_derivation_count: int = Field(ge=0)
    derivation_language_cue_count: int = Field(ge=0)
    shared_calculation_count: int = Field(ge=0)
    shared_experiment_count: int = Field(ge=0)
    same_subject_separate_provider_count: int = Field(ge=0)
    shared_measurement_group_count: int = Field(ge=0)

    eligible_for_independence_followup_count: int = Field(ge=0)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    signal_counts: dict[str, int] = Field(default_factory=dict)

    candidates: list[CoMeasurementCandidateCritique] = Field(
        default_factory=list
    )

    diagnostic_only: Literal[True] = True
    measurement_independence_verified: Literal[False] = False
    independence_certification_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class CoMeasurementCriticReport(StrictModel):
    schema_version: Literal[
        "co-measurement-candidate-critic-v1"
    ] = "co-measurement-candidate-critic-v1"

    requirement_count: int = Field(ge=0)
    resolution_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)

    both_metric_identity_exact_count: int = Field(ge=0)
    eligible_for_independence_followup_count: int = Field(ge=0)

    issue_counts: dict[str, int] = Field(default_factory=dict)
    signal_counts: dict[str, int] = Field(default_factory=dict)
    requirements: list[CoMeasurementRequirementCritique] = Field(
        default_factory=list
    )

    metric_identity_uses_source_expression: Literal[False] = False
    diagnostic_only: Literal[True] = True
    candidate_is_independence_witness: Literal[False] = False
    measurement_independence_verified_count: Literal[0] = 0
    independence_certification_authority: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    experiment_selection_performed: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


_DASHES = "‐-‒–—−"
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by",
    "for", "from", "in", "into", "is", "of", "on", "or", "that",
    "the", "their", "to", "under", "with",
}

_DERIVATION_PATTERNS = (
    re.compile(r"\bderived\s+from\b", re.I),
    re.compile(r"\bcalculated\s+from\b", re.I),
    re.compile(r"\bcomputed\s+from\b", re.I),
    re.compile(r"\bestimated\s+from\b", re.I),
    re.compile(r"\bpredicted\b.{0,120}\bfrom\b", re.I),
    re.compile(r"\busing\b.{0,160}\bas\b", re.I),
)


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    for dash in _DASHES:
        text = text.replace(dash, "-")
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: object) -> set[str]:
    raw = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", _normalize(value))
    out = set()
    for token in raw:
        parts = [part for part in token.split("-") if part]
        candidates = list(parts)
        if len(parts) > 1:
            candidates.append("".join(parts))
        for candidate in candidates:
            if (
                len(candidate) > 1
                and candidate not in _STOPWORDS
            ):
                out.add(candidate)
    return out


def _identity_tokens(graph: nx.Graph, measurement_id: str) -> set[str]:
    attrs = dict(graph.nodes[measurement_id])
    return _tokens(
        " ".join(
            [
                str(attrs.get("metric_id", "") or ""),
                str(attrs.get("metric", "") or ""),
                str(attrs.get("label", "") or ""),
            ]
        )
    )


def _observable_match(
    graph: nx.Graph,
    *,
    measurement_id: str,
    observable: str,
) -> tuple[float, list[str], list[str]]:
    need = _tokens(observable)
    identity = _identity_tokens(graph, measurement_id)
    if not need:
        return 0.0, [], []

    matched = sorted(need & identity)
    unmatched = sorted(need - identity)
    return (
        float(len(matched) / len(need)),
        matched,
        unmatched,
    )


def _relation(attrs: dict) -> str:
    return str(
        attrs.get("relation")
        or attrs.get("title")
        or ""
    ).strip().upper()


def _edge_attrs(
    graph: nx.Graph,
    source: str,
    target: str,
) -> list[dict]:
    if not graph.has_edge(source, target):
        return []
    if graph.is_multigraph():
        raw = graph.get_edge_data(source, target) or {}
        return [dict(attrs) for attrs in raw.values()]
    return [dict(graph.get_edge_data(source, target) or {})]


def _direct_derivation_present(
    graph: nx.Graph,
    left: str,
    right: str,
) -> bool:
    for source, target in ((left, right), (right, left)):
        for attrs in _edge_attrs(graph, source, target):
            if _relation(attrs) == "DERIVED_FROM":
                return True
    return False


def _measurement_groups(
    graph: nx.Graph,
    measurement_id: str,
) -> set[str]:
    groups = set()
    if graph.is_multigraph():
        edges = graph.out_edges(
            measurement_id,
            keys=True,
            data=True,
        )
        for _, target, _, attrs in edges:
            if _relation(dict(attrs)) == "IN_MEASUREMENT_GROUP":
                groups.add(str(target))
    else:
        for _, target, attrs in graph.out_edges(
            measurement_id,
            data=True,
        ):
            if _relation(dict(attrs)) == "IN_MEASUREMENT_GROUP":
                groups.add(str(target))
    return groups


def _provider_types(
    graph: nx.Graph,
    provider_ids: list[str],
) -> list[str]:
    return sorted(
        {
            str(graph.nodes[provider_id].get("type", ""))
            for provider_id in provider_ids
            if graph.has_node(provider_id)
            and str(graph.nodes[provider_id].get("type", ""))
        }
    )


def _source_expression(
    graph: nx.Graph,
    measurement_id: str,
) -> str:
    return str(
        graph.nodes[measurement_id].get("source_expression", "")
        or ""
    )


def _derivation_language_cue(
    graph: nx.Graph,
    candidate: CoMeasurementContextCandidate,
) -> bool:
    text = " ".join(
        [
            _source_expression(
                graph,
                candidate.target_measurement_node_id,
            ),
            _source_expression(
                graph,
                candidate.anchor_measurement_node_id,
            ),
        ]
    )
    return any(pattern.search(text) for pattern in _DERIVATION_PATTERNS)


def critique_co_measurement_candidates(
    *,
    graph: nx.Graph,
    requirements: OperationalizationWitnessRequirementSet,
    resolutions: CoMeasurementWitnessResolutionSet,
) -> CoMeasurementCriticReport:
    requirement_by_id = {
        row.requirement_id: row
        for row in requirements.requirements
    }
    resolution_by_id = {
        row.requirement_id: row
        for row in resolutions.resolutions
    }

    if set(requirement_by_id) != set(resolution_by_id):
        raise ValueError(
            "requirement/co-measurement resolution sets differ"
        )

    requirement_rows = []

    for requirement_id in sorted(requirement_by_id):
        requirement = requirement_by_id[requirement_id]
        resolution = resolution_by_id[requirement_id]

        rows = []

        for candidate in resolution.candidates:
            target_coverage, target_matched, target_unmatched = (
                _observable_match(
                    graph,
                    measurement_id=(
                        candidate.target_measurement_node_id
                    ),
                    observable=(
                        requirement.requested_target_observable
                    ),
                )
            )
            anchor_coverage, anchor_matched, anchor_unmatched = (
                _observable_match(
                    graph,
                    measurement_id=(
                        candidate.anchor_measurement_node_id
                    ),
                    observable=requirement.anchor_observable,
                )
            )

            direct_derivation = _direct_derivation_present(
                graph,
                candidate.target_measurement_node_id,
                candidate.anchor_measurement_node_id,
            )
            derivation_cue = _derivation_language_cue(
                graph,
                candidate,
            )

            target_provider_types = _provider_types(
                graph,
                candidate.target_provider_ids,
            )
            anchor_provider_types = _provider_types(
                graph,
                candidate.anchor_provider_ids,
            )
            shared_provider_types = _provider_types(
                graph,
                candidate.shared_provider_ids,
            )

            target_groups = _measurement_groups(
                graph,
                candidate.target_measurement_node_id,
            )
            anchor_groups = _measurement_groups(
                graph,
                candidate.anchor_measurement_node_id,
            )
            shared_group = bool(target_groups & anchor_groups)

            issues: list[CoMeasurementCriticIssueCode] = []
            signals: list[CoMeasurementContextSignal] = []

            if target_coverage < 1.0:
                issues.append(
                    "TARGET_METRIC_IDENTITY_UNRESOLVED"
                )
            if anchor_coverage < 1.0:
                issues.append(
                    "ANCHOR_METRIC_IDENTITY_UNRESOLVED"
                )
            if direct_derivation:
                issues.append(
                    "EXPLICIT_MEASUREMENT_DERIVATION_PRESENT"
                )
            if derivation_cue and not direct_derivation:
                issues.append(
                    "DERIVATION_LANGUAGE_CUE_UNRESOLVED"
                )
            if "Calculation" in shared_provider_types:
                issues.append(
                    "SHARED_CALCULATION_DEPENDENCY_UNRESOLVED"
                )

            if "Experiment" in shared_provider_types:
                signals.append("SHARED_EXPERIMENT_CONTEXT")
            if (
                candidate.same_subject_id
                and not candidate.shared_provider_present
            ):
                signals.append(
                    "SAME_SUBJECT_SEPARATE_PROVIDER_CONTEXT"
                )
            if shared_group:
                signals.append(
                    "SHARED_MEASUREMENT_GROUP_CONTEXT"
                )

            exact_both = bool(
                target_coverage == 1.0
                and anchor_coverage == 1.0
            )

            # Follow-up eligibility is deliberately conservative. It does
            # not certify independence; it only identifies candidates worth
            # a later source-level method/dependency review.
            eligible_followup = bool(
                exact_both
                and not direct_derivation
                and "Calculation" not in shared_provider_types
                and (
                    "Experiment" in shared_provider_types
                    or (
                        candidate.same_subject_id
                        and not candidate.shared_provider_present
                    )
                    or shared_group
                )
            )

            rows.append(
                CoMeasurementCandidateCritique(
                    requirement_id=requirement_id,
                    output_experiment_id=(
                        requirement.output_experiment_id
                    ),
                    source_paper_id=candidate.source_paper_id,
                    target_candidate_id=(
                        candidate.target_candidate_id
                    ),
                    anchor_candidate_id=(
                        candidate.anchor_candidate_id
                    ),
                    target_measurement_node_id=(
                        candidate.target_measurement_node_id
                    ),
                    anchor_measurement_node_id=(
                        candidate.anchor_measurement_node_id
                    ),
                    target_metric_id=candidate.target_metric_id,
                    anchor_metric_id=candidate.anchor_metric_id,
                    target_metric_identity_coverage=(
                        target_coverage
                    ),
                    anchor_metric_identity_coverage=(
                        anchor_coverage
                    ),
                    target_matched_tokens=target_matched,
                    target_unmatched_tokens=target_unmatched,
                    anchor_matched_tokens=anchor_matched,
                    anchor_unmatched_tokens=anchor_unmatched,
                    direct_derivation_present=direct_derivation,
                    derivation_language_cue_present=derivation_cue,
                    target_provider_types=target_provider_types,
                    anchor_provider_types=anchor_provider_types,
                    shared_provider_types=shared_provider_types,
                    same_subject_id=candidate.same_subject_id,
                    shared_provider_present=(
                        candidate.shared_provider_present
                    ),
                    shared_measurement_group_present=shared_group,
                    issues=issues,
                    context_signals=signals,
                    exact_metric_identity_on_both_sides=exact_both,
                    eligible_for_independence_followup=(
                        eligible_followup
                    ),
                )
            )

        issue_counts = Counter(
            issue
            for row in rows
            for issue in row.issues
        )
        signal_counts = Counter(
            signal
            for row in rows
            for signal in row.context_signals
        )

        requirement_rows.append(
            CoMeasurementRequirementCritique(
                requirement_id=requirement_id,
                output_experiment_id=(
                    requirement.output_experiment_id
                ),
                candidate_count=len(rows),
                target_metric_identity_exact_count=sum(
                    row.target_metric_identity_coverage == 1.0
                    for row in rows
                ),
                anchor_metric_identity_exact_count=sum(
                    row.anchor_metric_identity_coverage == 1.0
                    for row in rows
                ),
                both_metric_identity_exact_count=sum(
                    row.exact_metric_identity_on_both_sides
                    for row in rows
                ),
                explicit_derivation_count=sum(
                    row.direct_derivation_present
                    for row in rows
                ),
                derivation_language_cue_count=sum(
                    row.derivation_language_cue_present
                    for row in rows
                ),
                shared_calculation_count=sum(
                    "Calculation" in row.shared_provider_types
                    for row in rows
                ),
                shared_experiment_count=sum(
                    "Experiment" in row.shared_provider_types
                    for row in rows
                ),
                same_subject_separate_provider_count=sum(
                    "SAME_SUBJECT_SEPARATE_PROVIDER_CONTEXT"
                    in row.context_signals
                    for row in rows
                ),
                shared_measurement_group_count=sum(
                    row.shared_measurement_group_present
                    for row in rows
                ),
                eligible_for_independence_followup_count=sum(
                    row.eligible_for_independence_followup
                    for row in rows
                ),
                issue_counts=dict(sorted(issue_counts.items())),
                signal_counts=dict(sorted(signal_counts.items())),
                candidates=rows,
            )
        )

    overall_issues = Counter(
        issue
        for requirement in requirement_rows
        for row in requirement.candidates
        for issue in row.issues
    )
    overall_signals = Counter(
        signal
        for requirement in requirement_rows
        for row in requirement.candidates
        for signal in row.context_signals
    )

    return CoMeasurementCriticReport(
        requirement_count=requirements.requirement_count,
        resolution_count=resolutions.resolution_count,
        candidate_count=sum(
            row.candidate_count
            for row in requirement_rows
        ),
        both_metric_identity_exact_count=sum(
            row.both_metric_identity_exact_count
            for row in requirement_rows
        ),
        eligible_for_independence_followup_count=sum(
            row.eligible_for_independence_followup_count
            for row in requirement_rows
        ),
        issue_counts=dict(sorted(overall_issues.items())),
        signal_counts=dict(sorted(overall_signals.items())),
        requirements=requirement_rows,
    )
