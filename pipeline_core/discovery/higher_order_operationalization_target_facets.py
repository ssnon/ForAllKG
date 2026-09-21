from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_operationalization_resolver import (
    MeasurementOperationalizationCandidate,
    _measurement_candidates,
)
from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirementSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FacetSupportStatus = Literal[
    "EXACT_METRIC_IDENTITY_SUPPORT",
    "PARTIAL_METRIC_IDENTITY_SUPPORT",
    "NO_METRIC_IDENTITY_SUPPORT",
]


class OperationalizationTargetFacet(StrictModel):
    facet_id: str
    original_target: str
    facet_text: str
    head_text: str
    qualifier_text: str
    facet_term: str
    decomposition_mode: Literal[
        "coordinated_head_facet",
        "single_target",
    ]

    decomposition_is_scientific_equivalence: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class TargetFacetMeasurementCandidate(StrictModel):
    facet_id: str
    facet_text: str

    candidate_id: str
    measurement_node_id: str
    metric_id: str
    metric: str
    source_expression: str
    subject_id: str
    source_paper_ids: list[str] = Field(default_factory=list)
    provider_types: list[str] = Field(default_factory=list)

    retrieval_lexical_coverage: float
    metric_identity_coverage: float
    matched_identity_tokens: list[str] = Field(default_factory=list)
    unmatched_facet_tokens: list[str] = Field(default_factory=list)

    exact_metric_identity: bool
    partial_metric_identity: bool

    candidate_is_operationalization_witness: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class TargetFacetResolution(StrictModel):
    requirement_id: str
    output_experiment_id: str
    original_target: str

    facet: OperationalizationTargetFacet
    status: FacetSupportStatus

    retrieval_candidate_count: int = Field(ge=0)
    exact_metric_identity_candidate_count: int = Field(ge=0)
    partial_metric_identity_candidate_count: int = Field(ge=0)
    best_metric_identity_coverage: float

    candidates: list[TargetFacetMeasurementCandidate] = Field(
        default_factory=list
    )

    target_facet_operationalization_verified: Literal[False] = False
    operationalization_bridge_authority: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    selection_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class TargetFacetRequirementResolution(StrictModel):
    requirement_id: str
    output_experiment_id: str
    original_target: str

    decomposition_mode: str
    facet_count: int = Field(ge=1)
    exact_supported_facet_count: int = Field(ge=0)
    partial_supported_facet_count: int = Field(ge=0)
    unsupported_facet_count: int = Field(ge=0)
    all_facets_exactly_supported: bool

    facets: list[TargetFacetResolution] = Field(default_factory=list)

    target_operationalization_verified: Literal[False] = False
    operationalization_bridge_authority: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class TargetFacetResolutionSet(StrictModel):
    schema_version: Literal[
        "target-facet-operationalization-resolution-v1"
    ] = "target-facet-operationalization-resolution-v1"

    requirement_count: int = Field(ge=0)
    resolution_count: int = Field(ge=0)
    total_facet_count: int = Field(ge=0)

    exact_supported_facet_count: int = Field(ge=0)
    partial_supported_facet_count: int = Field(ge=0)
    unsupported_facet_count: int = Field(ge=0)

    status_counts: dict[str, int] = Field(default_factory=dict)
    resolutions: list[TargetFacetRequirementResolution] = Field(
        default_factory=list
    )

    domain_specific_synonymy_inferred: Literal[False] = False
    decomposition_is_scientific_equivalence: Literal[False] = False
    metric_identity_uses_source_expression: Literal[False] = False
    target_operationalization_verified_count: Literal[0] = 0
    operationalization_bridge_authority: Literal[False] = False
    measurement_independence_verified_count: Literal[0] = 0
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


def _stable_id(prefix: str, *parts: object) -> str:
    import hashlib

    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    for dash in _DASHES:
        text = text.replace(dash, "-")
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _surface_tokens(value: object) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", _normalize(value))


def _identity_tokens(value: object) -> set[str]:
    raw = _surface_tokens(value)
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


def decompose_target_facets(
    target: str,
) -> list[OperationalizationTargetFacet]:
    normalized = " ".join(str(target).split()).strip()
    if not normalized:
        raise ValueError("target must be non-empty")

    parts = [
        " ".join(part.split()).strip()
        for part in re.split(
            r"\s*(?:,|\band\b|\bor\b)\s*",
            normalized,
            flags=re.IGNORECASE,
        )
        if " ".join(part.split()).strip()
    ]

    first = _surface_tokens(parts[0]) if parts else []
    later = [_surface_tokens(part) for part in parts[1:]]

    coordinated = bool(
        len(parts) >= 2
        and len(first) >= 2
        and all(len(tokens) == 1 for tokens in later)
    )

    if not coordinated:
        return [
            OperationalizationTargetFacet(
                facet_id=_stable_id(
                    "operationalization_target_facet",
                    normalized,
                    normalized,
                ),
                original_target=normalized,
                facet_text=normalized,
                head_text=normalized,
                qualifier_text="",
                facet_term=normalized,
                decomposition_mode="single_target",
            )
        ]

    head = first[-2]
    first_facet = first[-1]
    qualifiers = first[:-2]
    facet_terms = [first_facet] + [
        tokens[0]
        for tokens in later
    ]

    prefix = " ".join([*qualifiers, head]).strip()
    qualifier_text = " ".join(qualifiers).strip()

    return [
        OperationalizationTargetFacet(
            facet_id=_stable_id(
                "operationalization_target_facet",
                normalized,
                prefix,
                facet_term,
            ),
            original_target=normalized,
            facet_text=" ".join([prefix, facet_term]).strip(),
            head_text=head,
            qualifier_text=qualifier_text,
            facet_term=facet_term,
            decomposition_mode="coordinated_head_facet",
        )
        for facet_term in facet_terms
    ]


def _candidate_identity_text(
    graph: nx.Graph,
    measurement_id: str,
) -> str:
    attrs = dict(graph.nodes[measurement_id])
    return " ".join(
        [
            str(attrs.get("metric_id", "") or ""),
            str(attrs.get("metric", "") or ""),
            str(attrs.get("label", "") or ""),
        ]
    )


def _provider_types(
    graph: nx.Graph,
    candidate: MeasurementOperationalizationCandidate,
) -> list[str]:
    return sorted(
        {
            str(graph.nodes[row.provider_id].get("type", ""))
            for row in candidate.providers
            if graph.has_node(row.provider_id)
            and str(graph.nodes[row.provider_id].get("type", ""))
        }
    )


def _facet_candidate(
    *,
    graph: nx.Graph,
    facet: OperationalizationTargetFacet,
    candidate: MeasurementOperationalizationCandidate,
) -> TargetFacetMeasurementCandidate:
    facet_tokens = _identity_tokens(facet.facet_text)
    metric_tokens = _identity_tokens(
        _candidate_identity_text(
            graph,
            candidate.measurement_node_id,
        )
    )

    matched = sorted(facet_tokens & metric_tokens)
    unmatched = sorted(facet_tokens - metric_tokens)
    coverage = (
        float(len(matched) / len(facet_tokens))
        if facet_tokens
        else 0.0
    )

    return TargetFacetMeasurementCandidate(
        facet_id=facet.facet_id,
        facet_text=facet.facet_text,
        candidate_id=candidate.candidate_id,
        measurement_node_id=candidate.measurement_node_id,
        metric_id=str(candidate.metric_id or ""),
        metric=str(candidate.metric or ""),
        source_expression=str(candidate.source_expression or ""),
        subject_id=str(candidate.subject_id or ""),
        source_paper_ids=list(candidate.source_paper_ids),
        provider_types=_provider_types(graph, candidate),
        retrieval_lexical_coverage=float(candidate.lexical_coverage),
        metric_identity_coverage=coverage,
        matched_identity_tokens=matched,
        unmatched_facet_tokens=unmatched,
        exact_metric_identity=bool(coverage == 1.0),
        partial_metric_identity=bool(0.0 < coverage < 1.0),
    )


def resolve_target_facet_operationalizations(
    *,
    graph: nx.Graph,
    requirements: OperationalizationWitnessRequirementSet,
    minimum_retrieval_coverage: float = 0.25,
    max_retrieval_candidates_per_facet: int = 20000,
    max_output_candidates_per_facet: int = 100,
) -> TargetFacetResolutionSet:
    if not (0.0 < minimum_retrieval_coverage <= 1.0):
        raise ValueError(
            "minimum_retrieval_coverage must be in (0, 1]"
        )
    if max_retrieval_candidates_per_facet < 1:
        raise ValueError(
            "max_retrieval_candidates_per_facet must be >= 1"
        )
    if max_output_candidates_per_facet < 1:
        raise ValueError(
            "max_output_candidates_per_facet must be >= 1"
        )

    requirement_rows = []

    for requirement in requirements.requirements:
        facets = decompose_target_facets(
            requirement.requested_target_observable
        )
        facet_rows = []

        for facet in facets:
            raw_candidates = _measurement_candidates(
                graph,
                observable=facet.facet_text,
                role="target",
                minimum_coverage=minimum_retrieval_coverage,
                max_candidates=max_retrieval_candidates_per_facet,
            )

            candidates = [
                _facet_candidate(
                    graph=graph,
                    facet=facet,
                    candidate=candidate,
                )
                for candidate in raw_candidates
            ]

            candidates.sort(
                key=lambda row: (
                    not row.exact_metric_identity,
                    -row.metric_identity_coverage,
                    -row.retrieval_lexical_coverage,
                    row.metric_id,
                    row.measurement_node_id,
                )
            )

            exact_count = sum(
                row.exact_metric_identity
                for row in candidates
            )
            partial_count = sum(
                row.partial_metric_identity
                for row in candidates
            )
            best_identity = max(
                (
                    row.metric_identity_coverage
                    for row in candidates
                ),
                default=0.0,
            )

            if exact_count:
                status: FacetSupportStatus = (
                    "EXACT_METRIC_IDENTITY_SUPPORT"
                )
            elif partial_count:
                status = "PARTIAL_METRIC_IDENTITY_SUPPORT"
            else:
                status = "NO_METRIC_IDENTITY_SUPPORT"

            facet_rows.append(
                TargetFacetResolution(
                    requirement_id=requirement.requirement_id,
                    output_experiment_id=(
                        requirement.output_experiment_id
                    ),
                    original_target=(
                        requirement.requested_target_observable
                    ),
                    facet=facet,
                    status=status,
                    retrieval_candidate_count=len(candidates),
                    exact_metric_identity_candidate_count=exact_count,
                    partial_metric_identity_candidate_count=partial_count,
                    best_metric_identity_coverage=best_identity,
                    candidates=candidates[
                        :max_output_candidates_per_facet
                    ],
                )
            )

        exact_supported = sum(
            row.status == "EXACT_METRIC_IDENTITY_SUPPORT"
            for row in facet_rows
        )
        partial_supported = sum(
            row.status == "PARTIAL_METRIC_IDENTITY_SUPPORT"
            for row in facet_rows
        )
        unsupported = sum(
            row.status == "NO_METRIC_IDENTITY_SUPPORT"
            for row in facet_rows
        )

        requirement_rows.append(
            TargetFacetRequirementResolution(
                requirement_id=requirement.requirement_id,
                output_experiment_id=(
                    requirement.output_experiment_id
                ),
                original_target=(
                    requirement.requested_target_observable
                ),
                decomposition_mode=(
                    facet_rows[0].facet.decomposition_mode
                ),
                facet_count=len(facet_rows),
                exact_supported_facet_count=exact_supported,
                partial_supported_facet_count=partial_supported,
                unsupported_facet_count=unsupported,
                all_facets_exactly_supported=bool(
                    exact_supported == len(facet_rows)
                ),
                facets=facet_rows,
            )
        )

    statuses = Counter(
        facet.status
        for requirement in requirement_rows
        for facet in requirement.facets
    )

    return TargetFacetResolutionSet(
        requirement_count=requirements.requirement_count,
        resolution_count=len(requirement_rows),
        total_facet_count=sum(
            row.facet_count
            for row in requirement_rows
        ),
        exact_supported_facet_count=sum(
            row.exact_supported_facet_count
            for row in requirement_rows
        ),
        partial_supported_facet_count=sum(
            row.partial_supported_facet_count
            for row in requirement_rows
        ),
        unsupported_facet_count=sum(
            row.unsupported_facet_count
            for row in requirement_rows
        ),
        status_counts=dict(sorted(statuses.items())),
        resolutions=requirement_rows,
    )
