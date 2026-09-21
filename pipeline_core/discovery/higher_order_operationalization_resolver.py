from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirement,
    OperationalizationWitnessRequirementSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ResolutionStatus = Literal[
    "SUPPORTED_DISTINCT_OPERATIONALIZATION",
    "SAME_PROVIDER_OR_METHOD_UNRESOLVED",
    "PROVIDER_IDENTITY_UNRESOLVED",
    "ONLY_TARGET_FOUND",
    "ONLY_ANCHOR_FOUND",
    "NO_MATCH",
]


class MeasurementProviderCandidate(StrictModel):
    provider_id: str
    provider_type: Literal["Experiment", "Calculation"]
    label: str
    method_identity: str
    method_identity_explicit: bool
    source_paper_id: str | None = None


class MeasurementOperationalizationCandidate(StrictModel):
    candidate_id: str
    role: Literal["target", "anchor"]
    observable: str

    measurement_node_id: str
    metric_id: str
    metric: str
    source_expression: str
    description: str
    subject_id: str

    lexical_coverage: float
    matched_token_count: int
    observable_token_count: int

    providers: list[MeasurementProviderCandidate] = Field(
        default_factory=list
    )
    provider_count: int = Field(ge=0)
    provenance_present: bool
    source_paper_ids: list[str] = Field(default_factory=list)

    measurement_independence_verified: Literal[False] = False
    evidence_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class OperationalizationPairCandidate(StrictModel):
    target_candidate_id: str
    anchor_candidate_id: str
    target_measurement_node_id: str
    anchor_measurement_node_id: str

    target_provider_ids: list[str] = Field(default_factory=list)
    anchor_provider_ids: list[str] = Field(default_factory=list)
    distinct_provider_ids_present: bool
    explicit_method_identity_present_on_both_sides: bool
    distinct_method_identity_present: bool
    provenance_present_on_both_sides: bool

    classification: Literal[
        "SUPPORTED_DISTINCT_OPERATIONALIZATION",
        "SAME_PROVIDER_OR_METHOD_UNRESOLVED",
        "PROVIDER_IDENTITY_UNRESOLVED",
    ]

    measurement_independence_verified: Literal[False] = False
    independence_certification_authority: Literal[False] = False


class OperationalizationWitnessResolution(StrictModel):
    requirement_id: str
    output_experiment_id: str
    requested_target_observable: str
    anchor_observable: str

    status: ResolutionStatus
    target_candidates: list[MeasurementOperationalizationCandidate] = Field(
        default_factory=list
    )
    anchor_candidates: list[MeasurementOperationalizationCandidate] = Field(
        default_factory=list
    )
    pair_candidates: list[OperationalizationPairCandidate] = Field(
        default_factory=list
    )

    target_candidate_count: int = Field(ge=0)
    anchor_candidate_count: int = Field(ge=0)
    pair_candidate_count: int = Field(ge=0)

    candidate_inspiration_involved: bool = False

    corpus_lookup_performed: Literal[True] = True
    measurement_independence_verified: Literal[False] = False
    independence_certification_authority: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class OperationalizationWitnessResolutionSet(StrictModel):
    schema_version: Literal[
        "operationalization-witness-resolution-set-v1"
    ] = "operationalization-witness-resolution-set-v1"

    requirement_count: int = Field(ge=0)
    resolution_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)

    measurement_candidate_count: int = Field(ge=0)
    pair_candidate_count: int = Field(ge=0)
    distinct_operationalization_resolution_count: int = Field(ge=0)

    graph_node_count: int = Field(ge=0)
    graph_edge_count: int = Field(ge=0)

    resolutions: list[OperationalizationWitnessResolution] = Field(
        default_factory=list
    )

    corpus_lookup_performed: Literal[True] = True
    lexical_candidate_retrieval_only: Literal[True] = True
    measurement_independence_verified_count: Literal[0] = 0
    independence_certification_authority: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    experiment_selection_performed: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


_DASHES = "‐-‒–—−"
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by",
    "can", "do", "does", "for", "from", "has", "have", "how", "in",
    "into", "is", "may", "of", "on", "or", "such", "than", "that",
    "the", "their", "to", "under", "with",
}


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _normalize(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    for dash in _DASHES:
        value = value.replace(dash, "-")
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: object) -> set[str]:
    raw = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        _normalize(text),
    )
    out: set[str] = set()
    for token in raw:
        parts = [part for part in token.split("-") if part]
        candidates = list(parts)
        if len(parts) > 1:
            candidates.append("".join(parts))
        for candidate in candidates:
            if len(candidate) > 1 and candidate not in _STOPWORDS:
                out.add(candidate)
    return out


def _coverage(query: str, text: str) -> tuple[float, int, int]:
    required = _tokens(query)
    if not required:
        return 0.0, 0, 0
    present = required & _tokens(text)
    return (
        float(len(present) / len(required)),
        len(present),
        len(required),
    )


def _measurement_text(attrs: dict[str, object]) -> str:
    return "\n".join(
        str(attrs.get(field, "") or "")
        for field in (
            "metric",
            "label",
            "metric_id",
            "source_expression",
            "description",
        )
    )


def _provider_method_identity(
    attrs: dict[str, object],
) -> tuple[str, bool]:
    node_type = str(attrs.get("type", "")).strip()

    if node_type == "Experiment":
        parts = [
            str(attrs.get("method_label", "") or "").strip(),
            str(attrs.get("raw_method_name", "") or "").strip(),
            str(attrs.get("experiment_family", "") or "").strip(),
        ]
    elif node_type == "Calculation":
        parts = [
            str(attrs.get("method_details", "") or "").strip(),
            str(attrs.get("calculation_type", "") or "").strip(),
        ]
    else:
        return "", False

    explicit = [part for part in parts if part]
    return " | ".join(explicit), bool(explicit)


def _edge_relation(attrs: dict[str, object]) -> str:
    return str(
        attrs.get("relation")
        or attrs.get("title")
        or ""
    ).strip().upper()


def _iter_in_edges(
    graph: nx.Graph,
    node_id: str,
):
    if graph.is_multigraph():
        yield from graph.in_edges(
            node_id,
            keys=True,
            data=True,
        )
    else:
        for source, target, attrs in graph.in_edges(
            node_id,
            data=True,
        ):
            yield source, target, None, attrs


def _providers_for_measurement(
    graph: nx.Graph,
    measurement_id: str,
) -> list[MeasurementProviderCandidate]:
    rows = {}

    for source, _, _, edge_attrs in _iter_in_edges(
        graph,
        measurement_id,
    ):
        if _edge_relation(dict(edge_attrs)) != "HAS_MEASUREMENT":
            continue

        attrs = dict(graph.nodes[source])
        node_type = str(attrs.get("type", "")).strip()
        if node_type not in {"Experiment", "Calculation"}:
            continue

        method_identity, explicit = _provider_method_identity(attrs)

        paper_id = str(
            attrs.get("source_paper_id")
            or edge_attrs.get("source_paper_id")
            or edge_attrs.get("paper_id")
            or ""
        ).strip()

        provider = MeasurementProviderCandidate(
            provider_id=str(source),
            provider_type=node_type,
            label=str(attrs.get("label", "") or ""),
            method_identity=method_identity,
            method_identity_explicit=explicit,
            source_paper_id=paper_id or None,
        )
        rows[provider.provider_id] = provider

    return [
        rows[key]
        for key in sorted(rows)
    ]


def _measurement_provenance(
    graph: nx.Graph,
    measurement_id: str,
    providers: list[MeasurementProviderCandidate],
) -> tuple[bool, list[str]]:
    papers = {
        value
        for value in (
            str(
                graph.nodes[measurement_id].get(
                    "source_paper_id",
                    "",
                )
                or ""
            ).strip(),
            *[
                provider.source_paper_id or ""
                for provider in providers
            ],
        )
        if value
    }

    edge_provenance = False

    for _, _, _, attrs in _iter_in_edges(graph, measurement_id):
        attrs = dict(attrs)
        if _edge_relation(attrs) != "HAS_MEASUREMENT":
            continue

        if any(
            str(attrs.get(field, "") or "").strip()
            for field in (
                "paper_id",
                "source_paper_id",
                "document_id",
                "evidence_pointers_json",
                "evidence_text",
            )
        ):
            edge_provenance = True

        paper = str(
            attrs.get("source_paper_id")
            or attrs.get("paper_id")
            or ""
        ).strip()
        if paper:
            papers.add(paper)

    return bool(papers or edge_provenance), sorted(papers)


def _measurement_candidates(
    graph: nx.Graph,
    *,
    observable: str,
    role: Literal["target", "anchor"],
    minimum_coverage: float,
    max_candidates: int,
) -> list[MeasurementOperationalizationCandidate]:
    rows = []

    for node_id, raw_attrs in graph.nodes(data=True):
        attrs = dict(raw_attrs)
        if str(attrs.get("type", "")).strip() != "Measurement":
            continue

        text = _measurement_text(attrs)
        score, matched, total = _coverage(observable, text)

        if score < minimum_coverage:
            continue

        providers = _providers_for_measurement(
            graph,
            str(node_id),
        )
        provenance_present, paper_ids = _measurement_provenance(
            graph,
            str(node_id),
            providers,
        )

        rows.append(
            MeasurementOperationalizationCandidate(
                candidate_id=_stable_id(
                    "measurement_operationalization_candidate",
                    role,
                    observable,
                    node_id,
                ),
                role=role,
                observable=observable,
                measurement_node_id=str(node_id),
                metric_id=str(attrs.get("metric_id", "") or ""),
                metric=str(
                    attrs.get("metric")
                    or attrs.get("label")
                    or ""
                ),
                source_expression=str(
                    attrs.get("source_expression", "") or ""
                ),
                description=str(attrs.get("description", "") or ""),
                subject_id=str(attrs.get("subject_id", "") or ""),
                lexical_coverage=score,
                matched_token_count=matched,
                observable_token_count=total,
                providers=providers,
                provider_count=len(providers),
                provenance_present=provenance_present,
                source_paper_ids=paper_ids,
            )
        )

    rows.sort(
        key=lambda row: (
            -row.lexical_coverage,
            -row.matched_token_count,
            -row.provider_count,
            row.measurement_node_id,
        )
    )

    return rows[:max_candidates]


def _pair_classification(
    target: MeasurementOperationalizationCandidate,
    anchor: MeasurementOperationalizationCandidate,
) -> OperationalizationPairCandidate:
    target_provider_ids = {
        row.provider_id
        for row in target.providers
    }
    anchor_provider_ids = {
        row.provider_id
        for row in anchor.providers
    }

    distinct_provider_ids_present = bool(
        target_provider_ids
        and anchor_provider_ids
        and target_provider_ids.isdisjoint(anchor_provider_ids)
    )

    target_methods = {
        _normalize(row.method_identity)
        for row in target.providers
        if row.method_identity_explicit
        and _normalize(row.method_identity)
    }
    anchor_methods = {
        _normalize(row.method_identity)
        for row in anchor.providers
        if row.method_identity_explicit
        and _normalize(row.method_identity)
    }

    explicit_method_identity_present_on_both_sides = bool(
        target_methods
        and anchor_methods
    )
    distinct_method_identity_present = bool(
        explicit_method_identity_present_on_both_sides
        and target_methods.isdisjoint(anchor_methods)
    )
    provenance_present_on_both_sides = bool(
        target.provenance_present
        and anchor.provenance_present
    )

    if (
        distinct_provider_ids_present
        and explicit_method_identity_present_on_both_sides
        and distinct_method_identity_present
        and provenance_present_on_both_sides
    ):
        classification = "SUPPORTED_DISTINCT_OPERATIONALIZATION"
    elif (
        not target_provider_ids
        or not anchor_provider_ids
        or not explicit_method_identity_present_on_both_sides
        or not provenance_present_on_both_sides
    ):
        classification = "PROVIDER_IDENTITY_UNRESOLVED"
    else:
        classification = "SAME_PROVIDER_OR_METHOD_UNRESOLVED"

    return OperationalizationPairCandidate(
        target_candidate_id=target.candidate_id,
        anchor_candidate_id=anchor.candidate_id,
        target_measurement_node_id=target.measurement_node_id,
        anchor_measurement_node_id=anchor.measurement_node_id,
        target_provider_ids=sorted(target_provider_ids),
        anchor_provider_ids=sorted(anchor_provider_ids),
        distinct_provider_ids_present=distinct_provider_ids_present,
        explicit_method_identity_present_on_both_sides=(
            explicit_method_identity_present_on_both_sides
        ),
        distinct_method_identity_present=distinct_method_identity_present,
        provenance_present_on_both_sides=provenance_present_on_both_sides,
        classification=classification,
    )


def _resolve_requirement(
    graph: nx.Graph,
    requirement: OperationalizationWitnessRequirement,
    *,
    minimum_coverage: float,
    max_candidates_per_observable: int,
    max_pairs: int,
) -> OperationalizationWitnessResolution:
    target = _measurement_candidates(
        graph,
        observable=requirement.requested_target_observable,
        role="target",
        minimum_coverage=minimum_coverage,
        max_candidates=max_candidates_per_observable,
    )
    anchor = _measurement_candidates(
        graph,
        observable=requirement.anchor_observable,
        role="anchor",
        minimum_coverage=minimum_coverage,
        max_candidates=max_candidates_per_observable,
    )

    pairs = [
        _pair_classification(left, right)
        for left in target
        for right in anchor
        if left.measurement_node_id != right.measurement_node_id
    ]

    rank = {
        "SUPPORTED_DISTINCT_OPERATIONALIZATION": 0,
        "SAME_PROVIDER_OR_METHOD_UNRESOLVED": 1,
        "PROVIDER_IDENTITY_UNRESOLVED": 2,
    }
    pairs.sort(
        key=lambda row: (
            rank[row.classification],
            row.target_measurement_node_id,
            row.anchor_measurement_node_id,
        )
    )
    pairs = pairs[:max_pairs]

    if not target and not anchor:
        status: ResolutionStatus = "NO_MATCH"
    elif target and not anchor:
        status = "ONLY_TARGET_FOUND"
    elif anchor and not target:
        status = "ONLY_ANCHOR_FOUND"
    elif any(
        row.classification == "SUPPORTED_DISTINCT_OPERATIONALIZATION"
        for row in pairs
    ):
        status = "SUPPORTED_DISTINCT_OPERATIONALIZATION"
    elif any(
        row.classification == "SAME_PROVIDER_OR_METHOD_UNRESOLVED"
        for row in pairs
    ):
        status = "SAME_PROVIDER_OR_METHOD_UNRESOLVED"
    else:
        status = "PROVIDER_IDENTITY_UNRESOLVED"

    return OperationalizationWitnessResolution(
        requirement_id=requirement.requirement_id,
        output_experiment_id=requirement.output_experiment_id,
        requested_target_observable=(
            requirement.requested_target_observable
        ),
        anchor_observable=requirement.anchor_observable,
        status=status,
        target_candidates=target,
        anchor_candidates=anchor,
        pair_candidates=pairs,
        target_candidate_count=len(target),
        anchor_candidate_count=len(anchor),
        pair_candidate_count=len(pairs),
        candidate_inspiration_involved=(
            requirement.candidate_inspiration_involved
        ),
    )


def resolve_operationalization_witnesses(
    *,
    graph: nx.Graph,
    requirements: OperationalizationWitnessRequirementSet,
    minimum_coverage: float = 0.50,
    max_candidates_per_observable: int = 20,
    max_pairs: int = 40,
) -> OperationalizationWitnessResolutionSet:
    if minimum_coverage <= 0.0 or minimum_coverage > 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if max_candidates_per_observable < 1:
        raise ValueError("max_candidates_per_observable must be >= 1")
    if max_pairs < 1:
        raise ValueError("max_pairs must be >= 1")

    rows = [
        _resolve_requirement(
            graph,
            requirement,
            minimum_coverage=minimum_coverage,
            max_candidates_per_observable=max_candidates_per_observable,
            max_pairs=max_pairs,
        )
        for requirement in requirements.requirements
    ]

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    candidate_ids = {
        candidate.candidate_id
        for row in rows
        for candidate in (
            *row.target_candidates,
            *row.anchor_candidates,
        )
    }

    return OperationalizationWitnessResolutionSet(
        requirement_count=requirements.requirement_count,
        resolution_count=len(rows),
        status_counts=dict(sorted(status_counts.items())),
        measurement_candidate_count=len(candidate_ids),
        pair_candidate_count=sum(
            row.pair_candidate_count
            for row in rows
        ),
        distinct_operationalization_resolution_count=sum(
            row.status == "SUPPORTED_DISTINCT_OPERATIONALIZATION"
            for row in rows
        ),
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
        resolutions=rows,
    )
