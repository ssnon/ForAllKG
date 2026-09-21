from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_operationalization_target_facets import (
    OperationalizationTargetFacet,
    TargetFacetResolutionSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


BridgeCandidateStatus = Literal[
    "CORPUS_REPEATED_BRIDGE_CANDIDATE",
    "SINGLE_PAPER_BRIDGE_CANDIDATE",
    "NO_BRIDGE_CANDIDATE",
]


class OperationalizationBridgeSupport(StrictModel):
    measurement_node_id: str
    metric_id: str
    metric_label: str
    source_expression: str
    source_paper_ids: list[str] = Field(default_factory=list)
    subject_id: str
    provider_types: list[str] = Field(default_factory=list)

    metric_identity_matched_tokens: list[str] = Field(default_factory=list)
    source_expression_matched_tokens: list[str] = Field(default_factory=list)

    metric_supports_qualifier: bool
    source_expression_supports_head_and_facet: bool

    support_is_bridge_certification: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class OperationalizationBridgeCandidate(StrictModel):
    schema_version: Literal[
        "operationalization-bridge-candidate-v1"
    ] = "operationalization-bridge-candidate-v1"

    bridge_candidate_id: str
    requirement_id: str
    output_experiment_id: str

    original_target: str
    facet_id: str
    facet_text: str
    qualifier_text: str
    head_text: str
    facet_term: str

    metric_id: str
    metric_labels: list[str] = Field(default_factory=list)

    support_measurement_count: int = Field(ge=1)
    support_paper_count: int = Field(ge=1)
    support_subject_count: int = Field(ge=0)
    support_provider_types: list[str] = Field(default_factory=list)

    status: BridgeCandidateStatus
    supports: list[OperationalizationBridgeSupport] = Field(
        default_factory=list
    )

    bridge_kind: Literal[
        "corpus_observed_operationalization_candidate"
    ] = "corpus_observed_operationalization_candidate"

    scientific_equivalence_asserted: Literal[False] = False
    operationalization_bridge_verified: Literal[False] = False
    operationalization_bridge_authority: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    selection_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class TargetFacetBridgeResolution(StrictModel):
    requirement_id: str
    output_experiment_id: str
    facet_id: str
    facet_text: str
    status: BridgeCandidateStatus

    candidate_count: int = Field(ge=0)
    repeated_candidate_count: int = Field(ge=0)
    single_paper_candidate_count: int = Field(ge=0)

    candidates: list[OperationalizationBridgeCandidate] = Field(
        default_factory=list
    )

    operationalization_bridge_verified: Literal[False] = False
    operationalization_bridge_authority: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    selection_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class TargetFacetBridgeResolutionSet(StrictModel):
    schema_version: Literal[
        "target-facet-operationalization-bridge-resolution-v1"
    ] = "target-facet-operationalization-bridge-resolution-v1"

    requirement_count: int = Field(ge=0)
    facet_count: int = Field(ge=0)
    resolution_count: int = Field(ge=0)

    candidate_count: int = Field(ge=0)
    repeated_candidate_count: int = Field(ge=0)
    single_paper_candidate_count: int = Field(ge=0)
    unresolved_facet_count: int = Field(ge=0)

    status_counts: dict[str, int] = Field(default_factory=dict)
    resolutions: list[TargetFacetBridgeResolution] = Field(
        default_factory=list
    )

    metric_identity_uses_source_expression: Literal[False] = False
    source_expression_used_as_bridge_evidence_only: Literal[True] = True
    scientific_equivalence_asserted: Literal[False] = False
    operationalization_bridge_verified_count: Literal[0] = 0
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


def _tokens(value: object) -> set[str]:
    """
    Deterministic orthographic token forms only.

    In addition to ordinary tokens and hyphen-compacted forms, expose
    compacted adjacent surface-token forms. This treats spelling variants
    such as ``hotspot``, ``hot-spot``, and ``hot spot`` as the same lexical
    form without introducing scientific synonymy.

    A trailing plural ``s`` is stripped only from the second member of an
    adjacent compacted form, so ``hot spots`` can expose ``hotspot`` while
    standalone domain tokens such as ``SERS`` are left untouched.
    """
    raw = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", _normalize(value))
    out = set()
    surface_parts: list[str] = []

    for token in raw:
        parts = [part for part in token.split("-") if part]
        surface_parts.extend(parts)

        candidates = list(parts)
        if len(parts) > 1:
            joined = "".join(parts)
            candidates.append(joined)

            if (
                len(parts[-1]) > 3
                and parts[-1].endswith("s")
                and not parts[-1].endswith("ss")
            ):
                candidates.append(
                    "".join([*parts[:-1], parts[-1][:-1]])
                )

        for candidate in candidates:
            if len(candidate) > 1 and candidate not in _STOPWORDS:
                out.add(candidate)

    for left, right in zip(surface_parts, surface_parts[1:]):
        joined = left + right
        if len(joined) > 1:
            out.add(joined)

        if (
            len(right) > 3
            and right.endswith("s")
            and not right.endswith("ss")
        ):
            singular_joined = left + right[:-1]
            if len(singular_joined) > 1:
                out.add(singular_joined)

    return out

def _required_terms(value: object) -> set[str]:
    """
    Semantic terms required by the requested facet/qualifier.

    Unlike ``_tokens``, this function does not create adjacent phrase
    compounds. Required semantic units must not become stricter merely
    because candidate-side orthographic normalization exposes additional
    searchable forms.
    """
    raw = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", _normalize(value))
    out = set()

    for token in raw:
        parts = [part for part in token.split("-") if part]
        candidates = list(parts)

        if len(parts) > 1:
            candidates.append("".join(parts))

        for candidate in candidates:
            if len(candidate) > 1 and candidate not in _STOPWORDS:
                out.add(candidate)

    return out


def _metric_identity_text(attrs: dict) -> str:
    return " ".join(
        [
            str(attrs.get("metric_id", "") or ""),
            str(attrs.get("metric", "") or ""),
            str(attrs.get("label", "") or ""),
        ]
    )


def _provider_types_for_measurement(
    graph: nx.Graph,
    measurement_id: str,
) -> list[str]:
    rows = set()
    if graph.is_multigraph():
        incoming = graph.in_edges(
            measurement_id,
            keys=True,
            data=True,
        )
        for source, _, _, attrs in incoming:
            relation = str(
                attrs.get("relation")
                or attrs.get("title")
                or ""
            ).strip().upper()
            if relation != "HAS_MEASUREMENT":
                continue
            if not graph.has_node(source):
                continue
            provider_type = str(
                graph.nodes[source].get("type", "") or ""
            ).strip()
            if provider_type in {"Experiment", "Calculation"}:
                rows.add(provider_type)
    else:
        for source, _, attrs in graph.in_edges(
            measurement_id,
            data=True,
        ):
            relation = str(
                attrs.get("relation")
                or attrs.get("title")
                or ""
            ).strip().upper()
            if relation != "HAS_MEASUREMENT":
                continue
            if not graph.has_node(source):
                continue
            provider_type = str(
                graph.nodes[source].get("type", "") or ""
            ).strip()
            if provider_type in {"Experiment", "Calculation"}:
                rows.add(provider_type)

    return sorted(rows)


def _qualifier_support_tokens(
    facet: OperationalizationTargetFacet,
) -> set[str]:
    qualifier = _required_terms(facet.qualifier_text)
    if qualifier:
        return qualifier
    # For non-qualified targets, require the head on the metric identity
    # side so a source-expression coincidence alone cannot create a bridge.
    return _required_terms(facet.head_text)


def _source_required_tokens(
    facet: OperationalizationTargetFacet,
) -> set[str]:
    return _required_terms(
        f"{facet.head_text} {facet.facet_term}"
    )


def _bridge_support_for_measurement(
    *,
    graph: nx.Graph,
    facet: OperationalizationTargetFacet,
    measurement_id: str,
) -> OperationalizationBridgeSupport | None:
    attrs = dict(graph.nodes[measurement_id])

    metric_id = str(attrs.get("metric_id", "") or "").strip()
    if not metric_id:
        return None

    metric_tokens = _tokens(_metric_identity_text(attrs))
    source_expression = str(
        attrs.get("source_expression", "") or ""
    ).strip()
    source_tokens = _tokens(source_expression)

    qualifier_required = _qualifier_support_tokens(facet)
    source_required = _source_required_tokens(facet)

    metric_supports = bool(
        qualifier_required
        and qualifier_required.issubset(metric_tokens)
    )
    source_supports = bool(
        source_required
        and source_required.issubset(source_tokens)
    )

    if not (metric_supports and source_supports):
        return None

    paper_ids = []
    for value in (
        attrs.get("source_paper_id"),
        attrs.get("paper_id"),
    ):
        text = str(value or "").strip()
        if text and text not in paper_ids:
            paper_ids.append(text)

    subject_id = str(attrs.get("subject_id", "") or "").strip()

    return OperationalizationBridgeSupport(
        measurement_node_id=str(measurement_id),
        metric_id=metric_id,
        metric_label=str(
            attrs.get("metric")
            or attrs.get("label")
            or ""
        ),
        source_expression=source_expression,
        source_paper_ids=paper_ids,
        subject_id=subject_id,
        provider_types=_provider_types_for_measurement(
            graph,
            str(measurement_id),
        ),
        metric_identity_matched_tokens=sorted(
            qualifier_required & metric_tokens
        ),
        source_expression_matched_tokens=sorted(
            source_required & source_tokens
        ),
        metric_supports_qualifier=metric_supports,
        source_expression_supports_head_and_facet=source_supports,
    )


def resolve_target_facet_bridge_candidates(
    *,
    graph: nx.Graph,
    facet_resolutions: TargetFacetResolutionSet,
    minimum_distinct_papers_for_repeated_bridge: int = 2,
    max_supports_per_candidate: int = 50,
) -> TargetFacetBridgeResolutionSet:
    if minimum_distinct_papers_for_repeated_bridge < 2:
        raise ValueError(
            "minimum_distinct_papers_for_repeated_bridge must be >= 2"
        )
    if max_supports_per_candidate < 1:
        raise ValueError(
            "max_supports_per_candidate must be >= 1"
        )

    measurement_ids = [
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("type", "")) == "Measurement"
    ]

    rows = []

    for requirement in facet_resolutions.resolutions:
        for facet_resolution in requirement.facets:
            facet = facet_resolution.facet
            grouped: dict[
                str,
                list[OperationalizationBridgeSupport],
            ] = defaultdict(list)

            for measurement_id in measurement_ids:
                support = _bridge_support_for_measurement(
                    graph=graph,
                    facet=facet,
                    measurement_id=measurement_id,
                )
                if support is not None:
                    grouped[support.metric_id].append(support)

            candidates = []

            for metric_id in sorted(grouped):
                supports = grouped[metric_id]

                paper_ids = sorted(
                    {
                        paper_id
                        for support in supports
                        for paper_id in support.source_paper_ids
                        if paper_id
                    }
                )
                subjects = sorted(
                    {
                        support.subject_id
                        for support in supports
                        if support.subject_id
                    }
                )
                provider_types = sorted(
                    {
                        provider_type
                        for support in supports
                        for provider_type in support.provider_types
                    }
                )
                metric_labels = sorted(
                    {
                        support.metric_label
                        for support in supports
                        if support.metric_label
                    }
                )

                if (
                    len(paper_ids)
                    >= minimum_distinct_papers_for_repeated_bridge
                ):
                    status: BridgeCandidateStatus = (
                        "CORPUS_REPEATED_BRIDGE_CANDIDATE"
                    )
                else:
                    status = "SINGLE_PAPER_BRIDGE_CANDIDATE"

                candidates.append(
                    OperationalizationBridgeCandidate(
                        bridge_candidate_id=_stable_id(
                            "operationalization_bridge_candidate",
                            requirement.requirement_id,
                            facet.facet_id,
                            metric_id,
                        ),
                        requirement_id=requirement.requirement_id,
                        output_experiment_id=(
                            requirement.output_experiment_id
                        ),
                        original_target=requirement.original_target,
                        facet_id=facet.facet_id,
                        facet_text=facet.facet_text,
                        qualifier_text=facet.qualifier_text,
                        head_text=facet.head_text,
                        facet_term=facet.facet_term,
                        metric_id=metric_id,
                        metric_labels=metric_labels,
                        support_measurement_count=len(supports),
                        support_paper_count=len(paper_ids),
                        support_subject_count=len(subjects),
                        support_provider_types=provider_types,
                        status=status,
                        supports=supports[:max_supports_per_candidate],
                    )
                )

            candidates.sort(
                key=lambda row: (
                    row.status != "CORPUS_REPEATED_BRIDGE_CANDIDATE",
                    -row.support_paper_count,
                    -row.support_measurement_count,
                    row.metric_id,
                )
            )

            if any(
                row.status == "CORPUS_REPEATED_BRIDGE_CANDIDATE"
                for row in candidates
            ):
                status: BridgeCandidateStatus = (
                    "CORPUS_REPEATED_BRIDGE_CANDIDATE"
                )
            elif candidates:
                status = "SINGLE_PAPER_BRIDGE_CANDIDATE"
            else:
                status = "NO_BRIDGE_CANDIDATE"

            rows.append(
                TargetFacetBridgeResolution(
                    requirement_id=requirement.requirement_id,
                    output_experiment_id=(
                        requirement.output_experiment_id
                    ),
                    facet_id=facet.facet_id,
                    facet_text=facet.facet_text,
                    status=status,
                    candidate_count=len(candidates),
                    repeated_candidate_count=sum(
                        row.status
                        == "CORPUS_REPEATED_BRIDGE_CANDIDATE"
                        for row in candidates
                    ),
                    single_paper_candidate_count=sum(
                        row.status
                        == "SINGLE_PAPER_BRIDGE_CANDIDATE"
                        for row in candidates
                    ),
                    candidates=candidates,
                )
            )

    status_counts = Counter(row.status for row in rows)

    return TargetFacetBridgeResolutionSet(
        requirement_count=facet_resolutions.requirement_count,
        facet_count=facet_resolutions.total_facet_count,
        resolution_count=len(rows),
        candidate_count=sum(row.candidate_count for row in rows),
        repeated_candidate_count=sum(
            row.repeated_candidate_count for row in rows
        ),
        single_paper_candidate_count=sum(
            row.single_paper_candidate_count for row in rows
        ),
        unresolved_facet_count=sum(
            row.status == "NO_BRIDGE_CANDIDATE"
            for row in rows
        ),
        status_counts=dict(sorted(status_counts.items())),
        resolutions=rows,
    )
