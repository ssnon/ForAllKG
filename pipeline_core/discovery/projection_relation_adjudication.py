from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.relation_ir_registry import (
    get_relation_typing_adapter,
)
from pipeline_core.discovery.counterevidence_projection_retrieval import (
    CounterevidenceProjectionQueryPlan,
)
from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
    GroundedFactorRelationProjection,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIR,
    ScientificRelationIRReport,
    assess_relation_document_compatibility,
    lexical_content_tokens,
)
from pipeline_core.discovery.supporting_projection_retrieval import (
    SupportingProjectionQueryPlan,
)
from pipeline_core.discovery.scientific_relation_second_pass import (
    ScientificRelationSecondPassPlan,
)
from pipeline_core.domain.domain_profile import (
    ScientificDomainProfile,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ProjectionPriorArtRelationship = Literal[
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "LOWER_ORDER_RELATION_PRIOR_ART",
    "DIRECTIONAL_COUNTEREVIDENCE",
    "CONTEXTUAL_CONFLICT",
    "CONFLICTING_PRIOR_ART",
    "COMPONENT_ONLY",
    "TITLE_ONLY_NEIGHBOR",
    "UNRELATED",
    "INSUFFICIENT_METADATA",
]

CompiledClaimRelationState = Literal[
    "DIRECT_RELATION_FOUND",
    "LOWER_ORDER_RELATION_FOUND",
    "COUNTEREVIDENCE_FOUND",
    "CONFLICTING_RELATION_FOUND",
    "MIXED_RELATION_SIGNALS",
    "COMPONENTS_OR_CONTEXT_ONLY",
    "NO_MATERIAL_RELATION_SIGNAL",
    "INSUFFICIENT_METADATA",
]


_STRONG_ABSTRACT_RELATIONSHIPS = frozenset(
    {
        "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART",
        "LOWER_ORDER_RELATION_PRIOR_ART",
        "DIRECTIONAL_COUNTEREVIDENCE",
        "CONTEXTUAL_CONFLICT",
        "CONFLICTING_PRIOR_ART",
    }
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _normalize(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _exact_span_present(span: str, source: str) -> bool:
    needle = _normalize(span)
    haystack = _normalize(source)
    return bool(needle and needle in haystack)


def _work_document(work: PriorArtWork) -> str:
    return "\n".join(
        value
        for value in (
            str(work.title or "").strip(),
            str(work.abstract or "").strip(),
        )
        if value
    )


def _work_identity_key(work: PriorArtWork) -> str:
    doi = str(work.doi or "").strip().casefold()
    doi = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
    )
    doi = re.sub(r"^doi:\s*", "", doi)
    if doi:
        return "doi:" + doi

    title = _normalize(work.title)
    if title:
        return "title:" + title + "|" + str(work.year or "")

    return "work:" + work.work_id


class ProjectionRelationWorkCandidate(StrictModel):
    review_work_id: str
    source_work_ids: list[str] = Field(min_length=1)

    title: str
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    citation_count: int | None = None

    typed_compatibility_state: str
    relation_type_labels: list[str] = Field(default_factory=list)
    document_type_labels: list[str] = Field(default_factory=list)
    document_ambiguity_labels: list[str] = Field(default_factory=list)
    document_domain_labels: list[str] = Field(default_factory=list)
    document_scope_features: list[str] = Field(default_factory=list)
    typed_reason_codes: list[str] = Field(default_factory=list)

    endpoint_lexical_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    endpoint_discriminative_coverages: list[float] = Field(
        default_factory=list
    )
    endpoint_discriminative_match_counts: list[int] = Field(
        default_factory=list
    )
    endpoint_required_match_counts: list[int] = Field(
        default_factory=list
    )
    endpoint_supported_count: int = Field(default=0, ge=0)
    all_endpoints_supported: bool = False
    relation_anchor_tier: Literal[
        "PAIR_OR_MULTI_ENDPOINT_ANCHORED",
        "PARTIAL_ENDPOINT_ANCHORED",
        "CONTEXT_ONLY",
    ] = "CONTEXT_ONLY"
    identity_lexical_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    supporting_projection_ids: list[str] = Field(default_factory=list)
    supporting_projection_kinds: list[str] = Field(default_factory=list)
    supporting_factor_orders: list[int] = Field(default_factory=list)

    counterevidence_projection_ids: list[str] = Field(default_factory=list)
    counterevidence_modes: list[str] = Field(default_factory=list)

    selected_lanes: list[
        Literal[
            "SUPPORTING_PRIOR_ART",
            "COUNTEREVIDENCE_PRIOR_ART",
            "SEMANTIC_SECOND_PASS_SUPPORTING",
            "SEMANTIC_SECOND_PASS_COUNTEREVIDENCE",
        ]
    ] = Field(default_factory=list)

    second_pass_projection_ids: list[str] = Field(default_factory=list)
    second_pass_roles: list[str] = Field(default_factory=list)

    selection_score: float
    selected_for_review: Literal[True] = True

    retrieval_is_not_relation_evidence: Literal[True] = True
    novelty_authority: Literal[False] = False


class ProjectionRelationClaimCandidateSet(StrictModel):
    hypothesis_id: str
    claim_id: str
    relation_ir_id: str
    claim_text: str

    endpoint_terms: list[str] = Field(min_length=2)
    identity_terms: list[str] = Field(default_factory=list)
    scope_terms: list[str] = Field(default_factory=list)
    directional_terms: list[str] = Field(default_factory=list)

    projection_ids: list[str] = Field(default_factory=list)
    full_projection_ids: list[str] = Field(default_factory=list)
    lower_order_projection_ids: list[str] = Field(default_factory=list)

    candidates: list[ProjectionRelationWorkCandidate]
    candidate_count: int = Field(ge=0)
    abstract_candidate_count: int = Field(ge=0)
    typed_identity_excluded_work_count: int = Field(ge=0)
    source_unique_work_count: int = Field(ge=0)

    max_review_works: int = Field(ge=1)
    balanced_lane_selection: Literal[True] = True

    @model_validator(mode="after")
    def validate_counts(self) -> "ProjectionRelationClaimCandidateSet":
        if not self.projection_ids and self.candidates:
            raise ValueError(
                "unprojected relation cannot carry bounded review candidates"
            )

        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count mismatch")
        if self.abstract_candidate_count != sum(
            bool(row.abstract)
            for row in self.candidates
        ):
            raise ValueError("abstract_candidate_count mismatch")
        if self.candidate_count > self.max_review_works:
            raise ValueError("candidate set exceeded max_review_works")
        return self


class ProjectionRelationCandidateReport(StrictModel):
    schema_version: Literal[
        "projection-relation-candidate-report-v1"
    ] = "projection-relation-candidate-report-v1"

    report_id: str
    source_relation_ir_report_id: str
    source_projection_report_id: str
    source_supporting_query_plan_id: str
    source_supporting_prior_art_packet_id: str
    source_counterevidence_query_plan_id: str
    source_counterevidence_prior_art_packet_id: str

    claims: list[ProjectionRelationClaimCandidateSet]
    claim_count: int = Field(ge=0)
    total_review_candidate_count: int = Field(ge=0)

    diagnostic_only: Literal[True] = True
    relation_adjudication_performed: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ProjectionRelationCandidateReport":
        if self.claim_count != len(self.claims):
            raise ValueError("claim_count mismatch")
        if self.total_review_candidate_count != sum(
            row.candidate_count
            for row in self.claims
        ):
            raise ValueError("total_review_candidate_count mismatch")
        return self


class ProjectionRelationMatchDraft(StrictModel):
    work_id: str
    relationship: ProjectionPriorArtRelationship
    confidence: float = Field(ge=0.0, le=1.0)

    basis_projection_ids: list[str] = Field(default_factory=list)
    counterevidence_modes: list[str] = Field(default_factory=list)
    second_pass_roles: list[str] = Field(default_factory=list)

    evidence_span: str = ""
    rationale: str = Field(min_length=1)


class ProjectionRelationClaimReviewDraft(StrictModel):
    matches: list[ProjectionRelationMatchDraft] = Field(default_factory=list)
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_work_ids(
        self,
    ) -> "ProjectionRelationClaimReviewDraft":
        ids = [row.work_id for row in self.matches]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "projection relation draft must contain at most one match per work"
            )
        return self


class CompiledProjectionRelationMatch(StrictModel):
    work_id: str
    source_work_ids: list[str]
    relationship: ProjectionPriorArtRelationship
    original_relationship: ProjectionPriorArtRelationship
    confidence: float

    basis_projection_ids: list[str] = Field(default_factory=list)
    counterevidence_modes: list[str] = Field(default_factory=list)
    second_pass_roles: list[str] = Field(default_factory=list)

    evidence_span: str = ""
    rationale: str

    typed_compatibility_state: str
    abstract_available: bool
    deterministic_reason_codes: list[str] = Field(default_factory=list)


class ProjectionRelationClaimReview(StrictModel):
    hypothesis_id: str
    claim_id: str
    relation_ir_id: str
    relation_state: CompiledClaimRelationState

    matches: list[CompiledProjectionRelationMatch]

    # Audit coverage. All candidates are presented and exhaustive
    # classification is requested. If the model nevertheless omits a work,
    # it remains explicitly unclassified and therefore cannot contribute to
    # bounded review closure. Keep reviewed_work_count as the number of
    # emitted compiled match records.
    presented_work_count: int = Field(ge=0)
    classified_work_count: int = Field(ge=0)
    unclassified_work_ids: list[str] = Field(default_factory=list)
    reviewed_work_count: int = Field(ge=0)

    direct_prior_art_work_ids: list[str] = Field(default_factory=list)
    partial_prior_art_work_ids: list[str] = Field(default_factory=list)
    lower_order_prior_art_work_ids: list[str] = Field(default_factory=list)
    directional_counterevidence_work_ids: list[str] = Field(default_factory=list)
    contextual_conflict_work_ids: list[str] = Field(default_factory=list)
    conflicting_prior_art_work_ids: list[str] = Field(default_factory=list)
    component_only_work_ids: list[str] = Field(default_factory=list)

    reviewer_unknown_work_ids: list[str] = Field(default_factory=list)
    interpretation: str

    no_absence_inference: Literal[True] = True
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_count(self) -> "ProjectionRelationClaimReview":
        if self.classified_work_count != len(self.matches):
            raise ValueError("classified_work_count mismatch")
        if self.reviewed_work_count != self.classified_work_count:
            raise ValueError(
                "legacy reviewed_work_count must equal classified_work_count"
            )
        if (
            self.presented_work_count
            != self.classified_work_count
            + len(self.unclassified_work_ids)
        ):
            raise ValueError("presented adjudication coverage mismatch")
        return self


class ProjectionRelationAdjudicationReport(StrictModel):
    schema_version: Literal[
        "projection-relation-adjudication-report-v1"
    ] = "projection-relation-adjudication-report-v1"

    report_id: str
    source_candidate_report_id: str
    backend_name: str
    model_name: str

    reviews: list[ProjectionRelationClaimReview]
    reviewed_claim_count: int = Field(ge=0)

    presented_work_count: int = Field(ge=0)
    classified_work_count: int = Field(ge=0)
    unclassified_work_count: int = Field(ge=0)

    # Backward-compatible alias for emitted/classified match rows.
    reviewed_work_count: int = Field(ge=0)
    llm_calls_performed: int = Field(ge=0)

    direct_signal_work_count: int = Field(ge=0)
    lower_order_signal_work_count: int = Field(ge=0)
    counterevidence_signal_work_count: int = Field(ge=0)
    conflicting_signal_work_count: int = Field(ge=0)

    epistemic_usage: Literal[
        "bounded_relation_adjudication_not_novelty_certification"
    ] = "bounded_relation_adjudication_not_novelty_certification"

    diagnostic_only: Literal[True] = True
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ProjectionRelationAdjudicationReport":
        if self.reviewed_claim_count != len(self.reviews):
            raise ValueError("reviewed_claim_count mismatch")

        expected_presented = sum(
            row.presented_work_count
            for row in self.reviews
        )
        expected_classified = sum(
            row.classified_work_count
            for row in self.reviews
        )
        expected_unclassified = sum(
            len(row.unclassified_work_ids)
            for row in self.reviews
        )

        if self.presented_work_count != expected_presented:
            raise ValueError("presented_work_count mismatch")
        if self.classified_work_count != expected_classified:
            raise ValueError("classified_work_count mismatch")
        if self.unclassified_work_count != expected_unclassified:
            raise ValueError("unclassified_work_count mismatch")
        if self.reviewed_work_count != expected_classified:
            raise ValueError(
                "legacy reviewed_work_count must equal classified_work_count"
            )
        if (
            self.presented_work_count
            != self.classified_work_count
            + self.unclassified_work_count
        ):
            raise ValueError("aggregate adjudication coverage mismatch")
        return self


def _projection_maps(
    projection_report: GroundedFactorProjectionReport,
) -> tuple[
    dict[str, GroundedFactorRelationProjection],
    dict[str, list[GroundedFactorRelationProjection]],
]:
    by_id: dict[str, GroundedFactorRelationProjection] = {}
    by_claim: dict[str, list[GroundedFactorRelationProjection]] = defaultdict(list)

    for projection_set in projection_report.projection_sets:
        for row in projection_set.projections:
            by_id[row.projection_id] = row
            by_claim[row.claim_id].append(row)

    return by_id, by_claim


def _support_query_projection_map(
    plan: SupportingProjectionQueryPlan,
) -> dict[str, set[str]]:
    return {
        row.transport_query_id: set(row.projection_ids)
        for row in plan.transport_queries
    }


def _counter_query_projection_map(
    plan: CounterevidenceProjectionQueryPlan,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
]:
    projection_map = {
        row.transport_query_id: set(row.projection_ids)
        for row in plan.transport_queries
    }
    mode_map = {
        row.transport_query_id: set(row.counterevidence_modes)
        for row in plan.transport_queries
    }
    return projection_map, mode_map


def _second_pass_query_maps(
    plan: ScientificRelationSecondPassPlan | None,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
]:
    if plan is None:
        return {}, {}

    projection_map = {
        row.query_id: set(row.basis_projection_ids)
        for row in plan.bindings
    }
    role_map = {
        row.query_id: {row.role}
        for row in plan.bindings
    }
    return projection_map, role_map


def _work_query_provenance(
    *,
    work: PriorArtWork,
    supporting_query_projection: dict[str, set[str]],
    counter_query_projection: dict[str, set[str]],
    counter_query_modes: dict[str, set[str]],
) -> tuple[set[str], set[str], set[str]]:
    supporting: set[str] = set()
    counter: set[str] = set()
    modes: set[str] = set()

    for query_id in work.retrieval_query_ids:
        supporting.update(
            supporting_query_projection.get(query_id, set())
        )
        counter.update(
            counter_query_projection.get(query_id, set())
        )
        modes.update(
            counter_query_modes.get(query_id, set())
        )

    return supporting, counter, modes


def _merge_work_group(
    works: list[PriorArtWork],
) -> PriorArtWork:
    if not works:
        raise ValueError("cannot merge empty work group")

    ordered = sorted(
        works,
        key=lambda row: (
            -len(str(row.abstract or "")),
            -int(bool(row.doi)),
            -int(row.citation_count or 0),
            row.work_id,
        ),
    )
    base = ordered[0].model_copy(deep=True)

    base.providers = list(
        dict.fromkeys(
            provider
            for row in works
            for provider in row.providers
        )
    )
    base.retrieval_query_ids = list(
        dict.fromkeys(
            query_id
            for row in works
            for query_id in row.retrieval_query_ids
        )
    )
    base.retrieval_claim_ids = list(
        dict.fromkeys(
            claim_id
            for row in works
            for claim_id in row.retrieval_claim_ids
        )
    )

    return base


def _matches_patterns(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    return any(
        re.search(pattern, text, re.I)
        for pattern in patterns
    )


def _relation_reference_text(
    relation: ScientificRelationIR,
) -> str:
    """Structured relation context for candidate compatibility.

    Candidate ranking must not reuse strong_scope_compatibility() as though its
    low-scope result were a hard exclusion. That function was designed for
    strong conflict/directness checks. Here we preserve the full relation
    structure and interpret domain/scope overlap as ranking context.
    """
    return " ".join(
        value
        for value in [
            relation.claim_text,
            *[
                row.surface_text
                for row in relation.endpoint_concepts
            ],
            *[
                row.surface_text
                for row in relation.identity_concepts
            ],
            *[
                row.surface_text
                for row in relation.scope_qualifiers
            ],
            *[
                row.surface_text
                for row in relation.directional_qualifiers
            ],
            relation.observable_concept.surface_text,
            *[
                label.replace("_", " ")
                for label in relation.relation_type_labels
            ],
        ]
        if str(value).strip()
    )


def _term_coverage(
    terms: list[str],
    document: str,
) -> float:
    if not terms:
        return 0.0

    document_tokens = set(
        lexical_content_tokens(document)
    )
    if not document_tokens:
        return 0.0

    scores: list[float] = []

    for term in terms:
        tokens = set(
            lexical_content_tokens(term)
        )
        if not tokens:
            continue
        scores.append(
            len(tokens & document_tokens)
            / len(tokens)
        )

    if not scores:
        return 0.0

    return sum(scores) / len(scores)



def _endpoint_discriminative_evidence(
    relation: ScientificRelationIR,
    document: str,
) -> tuple[
    list[float],
    list[int],
    list[int],
]:
    """Endpoint-specific lexical evidence for bounded candidate ranking.

    Shared endpoint tokens are removed before scoring whenever possible.
    A strong endpoint support requires more than a single generic token:
      - one-token basis: 1 matched token;
      - basis with >=2 tokens: at least 2 matched tokens.

    This is deliberately a ranking criterion, not a scientific exclusion.
    Synonym-heavy papers may remain in neighboring/context pools and can be
    recovered by later semantic adjudication if selected.
    """
    document_tokens = set(
        lexical_content_tokens(document)
    )
    endpoint_token_sets = [
        set(
            lexical_content_tokens(
                row.surface_text
            )
        )
        for row in relation.endpoint_concepts
    ]

    coverages: list[float] = []
    match_counts: list[int] = []
    required_counts: list[int] = []

    for index, tokens in enumerate(endpoint_token_sets):
        if not tokens:
            coverages.append(0.0)
            match_counts.append(0)
            required_counts.append(1)
            continue

        other_tokens = set().union(
            *[
                other
                for other_index, other in enumerate(endpoint_token_sets)
                if other_index != index
            ]
        )
        discriminative = tokens - other_tokens
        basis = discriminative or tokens

        matched = basis & document_tokens
        required = 1 if len(basis) == 1 else 2

        coverages.append(
            len(matched) / len(basis)
        )
        match_counts.append(
            len(matched)
        )
        required_counts.append(
            required
        )

    return (
        coverages,
        match_counts,
        required_counts,
    )



def _relation_anchor_state(
    match_counts: list[int],
    required_counts: list[int],
) -> tuple[
    int,
    bool,
    Literal[
        "PAIR_OR_MULTI_ENDPOINT_ANCHORED",
        "PARTIAL_ENDPOINT_ANCHORED",
        "CONTEXT_ONLY",
    ],
]:
    if len(match_counts) != len(required_counts):
        raise ValueError(
            "endpoint match/requirement count length mismatch"
        )

    supported_flags = [
        matched >= required
        for matched, required in zip(
            match_counts,
            required_counts,
            strict=True,
        )
    ]
    supported = sum(supported_flags)
    all_supported = bool(supported_flags) and all(
        supported_flags
    )

    if all_supported:
        tier = "PAIR_OR_MULTI_ENDPOINT_ANCHORED"
    elif supported:
        tier = "PARTIAL_ENDPOINT_ANCHORED"
    else:
        tier = "CONTEXT_ONLY"

    return supported, all_supported, tier


def _candidate_compatibility(
    *,
    relation: ScientificRelationIR,
    document: str,
    domain_profile: ScientificDomainProfile,
) -> tuple[
    str,
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """Candidate-selection compatibility distinct from strong conflict scope.

    The previous 0071 implementation reused strong_scope_compatibility() as a
    candidate-ranking state. In SERS this marked many relevant papers as
    DOMAIN_OR_SCOPE_MISMATCH simply because not every noncritical scope feature
    overlapped. Candidate selection instead needs:
      - hard typed identity protection,
      - explicit incompatible-domain protection,
      - soft domain/scope ranking for neighboring literature.
    """
    adapter = get_relation_typing_adapter(
        domain_profile.profile_id
    )

    relation_types = list(
        dict.fromkeys(
            relation.relation_type_labels
        )
    )
    relation_type_set = set(relation_types)

    document_types = list(
        dict.fromkeys(
            adapter.type_labels(
                surface_text=document,
                relation_context=document,
            )
        )
    )
    document_type_set = set(document_types)

    document_ambiguity = list(
        dict.fromkeys(
            adapter.ambiguity_labels(
                surface_text=document,
                relation_context=document,
                type_labels=tuple(document_types),
            )
        )
    )

    relation_ambiguity = list(
        dict.fromkeys(
            label
            for concept in [
                *relation.endpoint_concepts,
                *relation.identity_concepts,
                relation.observable_concept,
            ]
            for label in concept.ambiguity_labels
        )
    )

    conflicts = adapter.conflicting_type_pairs(
        relation_types=tuple(relation_types),
        document_types=tuple(document_types),
    )

    reference = _relation_reference_text(
        relation
    )
    document_domains = sorted(
        domain_profile.novelty.domains(
            document
        )
    )
    document_scope = sorted(
        domain_profile.novelty.scope_features(
            document
        )
    )
    relation_domains = set(
        domain_profile.novelty.domains(
            reference
        )
    )
    relation_scope = set(
        domain_profile.novelty.scope_features(
            reference
        )
    )
    document_domain_set = set(
        document_domains
    )
    document_scope_set = set(
        document_scope
    )

    reasons: list[str] = []

    if relation_ambiguity:
        return (
            "AMBIGUOUS_RELATION_IDENTITY",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            [
                "relation_concept_identity_ambiguous",
                *relation_ambiguity,
            ],
        )

    if conflicts:
        return (
            "TYPE_IDENTITY_CONFLICT",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            [
                "typed_concept_identity_conflict",
                *[
                    left + "!=" + right
                    for left, right in conflicts
                ],
            ],
        )

    # A document that literally uses an overloaded identity token without the
    # physical context required by the typed relation should not consume the
    # bounded review budget. This catches e.g. cultural/population "hotspot"
    # records while retaining SERS/plasmonic hotspot documents.
    if (
        "electromagnetic_plasmonic_hotspot"
        in relation_type_set
        and "hotspot_without_typed_physical_context"
        in document_ambiguity
    ):
        return (
            "AMBIGUOUS_DOCUMENT_IDENTITY",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            [
                "overloaded_hotspot_without_required_physical_identity"
            ],
        )

    claim_context = _matches_patterns(
        reference,
        domain_profile.novelty.claim_context_patterns,
    )
    explicit_document_mismatch = _matches_patterns(
        document,
        domain_profile.novelty.document_mismatch_patterns,
    )
    explicit_document_compatible = _matches_patterns(
        document,
        domain_profile.novelty.document_compatible_patterns,
    )

    if (
        claim_context
        and explicit_document_mismatch
        and not explicit_document_compatible
    ):
        return (
            "EXPLICIT_DOMAIN_MISMATCH",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            [domain_profile.novelty.domain_mismatch_reason],
        )

    shared_relation_types = relation_type_set & document_type_set

    # A shared scientific concept type is not, by itself, domain identity.
    # Generic concepts such as measurement reproducibility, composition, or
    # nanostructure design occur across many fields. Promote a type overlap to
    # TYPED_COMPATIBLE only when the document also carries explicit compatible
    # domain context. Without that context, retain the work as neighboring
    # literature rather than spending the highest-priority in-domain review
    # budget on it.
    if shared_relation_types and (
        relation_domains & document_domain_set
        or explicit_document_compatible
    ):
        return (
            "TYPED_COMPATIBLE",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            reasons,
        )

    if relation_domains & document_domain_set:
        return (
            "DOMAIN_COMPATIBLE",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            reasons,
        )

    if explicit_document_compatible:
        return (
            "DOMAIN_COMPATIBLE",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            reasons,
        )

    if shared_relation_types:
        return (
            "NEIGHBORING_SCOPE",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            ["typed_concept_overlap_without_explicit_domain_compatibility"],
        )

    if relation_scope & document_scope_set:
        return (
            "NEIGHBORING_SCOPE",
            relation_types,
            document_types,
            document_ambiguity,
            document_domains,
            ["partial_structured_scope_overlap"],
        )

    return (
        "UNCERTAIN",
        relation_types,
        document_types,
        document_ambiguity,
        document_domains,
        ["insufficient_explicit_candidate_compatibility"],
    )


def _selection_score(
    *,
    work: PriorArtWork,
    compatibility_state: str,
    endpoint_coverage: float,
    endpoint_discriminative_coverages: list[float],
    endpoint_supported_count: int,
    all_endpoints_supported: bool,
    identity_coverage: float,
    supporting_projections: set[str],
    counter_projections: set[str],
    counter_modes: set[str],
    second_pass_roles: set[str],
    projection_by_id: dict[str, GroundedFactorRelationProjection],
) -> float:
    """Bounded candidate-ranking score, not scientific evidence.

    Retrieval provenance is deliberately capped. The previous score added a
    bonus for every hit projection, allowing irrelevant records retrieved by
    many broad queries to outrank scientifically compatible literature.
    """
    score = 0.0

    if work.abstract:
        score += 12.0
    if work.doi:
        score += 0.5

    state_bonus = {
        "TYPED_COMPATIBLE": 7.0,
        "DOMAIN_COMPATIBLE": 6.0,
        "COMPATIBLE": 6.0,  # backward-compatible fixture/state
        "NEIGHBORING_SCOPE": 2.5,
        "UNCERTAIN": 0.0,
        "EXPLICIT_DOMAIN_MISMATCH": -6.0,
        "AMBIGUOUS_DOCUMENT_IDENTITY": -12.0,
        "AMBIGUOUS_RELATION_IDENTITY": -16.0,
        "TYPE_IDENTITY_CONFLICT": -20.0,
        "DOMAIN_OR_SCOPE_MISMATCH": -3.0,  # legacy artifact compatibility
    }
    score += state_bonus.get(
        compatibility_state,
        0.0,
    )

    score += 4.0 * max(
        0.0,
        min(1.0, endpoint_coverage),
    )

    if endpoint_discriminative_coverages:
        mean_discriminative = (
            sum(endpoint_discriminative_coverages)
            / len(endpoint_discriminative_coverages)
        )
        score += 8.0 * mean_discriminative

    # Anchor tier is handled lexicographically by the bounded selector;
    # keep only a small within-tier score contribution here.
    score += 1.0 * endpoint_supported_count
    if all_endpoints_supported:
        score += 2.0

    score += 2.0 * max(
        0.0,
        min(1.0, identity_coverage),
    )

    # Capped provenance contribution: retrieval breadth is useful for recall
    # but is not relation evidence.
    support_kinds = {
        projection_by_id[projection_id].projection_kind
        for projection_id in supporting_projections
        if projection_id in projection_by_id
    }
    if "FULL_RELATION" in support_kinds:
        score += 2.0
    if "LOWER_ORDER_FACTOR_SUBSET" in support_kinds:
        score += 1.0
    if "BASE_RELATION" in support_kinds:
        score += 0.5

    if counter_projections:
        score += 1.5
    if "DISTRIBUTIONAL_DIVERGENCE" in counter_modes:
        score += 0.5
    if second_pass_roles:
        score += 1.0

    # Citation count is only a stable tie-break contribution, never evidence.
    if work.citation_count:
        score += min(
            0.5,
            math.log10(max(1, work.citation_count)) * 0.12,
        )

    return round(score, 6)


def _candidate_for_group(
    *,
    relation: ScientificRelationIR,
    works: list[PriorArtWork],
    source_work_ids: list[str],
    domain_profile: ScientificDomainProfile,
    supporting_query_projection: dict[str, set[str]],
    counter_query_projection: dict[str, set[str]],
    counter_query_modes: dict[str, set[str]],
    second_pass_query_projection: dict[str, set[str]],
    second_pass_query_roles: dict[str, set[str]],
    projection_by_id: dict[str, GroundedFactorRelationProjection],
) -> ProjectionRelationWorkCandidate:
    merged = _merge_work_group(works)
    document = _work_document(
        merged
    )

    (
        compatibility_state,
        relation_types,
        document_types,
        document_ambiguity,
        document_domains,
        compatibility_reasons,
    ) = _candidate_compatibility(
        relation=relation,
        document=document,
        domain_profile=domain_profile,
    )

    document_scope = sorted(
        domain_profile.novelty.scope_features(
            document
        )
    )

    endpoint_coverage = _term_coverage(
        [
            row.surface_text
            for row in relation.endpoint_concepts
        ],
        document,
    )
    (
        endpoint_discriminative_coverages,
        endpoint_discriminative_match_counts,
        endpoint_required_match_counts,
    ) = _endpoint_discriminative_evidence(
        relation,
        document,
    )
    (
        endpoint_supported_count,
        all_endpoints_supported,
        relation_anchor_tier,
    ) = _relation_anchor_state(
        endpoint_discriminative_match_counts,
        endpoint_required_match_counts,
    )

    identity_coverage = _term_coverage(
        [
            row.surface_text
            for row in relation.identity_concepts
        ],
        document,
    )

    supporting: set[str] = set()
    counter: set[str] = set()
    modes: set[str] = set()
    second_pass_projections: set[str] = set()
    second_pass_roles: set[str] = set()

    for work in works:
        work_support, work_counter, work_modes = _work_query_provenance(
            work=work,
            supporting_query_projection=supporting_query_projection,
            counter_query_projection=counter_query_projection,
            counter_query_modes=counter_query_modes,
        )
        supporting.update(work_support)
        counter.update(work_counter)
        modes.update(work_modes)

        for query_id in work.retrieval_query_ids:
            second_pass_projections.update(
                second_pass_query_projection.get(
                    query_id,
                    set(),
                )
            )
            second_pass_roles.update(
                second_pass_query_roles.get(
                    query_id,
                    set(),
                )
            )

    support_rows = [
        projection_by_id[projection_id]
        for projection_id in sorted(supporting)
        if projection_id in projection_by_id
    ]

    lanes: list[
        Literal[
            "SUPPORTING_PRIOR_ART",
            "COUNTEREVIDENCE_PRIOR_ART",
            "SEMANTIC_SECOND_PASS_SUPPORTING",
            "SEMANTIC_SECOND_PASS_COUNTEREVIDENCE",
        ]
    ] = []
    if supporting:
        lanes.append("SUPPORTING_PRIOR_ART")
    if counter or modes:
        lanes.append("COUNTEREVIDENCE_PRIOR_ART")
    if second_pass_roles & {
        "SEMANTIC_ENDPOINT_PAIR",
        "SEMANTIC_FULL_RELATION",
    }:
        lanes.append("SEMANTIC_SECOND_PASS_SUPPORTING")
    if "SEMANTIC_COUNTEREVIDENCE" in second_pass_roles:
        lanes.append("SEMANTIC_SECOND_PASS_COUNTEREVIDENCE")

    return ProjectionRelationWorkCandidate(
        review_work_id=merged.work_id,
        source_work_ids=sorted(set(source_work_ids)),
        title=merged.title,
        year=merged.year,
        doi=merged.doi,
        url=merged.url,
        abstract=merged.abstract,
        citation_count=merged.citation_count,
        typed_compatibility_state=compatibility_state,
        relation_type_labels=relation_types,
        document_type_labels=document_types,
        document_ambiguity_labels=document_ambiguity,
        document_domain_labels=document_domains,
        document_scope_features=document_scope,
        typed_reason_codes=list(
            dict.fromkeys(
                compatibility_reasons
            )
        ),
        endpoint_lexical_coverage=endpoint_coverage,
        endpoint_discriminative_coverages=(
            endpoint_discriminative_coverages
        ),
        endpoint_discriminative_match_counts=(
            endpoint_discriminative_match_counts
        ),
        endpoint_required_match_counts=(
            endpoint_required_match_counts
        ),
        endpoint_supported_count=endpoint_supported_count,
        all_endpoints_supported=all_endpoints_supported,
        relation_anchor_tier=relation_anchor_tier,
        identity_lexical_coverage=identity_coverage,
        supporting_projection_ids=sorted(supporting),
        supporting_projection_kinds=list(
            dict.fromkeys(
                row.projection_kind
                for row in support_rows
            )
        ),
        supporting_factor_orders=sorted(
            {
                row.factor_order
                for row in support_rows
            }
        ),
        counterevidence_projection_ids=sorted(counter),
        counterevidence_modes=sorted(modes),
        selected_lanes=lanes,
        second_pass_projection_ids=sorted(
            second_pass_projections
        ),
        second_pass_roles=sorted(
            second_pass_roles
        ),
        selection_score=_selection_score(
            work=merged,
            compatibility_state=compatibility_state,
            endpoint_coverage=endpoint_coverage,
            endpoint_discriminative_coverages=(
                endpoint_discriminative_coverages
            ),
            endpoint_supported_count=endpoint_supported_count,
            all_endpoints_supported=all_endpoints_supported,
            identity_coverage=identity_coverage,
            supporting_projections=supporting,
            counter_projections=counter,
            counter_modes=modes,
            second_pass_roles=second_pass_roles,
            projection_by_id=projection_by_id,
        ),
    )


def _balanced_select(
    rows: list[ProjectionRelationWorkCandidate],
    *,
    max_review_works: int,
) -> list[ProjectionRelationWorkCandidate]:
    if max_review_works < 1:
        raise ValueError("max_review_works must be >= 1")

    hard_excluded_states = {
        "TYPE_IDENTITY_CONFLICT",
        "AMBIGUOUS_RELATION_IDENTITY",
        "AMBIGUOUS_DOCUMENT_IDENTITY",
    }

    normal = [
        row
        for row in rows
        if (
            row.typed_compatibility_state
            not in hard_excluded_states
            and row.typed_compatibility_state
            != "EXPLICIT_DOMAIN_MISMATCH"
        )
    ]
    contextual = [
        row
        for row in rows
        if (
            row.typed_compatibility_state
            == "EXPLICIT_DOMAIN_MISMATCH"
            and "COUNTEREVIDENCE_PRIOR_ART"
            in row.selected_lanes
        )
    ]

    def key(row: ProjectionRelationWorkCandidate) -> tuple:
        # Relation anchoring precedes retrieval breadth. A paper that touches
        # both endpoint-specific lexical bases is a better bounded adjudication
        # candidate than a broadly retrieved domain review that only mentions
        # one side of the proposed relation.
        in_domain = row.typed_compatibility_state in {
            "TYPED_COMPATIBLE",
            "DOMAIN_COMPATIBLE",
            "COMPATIBLE",
        }
        neighboring = row.typed_compatibility_state in {
            "NEIGHBORING_SCOPE",
            "UNCERTAIN",
        }
        pair = (
            row.relation_anchor_tier
            == "PAIR_OR_MULTI_ENDPOINT_ANCHORED"
        )
        partial = (
            row.relation_anchor_tier
            == "PARTIAL_ENDPOINT_ANCHORED"
        )

        # Scientific review budget priority:
        # 5 in-domain + both endpoints
        # 4 in-domain + one endpoint
        # 3 neighboring + both endpoints
        # 2 in-domain context-only
        # 1 neighboring + one endpoint
        # 0 remaining/contextual fallback
        if in_domain and pair:
            relevance_bucket = 5
        elif in_domain and partial:
            relevance_bucket = 4
        elif neighboring and pair:
            relevance_bucket = 3
        elif in_domain:
            relevance_bucket = 2
        elif neighboring and partial:
            relevance_bucket = 1
        else:
            relevance_bucket = 0

        return (
            -int(bool(row.abstract)),
            -relevance_bucket,
            -row.endpoint_supported_count,
            -row.selection_score,
            -(row.citation_count or 0),
            row.review_work_id,
        )

    normal.sort(key=key)
    contextual.sort(key=key)

    contextual_quota = min(
        len(contextual),
        max(0, max_review_works // 10),
    )
    normal_budget = max_review_works - contextual_quota

    supporting = [
        row
        for row in normal
        if (
            "SUPPORTING_PRIOR_ART" in row.selected_lanes
            or "SEMANTIC_SECOND_PASS_SUPPORTING" in row.selected_lanes
        )
    ]
    counter = [
        row
        for row in normal
        if (
            "COUNTEREVIDENCE_PRIOR_ART" in row.selected_lanes
            or "SEMANTIC_SECOND_PASS_COUNTEREVIDENCE" in row.selected_lanes
        )
    ]

    support_quota = (normal_budget + 1) // 2
    counter_quota = normal_budget // 2

    selected: dict[str, ProjectionRelationWorkCandidate] = {}

    for row in supporting[:support_quota]:
        selected[row.review_work_id] = row

    for row in counter[:counter_quota]:
        selected[row.review_work_id] = row

    for row in normal:
        if len(selected) >= normal_budget:
            break
        selected.setdefault(row.review_work_id, row)

    # Preserve only a small explicit-domain-mismatch budget for genuine
    # neighboring-scope/contextual-conflict discovery.
    for row in contextual[:contextual_quota]:
        selected.setdefault(row.review_work_id, row)

    result = list(selected.values())
    result.sort(key=key)
    return result[:max_review_works]


def build_projection_relation_candidate_report(
    *,
    relation_ir_report: ScientificRelationIRReport,
    projection_report: GroundedFactorProjectionReport,
    supporting_plan: SupportingProjectionQueryPlan,
    supporting_packet: PriorArtPacket,
    counterevidence_plan: CounterevidenceProjectionQueryPlan,
    counterevidence_packet: PriorArtPacket,
    domain_profile: ScientificDomainProfile,
    max_review_works_per_claim: int = 20,
    second_pass_plan: ScientificRelationSecondPassPlan | None = None,
    second_pass_packet: PriorArtPacket | None = None,
) -> ProjectionRelationCandidateReport:
    if max_review_works_per_claim < 1:
        raise ValueError(
            "max_review_works_per_claim must be >= 1"
        )

    if (
        supporting_packet.source_portfolio_id
        != supporting_plan.source_portfolio_id
    ):
        raise ValueError("supporting packet/plan portfolio mismatch")
    if (
        counterevidence_packet.source_portfolio_id
        != counterevidence_plan.source_portfolio_id
    ):
        raise ValueError("counterevidence packet/plan portfolio mismatch")
    if (
        supporting_plan.source_portfolio_id
        != counterevidence_plan.source_portfolio_id
    ):
        raise ValueError(
            "supporting/counterevidence portfolio mismatch"
        )

    if (second_pass_plan is None) != (second_pass_packet is None):
        raise ValueError(
            "second_pass_plan and second_pass_packet must be supplied together"
        )

    projection_by_id, projection_by_claim = _projection_maps(
        projection_report
    )
    support_query_map = _support_query_projection_map(
        supporting_plan
    )
    counter_query_map, counter_mode_map = (
        _counter_query_projection_map(
            counterevidence_plan
        )
    )
    (
        second_pass_query_map,
        second_pass_role_map,
    ) = _second_pass_query_maps(
        second_pass_plan
    )

    relation_by_claim = {
        row.claim_id: row
        for row in relation_ir_report.relations
    }

    all_works = [
        *supporting_packet.works,
        *counterevidence_packet.works,
        *(
            second_pass_packet.works
            if second_pass_packet is not None
            else []
        ),
    ]

    works_by_claim: dict[str, list[PriorArtWork]] = defaultdict(list)

    known_claim_ids = set(relation_by_claim)
    for work in all_works:
        claim_ids = set(work.retrieval_claim_ids) & known_claim_ids
        for claim_id in claim_ids:
            works_by_claim[claim_id].append(work)

    claim_rows: list[ProjectionRelationClaimCandidateSet] = []

    for claim_id, relation in relation_by_claim.items():
        source_works = works_by_claim.get(claim_id, [])

        grouped: dict[str, list[PriorArtWork]] = defaultdict(list)
        source_ids_by_group: dict[str, list[str]] = defaultdict(list)

        for work in source_works:
            key = _work_identity_key(work)
            grouped[key].append(work)
            source_ids_by_group[key].append(work.work_id)

        all_candidates = [
            _candidate_for_group(
                relation=relation,
                works=group,
                source_work_ids=source_ids_by_group[key],
                domain_profile=domain_profile,
                supporting_query_projection=support_query_map,
                counter_query_projection=counter_query_map,
                counter_query_modes=counter_mode_map,
                second_pass_query_projection=second_pass_query_map,
                second_pass_query_roles=second_pass_role_map,
                projection_by_id=projection_by_id,
            )
            for key, group in grouped.items()
        ]

        selected = _balanced_select(
            all_candidates,
            max_review_works=max_review_works_per_claim,
        )

        claim_projections = projection_by_claim.get(
            claim_id,
            [],
        )
        projection_ids = [
            row.projection_id
            for row in claim_projections
        ]
        full_ids = [
            row.projection_id
            for row in claim_projections
            if row.projection_kind == "FULL_RELATION"
        ]
        lower_ids = [
            row.projection_id
            for row in claim_projections
            if row.projection_kind
            in {
                "BASE_RELATION",
                "LOWER_ORDER_FACTOR_SUBSET",
            }
        ]

        claim_rows.append(
            ProjectionRelationClaimCandidateSet(
                hypothesis_id=relation.hypothesis_id,
                claim_id=claim_id,
                relation_ir_id=relation.relation_ir_id,
                claim_text=relation.claim_text,
                endpoint_terms=[
                    row.surface_text
                    for row in relation.endpoint_concepts
                ],
                identity_terms=[
                    row.surface_text
                    for row in relation.identity_concepts
                ],
                scope_terms=[
                    row.surface_text
                    for row in relation.scope_qualifiers
                ],
                directional_terms=[
                    row.surface_text
                    for row in relation.directional_qualifiers
                ],
                projection_ids=projection_ids,
                full_projection_ids=full_ids,
                lower_order_projection_ids=lower_ids,
                candidates=selected,
                candidate_count=len(selected),
                abstract_candidate_count=sum(
                    bool(row.abstract)
                    for row in selected
                ),
                typed_identity_excluded_work_count=sum(
                    row.typed_compatibility_state
                    in {
                        "TYPE_IDENTITY_CONFLICT",
                        "AMBIGUOUS_RELATION_IDENTITY",
                        "AMBIGUOUS_DOCUMENT_IDENTITY",
                    }
                    for row in all_candidates
                ),
                source_unique_work_count=len(grouped),
                max_review_works=max_review_works_per_claim,
            )
        )

    claim_rows.sort(
        key=lambda row: (
            row.hypothesis_id,
            row.claim_id,
        )
    )

    return ProjectionRelationCandidateReport(
        report_id=_stable_id(
            "projection_relation_candidate_report",
            relation_ir_report.report_id,
            projection_report.report_id,
            supporting_plan.plan_id,
            supporting_packet.packet_id,
            counterevidence_plan.plan_id,
            counterevidence_packet.packet_id,
            max_review_works_per_claim,
            *[
                (
                    row.claim_id,
                    *[
                        candidate.review_work_id
                        for candidate in row.candidates
                    ],
                )
                for row in claim_rows
            ],
        ),
        source_relation_ir_report_id=relation_ir_report.report_id,
        source_projection_report_id=projection_report.report_id,
        source_supporting_query_plan_id=supporting_plan.plan_id,
        source_supporting_prior_art_packet_id=supporting_packet.packet_id,
        source_counterevidence_query_plan_id=counterevidence_plan.plan_id,
        source_counterevidence_prior_art_packet_id=counterevidence_packet.packet_id,
        claims=claim_rows,
        claim_count=len(claim_rows),
        total_review_candidate_count=sum(
            row.candidate_count
            for row in claim_rows
        ),
    )


_ADJUDICATION_SYSTEM = """You adjudicate the relationship between bounded prior-art records and ONE typed scientific relation.

This is NOT a novelty judgment and NOT a truth judgment. You must not say that a claim is novel, unprecedented, first, absent from the literature, or non-obvious.

Use ONLY the supplied title/abstract metadata and the supplied relation/projection structure.

ALLOWED RELATIONSHIPS
- DIRECT_PRIOR_ART: abstract-backed record explicitly states/tests essentially the same FULL scientific relation.
- PARTIAL_PRIOR_ART: abstract-backed record establishes a substantial part of the same relation nucleus but not the full typed relation.
- LOWER_ORDER_RELATION_PRIOR_ART: abstract-backed record explicitly establishes a BASE or proper lower-order projection while not establishing the FULL relation.
- DIRECTIONAL_COUNTEREVIDENCE: abstract-backed neighboring evidence materially challenges the proposed ordered relation or shows weak/absent/opposite/regime-dependent behavior.
- CONTEXTUAL_CONFLICT: abstract-backed evidence undermines a bridge/assumption in a scientifically neighboring but scope-mismatched context.
- CONFLICTING_PRIOR_ART: abstract-backed evidence directly reports a materially opposing relation in sufficiently overlapping scientific scope.
- COMPONENT_ONLY: record discusses relevant variables/components but does not establish the required multivariable relation.
- TITLE_ONLY_NEIGHBOR: relevant title but no abstract-backed relation can be verified.
- UNRELATED: record does not materially bear on the relation.
- INSUFFICIENT_METADATA: supplied metadata is insufficient to judge.

PROJECTION CONTRACT
- DIRECT_PRIOR_ART must cite at least one FULL_RELATION projection ID.
- LOWER_ORDER_RELATION_PRIOR_ART must cite at least one BASE_RELATION or LOWER_ORDER_FACTOR_SUBSET projection ID.
- Do not infer a relation from co-mention or separate main effects.
- Retrieval provenance is search provenance only. A paper retrieved by a counterevidence query is not automatically counterevidence.

COUNTEREVIDENCE CONTRACT
- counterevidence_modes may only be copied from modes actually listed for that work.
- second_pass_roles may only be copied from second_pass_roles actually listed for that work.
- SEMANTIC_COUNTEREVIDENCE is RETRIEVAL PROVENANCE ONLY. It does not itself establish counterevidence.
- Use DIRECTIONAL_COUNTEREVIDENCE only when the abstract materially weakens the proposed relation/direction.
- A DIRECTIONAL_COUNTEREVIDENCE record must have either a listed legacy counterevidence_mode or listed SEMANTIC_COUNTEREVIDENCE second-pass provenance, plus an exact supporting abstract span.
- A bridge-decoupling result in a neighboring scope may be CONTEXTUAL_CONFLICT rather than exact contradiction.
- Use CONFLICTING_PRIOR_ART only for a materially opposing relation in substantially overlapping scope.

EVIDENCE-SPAN CONTRACT
- For every strong relationship (DIRECT, PARTIAL, LOWER_ORDER, DIRECTIONAL_COUNTEREVIDENCE, CONTEXTUAL_CONFLICT, CONFLICTING), evidence_span MUST be one exact contiguous span copied from the supplied ABSTRACT, not a paraphrase.
- If no exact abstract span supports the relationship, do not use a strong relationship.
- For COMPONENT_ONLY/TITLE_ONLY_NEIGHBOR/UNRELATED/INSUFFICIENT_METADATA, evidence_span may be empty.

WORK-ID CONTRACT
- work_id must be copied byte-for-byte from ALLOWED_WORK_IDS.
- basis_projection_ids must be copied from ALLOWED_PROJECTION_IDS.
- Do not invent IDs.

Return exactly one record for every work in ALLOWED_WORK_IDS.
Return at most one record per work. Do not omit any allowed work.
If a record has no material bearing, use UNRELATED; if metadata is insufficient,
use INSUFFICIENT_METADATA. Do not use omission as a relationship label.
Your interpretation must be bounded to the supplied reviewed set."""


def _prompt_for_claim(
    row: ProjectionRelationClaimCandidateSet,
    projection_by_id: dict[str, GroundedFactorRelationProjection],
) -> str:
    lines = [
        "TYPED RELATION",
        "==============",
        f"hypothesis_id: {row.hypothesis_id}",
        f"claim_id: {row.claim_id}",
        f"claim_text: {row.claim_text}",
        f"endpoints: {row.endpoint_terms!r}",
        f"identity_terms: {row.identity_terms!r}",
        f"scope_terms: {row.scope_terms!r}",
        f"directional_terms: {row.directional_terms!r}",
        "",
        "PROJECTIONS",
        "===========",
    ]

    for projection_id in row.projection_ids:
        projection = projection_by_id[projection_id]
        labels = [
            binding.group_label
            for binding in projection.factor_bindings
        ]
        lines.extend(
            [
                f"projection_id: {projection.projection_id}",
                f"kind: {projection.projection_kind}",
                f"factor_order: {projection.factor_order}",
                f"retained_factors: {labels!r}",
                f"target_terms: {projection.canonical_search_terms!r}",
                "",
            ]
        )

    lines.extend(
        [
            "REVIEW CANDIDATES",
            "=================",
        ]
    )

    for index, work in enumerate(row.candidates, start=1):
        lines.extend(
            [
                f"[{index}] work_id={work.review_work_id}",
                f"title: {work.title}",
                f"year: {work.year}",
                f"doi: {work.doi}",
                f"typed_compatibility_state: {work.typed_compatibility_state}",
                f"supporting_projection_ids: {work.supporting_projection_ids!r}",
                f"counterevidence_projection_ids: {work.counterevidence_projection_ids!r}",
                f"counterevidence_modes: {work.counterevidence_modes!r}",
                f"second_pass_roles: {work.second_pass_roles!r}",
                "abstract:",
                (
                    work.abstract
                    if work.abstract
                    else "[NO ABSTRACT AVAILABLE]"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "ALLOWED_WORK_IDS",
            "================",
            *[
                row.review_work_id
                for row in row.candidates
            ],
            "",
            "ALLOWED_PROJECTION_IDS",
            "======================",
            *row.projection_ids,
            "",
            "CLASSIFICATION COVERAGE CONTRACT",
            "Return exactly one match record for every ALLOWED_WORK_ID.",
            "Every presented work must receive one allowed relationship label.",
            "",
            "Do not infer literature-wide absence.",
            "Do not make a novelty or non-obviousness judgment.",
        ]
    )

    return "\n".join(lines)


class InstructorProjectionRelationAdjudicationBackend:
    backend_name = "instructor_projection_relation_adjudication"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        capture_prompts: bool = False,
    ) -> None:
        self.model_name = model
        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)
        self.base_url = base_url
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.capture_prompts = bool(capture_prompts)
        self.prompt_records: list[dict[str, str]] = []
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env}."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Projection relation adjudication requires openai + instructor."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=instructor.Mode.JSON,
        )
        return self._client

    def review(
        self,
        *,
        candidate_set: ProjectionRelationClaimCandidateSet,
        projection_by_id: dict[str, GroundedFactorRelationProjection],
    ) -> ProjectionRelationClaimReviewDraft:
        user = _prompt_for_claim(
            candidate_set,
            projection_by_id,
        )

        if self.capture_prompts:
            self.prompt_records.append(
                {
                    "claim_id": candidate_set.claim_id,
                    "system_prompt": _ADJUDICATION_SYSTEM,
                    "user_prompt": user,
                }
            )

        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=ProjectionRelationClaimReviewDraft,
            messages=[
                {
                    "role": "system",
                    "content": _ADJUDICATION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            temperature=0.0,
            max_retries=self.parse_retries,
            telemetry_context={
                "pipeline":
                    "projection_relation_adjudication_shadow",
                "stage":
                    "bounded_relation_adjudication",
                "call_kind":
                    "prior_art_review",
                "claim_id":
                    candidate_set.claim_id,
            },
        )

        if not isinstance(
            result,
            ProjectionRelationClaimReviewDraft,
        ):
            result = ProjectionRelationClaimReviewDraft.model_validate(
                result
            )

        return result


def _downgrade(
    relationship: ProjectionPriorArtRelationship,
    *,
    candidate: ProjectionRelationWorkCandidate,
    basis_projection_ids: list[str],
    counterevidence_modes: list[str],
    second_pass_roles: list[str],
    evidence_span: str,
    projection_by_id: dict[str, GroundedFactorRelationProjection],
) -> tuple[
    ProjectionPriorArtRelationship,
    list[str],
]:
    reasons: list[str] = []
    compiled = relationship

    abstract = str(candidate.abstract or "")

    if relationship in _STRONG_ABSTRACT_RELATIONSHIPS:
        if not abstract:
            return (
                "TITLE_ONLY_NEIGHBOR",
                ["strong_relation_downgraded_without_abstract"],
            )
        if not _exact_span_present(
            evidence_span,
            abstract,
        ):
            return (
                "INSUFFICIENT_METADATA",
                ["strong_relation_downgraded_without_exact_abstract_span"],
            )

    if candidate.typed_compatibility_state in {
        "TYPE_IDENTITY_CONFLICT",
        "AMBIGUOUS_RELATION_IDENTITY",
        "AMBIGUOUS_DOCUMENT_IDENTITY",
    } and relationship in _STRONG_ABSTRACT_RELATIONSHIPS:
        return (
            "UNRELATED",
            ["strong_relation_downgraded_for_typed_identity_conflict"],
        )

    if (
        candidate.typed_compatibility_state == "EXPLICIT_DOMAIN_MISMATCH"
        and relationship in _STRONG_ABSTRACT_RELATIONSHIPS
    ):
        return (
            "UNRELATED",
            ["strong_relation_downgraded_for_explicit_domain_mismatch"],
        )

    valid_projection_ids = [
        projection_id
        for projection_id in basis_projection_ids
        if projection_id in projection_by_id
    ]

    if relationship == "DIRECT_PRIOR_ART":
        if not any(
            projection_by_id[projection_id].projection_kind
            == "FULL_RELATION"
            for projection_id in valid_projection_ids
        ):
            compiled = "COMPONENT_ONLY"
            reasons.append(
                "direct_prior_art_requires_full_projection_basis"
            )
        elif (
            not candidate.all_endpoints_supported
            or candidate.relation_anchor_tier
            != "PAIR_OR_MULTI_ENDPOINT_ANCHORED"
        ):
            compiled = (
                "PARTIAL_PRIOR_ART"
                if candidate.endpoint_supported_count > 0
                else "COMPONENT_ONLY"
            )
            reasons.append(
                "direct_prior_art_requires_pair_or_multi_endpoint_anchor"
            )
        elif (
            candidate.typed_compatibility_state
            not in {
                "COMPATIBLE",
                "TYPED_COMPATIBLE",
                "DOMAIN_COMPATIBLE",
            }
        ):
            compiled = "PARTIAL_PRIOR_ART"
            reasons.append(
                "direct_prior_art_downgraded_for_scope_uncertainty"
            )

    elif relationship == "LOWER_ORDER_RELATION_PRIOR_ART":
        if not any(
            projection_by_id[projection_id].projection_kind
            in {
                "BASE_RELATION",
                "LOWER_ORDER_FACTOR_SUBSET",
            }
            for projection_id in valid_projection_ids
        ):
            compiled = "COMPONENT_ONLY"
            reasons.append(
                "lower_order_prior_art_requires_lower_order_projection_basis"
            )
        elif (
            candidate.typed_compatibility_state
            in {
                "DOMAIN_OR_SCOPE_MISMATCH",
                "EXPLICIT_DOMAIN_MISMATCH",
            }
        ):
            compiled = "COMPONENT_ONLY"
            reasons.append(
                "lower_order_prior_art_downgraded_for_scope_mismatch"
            )

    elif relationship == "DIRECTIONAL_COUNTEREVIDENCE":
        valid_modes = set(counterevidence_modes) & set(
            candidate.counterevidence_modes
        )
        valid_second_pass_roles = set(second_pass_roles) & set(
            candidate.second_pass_roles
        )
        has_counter_retrieval_provenance = bool(
            valid_modes
            or "SEMANTIC_COUNTEREVIDENCE"
            in valid_second_pass_roles
        )
        if not has_counter_retrieval_provenance:
            compiled = "COMPONENT_ONLY"
            reasons.append(
                "directional_counterevidence_requires_retrieved_counterevidence_provenance"
            )

    elif relationship == "CONFLICTING_PRIOR_ART":
        if (
            candidate.typed_compatibility_state
            not in {
                "COMPATIBLE",
                "TYPED_COMPATIBLE",
                "DOMAIN_COMPATIBLE",
            }
        ):
            compiled = "CONTEXTUAL_CONFLICT"
            reasons.append(
                "conflicting_prior_art_downgraded_for_scope_mismatch"
            )

    return compiled, reasons


def _relation_state(
    matches: list[CompiledProjectionRelationMatch],
) -> CompiledClaimRelationState:
    direct = any(
        row.relationship == "DIRECT_PRIOR_ART"
        for row in matches
    )
    conflicting = any(
        row.relationship == "CONFLICTING_PRIOR_ART"
        for row in matches
    )
    counter = any(
        row.relationship
        in {
            "DIRECTIONAL_COUNTEREVIDENCE",
            "CONTEXTUAL_CONFLICT",
        }
        for row in matches
    )
    lower = any(
        row.relationship == "LOWER_ORDER_RELATION_PRIOR_ART"
        for row in matches
    )

    signal_count = sum(
        (direct, conflicting, counter, lower)
    )
    if signal_count >= 2:
        return "MIXED_RELATION_SIGNALS"
    if direct:
        return "DIRECT_RELATION_FOUND"
    if conflicting:
        return "CONFLICTING_RELATION_FOUND"
    if counter:
        return "COUNTEREVIDENCE_FOUND"
    if lower:
        return "LOWER_ORDER_RELATION_FOUND"

    if any(
        row.relationship
        in {
            "COMPONENT_ONLY",
            "PARTIAL_PRIOR_ART",
            "CONTEXTUAL_CONFLICT",
        }
        for row in matches
    ):
        return "COMPONENTS_OR_CONTEXT_ONLY"

    if matches and all(
        row.relationship
        in {
            "INSUFFICIENT_METADATA",
            "TITLE_ONLY_NEIGHBOR",
        }
        for row in matches
    ):
        return "INSUFFICIENT_METADATA"

    return "NO_MATERIAL_RELATION_SIGNAL"


def compile_projection_relation_claim_review(
    *,
    candidate_set: ProjectionRelationClaimCandidateSet,
    draft: ProjectionRelationClaimReviewDraft,
    projection_by_id: dict[str, GroundedFactorRelationProjection],
) -> ProjectionRelationClaimReview:
    candidates = {
        row.review_work_id: row
        for row in candidate_set.candidates
    }
    allowed_projection_ids = set(
        candidate_set.projection_ids
    )

    unknown = sorted(
        {
            row.work_id
            for row in draft.matches
        }
        - set(candidates)
    )

    compiled: list[CompiledProjectionRelationMatch] = []

    for row in draft.matches:
        candidate = candidates.get(row.work_id)
        if candidate is None:
            continue

        basis_projection_ids = [
            projection_id
            for projection_id in row.basis_projection_ids
            if projection_id in allowed_projection_ids
        ]
        invalid_projection_ids = sorted(
            set(row.basis_projection_ids)
            - allowed_projection_ids
        )

        valid_modes = [
            mode
            for mode in row.counterevidence_modes
            if mode in candidate.counterevidence_modes
        ]
        invalid_modes = sorted(
            set(row.counterevidence_modes)
            - set(candidate.counterevidence_modes)
        )

        valid_second_pass_roles = [
            role
            for role in row.second_pass_roles
            if role in candidate.second_pass_roles
        ]
        invalid_second_pass_roles = sorted(
            set(row.second_pass_roles)
            - set(candidate.second_pass_roles)
        )

        relationship, reasons = _downgrade(
            row.relationship,
            candidate=candidate,
            basis_projection_ids=basis_projection_ids,
            counterevidence_modes=valid_modes,
            second_pass_roles=valid_second_pass_roles,
            evidence_span=row.evidence_span,
            projection_by_id=projection_by_id,
        )

        if invalid_projection_ids:
            reasons.append(
                "unknown_or_out_of_claim_projection_id_dropped"
            )
        if invalid_modes:
            reasons.append(
                "unretrieved_counterevidence_mode_dropped"
            )
        if invalid_second_pass_roles:
            reasons.append(
                "unretrieved_second_pass_role_dropped"
            )

        compiled.append(
            CompiledProjectionRelationMatch(
                work_id=candidate.review_work_id,
                source_work_ids=list(candidate.source_work_ids),
                relationship=relationship,
                original_relationship=row.relationship,
                confidence=row.confidence,
                basis_projection_ids=basis_projection_ids,
                counterevidence_modes=valid_modes,
                second_pass_roles=valid_second_pass_roles,
                evidence_span=row.evidence_span,
                rationale=row.rationale,
                typed_compatibility_state=(
                    candidate.typed_compatibility_state
                ),
                abstract_available=bool(candidate.abstract),
                deterministic_reason_codes=list(
                    dict.fromkeys(reasons)
                ),
            )
        )

    compiled.sort(
        key=lambda row: (
            -row.confidence,
            row.relationship,
            row.work_id,
        )
    )

    classified_ids = {
        row.work_id
        for row in compiled
    }
    unclassified_ids = sorted(
        set(candidates)
        - classified_ids
    )

    def ids(relationship: str) -> list[str]:
        return [
            row.work_id
            for row in compiled
            if row.relationship == relationship
        ]

    return ProjectionRelationClaimReview(
        hypothesis_id=candidate_set.hypothesis_id,
        claim_id=candidate_set.claim_id,
        relation_ir_id=candidate_set.relation_ir_id,
        relation_state=(
            "INSUFFICIENT_METADATA"
            if not candidate_set.projection_ids
            else _relation_state(compiled)
        ),
        matches=compiled,
        presented_work_count=len(candidate_set.candidates),
        classified_work_count=len(compiled),
        unclassified_work_ids=unclassified_ids,
        reviewed_work_count=len(compiled),
        direct_prior_art_work_ids=ids("DIRECT_PRIOR_ART"),
        partial_prior_art_work_ids=ids("PARTIAL_PRIOR_ART"),
        lower_order_prior_art_work_ids=ids(
            "LOWER_ORDER_RELATION_PRIOR_ART"
        ),
        directional_counterevidence_work_ids=ids(
            "DIRECTIONAL_COUNTEREVIDENCE"
        ),
        contextual_conflict_work_ids=ids(
            "CONTEXTUAL_CONFLICT"
        ),
        conflicting_prior_art_work_ids=ids(
            "CONFLICTING_PRIOR_ART"
        ),
        component_only_work_ids=ids("COMPONENT_ONLY"),
        reviewer_unknown_work_ids=unknown,
        interpretation=draft.interpretation,
    )


def build_projection_relation_adjudication_report(
    *,
    candidate_report: ProjectionRelationCandidateReport,
    drafts_by_claim_id: dict[str, ProjectionRelationClaimReviewDraft],
    projection_report: GroundedFactorProjectionReport,
    backend_name: str,
    model_name: str,
    llm_calls_performed: int,
) -> ProjectionRelationAdjudicationReport:
    projection_by_id, _projection_by_claim = _projection_maps(
        projection_report
    )

    reviews: list[ProjectionRelationClaimReview] = []

    for candidate_set in candidate_report.claims:
        draft = drafts_by_claim_id.get(
            candidate_set.claim_id
        )
        if draft is None:
            raise ValueError(
                "missing projection relation adjudication draft for claim "
                + candidate_set.claim_id
            )
        reviews.append(
            compile_projection_relation_claim_review(
                candidate_set=candidate_set,
                draft=draft,
                projection_by_id=projection_by_id,
            )
        )

    direct_ids = {
        work_id
        for row in reviews
        for work_id in row.direct_prior_art_work_ids
    }
    lower_ids = {
        work_id
        for row in reviews
        for work_id in row.lower_order_prior_art_work_ids
    }
    counter_ids = {
        work_id
        for row in reviews
        for work_id in (
            row.directional_counterevidence_work_ids
            + row.contextual_conflict_work_ids
        )
    }
    conflicting_ids = {
        work_id
        for row in reviews
        for work_id in row.conflicting_prior_art_work_ids
    }

    return ProjectionRelationAdjudicationReport(
        report_id=_stable_id(
            "projection_relation_adjudication_report",
            candidate_report.report_id,
            backend_name,
            model_name,
            *[
                (
                    row.claim_id,
                    row.relation_state,
                    *[
                        match.work_id + ":" + match.relationship
                        for match in row.matches
                    ],
                )
                for row in reviews
            ],
        ),
        source_candidate_report_id=candidate_report.report_id,
        backend_name=backend_name,
        model_name=model_name,
        reviews=reviews,
        reviewed_claim_count=len(reviews),
        presented_work_count=sum(
            row.presented_work_count
            for row in reviews
        ),
        classified_work_count=sum(
            row.classified_work_count
            for row in reviews
        ),
        unclassified_work_count=sum(
            len(row.unclassified_work_ids)
            for row in reviews
        ),
        reviewed_work_count=sum(
            row.classified_work_count
            for row in reviews
        ),
        llm_calls_performed=llm_calls_performed,
        direct_signal_work_count=len(direct_ids),
        lower_order_signal_work_count=len(lower_ids),
        counterevidence_signal_work_count=len(counter_ids),
        conflicting_signal_work_count=len(conflicting_ids),
    )


def _subset_candidate_set_for_followup(
    candidate_set: ProjectionRelationClaimCandidateSet,
    work_ids: list[str],
) -> ProjectionRelationClaimCandidateSet:
    """Build a validated prompt-only subset for omitted-work follow-up.

    The original bounded candidate universe remains authoritative. This helper
    only narrows the next LLM prompt to work IDs that were presented previously
    but not classified by the model. No relationship is filled
    deterministically for an omitted work.
    """

    requested = set(work_ids)
    candidates = [
        row
        for row in candidate_set.candidates
        if row.review_work_id in requested
    ]
    if len(candidates) != len(requested):
        found = {row.review_work_id for row in candidates}
        missing = sorted(requested - found)
        raise ValueError(
            "follow-up work IDs are absent from original candidate set: "
            + ", ".join(missing)
        )

    body = candidate_set.model_dump(mode="python")
    body.update(
        {
            "candidates": candidates,
            "candidate_count": len(candidates),
            "abstract_candidate_count": sum(
                bool(row.abstract) for row in candidates
            ),
            "source_unique_work_count": len(
                {
                    source_work_id
                    for row in candidates
                    for source_work_id in row.source_work_ids
                }
            ),
        }
    )
    return ProjectionRelationClaimCandidateSet.model_validate(body)


def _review_claim_with_exhaustive_followup(
    *,
    candidate_set: ProjectionRelationClaimCandidateSet,
    projection_by_id: dict[str, GroundedFactorRelationProjection],
    backend: InstructorProjectionRelationAdjudicationBackend,
    max_exhaustive_rounds: int,
) -> tuple[ProjectionRelationClaimReviewDraft, int]:
    """Re-ask only omitted presented works, bounded and fail-closed.

    A model may return fewer match records than requested even though the
    prompt requires exhaustive classification. Rather than silently mapping
    omitted works to UNRELATED, subsequent rounds contain only the still
    unclassified IDs. If the bounded round budget is exhausted, those works
    remain omitted in the merged draft and therefore become explicit
    ``unclassified_work_ids`` during deterministic compilation.
    """

    if max_exhaustive_rounds < 1:
        raise ValueError("max_exhaustive_rounds must be >= 1")

    ordered_ids = [
        row.review_work_id for row in candidate_set.candidates
    ]
    allowed_ids = set(ordered_ids)
    remaining = list(ordered_ids)
    known_matches: dict[str, ProjectionRelationMatchDraft] = {}
    unknown_matches: dict[str, ProjectionRelationMatchDraft] = {}
    interpretations: list[str] = []
    calls = 0

    for _round in range(max_exhaustive_rounds):
        if not remaining:
            break

        prompt_set = _subset_candidate_set_for_followup(
            candidate_set,
            remaining,
        )
        draft = backend.review(
            candidate_set=prompt_set,
            projection_by_id=projection_by_id,
        )
        calls += 1
        if draft.interpretation.strip():
            interpretations.append(draft.interpretation.strip())

        round_allowed = set(remaining)
        for match in draft.matches:
            if match.work_id in round_allowed:
                known_matches.setdefault(match.work_id, match)
            elif match.work_id not in allowed_ids:
                # Preserve first-seen hallucinated IDs for deterministic audit
                # in compile_projection_relation_claim_review().
                unknown_matches.setdefault(match.work_id, match)
            # A previously classified allowed ID repeated in a later round is
            # ignored. The first bounded classification remains authoritative.

        remaining = [
            work_id
            for work_id in ordered_ids
            if work_id not in known_matches
        ]

    merged_matches = [
        known_matches[work_id]
        for work_id in ordered_ids
        if work_id in known_matches
    ]
    merged_matches.extend(
        unknown_matches[work_id]
        for work_id in sorted(unknown_matches)
    )

    interpretation = " | ".join(dict.fromkeys(interpretations))
    if not interpretation:
        interpretation = (
            "Bounded relation adjudication returned no interpretation."
        )

    return (
        ProjectionRelationClaimReviewDraft(
            matches=merged_matches,
            interpretation=interpretation,
        ),
        calls,
    )


def run_projection_relation_adjudication(
    *,
    candidate_report: ProjectionRelationCandidateReport,
    projection_report: GroundedFactorProjectionReport,
    backend: InstructorProjectionRelationAdjudicationBackend,
    max_exhaustive_rounds: int = 3,
) -> ProjectionRelationAdjudicationReport:
    if max_exhaustive_rounds < 1:
        raise ValueError("max_exhaustive_rounds must be >= 1")

    projection_by_id, _projection_by_claim = _projection_maps(
        projection_report
    )

    drafts: dict[str, ProjectionRelationClaimReviewDraft] = {}
    calls = 0

    for candidate_set in candidate_report.claims:
        if not candidate_set.candidates:
            if not candidate_set.projection_ids:
                interpretation = (
                    "The typed relation was not ready for grounded projection, "
                    "so bounded relation adjudication could not be performed."
                )
            else:
                interpretation = (
                    "No bounded review candidates were available."
                )
            drafts[candidate_set.claim_id] = (
                ProjectionRelationClaimReviewDraft(
                    matches=[],
                    interpretation=interpretation,
                )
            )
            continue

        draft, claim_calls = _review_claim_with_exhaustive_followup(
            candidate_set=candidate_set,
            projection_by_id=projection_by_id,
            backend=backend,
            max_exhaustive_rounds=max_exhaustive_rounds,
        )
        drafts[candidate_set.claim_id] = draft
        calls += claim_calls

    return build_projection_relation_adjudication_report(
        candidate_report=candidate_report,
        drafts_by_claim_id=drafts,
        projection_report=projection_report,
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        llm_calls_performed=calls,
    )


__all__ = [
    "InstructorProjectionRelationAdjudicationBackend",
    "ProjectionRelationAdjudicationReport",
    "ProjectionRelationCandidateReport",
    "ProjectionRelationClaimReviewDraft",
    "ProjectionRelationMatchDraft",
    "ProjectionRelationWorkCandidate",
    "build_projection_relation_adjudication_report",
    "build_projection_relation_candidate_report",
    "compile_projection_relation_claim_review",
    "run_projection_relation_adjudication",
]
