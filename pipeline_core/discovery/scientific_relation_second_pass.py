from __future__ import annotations

import hashlib
import json
import os
import re
from itertools import product
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
)
from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
    GroundedFactorRelationProjection,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureRetriever,
    canonicalize_prior_art_works,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIR,
    ScientificRelationIRReport,
)
from pipeline_core.domain.domain_profile import ScientificDomainProfile
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SecondPassQueryRole = Literal[
    "SEMANTIC_ENDPOINT_PAIR",
    "SEMANTIC_FULL_RELATION",
    "SEMANTIC_COUNTEREVIDENCE",
]


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


def _clean_text(value: object, *, limit: int = 160) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit].strip()


def _unique_text(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _clean_text(raw)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


_FORBIDDEN_QUERY_PATTERNS = (
    re.compile(r"https?://", re.I),
    re.compile(r"\bdoi\s*:", re.I),
    re.compile(r"\b10\.\d{4,9}/\S+", re.I),
)


def _safe_generated_phrase(value: object) -> str | None:
    text = _clean_text(value, limit=120)
    if not text:
        return None
    if any(pattern.search(text) for pattern in _FORBIDDEN_QUERY_PATTERNS):
        return None
    if len(text.split()) > 12:
        return None
    return text


def _base_doi(value: object) -> str | None:
    text = str(value or "").strip().casefold()
    if text.startswith("https://doi.org/"):
        text = text[len("https://doi.org/") :]
    if text.startswith("doi:"):
        text = text[4:]
    text = re.sub(r"\.s\d+$", "", text, flags=re.I)
    return text or None


def _is_supplementary_doi(value: object) -> bool:
    text = str(value or "").strip()
    return bool(re.search(r"\.s\d+$", text, re.I))


class EndpointSemanticExpansion(StrictModel):
    endpoint_concept_id: str
    source_text: str
    retrieval_aliases: list[str] = Field(default_factory=list)


class RelationSemanticExpansionDraft(StrictModel):
    endpoint_expansions: list[EndpointSemanticExpansion] = Field(
        default_factory=list
    )
    counterevidence_phrases: list[str] = Field(default_factory=list)
    notes: str = ""


class SemanticSecondPassQueryBinding(StrictModel):
    query_id: str
    relation_ir_id: str
    hypothesis_id: str
    claim_id: str
    role: SecondPassQueryRole
    basis_projection_ids: list[str] = Field(default_factory=list)

    query_text: str = Field(min_length=1)
    source_endpoint_terms: list[str] = Field(min_length=2)
    generated_endpoint_terms: list[str] = Field(default_factory=list)
    generated_counterevidence_terms: list[str] = Field(default_factory=list)
    domain_anchor_terms: list[str] = Field(default_factory=list)
    identity_anchor_terms: list[str] = Field(default_factory=list)

    semantic_expansion_is_retrieval_only: Literal[True] = True
    scientific_evidence_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    nonobviousness_authority: Literal[False] = False


class ScientificRelationSecondPassPlan(StrictModel):
    schema_version: Literal[
        "scientific-relation-second-pass-plan-v1"
    ] = "scientific-relation-second-pass-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relation_ir_report_id: str
    source_projection_report_id: str
    source_portfolio_id: str

    bindings: list[SemanticSecondPassQueryBinding]
    query_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    max_queries_per_claim: int = Field(ge=1)

    retrieval_semantics: Literal[
        "semantic_vocabulary_expansion_only"
    ] = "semantic_vocabulary_expansion_only"

    source_relation_mutated: Literal[False] = False
    relation_adjudication_performed: Literal[False] = False
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificRelationSecondPassPlan":
        if self.query_count != len(self.bindings):
            raise ValueError("second-pass query_count mismatch")
        if self.query_count > self.claim_count * self.max_queries_per_claim:
            raise ValueError("second-pass plan exceeded bounded query budget")

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("second-pass plan SHA mismatch")
        if observed_id != (
            "scientific_relation_second_pass_plan:"
            + expected_sha[:20]
        ):
            raise ValueError("second-pass plan ID mismatch")
        return self


class SecondPassResolutionTarget(StrictModel):
    source_work_id: str
    claim_id: str | None = None
    reason: Literal[
        "SUPPLEMENTARY_DOI_BASE_LOOKUP",
        "MISSING_ABSTRACT_DOI_LOOKUP",
        "MISSING_ABSTRACT_TITLE_LOOKUP",
    ]
    query_text: str = Field(min_length=1)


class ScientificRelationSecondPassReport(StrictModel):
    schema_version: Literal[
        "scientific-relation-second-pass-report-v1"
    ] = "scientific-relation-second-pass-report-v1"

    report_id: str
    source_second_pass_plan_id: str
    source_delta_packet_id: str
    source_resolution_packet_id: str | None = None
    source_resolved_packet_id: str

    semantic_query_count: int = Field(ge=0)
    resolution_query_count: int = Field(ge=0)
    raw_second_pass_work_count: int = Field(ge=0)
    resolved_unique_work_count: int = Field(ge=0)

    abstract_work_count_before_resolution: int = Field(ge=0)
    abstract_work_count_after_resolution: int = Field(ge=0)
    supplementary_doi_target_count: int = Field(ge=0)
    missing_abstract_target_count: int = Field(ge=0)

    diagnostic_only: Literal[True] = True
    relation_adjudication_performed: Literal[False] = False
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False


def _domain_anchor_terms(
    profile: ScientificDomainProfile,
) -> list[str]:
    # Labels are configured domain names, not model-generated evidence.
    # Keep only human-readable labels suitable for bibliographic search.
    values: list[str] = []
    for label, _patterns in profile.novelty.domain_patterns:
        cleaned = label.replace("_", " ").strip()
        if cleaned:
            values.append(cleaned)
    return _unique_text(values)[:2]


def _projection_rows(
    projection_report: GroundedFactorProjectionReport,
    claim_id: str,
) -> list[GroundedFactorRelationProjection]:
    for row in projection_report.projection_sets:
        if row.claim_id == claim_id:
            return list(row.projections)
    return []


def _projection_ids(
    projections: list[GroundedFactorRelationProjection],
    kind: str,
) -> list[str]:
    return [
        row.projection_id
        for row in projections
        if row.projection_kind == kind
    ]


def _full_identity_aliases(
    projections: list[GroundedFactorRelationProjection],
) -> list[str]:
    full = next(
        (
            row
            for row in projections
            if row.projection_kind == "FULL_RELATION"
        ),
        None,
    )
    if full is None:
        return []

    return _unique_text(
        [
            binding.exact_source_aliases[0]
            for binding in full.factor_bindings
            if binding.exact_source_aliases
        ]
    )


def _validated_aliases(
    relation: ScientificRelationIR,
    draft: RelationSemanticExpansionDraft,
) -> dict[str, list[str]]:
    endpoint_by_id = {
        row.concept_id: row
        for row in relation.endpoint_concepts
    }

    output: dict[str, list[str]] = {
        row.concept_id: []
        for row in relation.endpoint_concepts
    }

    for expansion in draft.endpoint_expansions:
        source = endpoint_by_id.get(expansion.endpoint_concept_id)
        if source is None:
            continue
        if _clean_text(expansion.source_text).casefold() != (
            _clean_text(source.surface_text).casefold()
        ):
            continue

        aliases = [
            safe
            for value in expansion.retrieval_aliases
            if (safe := _safe_generated_phrase(value))
        ]
        output[source.concept_id] = _unique_text(aliases)[:3]

    return output


def _endpoint_pair_variants(
    relation: ScientificRelationIR,
    aliases_by_id: dict[str, list[str]],
    *,
    max_pairs: int,
) -> list[tuple[list[str], list[str]]]:
    groups: list[list[tuple[str, bool]]] = []

    for endpoint in relation.endpoint_concepts:
        generated = aliases_by_id.get(endpoint.concept_id, [])
        values = [
            (alias, True)
            for alias in generated
        ]
        # Always retain the exact source endpoint as a fallback. This does not
        # turn the semantic alias into evidence; it only prevents an empty plan.
        values.append((endpoint.surface_text, False))

        unique: list[tuple[str, bool]] = []
        seen: set[str] = set()
        for text, generated_flag in values:
            key = _clean_text(text).casefold()
            if key and key not in seen:
                seen.add(key)
                unique.append((_clean_text(text), generated_flag))
        groups.append(unique[:4])

    rows: list[tuple[list[str], list[str]]] = []
    seen_queries: set[str] = set()

    for combination in product(*groups):
        terms = [text for text, _flag in combination]
        generated_terms = [
            text
            for text, flag in combination
            if flag
        ]
        key = " ".join(terms).casefold()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        rows.append((terms, generated_terms))
        if len(rows) >= max_pairs:
            break

    return rows


def _binding(
    *,
    relation: ScientificRelationIR,
    role: SecondPassQueryRole,
    basis_projection_ids: list[str],
    source_endpoint_terms: list[str],
    query_terms: list[str],
    generated_endpoint_terms: list[str],
    generated_counterevidence_terms: list[str],
    domain_anchor_terms: list[str],
    identity_anchor_terms: list[str],
) -> SemanticSecondPassQueryBinding:
    query_text = " ".join(
        _unique_text(query_terms)
    )
    return SemanticSecondPassQueryBinding(
        query_id=_stable_id(
            "scientific_relation_second_pass_query",
            relation.relation_ir_id,
            role,
            query_text,
        ),
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        role=role,
        basis_projection_ids=list(
            dict.fromkeys(basis_projection_ids)
        ),
        query_text=query_text,
        source_endpoint_terms=source_endpoint_terms,
        generated_endpoint_terms=generated_endpoint_terms,
        generated_counterevidence_terms=generated_counterevidence_terms,
        domain_anchor_terms=domain_anchor_terms,
        identity_anchor_terms=identity_anchor_terms,
    )


def compile_second_pass_bindings(
    *,
    relation: ScientificRelationIR,
    projection_report: GroundedFactorProjectionReport,
    draft: RelationSemanticExpansionDraft,
    domain_profile: ScientificDomainProfile,
    max_queries_per_claim: int = 8,
) -> list[SemanticSecondPassQueryBinding]:
    if max_queries_per_claim < 1:
        raise ValueError("max_queries_per_claim must be >= 1")
    if relation.typing_status != "READY":
        return []

    projections = _projection_rows(
        projection_report,
        relation.claim_id,
    )
    if not projections:
        return []

    aliases_by_id = _validated_aliases(
        relation,
        draft,
    )
    source_endpoints = [
        row.surface_text
        for row in relation.endpoint_concepts
    ]
    domain_anchors = _domain_anchor_terms(
        domain_profile
    )
    identity_aliases = _full_identity_aliases(
        projections
    )

    base_ids = _projection_ids(
        projections,
        "BASE_RELATION",
    )
    full_ids = _projection_ids(
        projections,
        "FULL_RELATION",
    )

    counter_terms = _unique_text(
        [
            safe
            for value in draft.counterevidence_phrases
            if (safe := _safe_generated_phrase(value))
        ]
    )[:3]

    # Reserve role capacity before endpoint-pair enumeration. The previous
    # planner filled the entire bounded budget with endpoint-pair combinations,
    # starving SEMANTIC_FULL_RELATION and SEMANTIC_COUNTEREVIDENCE whenever
    # both endpoints had several aliases.
    #
    # Priority:
    #   1) always preserve at least one endpoint-pair slot;
    #   2) reserve up to two counterevidence slots when such search vocabulary
    #      exists, because the first-pass retrieval already has supporting
    #      projections but no semantic counter-vocabulary;
    #   3) reserve one identity-conditioned FULL slot when budget remains.
    counter_budget = min(
        2,
        len(counter_terms),
        max(0, max_queries_per_claim - 1),
    )
    remaining_after_counter = (
        max_queries_per_claim
        - counter_budget
        - 1  # mandatory endpoint-pair slot
    )
    full_budget = (
        1
        if identity_aliases
        and remaining_after_counter >= 1
        else 0
    )
    endpoint_budget = (
        max_queries_per_claim
        - counter_budget
        - full_budget
    )

    rows: list[SemanticSecondPassQueryBinding] = []

    # Generate enough pair variants to supply endpoint, FULL, and counter roles
    # without letting the endpoint role consume all bounded capacity.
    pair_capacity = max(
        2,
        endpoint_budget + full_budget + counter_budget,
    )
    pairs = _endpoint_pair_variants(
        relation,
        aliases_by_id,
        max_pairs=pair_capacity,
    )

    endpoint_rows = 0
    for terms, generated_terms in pairs:
        if endpoint_rows >= endpoint_budget:
            break
        if not generated_terms:
            continue
        rows.append(
            _binding(
                relation=relation,
                role="SEMANTIC_ENDPOINT_PAIR",
                basis_projection_ids=base_ids,
                source_endpoint_terms=source_endpoints,
                query_terms=[
                    *terms,
                    *domain_anchors,
                ],
                generated_endpoint_terms=generated_terms,
                generated_counterevidence_terms=[],
                domain_anchor_terms=domain_anchors,
                identity_anchor_terms=[],
            )
        )
        endpoint_rows += 1

    # Use the first semantically expanded pair for identity-conditioned FULL
    # retrieval. This remains retrieval intent only.
    expanded_pairs = [
        (terms, generated_terms)
        for terms, generated_terms in pairs
        if generated_terms
    ]

    if (
        full_budget
        and identity_aliases
        and expanded_pairs
    ):
        terms, generated_terms = expanded_pairs[0]
        rows.append(
            _binding(
                relation=relation,
                role="SEMANTIC_FULL_RELATION",
                basis_projection_ids=full_ids,
                source_endpoint_terms=source_endpoints,
                query_terms=[
                    *terms,
                    *identity_aliases,
                    *domain_anchors,
                ],
                generated_endpoint_terms=generated_terms,
                generated_counterevidence_terms=[],
                domain_anchor_terms=domain_anchors,
                identity_anchor_terms=identity_aliases,
            )
        )

    # Counterevidence uses expanded endpoint terminology but deliberately omits
    # synthetic identity terms so bridge-breaking literature can be recovered.
    counter_pair_cursor = 0
    for counter_term in counter_terms[:counter_budget]:
        if expanded_pairs:
            (
                terms,
                generated_terms,
            ) = expanded_pairs[
                counter_pair_cursor
                % len(expanded_pairs)
            ]
            counter_pair_cursor += 1
        else:
            terms = source_endpoints
            generated_terms = []

        rows.append(
            _binding(
                relation=relation,
                role="SEMANTIC_COUNTEREVIDENCE",
                basis_projection_ids=base_ids,
                source_endpoint_terms=source_endpoints,
                query_terms=[
                    *terms,
                    counter_term,
                    *domain_anchors,
                ],
                generated_endpoint_terms=generated_terms,
                generated_counterevidence_terms=[counter_term],
                domain_anchor_terms=domain_anchors,
                identity_anchor_terms=[],
            )
        )

    # Literal query dedup; semantic equivalence is never inferred.
    deduped: list[SemanticSecondPassQueryBinding] = []
    seen: set[str] = set()
    for row in rows:
        key = (
            row.hypothesis_id
            + "|"
            + row.claim_id
            + "|"
            + row.query_text.casefold()
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped[:max_queries_per_claim]



def build_second_pass_plan(
    *,
    relation_ir_report: ScientificRelationIRReport,
    projection_report: GroundedFactorProjectionReport,
    drafts_by_claim_id: dict[str, RelationSemanticExpansionDraft],
    domain_profile: ScientificDomainProfile,
    source_portfolio_id: str,
    max_queries_per_claim: int = 8,
) -> ScientificRelationSecondPassPlan:
    bindings: list[SemanticSecondPassQueryBinding] = []

    for relation in relation_ir_report.relations:
        draft = drafts_by_claim_id.get(
            relation.claim_id,
            RelationSemanticExpansionDraft(),
        )
        bindings.extend(
            compile_second_pass_bindings(
                relation=relation,
                projection_report=projection_report,
                draft=draft,
                domain_profile=domain_profile,
                max_queries_per_claim=max_queries_per_claim,
            )
        )

    body = {
        "schema_version":
            "scientific-relation-second-pass-plan-v1",
        "source_relation_ir_report_id":
            relation_ir_report.report_id,
        "source_projection_report_id":
            projection_report.report_id,
        "source_portfolio_id":
            source_portfolio_id,
        "bindings": [
            row.model_dump(mode="json")
            for row in bindings
        ],
        "query_count":
            len(bindings),
        "claim_count":
            len(relation_ir_report.relations),
        "max_queries_per_claim":
            max_queries_per_claim,
        "retrieval_semantics":
            "semantic_vocabulary_expansion_only",
        "source_relation_mutated":
            False,
        "relation_adjudication_performed":
            False,
        "absence_based_novelty_authorized":
            False,
        "positive_nonobviousness_authority_created":
            False,
        "production_selection_changed":
            False,
    }
    digest = _sha256_json(body)
    return ScientificRelationSecondPassPlan(
        **body,
        plan_id=(
            "scientific_relation_second_pass_plan:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


def build_second_pass_transport_plan(
    plan: ScientificRelationSecondPassPlan,
) -> LiteratureQueryPlan:
    queries = [
        LiteratureQuery(
            query_id=row.query_id,
            hypothesis_id=row.hypothesis_id,
            claim_id=row.claim_id,
            query_kind=(
                "claim_diagnostic"
                if row.role == "SEMANTIC_COUNTEREVIDENCE"
                else "claim_variant"
            ),
            query_text=row.query_text,
        )
        for row in plan.bindings
    ]

    body = {
        "schema_version":
            "literature-query-plan-v1",
        "source_portfolio_id":
            plan.source_portfolio_id,
        "queries": [
            row.model_dump(mode="json")
            for row in queries
        ],
        "claims": [],
        "policy_version":
            "external-novelty-query-policy-v1",
    }
    digest = _sha256_json(body)
    return LiteratureQueryPlan(
        **body,
        plan_id=(
            "literature_query_plan:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


def build_resolution_targets(
    packet: PriorArtPacket,
    *,
    max_queries: int = 24,
) -> list[SecondPassResolutionTarget]:
    if max_queries < 0:
        raise ValueError("max_queries must be >= 0")

    rows: list[SecondPassResolutionTarget] = []
    seen: set[tuple[str, str]] = set()

    for work in packet.works:
        claim_id = (
            work.retrieval_claim_ids[0]
            if work.retrieval_claim_ids
            else None
        )

        doi = _base_doi(work.doi)
        supplementary = _is_supplementary_doi(work.doi)
        missing_abstract = not bool(
            str(work.abstract or "").strip()
        )

        if supplementary and doi:
            key = (work.work_id, "base-doi:" + doi)
            if key not in seen:
                seen.add(key)
                rows.append(
                    SecondPassResolutionTarget(
                        source_work_id=work.work_id,
                        claim_id=claim_id,
                        reason="SUPPLEMENTARY_DOI_BASE_LOOKUP",
                        query_text=doi,
                    )
                )

        if missing_abstract and doi:
            key = (work.work_id, "doi:" + doi)
            if key not in seen:
                seen.add(key)
                rows.append(
                    SecondPassResolutionTarget(
                        source_work_id=work.work_id,
                        claim_id=claim_id,
                        reason="MISSING_ABSTRACT_DOI_LOOKUP",
                        query_text=doi,
                    )
                )
        elif missing_abstract and work.title:
            title = _clean_text(
                work.title,
                limit=300,
            )
            key = (
                work.work_id,
                "title:" + title.casefold(),
            )
            if title and key not in seen:
                seen.add(key)
                rows.append(
                    SecondPassResolutionTarget(
                        source_work_id=work.work_id,
                        claim_id=claim_id,
                        reason="MISSING_ABSTRACT_TITLE_LOOKUP",
                        query_text=title,
                    )
                )

        if len(rows) >= max_queries:
            break

    return rows[:max_queries]


def build_resolution_transport_plan(
    *,
    source_portfolio_id: str,
    targets: list[SecondPassResolutionTarget],
) -> LiteratureQueryPlan:
    queries = [
        LiteratureQuery(
            query_id=_stable_id(
                "second_pass_resolution_query",
                target.source_work_id,
                target.reason,
                target.query_text,
            ),
            hypothesis_id="resolution:" + target.source_work_id,
            claim_id=target.claim_id,
            query_kind="claim_exact_verification",
            query_text=target.query_text,
        )
        for target in targets
    ]

    body = {
        "schema_version":
            "literature-query-plan-v1",
        "source_portfolio_id":
            source_portfolio_id,
        "queries": [
            row.model_dump(mode="json")
            for row in queries
        ],
        "claims": [],
        "policy_version":
            "external-novelty-query-policy-v1",
    }
    digest = _sha256_json(body)
    return LiteratureQueryPlan(
        **body,
        plan_id=(
            "literature_query_plan:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


def _empty_packet(
    plan: LiteratureQueryPlan,
) -> PriorArtPacket:
    body = {
        "schema_version": "prior-art-packet-v1",
        "packet_id": _stable_id(
            "prior_art_packet",
            plan.plan_id,
            "empty",
        ),
        "source_portfolio_id": plan.source_portfolio_id,
        "source_query_plan_id": plan.plan_id,
        "searched_at_utc": "1970-01-01T00:00:00+00:00",
        "providers_requested": [],
        "works": [],
        "executions": [],
        "raw_work_count": 0,
        "canonical_work_count": 0,
        "deduplicated_work_count": 0,
        "supplementary_records_collapsed": 0,
        "epistemic_usage": "prior_art_only_not_positive_premise",
    }
    return PriorArtPacket(
        **body,
        packet_sha256=_sha256_json(body),
    )


def _normalized_title(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _resolution_identity_match(
    *,
    source: PriorArtWork,
    candidate: PriorArtWork,
) -> bool:
    source_doi = _base_doi(source.doi)
    candidate_doi = _base_doi(candidate.doi)

    if source_doi and candidate_doi:
        return source_doi == candidate_doi

    source_title = _normalized_title(source.title)
    candidate_title = _normalized_title(candidate.title)

    return bool(
        len(source_title) >= 20
        and source_title == candidate_title
    )


def filter_resolution_packet_to_source_identities(
    *,
    semantic_packet: PriorArtPacket,
    resolution_packet: PriorArtPacket,
    resolution_plan: LiteratureQueryPlan,
    resolution_targets: list[SecondPassResolutionTarget],
) -> PriorArtPacket:
    if len(resolution_plan.queries) != len(resolution_targets):
        raise ValueError(
            "resolution plan/target length mismatch"
        )

    source_by_id = {
        work.work_id: work
        for work in semantic_packet.works
    }
    source_by_query_id: dict[str, PriorArtWork] = {}

    for query, target in zip(
        resolution_plan.queries,
        resolution_targets,
        strict=True,
    ):
        source = source_by_id.get(
            target.source_work_id
        )
        if source is not None:
            source_by_query_id[query.query_id] = source

    accepted: list[PriorArtWork] = []
    seen: set[str] = set()

    for candidate in resolution_packet.works:
        matched = False
        for query_id in candidate.retrieval_query_ids:
            source = source_by_query_id.get(query_id)
            if source is None:
                continue
            if _resolution_identity_match(
                source=source,
                candidate=candidate,
            ):
                matched = True
                break

        if matched and candidate.work_id not in seen:
            seen.add(candidate.work_id)
            accepted.append(candidate)

    body = {
        "schema_version":
            "prior-art-packet-v1",
        "packet_id": _stable_id(
            "prior_art_packet",
            resolution_packet.packet_id,
            "identity-filtered",
            *[
                work.work_id
                for work in accepted
            ],
        ),
        "source_portfolio_id":
            resolution_packet.source_portfolio_id,
        "source_query_plan_id":
            resolution_packet.source_query_plan_id,
        "searched_at_utc":
            resolution_packet.searched_at_utc,
        "providers_requested":
            list(resolution_packet.providers_requested),
        "works": [
            work.model_dump(mode="json")
            for work in accepted
        ],
        "executions": [
            row.model_dump(mode="json")
            for row in resolution_packet.executions
        ],
        "raw_work_count":
            resolution_packet.raw_work_count,
        "canonical_work_count":
            len(accepted),
        "deduplicated_work_count":
            max(
                0,
                resolution_packet.raw_work_count
                - len(accepted),
            ),
        "supplementary_records_collapsed":
            resolution_packet.supplementary_records_collapsed,
        "epistemic_usage":
            "prior_art_only_not_positive_premise",
    }

    return PriorArtPacket(
        **body,
        packet_sha256=_sha256_json(body),
    )


def merge_second_pass_resolution_packets(
    *,
    semantic_packet: PriorArtPacket,
    resolution_packet: PriorArtPacket,
    semantic_transport_plan: LiteratureQueryPlan,
    resolution_plan: LiteratureQueryPlan,
    resolution_targets: list[SecondPassResolutionTarget],
) -> PriorArtPacket:
    filtered_resolution_packet = (
        filter_resolution_packet_to_source_identities(
            semantic_packet=semantic_packet,
            resolution_packet=resolution_packet,
            resolution_plan=resolution_plan,
            resolution_targets=resolution_targets,
        )
    )

    merged_input = [
        *semantic_packet.works,
        *filtered_resolution_packet.works,
    ]
    canonical, supplementary_collapsed = (
        canonicalize_prior_art_works(
            merged_input
        )
    )
    works = sorted(
        canonical,
        key=lambda row: (
            -(row.citation_count or 0),
            -(row.year or 0),
            row.title.casefold(),
        ),
    )
    executions: list[QueryExecution] = [
        *semantic_packet.executions,
        *filtered_resolution_packet.executions,
    ]
    raw_count = (
        semantic_packet.raw_work_count
        + filtered_resolution_packet.raw_work_count
    )

    body = {
        "schema_version":
            "prior-art-packet-v1",
        "packet_id": _stable_id(
            "prior_art_packet",
            semantic_transport_plan.plan_id,
            semantic_packet.packet_id,
            resolution_packet.packet_id,
            "resolved-second-pass",
            *[
                row.work_id
                for row in works
            ],
        ),
        "source_portfolio_id":
            semantic_transport_plan.source_portfolio_id,
        # Bind the resolved packet to the semantic retrieval plan because the
        # resolution pass only enriches works already found by that plan.
        "source_query_plan_id":
            semantic_transport_plan.plan_id,
        "searched_at_utc":
            (
                filtered_resolution_packet.searched_at_utc
                if filtered_resolution_packet.works
                else semantic_packet.searched_at_utc
            ),
        "providers_requested": sorted(
            set(semantic_packet.providers_requested)
            | set(filtered_resolution_packet.providers_requested)
        ),
        "works": [
            row.model_dump(mode="json")
            for row in works
        ],
        "executions": [
            row.model_dump(mode="json")
            for row in executions
        ],
        "raw_work_count":
            raw_count,
        "canonical_work_count":
            len(works),
        "deduplicated_work_count":
            max(0, raw_count - len(works)),
        "supplementary_records_collapsed": (
            semantic_packet.supplementary_records_collapsed
            + filtered_resolution_packet.supplementary_records_collapsed
            + supplementary_collapsed
        ),
        "epistemic_usage":
            "prior_art_only_not_positive_premise",
    }

    return PriorArtPacket(
        **body,
        packet_sha256=_sha256_json(body),
    )


_SEMANTIC_EXPANSION_SYSTEM = """You generate bounded bibliographic SEARCH VOCABULARY for one typed scientific relation.

This is retrieval assistance only. You are NOT judging truth, novelty, prior art, non-obviousness, or evidence.

For each supplied endpoint:
- copy endpoint_concept_id and source_text exactly;
- provide up to 3 short retrieval aliases that scientists may use for the same endpoint concept;
- aliases may use scientifically standard synonyms, measurement terminology, or mechanism terminology;
- do not add a new experimental variable, material, causal claim, paper, author, DOI, citation, or result.

Also provide up to 3 short counterevidence search phrases that could retrieve literature where the two endpoints are decoupled, independent, non-correlated, differently heterogeneous, or otherwise fail to move together.
- These phrases are search vocabulary only.
- Do not claim that counterevidence exists.

Prefer terminology likely to occur in titles or abstracts.
Do not emit URLs or DOI-like strings.
Do not name specific papers or authors.
"""


def _semantic_expansion_prompt(
    relation: ScientificRelationIR,
    domain_profile: ScientificDomainProfile,
) -> str:
    lines = [
        "DOMAIN PROFILE",
        "==============",
        domain_profile.description,
        "",
        "TYPED RELATION",
        "==============",
        f"claim_id: {relation.claim_id}",
        f"claim_text: {relation.claim_text}",
        f"type_labels: {relation.relation_type_labels!r}",
        f"domain_labels: {relation.relation_domain_labels!r}",
        f"scope_features: {relation.relation_scope_features!r}",
        "",
        "ENDPOINTS",
        "=========",
    ]
    for endpoint in relation.endpoint_concepts:
        lines.extend(
            [
                f"endpoint_concept_id: {endpoint.concept_id}",
                f"source_text: {endpoint.surface_text}",
                f"type_labels: {endpoint.type_labels!r}",
                "",
            ]
        )

    if relation.identity_concepts:
        lines.extend(
            [
                "IDENTITY CONTEXT",
                "================",
                *[
                    row.surface_text
                    for row in relation.identity_concepts
                ],
                "",
            ]
        )

    lines.extend(
        [
            "Return retrieval aliases only.",
            "Do not make a literature-wide claim.",
        ]
    )
    return "\n".join(lines)


class InstructorSemanticExpansionBackend:
    backend_name = "instructor_scientific_relation_semantic_expansion"

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
                "Semantic second-pass expansion requires openai + instructor."
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

    def expand(
        self,
        *,
        relation: ScientificRelationIR,
        domain_profile: ScientificDomainProfile,
    ) -> RelationSemanticExpansionDraft:
        user = _semantic_expansion_prompt(
            relation,
            domain_profile,
        )
        if self.capture_prompts:
            self.prompt_records.append(
                {
                    "claim_id": relation.claim_id,
                    "system_prompt": _SEMANTIC_EXPANSION_SYSTEM,
                    "user_prompt": user,
                }
            )

        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=RelationSemanticExpansionDraft,
            messages=[
                {
                    "role": "system",
                    "content": _SEMANTIC_EXPANSION_SYSTEM,
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
                    "scientific_relation_second_pass_shadow",
                "stage":
                    "semantic_query_expansion",
                "call_kind":
                    "retrieval_query_expansion",
                "claim_id":
                    relation.claim_id,
            },
        )

        if not isinstance(
            result,
            RelationSemanticExpansionDraft,
        ):
            result = RelationSemanticExpansionDraft.model_validate(
                result
            )
        return result


def run_semantic_expansion(
    *,
    relation_ir_report: ScientificRelationIRReport,
    domain_profile: ScientificDomainProfile,
    backend: InstructorSemanticExpansionBackend,
) -> dict[str, RelationSemanticExpansionDraft]:
    drafts: dict[str, RelationSemanticExpansionDraft] = {}
    for relation in relation_ir_report.relations:
        if relation.typing_status != "READY":
            drafts[relation.claim_id] = RelationSemanticExpansionDraft()
            continue
        drafts[relation.claim_id] = backend.expand(
            relation=relation,
            domain_profile=domain_profile,
        )
    return drafts


__all__ = [
    "InstructorSemanticExpansionBackend",
    "RelationSemanticExpansionDraft",
    "ScientificRelationSecondPassPlan",
    "ScientificRelationSecondPassReport",
    "SemanticSecondPassQueryBinding",
    "SecondPassResolutionTarget",
    "build_resolution_targets",
    "build_resolution_transport_plan",
    "build_second_pass_plan",
    "build_second_pass_transport_plan",
    "compile_second_pass_bindings",
    "filter_resolution_packet_to_source_identities",
    "merge_second_pass_resolution_packets",
    "run_semantic_expansion",
]
