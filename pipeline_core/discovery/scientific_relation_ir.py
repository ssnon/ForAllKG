from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.domain.domain_profile import ScientificDomainProfile
from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RelationIRSourceContract = Literal[
    "atomic-cross-lane-scientific-synthesis-report-v1",
    "relational-atomic-projection-report-v1",
]


ConceptRole = Literal[
    "RELATION_ENDPOINT",
    "BRANCH_IDENTITY",
    "SCOPE_QUALIFIER",
    "DIRECTIONAL_QUALIFIER",
    "OBSERVABLE",
]

RelationTypingStatus = Literal[
    "READY",
    "PARTIAL",
    "AMBIGUOUS",
    "STRUCTURALLY_INVALID",
]

DocumentCompatibilityState = Literal[
    "COMPATIBLE",
    "UNCERTAIN",
    "AMBIGUOUS_RELATION_IDENTITY",
    "TYPE_IDENTITY_CONFLICT",
    "DOMAIN_OR_SCOPE_MISMATCH",
]


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _normalize(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−-]+", " ", text)
    text = re.sub(r"[^\w\s+*/().]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


_GENERIC_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "within",
        "under",
        "higher",
        "lower",
        "increase",
        "increases",
        "increased",
        "decrease",
        "decreases",
        "decreased",
        "greater",
        "less",
        "more",
    }
)


def lexical_content_tokens(value: object) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token
            for token in _normalize(value).split()
            if len(token) >= 2 and token not in _GENERIC_TOKENS
        )
    )


def _surface_contains(text: str, phrase: str) -> bool:
    needle = _normalize(phrase)
    haystack = _normalize(text)
    return bool(needle and needle in haystack)


class RelationTypingAdapter(Protocol):
    """Domain adapter for concept identity typing.

    Implementations may use exact lexical/domain rules, registries, or other
    deterministic authorities. They must not infer novelty, truth, or prior-art
    relationships.
    """

    adapter_id: str

    def type_labels(
        self,
        *,
        surface_text: str,
        relation_context: str,
    ) -> tuple[str, ...]: ...

    def ambiguity_labels(
        self,
        *,
        surface_text: str,
        relation_context: str,
        type_labels: tuple[str, ...],
    ) -> tuple[str, ...]: ...

    def conflicting_type_pairs(
        self,
        *,
        relation_types: tuple[str, ...],
        document_types: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]: ...


@dataclass(frozen=True)
class NullRelationTypingAdapter:
    adapter_id: str = "null_relation_typing_v1"

    def type_labels(
        self,
        *,
        surface_text: str,
        relation_context: str,
    ) -> tuple[str, ...]:
        del surface_text, relation_context
        return ()

    def ambiguity_labels(
        self,
        *,
        surface_text: str,
        relation_context: str,
        type_labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        del surface_text, relation_context, type_labels
        return ()

    def conflicting_type_pairs(
        self,
        *,
        relation_types: tuple[str, ...],
        document_types: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        del relation_types, document_types
        return ()


class ScientificConceptIR(StrictModel):
    concept_id: str
    role: ConceptRole
    source_field: str
    source_index: int = Field(ge=0)

    surface_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    lexical_tokens: list[str] = Field(default_factory=list)

    type_labels: list[str] = Field(default_factory=list)
    ambiguity_labels: list[str] = Field(default_factory=list)

    literal_in_claim_text: bool
    literal_in_required_bridge: bool | None = None

    semantic_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class ScientificRelationIR(StrictModel):
    schema_version: Literal[
        "scientific-relation-ir-v1"
    ] = "scientific-relation-ir-v1"

    relation_ir_id: str
    hypothesis_id: str
    claim_id: str
    claim_kind: str
    novelty_selection_role: str
    claim_text: str

    endpoint_concepts: list[ScientificConceptIR] = Field(min_length=2)
    identity_concepts: list[ScientificConceptIR] = Field(default_factory=list)
    scope_qualifiers: list[ScientificConceptIR] = Field(default_factory=list)
    directional_qualifiers: list[ScientificConceptIR] = Field(
        default_factory=list
    )
    observable_concept: ScientificConceptIR

    relation_type_labels: list[str] = Field(default_factory=list)
    relation_domain_labels: list[str] = Field(default_factory=list)
    relation_scope_features: list[str] = Field(default_factory=list)

    typing_status: RelationTypingStatus
    reason_codes: list[str] = Field(default_factory=list)

    source_contract: RelationIRSourceContract = (
        "atomic-cross-lane-scientific-synthesis-report-v1"
    )
    source_exact_spans_only: Literal[True] = True

    scientific_truth_authority: Literal[False] = False
    prior_art_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> "ScientificRelationIR":
        endpoint_ids = [row.concept_id for row in self.endpoint_concepts]
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("duplicate endpoint concept IDs")
        if self.typing_status == "READY" and self.reason_codes:
            if any(
                code.startswith("ambiguous_concept:")
                or code.startswith("missing_literal:")
                for code in self.reason_codes
            ):
                raise ValueError(
                    "READY relation cannot carry ambiguity/literal failures"
                )
        return self


class ScientificRelationIRReport(StrictModel):
    schema_version: Literal[
        "scientific-relation-ir-report-v1"
    ] = "scientific-relation-ir-report-v1"

    report_id: str
    source_atomic_report_id: str
    source_contract: RelationIRSourceContract = (
        "atomic-cross-lane-scientific-synthesis-report-v1"
    )
    domain_profile_id: str
    typing_adapter_id: str
    relations: list[ScientificRelationIR]
    relation_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    ambiguous_count: int = Field(ge=0)
    structurally_invalid_count: int = Field(ge=0)

    diagnostic_only: Literal[True] = True
    query_plan_changed: Literal[False] = False
    prior_art_review_changed: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificRelationIRReport":
        if self.relation_count != len(self.relations):
            raise ValueError("relation_count mismatch")
        expected = {
            "READY": self.ready_count,
            "PARTIAL": self.partial_count,
            "AMBIGUOUS": self.ambiguous_count,
            "STRUCTURALLY_INVALID": self.structurally_invalid_count,
        }
        for state, count in expected.items():
            actual = sum(
                row.typing_status == state
                for row in self.relations
            )
            if actual != count:
                raise ValueError(
                    f"typing count mismatch for {state}: {count} != {actual}"
                )
        return self


class RelationDocumentCompatibility(StrictModel):
    schema_version: Literal[
        "relation-document-compatibility-v1"
    ] = "relation-document-compatibility-v1"

    relation_ir_id: str
    state: DocumentCompatibilityState

    relation_type_labels: list[str] = Field(default_factory=list)
    document_type_labels: list[str] = Field(default_factory=list)
    document_ambiguity_labels: list[str] = Field(default_factory=list)
    conflicting_type_pairs: list[list[str]] = Field(default_factory=list)

    domain_relevance: float = Field(ge=0.0, le=1.0)
    scope_relevance: float = Field(ge=0.0, le=1.0)
    scope_reason_codes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    strong_prior_art_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


def _concept(
    *,
    spec: CompiledAtomicSpecification,
    hypothesis_id: str,
    role: ConceptRole,
    source_field: str,
    source_index: int,
    surface_text: str,
    relation_context: str,
    adapter: RelationTypingAdapter,
) -> ScientificConceptIR:
    normalized = _normalize(surface_text)
    if not normalized:
        raise ValueError(
            f"empty scientific relation concept: {source_field}[{source_index}]"
        )

    type_labels = tuple(
        dict.fromkeys(
            adapter.type_labels(
                surface_text=surface_text,
                relation_context=relation_context,
            )
        )
    )
    ambiguity_labels = tuple(
        dict.fromkeys(
            adapter.ambiguity_labels(
                surface_text=surface_text,
                relation_context=relation_context,
                type_labels=type_labels,
            )
        )
    )

    bridge_required = role in {
        "RELATION_ENDPOINT",
        "BRANCH_IDENTITY",
        "SCOPE_QUALIFIER",
        "DIRECTIONAL_QUALIFIER",
    }

    return ScientificConceptIR(
        concept_id=_stable_id(
            "scientific_concept",
            hypothesis_id,
            spec.claim_id,
            role,
            source_field,
            source_index,
            normalized,
        ),
        role=role,
        source_field=source_field,
        source_index=source_index,
        surface_text=surface_text,
        normalized_text=normalized,
        lexical_tokens=list(lexical_content_tokens(surface_text)),
        type_labels=list(type_labels),
        ambiguity_labels=list(ambiguity_labels),
        literal_in_claim_text=_surface_contains(
            spec.text,
            surface_text,
        ),
        literal_in_required_bridge=(
            _surface_contains(
                spec.required_bridge,
                surface_text,
            )
            if bridge_required
            else None
        ),
    )


def compile_atomic_specification_relation_ir(
    *,
    hypothesis_id: str,
    spec: CompiledAtomicSpecification,
    domain_profile: ScientificDomainProfile,
    typing_adapter: RelationTypingAdapter,
    source_contract: RelationIRSourceContract = (
        "atomic-cross-lane-scientific-synthesis-report-v1"
    ),
) -> ScientificRelationIR:
    relation_context = " ".join(
        value
        for value in (
            spec.text,
            spec.required_bridge,
            spec.observable,
            spec.predicted_observation,
            spec.falsification_condition,
        )
        if str(value).strip()
    )

    endpoints = [
        _concept(
            spec=spec,
            hypothesis_id=hypothesis_id,
            role="RELATION_ENDPOINT",
            source_field="relation_endpoint_anchors",
            source_index=index,
            surface_text=value,
            relation_context=relation_context,
            adapter=typing_adapter,
        )
        for index, value in enumerate(spec.relation_endpoint_anchors)
    ]
    identities = [
        _concept(
            spec=spec,
            hypothesis_id=hypothesis_id,
            role="BRANCH_IDENTITY",
            source_field="prior_art_identity_terms",
            source_index=index,
            surface_text=value,
            relation_context=relation_context,
            adapter=typing_adapter,
        )
        for index, value in enumerate(spec.prior_art_identity_terms)
    ]
    scopes = [
        _concept(
            spec=spec,
            hypothesis_id=hypothesis_id,
            role="SCOPE_QUALIFIER",
            source_field="scope_qualifier_spans",
            source_index=index,
            surface_text=value,
            relation_context=relation_context,
            adapter=typing_adapter,
        )
        for index, value in enumerate(spec.scope_qualifier_spans)
    ]
    directions = [
        _concept(
            spec=spec,
            hypothesis_id=hypothesis_id,
            role="DIRECTIONAL_QUALIFIER",
            source_field="directional_qualifier_spans",
            source_index=index,
            surface_text=value,
            relation_context=relation_context,
            adapter=typing_adapter,
        )
        for index, value in enumerate(spec.directional_qualifier_spans)
    ]
    observable = _concept(
        spec=spec,
        hypothesis_id=hypothesis_id,
        role="OBSERVABLE",
        source_field="observable",
        source_index=0,
        surface_text=spec.observable,
        relation_context=relation_context,
        adapter=typing_adapter,
    )

    all_concepts = [
        *endpoints,
        *identities,
        *scopes,
        *directions,
        observable,
    ]

    relation_types = tuple(
        dict.fromkeys(
            label
            for concept in all_concepts
            for label in concept.type_labels
        )
    )
    ambiguity = [
        (concept, label)
        for concept in all_concepts
        for label in concept.ambiguity_labels
    ]

    reasons: list[str] = []

    if len(endpoints) < 2:
        reasons.append("relation_requires_at_least_two_endpoints")

    for concept in [
        *endpoints,
        *identities,
        *scopes,
        *directions,
    ]:
        if not concept.literal_in_claim_text:
            reasons.append(
                "missing_literal:claim_text:"
                + concept.source_field
                + ":"
                + str(concept.source_index)
            )
        if concept.literal_in_required_bridge is not True:
            reasons.append(
                "missing_literal:required_bridge:"
                + concept.source_field
                + ":"
                + str(concept.source_index)
            )

    for concept, label in ambiguity:
        reasons.append(
            "ambiguous_concept:"
            + concept.role
            + ":"
            + label
        )

    endpoint_token_sets = [
        set(row.lexical_tokens)
        for row in endpoints
        if row.lexical_tokens
    ]
    if len(endpoint_token_sets) >= 2:
        intersection = set.intersection(*endpoint_token_sets)
        if intersection and all(
            tokens == intersection
            for tokens in endpoint_token_sets
        ):
            reasons.append(
                "endpoint_identity_not_distinct"
            )

    structural_invalid = any(
        code.startswith("missing_literal:")
        or code == "relation_requires_at_least_two_endpoints"
        for code in reasons
    )

    if structural_invalid:
        status: RelationTypingStatus = "STRUCTURALLY_INVALID"
    elif ambiguity:
        status = "AMBIGUOUS"
    elif relation_types:
        status = "READY"
    else:
        # No explicit type rule fired. The relation is structurally usable
        # but remains only partially typed; this must not be treated as a
        # negative scientific judgment.
        status = "PARTIAL"
        reasons.append("no_explicit_concept_type_rule_matched")

    return ScientificRelationIR(
        relation_ir_id=_stable_id(
            "scientific_relation_ir",
            hypothesis_id,
            spec.claim_id,
            spec.text,
            *spec.relation_endpoint_anchors,
            *spec.prior_art_identity_terms,
        ),
        hypothesis_id=hypothesis_id,
        claim_id=spec.claim_id,
        claim_kind=spec.kind,
        novelty_selection_role=spec.novelty_selection_role,
        claim_text=spec.text,
        endpoint_concepts=endpoints,
        identity_concepts=identities,
        scope_qualifiers=scopes,
        directional_qualifiers=directions,
        observable_concept=observable,
        relation_type_labels=list(relation_types),
        relation_domain_labels=sorted(
            domain_profile.novelty.domains(relation_context)
        ),
        relation_scope_features=sorted(
            domain_profile.novelty.scope_features(relation_context)
        ),
        typing_status=status,
        reason_codes=list(dict.fromkeys(reasons)),
        source_contract=source_contract,
    )


def compile_atomic_specifications_relation_ir_report(
    *,
    source_atomic_report_id: str,
    source_contract: RelationIRSourceContract,
    specifications: list[tuple[str, CompiledAtomicSpecification]],
    domain_profile: ScientificDomainProfile,
    typing_adapter: RelationTypingAdapter | None = None,
) -> ScientificRelationIRReport:
    adapter: RelationTypingAdapter = (
        typing_adapter
        if typing_adapter is not None
        else NullRelationTypingAdapter()
    )

    relations = [
        compile_atomic_specification_relation_ir(
            hypothesis_id=hypothesis_id,
            spec=spec,
            domain_profile=domain_profile,
            typing_adapter=adapter,
            source_contract=source_contract,
        )
        for hypothesis_id, spec in specifications
    ]

    return ScientificRelationIRReport(
        report_id=_stable_id(
            "scientific_relation_ir_report",
            source_atomic_report_id,
            domain_profile.profile_id,
            adapter.adapter_id,
            *[row.relation_ir_id for row in relations],
        ),
        source_atomic_report_id=source_atomic_report_id,
        source_contract=source_contract,
        domain_profile_id=domain_profile.profile_id,
        typing_adapter_id=adapter.adapter_id,
        relations=relations,
        relation_count=len(relations),
        ready_count=sum(row.typing_status == "READY" for row in relations),
        partial_count=sum(row.typing_status == "PARTIAL" for row in relations),
        ambiguous_count=sum(
            row.typing_status == "AMBIGUOUS"
            for row in relations
        ),
        structurally_invalid_count=sum(
            row.typing_status == "STRUCTURALLY_INVALID"
            for row in relations
        ),
    )


def compile_atomic_report_relation_ir(
    *,
    report: object,
    domain_profile: ScientificDomainProfile,
    typing_adapter: RelationTypingAdapter | None = None,
) -> ScientificRelationIRReport:
    """Compatibility shim for the former cross-lane-specific core entrypoint."""

    from pipeline_core.discovery.reframing.atomic_cross_lane_relation_ir_adapter import (
        compile_atomic_report_relation_ir as compile_cross_lane_report,
    )

    return compile_cross_lane_report(
        report=report,
        domain_profile=domain_profile,
        typing_adapter=typing_adapter,
    )


def assess_relation_document_compatibility(
    *,
    relation: ScientificRelationIR,
    document_text: str,
    domain_profile: ScientificDomainProfile,
    typing_adapter: RelationTypingAdapter,
    min_domain: float = 0.75,
    min_scope: float = 0.75,
) -> RelationDocumentCompatibility:
    """Diagnostic typed-identity compatibility for one prior-art document.

    This function does not change production prior-art review. It is intended
    to expose semantic collisions before such a gate is promoted.
    """

    document_types = tuple(
        dict.fromkeys(
            typing_adapter.type_labels(
                surface_text=document_text,
                relation_context=document_text,
            )
        )
    )
    document_ambiguity = tuple(
        dict.fromkeys(
            typing_adapter.ambiguity_labels(
                surface_text=document_text,
                relation_context=document_text,
                type_labels=document_types,
            )
        )
    )
    relation_types = tuple(relation.relation_type_labels)

    conflicts = typing_adapter.conflicting_type_pairs(
        relation_types=relation_types,
        document_types=document_types,
    )

    compatible, domain, scope, scope_reasons = (
        domain_profile.novelty.strong_scope_compatibility(
            relation.claim_text,
            document_text,
            min_domain=min_domain,
            min_scope=min_scope,
        )
    )

    reasons: list[str] = []

    relation_ambiguity = [
        label
        for concept in (
            [
                *relation.endpoint_concepts,
                *relation.identity_concepts,
                relation.observable_concept,
            ]
        )
        for label in concept.ambiguity_labels
    ]

    if relation_ambiguity:
        state: DocumentCompatibilityState = (
            "AMBIGUOUS_RELATION_IDENTITY"
        )
        reasons.append("relation_concept_identity_ambiguous")
    elif conflicts:
        state = "TYPE_IDENTITY_CONFLICT"
        reasons.append("typed_concept_identity_conflict")
    elif not compatible:
        state = "DOMAIN_OR_SCOPE_MISMATCH"
        reasons.extend(scope_reasons)
    elif document_ambiguity:
        state = "UNCERTAIN"
        reasons.append("document_concept_identity_ambiguous")
    elif relation_types and document_types:
        state = "COMPATIBLE"
    else:
        state = "UNCERTAIN"
        reasons.append("insufficient_explicit_type_identity")

    return RelationDocumentCompatibility(
        relation_ir_id=relation.relation_ir_id,
        state=state,
        relation_type_labels=list(relation_types),
        document_type_labels=list(document_types),
        document_ambiguity_labels=list(document_ambiguity),
        conflicting_type_pairs=[
            [left, right]
            for left, right in conflicts
        ],
        domain_relevance=domain,
        scope_relevance=scope,
        scope_reason_codes=scope_reasons,
        reason_codes=list(dict.fromkeys(reasons)),
    )


__all__ = [
    "NullRelationTypingAdapter",
    "RelationDocumentCompatibility",
    "RelationIRSourceContract",
    "RelationTypingAdapter",
    "ScientificConceptIR",
    "ScientificRelationIR",
    "ScientificRelationIRReport",
    "assess_relation_document_compatibility",
    "compile_atomic_report_relation_ir",
    "compile_atomic_specification_relation_ir",
    "compile_atomic_specifications_relation_ir_report",
    "lexical_content_tokens",
]
