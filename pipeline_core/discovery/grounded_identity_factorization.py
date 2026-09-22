from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.novelty_closure_review import (
    _identity_content_tokens,
)
from pipeline_core.discovery.reframing.grounded_identity_constituent_shadow import (
    GroundedIdentityAnnotation,
    GroundedIdentityAnnotationReport,
    GroundedIdentityConstituentGroup,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificConceptIR,
    ScientificRelationIR,
    ScientificRelationIRReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FactorizationStatus = Literal[
    "READY",
    "RELATION_NOT_READY",
    "ANNOTATION_MISSING",
    "IDENTITY_BINDING_MISMATCH",
    "INSUFFICIENT_DISTINCT_FACTORS",
]


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _identity_tokens(value: object) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _identity_content_tokens(
                str(value or "")
            )
        )
    )


class GroundedIdentityFactorSpan(StrictModel):
    candidate_id: str
    candidate_ref: str
    exact_source_text: str
    matched_source_paths: list[str] = Field(min_length=1)

    source_exact_span: Literal[True] = True


class GroundedIdentityFactor(StrictModel):
    factor_id: str
    relation_ir_id: str
    claim_id: str
    source_identity_concept_id: str
    source_identity_term: str

    group_label: str
    identity_basis_tokens: list[str] = Field(min_length=1)
    exclusive_identity_basis_tokens: list[str] = Field(min_length=1)

    grounded_spans: list[GroundedIdentityFactorSpan] = Field(min_length=1)
    query_aliases: list[str] = Field(min_length=1)

    factor_semantics_derived_from_identity_tokens_only: Literal[True] = True
    query_aliases_are_exact_source_spans: Literal[True] = True
    scientific_equivalence_inferred: Literal[False] = False
    novelty_authority: Literal[False] = False


class RelationIdentityFactorization(StrictModel):
    relation_ir_id: str
    hypothesis_id: str
    claim_id: str

    factorization_status: FactorizationStatus
    source_identity_concept_id: str | None = None
    source_identity_term: str | None = None

    source_identity_tokens: list[str] = Field(default_factory=list)
    grounded_factors: list[GroundedIdentityFactor] = Field(default_factory=list)
    preserved_identity_concept_ids: list[str] = Field(default_factory=list)

    grounded_factor_count: int = Field(ge=0)
    effective_identity_factor_count: int = Field(ge=0)
    complete_identity_token_coverage: bool = False
    distinct_factor_basis_validated: bool = False

    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "RelationIdentityFactorization":
        if self.grounded_factor_count != len(self.grounded_factors):
            raise ValueError("grounded_factor_count mismatch")
        expected_effective = (
            len(self.grounded_factors)
            + len(self.preserved_identity_concept_ids)
        )
        if self.effective_identity_factor_count != expected_effective:
            raise ValueError("effective_identity_factor_count mismatch")

        if self.factorization_status == "READY":
            if self.grounded_factor_count < 2:
                raise ValueError(
                    "READY factorization requires at least two grounded factors"
                )
            if not self.complete_identity_token_coverage:
                raise ValueError(
                    "READY factorization requires complete identity token coverage"
                )
            if not self.distinct_factor_basis_validated:
                raise ValueError(
                    "READY factorization requires distinct factor basis"
                )
        return self


class GroundedIdentityFactorizationReport(StrictModel):
    schema_version: Literal[
        "grounded-identity-factorization-report-v1"
    ] = "grounded-identity-factorization-report-v1"

    report_id: str
    source_relation_ir_report_id: str
    source_annotation_candidate_portfolio_id: str
    source_annotation_atomic_report_id: str

    relations: list[RelationIdentityFactorization]
    relation_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    not_ready_count: int = Field(ge=0)
    total_grounded_factor_count: int = Field(ge=0)

    source_grounded_exact_span_required: Literal[True] = True
    identity_factor_semantics_are_lexical_not_inferred: Literal[True] = True

    diagnostic_only: Literal[True] = True
    relation_ir_mutated: Literal[False] = False
    query_plan_changed: Literal[False] = False
    retrieval_changed: Literal[False] = False
    prior_art_review_changed: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "GroundedIdentityFactorizationReport":
        if self.relation_count != len(self.relations):
            raise ValueError("relation_count mismatch")
        if self.ready_count != sum(
            row.factorization_status == "READY"
            for row in self.relations
        ):
            raise ValueError("ready_count mismatch")
        if self.not_ready_count != (
            self.relation_count - self.ready_count
        ):
            raise ValueError("not_ready_count mismatch")
        if self.total_grounded_factor_count != sum(
            row.grounded_factor_count
            for row in self.relations
        ):
            raise ValueError("total_grounded_factor_count mismatch")
        return self


def _matching_identity_concept(
    relation: ScientificRelationIR,
    annotation: GroundedIdentityAnnotation,
) -> ScientificConceptIR | None:
    expected = _normalized(annotation.identity_term)
    rows = [
        concept
        for concept in relation.identity_concepts
        if _normalized(concept.surface_text) == expected
    ]
    if len(rows) != 1:
        return None
    return rows[0]


def _group_basis_tokens(
    *,
    group: GroundedIdentityConstituentGroup,
    identity_token_set: set[str],
) -> set[str]:
    """Return identity-token basis supported by all exact-source alternatives.

    Each exact source span may contain additional scientific words. Those words
    are retained as retrieval aliases but do NOT become factor semantics.

    For multiple aliases in one group, every alias must share at least one
    common identity token. This prevents a model-created group from silently
    joining unrelated constituents under one label.
    """

    overlaps: list[set[str]] = []
    for span in group.spans:
        span_tokens = set(
            _identity_tokens(
                span.exact_source_text
            )
        )
        overlap = span_tokens & identity_token_set
        if not overlap:
            return set()
        overlaps.append(overlap)

    if not overlaps:
        return set()

    common = set.intersection(*overlaps)
    if not common:
        return set()

    # The union captures the identity content represented by alternative exact
    # spans, while `common` guarantees they still refer to a shared constituent.
    return set.union(*overlaps)


def _factorize_ready_binding(
    *,
    relation: ScientificRelationIR,
    annotation: GroundedIdentityAnnotation,
    source_identity: ScientificConceptIR,
) -> tuple[
    list[GroundedIdentityFactor],
    bool,
    bool,
    list[str],
]:
    identity_tokens = list(
        _identity_tokens(
            source_identity.surface_text
        )
    )
    identity_token_set = set(identity_tokens)
    reasons: list[str] = []

    if len(identity_token_set) < 2:
        return (
            [],
            False,
            False,
            ["identity_has_fewer_than_two_content_tokens"],
        )

    group_basis: list[set[str]] = []
    for group in annotation.groups:
        basis = _group_basis_tokens(
            group=group,
            identity_token_set=identity_token_set,
        )
        if not basis:
            reasons.append(
                "grounded_group_lacks_shared_identity_basis:"
                + group.label
            )
        group_basis.append(basis)

    if reasons:
        return [], False, False, reasons

    union_basis = set().union(*group_basis)
    complete = union_basis == identity_token_set
    if not complete:
        missing = sorted(identity_token_set - union_basis)
        reasons.append(
            "identity_token_coverage_incomplete:"
            + ",".join(missing)
        )

    # Every factor must contribute at least one identity token not carried by
    # any other factor. Otherwise it is not a distinct projection dimension.
    exclusive_by_index: list[set[str]] = []
    distinct = True

    for index, basis in enumerate(group_basis):
        others = set().union(
            *[
                row
                for other_index, row in enumerate(group_basis)
                if other_index != index
            ]
        )
        exclusive = basis - others
        exclusive_by_index.append(exclusive)
        if not exclusive:
            distinct = False
            reasons.append(
                "grounded_group_has_no_exclusive_identity_basis:"
                + annotation.groups[index].label
            )

    if len(group_basis) < 2:
        distinct = False
        reasons.append(
            "fewer_than_two_grounded_identity_groups"
        )

    if not complete or not distinct:
        return [], complete, distinct, reasons

    factors: list[GroundedIdentityFactor] = []

    for index, (group, basis, exclusive) in enumerate(
        zip(
            annotation.groups,
            group_basis,
            exclusive_by_index,
            strict=True,
        )
    ):
        grounded_spans = [
            GroundedIdentityFactorSpan(
                candidate_id=span.candidate_id,
                candidate_ref=span.candidate_ref,
                exact_source_text=span.exact_source_text,
                matched_source_paths=list(
                    span.matched_source_paths
                ),
            )
            for span in group.spans
        ]
        aliases = list(
            dict.fromkeys(
                span.exact_source_text
                for span in group.spans
            )
        )

        ordered_basis = [
            token
            for token in identity_tokens
            if token in basis
        ]
        ordered_exclusive = [
            token
            for token in identity_tokens
            if token in exclusive
        ]

        factors.append(
            GroundedIdentityFactor(
                factor_id=_stable_id(
                    "grounded_identity_factor",
                    relation.relation_ir_id,
                    source_identity.concept_id,
                    index,
                    group.label,
                    *ordered_basis,
                    *aliases,
                ),
                relation_ir_id=relation.relation_ir_id,
                claim_id=relation.claim_id,
                source_identity_concept_id=source_identity.concept_id,
                source_identity_term=source_identity.surface_text,
                group_label=group.label,
                identity_basis_tokens=ordered_basis,
                exclusive_identity_basis_tokens=ordered_exclusive,
                grounded_spans=grounded_spans,
                query_aliases=aliases,
            )
        )

    return factors, complete, distinct, reasons


def factorize_relation_identity(
    *,
    relation: ScientificRelationIR,
    annotation: GroundedIdentityAnnotation | None,
) -> RelationIdentityFactorization:
    if relation.typing_status != "READY":
        return RelationIdentityFactorization(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            factorization_status="RELATION_NOT_READY",
            grounded_factor_count=0,
            effective_identity_factor_count=len(
                relation.identity_concepts
            ),
            preserved_identity_concept_ids=[
                row.concept_id
                for row in relation.identity_concepts
            ],
            reason_codes=[
                "relation_typing_status:"
                + relation.typing_status
            ],
        )

    if annotation is None:
        return RelationIdentityFactorization(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            factorization_status="ANNOTATION_MISSING",
            grounded_factor_count=0,
            effective_identity_factor_count=len(
                relation.identity_concepts
            ),
            preserved_identity_concept_ids=[
                row.concept_id
                for row in relation.identity_concepts
            ],
            reason_codes=["no_grounded_identity_annotation"],
        )

    source_identity = _matching_identity_concept(
        relation,
        annotation,
    )
    if source_identity is None:
        return RelationIdentityFactorization(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            factorization_status="IDENTITY_BINDING_MISMATCH",
            source_identity_term=annotation.identity_term,
            grounded_factor_count=0,
            effective_identity_factor_count=len(
                relation.identity_concepts
            ),
            preserved_identity_concept_ids=[
                row.concept_id
                for row in relation.identity_concepts
            ],
            reason_codes=[
                "annotation_identity_does_not_bind_exactly_one_relation_identity"
            ],
        )

    (
        factors,
        complete,
        distinct,
        reasons,
    ) = _factorize_ready_binding(
        relation=relation,
        annotation=annotation,
        source_identity=source_identity,
    )

    preserved = [
        row.concept_id
        for row in relation.identity_concepts
        if row.concept_id != source_identity.concept_id
    ]

    if len(factors) < 2:
        return RelationIdentityFactorization(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            factorization_status="INSUFFICIENT_DISTINCT_FACTORS",
            source_identity_concept_id=source_identity.concept_id,
            source_identity_term=source_identity.surface_text,
            source_identity_tokens=list(
                _identity_tokens(
                    source_identity.surface_text
                )
            ),
            grounded_factor_count=0,
            effective_identity_factor_count=len(
                relation.identity_concepts
            ),
            preserved_identity_concept_ids=[
                row.concept_id
                for row in relation.identity_concepts
            ],
            complete_identity_token_coverage=complete,
            distinct_factor_basis_validated=distinct,
            reason_codes=list(
                dict.fromkeys(
                    [
                        *reasons,
                        "grounded_identity_not_promoted_to_projection_factors",
                    ]
                )
            ),
        )

    return RelationIdentityFactorization(
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        factorization_status="READY",
        source_identity_concept_id=source_identity.concept_id,
        source_identity_term=source_identity.surface_text,
        source_identity_tokens=list(
            _identity_tokens(
                source_identity.surface_text
            )
        ),
        grounded_factors=factors,
        preserved_identity_concept_ids=preserved,
        grounded_factor_count=len(factors),
        effective_identity_factor_count=(
            len(factors) + len(preserved)
        ),
        complete_identity_token_coverage=complete,
        distinct_factor_basis_validated=distinct,
        reason_codes=list(dict.fromkeys(reasons)),
    )


def build_grounded_identity_factorization_report(
    *,
    relation_ir_report: ScientificRelationIRReport,
    annotation_report: GroundedIdentityAnnotationReport,
) -> GroundedIdentityFactorizationReport:
    annotation_by_claim = {
        row.claim_id: row
        for row in annotation_report.annotations
    }

    relation_claim_ids = {
        row.claim_id
        for row in relation_ir_report.relations
    }
    unknown_annotations = sorted(
        set(annotation_by_claim) - relation_claim_ids
    )
    if unknown_annotations:
        raise ValueError(
            "grounded identity annotation contains claims absent from "
            "ScientificRelationIR: "
            + ", ".join(unknown_annotations)
        )

    rows = [
        factorize_relation_identity(
            relation=relation,
            annotation=annotation_by_claim.get(
                relation.claim_id
            ),
        )
        for relation in relation_ir_report.relations
    ]

    return GroundedIdentityFactorizationReport(
        report_id=_stable_id(
            "grounded_identity_factorization_report",
            relation_ir_report.report_id,
            annotation_report.source_atomic_synthesis_report_id,
            *[
                (
                    row.relation_ir_id,
                    row.factorization_status,
                    *[
                        factor.factor_id
                        for factor in row.grounded_factors
                    ],
                )
                for row in rows
            ],
        ),
        source_relation_ir_report_id=relation_ir_report.report_id,
        source_annotation_candidate_portfolio_id=(
            annotation_report.source_candidate_portfolio_id
        ),
        source_annotation_atomic_report_id=(
            annotation_report.source_atomic_synthesis_report_id
        ),
        relations=rows,
        relation_count=len(rows),
        ready_count=sum(
            row.factorization_status == "READY"
            for row in rows
        ),
        not_ready_count=sum(
            row.factorization_status != "READY"
            for row in rows
        ),
        total_grounded_factor_count=sum(
            row.grounded_factor_count
            for row in rows
        ),
    )


__all__ = [
    "GroundedIdentityFactor",
    "GroundedIdentityFactorizationReport",
    "RelationIdentityFactorization",
    "build_grounded_identity_factorization_report",
    "factorize_relation_identity",
]
