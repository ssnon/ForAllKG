from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.scientific_relation_ir import (
    ScientificConceptIR,
    ScientificRelationIR,
    ScientificRelationIRReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ProjectionKind = Literal[
    "BASE_RELATION",
    "LOWER_ORDER_IDENTITY_SUBSET",
    "FULL_RELATION",
]


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _unique_text(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(str(raw or "").split()).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


class ScientificRelationProjection(StrictModel):
    schema_version: Literal[
        "scientific-relation-projection-v1"
    ] = "scientific-relation-projection-v1"

    projection_id: str
    relation_ir_id: str
    hypothesis_id: str
    claim_id: str
    projection_kind: ProjectionKind

    retained_identity_concept_ids: list[str] = Field(default_factory=list)
    omitted_identity_concept_ids: list[str] = Field(default_factory=list)
    identity_order: int = Field(ge=0)

    endpoint_terms: list[str] = Field(min_length=2)
    retained_identity_terms: list[str] = Field(default_factory=list)
    scope_terms: list[str] = Field(default_factory=list)
    directional_terms: list[str] = Field(default_factory=list)

    search_terms: list[str] = Field(min_length=2)
    search_query: str = Field(min_length=1)

    source_exact_spans_only: Literal[True] = True
    endpoints_always_retained: Literal[True] = True
    scope_qualifiers_always_retained: Literal[True] = True
    directional_qualifiers_always_retained: Literal[True] = True

    relation_typing_status: str
    eligible_for_future_typed_retrieval: bool
    reason_codes: list[str] = Field(default_factory=list)

    novelty_authority: Literal[False] = False
    prior_art_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ScientificRelationProjectionSet(StrictModel):
    schema_version: Literal[
        "scientific-relation-projection-set-v1"
    ] = "scientific-relation-projection-set-v1"

    relation_ir_id: str
    hypothesis_id: str
    claim_id: str
    identity_concept_count: int = Field(ge=0)
    max_lower_order_identity_subset_size: int = Field(ge=1)
    max_projections_per_relation: int = Field(ge=2)

    projections: list[ScientificRelationProjection]
    projection_count: int = Field(ge=0)
    truncated: bool = False
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificRelationProjectionSet":
        if self.projection_count != len(self.projections):
            raise ValueError("projection_count mismatch")
        ids = [row.projection_id for row in self.projections]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate relation projection_id")
        return self


class ScientificRelationProjectionReport(StrictModel):
    schema_version: Literal[
        "scientific-relation-projection-report-v1"
    ] = "scientific-relation-projection-report-v1"

    report_id: str
    source_relation_ir_report_id: str
    projection_sets: list[ScientificRelationProjectionSet]
    relation_count: int = Field(ge=0)
    projection_count: int = Field(ge=0)
    future_retrieval_eligible_projection_count: int = Field(ge=0)
    truncated_relation_count: int = Field(ge=0)

    projection_policy: Literal[
        "bounded_exact_identity_subset_v1"
    ] = "bounded_exact_identity_subset_v1"

    diagnostic_only: Literal[True] = True
    query_plan_changed: Literal[False] = False
    retrieval_changed: Literal[False] = False
    prior_art_review_changed: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificRelationProjectionReport":
        if self.relation_count != len(self.projection_sets):
            raise ValueError("relation_count mismatch")
        actual_projection_count = sum(
            row.projection_count
            for row in self.projection_sets
        )
        if self.projection_count != actual_projection_count:
            raise ValueError("projection_count mismatch")
        return self


def _projection_query_terms(
    relation: ScientificRelationIR,
    retained_identities: tuple[ScientificConceptIR, ...],
) -> list[str]:
    # Relation endpoints carry the lower-order relation itself.
    # Scope and direction remain fixed because dropping them would change the
    # scientific applicability/direction rather than merely project away a
    # distinguishing identity factor.
    return _unique_text(
        [
            *[
                row.surface_text
                for row in relation.endpoint_concepts
            ],
            *[
                row.surface_text
                for row in retained_identities
            ],
            *[
                row.surface_text
                for row in relation.scope_qualifiers
            ],
            *[
                row.surface_text
                for row in relation.directional_qualifiers
            ],
        ]
    )


def _make_projection(
    *,
    relation: ScientificRelationIR,
    kind: ProjectionKind,
    retained_identities: tuple[ScientificConceptIR, ...],
) -> ScientificRelationProjection:
    all_ids = [
        row.concept_id
        for row in relation.identity_concepts
    ]
    retained_ids = [
        row.concept_id
        for row in retained_identities
    ]
    retained_set = set(retained_ids)
    omitted_ids = [
        value
        for value in all_ids
        if value not in retained_set
    ]

    search_terms = _projection_query_terms(
        relation,
        retained_identities,
    )
    if len(search_terms) < 2:
        raise ValueError(
            "scientific relation projection requires at least two search terms"
        )

    eligible = relation.typing_status == "READY"
    reasons: list[str] = []
    if not eligible:
        reasons.append(
            "relation_not_ready_for_typed_retrieval:"
            + relation.typing_status
        )

    if kind == "BASE_RELATION":
        reasons.append(
            "all_branch_identity_factors_projected_away"
        )
    elif kind == "LOWER_ORDER_IDENTITY_SUBSET":
        reasons.append(
            "proper_identity_subset_projection"
        )
    else:
        reasons.append(
            "full_identity_relation_preserved"
        )

    return ScientificRelationProjection(
        projection_id=_stable_id(
            "scientific_relation_projection",
            relation.relation_ir_id,
            kind,
            *retained_ids,
            *search_terms,
        ),
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        projection_kind=kind,
        retained_identity_concept_ids=retained_ids,
        omitted_identity_concept_ids=omitted_ids,
        identity_order=len(retained_ids),
        endpoint_terms=[
            row.surface_text
            for row in relation.endpoint_concepts
        ],
        retained_identity_terms=[
            row.surface_text
            for row in retained_identities
        ],
        scope_terms=[
            row.surface_text
            for row in relation.scope_qualifiers
        ],
        directional_terms=[
            row.surface_text
            for row in relation.directional_qualifiers
        ],
        search_terms=search_terms,
        search_query=" ".join(search_terms),
        relation_typing_status=relation.typing_status,
        eligible_for_future_typed_retrieval=eligible,
        reason_codes=reasons,
    )


def compile_relation_projection_set(
    relation: ScientificRelationIR,
    *,
    max_lower_order_identity_subset_size: int = 2,
    max_projections_per_relation: int = 12,
) -> ScientificRelationProjectionSet:
    """Compile bounded proper lower-order projections from one typed relation.

    Projection semantics:
    - endpoints are NEVER projected away;
    - scope/directional qualifiers are NEVER projected away;
    - only BRANCH_IDENTITY concepts are projected;
    - BASE_RELATION removes every branch identity;
    - proper non-empty identity subsets form LOWER_ORDER projections;
    - FULL_RELATION preserves every branch identity.

    The compiler uses only exact source spans already present in
    ScientificRelationIR. It invents no synonyms or scientific equivalences.
    """

    if max_lower_order_identity_subset_size < 1:
        raise ValueError(
            "max_lower_order_identity_subset_size must be >= 1"
        )
    if max_projections_per_relation < 2:
        raise ValueError(
            "max_projections_per_relation must be >= 2"
        )
    if len(relation.endpoint_concepts) < 2:
        raise ValueError(
            "relation projection requires at least two endpoints"
        )

    identities = tuple(relation.identity_concepts)
    projections: list[ScientificRelationProjection] = []

    projections.append(
        _make_projection(
            relation=relation,
            kind="BASE_RELATION",
            retained_identities=(),
        )
    )

    proper_subsets: list[tuple[ScientificConceptIR, ...]] = []
    identity_count = len(identities)

    if identity_count >= 2:
        upper = min(
            max_lower_order_identity_subset_size,
            identity_count - 1,
        )
        for size in range(1, upper + 1):
            proper_subsets.extend(
                combinations(
                    identities,
                    size,
                )
            )

    # Deterministic bounded projection budget. Reserve one slot for FULL.
    available_lower_order_slots = max(
        0,
        max_projections_per_relation - 2,
    )
    selected_subsets = proper_subsets[
        :available_lower_order_slots
    ]
    truncated = len(selected_subsets) < len(proper_subsets)

    for subset in selected_subsets:
        projections.append(
            _make_projection(
                relation=relation,
                kind="LOWER_ORDER_IDENTITY_SUBSET",
                retained_identities=subset,
            )
        )

    if identities:
        projections.append(
            _make_projection(
                relation=relation,
                kind="FULL_RELATION",
                retained_identities=identities,
            )
        )
    else:
        # With no branch identities, BASE is already the full relation.
        # Do not duplicate the same scientific target under a second label.
        truncated = False

    reasons: list[str] = []
    if identity_count == 0:
        reasons.append(
            "no_branch_identity_factors_base_equals_full"
        )
    elif identity_count == 1:
        reasons.append(
            "single_identity_factor_no_proper_nonempty_subset"
        )
    if truncated:
        reasons.append(
            "lower_order_projection_budget_truncated"
        )
    if relation.typing_status != "READY":
        reasons.append(
            "relation_not_ready_for_future_typed_retrieval"
        )

    return ScientificRelationProjectionSet(
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        identity_concept_count=identity_count,
        max_lower_order_identity_subset_size=(
            max_lower_order_identity_subset_size
        ),
        max_projections_per_relation=max_projections_per_relation,
        projections=projections,
        projection_count=len(projections),
        truncated=truncated,
        reason_codes=reasons,
    )


def compile_relation_projection_report(
    relation_ir_report: ScientificRelationIRReport,
    *,
    max_lower_order_identity_subset_size: int = 2,
    max_projections_per_relation: int = 12,
) -> ScientificRelationProjectionReport:
    sets = [
        compile_relation_projection_set(
            relation,
            max_lower_order_identity_subset_size=(
                max_lower_order_identity_subset_size
            ),
            max_projections_per_relation=max_projections_per_relation,
        )
        for relation in relation_ir_report.relations
    ]

    return ScientificRelationProjectionReport(
        report_id=_stable_id(
            "scientific_relation_projection_report",
            relation_ir_report.report_id,
            max_lower_order_identity_subset_size,
            max_projections_per_relation,
            *[
                projection.projection_id
                for row in sets
                for projection in row.projections
            ],
        ),
        source_relation_ir_report_id=relation_ir_report.report_id,
        projection_sets=sets,
        relation_count=len(sets),
        projection_count=sum(
            row.projection_count
            for row in sets
        ),
        future_retrieval_eligible_projection_count=sum(
            projection.eligible_for_future_typed_retrieval
            for row in sets
            for projection in row.projections
        ),
        truncated_relation_count=sum(
            row.truncated
            for row in sets
        ),
    )


__all__ = [
    "ScientificRelationProjection",
    "ScientificRelationProjectionReport",
    "ScientificRelationProjectionSet",
    "compile_relation_projection_report",
    "compile_relation_projection_set",
]
