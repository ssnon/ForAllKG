from __future__ import annotations

import hashlib
from itertools import combinations, product
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.grounded_identity_factorization import (
    GroundedIdentityFactor,
    GroundedIdentityFactorizationReport,
    RelationIdentityFactorization,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIR,
    ScientificRelationIRReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


GroundedProjectionKind = Literal[
    "BASE_RELATION",
    "LOWER_ORDER_FACTOR_SUBSET",
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


class GroundedProjectionFactorBinding(StrictModel):
    factor_id: str
    group_label: str
    identity_basis_tokens: list[str] = Field(min_length=1)
    exclusive_identity_basis_tokens: list[str] = Field(min_length=1)
    exact_source_aliases: list[str] = Field(min_length=1)

    aliases_are_source_exact: Literal[True] = True
    factor_semantics_are_identity_lexical_basis_only: Literal[True] = True


class GroundedFactorRelationProjection(StrictModel):
    schema_version: Literal[
        "grounded-factor-relation-projection-v1"
    ] = "grounded-factor-relation-projection-v1"

    projection_id: str
    relation_ir_id: str
    hypothesis_id: str
    claim_id: str
    projection_kind: GroundedProjectionKind

    source_identity_term: str
    retained_factor_ids: list[str] = Field(default_factory=list)
    omitted_factor_ids: list[str] = Field(default_factory=list)
    factor_order: int = Field(ge=0)
    factor_bindings: list[GroundedProjectionFactorBinding] = Field(
        default_factory=list
    )

    endpoint_terms: list[str] = Field(min_length=2)
    scope_terms: list[str] = Field(default_factory=list)
    directional_terms: list[str] = Field(default_factory=list)

    canonical_search_terms: list[str] = Field(min_length=2)
    canonical_search_query: str = Field(min_length=1)
    exact_source_query_variants: list[str] = Field(min_length=1)
    exact_source_query_variant_count: int = Field(ge=1)

    relation_typing_status: str
    factorization_status: str
    eligible_for_future_typed_retrieval: bool
    reason_codes: list[str] = Field(default_factory=list)

    endpoints_always_retained: Literal[True] = True
    scope_qualifiers_always_retained: Literal[True] = True
    directional_qualifiers_always_retained: Literal[True] = True
    source_exact_factor_aliases_only: Literal[True] = True

    novelty_authority: Literal[False] = False
    prior_art_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "GroundedFactorRelationProjection":
        if self.factor_order != len(self.retained_factor_ids):
            raise ValueError("factor_order mismatch")
        if self.factor_order != len(self.factor_bindings):
            raise ValueError("factor binding count mismatch")
        if (
            self.exact_source_query_variant_count
            != len(self.exact_source_query_variants)
        ):
            raise ValueError("exact source query variant count mismatch")
        return self


class GroundedFactorProjectionSet(StrictModel):
    schema_version: Literal[
        "grounded-factor-projection-set-v1"
    ] = "grounded-factor-projection-set-v1"

    relation_ir_id: str
    hypothesis_id: str
    claim_id: str
    source_identity_term: str | None = None
    factorization_status: str

    grounded_factor_count: int = Field(ge=0)
    max_lower_order_factor_subset_size: int = Field(ge=1)
    max_projections_per_relation: int = Field(ge=2)
    max_alias_query_variants_per_projection: int = Field(ge=1)

    projections: list[GroundedFactorRelationProjection]
    projection_count: int = Field(ge=0)
    truncated: bool = False
    planning_ready: bool
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "GroundedFactorProjectionSet":
        if self.projection_count != len(self.projections):
            raise ValueError("projection_count mismatch")
        ids = [row.projection_id for row in self.projections]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate factor projection ID")
        if self.planning_ready and self.projection_count < 2:
            raise ValueError(
                "planning-ready factorized relation requires BASE and FULL"
            )
        return self


class GroundedFactorProjectionReport(StrictModel):
    schema_version: Literal[
        "grounded-factor-projection-report-v1"
    ] = "grounded-factor-projection-report-v1"

    report_id: str
    source_relation_ir_report_id: str
    source_factorization_report_id: str

    projection_sets: list[GroundedFactorProjectionSet]
    relation_count: int = Field(ge=0)
    planning_ready_relation_count: int = Field(ge=0)
    projection_count: int = Field(ge=0)
    future_retrieval_eligible_projection_count: int = Field(ge=0)
    truncated_relation_count: int = Field(ge=0)

    projection_policy: Literal[
        "source_grounded_factor_subset_v1"
    ] = "source_grounded_factor_subset_v1"

    diagnostic_only: Literal[True] = True
    query_plan_changed: Literal[False] = False
    retrieval_changed: Literal[False] = False
    prior_art_review_changed: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "GroundedFactorProjectionReport":
        if self.relation_count != len(self.projection_sets):
            raise ValueError("relation_count mismatch")
        if self.planning_ready_relation_count != sum(
            row.planning_ready
            for row in self.projection_sets
        ):
            raise ValueError("planning_ready_relation_count mismatch")
        if self.projection_count != sum(
            row.projection_count
            for row in self.projection_sets
        ):
            raise ValueError("projection_count mismatch")
        return self


def _binding(
    factor: GroundedIdentityFactor,
) -> GroundedProjectionFactorBinding:
    return GroundedProjectionFactorBinding(
        factor_id=factor.factor_id,
        group_label=factor.group_label,
        identity_basis_tokens=list(
            factor.identity_basis_tokens
        ),
        exclusive_identity_basis_tokens=list(
            factor.exclusive_identity_basis_tokens
        ),
        exact_source_aliases=list(
            factor.query_aliases
        ),
    )


def _base_fixed_terms(
    relation: ScientificRelationIR,
) -> list[str]:
    return _unique_text(
        [
            *[
                row.surface_text
                for row in relation.endpoint_concepts
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


def _query_variants(
    *,
    fixed_terms: list[str],
    factors: tuple[GroundedIdentityFactor, ...],
    max_variants: int,
) -> list[str]:
    if max_variants < 1:
        raise ValueError("max_variants must be >= 1")

    if not factors:
        query = " ".join(_unique_text(fixed_terms))
        if not query:
            raise ValueError("empty factor projection query")
        return [query]

    alias_groups = [
        tuple(_unique_text(factor.query_aliases))
        for factor in factors
    ]
    if any(not group for group in alias_groups):
        raise ValueError(
            "grounded projection factor missing exact source query alias"
        )

    variants: list[str] = []
    seen: set[str] = set()

    for aliases in product(*alias_groups):
        query = " ".join(
            _unique_text(
                [
                    *fixed_terms,
                    *aliases,
                ]
            )
        )
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            variants.append(query)
        if len(variants) >= max_variants:
            break

    if not variants:
        raise ValueError("no exact-source factor query variants")
    return variants


def _projection(
    *,
    relation: ScientificRelationIR,
    factorization: RelationIdentityFactorization,
    kind: GroundedProjectionKind,
    retained: tuple[GroundedIdentityFactor, ...],
    all_factors: tuple[GroundedIdentityFactor, ...],
    max_alias_query_variants: int,
) -> GroundedFactorRelationProjection:
    retained_ids = [
        row.factor_id
        for row in retained
    ]
    retained_set = set(retained_ids)
    omitted_ids = [
        row.factor_id
        for row in all_factors
        if row.factor_id not in retained_set
    ]

    fixed_terms = _base_fixed_terms(relation)
    variants = _query_variants(
        fixed_terms=fixed_terms,
        factors=retained,
        max_variants=max_alias_query_variants,
    )

    canonical_terms = _unique_text(
        [
            *fixed_terms,
            *[
                factor.query_aliases[0]
                for factor in retained
            ],
        ]
    )

    reasons: list[str] = []
    if kind == "BASE_RELATION":
        reasons.append(
            "all_grounded_identity_factors_projected_away"
        )
    elif kind == "LOWER_ORDER_FACTOR_SUBSET":
        reasons.append(
            "proper_grounded_factor_subset_projection"
        )
    else:
        reasons.append(
            "all_grounded_identity_factors_retained"
        )

    eligible = (
        relation.typing_status == "READY"
        and factorization.factorization_status == "READY"
    )
    if not eligible:
        reasons.append(
            "relation_or_factorization_not_ready_for_future_typed_retrieval"
        )

    return GroundedFactorRelationProjection(
        projection_id=_stable_id(
            "grounded_factor_relation_projection",
            relation.relation_ir_id,
            kind,
            *retained_ids,
            *variants,
        ),
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        projection_kind=kind,
        source_identity_term=str(
            factorization.source_identity_term or ""
        ),
        retained_factor_ids=retained_ids,
        omitted_factor_ids=omitted_ids,
        factor_order=len(retained),
        factor_bindings=[
            _binding(factor)
            for factor in retained
        ],
        endpoint_terms=[
            row.surface_text
            for row in relation.endpoint_concepts
        ],
        scope_terms=[
            row.surface_text
            for row in relation.scope_qualifiers
        ],
        directional_terms=[
            row.surface_text
            for row in relation.directional_qualifiers
        ],
        canonical_search_terms=canonical_terms,
        canonical_search_query=variants[0],
        exact_source_query_variants=variants,
        exact_source_query_variant_count=len(variants),
        relation_typing_status=relation.typing_status,
        factorization_status=factorization.factorization_status,
        eligible_for_future_typed_retrieval=eligible,
        reason_codes=reasons,
    )


def compile_grounded_factor_projection_set(
    *,
    relation: ScientificRelationIR,
    factorization: RelationIdentityFactorization,
    max_lower_order_factor_subset_size: int = 2,
    max_projections_per_relation: int = 12,
    max_alias_query_variants_per_projection: int = 4,
) -> GroundedFactorProjectionSet:
    if relation.relation_ir_id != factorization.relation_ir_id:
        raise ValueError(
            "relation/factorization relation_ir_id mismatch"
        )
    if relation.claim_id != factorization.claim_id:
        raise ValueError(
            "relation/factorization claim_id mismatch"
        )
    if max_lower_order_factor_subset_size < 1:
        raise ValueError(
            "max_lower_order_factor_subset_size must be >= 1"
        )
    if max_projections_per_relation < 2:
        raise ValueError(
            "max_projections_per_relation must be >= 2"
        )
    if max_alias_query_variants_per_projection < 1:
        raise ValueError(
            "max_alias_query_variants_per_projection must be >= 1"
        )

    if (
        relation.typing_status != "READY"
        or factorization.factorization_status != "READY"
        or factorization.grounded_factor_count < 2
    ):
        return GroundedFactorProjectionSet(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            source_identity_term=factorization.source_identity_term,
            factorization_status=factorization.factorization_status,
            grounded_factor_count=factorization.grounded_factor_count,
            max_lower_order_factor_subset_size=(
                max_lower_order_factor_subset_size
            ),
            max_projections_per_relation=max_projections_per_relation,
            max_alias_query_variants_per_projection=(
                max_alias_query_variants_per_projection
            ),
            projections=[],
            projection_count=0,
            planning_ready=False,
            reason_codes=[
                "factorized_projection_requires_ready_typed_relation",
                "factorized_projection_requires_ready_grounded_factorization",
            ],
        )

    factors = tuple(factorization.grounded_factors)
    count = len(factors)

    projections: list[GroundedFactorRelationProjection] = [
        _projection(
            relation=relation,
            factorization=factorization,
            kind="BASE_RELATION",
            retained=(),
            all_factors=factors,
            max_alias_query_variants=(
                max_alias_query_variants_per_projection
            ),
        )
    ]

    upper = min(
        max_lower_order_factor_subset_size,
        count - 1,
    )
    proper_subsets: list[tuple[GroundedIdentityFactor, ...]] = []
    for size in range(1, upper + 1):
        proper_subsets.extend(
            combinations(
                factors,
                size,
            )
        )

    available_subset_slots = max(
        0,
        max_projections_per_relation - 2,
    )
    selected = proper_subsets[:available_subset_slots]
    truncated = len(selected) < len(proper_subsets)

    for subset in selected:
        projections.append(
            _projection(
                relation=relation,
                factorization=factorization,
                kind="LOWER_ORDER_FACTOR_SUBSET",
                retained=subset,
                all_factors=factors,
                max_alias_query_variants=(
                    max_alias_query_variants_per_projection
                ),
            )
        )

    projections.append(
        _projection(
            relation=relation,
            factorization=factorization,
            kind="FULL_RELATION",
            retained=factors,
            all_factors=factors,
            max_alias_query_variants=(
                max_alias_query_variants_per_projection
            ),
        )
    )

    reasons: list[str] = [
        "source_grounded_factorization_used_for_projection"
    ]
    if truncated:
        reasons.append(
            "grounded_factor_projection_budget_truncated"
        )

    return GroundedFactorProjectionSet(
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        source_identity_term=factorization.source_identity_term,
        factorization_status=factorization.factorization_status,
        grounded_factor_count=len(factors),
        max_lower_order_factor_subset_size=(
            max_lower_order_factor_subset_size
        ),
        max_projections_per_relation=max_projections_per_relation,
        max_alias_query_variants_per_projection=(
            max_alias_query_variants_per_projection
        ),
        projections=projections,
        projection_count=len(projections),
        truncated=truncated,
        planning_ready=True,
        reason_codes=reasons,
    )


def compile_grounded_factor_projection_report(
    *,
    relation_ir_report: ScientificRelationIRReport,
    factorization_report: GroundedIdentityFactorizationReport,
    max_lower_order_factor_subset_size: int = 2,
    max_projections_per_relation: int = 12,
    max_alias_query_variants_per_projection: int = 4,
) -> GroundedFactorProjectionReport:
    factorization_by_relation = {
        row.relation_ir_id: row
        for row in factorization_report.relations
    }

    relation_ids = {
        row.relation_ir_id
        for row in relation_ir_report.relations
    }
    unknown = sorted(
        set(factorization_by_relation) - relation_ids
    )
    if unknown:
        raise ValueError(
            "factorization contains unknown relation IR IDs: "
            + ", ".join(unknown)
        )

    sets: list[GroundedFactorProjectionSet] = []

    for relation in relation_ir_report.relations:
        factorization = factorization_by_relation.get(
            relation.relation_ir_id
        )
        if factorization is None:
            raise ValueError(
                "missing grounded identity factorization for relation: "
                + relation.relation_ir_id
            )

        sets.append(
            compile_grounded_factor_projection_set(
                relation=relation,
                factorization=factorization,
                max_lower_order_factor_subset_size=(
                    max_lower_order_factor_subset_size
                ),
                max_projections_per_relation=(
                    max_projections_per_relation
                ),
                max_alias_query_variants_per_projection=(
                    max_alias_query_variants_per_projection
                ),
            )
        )

    return GroundedFactorProjectionReport(
        report_id=_stable_id(
            "grounded_factor_projection_report",
            relation_ir_report.report_id,
            factorization_report.report_id,
            max_lower_order_factor_subset_size,
            max_projections_per_relation,
            max_alias_query_variants_per_projection,
            *[
                projection.projection_id
                for row in sets
                for projection in row.projections
            ],
        ),
        source_relation_ir_report_id=relation_ir_report.report_id,
        source_factorization_report_id=factorization_report.report_id,
        projection_sets=sets,
        relation_count=len(sets),
        planning_ready_relation_count=sum(
            row.planning_ready
            for row in sets
        ),
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
    "GroundedFactorProjectionReport",
    "GroundedFactorProjectionSet",
    "GroundedFactorRelationProjection",
    "compile_grounded_factor_projection_report",
    "compile_grounded_factor_projection_set",
]
