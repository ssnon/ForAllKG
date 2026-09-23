from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
    GroundedFactorProjectionSet,
    GroundedFactorRelationProjection,
    GroundedProjectionFactorBinding,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_projection import (
    RelationalAtomicProjectionReport,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIR,
    ScientificRelationIRReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AtomicIdentityFactorizationStatus = Literal[
    "ATOMIC_IDENTITY_READY",
    "RELATION_NOT_READY",
    "IDENTITY_CARDINALITY_UNSUPPORTED",
    "SOURCE_GROUNDING_MISSING",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _normalize_space(value: object) -> str:
    return " ".join(str(value or "").split())


def _candidate_fields(
    candidate: HypothesisCard,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("title", candidate.title),
        ("hypothesis_statement", candidate.hypothesis_statement),
        ("inferential_bridge", candidate.inferential_bridge),
    ]
    rows.extend(
        (f"assumptions[{index}]", value)
        for index, value in enumerate(candidate.assumptions)
    )
    for index, row in enumerate(candidate.predicted_observations):
        rows.append(
            (f"predicted_observations[{index}].observable", row.observable)
        )
        rows.append(
            (f"predicted_observations[{index}].rationale", row.rationale)
        )
    for index, row in enumerate(candidate.falsification_criteria):
        rows.append(
            (f"falsification_criteria[{index}].observable", row.observable)
        )
        rows.append(
            (
                f"falsification_criteria[{index}].falsifying_outcome",
                row.falsifying_outcome,
            )
        )
    return [
        (path, str(value))
        for path, value in rows
        if str(value or "").strip()
    ]


def _literal_identity_occurrences(
    candidate: HypothesisCard,
    identity_term: str,
) -> list[tuple[str, str]]:
    """Return source-exact occurrences, allowing only case/whitespace variance."""

    tokens = _normalize_space(identity_term).split()
    if not tokens:
        return []

    pattern = re.compile(
        r"(?<!\w)"
        + r"\s+".join(re.escape(token) for token in tokens)
        + r"(?!\w)",
        flags=re.IGNORECASE,
    )
    hits: list[tuple[str, str]] = []
    for path, value in _candidate_fields(candidate):
        match = pattern.search(value)
        if match is None:
            continue
        hits.append((path, match.group(0)))
    return hits


def _unique_alias_groups(
    hits: list[tuple[str, str]],
) -> list[tuple[str, list[str]]]:
    aliases: dict[str, tuple[str, list[str]]] = {}
    for path, exact_text in hits:
        key = _normalize_space(exact_text).casefold()
        if key not in aliases:
            aliases[key] = (_normalize_space(exact_text), [])
        aliases[key][1].append(path)
    return [
        (text, sorted(set(paths)))
        for text, paths in aliases.values()
    ]


class RelationalAtomicIdentitySourceSpan(StrictModel):
    candidate_id: str
    exact_source_text: str = Field(min_length=1)
    matched_source_paths: list[str] = Field(min_length=1)

    source_exact_span: Literal[True] = True
    semantic_paraphrase_used: Literal[False] = False


class RelationalAtomicIdentityFactorization(StrictModel):
    relation_ir_id: str
    hypothesis_id: str
    claim_id: str
    candidate_hypothesis_id: str

    factorization_status: AtomicIdentityFactorizationStatus
    source_identity_concept_id: str | None = None
    source_identity_term: str | None = None
    identity_basis_tokens: list[str] = Field(default_factory=list)
    source_spans: list[RelationalAtomicIdentitySourceSpan] = Field(
        default_factory=list
    )

    factor_id: str | None = None
    exact_source_aliases: list[str] = Field(default_factory=list)

    reason_codes: list[str] = Field(default_factory=list)

    atomic_identity_preserved_without_decomposition: Literal[True] = True
    synthetic_constituent_decomposition_performed: Literal[False] = False
    scientific_equivalence_inferred: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_state(self) -> "RelationalAtomicIdentityFactorization":
        ready = self.factorization_status == "ATOMIC_IDENTITY_READY"
        if ready:
            if self.source_identity_concept_id is None:
                raise ValueError(
                    "ATOMIC_IDENTITY_READY requires source identity concept"
                )
            if not self.source_identity_term:
                raise ValueError(
                    "ATOMIC_IDENTITY_READY requires source identity term"
                )
            if not self.identity_basis_tokens:
                raise ValueError(
                    "ATOMIC_IDENTITY_READY requires identity basis tokens"
                )
            if not self.source_spans or not self.exact_source_aliases:
                raise ValueError(
                    "ATOMIC_IDENTITY_READY requires exact source grounding"
                )
            if self.factor_id is None:
                raise ValueError(
                    "ATOMIC_IDENTITY_READY requires factor_id"
                )
            if self.reason_codes:
                raise ValueError(
                    "ATOMIC_IDENTITY_READY cannot carry failure reason_codes"
                )
        else:
            if self.factor_id is not None:
                raise ValueError(
                    "non-ready atomic identity cannot carry factor_id"
                )
            if self.source_spans or self.exact_source_aliases:
                raise ValueError(
                    "non-ready atomic identity cannot carry source aliases"
                )
            if not self.reason_codes:
                raise ValueError(
                    "non-ready atomic identity requires reason_codes"
                )
        return self


class RelationalAtomicIdentityFactorizationReport(StrictModel):
    schema_version: Literal[
        "relational-atomic-identity-factorization-report-v1"
    ] = "relational-atomic-identity-factorization-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relational_projection_report_id: str
    source_relation_ir_report_id: str

    relations: list[RelationalAtomicIdentityFactorization]
    relation_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    not_ready_count: int = Field(ge=0)

    source_candidate_artifacts_hash_checked: Literal[True] = True
    source_exact_identity_required: Literal[True] = True
    atomic_identity_preserved_without_decomposition: Literal[True] = True
    synthetic_constituent_decomposition_performed: Literal[False] = False

    scientific_content_added: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "RelationalAtomicIdentityFactorizationReport":
        if self.relation_count != len(self.relations):
            raise ValueError("relation_count mismatch")
        ready = sum(
            row.factorization_status == "ATOMIC_IDENTITY_READY"
            for row in self.relations
        )
        if self.ready_count != ready:
            raise ValueError("ready_count mismatch")
        if self.not_ready_count != len(self.relations) - ready:
            raise ValueError("not_ready_count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("atomic identity factorization SHA mismatch")
        if observed_id != (
            "relational_atomic_identity_factorization:"
            + expected_sha[:20]
        ):
            raise ValueError("atomic identity factorization ID mismatch")
        return self


def factorize_relational_atomic_identity(
    *,
    relation: ScientificRelationIR,
    candidate: HypothesisCard,
    candidate_hypothesis_id: str,
) -> RelationalAtomicIdentityFactorization:
    if relation.typing_status != "READY":
        return RelationalAtomicIdentityFactorization(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            candidate_hypothesis_id=candidate_hypothesis_id,
            factorization_status="RELATION_NOT_READY",
            reason_codes=[
                "relation_typing_status:" + relation.typing_status
            ],
        )

    if len(relation.identity_concepts) != 1:
        return RelationalAtomicIdentityFactorization(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            candidate_hypothesis_id=candidate_hypothesis_id,
            factorization_status="IDENTITY_CARDINALITY_UNSUPPORTED",
            reason_codes=[
                "relational_atomic_identity_requires_exactly_one_identity_concept"
            ],
        )

    identity = relation.identity_concepts[0]
    basis = list(dict.fromkeys(identity.lexical_tokens))
    if not basis:
        return RelationalAtomicIdentityFactorization(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            candidate_hypothesis_id=candidate_hypothesis_id,
            factorization_status="SOURCE_GROUNDING_MISSING",
            source_identity_concept_id=identity.concept_id,
            source_identity_term=identity.surface_text,
            reason_codes=["atomic_identity_has_no_lexical_basis_tokens"],
        )

    hits = _literal_identity_occurrences(
        candidate,
        identity.surface_text,
    )
    aliases = _unique_alias_groups(hits)
    if not aliases:
        return RelationalAtomicIdentityFactorization(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            candidate_hypothesis_id=candidate_hypothesis_id,
            factorization_status="SOURCE_GROUNDING_MISSING",
            source_identity_concept_id=identity.concept_id,
            source_identity_term=identity.surface_text,
            identity_basis_tokens=basis,
            reason_codes=[
                "atomic_identity_not_exact_in_candidate_hypothesis_source"
            ],
        )

    spans = [
        RelationalAtomicIdentitySourceSpan(
            candidate_id=candidate_hypothesis_id,
            exact_source_text=exact_text,
            matched_source_paths=paths,
        )
        for exact_text, paths in aliases
    ]
    exact_aliases = [
        row.exact_source_text
        for row in spans
    ]
    factor_id = _stable_id(
        "relational_atomic_identity_factor",
        relation.relation_ir_id,
        identity.concept_id,
        *basis,
        *exact_aliases,
    )

    return RelationalAtomicIdentityFactorization(
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        candidate_hypothesis_id=candidate_hypothesis_id,
        factorization_status="ATOMIC_IDENTITY_READY",
        source_identity_concept_id=identity.concept_id,
        source_identity_term=identity.surface_text,
        identity_basis_tokens=basis,
        source_spans=spans,
        factor_id=factor_id,
        exact_source_aliases=exact_aliases,
    )


def _candidate_card(
    *,
    plan: RelationalAtomicBindingPlan,
    final_hypothesis_id: str,
    candidate_hypothesis_id: str,
) -> HypothesisCard:
    matches = [
        row
        for row in plan.hypotheses
        if row.final_hypothesis_id == final_hypothesis_id
        and row.candidate_hypothesis_id == candidate_hypothesis_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "binding plan must resolve exactly one candidate/final lineage"
        )
    hypothesis_plan = matches[0]

    portfolio_path = Path(hypothesis_plan.source_candidate_portfolio)
    if not portfolio_path.is_file():
        raise ValueError(
            "candidate source portfolio missing: " + str(portfolio_path)
        )
    if _sha256_file(portfolio_path) != (
        hypothesis_plan.source_candidate_portfolio_sha256
    ):
        raise ValueError(
            "candidate source portfolio changed after binding-plan freeze"
        )

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    cards = [
        row
        for row in portfolio.hypotheses
        if row.hypothesis_id == candidate_hypothesis_id
    ]
    if len(cards) != 1:
        raise ValueError(
            "candidate source portfolio must resolve exactly one hypothesis"
        )
    return cards[0]


def build_relational_atomic_identity_factorization_report(
    *,
    plan: RelationalAtomicBindingPlan,
    projection_report: RelationalAtomicProjectionReport,
    relation_ir_report: ScientificRelationIRReport,
) -> RelationalAtomicIdentityFactorizationReport:
    if projection_report.source_binding_plan_id != plan.plan_id:
        raise ValueError(
            "relational projection/source binding plan ID mismatch"
        )
    if projection_report.source_binding_plan_sha256 != plan.plan_sha256:
        raise ValueError(
            "relational projection/source binding plan SHA mismatch"
        )
    if relation_ir_report.source_contract != (
        "relational-atomic-projection-report-v1"
    ):
        raise ValueError(
            "relational identity factorization requires relational source contract"
        )
    if relation_ir_report.source_atomic_report_id != projection_report.report_id:
        raise ValueError(
            "relation IR/relational projection provenance mismatch"
        )

    projected_rows = {
        (row.final_hypothesis_id, row.claim_id): row
        for row in projection_report.rows
        if row.projection_status == "PROJECTED"
    }
    if len(projected_rows) != sum(
        row.projection_status == "PROJECTED"
        for row in projection_report.rows
    ):
        raise ValueError(
            "duplicate projected final-hypothesis/claim lineage"
        )

    rows: list[RelationalAtomicIdentityFactorization] = []
    for relation in relation_ir_report.relations:
        source = projected_rows.get(
            (relation.hypothesis_id, relation.claim_id)
        )
        if source is None:
            raise ValueError(
                "relation IR claim absent from relational projection: "
                + relation.claim_id
            )

        candidate = _candidate_card(
            plan=plan,
            final_hypothesis_id=relation.hypothesis_id,
            candidate_hypothesis_id=source.candidate_hypothesis_id,
        )
        rows.append(
            factorize_relational_atomic_identity(
                relation=relation,
                candidate=candidate,
                candidate_hypothesis_id=source.candidate_hypothesis_id,
            )
        )

    body = {
        "schema_version":
            "relational-atomic-identity-factorization-report-v1",
        "source_binding_plan_id": plan.plan_id,
        "source_binding_plan_sha256": plan.plan_sha256,
        "source_relational_projection_report_id": projection_report.report_id,
        "source_relation_ir_report_id": relation_ir_report.report_id,
        "relations": [row.model_dump(mode="json") for row in rows],
        "relation_count": len(rows),
        "ready_count": sum(
            row.factorization_status == "ATOMIC_IDENTITY_READY"
            for row in rows
        ),
        "not_ready_count": sum(
            row.factorization_status != "ATOMIC_IDENTITY_READY"
            for row in rows
        ),
        "source_candidate_artifacts_hash_checked": True,
        "source_exact_identity_required": True,
        "atomic_identity_preserved_without_decomposition": True,
        "synthetic_constituent_decomposition_performed": False,
        "scientific_content_added": False,
        "novelty_assessment_performed": False,
        "verifier_result_observed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalAtomicIdentityFactorizationReport(
        **body,
        report_id=(
            "relational_atomic_identity_factorization:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


def _unique_text(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _normalize_space(raw).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _fixed_terms(relation: ScientificRelationIR) -> list[str]:
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


def compile_relational_atomic_factor_projection_set(
    *,
    relation: ScientificRelationIR,
    factorization: RelationalAtomicIdentityFactorization,
    max_alias_query_variants_per_projection: int = 4,
) -> GroundedFactorProjectionSet:
    if relation.relation_ir_id != factorization.relation_ir_id:
        raise ValueError("relation/factorization relation_ir_id mismatch")
    if relation.claim_id != factorization.claim_id:
        raise ValueError("relation/factorization claim_id mismatch")
    if max_alias_query_variants_per_projection < 1:
        raise ValueError(
            "max_alias_query_variants_per_projection must be >= 1"
        )

    if factorization.factorization_status != "ATOMIC_IDENTITY_READY":
        return GroundedFactorProjectionSet(
            relation_ir_id=relation.relation_ir_id,
            hypothesis_id=relation.hypothesis_id,
            claim_id=relation.claim_id,
            source_identity_term=factorization.source_identity_term,
            factorization_status=factorization.factorization_status,
            grounded_factor_count=0,
            max_lower_order_factor_subset_size=1,
            max_projections_per_relation=2,
            max_alias_query_variants_per_projection=(
                max_alias_query_variants_per_projection
            ),
            projections=[],
            projection_count=0,
            truncated=False,
            planning_ready=False,
            reason_codes=[
                "relational_atomic_identity_not_source_grounded"
            ],
        )

    assert factorization.factor_id is not None
    assert factorization.source_identity_term is not None
    aliases = list(factorization.exact_source_aliases)
    if not aliases:
        raise ValueError("ready atomic identity factorization lacks aliases")

    fixed = _fixed_terms(relation)
    factor_binding = GroundedProjectionFactorBinding(
        factor_id=factorization.factor_id,
        group_label="atomic_branch_identity",
        identity_basis_tokens=list(
            factorization.identity_basis_tokens
        ),
        exclusive_identity_basis_tokens=list(
            factorization.identity_basis_tokens
        ),
        exact_source_aliases=aliases,
    )

    base_query = " ".join(fixed)
    base = GroundedFactorRelationProjection(
        projection_id=_stable_id(
            "grounded_factor_relation_projection",
            relation.relation_ir_id,
            "BASE_RELATION",
            base_query,
        ),
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        projection_kind="BASE_RELATION",
        source_identity_term=factorization.source_identity_term,
        retained_factor_ids=[],
        omitted_factor_ids=[factorization.factor_id],
        factor_order=0,
        factor_bindings=[],
        endpoint_terms=[
            row.surface_text for row in relation.endpoint_concepts
        ],
        scope_terms=[
            row.surface_text for row in relation.scope_qualifiers
        ],
        directional_terms=[
            row.surface_text for row in relation.directional_qualifiers
        ],
        canonical_search_terms=fixed,
        canonical_search_query=base_query,
        exact_source_query_variants=[base_query],
        exact_source_query_variant_count=1,
        relation_typing_status=relation.typing_status,
        factorization_status=factorization.factorization_status,
        eligible_for_future_typed_retrieval=(
            relation.typing_status == "READY"
        ),
        reason_codes=[
            "relational_atomic_identity_projected_away"
        ],
    )

    variants = [
        " ".join(_unique_text([*fixed, alias]))
        for alias in aliases[:max_alias_query_variants_per_projection]
    ]
    variants = _unique_text(variants)
    if not variants:
        raise ValueError("no FULL atomic-identity query variants")

    canonical_terms = _unique_text([*fixed, aliases[0]])
    full = GroundedFactorRelationProjection(
        projection_id=_stable_id(
            "grounded_factor_relation_projection",
            relation.relation_ir_id,
            "FULL_RELATION",
            factorization.factor_id,
            *variants,
        ),
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        projection_kind="FULL_RELATION",
        source_identity_term=factorization.source_identity_term,
        retained_factor_ids=[factorization.factor_id],
        omitted_factor_ids=[],
        factor_order=1,
        factor_bindings=[factor_binding],
        endpoint_terms=[
            row.surface_text for row in relation.endpoint_concepts
        ],
        scope_terms=[
            row.surface_text for row in relation.scope_qualifiers
        ],
        directional_terms=[
            row.surface_text for row in relation.directional_qualifiers
        ],
        canonical_search_terms=canonical_terms,
        canonical_search_query=variants[0],
        exact_source_query_variants=variants,
        exact_source_query_variant_count=len(variants),
        relation_typing_status=relation.typing_status,
        factorization_status=factorization.factorization_status,
        eligible_for_future_typed_retrieval=(
            relation.typing_status == "READY"
        ),
        reason_codes=[
            "relational_atomic_identity_retained_whole"
        ],
    )

    return GroundedFactorProjectionSet(
        relation_ir_id=relation.relation_ir_id,
        hypothesis_id=relation.hypothesis_id,
        claim_id=relation.claim_id,
        source_identity_term=factorization.source_identity_term,
        factorization_status=factorization.factorization_status,
        grounded_factor_count=1,
        max_lower_order_factor_subset_size=1,
        max_projections_per_relation=2,
        max_alias_query_variants_per_projection=(
            max_alias_query_variants_per_projection
        ),
        projections=[base, full],
        projection_count=2,
        truncated=False,
        planning_ready=True,
        reason_codes=[
            "relational_atomic_identity_preserved_as_single_source_grounded_factor",
            "no_synthetic_constituent_decomposition",
            "no_proper_nonempty_lower_order_factor_subset_for_single_identity",
        ],
    )


def compile_relational_atomic_grounded_factor_projection_report(
    *,
    relation_ir_report: ScientificRelationIRReport,
    factorization_report: RelationalAtomicIdentityFactorizationReport,
    max_alias_query_variants_per_projection: int = 4,
) -> GroundedFactorProjectionReport:
    if factorization_report.source_relation_ir_report_id != (
        relation_ir_report.report_id
    ):
        raise ValueError(
            "factorization/relation IR report provenance mismatch"
        )

    factorization_by_relation = {
        row.relation_ir_id: row
        for row in factorization_report.relations
    }
    if len(factorization_by_relation) != len(
        factorization_report.relations
    ):
        raise ValueError("duplicate relation in atomic factorization report")

    relation_ids = {
        row.relation_ir_id for row in relation_ir_report.relations
    }
    if set(factorization_by_relation) != relation_ids:
        raise ValueError(
            "atomic factorization population must exactly match relation IR"
        )

    sets = [
        compile_relational_atomic_factor_projection_set(
            relation=relation,
            factorization=factorization_by_relation[
                relation.relation_ir_id
            ],
            max_alias_query_variants_per_projection=(
                max_alias_query_variants_per_projection
            ),
        )
        for relation in relation_ir_report.relations
    ]

    return GroundedFactorProjectionReport(
        report_id=_stable_id(
            "grounded_factor_projection_report",
            relation_ir_report.report_id,
            factorization_report.report_id,
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
            row.planning_ready for row in sets
        ),
        projection_count=sum(
            row.projection_count for row in sets
        ),
        future_retrieval_eligible_projection_count=sum(
            projection.eligible_for_future_typed_retrieval
            for row in sets
            for projection in row.projections
        ),
        truncated_relation_count=0,
    )


__all__ = [
    "RelationalAtomicIdentityFactorization",
    "RelationalAtomicIdentityFactorizationReport",
    "RelationalAtomicIdentitySourceSpan",
    "build_relational_atomic_identity_factorization_report",
    "compile_relational_atomic_factor_projection_set",
    "compile_relational_atomic_grounded_factor_projection_report",
    "factorize_relational_atomic_identity",
]
