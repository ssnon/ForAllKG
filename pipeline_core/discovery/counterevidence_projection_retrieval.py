from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from itertools import product
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
    GroundedFactorRelationProjection,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CounterevidenceMode = Literal[
    "NULL_RELATION",
    "INDEPENDENCE",
    "DECOUPLING",
    "OPPOSITE_DIRECTION",
    "DISTRIBUTIONAL_DIVERGENCE",
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


def _normalize(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−-]+", " ", text)
    text = re.sub(r"[^\w\s+*/().]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _unique_text(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(str(raw or "").split()).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


_DIRECTION_INVERSION = {
    "lower": "higher",
    "higher": "lower",
    "decrease": "increase",
    "decreases": "increases",
    "decreased": "increased",
    "decreasing": "increasing",
    "increase": "decrease",
    "increases": "decreases",
    "increased": "decreased",
    "increasing": "decreasing",
    "negative correlation": "positive correlation",
    "positive correlation": "negative correlation",
    "negatively correlated": "positively correlated",
    "positively correlated": "negatively correlated",
    "inverse relationship": "positive relationship",
    "positive relationship": "inverse relationship",
}


_DISTRIBUTIONAL_TOKENS = frozenset(
    {
        "heterogeneity",
        "heterogeneous",
        "homogeneity",
        "homogeneous",
        "uniformity",
        "uniform",
        "variance",
        "variability",
        "variation",
    }
)


_MODE_TERMS: dict[
    CounterevidenceMode,
    tuple[str, ...],
] = {
    "NULL_RELATION": (
        "no association",
        "no correlation",
    ),
    "INDEPENDENCE": (
        "independent",
        "independence",
    ),
    "DECOUPLING": (
        "decoupled",
        "decoupling",
    ),
    # OPPOSITE_DIRECTION is generated from explicit directional terms.
    "OPPOSITE_DIRECTION": (),
    "DISTRIBUTIONAL_DIVERGENCE": (
        "homogeneous heterogeneous",
        "uniform heterogeneous",
    ),
}


class CounterevidenceQueryBinding(StrictModel):
    query_binding_id: str
    projection_id: str
    relation_ir_id: str
    hypothesis_id: str
    claim_id: str

    projection_kind: str
    factor_order: int = Field(ge=0)
    retained_factor_ids: list[str] = Field(default_factory=list)
    retained_factor_labels: list[str] = Field(default_factory=list)

    counterevidence_mode: CounterevidenceMode
    source_alias_variant_index: int = Field(ge=0)
    query_variant_index: int = Field(ge=0)

    query_text: str = Field(min_length=1)
    fixed_scientific_terms: list[str] = Field(min_length=2)
    counterevidence_operator_terms: list[str] = Field(min_length=1)

    search_intent_only: Literal[True] = True
    scientific_counterevidence_established: Literal[False] = False
    novelty_authority: Literal[False] = False


class CounterevidenceTransportQuery(StrictModel):
    transport_query_id: str
    hypothesis_id: str
    claim_id: str
    query_text: str = Field(min_length=1)

    projection_ids: list[str] = Field(min_length=1)
    counterevidence_modes: list[CounterevidenceMode] = Field(min_length=1)
    binding_ids: list[str] = Field(min_length=1)

    query_kind_adapter: Literal[
        "claim_diagnostic"
    ] = "claim_diagnostic"

    literal_transport_dedup_only: Literal[True] = True
    semantic_equivalence_inferred: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_bindings(self) -> "CounterevidenceTransportQuery":
        if len(self.binding_ids) < len(self.projection_ids):
            raise ValueError(
                "counterevidence transport has fewer binding IDs "
                "than projection IDs"
            )
        return self


class CounterevidenceProjectionQueryPlan(StrictModel):
    schema_version: Literal[
        "counterevidence-projection-query-plan-v1"
    ] = "counterevidence-projection-query-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_portfolio_id: str
    source_projection_report_id: str

    max_lower_order_factor_order: int = Field(ge=0)
    max_source_alias_variants_per_projection: int = Field(ge=1)

    bindings: list[CounterevidenceQueryBinding]
    transport_queries: list[CounterevidenceTransportQuery]

    selected_projection_count: int = Field(ge=0)
    counterevidence_binding_count: int = Field(ge=0)
    transport_query_count: int = Field(ge=0)
    deduplicated_network_query_count: int = Field(ge=0)

    modes_present: list[CounterevidenceMode] = Field(default_factory=list)

    retrieval_intent: Literal[
        "bridge_breaking_or_directionally_conflicting_prior_art"
    ] = "bridge_breaking_or_directionally_conflicting_prior_art"

    relation_adjudication_authorized: Literal[False] = False
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "CounterevidenceProjectionQueryPlan":
        if self.counterevidence_binding_count != len(self.bindings):
            raise ValueError("counterevidence_binding_count mismatch")
        if self.transport_query_count != len(self.transport_queries):
            raise ValueError("transport_query_count mismatch")
        expected_dedup = (
            len(self.bindings) - len(self.transport_queries)
        )
        if self.deduplicated_network_query_count != expected_dedup:
            raise ValueError("deduplicated_network_query_count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        expected_id = (
            "counterevidence_projection_query_plan:"
            + expected_sha[:20]
        )
        if observed_sha != expected_sha:
            raise ValueError(
                "counterevidence projection query plan SHA mismatch"
            )
        if observed_id != expected_id:
            raise ValueError(
                "counterevidence projection query plan ID mismatch"
            )
        return self


class CounterevidenceCoverageRow(StrictModel):
    projection_id: str
    hypothesis_id: str
    claim_id: str
    projection_kind: str
    factor_order: int = Field(ge=0)
    counterevidence_mode: CounterevidenceMode

    transport_query_ids: list[str] = Field(default_factory=list)
    query_count: int = Field(ge=0)
    successful_provider_query_count: int = Field(ge=0)
    failed_provider_query_count: int = Field(ge=0)

    unique_work_count: int = Field(ge=0)
    abstract_work_count: int = Field(ge=0)
    work_ids: list[str] = Field(default_factory=list)

    retrieval_coverage_only: Literal[True] = True
    counterevidence_relationship_assigned: Literal[False] = False
    absence_is_novelty: Literal[False] = False


class CounterevidenceProjectionRetrievalReport(StrictModel):
    schema_version: Literal[
        "counterevidence-projection-retrieval-report-v1"
    ] = "counterevidence-projection-retrieval-report-v1"

    report_id: str
    source_counterevidence_query_plan_id: str
    source_transport_query_plan_id: str
    source_prior_art_packet_id: str

    projection_mode_coverage: list[CounterevidenceCoverageRow]
    coverage_row_count: int = Field(ge=0)
    selected_projection_count: int = Field(ge=0)
    transport_query_count: int = Field(ge=0)

    successful_provider_query_count: int = Field(ge=0)
    failed_provider_query_count: int = Field(ge=0)
    unique_work_count: int = Field(ge=0)
    abstract_work_count: int = Field(ge=0)

    retrieval_lane: Literal[
        "COUNTEREVIDENCE_PRIOR_ART"
    ] = "COUNTEREVIDENCE_PRIOR_ART"

    diagnostic_only: Literal[True] = True
    relation_adjudication_performed: Literal[False] = False
    counterevidence_relationship_assigned: Literal[False] = False
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "CounterevidenceProjectionRetrievalReport":
        if self.coverage_row_count != len(self.projection_mode_coverage):
            raise ValueError("coverage_row_count mismatch")
        return self


def _projection_selected(
    projection: GroundedFactorRelationProjection,
    *,
    max_lower_order_factor_order: int,
) -> bool:
    if projection.projection_kind in {
        "BASE_RELATION",
        "FULL_RELATION",
    }:
        return True
    return (
        projection.projection_kind
        == "LOWER_ORDER_FACTOR_SUBSET"
        and projection.factor_order
        <= max_lower_order_factor_order
    )


def _factor_alias_variants(
    projection: GroundedFactorRelationProjection,
    *,
    max_variants: int,
) -> list[list[str]]:
    if not projection.factor_bindings:
        return [[]]

    groups = [
        binding.exact_source_aliases
        for binding in projection.factor_bindings
    ]

    variants: list[list[str]] = []
    for values in product(*groups):
        variants.append(_unique_text(list(values)))
        if len(variants) >= max_variants:
            break

    return variants or [[]]


def _fixed_scientific_terms(
    projection: GroundedFactorRelationProjection,
    aliases: list[str],
) -> list[str]:
    # Deliberately exclude the original directional qualifiers here:
    # null/independence/decoupling search should not simultaneously assert
    # the original direction. Scope qualifiers are preserved.
    return _unique_text(
        [
            *projection.endpoint_terms,
            *projection.scope_terms,
            *aliases,
        ]
    )


def _opposite_direction_terms(
    projection: GroundedFactorRelationProjection,
) -> list[str]:
    output: list[str] = []

    for term in projection.directional_terms:
        normalized = _normalize(term)

        # Exact whole-phrase inversion first.
        direct = _DIRECTION_INVERSION.get(normalized)
        if direct:
            output.append(direct)
            continue

        # Conservative token/phrase replacement inside an explicit
        # directional qualifier. Only known relation operators are changed.
        replaced = normalized
        changed = False
        for source in sorted(
            _DIRECTION_INVERSION,
            key=len,
            reverse=True,
        ):
            if source in replaced:
                replaced = replaced.replace(
                    source,
                    _DIRECTION_INVERSION[source],
                )
                changed = True
                break

        if changed and replaced:
            output.append(replaced)

    return _unique_text(output)


def _has_distributional_endpoint(
    projection: GroundedFactorRelationProjection,
) -> bool:
    tokens = {
        token
        for term in projection.endpoint_terms
        for token in _normalize(term).split()
    }
    return bool(tokens & _DISTRIBUTIONAL_TOKENS)


def _binding_rows_for_projection(
    projection: GroundedFactorRelationProjection,
    *,
    max_source_alias_variants: int,
) -> list[CounterevidenceQueryBinding]:
    alias_variants = _factor_alias_variants(
        projection,
        max_variants=max_source_alias_variants,
    )

    modes: list[CounterevidenceMode] = [
        "NULL_RELATION",
        "INDEPENDENCE",
        "DECOUPLING",
    ]

    opposite = _opposite_direction_terms(projection)
    if opposite:
        modes.append("OPPOSITE_DIRECTION")

    if _has_distributional_endpoint(projection):
        modes.append("DISTRIBUTIONAL_DIVERGENCE")

    labels = [
        binding.group_label
        for binding in projection.factor_bindings
    ]

    rows: list[CounterevidenceQueryBinding] = []

    for source_index, aliases in enumerate(alias_variants):
        fixed = _fixed_scientific_terms(
            projection,
            aliases,
        )

        for mode in modes:
            if mode == "OPPOSITE_DIRECTION":
                operator_variants = [opposite]
            else:
                operator_variants = [
                    [term]
                    for term in _MODE_TERMS[mode]
                ]

            for query_index, operator_terms in enumerate(
                operator_variants
            ):
                query_terms = _unique_text(
                    [
                        *fixed,
                        *operator_terms,
                    ]
                )
                query_text = " ".join(query_terms)
                if not query_text:
                    continue

                rows.append(
                    CounterevidenceQueryBinding(
                        query_binding_id=_stable_id(
                            "counterevidence_query_binding",
                            projection.projection_id,
                            source_index,
                            mode,
                            query_index,
                            query_text,
                        ),
                        projection_id=projection.projection_id,
                        relation_ir_id=projection.relation_ir_id,
                        hypothesis_id=projection.hypothesis_id,
                        claim_id=projection.claim_id,
                        projection_kind=projection.projection_kind,
                        factor_order=projection.factor_order,
                        retained_factor_ids=list(
                            projection.retained_factor_ids
                        ),
                        retained_factor_labels=labels,
                        counterevidence_mode=mode,
                        source_alias_variant_index=source_index,
                        query_variant_index=query_index,
                        query_text=query_text,
                        fixed_scientific_terms=fixed,
                        counterevidence_operator_terms=operator_terms,
                    )
                )

    return rows


def build_counterevidence_projection_query_plan(
    *,
    portfolio: HypothesisPortfolio,
    projection_report: GroundedFactorProjectionReport,
    max_lower_order_factor_order: int = 1,
    max_source_alias_variants_per_projection: int = 2,
) -> CounterevidenceProjectionQueryPlan:
    if max_lower_order_factor_order < 0:
        raise ValueError(
            "max_lower_order_factor_order must be >= 0"
        )
    if max_source_alias_variants_per_projection < 1:
        raise ValueError(
            "max_source_alias_variants_per_projection must be >= 1"
        )

    hypothesis_ids = {
        row.hypothesis_id
        for row in portfolio.hypotheses
    }

    selected: list[GroundedFactorRelationProjection] = []
    for projection_set in projection_report.projection_sets:
        if not projection_set.planning_ready:
            continue
        for projection in projection_set.projections:
            if not projection.eligible_for_future_typed_retrieval:
                continue
            if _projection_selected(
                projection,
                max_lower_order_factor_order=(
                    max_lower_order_factor_order
                ),
            ):
                selected.append(projection)

    unknown = sorted(
        {
            row.hypothesis_id
            for row in selected
        }
        - hypothesis_ids
    )
    if unknown:
        raise ValueError(
            "counterevidence projection references unknown hypotheses: "
            + ", ".join(unknown)
        )

    bindings = [
        binding
        for projection in selected
        for binding in _binding_rows_for_projection(
            projection,
            max_source_alias_variants=(
                max_source_alias_variants_per_projection
            ),
        )
    ]

    # Only literal normalized transport query equality is deduplicated.
    grouped: dict[
        tuple[str, str, str],
        list[CounterevidenceQueryBinding],
    ] = defaultdict(list)

    for row in bindings:
        key = (
            row.hypothesis_id,
            row.claim_id,
            _normalize(row.query_text),
        )
        grouped[key].append(row)

    transport: list[CounterevidenceTransportQuery] = []

    for (
        hypothesis_id,
        claim_id,
        normalized_query,
    ), rows in grouped.items():
        transport.append(
            CounterevidenceTransportQuery(
                transport_query_id=_stable_id(
                    "counterevidence_transport_query",
                    portfolio.portfolio_id,
                    hypothesis_id,
                    claim_id,
                    normalized_query,
                ),
                hypothesis_id=hypothesis_id,
                claim_id=claim_id,
                query_text=rows[0].query_text,
                projection_ids=list(
                    dict.fromkeys(
                        row.projection_id
                        for row in rows
                    )
                ),
                counterevidence_modes=list(
                    dict.fromkeys(
                        row.counterevidence_mode
                        for row in rows
                    )
                ),
                binding_ids=[
                    row.query_binding_id
                    for row in rows
                ],
            )
        )

    transport.sort(
        key=lambda row: (
            row.hypothesis_id,
            row.claim_id,
            row.query_text.casefold(),
        )
    )

    body = {
        "schema_version":
            "counterevidence-projection-query-plan-v1",
        "source_portfolio_id":
            portfolio.portfolio_id,
        "source_projection_report_id":
            projection_report.report_id,
        "max_lower_order_factor_order":
            max_lower_order_factor_order,
        "max_source_alias_variants_per_projection":
            max_source_alias_variants_per_projection,
        "bindings": [
            row.model_dump(mode="json")
            for row in bindings
        ],
        "transport_queries": [
            row.model_dump(mode="json")
            for row in transport
        ],
        "selected_projection_count":
            len(selected),
        "counterevidence_binding_count":
            len(bindings),
        "transport_query_count":
            len(transport),
        "deduplicated_network_query_count":
            len(bindings) - len(transport),
        "modes_present": list(
            dict.fromkeys(
                row.counterevidence_mode
                for row in bindings
            )
        ),
        "retrieval_intent":
            "bridge_breaking_or_directionally_conflicting_prior_art",
        "relation_adjudication_authorized":
            False,
        "absence_based_novelty_authorized":
            False,
        "positive_nonobviousness_authority_created":
            False,
        "production_selection_changed":
            False,
    }
    digest = _sha256_json(body)

    return CounterevidenceProjectionQueryPlan(
        **body,
        plan_id=(
            "counterevidence_projection_query_plan:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


def build_counterevidence_transport_plan(
    plan: CounterevidenceProjectionQueryPlan,
) -> LiteratureQueryPlan:
    queries = [
        LiteratureQuery(
            query_id=row.transport_query_id,
            hypothesis_id=row.hypothesis_id,
            claim_id=row.claim_id,
            query_kind="claim_diagnostic",
            query_text=row.query_text,
        )
        for row in plan.transport_queries
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


def build_counterevidence_retrieval_report(
    *,
    counterevidence_plan: CounterevidenceProjectionQueryPlan,
    transport_plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
) -> CounterevidenceProjectionRetrievalReport:
    if packet.source_query_plan_id != transport_plan.plan_id:
        raise ValueError(
            "counterevidence packet/query plan mismatch"
        )
    if packet.source_portfolio_id != counterevidence_plan.source_portfolio_id:
        raise ValueError(
            "counterevidence packet/source portfolio mismatch"
        )

    transport_ids_by_binding: dict[str, set[str]] = defaultdict(set)

    for query in counterevidence_plan.transport_queries:
        for binding_id in query.binding_ids:
            transport_ids_by_binding[binding_id].add(
                query.transport_query_id
            )

    success_pairs = {
        (row.query_id, row.provider)
        for row in packet.executions
        if row.success
    }
    failed_pairs = {
        (row.query_id, row.provider)
        for row in packet.executions
        if not row.success
    }

    grouped_bindings: dict[
        tuple[str, CounterevidenceMode],
        list[CounterevidenceQueryBinding],
    ] = defaultdict(list)

    for row in counterevidence_plan.bindings:
        grouped_bindings[
            (
                row.projection_id,
                row.counterevidence_mode,
            )
        ].append(row)

    coverage: list[CounterevidenceCoverageRow] = []

    for (
        projection_id,
        mode,
    ), bindings in grouped_bindings.items():
        first = bindings[0]
        query_ids = sorted(
            {
                query_id
                for binding in bindings
                for query_id in transport_ids_by_binding.get(
                    binding.query_binding_id,
                    set(),
                )
            }
        )
        query_set = set(query_ids)

        works = [
            work
            for work in packet.works
            if query_set
            & set(work.retrieval_query_ids)
        ]
        work_ids = sorted(
            {
                work.work_id
                for work in works
            }
        )

        coverage.append(
            CounterevidenceCoverageRow(
                projection_id=projection_id,
                hypothesis_id=first.hypothesis_id,
                claim_id=first.claim_id,
                projection_kind=first.projection_kind,
                factor_order=first.factor_order,
                counterevidence_mode=mode,
                transport_query_ids=query_ids,
                query_count=len(query_ids),
                successful_provider_query_count=sum(
                    query_id in query_set
                    for query_id, _provider in success_pairs
                ),
                failed_provider_query_count=sum(
                    query_id in query_set
                    for query_id, _provider in failed_pairs
                ),
                unique_work_count=len(work_ids),
                abstract_work_count=sum(
                    bool(work.abstract)
                    for work in works
                ),
                work_ids=work_ids,
            )
        )

    coverage.sort(
        key=lambda row: (
            row.claim_id,
            row.factor_order,
            row.projection_kind,
            row.counterevidence_mode,
            row.projection_id,
        )
    )

    return CounterevidenceProjectionRetrievalReport(
        report_id=_stable_id(
            "counterevidence_projection_retrieval_report",
            counterevidence_plan.plan_id,
            transport_plan.plan_id,
            packet.packet_id,
        ),
        source_counterevidence_query_plan_id=counterevidence_plan.plan_id,
        source_transport_query_plan_id=transport_plan.plan_id,
        source_prior_art_packet_id=packet.packet_id,
        projection_mode_coverage=coverage,
        coverage_row_count=len(coverage),
        selected_projection_count=(
            counterevidence_plan.selected_projection_count
        ),
        transport_query_count=(
            counterevidence_plan.transport_query_count
        ),
        successful_provider_query_count=sum(
            row.success
            for row in packet.executions
        ),
        failed_provider_query_count=sum(
            not row.success
            for row in packet.executions
        ),
        unique_work_count=len(
            {
                work.work_id
                for work in packet.works
            }
        ),
        abstract_work_count=sum(
            bool(work.abstract)
            for work in packet.works
        ),
    )


__all__ = [
    "CounterevidenceProjectionQueryPlan",
    "CounterevidenceProjectionRetrievalReport",
    "build_counterevidence_projection_query_plan",
    "build_counterevidence_retrieval_report",
    "build_counterevidence_transport_plan",
]
