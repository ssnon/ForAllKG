from __future__ import annotations

import hashlib
import json
from collections import defaultdict
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


class SupportingProjectionQueryBinding(StrictModel):
    projection_id: str
    relation_ir_id: str
    hypothesis_id: str
    claim_id: str
    projection_kind: str
    factor_order: int = Field(ge=0)
    retained_factor_ids: list[str] = Field(default_factory=list)
    retained_factor_labels: list[str] = Field(default_factory=list)
    query_variant_index: int = Field(ge=0)
    query_text: str = Field(min_length=1)

    source_grounded_projection: Literal[True] = True
    supporting_retrieval_only: Literal[True] = True
    novelty_authority: Literal[False] = False


class SupportingProjectionTransportQuery(StrictModel):
    transport_query_id: str
    hypothesis_id: str
    claim_id: str
    query_text: str = Field(min_length=1)

    projection_ids: list[str] = Field(min_length=1)
    projection_kinds: list[str] = Field(min_length=1)
    factor_orders: list[int] = Field(min_length=1)
    binding_count: int = Field(ge=1)

    query_kind_adapter: Literal[
        "claim_variant"
    ] = "claim_variant"

    network_query_deduplicated: Literal[True] = True
    semantic_equivalence_inferred: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_binding_count(
        self,
    ) -> "SupportingProjectionTransportQuery":
        if self.binding_count != len(self.projection_ids):
            raise ValueError(
                "transport query binding_count mismatch"
            )
        return self


class SupportingProjectionQueryPlan(StrictModel):
    schema_version: Literal[
        "supporting-projection-query-plan-v1"
    ] = "supporting-projection-query-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_portfolio_id: str
    source_projection_report_id: str

    bindings: list[SupportingProjectionQueryBinding]
    transport_queries: list[SupportingProjectionTransportQuery]

    projection_binding_count: int = Field(ge=0)
    transport_query_count: int = Field(ge=0)
    deduplicated_network_query_count: int = Field(ge=0)

    epistemic_usage: Literal[
        "supporting_prior_art_retrieval_only_no_relation_judgment"
    ] = "supporting_prior_art_retrieval_only_no_relation_judgment"

    no_relation_adjudication: Literal[True] = True
    no_absence_based_novelty: Literal[True] = True
    no_positive_nonobviousness_authority: Literal[True] = True
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "SupportingProjectionQueryPlan":
        if self.projection_binding_count != len(self.bindings):
            raise ValueError("projection_binding_count mismatch")
        if self.transport_query_count != len(self.transport_queries):
            raise ValueError("transport_query_count mismatch")
        expected_dedup = (
            len(self.bindings) - len(self.transport_queries)
        )
        if self.deduplicated_network_query_count != expected_dedup:
            raise ValueError(
                "deduplicated_network_query_count mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        expected_id = (
            "supporting_projection_query_plan:"
            + expected_sha[:20]
        )
        if observed_sha != expected_sha:
            raise ValueError(
                "supporting projection query plan SHA mismatch"
            )
        if observed_id != expected_id:
            raise ValueError(
                "supporting projection query plan ID mismatch"
            )
        return self


class SupportingProjectionCoverageRow(StrictModel):
    projection_id: str
    hypothesis_id: str
    claim_id: str
    projection_kind: str
    factor_order: int = Field(ge=0)

    transport_query_ids: list[str] = Field(default_factory=list)
    query_count: int = Field(ge=0)
    successful_provider_query_count: int = Field(ge=0)
    failed_provider_query_count: int = Field(ge=0)

    unique_work_count: int = Field(ge=0)
    abstract_work_count: int = Field(ge=0)
    doi_work_count: int = Field(ge=0)

    work_ids: list[str] = Field(default_factory=list)

    retrieval_coverage_only: Literal[True] = True
    relation_status_assigned: Literal[False] = False
    absence_is_novelty: Literal[False] = False


class SupportingProjectionRetrievalReport(StrictModel):
    schema_version: Literal[
        "supporting-projection-retrieval-report-v1"
    ] = "supporting-projection-retrieval-report-v1"

    report_id: str
    source_supporting_query_plan_id: str
    source_transport_query_plan_id: str
    source_prior_art_packet_id: str

    projection_coverage: list[SupportingProjectionCoverageRow]
    projection_count: int = Field(ge=0)
    transport_query_count: int = Field(ge=0)
    successful_provider_query_count: int = Field(ge=0)
    failed_provider_query_count: int = Field(ge=0)
    unique_work_count: int = Field(ge=0)
    abstract_work_count: int = Field(ge=0)

    retrieval_lane: Literal[
        "SUPPORTING_PRIOR_ART"
    ] = "SUPPORTING_PRIOR_ART"

    diagnostic_only: Literal[True] = True
    relation_adjudication_performed: Literal[False] = False
    counterevidence_search_performed: Literal[False] = False
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "SupportingProjectionRetrievalReport":
        if self.projection_count != len(self.projection_coverage):
            raise ValueError("projection_count mismatch")
        return self


def _projection_bindings(
    report: GroundedFactorProjectionReport,
) -> list[SupportingProjectionQueryBinding]:
    rows: list[SupportingProjectionQueryBinding] = []

    for projection_set in report.projection_sets:
        if not projection_set.planning_ready:
            continue

        for projection in projection_set.projections:
            if not projection.eligible_for_future_typed_retrieval:
                continue

            labels = [
                binding.group_label
                for binding in projection.factor_bindings
            ]

            for index, query_text in enumerate(
                projection.exact_source_query_variants
            ):
                rows.append(
                    SupportingProjectionQueryBinding(
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
                        query_variant_index=index,
                        query_text=query_text,
                    )
                )

    return rows


def build_supporting_projection_query_plan(
    *,
    portfolio: HypothesisPortfolio,
    projection_report: GroundedFactorProjectionReport,
) -> SupportingProjectionQueryPlan:
    portfolio_hypothesis_ids = {
        row.hypothesis_id
        for row in portfolio.hypotheses
    }

    bindings = _projection_bindings(
        projection_report
    )

    unknown_hypotheses = sorted(
        {
            row.hypothesis_id
            for row in bindings
        }
        - portfolio_hypothesis_ids
    )
    if unknown_hypotheses:
        raise ValueError(
            "projection query binding references hypotheses absent from "
            "source portfolio: "
            + ", ".join(unknown_hypotheses)
        )

    # Deduplicate only literal transport query identity within the same
    # hypothesis/claim. No scientific-equivalence inference is performed.
    grouped: dict[
        tuple[str, str, str],
        list[SupportingProjectionQueryBinding],
    ] = defaultdict(list)

    for row in bindings:
        key = (
            row.hypothesis_id,
            row.claim_id,
            " ".join(row.query_text.split()).casefold(),
        )
        grouped[key].append(row)

    transport_queries: list[SupportingProjectionTransportQuery] = []

    for (
        hypothesis_id,
        claim_id,
        normalized_query,
    ), rows in grouped.items():
        query_text = rows[0].query_text
        transport_queries.append(
            SupportingProjectionTransportQuery(
                transport_query_id=_stable_id(
                    "supporting_projection_transport_query",
                    portfolio.portfolio_id,
                    hypothesis_id,
                    claim_id,
                    normalized_query,
                ),
                hypothesis_id=hypothesis_id,
                claim_id=claim_id,
                query_text=query_text,
                projection_ids=list(
                    dict.fromkeys(
                        row.projection_id
                        for row in rows
                    )
                ),
                projection_kinds=list(
                    dict.fromkeys(
                        row.projection_kind
                        for row in rows
                    )
                ),
                factor_orders=list(
                    dict.fromkeys(
                        row.factor_order
                        for row in rows
                    )
                ),
                binding_count=len(
                    {
                        row.projection_id
                        for row in rows
                    }
                ),
            )
        )

    transport_queries.sort(
        key=lambda row: (
            row.hypothesis_id,
            row.claim_id,
            row.query_text.casefold(),
        )
    )

    body = {
        "schema_version":
            "supporting-projection-query-plan-v1",
        "source_portfolio_id":
            portfolio.portfolio_id,
        "source_projection_report_id":
            projection_report.report_id,
        "bindings": [
            row.model_dump(mode="json")
            for row in bindings
        ],
        "transport_queries": [
            row.model_dump(mode="json")
            for row in transport_queries
        ],
        "projection_binding_count":
            len(bindings),
        "transport_query_count":
            len(transport_queries),
        "deduplicated_network_query_count":
            len(bindings) - len(transport_queries),
        "epistemic_usage":
            "supporting_prior_art_retrieval_only_no_relation_judgment",
        "no_relation_adjudication":
            True,
        "no_absence_based_novelty":
            True,
        "no_positive_nonobviousness_authority":
            True,
        "production_selection_changed":
            False,
    }
    digest = _sha256_json(body)
    return SupportingProjectionQueryPlan(
        **body,
        plan_id=(
            "supporting_projection_query_plan:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


def build_transport_literature_query_plan(
    plan: SupportingProjectionQueryPlan,
) -> LiteratureQueryPlan:
    queries = [
        LiteratureQuery(
            query_id=row.transport_query_id,
            hypothesis_id=row.hypothesis_id,
            claim_id=row.claim_id,
            # Existing production transport contract is intentionally reused
            # without adding a new query_kind enum to production schemas.
            query_kind="claim_variant",
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
        # Transport retrieval does not require canonical NoveltyClaim rows.
        # Relation/projection provenance remains in the supporting plan.
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


def build_supporting_projection_retrieval_report(
    *,
    supporting_plan: SupportingProjectionQueryPlan,
    transport_plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
) -> SupportingProjectionRetrievalReport:
    if packet.source_query_plan_id != transport_plan.plan_id:
        raise ValueError(
            "supporting projection packet/query plan mismatch"
        )
    if packet.source_portfolio_id != supporting_plan.source_portfolio_id:
        raise ValueError(
            "supporting projection packet/source portfolio mismatch"
        )

    transport_by_projection: dict[str, set[str]] = defaultdict(set)
    binding_by_projection: dict[
        str,
        SupportingProjectionQueryBinding,
    ] = {}

    for row in supporting_plan.bindings:
        binding_by_projection.setdefault(
            row.projection_id,
            row,
        )

    for query in supporting_plan.transport_queries:
        for projection_id in query.projection_ids:
            transport_by_projection[projection_id].add(
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

    coverage_rows: list[SupportingProjectionCoverageRow] = []

    for projection_id in sorted(binding_by_projection):
        binding = binding_by_projection[projection_id]
        query_ids = sorted(
            transport_by_projection.get(
                projection_id,
                set(),
            )
        )
        query_id_set = set(query_ids)

        works = [
            work
            for work in packet.works
            if query_id_set
            & set(work.retrieval_query_ids)
        ]
        work_ids = sorted(
            {
                work.work_id
                for work in works
            }
        )

        coverage_rows.append(
            SupportingProjectionCoverageRow(
                projection_id=projection_id,
                hypothesis_id=binding.hypothesis_id,
                claim_id=binding.claim_id,
                projection_kind=binding.projection_kind,
                factor_order=binding.factor_order,
                transport_query_ids=query_ids,
                query_count=len(query_ids),
                successful_provider_query_count=sum(
                    query_id in query_id_set
                    for query_id, _provider
                    in success_pairs
                ),
                failed_provider_query_count=sum(
                    query_id in query_id_set
                    for query_id, _provider
                    in failed_pairs
                ),
                unique_work_count=len(work_ids),
                abstract_work_count=sum(
                    bool(work.abstract)
                    for work in works
                ),
                doi_work_count=sum(
                    bool(work.doi)
                    for work in works
                ),
                work_ids=work_ids,
            )
        )

    unique_work_ids = {
        work.work_id
        for work in packet.works
    }

    return SupportingProjectionRetrievalReport(
        report_id=_stable_id(
            "supporting_projection_retrieval_report",
            supporting_plan.plan_id,
            transport_plan.plan_id,
            packet.packet_id,
            *[
                (
                    row.projection_id,
                    row.unique_work_count,
                    row.abstract_work_count,
                )
                for row in coverage_rows
            ],
        ),
        source_supporting_query_plan_id=supporting_plan.plan_id,
        source_transport_query_plan_id=transport_plan.plan_id,
        source_prior_art_packet_id=packet.packet_id,
        projection_coverage=coverage_rows,
        projection_count=len(coverage_rows),
        transport_query_count=len(
            supporting_plan.transport_queries
        ),
        successful_provider_query_count=sum(
            row.success
            for row in packet.executions
        ),
        failed_provider_query_count=sum(
            not row.success
            for row in packet.executions
        ),
        unique_work_count=len(unique_work_ids),
        abstract_work_count=sum(
            bool(work.abstract)
            for work in packet.works
        ),
    )


__all__ = [
    "SupportingProjectionQueryPlan",
    "SupportingProjectionRetrievalReport",
    "build_supporting_projection_query_plan",
    "build_supporting_projection_retrieval_report",
    "build_transport_literature_query_plan",
]
