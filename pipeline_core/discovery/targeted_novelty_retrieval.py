from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureRetriever,
    canonicalize_prior_art_works,
)
from pipeline_core.discovery.novelty_refinement_contracts import NoveltyGap


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _targeted_query_kind(
    target: TargetedGapQuery,
) -> str:
    """Map targeted semantic role into audit-visible query provenance."""

    if (
        target.query_role
        == "exact_higher_order_verification"
    ):
        return "claim_exact_verification"

    return "claim_variant"


def build_augmented_query_plan(
    base: LiteratureQueryPlan,
    gap: NoveltyGap,
) -> tuple[LiteratureQueryPlan, LiteratureQueryPlan]:
    """Return (augmented full plan, delta-only plan).

    The delta plan is only for network retrieval. The augmented plan is the
    audit/reassessment plan and keeps original claim provenance intact.
    """
    new_queries = []
    for index, target in enumerate(gap.targeted_queries, start=1):
        new_queries.append(
            LiteratureQuery(
                query_id=_stable_id(
                    "literature_query",
                    base.plan_id,
                    gap.gap_id,
                    index,
                    target.claim_id,
                    target.query_role,
                    target.query_text,
                ),
                hypothesis_id=gap.hypothesis_id,
                claim_id=target.claim_id,
                query_kind=_targeted_query_kind(
                    target
                ),
                query_text=target.query_text,
            )
        )

    all_queries = list(base.queries) + new_queries
    full_id = _stable_id(
        "literature_query_plan",
        base.source_portfolio_id,
        base.plan_id,
        gap.gap_id,
        *(x.query_id for x in new_queries),
    )
    full_body = {
        "schema_version": "literature-query-plan-v1",
        "plan_id": full_id,
        "source_portfolio_id": base.source_portfolio_id,
        "queries": [x.model_dump(mode="json") for x in all_queries],
        "claims": [x.model_dump(mode="json") for x in base.claims],
        "policy_version": "external-novelty-query-policy-v1",
    }
    full = LiteratureQueryPlan(**full_body, plan_sha256=_sha256_json(full_body))

    delta_id = _stable_id(
        "literature_query_plan",
        base.source_portfolio_id,
        gap.gap_id,
        "delta",
        *(x.query_id for x in new_queries),
    )
    delta_body = {
        "schema_version": "literature-query-plan-v1",
        "plan_id": delta_id,
        "source_portfolio_id": base.source_portfolio_id,
        "queries": [x.model_dump(mode="json") for x in new_queries],
        "claims": [x.model_dump(mode="json") for x in base.claims],
        "policy_version": "external-novelty-query-policy-v1",
    }
    delta = LiteratureQueryPlan(**delta_body, plan_sha256=_sha256_json(delta_body))
    return full, delta


def merge_prior_art_packets(
    base: PriorArtPacket,
    delta: PriorArtPacket,
    augmented_plan: LiteratureQueryPlan,
) -> PriorArtPacket:
    """Merge old + targeted evidence using the shared canonical identity seam."""
    merged_input = list(base.works) + list(delta.works)
    canonical, cross_packet_supplementary_collapsed = (
        canonicalize_prior_art_works(merged_input)
    )
    works = sorted(
        canonical,
        key=lambda x: (-(x.citation_count or 0), -(x.year or 0), x.title.lower()),
    )
    executions: list[QueryExecution] = list(base.executions) + list(delta.executions)
    raw_count = (
        (base.raw_work_count or len(base.works))
        + (delta.raw_work_count or len(delta.works))
    )
    packet_id = _stable_id(
        "prior_art_packet",
        augmented_plan.plan_id,
        base.packet_id,
        delta.packet_id,
        *(x.work_id for x in works),
    )
    body = {
        "schema_version": "prior-art-packet-v1",
        "packet_id": packet_id,
        "source_portfolio_id": augmented_plan.source_portfolio_id,
        "source_query_plan_id": augmented_plan.plan_id,
        "searched_at_utc": delta.searched_at_utc,
        "providers_requested": sorted(
            set(base.providers_requested) | set(delta.providers_requested)
        ),
        "works": [x.model_dump(mode="json") for x in works],
        "executions": [x.model_dump(mode="json") for x in executions],
        "raw_work_count": raw_count,
        "canonical_work_count": len(works),
        "deduplicated_work_count": max(0, raw_count - len(works)),
        "supplementary_records_collapsed": (
            base.supplementary_records_collapsed
            + delta.supplementary_records_collapsed
            + cross_packet_supplementary_collapsed
        ),
        "epistemic_usage": "prior_art_only_not_positive_premise",
    }
    return PriorArtPacket(**body, packet_sha256=_sha256_json(body))


@dataclass(frozen=True)
class TargetedRetrievalOutcome:
    augmented_plan: LiteratureQueryPlan
    delta_plan: LiteratureQueryPlan
    delta_packet: PriorArtPacket
    merged_packet: PriorArtPacket


class TargetedNoveltyRetriever:
    def __init__(self, retriever: LiteratureRetriever) -> None:
        self.retriever = retriever

    def retrieve(
        self,
        base_plan: LiteratureQueryPlan,
        base_packet: PriorArtPacket,
        gap: NoveltyGap,
    ) -> TargetedRetrievalOutcome:
        augmented, delta_plan = build_augmented_query_plan(base_plan, gap)
        delta = self.retriever.retrieve(delta_plan).packet
        # Delta packet references delta_plan; merged packet is deliberately
        # rebound to the augmented plan for deterministic reassessment.
        merged = merge_prior_art_packets(base_packet, delta, augmented)
        return TargetedRetrievalOutcome(
            augmented_plan=augmented,
            delta_plan=delta_plan,
            delta_packet=delta,
            merged_packet=merged,
        )
