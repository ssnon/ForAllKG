from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from dac_her.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
)
from dac_her.literature_retrieval import LiteratureRetriever
from dac_her.novelty_refinement_contracts import NoveltyGap


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


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
                query_kind="claim_variant",
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


def _work_key(work: PriorArtWork) -> str:
    doi = (work.doi or "").lower().strip()
    if doi:
        if doi.startswith("https://doi.org/"):
            doi = doi[16:]
        return "doi:" + doi
    return "title:" + " ".join(work.title.lower().split())


def _merge_work(left: PriorArtWork, right: PriorArtWork) -> PriorArtWork:
    abstracts = [x for x in [left.abstract, right.abstract] if x]
    abstract = max(abstracts, key=len) if abstracts else None
    return PriorArtWork(
        work_id=left.work_id,
        title=left.title if len(left.title) >= len(right.title) else right.title,
        year=left.year if left.year is not None else right.year,
        publication_date=left.publication_date or right.publication_date,
        doi=left.doi or right.doi,
        url=left.url or right.url,
        open_access_url=left.open_access_url or right.open_access_url,
        abstract=abstract,
        authors=sorted(set(left.authors) | set(right.authors)),
        venue=left.venue or right.venue,
        citation_count=max(
            [x for x in [left.citation_count, right.citation_count] if x is not None],
            default=None,
        ),
        providers=sorted(set(left.providers) | set(right.providers)),
        provider_ids={**left.provider_ids, **right.provider_ids},
        retrieval_query_ids=sorted(
            set(left.retrieval_query_ids) | set(right.retrieval_query_ids)
        ),
        retrieval_claim_ids=sorted(
            set(left.retrieval_claim_ids) | set(right.retrieval_claim_ids)
        ),
    )


def merge_prior_art_packets(
    base: PriorArtPacket,
    delta: PriorArtPacket,
    augmented_plan: LiteratureQueryPlan,
) -> PriorArtPacket:
    """Merge old + targeted search evidence under the augmented query plan."""
    index: dict[str, PriorArtWork] = {}
    for work in list(base.works) + list(delta.works):
        key = _work_key(work)
        index[key] = _merge_work(index[key], work) if key in index else work
    works = sorted(
        index.values(),
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
