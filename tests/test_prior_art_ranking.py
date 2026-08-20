from __future__ import annotations

import numpy as np

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
    PriorArtWork,
)
from dac_her.prior_art_matching import PriorArtRanker


class FlatEncoder:
    def encode_query(self, text: str):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_documents(self, texts, *, batch_size=32):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_token_coverage_and_reaction_domain_break_semantic_ties() -> None:
    claim = NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text="nitrogen coordination charge donation d band tuning hydrogen evolution reaction",
        rationale="fixture",
        search_concepts=[
            "nitrogen coordination charge donation",
            "d band tuning hydrogen evolution",
        ],
        search_queries=["nitrogen coordination d band HER"],
    )
    query = LiteratureQuery(
        query_id="q1",
        hypothesis_id="h1",
        claim_id="c1",
        query_kind="claim_primary",
        query_text="x",
    )
    plan = LiteratureQueryPlan(
        plan_id="p", plan_sha256="x", source_portfolio_id="portfolio", queries=[query]
    )
    her = PriorArtWork(
        work_id="her",
        title="Nitrogen coordination tunes d-band charge transfer for hydrogen evolution",
        abstract="Nitrogen coordination changes charge donation and HER intermediate binding.",
        retrieval_query_ids=["q1"],
        retrieval_claim_ids=["c1"],
    )
    orr = PriorArtWork(
        work_id="orr",
        title="Nitrogen coordination tunes charge transfer for oxygen reduction",
        abstract="An ORR catalyst with nitrogen coordination and charge transfer.",
        retrieval_query_ids=["q1"],
        retrieval_claim_ids=["c1"],
    )
    packet = PriorArtPacket(
        packet_id="packet",
        packet_sha256="x",
        source_portfolio_id="portfolio",
        source_query_plan_id="p",
        searched_at_utc="2026-08-10T00:00:00+00:00",
        works=[orr, her],
    )
    ranked = PriorArtRanker(FlatEncoder()).rank(claim, packet, plan).ranked_works
    assert ranked[0].work_id == "her"
    assert ranked[0].lexical_coverage > 0.0
    assert ranked[0].reaction_domain_relevance > ranked[1].reaction_domain_relevance
