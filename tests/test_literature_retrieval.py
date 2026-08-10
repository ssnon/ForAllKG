from __future__ import annotations

from dac_her.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
    PriorArtPacket,
    PriorArtWork,
)
from dac_her.literature_retrieval import (
    LiteratureRetriever,
    canonicalize_prior_art_packet,
)


class ProviderA:
    provider_name = "a"

    def search(self, query, *, limit):
        return [
            PriorArtWork(
                work_id="wa",
                title="Charge Transfer Drives Hydrogen Adsorption at a Metal Interface",
                doi="10.1/example",
                abstract="short abstract",
                providers=["a"],
                provider_ids={"a": "1"},
                retrieval_query_ids=[query.query_id],
                retrieval_claim_ids=[query.claim_id] if query.claim_id else [],
            )
        ]


class ProviderB:
    provider_name = "b"

    def search(self, query, *, limit):
        return [
            PriorArtWork(
                work_id="wb",
                title="Charge Transfer Drives Hydrogen Adsorption at a Metal Interface",
                doi="10.1/example.s001",
                abstract="a much longer abstract with more metadata",
                providers=["b"],
                provider_ids={"b": "2"},
                retrieval_query_ids=[query.query_id],
                retrieval_claim_ids=[query.claim_id] if query.claim_id else [],
            ),
            PriorArtWork(
                work_id="wc",
                title="Charge Transfer Drives Hydrogen Adsorption at a Metal Interface",
                doi=None,
                abstract=None,
                providers=["b"],
                provider_ids={"b": "3"},
                retrieval_query_ids=[query.query_id],
                retrieval_claim_ids=[query.claim_id] if query.claim_id else [],
            ),
        ]


def _plan() -> LiteratureQueryPlan:
    query = LiteratureQuery(
        query_id="q1",
        hypothesis_id="h1",
        claim_id="c1",
        query_kind="claim_primary",
        query_text="test query",
    )
    return LiteratureQueryPlan(
        plan_id="plan",
        plan_sha256="x",
        source_portfolio_id="portfolio",
        queries=[query],
        claims=[],
    )


def test_provider_results_collapse_main_si_and_exact_title_duplicates() -> None:
    packet = LiteratureRetriever([ProviderA(), ProviderB()]).retrieve(_plan()).packet
    assert packet.raw_work_count == 3
    assert packet.canonical_work_count == 1
    assert packet.deduplicated_work_count == 2
    assert packet.supplementary_records_collapsed == 1
    assert len(packet.works) == 1
    assert packet.works[0].providers == ["a", "b"]
    assert packet.works[0].doi == "10.1/example"
    assert packet.works[0].abstract.startswith("a much longer")


def test_alpha5_packet_can_be_recanonicalized_without_network() -> None:
    packet = PriorArtPacket(
        packet_id="old",
        packet_sha256="x",
        source_portfolio_id="portfolio",
        source_query_plan_id="plan",
        searched_at_utc="2026-08-10T00:00:00+00:00",
        providers_requested=["fixture"],
        works=[
            PriorArtWork(work_id="a", title="A sufficiently long duplicated paper title", doi="10.2/x.s001"),
            PriorArtWork(work_id="b", title="A sufficiently long duplicated paper title", doi="10.2/x.s002"),
        ],
    )
    updated = canonicalize_prior_art_packet(packet)
    assert len(updated.works) == 1
    assert updated.raw_work_count == 2
    assert updated.deduplicated_work_count == 1
    assert updated.supplementary_records_collapsed == 1
