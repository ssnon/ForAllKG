from __future__ import annotations

from dac_her.external_novelty_contracts import PriorArtPacket, PriorArtWork
from dac_her.literature_retrieval import canonicalize_prior_art_packet
from dac_her.canonicalization_doi_conflict_recheck import sha256_json


def _work(
    work_id: str,
    title: str,
    doi: str | None,
    provider: str,
) -> PriorArtWork:
    return PriorArtWork(
        work_id=work_id,
        title=title,
        doi=doi,
        providers=[provider],
        provider_ids={provider: work_id},
        retrieval_query_ids=["q1"],
        retrieval_claim_ids=["c1"],
    )


def _packet(works: list[PriorArtWork]) -> PriorArtPacket:
    body = {
        "schema_version": "prior-art-packet-v1",
        "packet_id": "raw:test",
        "source_portfolio_id": "portfolio:test",
        "source_query_plan_id": "plan:test",
        "searched_at_utc": "2026-08-16T00:00:00+00:00",
        "providers_requested": ["openalex", "crossref"],
        "works": [w.model_dump(mode="json") for w in works],
        "executions": [],
        "raw_work_count": len(works),
        "canonical_work_count": len(works),
        "deduplicated_work_count": 0,
        "supplementary_records_collapsed": 0,
        "epistemic_usage": "prior_art_only_not_positive_premise",
    }
    return PriorArtPacket(
        **body,
        packet_sha256=sha256_json(body),
    )


def test_distinct_doi_families_same_title_do_not_merge():
    title = "A sufficiently long identical title for DOI conflict testing"
    packet = _packet(
        [
            _work("a", title, "10.1000/a", "openalex"),
            _work("b", title, "10.1000/b", "crossref"),
        ]
    )
    canonical = canonicalize_prior_art_packet(packet)
    assert len(canonical.works) == 2
    assert {w.doi for w in canonical.works} == {
        "10.1000/a",
        "10.1000/b",
    }


def test_same_doi_family_still_merges():
    title = "A sufficiently long identical title for same DOI family testing"
    packet = _packet(
        [
            _work("a", title, "10.1000/a", "openalex"),
            _work("b", title, "10.1000/a", "crossref"),
        ]
    )
    canonical = canonicalize_prior_art_packet(packet)
    assert len(canonical.works) == 1
    assert set(canonical.works[0].providers) == {
        "openalex",
        "crossref",
    }


def test_one_doi_family_plus_doi_less_record_merges():
    title = "A sufficiently long identical title for DOI-less fallback testing"
    packet = _packet(
        [
            _work("a", title, "10.1000/a", "openalex"),
            _work("b", title, None, "crossref"),
        ]
    )
    canonical = canonicalize_prior_art_packet(packet)
    assert len(canonical.works) == 1
    assert canonical.works[0].doi == "10.1000/a"


def test_multiple_doi_families_plus_doi_less_stays_separate():
    title = "A sufficiently long identical title for ambiguous fallback testing"
    packet = _packet(
        [
            _work("a", title, "10.1000/a", "openalex"),
            _work("b", title, "10.1000/b", "crossref"),
            _work("c", title, None, "crossref"),
        ]
    )
    canonical = canonicalize_prior_art_packet(packet)
    assert len(canonical.works) == 3
