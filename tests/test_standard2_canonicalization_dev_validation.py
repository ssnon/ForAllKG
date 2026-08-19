from __future__ import annotations

from dac_her.external_novelty_contracts import (
    PriorArtPacket,
    PriorArtWork,
)
from dac_her.literature_retrieval import (
    canonicalize_prior_art_packet,
)
from campaigns.sers_standard2.canonicalization_dev_validation import (
    canonical_json,
    sha256_json,
    title_cross_doi_collision_groups,
)


def _work(
    *,
    work_id: str,
    title: str,
    doi: str | None,
    provider: str,
    query_id: str,
    claim_id: str | None = None,
    abstract: str | None = None,
) -> PriorArtWork:
    return PriorArtWork(
        work_id=work_id,
        title=title,
        doi=doi,
        abstract=abstract,
        providers=[provider],
        provider_ids={
            provider: work_id,
        },
        retrieval_query_ids=[
            query_id,
        ],
        retrieval_claim_ids=(
            [claim_id]
            if claim_id
            else []
        ),
    )


def _packet(
    works: list[PriorArtWork],
) -> PriorArtPacket:
    body = {
        "schema_version":
            "prior-art-packet-v1",
        "packet_id":
            "raw:test",
        "source_portfolio_id":
            "portfolio:test",
        "source_query_plan_id":
            "plan:test",
        "searched_at_utc":
            "2026-08-16T00:00:00+00:00",
        "providers_requested":
            ["openalex", "crossref"],
        "works": [
            row.model_dump(mode="json")
            for row in works
        ],
        "executions": [],
        "raw_work_count":
            len(works),
        "canonical_work_count":
            len(works),
        "deduplicated_work_count":
            0,
        "supplementary_records_collapsed":
            0,
        "epistemic_usage":
            "prior_art_only_not_positive_premise",
    }
    return PriorArtPacket(
        **body,
        packet_sha256=sha256_json(body),
    )


def test_same_doi_merges_and_preserves_provenance():
    works = [
        _work(
            work_id="oa",
            title="A sufficiently long catalyst paper title",
            doi="10.1000/example",
            provider="openalex",
            query_id="q1",
            claim_id="c1",
            abstract="short",
        ),
        _work(
            work_id="cr",
            title="A sufficiently long catalyst paper title",
            doi="https://doi.org/10.1000/example",
            provider="crossref",
            query_id="q2",
            claim_id="c2",
            abstract="a much longer abstract",
        ),
    ]
    canonical = canonicalize_prior_art_packet(
        _packet(works)
    )
    assert len(canonical.works) == 1
    row = canonical.works[0]
    assert set(row.providers) == {
        "openalex",
        "crossref",
    }
    assert set(row.retrieval_query_ids) == {
        "q1",
        "q2",
    }
    assert set(row.retrieval_claim_ids) == {
        "c1",
        "c2",
    }
    assert row.abstract == "a much longer abstract"


def test_recanonicalization_is_deterministic():
    packet = _packet(
        [
            _work(
                work_id="oa",
                title="Another sufficiently long catalyst paper title",
                doi="10.1000/example2",
                provider="openalex",
                query_id="q1",
            ),
            _work(
                work_id="cr",
                title="Another sufficiently long catalyst paper title",
                doi="10.1000/example2",
                provider="crossref",
                query_id="q2",
            ),
        ]
    )
    left = canonicalize_prior_art_packet(
        packet
    )
    right = canonicalize_prior_art_packet(
        packet
    )
    assert canonical_json(left) == canonical_json(
        right
    )


def test_title_collision_detector_flags_distinct_dois():
    works = [
        _work(
            work_id="a",
            title="Identical normalized title for collision testing",
            doi="10.1000/a",
            provider="openalex",
            query_id="q1",
        ),
        _work(
            work_id="b",
            title="Identical normalized title for collision testing",
            doi="10.1000/b",
            provider="crossref",
            query_id="q1",
        ),
    ]
    rows = title_cross_doi_collision_groups(
        works
    )
    assert len(rows) == 1
    assert (
        rows[0]["distinct_doi_family_count"]
        == 2
    )


def test_title_collision_detector_allows_same_doi_family():
    works = [
        _work(
            work_id="a",
            title="Identical normalized title for same DOI testing",
            doi="10.1000/a",
            provider="openalex",
            query_id="q1",
        ),
        _work(
            work_id="b",
            title="Identical normalized title for same DOI testing",
            doi="10.1000/a.s1",
            provider="crossref",
            query_id="q1",
        ),
    ]
    assert not title_cross_doi_collision_groups(
        works
    )
