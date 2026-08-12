from __future__ import annotations

import json

from dac_her.literature_discovery.providers import (
    OpenAlexProvider,
    TransportResponse,
    reconstruct_abstract,
)
from dac_her.literature_discovery.providers.base import LiteratureSearchRequest


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, timeout, headers):
        self.calls.append(
            {
                "url": url,
                "params": dict(params),
                "timeout": timeout,
                "headers": dict(headers),
            }
        )
        return self.responses.pop(0)


def _response(payload, status=200, headers=None):
    return TransportResponse(
        status=status,
        headers=headers or {},
        body=json.dumps(payload).encode("utf-8"),
    )


def _work(index: int, *, doi: str | None = None):
    return {
        "id": f"https://openalex.org/W{index}",
        "doi": doi,
        "title": f"Paper {index}",
        "publication_year": 2025,
        "abstract_inverted_index": {
            "Catalysis": [0],
            "works": [1],
            str(index): [2],
        },
        "primary_location": {
            "is_oa": True,
            "source": {"display_name": "Catalysis Journal"},
        },
        "cited_by_count": index,
        "relevance_score": 10.0 - index,
    }


def test_reconstruct_abstract_orders_positions():
    abstract = reconstruct_abstract({"world": [1], "Hello": [0], "again": [2]})
    assert abstract == "Hello world again"


def test_openalex_provider_uses_cursor_pagination_and_normalizes_records():
    transport = FakeTransport(
        [
            _response({
                "meta": {"next_cursor": "next-1"},
                "results": [_work(index, doi=f"https://doi.org/10.1/{index}") for index in range(100)],
            }),
            _response({
                "meta": {"next_cursor": None},
                "results": [_work(index, doi=f"https://doi.org/10.1/{index}") for index in range(100, 150)],
            }),
        ]
    )
    provider = OpenAlexProvider(
        api_key="test-key",
        transport=transport,
        sleep=lambda _: None,
    )
    records = provider.search(
        LiteratureSearchRequest(
            query="electrocatalyst reconstruction",
            mechanism_bucket="working_state_reconstruction",
            limit=150,
        )
    )

    assert len(records) == 150
    assert transport.calls[0]["params"]["cursor"] == "*"
    assert transport.calls[0]["params"]["per_page"] == "100"
    assert transport.calls[1]["params"]["cursor"] == "next-1"
    assert transport.calls[1]["params"]["per_page"] == "50"
    assert transport.calls[0]["params"]["api_key"] == "test-key"
    assert records[0].doi == "10.1/0"
    assert records[0].abstract == "Catalysis works 0"
    assert records[0].venue == "Catalysis Journal"
    assert records[0].provider_references[0].provider_id == "W0"
    assert records[0].source_depth == "abstract"


def test_openalex_provider_retries_429_and_honors_retry_after():
    transport = FakeTransport(
        [
            _response({"error": "rate limited"}, status=429, headers={"Retry-After": "3"}),
            _response({"meta": {"next_cursor": None}, "results": [_work(1)]}),
        ]
    )
    sleeps = []
    provider = OpenAlexProvider(
        transport=transport,
        sleep=sleeps.append,
        backoff_base_seconds=1.0,
        max_retries=2,
    )
    records = provider.search(
        LiteratureSearchRequest(
            query="active site",
            mechanism_bucket="active_site_attribution",
            limit=1,
        )
    )

    assert len(records) == 1
    assert sleeps == [3.0]
    assert len(transport.calls) == 2
