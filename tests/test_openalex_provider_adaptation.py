from __future__ import annotations

from email.message import Message
from urllib.error import HTTPError

import pytest

import dac_her.literature_retrieval as lr
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
)


def _query() -> LiteratureQuery:
    return LiteratureQuery(
        query_id="q1",
        hypothesis_id="h1",
        claim_id="c1",
        query_kind="claim_primary",
        query_text="hydrogen evolution catalyst",
    )


def test_reconstruct_openalex_abstract():
    value = {
        "Hydrogen": [0],
        "evolution": [1],
        "reaction": [2],
        "catalyst": [3],
    }
    assert (
        lr.reconstruct_openalex_abstract(value)
        == "Hydrogen evolution reaction catalyst"
    )


def test_reconstruct_openalex_abstract_rejects_collision():
    value = {
        "alpha": [0],
        "beta": [0],
    }
    assert (
        lr.reconstruct_openalex_abstract(value)
        is None
    )


def test_openalex_provider_maps_prior_art_work(monkeypatch):
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1234/test",
                "display_name": "A test work",
                "publication_year": 2025,
                "publication_date": "2025-02-03",
                "cited_by_count": 17,
                "abstract_inverted_index": {
                    "Test": [0],
                    "abstract": [1],
                },
                "authorships": [
                    {
                        "author": {
                            "display_name": "A. Author"
                        }
                    }
                ],
                "primary_location": {
                    "landing_page_url":
                        "https://example.org/article",
                    "source": {
                        "display_name": "Example Journal"
                    },
                },
                "best_oa_location": {
                    "pdf_url":
                        "https://example.org/article.pdf",
                    "landing_page_url":
                        "https://example.org/article",
                },
            }
        ]
    }

    captured = {}

    def fake_request(
        url,
        **kwargs,
    ):
        captured["url"] = url
        return payload

    monkeypatch.setattr(
        lr,
        "_request_json",
        fake_request,
    )
    provider = lr.OpenAlexProvider(
        api_key="oa-secret"
    )
    rows = provider.search(
        _query(),
        limit=5,
    )
    assert len(rows) == 1
    work = rows[0]
    assert work.title == "A test work"
    assert work.abstract == "Test abstract"
    assert work.doi == "10.1234/test"
    assert work.authors == ["A. Author"]
    assert work.venue == "Example Journal"
    assert work.citation_count == 17
    assert (
        work.open_access_url
        == "https://example.org/article.pdf"
    )
    assert work.providers == ["openalex"]
    assert work.retrieval_query_ids == ["q1"]
    assert "api_key=oa-secret" in captured["url"]


def test_openalex_provider_error_never_leaks_key(monkeypatch):
    headers = Message()

    def fake_request(
        url,
        **kwargs,
    ):
        raise HTTPError(
            url=url,
            code=429,
            msg="Too Many Requests",
            hdrs=headers,
            fp=None,
        )

    monkeypatch.setattr(
        lr,
        "_request_json",
        fake_request,
    )
    provider = lr.OpenAlexProvider(
        api_key="oa-super-secret"
    )
    with pytest.raises(
        lr.OpenAlexProviderError
    ) as observed:
        provider.search(
            _query(),
            limit=1,
        )
    text = str(observed.value)
    assert "oa-super-secret" not in text
    assert "429" in text


def test_openalex_provider_requires_key():
    with pytest.raises(
        ValueError,
        match="OPENALEX_API_KEY",
    ):
        lr.OpenAlexProvider(api_key="")
