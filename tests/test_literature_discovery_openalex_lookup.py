from __future__ import annotations

import json

from dac_her.literature_discovery.providers.openalex import (
    OpenAlexProvider,
    TransportResponse,
    openalex_location_metadata,
)


class _Transport:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, timeout, headers):
        self.calls.append((url, dict(params), timeout, dict(headers)))
        return TransportResponse(
            status=200,
            headers={},
            body=json.dumps(
                {
                    "id": "https://openalex.org/W123",
                    "doi": "https://doi.org/10.1234/example",
                    "locations": [],
                }
            ).encode("utf-8"),
        )


def test_targeted_doi_lookup_reuses_provider_transport_and_auth_params():
    transport = _Transport()
    provider = OpenAlexProvider(
        api_key="key",
        mailto="user@example.org",
        transport=transport,
        max_retries=0,
    )
    payload = provider.get_work("10.1234/example")
    assert payload["id"].endswith("W123")
    assert len(transport.calls) == 1
    url, params, _, _ = transport.calls[0]
    assert url.endswith("/works/https://doi.org/10.1234/example")
    assert params["api_key"] == "key"
    assert params["mailto"] == "user@example.org"


def test_targeted_openalex_id_lookup_uses_single_work_url():
    transport = _Transport()
    provider = OpenAlexProvider(
        transport=transport,
        max_retries=0,
    )
    provider.get_work("https://openalex.org/W123")
    assert transport.calls[0][0].endswith("/works/W123")


def test_public_location_metadata_preserves_source_type():
    metadata = openalex_location_metadata(
        {
            "is_oa": True,
            "pdf_url": "https://example.org/p.pdf",
            "source": {
                "id": "https://openalex.org/S1",
                "display_name": "Repository",
                "type": "repository",
            },
        }
    )
    assert metadata is not None
    assert metadata["source_type"] == "repository"
    assert metadata["source_name"] == "Repository"
