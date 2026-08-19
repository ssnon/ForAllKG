from types import SimpleNamespace

import pytest

from dac_her.corpus_acquisition.openalex_catalog_adapter import (
    OpenAlexCatalogProvider,
)
from scripts.expand_literature_catalog import (
    MAX_RESULTS_PER_QUERY,
    _validate_results_per_query,
)


class _CaptureOpenAlexProvider:
    def __init__(self) -> None:
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        return []


def test_openalex_catalog_adapter_forwards_deep_limit() -> None:
    underlying = _CaptureOpenAlexProvider()

    adapter = OpenAlexCatalogProvider(
        provider=underlying
    )

    query = SimpleNamespace(
        query_text="gold silver SERS nanogap",
        axis_id="nanogap",
    )

    result = adapter.search(
        query,
        limit=500,
    )

    assert result == []
    assert len(underlying.requests) == 1
    assert underlying.requests[0].limit == 500


def test_openalex_catalog_adapter_preserves_default_depth_contract() -> None:
    underlying = _CaptureOpenAlexProvider()

    adapter = OpenAlexCatalogProvider(
        provider=underlying
    )

    query = SimpleNamespace(
        query_text="gold silver SERS",
        axis_id="composition_ratio",
    )

    adapter.search(
        query,
        limit=100,
    )

    assert underlying.requests[0].limit == 100


@pytest.mark.parametrize(
    "value",
    [
        1,
        100,
        101,
        500,
        1000,
    ],
)
def test_expansion_cli_accepts_supported_depth(
    value: int,
) -> None:
    assert (
        _validate_results_per_query(value)
        == value
    )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        1001,
    ],
)
def test_expansion_cli_rejects_out_of_range_depth(
    value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 1 and 1000",
    ):
        _validate_results_per_query(value)


def test_expansion_cli_deep_ceiling_is_explicit() -> None:
    assert MAX_RESULTS_PER_QUERY == 1000
