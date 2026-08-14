from __future__ import annotations

from dac_her.literature_catalog import (
    canonicalize_catalog_works,
)
from dac_her.literature_catalog_contracts import (
    CatalogWork,
)


def _work(
    work_id: str,
    *,
    title: str,
    doi: str | None,
    provider: str,
    axis: str,
    abstract: str | None = None,
):
    return CatalogWork(
        work_id=work_id,
        title=title,
        doi=doi,
        abstract=abstract,
        providers=[provider],
        provider_ids={provider: work_id},
        retrieval_query_ids=[f"q:{axis}"],
        retrieval_axis_ids=[axis],
    )


def test_catalog_collapses_doi_family_and_preserves_provenance():
    rows = [
        _work(
            "a",
            title="A sufficiently specific SERS nanogap paper title",
            doi="10.1000/example",
            provider="crossref",
            axis="nanogap",
        ),
        _work(
            "b",
            title="A sufficiently specific SERS nanogap paper title",
            doi="10.1000/example.s1",
            provider="semantic_scholar",
            axis="mechanism",
            abstract="Longer abstract",
        ),
    ]
    canonical, supplementary = canonicalize_catalog_works(rows)
    assert len(canonical) == 1
    assert supplementary == 1
    work = canonical[0]
    assert work.doi == "10.1000/example"
    assert set(work.providers) == {"crossref", "semantic_scholar"}
    assert set(work.retrieval_axis_ids) == {"nanogap", "mechanism"}
    assert work.abstract == "Longer abstract"


def test_catalog_collapses_doi_and_doi_less_exact_title_duplicate():
    title = "Distinctive gold silver SERS shell thickness controlled study"
    rows = [
        _work(
            "a",
            title=title,
            doi="10.1000/main",
            provider="crossref",
            axis="shell",
        ),
        _work(
            "b",
            title=title,
            doi=None,
            provider="semantic_scholar",
            axis="shell",
        ),
    ]
    canonical, _ = canonicalize_catalog_works(rows)
    assert len(canonical) == 1
    assert canonical[0].doi == "10.1000/main"
