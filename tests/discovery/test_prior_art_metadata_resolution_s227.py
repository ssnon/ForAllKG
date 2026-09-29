from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtWork,
)
from pipeline_core.discovery.prior_art_metadata_resolution_s227 import (
    candidate_matches_source,
    doi_family,
    inherited_candidate,
    is_supplementary_doi,
    norm_title,
)


def _work(
    work_id,
    *,
    title,
    doi=None,
    abstract=None,
    provider="crossref",
):
    return PriorArtWork(
        work_id=work_id,
        title=title,
        doi=doi,
        abstract=abstract,
        providers=[provider],
        provider_ids={provider: work_id},
        retrieval_query_ids=["q-original"],
        retrieval_claim_ids=["c-original"],
    )


def test_s227_doi_family_collapses_supplement_suffix():
    assert (
        doi_family("10.1021/acs.jpclett.5c00101.s001")
        == "10.1021/acs.jpclett.5c00101"
    )
    assert is_supplementary_doi(
        "10.1021/acs.jpclett.5c00101.s001"
    )
    assert not is_supplementary_doi(
        "10.1021/acs.jpclett.5c00101"
    )


def test_s227_accepts_parent_article_from_supplement_family():
    source = _work(
        "supp",
        title="Supporting Information",
        doi="10.1021/acs.jpclett.5c00101.s001",
    )
    parent = _work(
        "parent",
        title="Main article",
        doi="10.1021/acs.jpclett.5c00101",
        abstract="main abstract",
        provider="openalex",
    )

    accepted, basis = candidate_matches_source(
        source,
        parent,
    )
    assert accepted is True
    assert basis == "DOI_FAMILY_EXACT"


def test_s227_rejects_different_strong_doi_family_even_sameish_title():
    source = _work(
        "a",
        title="Same Scientific Title",
        doi="10.1000/a",
    )
    candidate = _work(
        "b",
        title="Same Scientific Title",
        doi="10.1000/b",
        abstract="abstract",
    )

    accepted, basis = candidate_matches_source(
        source,
        candidate,
    )
    assert accepted is False
    assert basis == "DOI_FAMILY_MISMATCH"


def test_s227_title_fallback_requires_exact_normalized_title():
    source = _work(
        "a",
        title="A sufficiently long scientific title about catalysis",
    )
    same = _work(
        "b",
        title="A sufficiently-long scientific title about catalysis!",
        abstract="abstract",
    )
    other = _work(
        "c",
        title="A different sufficiently long scientific title",
        abstract="abstract",
    )

    accepted, basis = candidate_matches_source(
        source,
        same,
    )
    assert accepted is True
    assert basis == "EXACT_NORMALIZED_TITLE"

    accepted2, _ = candidate_matches_source(
        source,
        other,
    )
    assert accepted2 is False


def test_s227_metadata_candidate_does_not_create_query_coverage():
    source = _work(
        "a",
        title="A sufficiently long scientific title about spectroscopy",
        doi="10.1000/a",
    )
    candidate = PriorArtWork(
        work_id="b",
        title=source.title,
        doi=source.doi,
        abstract="resolved abstract",
        providers=["openalex"],
        provider_ids={"openalex": "W123"},
        retrieval_query_ids=["synthetic-s227-query"],
        retrieval_claim_ids=[],
    )

    inherited = inherited_candidate(
        candidate,
        source=source,
    )

    assert inherited.retrieval_query_ids == [
        "q-original"
    ]
    assert inherited.retrieval_claim_ids == [
        "c-original"
    ]
    assert inherited.providers == ["openalex"]
    assert inherited.abstract == "resolved abstract"


def test_s227_title_normalization_is_conservative_and_deterministic():
    assert (
        norm_title("A Study: Raman/SERS!")
        == "a study raman sers"
    )
