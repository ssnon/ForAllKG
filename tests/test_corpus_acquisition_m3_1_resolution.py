from __future__ import annotations

from dac_her.corpus_acquisition.supplementary_resolution import (
    _AnchorParser,
    _anchor_candidate,
    _crossref_candidates,
    _deduplicate_candidates,
)
from dac_her.literature_catalog_contracts import CatalogWork


def test_strong_supporting_information_anchor_is_high_confidence_file():
    row = _anchor_candidate(
        work_id="w1",
        source_page_url="https://publisher.example/article",
        resolved_source_page_url="https://publisher.example/article",
        href="/suppl/article_si.pdf",
        anchor_text="Supporting Information",
        title_hint="",
    )
    assert row is not None
    assert row.kind == "direct_file"
    assert row.confidence == "high"
    assert row.automatic_download_eligible is True


def test_generic_data_link_is_not_misclassified():
    row = _anchor_candidate(
        work_id="w1",
        source_page_url="https://publisher.example/article",
        resolved_source_page_url="https://publisher.example/article",
        href="/data/table.csv",
        anchor_text="Data",
        title_hint="",
    )
    assert row is None


def test_crossref_is_supplemented_by_doi_is_metadata_only():
    work = CatalogWork(work_id="w1", title="Paper", doi="10.1/main")
    rows = _crossref_candidates(
        work=work,
        relation={
            "is-supplemented-by": [
                {
                    "id-type": "doi",
                    "id": "10.1/supp",
                }
            ]
        },
    )
    assert len(rows) == 1
    assert rows[0].confidence == "high"
    assert rows[0].kind == "related_identifier"
    assert rows[0].automatic_download_eligible is False


def test_crossref_direct_uri_supplement_can_be_download_candidate():
    work = CatalogWork(work_id="w1", title="Paper", doi="10.1/main")
    rows = _crossref_candidates(
        work=work,
        relation={
            "is-supplemented-by": [
                {
                    "id-type": "uri",
                    "id": "https://repo.example/supplement.pdf",
                }
            ]
        },
    )
    assert rows[0].kind == "direct_file"
    assert rows[0].automatic_download_eligible is True


def test_anchor_parser_extracts_text_and_href():
    parser = _AnchorParser()
    parser.feed(
        '<html><a href="/x.pdf" title="SI">Supporting <b>Information</b></a></html>'
    )
    assert len(parser.links) == 1
    assert parser.links[0]["href"] == "/x.pdf"
    assert "Supporting" in parser.links[0]["text"]


def test_candidate_dedup_prefers_high_confidence():
    high = _anchor_candidate(
        work_id="w1",
        source_page_url="https://p.example/a",
        resolved_source_page_url="https://p.example/a",
        href="/supplement.pdf",
        anchor_text="Supporting Information",
        title_hint="",
    )
    medium = _anchor_candidate(
        work_id="w1",
        source_page_url="https://p.example/a",
        resolved_source_page_url="https://p.example/a",
        href="/supplement.pdf",
        anchor_text="Download",
        title_hint="",
    )
    assert high is not None and medium is not None
    rows = _deduplicate_candidates([medium, high])
    assert len(rows) == 1
    assert rows[0].confidence == "high"
