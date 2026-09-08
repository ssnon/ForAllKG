from __future__ import annotations

from pipeline_core.literature.acquisition.access_contracts import (
    AccessLocation,
    SourceAcquisitionPolicy,
)
from pipeline_core.literature.acquisition.main_landing_resolution import (
    PublicLandingMainPdfResolver,
    _extract_main_pdf_urls,
)
from pipeline_core.literature.catalog_contracts import CatalogWork


def test_extracts_main_pdf_meta_link_and_anchor():
    html = """
    <html><head>
      <meta name="citation_pdf_url" content="/paper.pdf">
      <link rel="alternate" type="application/pdf" href="/alternate.pdf">
    </head><body>
      <a href="/browser-pdf">Open PDF in Browser</a>
    </body></html>
    """
    rows = _extract_main_pdf_urls(
        html,
        resolved_page_url="https://publisher.example/article",
    )
    urls = {url for url, _ in rows}
    assert "https://publisher.example/paper.pdf" in urls
    assert "https://publisher.example/alternate.pdf" in urls
    assert "https://publisher.example/browser-pdf" in urls


def test_rejects_supplementary_pdf_candidates():
    html = """
    <html><head>
      <meta name="citation_pdf_url" content="/article_supplement.pdf">
    </head><body>
      <a href="/supp/article_si.pdf">Download PDF Supplementary Information</a>
    </body></html>
    """
    rows = _extract_main_pdf_urls(
        html,
        resolved_page_url="https://publisher.example/article",
    )
    assert rows == []


def test_resolver_promotes_public_landing_candidate_without_claiming_oa():
    def fake_fetch(url, **kwargs):
        return (
            '<html><meta name="citation_pdf_url" content="/main.pdf"></html>',
            "https://repo.example/article",
            "text/html",
        )

    policy = SourceAcquisitionPolicy(
        policy_id="test",
        use_public_landing_html=True,
        resolver_delay_seconds=0,
    )
    resolver = PublicLandingMainPdfResolver(
        policy=policy,
        fetcher=fake_fetch,
    )
    work = CatalogWork(
        work_id="w1",
        title="Paper",
        doi="10.1/example",
    )
    landing = AccessLocation(
        location_id="l1",
        resolver="unpaywall",
        url="https://repo.example/article",
        url_for_landing_page="https://repo.example/article",
        automatic_download_eligible=False,
    )

    probe = resolver.resolve(
        work,
        existing_locations=[landing],
    )
    assert probe.attempt.status == "success"
    assert len(probe.locations) == 1
    row = probe.locations[0]
    assert row.resolver == "public_landing_html"
    assert row.automatic_download_eligible is True
    assert row.url_for_pdf == "https://repo.example/main.pdf"
    assert row.is_oa is False
    assert "oa_status_unverified" in row.reason_codes


def test_supplementary_doi_is_not_treated_as_main_work():
    called = False

    def fake_fetch(url, **kwargs):
        nonlocal called
        called = True
        return (
            '<html><meta name="citation_pdf_url" content="/si.pdf"></html>',
            "https://publisher.example/si",
            "text/html",
        )

    policy = SourceAcquisitionPolicy(
        policy_id="test",
        use_public_landing_html=True,
        resolver_delay_seconds=0,
    )
    probe = PublicLandingMainPdfResolver(
        policy=policy,
        fetcher=fake_fetch,
    ).resolve(
        CatalogWork(
            work_id="w-si",
            title="Supporting information",
            doi="10.1021/example.s001",
        ),
        existing_locations=[],
    )
    assert probe.attempt.status == "skipped"
    assert probe.attempt.message == "supplementary_doi_not_main_work"
    assert probe.locations == []
    assert called is False
