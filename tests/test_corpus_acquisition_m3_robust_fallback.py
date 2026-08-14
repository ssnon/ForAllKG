from __future__ import annotations

from pathlib import Path

import dac_her.corpus_acquisition.artifact_acquisition as module
from dac_her.corpus_acquisition.access_contracts import (
    AccessLocation,
    AccessResolution,
    SourceAcquisitionPolicy,
)
from dac_her.corpus_acquisition.artifact_acquisition import (
    MainArtifactDownloader,
    ordered_download_locations,
)
from dac_her.literature_catalog_contracts import CatalogWork


class _Headers:
    def __init__(self, content_type):
        self.content_type = content_type

    def get(self, key, default=None):
        if key.casefold() == "content-type":
            return self.content_type
        return default


class _Response:
    def __init__(self, body, url, content_type):
        self.body = body
        self.url = url
        self.headers = _Headers(content_type)
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.url

    def read(self, size=-1):
        if self.offset >= len(self.body):
            return b""
        if size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def _location(location_id, url, best=False):
    return AccessLocation(
        location_id=location_id,
        resolver="unpaywall",
        url=url,
        url_for_pdf=url,
        url_for_landing_page="https://example.org/article",
        is_oa=True,
        is_best=best,
        automatic_download_eligible=True,
    )


def test_ordered_locations_deduplicate_url():
    a = _location("a", "https://a.example/p.pdf", best=True)
    b = _location("b", "https://a.example/p.pdf")
    c = _location("c", "https://c.example/p.pdf")
    resolution = AccessResolution(
        work_id="w",
        status="resolved_direct_pdf",
        locations=[c, b, a],
        selected_location_id="a",
        selected_download_url=a.url,
    )
    rows = ordered_download_locations(resolution)
    assert [row.location_id for row in rows] == ["a", "c"]


def test_downloader_falls_back_after_non_pdf(monkeypatch, tmp_path):
    first = _location("first", "https://one.example/p.pdf", best=True)
    second = _location("second", "https://two.example/p.pdf")
    resolution = AccessResolution(
        work_id="w",
        status="resolved_direct_pdf",
        locations=[first, second],
        selected_location_id="first",
        selected_download_url=first.url,
    )
    responses = {
        first.url: _Response(
            b"<html>blocked</html>",
            first.url,
            "text/html",
        ),
        second.url: _Response(
            b"%PDF-1.7\nvalid",
            second.url,
            "application/pdf",
        ),
    }

    def fake_urlopen(request, timeout):
        return responses[request.full_url]

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    policy = SourceAcquisitionPolicy(
        policy_id="p",
        retries=0,
        try_all_direct_pdf_locations=True,
    )
    artifact = MainArtifactDownloader(policy).acquire(
        work=CatalogWork(work_id="w", title="Paper"),
        resolution=resolution,
        output_root=tmp_path,
    )
    assert artifact.status == "downloaded"
    assert artifact.selected_location_id == "second"
    assert artifact.attempted_location_count == 2
    assert [x.status for x in artifact.download_attempts] == [
        "failed",
        "success",
    ]
    assert artifact.download_attempts[0].error_code == "not_pdf"
