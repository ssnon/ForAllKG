from __future__ import annotations

import os

from dac_her.corpus_acquisition.access_contracts import (
    SourceAcquisitionPolicy,
)
from dac_her.corpus_acquisition.oa_resolution import (
    OpenAccessResolver,
    _catalog_location,
    _location_from_unpaywall,
)
from dac_her.literature_catalog_contracts import CatalogWork


def _policy() -> SourceAcquisitionPolicy:
    return SourceAcquisitionPolicy(
        policy_id="test_policy",
        use_unpaywall=False,
    )


def test_catalog_oa_location_is_download_candidate():
    work = CatalogWork(
        work_id="w1",
        title="Example",
        open_access_url="https://example.org/paper.pdf",
    )
    location = _catalog_location(work)
    assert location is not None
    assert location.automatic_download_eligible is True
    assert location.resolver == "catalog_open_access"


def test_unpaywall_direct_pdf_preserves_license_version_host():
    row = {
        "url_for_pdf": "https://example.org/p.pdf",
        "url_for_landing_page": "https://example.org/article",
        "host_type": "repository",
        "version": "acceptedVersion",
        "license": "cc-by",
    }
    location = _location_from_unpaywall(
        work_id="w1",
        row=row,
        is_best=True,
    )
    assert location is not None
    assert location.is_best is True
    assert location.automatic_download_eligible is True
    assert location.license == "cc-by"
    assert location.version == "acceptedVersion"
    assert location.host_type == "repository"


def test_no_doi_and_no_catalog_oa_remains_unresolved():
    resolver = OpenAccessResolver(_policy())
    work = CatalogWork(work_id="w1", title="No OA")
    resolution = resolver.resolve(work)
    assert resolution.status == "unresolved"
    assert resolution.paywall_bypass_attempted is False


def test_catalog_fallback_resolves_without_unpaywall():
    resolver = OpenAccessResolver(_policy())
    work = CatalogWork(
        work_id="w1",
        title="OA",
        open_access_url="https://example.org/paper.pdf",
    )
    resolution = resolver.resolve(work)
    assert resolution.status == "resolved_direct_pdf"
    assert resolution.selected_download_url == "https://example.org/paper.pdf"
