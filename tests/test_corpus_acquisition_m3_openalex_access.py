from __future__ import annotations

from pipeline_core.literature.acquisition.access_contracts import (
    AccessLocation,
    SourceAcquisitionPolicy,
)
from pipeline_core.literature.acquisition.access_priority import (
    access_location_priority,
)
from pipeline_core.literature.acquisition.openalex_access import (
    OpenAlexAccessResolver,
    _locations_from_openalex_work,
)
from pipeline_core.literature.catalog_contracts import CatalogWork


class _FakeProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_work(self, identifier: str):
        assert identifier in {"10.1234/example", "W123"}
        repeated = {
            "is_oa": True,
            "landing_page_url": "https://repo.example.org/item/1",
            "pdf_url": "https://repo.example.org/item/1.pdf",
            "license": "cc-by",
            "version": "acceptedVersion",
            "source": {
                "id": "https://openalex.org/S1",
                "display_name": "Example Repository",
                "type": "repository",
            },
        }
        return {
            "best_oa_location": repeated,
            "primary_location": repeated,
            "locations": [
                repeated,
                {
                    "is_oa": True,
                    "landing_page_url": "https://publisher.example.org/a",
                    "pdf_url": "https://publisher.example.org/a.pdf",
                    "license": "cc-by",
                    "version": "publishedVersion",
                    "source": {
                        "id": "https://openalex.org/S2",
                        "display_name": "Example Journal",
                        "type": "journal",
                    },
                },
                {
                    "is_oa": False,
                    "landing_page_url": "https://closed.example.org/a",
                    "pdf_url": "https://closed.example.org/a.pdf",
                },
            ],
        }


def _policy(**overrides) -> SourceAcquisitionPolicy:
    values = {
        "policy_id": "test_openalex_policy",
        "use_unpaywall": False,
        "use_catalog_open_access_url": False,
        "use_openalex": True,
        "openalex_require_api_key": False,
        "resolver_delay_seconds": 0,
    }
    values.update(overrides)
    return SourceAcquisitionPolicy(**values)


def test_collects_all_explicit_oa_locations_and_deduplicates_roles():
    raw = _FakeProvider().get_work("10.1234/example")
    rows = _locations_from_openalex_work(work_id="w1", raw_work=raw)
    assert len(rows) == 2
    assert all(row.resolver == "openalex" for row in rows)
    assert all(row.automatic_download_eligible for row in rows)
    best = next(row for row in rows if row.is_best)
    assert "primary_location" in best.reason_codes
    assert best.license == "cc-by"
    assert best.version == "acceptedVersion"
    assert best.source_name == "Example Repository"
    assert best.host_type == "repository"


def test_openalex_adapter_uses_doi_and_preserves_access_contract(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    resolver = OpenAlexAccessResolver(
        _policy(),
        provider_factory=_FakeProvider,
    )
    probe = resolver.resolve(
        CatalogWork(
            work_id="w1",
            title="Example",
            doi="10.1234/example",
        )
    )
    assert probe.attempt.status == "success"
    assert len(probe.locations) == 2
    assert probe.locations[0].is_best is True


def test_missing_required_key_skips_only_openalex_lane(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    resolver = OpenAlexAccessResolver(
        _policy(openalex_require_api_key=True),
        provider_factory=_FakeProvider,
    )
    probe = resolver.resolve(
        CatalogWork(
            work_id="w1",
            title="Example",
            doi="10.1234/example",
        )
    )
    assert probe.locations == []
    assert probe.attempt.status == "skipped"
    assert "OPENALEX_API_KEY" in (probe.attempt.message or "")


def test_openalex_direct_pdf_precedes_catalog_fallback():
    openalex = AccessLocation(
        location_id="oa",
        resolver="openalex",
        url="https://repo.example.org/p.pdf",
        url_for_pdf="https://repo.example.org/p.pdf",
        is_best=False,
        automatic_download_eligible=True,
        reason_codes=["openalex_oa_location", "direct_pdf_url"],
    )
    catalog = AccessLocation(
        location_id="catalog",
        resolver="catalog_open_access",
        url="https://catalog.example.org/p.pdf",
        url_for_pdf="https://catalog.example.org/p.pdf",
        automatic_download_eligible=True,
    )
    assert access_location_priority(openalex) < access_location_priority(catalog)


def test_openaccessresolver_integrates_openalex_before_catalog(monkeypatch):
    from pipeline_core.literature.acquisition import oa_resolution as oa_module
    from pipeline_core.literature.acquisition.access_contracts import ResolverAttempt
    from pipeline_core.literature.acquisition.openalex_access import OpenAlexAccessProbe

    openalex_location = AccessLocation(
        location_id="oa-best",
        resolver="openalex",
        url="https://repo.example.org/best.pdf",
        url_for_pdf="https://repo.example.org/best.pdf",
        is_best=True,
        automatic_download_eligible=True,
        reason_codes=["openalex_oa_location", "best_oa_location", "direct_pdf_url"],
    )

    class _Resolver:
        def __init__(self, policy):
            self.policy = policy

        def resolve(self, work):
            return OpenAlexAccessProbe(
                attempt=ResolverAttempt(
                    resolver="openalex",
                    status="success",
                    message="locations=1",
                ),
                locations=[openalex_location],
            )

    monkeypatch.setattr(oa_module, "OpenAlexAccessResolver", _Resolver)
    resolver = oa_module.OpenAccessResolver(
        _policy(use_catalog_open_access_url=True)
    )
    resolution = resolver.resolve(
        CatalogWork(
            work_id="w1",
            title="Example",
            doi="10.1234/example",
            open_access_url="https://catalog.example.org/fallback.pdf",
        )
    )
    assert resolution.status == "resolved_direct_pdf"
    assert resolution.selected_location_id == "oa-best"
    assert resolution.locations[0].resolver == "openalex"
    assert any(
        attempt.resolver == "openalex" and attempt.status == "success"
        for attempt in resolution.resolver_attempts
    )
