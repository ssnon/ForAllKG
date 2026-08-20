from __future__ import annotations

import json

from dac_her.corpus_acquisition.access_contracts import (
    AccessLocation,
    SourceAcquisitionPolicy,
)
from dac_her.corpus_acquisition.access_priority import (
    access_location_priority,
)
from dac_her.corpus_acquisition.pmc_access import (
    PmcAwsAccessResolver,
    extract_pmcids,
)
from pipeline_core.literature.catalog_contracts import CatalogWork


def _existing_pmc_location() -> AccessLocation:
    return AccessLocation(
        location_id="loc:pmc",
        resolver="openalex",
        url=(
            "https://pmc.ncbi.nlm.nih.gov/"
            "articles/PMC12345678/"
            "pdf/example.pdf"
        ),
        url_for_pdf=(
            "https://pmc.ncbi.nlm.nih.gov/"
            "articles/PMC12345678/"
            "pdf/example.pdf"
        ),
        is_oa=True,
        automatic_download_eligible=True,
    )


def test_extract_pmcid_from_existing_location():
    rows = extract_pmcids(
        [_existing_pmc_location()]
    )

    assert rows == ["PMC12345678"]


def test_pmc_aws_current_dataset_produces_repository_pdf():
    list_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult
 xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
 <CommonPrefixes>
  <Prefix>PMC12345678.1/</Prefix>
 </CommonPrefixes>
 <CommonPrefixes>
  <Prefix>PMC12345678.2/</Prefix>
 </CommonPrefixes>
</ListBucketResult>
"""

    metadata = {
        "pmcid": "PMC12345678",
        "version": 2,
        "doi": "10.1000/example",
        "is_pmc_openaccess": True,
        "is_retracted": False,
        "license_code": "CC BY",
        "pdf_url": (
            "s3://pmc-oa-opendata/"
            "PMC12345678.2/"
            "PMC12345678.2.pdf?md5=abc"
        ),
    }

    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)

        if "list-type=2" in url:
            return list_xml

        if url.endswith(
            "/metadata/PMC12345678.2.json"
        ):
            return json.dumps(
                metadata
            ).encode("utf-8")

        raise AssertionError(
            f"unexpected URL: {url}"
        )

    policy = SourceAcquisitionPolicy(
        policy_id="test",
        use_pmc_aws=True,
        resolver_delay_seconds=0,
    )

    resolver = PmcAwsAccessResolver(
        policy=policy,
        request_bytes=fake_request,
    )

    work = CatalogWork(
        work_id="w1",
        title="Example",
        doi="10.1000/example",
    )

    probe = resolver.resolve(
        work,
        existing_locations=[
            _existing_pmc_location()
        ],
    )

    assert probe.attempt.status == "success"
    assert len(probe.locations) == 1

    location = probe.locations[0]

    assert location.resolver == "pmc_aws"
    assert location.source_id == "PMC12345678"
    assert location.source_name == "PubMed Central"
    assert location.host_type == "repository"
    assert location.license == "CC BY"
    assert location.version == "2"
    assert location.automatic_download_eligible is True

    assert location.url_for_pdf == (
        "https://pmc-oa-opendata."
        "s3.amazonaws.com/"
        "PMC12345678.2/"
        "PMC12345678.2.pdf?md5=abc"
    )

    assert (
        "pmc_aws_current_dataset"
        in location.reason_codes
    )

    assert len(calls) == 2


def test_pmc_aws_is_repository_first_priority():
    pmc = AccessLocation(
        location_id="pmc",
        resolver="pmc_aws",
        url="https://repo.example/main.pdf",
        url_for_pdf=(
            "https://repo.example/main.pdf"
        ),
        is_oa=True,
        automatic_download_eligible=True,
    )

    unpaywall = AccessLocation(
        location_id="upw",
        resolver="unpaywall",
        url="https://publisher.example/main.pdf",
        url_for_pdf=(
            "https://publisher.example/main.pdf"
        ),
        is_oa=True,
        is_best=True,
        automatic_download_eligible=True,
    )

    assert (
        access_location_priority(pmc)
        < access_location_priority(unpaywall)
    )


def test_pmc_aws_policy_is_opt_in_by_default():
    policy = SourceAcquisitionPolicy(
        policy_id="test"
    )

    assert policy.use_pmc_aws is False


def test_numeric_ncbi_pmc_url_is_normalized():
    location = AccessLocation(
        location_id="legacy-ncbi",
        resolver="openalex",
        url=(
            "https://www.ncbi.nlm.nih.gov/"
            "pmc/articles/7377325"
        ),
        is_oa=True,
        automatic_download_eligible=False,
    )

    assert extract_pmcids(
        [location]
    ) == ["PMC7377325"]


def test_numeric_current_pmc_url_is_normalized():
    location = AccessLocation(
        location_id="current-pmc",
        resolver="openalex",
        url=(
            "https://pmc.ncbi.nlm.nih.gov/"
            "articles/9268529/"
        ),
        is_oa=True,
        automatic_download_eligible=False,
    )

    assert extract_pmcids(
        [location]
    ) == ["PMC9268529"]


def test_prefixed_pmcid_behavior_is_preserved():
    location = AccessLocation(
        location_id="prefixed-pmc",
        resolver="openalex",
        url=(
            "https://pmc.ncbi.nlm.nih.gov/"
            "articles/PMC10254201/"
        ),
        is_oa=True,
        automatic_download_eligible=False,
    )

    assert extract_pmcids(
        [location]
    ) == ["PMC10254201"]


def test_numeric_path_on_unrelated_host_is_rejected():
    location = AccessLocation(
        location_id="unrelated",
        resolver="openalex",
        url=(
            "https://example.org/"
            "pmc/articles/7377325"
        ),
        is_oa=True,
        automatic_download_eligible=False,
    )

    assert extract_pmcids(
        [location]
    ) == []


def test_unrelated_ncbi_numeric_path_is_rejected():
    location = AccessLocation(
        location_id="unrelated-ncbi",
        resolver="openalex",
        url=(
            "https://www.ncbi.nlm.nih.gov/"
            "books/7377325"
        ),
        is_oa=True,
        automatic_download_eligible=False,
    )

    assert extract_pmcids(
        [location]
    ) == []
