from __future__ import annotations

from pipeline_core.literature.discovery.contracts import (
    LiteratureRecord,
    literature_paper_id,
    merge_literature_records,
    normalize_doi,
)


def test_doi_normalization_produces_provider_independent_paper_id():
    doi = normalize_doi("https://doi.org/10.1000/ABC.123")
    assert doi == "10.1000/abc.123"
    assert literature_paper_id(
        doi=doi,
        provider="openalex",
        provider_id="W1",
    ) == literature_paper_id(
        doi="doi:10.1000/ABC.123",
        provider="semantic_scholar",
        provider_id="S2-1",
    )


def test_repeated_discovery_merges_queries_buckets_and_richer_source():
    first = LiteratureRecord.from_provider_result(
        provider="openalex",
        provider_id="W123",
        title="Dynamic active sites",
        doi="10.1000/demo",
        discovery_query="dynamic active site electrocatalysis",
        mechanism_bucket="working_state_reconstruction",
    )
    second = LiteratureRecord.from_provider_result(
        provider="semantic_scholar",
        provider_id="S456",
        title="Dynamic active sites",
        abstract="Adsorbate coverage induces a reversible reconstruction.",
        doi="https://doi.org/10.1000/DEMO",
        discovery_query="adsorbate induced reconstruction",
        mechanism_bucket="descriptor_failure_counterexamples",
    )

    merged = merge_literature_records(first, second)
    assert merged.source_depth == "abstract"
    assert merged.abstract is not None
    assert merged.discovery_queries == (
        "adsorbate induced reconstruction",
        "dynamic active site electrocatalysis",
    )
    assert merged.mechanism_buckets == (
        "descriptor_failure_counterexamples",
        "working_state_reconstruction",
    )
    assert {item.provider for item in merged.provider_references} == {
        "openalex",
        "semantic_scholar",
    }
