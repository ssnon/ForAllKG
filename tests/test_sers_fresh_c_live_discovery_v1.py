from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import (
    HistoricalLedgerSource,
    make_historical_exclusion_ledger,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_BROAD_QUERIES,
    EXPECTED_PROVIDERS,
    TARGET_ACQUIRED_PAPERS,
    assert_complete_execution,
    build_fresh_queue,
    load_and_validate_protocol,
    make_access_locator_payload,
    make_blind_queue_payload,
    make_catalog_queries,
    project_packet_to_identity_only,
)
from dac_her.literature_catalog_contracts import (
    CatalogQueryExecution,
    CatalogWork,
    LiteratureCatalogPacket,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.run_sers_fresh_c_live_discovery_v1 import execute


def _ledger(*ids: str):
    return make_historical_exclusion_ledger(
        canonical_ids=list(ids),
        sources=[
            HistoricalLedgerSource(
                source_id="history:test",
                source_sha256="a" * 64,
                canonical_identity_count=len(ids),
            )
        ],
    )


def _packet(works: list[CatalogWork]) -> LiteratureCatalogPacket:
    return LiteratureCatalogPacket(
        catalog_id="catalog:test",
        catalog_sha256="b" * 64,
        acquisition_profile_id="sers_fresh_c_broad_domain_v1",
        searched_at_utc="2026-08-17T00:00:00+00:00",
        providers_requested=["semantic_scholar", "crossref"],
        queries=[],
        works=works,
        executions=[],
        raw_work_count=len(works),
        canonical_work_count=len(works),
        deduplicated_work_count=0,
        supplementary_records_collapsed=0,
        epistemic_usage="candidate_source_only_not_positive_premise",
    )


def test_protocol_pins_broad_nonsemantic_discovery_contract():
    protocol = load_and_validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert protocol.providers == EXPECTED_PROVIDERS
    assert protocol.broad_queries == EXPECTED_BROAD_QUERIES
    assert protocol.results_per_query == 100
    assert protocol.expected_provider_query_executions == 8
    assert protocol.max_raw_metadata_rows == 800
    assert protocol.target_acquired_papers == 25
    assert protocol.full_fresh_identity_queue_frozen is True
    assert protocol.queue_truncated_to_target is False
    assert protocol.scientific_fields_used_for_ordering is False
    assert protocol.raw_catalog_packet_persisted is False
    assert (
        protocol.human_inspection_of_identity_or_locator_artifacts_before_c1_allowed
        is False
    )
    assert protocol.same_epoch_rerun_after_start_allowed is False
    assert protocol.failed_epoch_consumes_fresh_reserve_c is False
    assert protocol.automatic_c0_1d_transition_allowed is False


def test_queries_are_neutral_and_deterministic():
    protocol = load_and_validate_protocol(DEFAULT_PROTOCOL_PATH)
    first = make_catalog_queries(protocol)
    second = make_catalog_queries(protocol)
    assert first == second
    assert [row.query_text for row in first] == EXPECTED_BROAD_QUERIES
    assert {row.axis_id for row in first} == {
        "fresh_c_broad_domain_identity_only"
    }
    assert len({row.query_id for row in first}) == 4


def test_projection_discards_title_abstract_and_citation_but_keeps_access_metadata():
    packet = _packet(
        [
            CatalogWork(
                work_id="work:1",
                title="A highly scientific title that must not be persisted",
                doi="10.1234/example",
                url="https://example.org/article",
                open_access_url="https://example.org/article.pdf",
                abstract="scientific result that must not be persisted",
                citation_count=999,
                providers=["semantic_scholar"],
                provider_ids={"semantic_scholar": "S2-1"},
            )
        ]
    )
    projection = project_packet_to_identity_only(packet)
    assert len(projection.identity_records) == 1
    assert projection.ambiguous_identity_excluded_count == 0
    locator = projection.locators[0].model_dump(mode="json")
    assert locator["canonical_id"] == "doi:10.1234/example"
    assert locator["open_access_urls"] == ["https://example.org/article.pdf"]
    serialized = json.dumps(locator)
    assert "scientific title" not in serialized
    assert "scientific result" not in serialized
    assert "citation_count" not in serialized
    assert "title" not in locator
    assert "abstract" not in locator


def test_duplicate_identity_merges_only_access_locators():
    packet = _packet(
        [
            CatalogWork(
                work_id="work:a",
                title="First metadata row title long enough",
                doi="10.1234/example",
                open_access_url="https://a.example/p.pdf",
                providers=["semantic_scholar"],
                provider_ids={"semantic_scholar": "A"},
            ),
            CatalogWork(
                work_id="work:b",
                title="Second metadata row title long enough",
                doi="10.1234/example",
                open_access_url="https://b.example/p.pdf",
                providers=["crossref"],
                provider_ids={"crossref": "B"},
            ),
        ]
    )
    projection = project_packet_to_identity_only(packet)
    assert projection.duplicate_merge_count == 1
    assert len(projection.identity_records) == 1
    locator = projection.locators[0]
    assert locator.catalog_work_ids == ["work:a", "work:b"]
    assert locator.open_access_urls == [
        "https://a.example/p.pdf",
        "https://b.example/p.pdf",
    ]
    assert locator.providers == ["crossref", "semantic_scholar"]


def test_historical_exclusion_precedes_blind_queue():
    packet = _packet(
        [
            CatalogWork(
                work_id="seen",
                title="Seen paper title long enough",
                doi="10.1000/seen",
            ),
            CatalogWork(
                work_id="fresh",
                title="Fresh paper title long enough",
                doi="10.1000/fresh",
            ),
        ]
    )
    projection = project_packet_to_identity_only(packet)
    queue, locators, excluded = build_fresh_queue(
        projection=projection,
        historical_ledger=_ledger("doi:10.1000/seen"),
    )
    assert excluded == 1
    assert [row.canonical_id for row in queue] == ["doi:10.1000/fresh"]
    assert [row.canonical_id for row in locators] == ["doi:10.1000/fresh"]


def test_full_queue_is_not_truncated_to_target():
    works = [
        CatalogWork(
            work_id=f"work:{index}",
            title=f"Fresh candidate paper number {index} with long title",
            doi=f"10.2000/fresh{index}",
        )
        for index in range(30)
    ]
    projection = project_packet_to_identity_only(_packet(works))
    queue, locators, excluded = build_fresh_queue(
        projection=projection,
        historical_ledger=_ledger("doi:10.9999/history"),
    )
    assert excluded == 0
    assert len(queue) == 30
    assert len(locators) == 30
    assert TARGET_ACQUIRED_PAPERS == 25
    payload = make_blind_queue_payload(
        protocol=load_and_validate_protocol(DEFAULT_PROTOCOL_PATH),
        queue=queue,
    )
    assert payload["queue_count"] == 30
    assert payload["queue_truncated_to_target"] is False


def test_access_locator_payload_has_no_scientific_metadata_fields():
    packet = _packet(
        [
            CatalogWork(
                work_id="work:1",
                title="Long title used as bibliographic transport only",
                doi="10.3000/fresh",
                abstract="must not persist",
                citation_count=123,
            )
        ]
    )
    projection = project_packet_to_identity_only(packet)
    payload = make_access_locator_payload(
        protocol=load_and_validate_protocol(DEFAULT_PROTOCOL_PATH),
        locators=projection.locators,
    )
    serialized = json.dumps(payload, sort_keys=True)
    assert '"title"' not in serialized
    assert '"abstract"' not in serialized
    assert '"citation_count"' not in serialized
    assert payload["scientific_interpretation_performed"] is False


def test_execution_completeness_requires_all_eight_successes():
    protocol = load_and_validate_protocol(DEFAULT_PROTOCOL_PATH)
    queries = make_catalog_queries(protocol)
    executions = [
        CatalogQueryExecution(
            query_id=query.query_id,
            axis_id="fresh_c_broad_domain_identity_only",
            provider=provider,
            success=True,
            result_count=10,
        )
        for query in queries
        for provider in protocol.providers
    ]
    assert_complete_execution(protocol=protocol, executions=executions)
    executions[3] = executions[3].model_copy(update={"success": False})
    with pytest.raises(RuntimeError):
        assert_complete_execution(protocol=protocol, executions=executions)


def test_execution_completeness_rejects_missing_execution():
    protocol = load_and_validate_protocol(DEFAULT_PROTOCOL_PATH)
    executions = [
        CatalogQueryExecution(
            query_id=f"q{index}",
            axis_id="fresh_c_broad_domain_identity_only",
            provider="semantic_scholar",
            success=True,
            result_count=10,
        )
        for index in range(7)
    ]
    with pytest.raises(RuntimeError):
        assert_complete_execution(protocol=protocol, executions=executions)


def test_started_marker_is_written_before_retrieval_call_in_source():
    source = inspect.getsource(execute)
    assert source.index("_atomic_json(started_path, started)") < source.index(
        "retriever.retrieve("
    )
    assert source.index("preflight(") < source.index(
        "_atomic_json(started_path, started)"
    )


def test_failure_policy_is_not_fresh_c_consumption():
    protocol = load_and_validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert protocol.same_epoch_rerun_after_start_allowed is False
    assert protocol.failure_authorizes_query_or_selection_tuning is False
    assert protocol.failed_epoch_consumes_fresh_reserve_c is False
    assert protocol.fresh_reserve_c_consumption_occurs_in_c0_1c is False



def test_ambiguous_identity_candidate_is_fail_closed_excluded():
    packet = _packet(
        [
            CatalogWork(
                work_id="ambiguous",
                title="SERS Au",
                doi=None,
            ),
            CatalogWork(
                work_id="fresh",
                title="A sufficiently long fresh paper title for identity",
                doi="10.4000/fresh",
            ),
        ]
    )
    projection = project_packet_to_identity_only(packet)
    assert projection.ambiguous_identity_excluded_count == 1
    assert [row.canonical_id for row in projection.identity_records] == [
        "doi:10.4000/fresh"
    ]
