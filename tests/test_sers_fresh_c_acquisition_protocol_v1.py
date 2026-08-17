from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import dac_her.fresh_c_acquisition as fresh_c
from dac_her.fresh_c_acquisition import (
    FRESH_C_BLIND_ORDER_NAMESPACE,
    FreshCIdentityRecord,
    FreshCPreConsumptionSemanticState,
    HistoricalExclusionLedger,
    HistoricalLedgerSource,
    assert_preconsumption_operation_allowed,
    assert_semantically_unread,
    canonical_identity_from_fields,
    load_and_validate_protocol,
    make_historical_exclusion_ledger,
    project_catalog_identity,
    rank_fresh_identities,
)


PROTOCOL = Path(
    "dac_her/sers_fresh_c_acquisition_protocol_v1.json"
)


def _source(source_id: str = "history:test") -> HistoricalLedgerSource:
    return HistoricalLedgerSource(
        source_id=source_id,
        source_sha256="a" * 64,
        canonical_identity_count=1,
    )


def _ledger(*ids: str):
    return make_historical_exclusion_ledger(
        canonical_ids=ids,
        sources=[_source()],
    )


def test_doi_identity_is_preferred_over_title():
    canonical_id, method = canonical_identity_from_fields(
        doi="https://doi.org/10.1234/ABC.Def",
        title="A scientifically meaningful title that must not rank",
    )
    assert canonical_id == "doi:10.1234/abc.def"
    assert method == "doi_family"


def test_title_fallback_returns_hash_not_cleartext():
    title = (
        "A sufficiently long bibliographic title for fallback identity"
    )
    canonical_id, method = canonical_identity_from_fields(
        doi=None,
        title=title,
    )
    assert method == "normalized_title_sha256"
    assert canonical_id.startswith("title_sha256:")
    assert title.casefold() not in canonical_id.casefold()
    digest = canonical_id.split(":", 1)[1]
    assert len(digest) == 64


def test_short_title_without_doi_fails_closed():
    with pytest.raises(ValueError):
        canonical_identity_from_fields(
            doi=None,
            title="short",
        )


def test_selector_record_rejects_scientific_fields():
    with pytest.raises(ValidationError):
        FreshCIdentityRecord.model_validate(
            {
                "canonical_id": "doi:10.1/example",
                "catalog_work_id": "work:1",
                "identity_method": "doi_family",
                "title": "must be rejected",
            }
        )
    with pytest.raises(ValidationError):
        FreshCIdentityRecord.model_validate(
            {
                "canonical_id": "doi:10.1/example",
                "catalog_work_id": "work:1",
                "identity_method": "doi_family",
                "abstract": "must be rejected",
            }
        )


def test_catalog_projection_discards_scientific_metadata():
    work = SimpleNamespace(
        doi="10.1234/example",
        title="Long title used only if DOI is absent",
        work_id="catalog_work:opaque",
        abstract="scientific content must not cross selector boundary",
        citation_count=999,
        open_access_url="https://example.org/paper.pdf",
    )
    row = project_catalog_identity(work)
    assert row.model_dump() == {
        "canonical_id": "doi:10.1234/example",
        "catalog_work_id": "catalog_work:opaque",
        "identity_method": "doi_family",
    }


def test_blind_order_is_deterministic_identity_only():
    candidates = [
        FreshCIdentityRecord(
            canonical_id=f"doi:10.1000/{suffix}",
            catalog_work_id=f"work:{suffix}",
            identity_method="doi_family",
        )
        for suffix in ("c", "a", "b")
    ]
    ledger = _ledger("doi:10.1000/history")

    first = rank_fresh_identities(
        candidates=candidates,
        historical_ledger=ledger,
    )
    second = rank_fresh_identities(
        candidates=list(reversed(candidates)),
        historical_ledger=ledger,
    )
    assert [row.canonical_id for row in first] == [
        row.canonical_id for row in second
    ]
    for row in first:
        expected = hashlib.sha256(
            (
                FRESH_C_BLIND_ORDER_NAMESPACE
                + "\0"
                + row.canonical_id
            ).encode("utf-8")
        ).hexdigest()
        assert row.score_sha256 == expected


def test_historical_overlap_is_excluded_before_ranking():
    ledger = _ledger("doi:10.1000/seen")
    candidates = [
        FreshCIdentityRecord(
            canonical_id="doi:10.1000/seen",
            catalog_work_id="work:seen",
            identity_method="doi_family",
        ),
        FreshCIdentityRecord(
            canonical_id="doi:10.1000/fresh",
            catalog_work_id="work:fresh",
            identity_method="doi_family",
        ),
    ]
    ranked = rank_fresh_identities(
        candidates=candidates,
        historical_ledger=ledger,
    )
    assert [row.canonical_id for row in ranked] == [
        "doi:10.1000/fresh"
    ]


def test_duplicate_candidate_identity_fails_closed():
    ledger = _ledger("doi:10.1000/history")
    row = FreshCIdentityRecord(
        canonical_id="doi:10.1000/fresh",
        catalog_work_id="work:fresh",
        identity_method="doi_family",
    )
    with pytest.raises(ValueError):
        rank_fresh_identities(
            candidates=[row, row],
            historical_ledger=ledger,
        )


def test_incomplete_historical_ledger_is_rejected():
    good = _ledger("doi:10.1000/history").model_dump(mode="json")
    good["completeness_asserted"] = False
    with pytest.raises(ValidationError):
        HistoricalExclusionLedger.model_validate(good)


def test_caller_supplied_blind_namespace_is_rejected():
    ledger = _ledger("doi:10.1000/history")
    row = FreshCIdentityRecord(
        canonical_id="doi:10.1000/fresh",
        catalog_work_id="work:fresh",
        identity_method="doi_family",
    )
    with pytest.raises(ValueError):
        rank_fresh_identities(
            candidates=[row],
            historical_ledger=ledger,
            namespace="caller-controlled-basis",
        )


def test_preconsumption_guard_blocks_semantic_operations():
    assert_preconsumption_operation_allowed("pdf_byte_download")
    assert_preconsumption_operation_allowed("sha256_hashing")
    with pytest.raises(PermissionError):
        assert_preconsumption_operation_allowed("pdf_text_extraction")
    with pytest.raises(PermissionError):
        assert_preconsumption_operation_allowed("hypothesis_evaluation")
    with pytest.raises(PermissionError):
        assert_preconsumption_operation_allowed("unknown_operation")


def test_semantic_state_is_fail_closed():
    assert_semantically_unread(FreshCPreConsumptionSemanticState())
    with pytest.raises(ValidationError):
        FreshCPreConsumptionSemanticState.model_validate(
            {
                "fresh_reserve_c_consumed": False,
                "semantic_read_performed": True,
            }
        )


def test_fresh_module_has_no_network_or_scientific_selector_dependency():
    source = inspect.getsource(fresh_c)
    for token in (
        "corpus_acquisition.candidate_selection",
        "from requests",
        "import requests",
        "from urllib.request",
        "urlopen(",
        "import httpx",
        "import aiohttp",
    ):
        assert token not in source


def test_blind_ranker_source_has_no_scientific_or_access_scoring():
    source = inspect.getsource(rank_fresh_identities)
    for token in (
        ".title",
        ".abstract",
        "citation_count",
        "matched_axes",
        "hypothesis",
        "novelty",
        "direction",
        "open_access",
    ):
        assert token not in source


def test_protocol_is_nonactivating_preregistration_only():
    protocol = load_and_validate_protocol(PROTOCOL)
    assert protocol.stage == "C0.1A"
    assert protocol.status == "PREREGISTRATION_ONLY"
    assert protocol.activation_preconditions_required == [
        "I0_FROZEN",
        "C0_0_EXISTING_RESERVE_PROVENANCE_AUDIT_PASS",
    ]
    assert protocol.discovery_scope_policy.providers == [
        "semantic_scholar",
        "crossref",
    ]
    assert protocol.discovery_scope_policy.broad_queries == [
        "surface enhanced Raman spectroscopy gold silver",
        "SERS gold silver",
        "surface enhanced Raman spectroscopy Au Ag",
        "SERS Au Ag",
    ]
    assert protocol.discovery_scope_policy.axis_queries_allowed is False
    assert protocol.discovery_scope_policy.hypothesis_terms_allowed is False
    assert protocol.discovery_scope_policy.novelty_gap_terms_allowed is False
    assert (
        protocol.discovery_scope_policy.title_or_abstract_scoring_allowed
        is False
    )
    assert (
        protocol.discovery_scope_policy.results_per_query_defined_in_preregistration
        is False
    )
    assert (
        protocol.discovery_scope_policy.search_depth_must_be_frozen_before_live_discovery
        is True
    )
    assert (
        protocol.blind_ordering_policy.ordering_input_fields
        == ["canonical_id"]
    )
    assert (
        protocol.blind_ordering_policy.scientific_fields_used
        is False
    )
    assert (
        protocol.blind_ordering_policy.target_count_defined_in_preregistration
        is False
    )
    assert protocol.access_failure_policy.oa_only_automatic_acquisition is True
    assert protocol.access_failure_policy.access_availability_used_for_blind_score is False
    assert (
        protocol.access_failure_policy.unresolved_or_download_failed_behavior
        == "record_inaccessible_then_continue_frozen_blind_order"
    )
    assert (
        protocol.access_failure_policy.replacement_basis
        == "next_identity_in_frozen_blind_order"
    )
    assert protocol.access_failure_policy.manual_replacement_allowed is False
    assert (
        protocol.access_failure_policy.scientific_content_based_replacement_allowed
        is False
    )
    assert protocol.access_failure_policy.failed_candidate_rank_is_reassigned is False
    assert protocol.reuse_policy.scientific_candidate_selector_reused is False
    assert protocol.safety.fresh_c_stage_activated is False
    assert protocol.safety.live_discovery_started is False
    assert protocol.safety.live_selection_started is False
    assert protocol.safety.live_acquisition_started is False
    assert protocol.safety.fresh_reserve_c_consumed is False
    assert protocol.safety.semantic_read_performed is False
    assert protocol.safety.network_calls == 0
    assert protocol.safety.llm_calls == 0
    assert protocol.safety.automatic_next_stage_authorized is False
    assert protocol.safety.stop_after_preregistration_freeze is True


def test_ranker_has_no_target_count_parameter():
    signature = inspect.signature(rank_fresh_identities)
    assert "limit" not in signature.parameters
    assert "count" not in signature.parameters
    assert "target_count" not in signature.parameters
