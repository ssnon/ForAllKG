from pathlib import Path

import pytest

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_content_acquisition_v1 import (
    canonical_id_to_doi,
    load_and_validate_protocol,
    locator_record_to_minimal_work,
)


def test_c01d_protocol_is_blind_first_25_successes():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_content_acquisition_v1_protocol.json")
    )
    assert p.upstream_blind_queue_count == 599
    assert p.target_successful_pdf_count == 25
    assert p.maximum_identity_attempts == 599
    assert p.selection_rule == (
        "first_25_successful_verified_oa_pdfs_in_frozen_blind_order"
    )
    assert p.manual_candidate_replacement_allowed is False
    assert p.hypothesis_aware_selection_allowed is False


def test_c01d_reuses_only_public_oa_resolution_and_pdf_validation():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_content_acquisition_v1_protocol.json")
    )
    assert p.use_unpaywall is True
    assert p.use_openalex is True
    assert p.use_catalog_open_access_url is True
    assert p.require_pdf_magic is True
    assert p.paywall_bypass_allowed is False
    assert p.pdf_text_extraction_allowed is False
    assert p.pdf_semantic_read_allowed is False


def test_canonical_doi_recovery():
    assert canonical_id_to_doi("doi:10.1000/ABC") == "10.1000/abc"
    assert canonical_id_to_doi("openalex:W1") is None


def test_locator_conversion_does_not_require_or_propagate_title():
    work = locator_record_to_minimal_work(
        {
            "canonical_id": "doi:10.1000/test",
            "doi": "10.1000/test",
            "url": "https://example.org/landing",
            "open_access_url": "https://example.org/test.pdf",
            "provider_ids": {"openalex": "W123"},
            "title": "THIS MUST NOT BE USED",
        }
    )
    assert work.title == "sealed_identity:doi:10.1000/test"
    assert "THIS MUST NOT BE USED" not in work.title
    assert work.doi == "10.1000/test"
    assert work.provider_ids["openalex"] == "W123"


def test_c01d_does_not_consume_reserve_c():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_content_acquisition_v1_protocol.json")
    )
    assert p.fresh_reserve_c_consumed_on_success is False
    assert p.scientific_reassessment_allowed is False
    assert p.positive_evidence_promotion_allowed is False
    assert p.llm_calls == 0
    assert p.automatic_c1_transition_allowed is False
