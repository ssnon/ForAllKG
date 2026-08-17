from pathlib import Path

from dac_her.fresh_c_c1a_r1_recovery_v1 import (
    load_and_validate_protocol,
    render_page_bounded_text,
)


def _protocol():
    return load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_c1a_r1_recovery_v1_protocol.json")
    )


def test_recovery_is_post_consumption_not_new_consumption():
    p = _protocol()
    assert p.fresh_reserve_c_already_consumed is True
    assert p.consumption_irreversible is True
    assert p.original_c1a_same_epoch_rerun_allowed is False
    assert p.recovery_same_epoch_rerun_after_start_allowed is False


def test_recovery_keeps_exact_same_source_set():
    p = _protocol()
    assert p.source_identity_count == 25
    assert p.source_identity_set_must_remain_exact is True
    assert p.source_pdf_sha256_set_must_remain_exact is True
    assert p.identity_replacement_allowed is False
    assert p.redownload_allowed is False
    assert p.prior_failed_outputs_reused is False


def test_recovery_has_general_structural_fallback():
    p = _protocol()
    assert p.primary_extractor == "pdfminer_six_full_page_text_v1"
    assert p.structural_repair_tool == "mutool_clean"
    assert (
        p.structural_repair_trigger
        == "primary_structural_failure_or_zero_page_or_zero_text"
    )
    assert p.original_pdf_overwrite_allowed is False
    assert p.repaired_derivative_sha256_required is True


def test_recovery_preserves_conservative_evidence_semantics():
    p = _protocol()
    assert p.direct_positive_evidence_from_materialized_text_allowed_later is True
    assert p.negative_absence_inference_from_any_single_paper_allowed is False
    assert p.repaired_derivative_completeness_claim_allowed is False
    assert p.scientific_reviewer_read_performed_in_recovery is False
    assert p.scientific_adjudication_performed_in_recovery is False
    assert p.hypothesis_state_mutation_allowed is False


def test_recovery_has_no_network_llm_ocr_or_auto_transition():
    p = _protocol()
    assert p.external_literature_lookup_allowed is False
    assert p.network_calls_allowed is False
    assert p.llm_calls == 0
    assert p.ocr_performed is False
    assert p.automatic_c1b_transition_allowed is False


def test_page_bounded_rendering():
    assert (
        render_page_bounded_text(["alpha", "beta"])
        == "[[PAGE 1]]\nalpha\n\n[[PAGE 2]]\nbeta\n"
    )
