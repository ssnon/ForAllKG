import pytest
from pydantic import ValidationError

from dac_her.fresh_c_c1b1_reviewer_contract_v1 import (
    DEFAULT_PROTOCOL_PATH,
    EvidenceLocator,
    FreshCFinalAdjudication,
    FreshCPaperReview,
    H1,
    H3,
    HypothesisPaperAssessment,
    validate_protocol,
)

def assessment(hypothesis_id=H1, label="IRRELEVANT", evidence=None):
    return HypothesisPaperAssessment(
        hypothesis_id=hypothesis_id,
        relation_label=label,
        scientific_rationale="Synthetic DEV rationale.",
        evidence=evidence or [],
    )

def evidence():
    return [EvidenceLocator(
        page_number=3,
        evidence_paraphrase="Synthetic evidence supports the tested relation.",
        verbatim_quote="Synthetic short quote only.",
    )]

def test_protocol_is_pre_read_freeze_only():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["scientific_text_read_during_c1b1"] is False
    assert p["network_calls_during_c1b1"] == 0
    assert p["llm_calls_during_c1b1"] == 0
    assert p["automatic_c1b2_transition_allowed"] is False

def test_substantive_relation_requires_page_evidence():
    with pytest.raises(ValidationError):
        assessment(label="DIRECT_PRIOR_ART")

def test_quote_is_bounded_to_20_words():
    with pytest.raises(ValidationError):
        EvidenceLocator(
            page_number=1,
            evidence_paraphrase="Synthetic.",
            verbatim_quote=" ".join(["word"] * 21),
        )

def test_each_paper_reviews_h1_and_h3_exactly_once():
    with pytest.raises(ValidationError):
        FreshCPaperReview(
            reserve_index=1,
            canonical_id="doi:synthetic",
            materialization_mode="DIRECT_ORIGINAL",
            assessments=[assessment(H1), assessment(H1)],
        )
    ok = FreshCPaperReview(
        reserve_index=1,
        canonical_id="doi:synthetic",
        materialization_mode="DIRECT_ORIGINAL",
        assessments=[assessment(H1), assessment(H3)],
    )
    assert {x.hypothesis_id for x in ok.assessments} == {H1, H3}

def test_repaired_provenance_is_exactly_reserve_14():
    with pytest.raises(ValidationError):
        FreshCPaperReview(
            reserve_index=14,
            canonical_id="doi:synthetic",
            materialization_mode="DIRECT_ORIGINAL",
            assessments=[assessment(H1), assessment(H3)],
        )
    ok = FreshCPaperReview(
        reserve_index=14,
        canonical_id="doi:synthetic",
        materialization_mode="STRUCTURALLY_REPAIRED_DERIVATIVE",
        assessments=[assessment(H1), assessment(H3)],
    )
    assert ok.paper_level_completeness_claim_made is False

def test_h2_cannot_enter_paper_review():
    with pytest.raises(ValidationError):
        HypothesisPaperAssessment(
            hypothesis_id="direction_aware_trend_hypothesis:8507f8cadfc46d8d80de",
            relation_label="IRRELEVANT",
            scientific_rationale="Synthetic.",
        )

def test_final_adjudication_has_no_upgrade_or_resurrection():
    result = FreshCFinalAdjudication(
        h1_fresh_c_verdict="FRESH_C_PRESERVES_PRE_C_BOUNDED_EXTENSION",
        h1_rationale="Synthetic H1 conclusion.",
        h3_fresh_c_verdict="FRESH_C_PRESERVES_PRE_C_RELATIONAL_GAP",
        h3_rationale="Synthetic H3 conclusion.",
    )
    assert result.h2_resurrected is False
    assert result.hypothesis_upgrade_performed is False
    assert result.hypothesis_rewrite_performed is False
    assert result.count_threshold_used is False
    assert result.literature_wide_novelty_claim_made is False

def test_direct_prior_art_with_grounding_validates():
    a = assessment(H3, "DIRECT_PRIOR_ART", evidence())
    assert a.evidence[0].page_number == 3
