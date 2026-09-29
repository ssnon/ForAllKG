from scripts.discovery.run_s243_portfolio_diversity_prospective_audit import (
    classify_evidence_discrimination,
)


def test_identical_evidence_is_non_discriminative():
    assert classify_evidence_discrimination(
        available_count=5,
        recommendation_count=1,
        signature_count=1,
        dominant_signature_fraction=1.0,
    ) == "NON_DISCRIMINATIVE"


def test_some_variation_but_dominant_is_low():
    assert classify_evidence_discrimination(
        available_count=5,
        recommendation_count=2,
        signature_count=2,
        dominant_signature_fraction=0.8,
    ) == "LOW_DISCRIMINATION"


def test_multiple_non_dominant_signatures_discriminate():
    assert classify_evidence_discrimination(
        available_count=5,
        recommendation_count=2,
        signature_count=4,
        dominant_signature_fraction=0.4,
    ) == "DISCRIMINATION_OBSERVED"
