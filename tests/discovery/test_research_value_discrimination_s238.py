from scripts.discovery.run_s238_research_value_discrimination_audit import (
    discrimination_status,
)


def test_identical_cards_are_non_discriminative():
    assert discrimination_status(
        card_count=9,
        unique_value_classes=1,
        unique_dimension_signatures=1,
        dominant_signature_fraction=1.0,
    ) == "NON_DISCRIMINATIVE"


def test_dominant_signature_is_low_discrimination():
    assert discrimination_status(
        card_count=10,
        unique_value_classes=2,
        unique_dimension_signatures=3,
        dominant_signature_fraction=0.8,
    ) == "LOW_DISCRIMINATION"


def test_diverse_signatures_are_observed_discrimination():
    assert discrimination_status(
        card_count=10,
        unique_value_classes=3,
        unique_dimension_signatures=4,
        dominant_signature_fraction=0.4,
    ) == "DISCRIMINATION_OBSERVED"


def test_one_card_is_insufficient():
    assert discrimination_status(
        card_count=1,
        unique_value_classes=1,
        unique_dimension_signatures=1,
        dominant_signature_fraction=1.0,
    ) == "INSUFFICIENT_SAMPLE"
