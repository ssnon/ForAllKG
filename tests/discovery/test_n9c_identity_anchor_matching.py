from pipeline_core.discovery.novelty_closure_review import (
    _abstract_contains_identity_anchor,
)


def matches(
    abstract: str,
    anchor: str,
) -> bool:
    return (
        _abstract_contains_identity_anchor(
            abstract=abstract,
            anchors=(anchor,),
        )
    )


def test_original_exact_identity_phrase_still_matches():
    assert matches(
        (
            "Laser power was explicitly varied while "
            "measuring the SERS response."
        ),
        "laser power",
    )


def test_short_anchor_allows_only_trivial_plural_normalization():
    assert matches(
        (
            "Different laser powers were examined during "
            "the measurement."
        ),
        "laser power",
    )


def test_short_anchor_does_not_allow_word_order_reversal():
    assert not matches(
        (
            "The power-dependent experiment used a laser "
            "source throughout the study."
        ),
        "laser power",
    )


def test_long_anchor_matches_q4_reordered_inflectional_phrase():
    assert matches(
        (
            "The analysis compares stabilization energies "
            "of oxygenated intermediates across the "
            "investigated catalyst configurations."
        ),
        (
            "relative oxygenated intermediate "
            "stabilization"
        ),
    )


def test_long_anchor_requires_at_least_seventy_five_percent_coverage():
    assert not matches(
        (
            "The study reports oxygenated intermediates "
            "under several reaction conditions."
        ),
        (
            "relative oxygenated intermediate "
            "stabilization"
        ),
    )


def test_three_token_anchor_cannot_drop_one_identity_token():
    assert not matches(
        (
            "Interparticle spacing was characterized "
            "under several conditions."
        ),
        (
            "critical interparticle spacing"
        ),
    )


def test_three_token_anchor_can_reorder_all_identity_tokens_locally():
    assert matches(
        (
            "Spacing at the critical interparticle "
            "configuration was measured directly."
        ),
        (
            "critical interparticle spacing"
        ),
    )


def test_matching_terms_scattered_across_abstract_do_not_create_identity():
    assert not matches(
        (
            "Relative catalytic trends were quantified. "
            "Many unrelated structural descriptors, "
            "electronic observables, kinetic measurements, "
            "spectroscopic features, and support properties "
            "were analyzed in separate experiments. "
            "Oxygenated intermediates were also catalogued. "
            "Additional unrelated analyses and long-range "
            "comparisons were then performed before the "
            "authors finally discussed stabilization."
        ),
        (
            "relative oxygenated intermediate "
            "stabilization"
        ),
    )


def test_two_of_four_long_anchor_tokens_are_insufficient():
    assert not matches(
        (
            "Intermediate stabilization was observed, "
            "but no other identity-bearing terms were "
            "reported."
        ),
        (
            "relative oxygenated intermediate "
            "stabilization"
        ),
    )


def test_empty_anchor_never_matches():
    assert not (
        _abstract_contains_identity_anchor(
            abstract="Any scientific abstract.",
            anchors=("",),
        )
    )
