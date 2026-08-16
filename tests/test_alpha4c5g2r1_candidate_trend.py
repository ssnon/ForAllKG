from __future__ import annotations

from dac_her.domains.sers_au_ag_trend_alpha4c5g2r1 import (
    _nanogap_size_cue,
    _resolved_claim_control,
    _resolved_direction_shape,
)


def test_plural_gap_sizes_is_quantitative_size():
    text = (
        "Double-shelled nanoboxes with smaller interior "
        "gap sizes exhibit stronger SERS intensity."
    )
    assert _nanogap_size_cue(text) is True
    assert _resolved_claim_control(text) == (
        "nanogap_size",
        "nanogap size",
    )


def test_large_gap_then_gap_decreases_is_size_control():
    text = (
        "The FEM-simulated SERS enhancement increases "
        "from a large Au nanocube gap to a maximum "
        "as the gap decreases."
    )
    assert _nanogap_size_cue(text) is True
    assert _resolved_claim_control(text) == (
        "nanogap_size",
        "nanogap size",
    )


def test_presence_only_claim_remains_non_size():
    text = (
        "The presence of an interior nanogap creates "
        "a strong electromagnetic hot spot."
    )
    assert _nanogap_size_cue(text) is False


def test_historical_direction_fallback_recovers_gap_decrease():
    text = (
        "The SERS enhancement factor increases significantly "
        "as the interior nanogap size decreases."
    )
    assert _resolved_direction_shape(
        text=text,
        control_key="nanogap_size",
    ) == ("negative", "monotonic")


def test_historical_direction_fallback_recovers_smaller_gap():
    text = (
        "Double-shelled Au/Ag nanoboxes with smaller "
        "interior gap sizes exhibit stronger SERS intensity."
    )
    assert _resolved_direction_shape(
        text=text,
        control_key="nanogap_size",
    ) == ("negative", "monotonic")


def test_comparative_pair_fallback():
    text = (
        "The SERS enhancement factor is greater for "
        "the 2-nm interior gap than for the 8-nm gap."
    )
    assert _resolved_claim_control(text) == (
        "nanogap_size",
        "nanogap size",
    )
    assert _resolved_direction_shape(
        text=text,
        control_key="nanogap_size",
    ) == ("negative", "monotonic")
