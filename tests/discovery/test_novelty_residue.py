from pipeline_core.discovery.novelty_residue import (
    classify_prior_art_disposition,
)


def test_direct_prior_art_is_saturated():
    assert (
        classify_prior_art_disposition(
            "DIRECT_PRIOR_ART"
        )
        == "SATURATED"
    )


def test_partial_prior_art_is_not_saturated():
    assert (
        classify_prior_art_disposition(
            "PARTIAL_PRIOR_ART"
        )
        == "UNRESOLVED_PARTIAL"
    )


def test_components_only_is_residual():
    assert (
        classify_prior_art_disposition(
            "COMPONENTS_ONLY"
        )
        == "RESIDUAL"
    )


def test_no_direct_match_is_residual():
    assert (
        classify_prior_art_disposition(
            "NO_DIRECT_MATCH_FOUND"
        )
        == "RESIDUAL"
    )


def test_insufficient_metadata_is_unresolved():
    assert (
        classify_prior_art_disposition(
            "INSUFFICIENT_METADATA"
        )
        == "UNRESOLVED"
    )
