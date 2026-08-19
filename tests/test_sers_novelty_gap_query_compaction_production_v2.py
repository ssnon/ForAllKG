from dac_her.novelty_gap_analysis import (
    _query_has_incomplete_tail,
    _query_terms,
)
SAMPLES = [
    (
        "In Au and Ag architectures containing polymer-separated inner and "
        "outer Ag layers connected by an Ag bridge, the separated Ag layers "
        "and bridge support coupled plasmonic modes that generate localized "
        "electromagnetic hotspots."
    ),
    (
        "The effect of nanogap size on SERS performance depends on comparable "
        "analyte access to the relevant hotspots, with the gap geometry "
        "influencing both electromagnetic coupling and molecular sampling."
    ),
    (
        "For matched Ag shell-coated core-satellite nanostructures and "
        "comparable analyte loading, SERS measured in solution after analyte "
        "mixing differs from SERS measured after drying the same preparation."
    ),
    (
        "Changing from solution-phase analyte mixing to dried-substrate "
        "preparation produces a qualitative change in the SERS spectral "
        "intensity distribution or peak profile for Ag shell-coated "
        "core-satellite measurements."
    ),
]


def test_production_query_terms_preserves_gap_geometry_predicate() -> None:
    query = _query_terms(SAMPLES[1])
    assert "gap geometry influencing" in query
    assert "electromagnetic coupling" in query
    assert "molecular sampling" in query
    assert _query_has_incomplete_tail(query) is False


def test_production_query_terms_preserves_solution_drying_contrast() -> None:
    query = _query_terms(SAMPLES[2])
    assert "solution after analyte mixing" in query
    assert "after drying the same preparation" in query
    assert _query_has_incomplete_tail(query) is False


def test_production_query_terms_avoids_dangling_preposition() -> None:
    query = _query_terms(SAMPLES[3])
    assert not query.endswith(" for")
    assert _query_has_incomplete_tail(query) is False


def test_production_short_claim_is_preserved_as_normalized_source_tokens() -> None:
    text = "Solution-phase SERS differs after drying the same preparation."
    assert _query_terms(text) == (
        "solution-phase sers differs after drying the same preparation"
    )
