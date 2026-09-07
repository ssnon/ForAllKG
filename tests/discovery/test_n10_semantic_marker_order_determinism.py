from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    _CONDITION_MARKER_RE,
    _markers,
)


def test_casefold_ties_use_raw_surface_as_stable_secondary_key() -> None:
    text = (
        "At comparable state A, the first relation applies; "
        "at comparable state B, the second relation applies."
    )
    assert _markers(_CONDITION_MARKER_RE, text) == [
        "At comparable",
        "at comparable",
    ]


def test_marker_dedup_and_primary_casefold_order_remain_intact() -> None:
    text = (
        "when A holds, When B holds, when C holds; "
        "At comparable D; at comparable E."
    )
    assert _markers(_CONDITION_MARKER_RE, text) == [
        "At comparable",
        "at comparable",
        "When",
        "when",
    ]
