from types import SimpleNamespace

from pipeline_core.discovery.novelty_claim_decomposition import (
    _plan_exact_source_atomic_recompile_shadow,
    _plan_scope_wrapper_exact_source_atomic_recompile_shadow,
)


SOURCE = (
    "Under the condition of comparable overall metal–hydrogen coupling, "
    "I further propose that a more balanced bonding–antibonding distribution "
    "promotes compatibility between hydrogen adsorption and H2 desorption."
)


def _claim(
    *,
    scope: list[str] | None = None,
    anchors: list[str] | None = None,
    directions: list[str] | None = None,
):
    return SimpleNamespace(
        local_id="claim_2",
        required_bridge="",
        prior_art_identity_terms=["bonding–antibonding distribution"],
        semantic_fidelity_binding=SimpleNamespace(
            relation_endpoint_anchors=(
                anchors
                if anchors is not None
                else [
                    "bonding–antibonding distribution",
                    "hydrogen adsorption",
                    "H2 desorption",
                ]
            ),
            scope_qualifier_spans=(
                scope
                if scope is not None
                else ["at comparable overall metal–hydrogen coupling"]
            ),
            directional_qualifier_spans=(
                directions
                if directions is not None
                else [
                    "a more balanced bonding–antibonding distribution",
                    "promotes",
                ]
            ),
        ),
    )


def _hypothesis(
    bridge: str = SOURCE,
    assumptions: list[str] | None = None,
):
    return SimpleNamespace(
        inferential_bridge=bridge,
        assumptions=list(assumptions or []),
    )


def _plans(hypothesis, claim):
    exact = _plan_exact_source_atomic_recompile_shadow(
        hypothesis,
        claim,
        sanitized_required_bridge="",
    )
    wrapper = _plan_scope_wrapper_exact_source_atomic_recompile_shadow(
        hypothesis,
        claim,
        sanitized_required_bridge="",
        exact_source_plan=exact,
    )
    return exact, wrapper


def test_unique_scope_wrapper_candidate_is_diagnostic_fallback_only() -> None:
    exact, wrapper = _plans(_hypothesis(), _claim())

    assert exact["classification"] == "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert wrapper["classification"] == (
        "SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    )
    assert wrapper["candidate_count"] == 1
    assert wrapper["diagnostic_only"] is True
    assert wrapper["production_authority"] is False
    assert wrapper["production_recompile_enabled"] is False
    assert wrapper["recompile_performed"] is False
    assert wrapper["semantic_scope_synonymy_allowed"] is False

    candidate = wrapper["candidates"][0]
    assert candidate["source_path"] == "inferential_bridge.unit[0]"
    assert candidate["exact_source_text"] == SOURCE
    assert candidate["binding_scope"] == (
        "at comparable overall metal–hydrogen coupling"
    )
    assert candidate["source_scope"] == (
        "Under the condition of comparable overall metal–hydrogen coupling"
    )
    assert candidate["scope_match"]["matched"] is True
    assert candidate["scope_match"]["normalized_source_core"] == (
        "comparable overall metal hydrogen coupling"
    )


def test_different_scope_content_is_not_wrapper_candidate() -> None:
    source = SOURCE.replace(
        "comparable overall metal–hydrogen coupling",
        "higher overall metal–hydrogen coupling",
    )
    _, wrapper = _plans(_hypothesis(source), _claim())

    assert wrapper["classification"] == (
        "NO_SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    )
    assert wrapper["candidate_count"] == 0


def test_different_scientific_factor_is_not_wrapper_candidate() -> None:
    source = SOURCE.replace(
        "overall metal–hydrogen coupling",
        "overall metal–metal coupling",
    )
    _, wrapper = _plans(_hypothesis(source), _claim())

    assert wrapper["classification"] == (
        "NO_SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    )
    assert wrapper["candidate_count"] == 0


def test_unapproved_scope_wrapper_is_not_candidate() -> None:
    source = SOURCE.replace(
        "Under the condition of",
        "Provided",
        1,
    )
    _, wrapper = _plans(_hypothesis(source), _claim())

    assert wrapper["classification"] == (
        "NO_SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    )
    assert wrapper["candidate_count"] == 0


def test_same_scope_wrapper_is_not_a_rebind_candidate() -> None:
    source = SOURCE.replace(
        "Under the condition of",
        "At",
        1,
    )
    exact, wrapper = _plans(_hypothesis(source), _claim())

    assert exact["classification"] == "EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert wrapper["classification"] == "NOT_ELIGIBLE_EXISTING_EXACT_SOURCE_PATH"
    assert wrapper["candidate_count"] == 0


def test_existing_exact_source_candidate_blocks_wrapper_fallback() -> None:
    claim = _claim(
        scope=[
            "Under the condition of comparable overall metal–hydrogen coupling"
        ]
    )
    exact, wrapper = _plans(_hypothesis(), claim)

    assert exact["classification"] == "EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert wrapper["classification"] == "NOT_ELIGIBLE_EXISTING_EXACT_SOURCE_PATH"
    assert wrapper["candidate_count"] == 0


def test_multiple_distinct_wrapper_candidates_are_ambiguous() -> None:
    second = SOURCE.replace(
        "I further propose that a more balanced",
        "A more balanced",
        1,
    )
    exact, wrapper = _plans(
        _hypothesis(SOURCE, [second]),
        _claim(),
    )

    assert exact["classification"] == "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert wrapper["classification"] == (
        "AMBIGUOUS_SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATES"
    )
    assert wrapper["candidate_count"] == 2


def test_scope_cardinality_must_be_exactly_one() -> None:
    exact, wrapper = _plans(
        _hypothesis(),
        _claim(
            scope=[
                "at comparable overall metal–hydrogen coupling",
                "at fixed pressure",
            ]
        ),
    )

    assert exact["classification"] == "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert wrapper["classification"] == "BINDING_UNUSABLE_SCOPE_CARDINALITY"
    assert wrapper["candidate_count"] == 0


def test_anaphoric_source_cannot_become_wrapper_candidate() -> None:
    source = (
        "Under the condition of comparable overall metal–hydrogen coupling, "
        "this bonding–antibonding relationship promotes compatibility between "
        "hydrogen adsorption and H2 desorption."
    )
    claim = _claim(
        anchors=[
            "bonding–antibonding relationship",
            "hydrogen adsorption",
            "H2 desorption",
        ],
        directions=["promotes"],
    )
    _, wrapper = _plans(_hypothesis(source), claim)

    assert wrapper["classification"] == (
        "NO_SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    )
    assert wrapper["candidate_count"] == 0
