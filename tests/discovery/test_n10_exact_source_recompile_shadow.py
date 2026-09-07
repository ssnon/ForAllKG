from types import SimpleNamespace

from pipeline_core.discovery.novelty_claim_decomposition import (
    _plan_exact_source_atomic_recompile_shadow,
)


def _claim(*, bridge: str = "", anchors: list[str] | None = None):
    return SimpleNamespace(
        local_id="claim_1",
        required_bridge=bridge,
        prior_art_identity_terms=["activation-deactivation compatibility"],
        semantic_fidelity_binding=SimpleNamespace(
            relation_endpoint_anchors=(
                anchors
                if anchors is not None
                else [
                    "Cell survival",
                    "compatibility between pathway activation and deactivation",
                ]
            ),
            scope_qualifier_spans=[],
            directional_qualifier_spans=[],
        ),
    )


def _hypothesis(bridge: str, assumptions: list[str] | None = None):
    return SimpleNamespace(
        inferential_bridge=bridge,
        assumptions=list(assumptions or []),
    )


def test_unique_exact_source_recompile_candidate_is_shadow_only() -> None:
    source = (
        "Scaffold localization modulates activation-deactivation compatibility. "
        "Cell survival is reported to be associated with compatibility "
        "between pathway activation and deactivation."
    )

    result = _plan_exact_source_atomic_recompile_shadow(
        _hypothesis(source),
        _claim(),
        sanitized_required_bridge="",
    )

    assert result["classification"] == "EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert result["candidate_count"] == 1
    assert result["diagnostic_only"] is True
    assert result["production_authority"] is False
    assert result["recompile_performed"] is False
    assert result["candidates"][0]["source_path"] == "inferential_bridge.unit[1]"
    assert result["candidates"][0]["exact_source_text"].startswith("Cell survival")


def test_existing_sanitized_bridge_never_requests_recompile() -> None:
    result = _plan_exact_source_atomic_recompile_shadow(
        _hypothesis(
            "Cell survival is reported to be associated with compatibility "
            "between pathway activation and deactivation."
        ),
        _claim(bridge="already safe"),
        sanitized_required_bridge="already safe",
    )

    assert result["classification"] == "NO_RECOMPILE_NEEDED"
    assert result["candidate_count"] == 0


def test_multiple_exact_candidates_are_never_selected() -> None:
    unit = (
        "Cell survival is associated with compatibility between pathway "
        "activation and deactivation."
    )
    result = _plan_exact_source_atomic_recompile_shadow(
        _hypothesis(unit, [unit]),
        _claim(),
        sanitized_required_bridge="",
    )

    # Identical exact propositions deduplicate deterministically.
    assert result["classification"] == "EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert result["candidate_count"] == 1

    different = (
        "Cell survival remains associated with compatibility between pathway "
        "activation and deactivation."
    )
    result = _plan_exact_source_atomic_recompile_shadow(
        _hypothesis(unit, [different]),
        _claim(),
        sanitized_required_bridge="",
    )
    assert result["classification"] == "AMBIGUOUS_EXACT_SOURCE_CANDIDATES"
    assert result["candidate_count"] == 2


def test_missing_relation_endpoints_cannot_authorize_recompile() -> None:
    result = _plan_exact_source_atomic_recompile_shadow(
        _hypothesis(
            "Cell survival is associated with compatibility between pathway "
            "activation and deactivation."
        ),
        _claim(anchors=[]),
        sanitized_required_bridge="",
    )

    assert result["classification"] == "BINDING_UNUSABLE_NO_RELATION_ENDPOINTS"
    assert result["candidate_count"] == 0


def test_anaphoric_exact_source_is_not_a_candidate() -> None:
    result = _plan_exact_source_atomic_recompile_shadow(
        _hypothesis(
            "Cell survival follows this activation-deactivation relationship."
        ),
        SimpleNamespace(
            local_id="claim_1",
            required_bridge="",
            prior_art_identity_terms=["activation-deactivation relationship"],
            semantic_fidelity_binding=SimpleNamespace(
                relation_endpoint_anchors=[
                    "Cell survival",
                    "activation-deactivation relationship",
                ],
                scope_qualifier_spans=[],
                directional_qualifier_spans=[],
            ),
        ),
        sanitized_required_bridge="",
    )

    assert result["classification"] == "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert result["candidate_count"] == 0
