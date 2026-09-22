from __future__ import annotations

from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicSpecificationDraft,
    _derive_base_relation_nucleus,
    _single_sentence,
    _surface_contains,
    _validate_atomic_text_contract,
)


def _draft() -> AtomicSpecificationDraft:
    return AtomicSpecificationDraft(
        local_id="a1",
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Accessible hotspot stability controls calibration transfer error.",
        rationale="A branch-specific mediated relation.",
        source_candidate_ids=["CANDIDATE_01"],
        prior_art_identity_terms=["accessible hotspot stability"],
        relation_endpoint_anchors=[
            "accessible hotspot stability",
            "calibration transfer error",
        ],
        directional_qualifier_spans=["controls"],
        required_bridge=(
            "Accessible hotspot stability controls calibration transfer error."
        ),
        observable="cross-batch calibration transfer error",
        predicted_observation=(
            "Cross-batch calibration transfer error decreases as accessible "
            "hotspot stability increases."
        ),
        falsification_condition=(
            "Cross-batch calibration transfer error does not decrease as "
            "accessible hotspot stability increases."
        ),
        search_queries=[
            "accessible hotspot stability calibration transfer error"
        ],
    )


def test_required_bridge_must_be_single_sentence():
    assert _single_sentence(
        "Accessible hotspot stability controls calibration transfer error."
    )
    assert not _single_sentence(
        "Accessible hotspot stability changes. Calibration transfer changes."
    )


def test_surface_contains_is_literal_after_conservative_normalization():
    assert _surface_contains(
        "Accessible-hotspot stability controls calibration transfer error.",
        "accessible hotspot stability",
    )


def test_atomic_text_contract_reuses_existing_branch_sanitizers():
    _validate_atomic_text_contract(_draft())


def test_atomic_text_contract_rejects_prediction_without_branch_identity():
    row = _draft().model_copy(
        update={
            "predicted_observation": (
                "Cross-batch calibration transfer error decreases."
            )
        }
    )
    try:
        _validate_atomic_text_contract(row)
    except ValueError as exc:
        assert "prediction fails existing branch sanitizer" in str(exc)
    else:
        raise AssertionError("expected branch-specific prediction rejection")


def test_base_relation_nucleus_is_compiler_derived_from_scientific_endpoints():
    row = _draft()
    assert _derive_base_relation_nucleus(row) == [
        "accessible hotspot stability",
        "calibration transfer error",
        "controls",
    ]


def test_generic_relation_operators_cannot_be_base_endpoints():
    row = _draft().model_copy(
        update={
            "relation_endpoint_anchors": [
                "associated with lower",
                "will have lower",
            ],
        }
    )
    try:
        _derive_base_relation_nucleus(row)
    except ValueError as exc:
        assert "no scientific content beyond generic relation operators" in str(exc)
    else:
        raise AssertionError("expected generic endpoint rejection")


def test_model_authored_relation_nucleus_terms_are_forbidden():
    payload = _draft().model_dump(mode="json")
    payload["relation_nucleus_terms"] = ["associated with lower"]
    try:
        AtomicSpecificationDraft.model_validate(payload)
    except Exception as exc:
        assert "relation_nucleus_terms" in str(exc)
    else:
        raise AssertionError("expected extra relation_nucleus_terms rejection")
