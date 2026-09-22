from __future__ import annotations

from pipeline_core.discovery.reframing.scientific_synthesis_specification_repair import (
    ScientificSynthesisRepairClaimDiagnostic,
    _ALLOWED_MISSING_FIELDS,
    _canonical_json,
    _stable_id,
)


def test_repair_is_limited_to_frozen_specification_fields():
    assert _ALLOWED_MISSING_FIELDS == {
        "required_bridge",
        "predicted_observation",
        "falsification_condition",
    }


def test_repair_identity_is_deterministic():
    assert _stable_id("x", "a", 1) == _stable_id("x", "a", 1)


def test_repair_identity_changes_with_source():
    assert _stable_id("x", "a") != _stable_id("x", "b")


def test_canonical_json_serializes_nested_pydantic_models():
    diagnostic = ScientificSynthesisRepairClaimDiagnostic(
        claim_id="claim:1",
        claim_text="A conditioned relation should be tested.",
        prior_art_identity_terms=["condition", "response"],
        missing_fields=["required_bridge"],
        reason_codes=["atomic_residue_under_specified", "missing_required_bridge"],
    )
    payload = _canonical_json([diagnostic])
    assert '"claim_id":"claim:1"' in payload
    assert '"missing_fields":["required_bridge"]' in payload
