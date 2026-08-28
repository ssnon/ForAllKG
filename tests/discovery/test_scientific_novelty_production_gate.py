from __future__ import annotations

from pipeline_core.discovery.novelty_refinement_runtime import (
    TargetedNoveltyRefinementRuntime,
)
from pipeline_core.discovery.scientific_novelty_production_gate import (
    build_scientific_novelty_fallback_gate,
)


def _batch(selection_class: str):
    return {
        "schema_version":
            "scientific-novelty-action-shadow-batch-v1",
        "source_external_report_id":
            "external:r1",
        "decisions": [
            {
                "hypothesis_id":
                    "hypothesis:h1",
                "decision": {
                    "selection_class":
                        selection_class,
                    "action":
                        (
                            "REAXIS_REQUIRED"
                            if selection_class == "INELIGIBLE"
                            else "KEEP_ELIGIBLE"
                        ),
                    "reason_codes":
                        ["TEST"],
                    "semantic_stable":
                        True,
                    "stable_semantic_tier":
                        (
                            "LOW"
                            if selection_class == "INELIGIBLE"
                            else "HIGH"
                        ),
                    "external_status":
                        "LITERATURE_SUPPORTED_EXTENSION",
                },
            }
        ],
    }


def test_ineligible_decision_disallows_original_fallback():
    gate = build_scientific_novelty_fallback_gate(
        _batch("INELIGIBLE")
    )

    row = gate["gates"][0]

    assert row[
        "fallback_allowed"
    ] is False

    assert row[
        "selection_class"
    ] == "INELIGIBLE"


def test_eligible_decision_preserves_original_fallback():
    gate = build_scientific_novelty_fallback_gate(
        _batch("ELIGIBLE")
    )

    assert gate[
        "gates"
    ][0]["fallback_allowed"] is True


def test_runtime_gate_cannot_reenable_destructive_external_status():
    gate_by_id = {
        "hypothesis:h1": {
            "fallback_allowed": True,
        }
    }

    allowed = (
        TargetedNoveltyRefinementRuntime
        ._original_fallback_allowed(
            "WELL_ESTABLISHED",
            hypothesis_id="hypothesis:h1",
            scientific_gate_by_id=gate_by_id,
        )
    )

    assert allowed is False


def test_runtime_gate_blocks_stable_low_style_fallback():
    gate_by_id = {
        "hypothesis:h1": {
            "fallback_allowed": False,
        }
    }

    allowed = (
        TargetedNoveltyRefinementRuntime
        ._original_fallback_allowed(
            "LITERATURE_SUPPORTED_EXTENSION",
            hypothesis_id="hypothesis:h1",
            scientific_gate_by_id=gate_by_id,
        )
    )

    assert allowed is False


def test_absent_gate_preserves_historical_fallback_behavior():
    allowed = (
        TargetedNoveltyRefinementRuntime
        ._original_fallback_allowed(
            "LITERATURE_SUPPORTED_EXTENSION",
            hypothesis_id="hypothesis:h1",
            scientific_gate_by_id=None,
        )
    )

    assert allowed is True
