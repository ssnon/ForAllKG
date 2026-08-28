from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.novelty_refinement_contracts import (
    RefinementAttempt,
)
from pipeline_core.discovery.novelty_refinement_runtime import (
    _post_generation_novelty_observability,
)


def _minimal_attempt(**updates):
    body = {
        "original_hypothesis_id":
            "hypothesis:original",
        "candidate_hypothesis_id":
            "hypothesis:candidate",
        "gap_id":
            "gap:test",
        "action":
            "keep",
        "decision":
            "kept_original",
        "original_external_status":
            "PLAUSIBLY_NOVEL",
        "interpretation":
            "test",
    }

    body.update(
        updates
    )

    return RefinementAttempt(
        **body
    )


def test_observability_fields_are_backward_compatible_defaults():
    attempt = _minimal_attempt()

    assert (
        attempt.post_generation_semantic_pass_1
        is None
    )
    assert (
        attempt.post_generation_semantic_pass_2
        is None
    )
    assert (
        attempt.post_generation_semantic_stable
        is None
    )
    assert (
        attempt.post_generation_scientific_action
        is None
    )
    assert (
        attempt.post_generation_selection_class
        is None
    )
    assert (
        attempt.post_generation_scientific_reason_codes
        == []
    )


def test_observability_helper_serializes_existing_assessment_only():
    assessment = SimpleNamespace(
        semantic_pass_1=SimpleNamespace(
            overall_tier="HIGH",
        ),
        semantic_pass_2=SimpleNamespace(
            overall_tier="HIGH",
        ),
        action_decision=SimpleNamespace(
            action="KEEP_ELIGIBLE",
            selection_class="ELIGIBLE",
            reason_codes=[
                "STABLE_SEMANTIC_HIGH",
                "NON_DESTRUCTIVE_EXTERNAL_STATUS",
            ],
        ),
    )

    result = (
        _post_generation_novelty_observability(
            assessment
        )
    )

    assert result == {
        "post_generation_semantic_pass_1":
            "HIGH",
        "post_generation_semantic_pass_2":
            "HIGH",
        "post_generation_semantic_stable":
            True,
        "post_generation_scientific_action":
            "KEEP_ELIGIBLE",
        "post_generation_selection_class":
            "ELIGIBLE",
        "post_generation_scientific_reason_codes":
            [
                "STABLE_SEMANTIC_HIGH",
                "NON_DESTRUCTIVE_EXTERNAL_STATUS",
            ],
    }


def test_observability_survives_model_copy_final_id_binding_style():
    attempt = _minimal_attempt()

    observed = attempt.model_copy(
        update={
            "post_generation_semantic_pass_1":
                "HIGH",
            "post_generation_semantic_pass_2":
                "HIGH",
            "post_generation_semantic_stable":
                True,
            "post_generation_scientific_action":
                "KEEP_ELIGIBLE",
            "post_generation_selection_class":
                "ELIGIBLE",
            "post_generation_scientific_reason_codes":
                ["STABLE_SEMANTIC_HIGH"],
        }
    )

    rebound = observed.model_copy(
        update={
            "final_hypothesis_id":
                "hypothesis:final",
        }
    )

    assert (
        rebound.final_hypothesis_id
        == "hypothesis:final"
    )
    assert (
        rebound.post_generation_semantic_pass_1
        == "HIGH"
    )
    assert (
        rebound.post_generation_semantic_pass_2
        == "HIGH"
    )
    assert (
        rebound.post_generation_semantic_stable
        is True
    )
    assert (
        rebound.post_generation_scientific_action
        == "KEEP_ELIGIBLE"
    )
    assert (
        rebound.post_generation_selection_class
        == "ELIGIBLE"
    )



def test_runtime_captures_post_generation_observability_for_both_candidate_paths():
    from pathlib import Path

    runtime = Path(
        "pipeline_core/discovery/"
        "novelty_refinement_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    compact = "".join(
        runtime.split()
    )

    assert (
        "post_generation_observability_by_candidate_id"
        "[reaxis_card.hypothesis_id]="
        in compact
    )

    assert (
        "post_generation_observability_by_candidate_id"
        "[refined.hypothesis_id]="
        in compact
    )

    assert (
        "POST_GENERATION_OBSERVABILITY_BINDING"
        in runtime
    )
