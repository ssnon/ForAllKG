from pathlib import Path

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)
from pipeline_core.discovery.nonobviousness_post_generation import (
    assert_candidate_final_authority_equivalent,
)


def _card(
    *,
    hypothesis_id: str,
    observation_id: str,
    criterion_id: str,
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="synthetic",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        source_report_id="report:1",
        source_report_sha256="report-sha",
        title="Moderator changes a local response slope",
        hypothesis_statement=(
            "Under matched conditions, factor M changes "
            "the local slope relating descriptor X to response Y."
        ),
        hypothesis_type="descriptor_mediation",
        premise_statement_ids=[
            "stmt:p1",
            "stmt:p2",
        ],
        gap_statement_ids=[
            "stmt:g1",
        ],
        inferential_bridge=(
            "The evidence motivates testing M as a moderator "
            "of the X-to-Y response rather than only an intercept."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id=observation_id,
                observable=(
                    "Local X-to-Y slopes differ across M."
                ),
                expected_direction="shift",
                rationale=(
                    "A moderation effect predicts "
                    "nonparallel local responses."
                ),
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id=criterion_id,
                observable=(
                    "Local X-to-Y slopes differ across M."
                ),
                falsifying_outcome=(
                    "The local slopes are indistinguishable."
                ),
            )
        ],
        assumptions=[
            "X is measured consistently.",
        ],
        source_paper_ids=[
            "paper:1",
            "paper:2",
        ],
        gap_paper_ids=[
            "paper:3",
        ],
        cross_paper_synthesis=True,
        candidate_dependency="none",
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=2,
            gap_count=1,
            source_paper_count=2,
            candidate_premise_count=0,
            reported_premise_count=2,
            synthesis_premise_count=0,
        ),
    )


def test_identity_only_rebinding_is_authority_equivalent():
    candidate = _card(
        hypothesis_id="hypothesis:candidate",
        observation_id="prediction:candidate",
        criterion_id="falsifier:candidate",
    )

    final = _card(
        hypothesis_id="hypothesis:final",
        observation_id="prediction:final",
        criterion_id="falsifier:final",
    )

    assert_candidate_final_authority_equivalent(
        candidate=candidate,
        final=final,
    )


def test_hypothesis_statement_drift_fails_closed():
    candidate = _card(
        hypothesis_id="hypothesis:candidate",
        observation_id="prediction:candidate",
        criterion_id="falsifier:candidate",
    )

    final = _card(
        hypothesis_id="hypothesis:final",
        observation_id="prediction:final",
        criterion_id="falsifier:final",
    ).model_copy(
        update={
            "hypothesis_statement":
                "A scientifically different hypothesis."
        }
    )

    with pytest.raises(
        ValueError,
        match="changed_fields=.*hypothesis_statement",
    ):
        assert_candidate_final_authority_equivalent(
            candidate=candidate,
            final=final,
        )


def test_inferential_bridge_drift_fails_closed():
    candidate = _card(
        hypothesis_id="hypothesis:candidate",
        observation_id="prediction:candidate",
        criterion_id="falsifier:candidate",
    )

    final = _card(
        hypothesis_id="hypothesis:final",
        observation_id="prediction:final",
        criterion_id="falsifier:final",
    ).model_copy(
        update={
            "inferential_bridge":
                "A different inferential bridge."
        }
    )

    with pytest.raises(
        ValueError,
        match="inferential_bridge",
    ):
        assert_candidate_final_authority_equivalent(
            candidate=candidate,
            final=final,
        )


def test_prediction_content_drift_fails_closed():
    candidate = _card(
        hypothesis_id="hypothesis:candidate",
        observation_id="prediction:candidate",
        criterion_id="falsifier:candidate",
    )

    final = _card(
        hypothesis_id="hypothesis:final",
        observation_id="prediction:final",
        criterion_id="falsifier:final",
    )

    changed_prediction = (
        final.predicted_observations[0]
        .model_copy(
            update={
                "observable":
                    "A scientifically different observable."
            }
        )
    )

    final = final.model_copy(
        update={
            "predicted_observations": [
                changed_prediction
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="predicted_observations",
    ):
        assert_candidate_final_authority_equivalent(
            candidate=candidate,
            final=final,
        )


def test_grounding_or_provenance_drift_fails_closed():
    candidate = _card(
        hypothesis_id="hypothesis:candidate",
        observation_id="prediction:candidate",
        criterion_id="falsifier:candidate",
    )

    final = _card(
        hypothesis_id="hypothesis:final",
        observation_id="prediction:final",
        criterion_id="falsifier:final",
    ).model_copy(
        update={
            "premise_statement_ids": [
                "stmt:other"
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="premise_statement_ids",
    ):
        assert_candidate_final_authority_equivalent(
            candidate=candidate,
            final=final,
        )


def test_enforcement_guard_runs_before_n9_shadow():
    text = Path(
        "scripts/discovery/"
        "enforce_alpha6_nonobviousness.py"
    ).read_text(
        encoding="utf-8"
    )

    guard_call = (
        "        assert_candidate_final_authority_equivalent("
    )

    shadow_call = (
        '            "build_nonobviousness_shadow",'
    )

    assert guard_call in text
    assert shadow_call in text

    assert (
        text.index(guard_call)
        < text.index(shadow_call)
    )


def test_enforcement_audits_final_identity_and_equivalence():
    text = Path(
        "scripts/discovery/"
        "enforce_alpha6_nonobviousness.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        '"final_hypothesis_id":'
        in text
    )

    assert (
        '"candidate_final_authority_equivalent":'
        in text
    )
