from __future__ import annotations

import json
from types import SimpleNamespace

from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisResponsivenessDraft,
)
from pipeline_core.discovery.question_hypothesis_responsiveness import (
    QuestionHypothesisResponsivenessPromptAssembler,
    evaluate_hypothesis_task_preservation,
)


def _hypothesis():
    return SimpleNamespace(
        hypothesis_id="hypothesis:test",
        title="Geometry may alter an unrelated saturation regime",
        hypothesis_statement=(
            "Au geometry may shift reporter-concentration saturation."
        ),
        hypothesis_type="design_lever_interaction",
        inferential_bridge=(
            "This is supporting mechanistic context."
        ),
    )


def _draft(
    *,
    status: str,
    role: str,
):
    direct = role == "DIRECT_ANSWER"

    return QuestionAxisResponsivenessDraft(
        requested_variable_preservation=(
            "YES" if direct else "NO"
        ),
        requested_outcome_preservation=(
            "YES" if direct else "NO"
        ),
        relation_nucleus_preservation=(
            "YES" if direct else "NO"
        ),
        axis_role=role,
        overall_status=status,
        rationale=(
            "The candidate was evaluated against the "
            "requested variables, outcome, and relation nucleus."
        ),
    )


class _Backend:
    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.calls = []

    def review(
        self,
        prompt,
        *,
        review_pass_index,
        debug_path=None,
    ):
        self.calls.append(
            review_pass_index
        )

        return SimpleNamespace(
            draft=self.drafts[
                review_pass_index - 1
            ]
        )


def test_prompt_uses_hypothesis_text_without_synthetic_sro():
    prompt = (
        QuestionHypothesisResponsivenessPromptAssembler()
        .build(
            question=(
                "Which excitation wavelength and reporter "
                "combination best fits this Au nanostructure?"
            ),
            hypothesis=_hypothesis(),
        )
    )

    payload = json.loads(
        prompt.user_prompt.split(
            "=============================\n",
            1,
        )[1]
    )

    assert (
        payload["hypothesis"]["hypothesis_id"]
        == "hypothesis:test"
    )

    assert (
        payload["hypothesis"]["hypothesis_statement"]
        == (
            "Au geometry may shift "
            "reporter-concentration saturation."
        )
    )

    assert "proposed_subject" not in prompt.user_prompt
    assert "proposed_relation" not in prompt.user_prompt
    assert "proposed_object" not in prompt.user_prompt


def test_stable_direct_is_eligible():
    backend = _Backend(
        [
            _draft(
                status="PASS",
                role="DIRECT_ANSWER",
            ),
            _draft(
                status="PASS",
                role="DIRECT_ANSWER",
            ),
        ]
    )

    assessment, stability = (
        evaluate_hypothesis_task_preservation(
            question="question",
            hypothesis=_hypothesis(),
            backend=backend,
        )
    )

    assert assessment.task_class == "DIRECT"
    assert assessment.decision_stable is True
    assert stability.stable_role == "DIRECT_ANSWER"
    assert backend.calls == [1, 2]


def test_stable_task_replacement_is_rejected_class():
    backend = _Backend(
        [
            _draft(
                status="FAIL",
                role="TASK_REPLACEMENT",
            ),
            _draft(
                status="FAIL",
                role="TASK_REPLACEMENT",
            ),
        ]
    )

    assessment, _ = (
        evaluate_hypothesis_task_preservation(
            question="question",
            hypothesis=_hypothesis(),
            backend=backend,
        )
    )

    assert assessment.task_class == "TASK_REPLACING"
    assert assessment.decision_stable is True


def test_two_pass_disagreement_fails_closed():
    backend = _Backend(
        [
            _draft(
                status="PASS",
                role="DIRECT_ANSWER",
            ),
            _draft(
                status="FAIL",
                role="TASK_REPLACEMENT",
            ),
        ]
    )

    assessment, _ = (
        evaluate_hypothesis_task_preservation(
            question="question",
            hypothesis=_hypothesis(),
            backend=backend,
        )
    )

    assert assessment.task_class == "UNRESOLVED"
    assert assessment.decision_stable is False
