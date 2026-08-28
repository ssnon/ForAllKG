from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)
from pipeline_core.discovery.question_axis_responsiveness import (
    summarize_question_axis_two_pass,
)
from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisTwoPassStability,
)
from pipeline_core.discovery.question_axis_responsiveness_prompt import (
    QuestionAxisResponsivenessPrompt,
    SYSTEM_PROMPT as AXIS_SYSTEM_PROMPT,
)
from pipeline_core.discovery.question_task_preservation_policy import (
    TaskPreservationAssessment,
    classify_task_preservation,
)


QUESTION_HYPOTHESIS_PROMPT_VERSION = (
    "question-hypothesis-responsiveness-v1"
)


HYPOTHESIS_ADAPTER_SYSTEM_PROMPT = (
    AXIS_SYSTEM_PROMPT
    + """

HYPOTHESIS ADAPTER
==================
For this review the candidate is a compiled scientific hypothesis rather
than a planner DiscoveryAxis.

Apply exactly the same requested-variable preservation,
requested-outcome preservation, relation-nucleus preservation, axis-role,
and overall-status semantics defined above.

Judge the candidate's PRIMARY scientific task from its title and
hypothesis_statement.

The inferential_bridge is supporting context only. It must not rescue a
candidate whose primary hypothesis replaces the user's requested relation.

Do not require or infer synthetic subject/relation/object fields.
Do not reward novelty, plausibility, grounding, or mechanistic interest
when the primary task is replaced.
"""
)


class HypothesisResponsivenessBackendProtocol(
    Protocol
):
    def review(
        self,
        prompt: Any,
        *,
        review_pass_index: int,
        debug_path: str | None = None,
    ) -> Any:
        ...


def _compact_json(
    payload: dict[str, Any],
) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


class QuestionHypothesisResponsivenessPromptAssembler:
    """
    Thin representation adapter.

    Evaluation semantics and output contracts remain those of the existing
    Question↔Axis responsiveness critic. Only the candidate representation
    changes from structured DiscoveryAxis S/R/O fields to the compiled
    hypothesis text already produced by Alpha6.
    """

    def build(
        self,
        *,
        question: str,
        hypothesis: HypothesisCard,
    ) -> QuestionAxisResponsivenessPrompt:

        question = str(
            question
        ).strip()

        if not question:
            raise ValueError(
                "question must be non-empty"
            )

        hypothesis_id = str(
            hypothesis.hypothesis_id
        ).strip()

        if not hypothesis_id:
            raise ValueError(
                "hypothesis_id must be non-empty"
            )

        statement = str(
            hypothesis.hypothesis_statement
        ).strip()

        if not statement:
            raise ValueError(
                "hypothesis_statement must be non-empty"
            )

        payload = {
            "question": question,
            "hypothesis": {
                "hypothesis_id":
                    hypothesis_id,
                "title":
                    str(
                        hypothesis.title
                    ).strip(),
                "hypothesis_statement":
                    statement,
                "hypothesis_type":
                    str(
                        hypothesis.hypothesis_type
                    ),
                "inferential_bridge":
                    str(
                        hypothesis.inferential_bridge
                    ).strip(),
            },
        }

        user_prompt = (
            "QUESTION RESPONSIVENESS INPUT\n"
            "=============================\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

        canonical = _compact_json(
            {
                "prompt_version":
                    QUESTION_HYPOTHESIS_PROMPT_VERSION,
                "system_prompt":
                    HYPOTHESIS_ADAPTER_SYSTEM_PROMPT,
                "user_prompt":
                    user_prompt,
            }
        )

        return (
            QuestionAxisResponsivenessPrompt(
                prompt_version=(
                    QUESTION_HYPOTHESIS_PROMPT_VERSION
                ),
                axis_id=hypothesis_id,
                system_prompt=(
                    HYPOTHESIS_ADAPTER_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                prompt_sha256=_sha256(
                    canonical
                ),
            )
        )


def evaluate_hypothesis_task_preservation(
    *,
    question: str,
    hypothesis: HypothesisCard,
    backend: HypothesisResponsivenessBackendProtocol,
    debug_path_prefix: str | None = None,
) -> tuple[
    TaskPreservationAssessment,
    QuestionAxisTwoPassStability,
]:
    assembler = (
        QuestionHypothesisResponsivenessPromptAssembler()
    )

    prompt = assembler.build(
        question=question,
        hypothesis=hypothesis,
    )

    debug_1 = (
        f"{debug_path_prefix}.pass1.json"
        if debug_path_prefix
        else None
    )

    debug_2 = (
        f"{debug_path_prefix}.pass2.json"
        if debug_path_prefix
        else None
    )

    generation_1 = backend.review(
        prompt,
        review_pass_index=1,
        debug_path=debug_1,
    )

    generation_2 = backend.review(
        prompt,
        review_pass_index=2,
        debug_path=debug_2,
    )

    stability = (
        summarize_question_axis_two_pass(
            generation_1.draft,
            generation_2.draft,
        )
    )

    assessment = classify_task_preservation(
        candidate_id=(
            hypothesis.hypothesis_id
        ),
        quality_eligible=True,
        stability=stability,
    )

    return (
        assessment,
        stability,
    )
