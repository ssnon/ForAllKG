from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)


QUESTION_AXIS_PROMPT_VERSION = (
    "question-axis-responsiveness-prompt-v1"
)


def _compact_json(
    value: object,
) -> str:
    return json.dumps(
        value,
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


SYSTEM_PROMPT = """
You are a scientific task-responsiveness critic.

Evaluate whether a proposed discovery axis remains responsive to the
scientific QUESTION.

Do NOT judge:
- truth,
- plausibility,
- external novelty,
- prior art,
- experimental feasibility,
- evidence quality.

Judge only QUESTION -> AXIS fidelity.

A discovery axis does NOT need to cover every variable or outcome in a
multi-part question. A scientifically meaningful sub-axis is allowed.

For example, if a question asks which excitation wavelength AND reporter
combination is appropriate, an axis studying how SERS signal varies with
excitation wavelength may be a legitimate DIRECT_ANSWER.

An axis fails when it replaces the task with a substantially different
primary scientific relation.

TASK_REPLACEMENT means that:
- a variable not requested as a primary variable becomes the main variable
  of study; and/or
- the requested outcome or relation is displaced by another main outcome,
  such that answering the axis would not substantially answer the original
  question.

A new moderator, condition, or contextual variable is allowed only when it
is subordinate to the requested relation.

Do not use superficial vocabulary overlap as proof of responsiveness.

Evaluate:

1. requested_variable_preservation
YES:
  The axis directly studies at least one variable requested by the question.
PARTIAL:
  A requested variable remains material but is secondary or indirect.
NO:
  Requested variables are absent or merely contextual.

2. requested_outcome_preservation
YES:
  The axis directly targets an outcome requested by the question.
PARTIAL:
  It targets a closely related downstream or proxy outcome.
NO:
  The requested outcome is displaced.

3. relation_nucleus_preservation
YES:
  Answering the axis would directly answer a material part of the question.
PARTIAL:
  The axis is a legitimate subordinate extension.
NO:
  The axis tests a substantially different scientific relation.

4. axis_role
DIRECT_ANSWER:
  Directly addresses a material part of the question.

SUBORDINATE_EXTENSION:
  Adds a condition, moderator, or context while preserving the requested
  scientific relation as primary.

TASK_REPLACEMENT:
  Substitutes another primary scientific question.

UNRELATED:
  Essentially unrelated.

Overall decision:
- DIRECT_ANSWER -> PASS.
- SUBORDINATE_EXTENSION -> PASS or WARNING.
- TASK_REPLACEMENT -> FAIL.
- UNRELATED -> FAIL.

Be conservative about FAIL. Novel specialization is allowed. Fail only when
the primary task has actually been displaced.

Return only the requested structured response.
""".strip()


@dataclass(frozen=True)
class QuestionAxisResponsivenessPrompt:
    prompt_version: str

    axis_id: str

    system_prompt: str
    user_prompt: str

    prompt_sha256: str


class QuestionAxisResponsivenessPromptAssembler:

    def build(
        self,
        *,
        question: str,
        axis: DiscoveryAxis,
    ) -> QuestionAxisResponsivenessPrompt:

        question = str(
            question
        ).strip()

        if not question:
            raise ValueError(
                "question must be non-empty"
            )

        payload = {
            "question": question,
            "axis": {
                "axis_id": axis.axis_id,
                "label": axis.label,
                "proposed_subject": (
                    axis.proposed_subject
                ),
                "proposed_relation": (
                    axis.proposed_relation
                ),
                "proposed_object": (
                    axis.proposed_object
                ),
                "entry_anchor_label": (
                    axis.entry_anchor_label
                ),
                "exit_anchor_label": (
                    axis.exit_anchor_label
                ),
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
                    QUESTION_AXIS_PROMPT_VERSION,

                "system_prompt":
                    SYSTEM_PROMPT,

                "user_prompt":
                    user_prompt,
            }
        )

        return (
            QuestionAxisResponsivenessPrompt(
                prompt_version=(
                    QUESTION_AXIS_PROMPT_VERSION
                ),
                axis_id=axis.axis_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                prompt_sha256=_sha256(
                    canonical
                ),
            )
        )
