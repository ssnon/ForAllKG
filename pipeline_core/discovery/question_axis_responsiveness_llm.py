from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Mapping,
)

from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisResponsivenessDraft,
)
from pipeline_core.discovery.question_axis_responsiveness_prompt import (
    QuestionAxisResponsivenessPrompt,
)
from pipeline_core.llm.openrouter_llm import (
    OpenRouterLLM,
)


@dataclass(frozen=True)
class QuestionAxisResponsivenessGeneration:
    draft: QuestionAxisResponsivenessDraft

    requested_model: str
    served_model: str

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class OpenRouterQuestionAxisResponsivenessBackend:
    """Evaluation-only backend.

    This backend has no axis repair, rewrite, selection, or planner mutation
    capability.
    """

    backend_name = (
        "openrouter_question_axis_responsiveness_critic"
    )

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str = "medium",
        telemetry_path:
            str | Path | None = None,
        telemetry_context:
            Mapping[str, Any] | None = None,
        default_debug_path:
            str | Path = (
                "control/"
                "question_axis_responsiveness_raw_response.json"
            ),
    ) -> None:

        self.model_name = str(
            model
        )

        self.temperature = float(
            temperature
        )

        self.reasoning_effort = str(
            reasoning_effort
        )

        self.llm = OpenRouterLLM(
            model=self.model_name,
            application_title=(
                "ForAllKG Question Axis "
                "Responsiveness Diagnostic"
            ),
            default_debug_path=(
                default_debug_path
            ),
            provider=provider,
            reproducible=True,
            zdr=True,
            telemetry_path=(
                telemetry_path
            ),
            telemetry_context={
                "pipeline":
                    "discovery_axis",

                "stage":
                    "question_axis_responsiveness_critic",

                **dict(
                    telemetry_context
                    or {}
                ),
            },
        )

    def review(
        self,
        prompt: QuestionAxisResponsivenessPrompt,
        *,
        review_pass_index: int,
        debug_path:
            str | Path | None = None,
    ) -> QuestionAxisResponsivenessGeneration:

        draft = (
            self.llm.generate_structured(
                system_prompt=(
                    prompt.system_prompt
                ),
                prompt=(
                    prompt.user_prompt
                ),
                response_model=(
                    QuestionAxisResponsivenessDraft
                ),
                temperature=(
                    self.temperature
                ),
                max_tokens=2500,
                reasoning_effort=(
                    self.reasoning_effort
                ),
                debug_path=(
                    debug_path
                ),
                telemetry_components={
                    "axis_id":
                        prompt.axis_id,

                    "prompt_version":
                        prompt.prompt_version,

                    "prompt_sha256":
                        prompt.prompt_sha256,
                },
                telemetry_context={
                    "axis_id":
                        prompt.axis_id,

                    "review_pass_index":
                        int(
                            review_pass_index
                        ),
                },
            )
        )

        usage = self.llm.last_usage

        return (
            QuestionAxisResponsivenessGeneration(
                draft=draft,

                requested_model=(
                    self.model_name
                ),

                served_model=(
                    usage.served_model
                    if usage is not None
                    else self.model_name
                ),

                input_tokens=(
                    usage.input_tokens
                    if usage is not None
                    else None
                ),

                output_tokens=(
                    usage.output_tokens
                    if usage is not None
                    else None
                ),

                total_tokens=(
                    usage.total_tokens
                    if usage is not None
                    else None
                ),
            )
        )
