from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipeline_core.discovery.nonobviousness_operator_generation_contracts import (
    N11OperatorGenerationDraft,
)
from pipeline_core.discovery.nonobviousness_operator_generation_prompt import (
    N11OperatorGenerationPrompt,
)
from pipeline_core.llm.openrouter_llm import (
    OpenRouterLLM,
)


@dataclass(frozen=True)
class N11OperatorGeneration:
    draft: N11OperatorGenerationDraft

    requested_model: str
    served_model: str

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class OpenRouterN11OperatorGenerationBackend:
    backend_name = (
        "openrouter_n11_operator_generator"
    )

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str = "medium",
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
        default_debug_path: str | Path = (
            "control/"
            "n11_operator_generation_raw_response.json"
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
                "ForAllKG N11 Operator-Conditioned Generator"
            ),

            default_debug_path=
                default_debug_path,

            provider=
                provider,

            reproducible=
                True,

            zdr=
                True,

            telemetry_path=
                telemetry_path,

            telemetry_context={
                "pipeline":
                    "nonobviousness",

                "stage":
                    "n11_operator_conditioned_generation",

                **dict(
                    telemetry_context
                    or {}
                ),
            },
        )

    def generate(
        self,
        prompt: N11OperatorGenerationPrompt,
        *,
        generation_pass_index: int = 1,
        debug_path: str | Path | None = None,
    ) -> N11OperatorGeneration:
        if generation_pass_index < 1:
            raise ValueError(
                "generation_pass_index must be positive"
            )

        draft = self.llm.generate_structured(
            system_prompt=
                prompt.system_prompt,

            prompt=
                prompt.user_prompt,

            response_model=
                N11OperatorGenerationDraft,

            temperature=
                self.temperature,

            max_tokens=
                4000,

            reasoning_effort=
                self.reasoning_effort,

            debug_path=
                debug_path,

            telemetry_components={
                "hypothesis_id":
                    prompt.hypothesis_id,

                "requested_operator":
                    prompt.requested_operator,

                "prompt_version":
                    prompt.prompt_version,

                "prompt_sha256":
                    prompt.prompt_sha256,

                "allowed_baseline_count":
                    len(
                        prompt
                        .authority
                        .allowed_baseline_statement_ids
                    ),

                "allowed_supplemental_node_count":
                    len(
                        prompt
                        .authority
                        .allowed_supplemental_node_ids
                    ),

                "allowed_gap_count":
                    len(
                        prompt
                        .authority
                        .allowed_gap_statement_ids
                    ),

                "allowed_shared_component_count":
                    len(
                        prompt
                        .authority
                        .allowed_shared_component_ids
                    ),

                "allowed_supplemental_component_count":
                    len(
                        prompt
                        .authority
                        .allowed_supplemental_only_component_ids
                    ),
            },

            telemetry_context={
                "hypothesis_id":
                    prompt.hypothesis_id,

                "generation_pass_index":
                    int(
                        generation_pass_index
                    ),
            },
        )

        usage = self.llm.last_usage

        return (
            N11OperatorGeneration(
                draft=
                    draft,

                requested_model=
                    self.model_name,

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
