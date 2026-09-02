from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipeline_core.discovery.nonobviousness_mechanism_semantic_prompt import (
    MechanismSemanticPrompt,
)
from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismSemanticDraft,
)
from pipeline_core.llm.openrouter_llm import (
    OpenRouterLLM,
)


@dataclass(frozen=True)
class MechanismSemanticGeneration:
    draft: MechanismSemanticDraft

    requested_model: str
    served_model: str

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class OpenRouterMechanismSemanticBackend:
    backend_name = (
        "openrouter_n11_mechanism_semantic_reviewer"
    )

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str = "medium",
        telemetry_path: str | Path | None = None,
        telemetry_context:
            Mapping[str, Any] | None = None,
        default_debug_path:
            str | Path = (
                "control/"
                "n11_mechanism_semantic_raw_response.json"
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
                "ForAllKG N11 Mechanism Semantic Reviewer"
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
                    "n11_mechanism_semantic_review",

                **dict(
                    telemetry_context
                    or {}
                ),
            },
        )

    def review(
        self,
        prompt: MechanismSemanticPrompt,
        *,
        review_pass_index: int = 1,
        debug_path: str | Path | None = None,
    ) -> MechanismSemanticGeneration:
        if review_pass_index < 1:
            raise ValueError(
                "review_pass_index must be positive"
            )

        draft = self.llm.generate_structured(
            system_prompt=
                prompt.system_prompt,

            prompt=
                prompt.user_prompt,

            response_model=
                MechanismSemanticDraft,

            temperature=
                self.temperature,

            max_tokens=
                3000,

            reasoning_effort=
                self.reasoning_effort,

            debug_path=
                debug_path,

            telemetry_components={
                "hypothesis_id":
                    prompt.hypothesis_id,

                "prompt_version":
                    prompt.prompt_version,

                "prompt_sha256":
                    prompt.prompt_sha256,

                "supply_geometry":
                    prompt.supply_geometry,

                "baseline_support_statement_count":
                    len(
                        prompt
                        .baseline_support_statement_ids
                    ),

                "supplemental_mechanism_node_count":
                    len(
                        prompt
                        .supplemental_mechanism_node_ids
                    ),
            },

            telemetry_context={
                "hypothesis_id":
                    prompt.hypothesis_id,

                "review_pass_index":
                    int(
                        review_pass_index
                    ),
            },
        )

        usage = self.llm.last_usage

        return MechanismSemanticGeneration(
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
