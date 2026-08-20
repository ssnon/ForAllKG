from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolioDraft
from dac_her.hypothesis_prompt import HypothesisPrompt
from pipeline_core.llm_telemetry import run_instructor_structured_call


@dataclass(frozen=True)
class HypothesisDraftGeneration:
    draft: HypothesisPortfolioDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class HypothesisDraftBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(self, prompt: HypothesisPrompt) -> HypothesisDraftGeneration: ...

    def repair(
        self,
        prompt: HypothesisPrompt,
        previous_draft: HypothesisPortfolioDraft,
        feedback: str,
    ) -> HypothesisDraftGeneration: ...


class InstructorOpenAICompatibleHypothesisBackend:
    """Structured HypothesisPortfolioDraft generation via an OpenAI-compatible API.

    Imports are intentionally lazy so deterministic v2.6.0/v2.6.1 unit tests do
    not require a configured network client. The default Instructor JSON mode
    mirrors the Graph Explorer backend.
    """

    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        extra_headers: dict[str, str] | None = None,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env} or pass api_key explicitly."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Hypothesis Maker LLM backend requires the installed 'openai' and 'instructor' packages."
            ) from exc

        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            available = sorted(name for name in dir(instructor.Mode) if name.isupper())
            raise ValueError(
                f"Unknown Instructor mode {self.instructor_mode!r}. Available modes include: {available}"
            )

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers

        raw_client = OpenAI(**kwargs)
        self._client = instructor.from_openai(raw_client, mode=mode)
        return self._client

    def _call(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
        semantic_components: Mapping[str, Any] | None = None,
    ) -> HypothesisDraftGeneration:
        client = self._get_client()
        context = {
            **self.telemetry_context,
            "pipeline": "hypothesis_maker",
            "stage": stage,
            "call_kind": stage,
        }
        draft, event = run_instructor_structured_call(
            client.chat.completions,
            model=self.model_name,
            response_model=HypothesisPortfolioDraft,
            messages=messages,
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context=context,
            semantic_components=semantic_components,
        )
        if not isinstance(draft, HypothesisPortfolioDraft):
            draft = HypothesisPortfolioDraft.model_validate(draft)
        return HypothesisDraftGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )

    def generate(self, prompt: HypothesisPrompt) -> HypothesisDraftGeneration:
        return self._call(
            [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            stage="generation",
        )

    def repair(
        self,
        prompt: HypothesisPrompt,
        previous_draft: HypothesisPortfolioDraft,
        feedback: str,
    ) -> HypothesisDraftGeneration:
        previous_payload = previous_draft.model_dump_json(indent=2)
        return self._call(
            [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
                {"role": "assistant", "content": previous_payload},
                {"role": "user", "content": feedback},
            ],
            stage="repair",
            semantic_components={
                "previous_stage_output": previous_payload,
                "issues": feedback,
            },
        )
