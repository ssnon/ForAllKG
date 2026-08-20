from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from pipeline_core.discovery.hypothesis_semantic_contracts import HypothesisSemanticReviewDraft
from pipeline_core.discovery.hypothesis_semantic_prompt import HypothesisSemanticPrompt
from pipeline_core.llm_telemetry import run_instructor_structured_call


@dataclass(frozen=True)
class HypothesisSemanticGeneration:
    draft: HypothesisSemanticReviewDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class HypothesisSemanticBackend(Protocol):
    backend_name: str
    model_name: str

    def review(
        self,
        prompt: HypothesisSemanticPrompt,
    ) -> HypothesisSemanticGeneration: ...


class InstructorOpenAICompatibleSemanticCriticBackend:
    """Structured semantic review through an OpenAI-compatible API.

    This backend evaluates only. It has no hypothesis rewrite/repair method.
    """

    backend_name = "instructor_openai_compatible_semantic_critic"

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
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Semantic critic backend requires installed 'openai' and 'instructor'."
            ) from exc

        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            available = sorted(name for name in dir(instructor.Mode) if name.isupper())
            raise ValueError(
                f"Unknown Instructor mode {self.instructor_mode!r}. "
                f"Available modes include: {available}"
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

    def review(
        self,
        prompt: HypothesisSemanticPrompt,
    ) -> HypothesisSemanticGeneration:
        client = self._get_client()
        context = {
            **self.telemetry_context,
            "pipeline": "hypothesis_validation",
            "stage": "semantic_critic",
            "call_kind": "review",
        }
        draft, event = run_instructor_structured_call(
            client.chat.completions,
            model=self.model_name,
            response_model=HypothesisSemanticReviewDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context=context,
        )
        if not isinstance(draft, HypothesisSemanticReviewDraft):
            draft = HypothesisSemanticReviewDraft.model_validate(draft)
        return HypothesisSemanticGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )
