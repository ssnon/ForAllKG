from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from pipeline_core.llm.llm_telemetry import (
    append_usage_event,
    build_usage_event,
    normalize_stage_name,
)


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMUsage:
    requested_model: str
    served_model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


def env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise ValueError(
        f"{name} must be a boolean value, "
        f"received: {value!r}"
    )


class OpenRouterLLM:
    def __init__(
        self,
        model: str,
        *,
        application_title: str,
        default_debug_path: str | Path,
        provider: str | None = None,
        reproducible: bool = True,
        zdr: bool = True,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not defined.")

        title = str(application_title).strip()

        if not title:
            raise ValueError(
                "application_title must be non-empty."
            )

        self.application_title = title
        self.default_debug_path = Path(
            default_debug_path
        )

        self.model = model
        self.provider = (
            provider
            or os.getenv("OPENROUTER_PROVIDER")
            or None)
        self.reproducible = reproducible
        self.zdr = env_bool(
            "OPENROUTER_ZDR",
            default=zdr,
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=300.0,
            max_retries=3,
            default_headers={
                "X-OpenRouter-Title": self.application_title,
                "X-OpenRouter-Metadata": "enabled",
            },
        )
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self.last_usage: LLMUsage | None = None
        self.last_call_metadata: dict[str, object] = {}
        self.last_telemetry_event: dict[str, Any] | None = None

    def _provider_options(self, *, structured: bool) -> dict:
        options = {
            "require_parameters": False,
            "data_collection": os.getenv(
                "OPENROUTER_DATA_COLLECTION",
                "deny",
            ),
            "allow_fallbacks": not self.reproducible,
        }
        if self.zdr:
            options["zdr"] = True
        if self.provider:
            options["only"] = [self.provider]
        return options

    def _record_usage(self, response) -> None:
        usage = response.usage
        self.last_usage = LLMUsage(
            requested_model=self.model,
            served_model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
        print("[LLM]", json.dumps(self.last_usage.__dict__, ensure_ascii=False))

    def _record_call_metadata(self, response) -> None:
        choice = response.choices[0]
        usage = response.usage
        self.last_call_metadata = {
            "requested_model": self.model,
            "served_model": response.model,
            "input_tokens": usage.prompt_tokens if usage is not None else None,
            "output_tokens": usage.completion_tokens if usage is not None else None,
            "total_tokens": usage.total_tokens if usage is not None else None,
            "finish_reason": choice.finish_reason,
        }

    def _record_telemetry(
        self,
        *,
        response: Any,
        system_prompt: str,
        prompt: str,
        response_schema: Any | None,
        semantic_components: Mapping[str, Any] | None,
        extra_messages: list[Mapping[str, Any]] | None,
        elapsed_seconds: float,
        outcome: str,
        rejection_reason: str | None,
        telemetry_context: Mapping[str, Any] | None,
    ) -> None:
        context = dict(self.telemetry_context)
        context.update(dict(telemetry_context or {}))
        event = build_usage_event(
            requested_model=self.model,
            completion=response,
            system_prompt=system_prompt,
            user_prompt=prompt,
            response_schema=response_schema,
            semantic_components=semantic_components,
            extra_messages=extra_messages,
            elapsed_seconds=elapsed_seconds,
            outcome=outcome,
            rejection_reason=rejection_reason,
            context=context,
            provider_usage_scope="direct_provider_call",
        )
        append_usage_event(self.telemetry_path, event)
        payload = event.to_dict()
        self.last_telemetry_event = payload
        # Existing strict_recovery already copies last_call_metadata into each
        # attempt_usages record, so telemetry becomes available there without
        # changing recovery policy or control flow.
        self.last_call_metadata["telemetry_event"] = payload

    @staticmethod
    def _structured_stage(response_model: type[BaseModel]) -> str:
        name = response_model.__name__
        return normalize_stage_name(name, response_model=name) or name

    @staticmethod
    def _image_data_url(path: str | Path) -> str:
        path = Path(path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def generate_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        telemetry_components: Mapping[str, Any] | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> str:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={
                "provider": self._provider_options(structured=False)
            },
        )
        self._record_usage(response)
        self._record_call_metadata(response)
        content = response.choices[0].message.content
        elapsed = time.perf_counter() - started
        call_context = {
            "stage": "text_generation",
            "call_kind": "text",
            **dict(telemetry_context or {}),
        }
        if not content or not content.strip():
            self._record_telemetry(
                response=response,
                system_prompt=system_prompt,
                prompt=prompt,
                response_schema=None,
                semantic_components=telemetry_components,
                extra_messages=None,
                elapsed_seconds=elapsed,
                outcome="error",
                rejection_reason="empty_response",
                telemetry_context=call_context,
            )
            raise RuntimeError("OpenRouter returned empty content.")
        self._record_telemetry(
            response=response,
            system_prompt=system_prompt,
            prompt=prompt,
            response_schema=None,
            semantic_components=telemetry_components,
            extra_messages=None,
            elapsed_seconds=elapsed,
            outcome="success",
            rejection_reason=None,
            telemetry_context=call_context,
        )
        return content.strip()

    def _validate_structured_content(
        self,
        *,
        content: str | None,
        response_model: type[T],
        debug_path: str | Path | None,
    ) -> T:
        if not content:
            raise RuntimeError("Structured request returned empty content.")
        try:
            return response_model.model_validate_json(content)
        except ValidationError as error:
            path = (
                Path(debug_path)
                if debug_path is not None
                else self.default_debug_path
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            # Print a bounded summary so the operator can distinguish malformed
            # JSON, missing required fields, and semantic shape errors without
            # opening a potentially very large raw response first.
            details = str(error)
            if len(details) > 3000:
                details = details[:3000] + "\n... validation output truncated ..."
            print(
                f"\n[VALIDATION ERROR]\n"
                f"Raw response saved to: {path}\n"
                f"Validation details:\n{details}",
                flush=True,
            )
            raise

    def generate_structured(
        self,
        *,
        system_prompt: str,
        prompt: str,
        response_model: type[T],
        temperature: float = 0.0,
        max_tokens: int = 4000,
        reasoning_effort: str = "minimal",
        debug_path: str | Path | None = None,
        telemetry_components: Mapping[str, Any] | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> T:
        schema = response_model.model_json_schema()
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
            extra_body={
                "provider": self._provider_options(structured=True),
                "reasoning": {
                    "effort": reasoning_effort,
                    "exclude": True,
                },
            },
        )
        self._record_usage(response)
        self._record_call_metadata(response)
        call_context = {
            "stage": self._structured_stage(response_model),
            "call_kind": "structured",
            "response_model": response_model.__name__,
            **dict(telemetry_context or {}),
        }
        try:
            result = self._validate_structured_content(
                content=response.choices[0].message.content,
                response_model=response_model,
                debug_path=debug_path,
            )
        except Exception as error:
            self._record_telemetry(
                response=response,
                system_prompt=system_prompt,
                prompt=prompt,
                response_schema=schema,
                semantic_components=telemetry_components,
                extra_messages=None,
                elapsed_seconds=time.perf_counter() - started,
                outcome="validation_error",
                rejection_reason=type(error).__name__,
                telemetry_context=call_context,
            )
            raise
        self._record_telemetry(
            response=response,
            system_prompt=system_prompt,
            prompt=prompt,
            response_schema=schema,
            semantic_components=telemetry_components,
            extra_messages=None,
            elapsed_seconds=time.perf_counter() - started,
            outcome="success",
            rejection_reason=None,
            telemetry_context=call_context,
        )
        return result

    def generate_structured_with_images(
        self,
        *,
        system_prompt: str,
        prompt: str,
        image_paths: list[str | Path],
        response_model: type[T],
        temperature: float = 0.0,
        max_tokens: int = 3000,
        reasoning_effort: str = "minimal",
        debug_path: str | Path | None = None,
        telemetry_components: Mapping[str, Any] | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> T:
        if not image_paths:
            raise ValueError("At least one image path is required.")

        content: list[dict] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._image_data_url(image_path),
                    "detail": "high",
                },
            })

        schema = response_model.model_json_schema()
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
            extra_body={
                "provider": self._provider_options(structured=True),
                "reasoning": {
                    "effort": reasoning_effort,
                    "exclude": True,
                },
            },
        )
        self._record_usage(response)
        self._record_call_metadata(response)
        call_context = {
            "stage": self._structured_stage(response_model),
            "call_kind": "structured_with_images",
            "response_model": response_model.__name__,
            **dict(telemetry_context or {}),
        }
        image_diagnostics = [
            {"name": Path(path).name, "detail": "high"}
            for path in image_paths
        ]
        components = {
            "images": image_diagnostics,
            **dict(telemetry_components or {}),
        }
        try:
            result = self._validate_structured_content(
                content=response.choices[0].message.content,
                response_model=response_model,
                debug_path=debug_path,
            )
        except Exception as error:
            self._record_telemetry(
                response=response,
                system_prompt=system_prompt,
                prompt=prompt,
                response_schema=schema,
                semantic_components=components,
                extra_messages=None,
                elapsed_seconds=time.perf_counter() - started,
                outcome="validation_error",
                rejection_reason=type(error).__name__,
                telemetry_context=call_context,
            )
            raise
        self._record_telemetry(
            response=response,
            system_prompt=system_prompt,
            prompt=prompt,
            response_schema=schema,
            semantic_components=components,
            extra_messages=None,
            elapsed_seconds=time.perf_counter() - started,
            outcome="success",
            rejection_reason=None,
            telemetry_context=call_context,
        )
        return result
