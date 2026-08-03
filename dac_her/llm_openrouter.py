from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError


load_dotenv()
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
        provider: str | None = None,
        reproducible: bool = True,
        zdr: bool = True,
    ) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not defined.")

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
                "X-OpenRouter-Title": "GraphAgents DAC-HER",
                "X-OpenRouter-Metadata": "enabled",
            },
        )
        self.last_usage: LLMUsage | None = None
        self.last_call_metadata: dict[str, object] = {}

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
    ) -> str:
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
        if not content or not content.strip():
            raise RuntimeError("OpenRouter returned empty content.")
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
                else Path(
                    "data_dac/debug/last_invalid_structured_response.json"
                )
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
    ) -> T:
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
                    "schema": response_model.model_json_schema(),
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
        return self._validate_structured_content(
            content=response.choices[0].message.content,
            response_model=response_model,
            debug_path=debug_path,
        )

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
                    "schema": response_model.model_json_schema(),
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
        return self._validate_structured_content(
            content=response.choices[0].message.content,
            response_model=response_model,
            debug_path=debug_path,
        )
