from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

from pydantic import BaseModel

# The project pins openai/python-dotenv, but the artifact-test container is
# intentionally minimal. Stub import-only dependencies before importing the
# backend; the test replaces the actual client with a fake below.
if "openai" not in sys.modules:
    openai_stub = ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub
if "dotenv" not in sys.modules:
    dotenv_stub = ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_stub

import pipeline_core.openrouter_llm as llm_module


class TinyDraft(BaseModel):
    value: str


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="resp-openrouter",
            model="served-model",
            usage=SimpleNamespace(
                prompt_tokens=444,
                completion_tokens=21,
                total_tokens=465,
                cost=0.03125,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=256,
                    cache_write_tokens=32,
                ),
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"value":"ok"}'),
                )
            ],
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_openrouter_structured_call_preserves_payload_and_records_telemetry(
    monkeypatch,
    tmp_path,
):
    fake_client = FakeClient()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake_client)

    telemetry_path = tmp_path / "openrouter.jsonl"
    llm = llm_module.OpenRouterLLM(
        model="requested-model",
        telemetry_path=telemetry_path,
        application_title="test-openrouter-client",
        default_debug_path=tmp_path / "invalid.json",
    )
    result = llm.generate_structured(
        system_prompt="system prompt",
        prompt="PAPER_ID:\npaper-1\n\nCHUNK_ID:\nchunk-1\n\nCORE_TEXT:\nsource text",
        response_model=TinyDraft,
        temperature=0.0,
        max_tokens=123,
    )

    assert result == TinyDraft(value="ok")
    sent = fake_client.completions.kwargs
    assert sent["model"] == "requested-model"
    assert sent["max_tokens"] == 123
    assert sent["messages"] == [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": "PAPER_ID:\npaper-1\n\nCHUNK_ID:\nchunk-1\n\nCORE_TEXT:\nsource text",
        },
    ]
    assert sent["response_format"]["json_schema"]["schema"] == TinyDraft.model_json_schema()

    assert llm.last_call_metadata["input_tokens"] == 444
    event = llm.last_call_metadata["telemetry_event"]
    assert event["provider_input_tokens"] == 444
    assert event["provider_output_tokens"] == 21
    assert event["provider_cost_credits"] == 0.03125
    assert event["provider_cached_input_tokens"] == 256
    assert event["provider_cache_write_tokens"] == 32
    assert event["provider_usage_scope"] == "direct_provider_call"
    assert event["paper_id"] == "paper-1"
    assert event["chunk_id"] == "chunk-1"
    assert event["pipeline"] == "extraction"
    assert event["stage"] == "TinyDraft"
    assert event["estimated_components"]["source"]["estimated_tokens"] > 0
    assert event["estimated_components"]["schema"]["fingerprint"]

    persisted = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert persisted["call_id"] == event["call_id"]
    assert "source text" not in telemetry_path.read_text(encoding="utf-8")
