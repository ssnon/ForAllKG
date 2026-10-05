import pytest

from pipeline_core.discovery.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
)


def test_external_novelty_structured_output_cap_defaults_to_8192(monkeypatch):
    monkeypatch.delenv(
        "EXTERNAL_NOVELTY_MAX_COMPLETION_TOKENS",
        raising=False,
    )
    backend = InstructorOpenAICompatibleExternalNoveltyBackend(
        model="test-model",
        api_key="test-key",
    )
    assert backend.max_structured_output_tokens == 8192


def test_external_novelty_structured_output_cap_can_be_overridden(monkeypatch):
    monkeypatch.setenv(
        "EXTERNAL_NOVELTY_MAX_COMPLETION_TOKENS",
        "12288",
    )
    backend = InstructorOpenAICompatibleExternalNoveltyBackend(
        model="test-model",
        api_key="test-key",
    )
    assert backend.max_structured_output_tokens == 12288


def test_external_novelty_structured_output_cap_rejects_too_small_budget(
    monkeypatch,
):
    monkeypatch.setenv(
        "EXTERNAL_NOVELTY_MAX_COMPLETION_TOKENS",
        "512",
    )
    with pytest.raises(ValueError):
        InstructorOpenAICompatibleExternalNoveltyBackend(
            model="test-model",
            api_key="test-key",
        )
