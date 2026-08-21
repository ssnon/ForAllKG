import pytest

from pipeline_core.discovery.hypothesis_llm import (
    HypothesisDraftBackend,
    HypothesisDraftGeneration,
    InstructorOpenAICompatibleHypothesisBackend,
)

from tests.support._hypothesis_v261_fixtures import make_valid_draft


class ProtocolFake:
    backend_name = "fake"
    model_name = "fake-model"

    def generate(self, prompt):
        return HypothesisDraftGeneration(draft=make_valid_draft())

    def repair(self, prompt, previous_draft, feedback):
        return HypothesisDraftGeneration(draft=make_valid_draft())


def test_backend_protocol_is_runtime_checkable():
    assert isinstance(ProtocolFake(), HypothesisDraftBackend)


def test_openai_compatible_backend_is_lazy_and_requires_key_only_on_call():
    backend = InstructorOpenAICompatibleHypothesisBackend(
        model="fake-model",
        api_key="",
    )
    assert backend.model_name == "fake-model"
    with pytest.raises(RuntimeError, match="No API key available"):
        backend._get_client()
