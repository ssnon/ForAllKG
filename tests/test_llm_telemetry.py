from __future__ import annotations

import json
from types import SimpleNamespace

from pipeline_core.llm_telemetry import (
    append_usage_event,
    build_usage_event,
    component_fingerprint,
    estimate_prompt_components,
    infer_prompt_context,
    instructor_create_with_completion,
)


def _completion(
    *,
    prompt_tokens=100,
    completion_tokens=20,
    model="served-model",
    cost=None,
    cached_tokens=None,
    cache_write_tokens=None,
):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    if cost is not None:
        usage.cost = cost
    if cached_tokens is not None or cache_write_tokens is not None:
        usage.prompt_tokens_details = SimpleNamespace(
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    return SimpleNamespace(
        id="resp-1",
        model=model,
        usage=usage,
        choices=[SimpleNamespace(finish_reason="stop")],
    )


def test_component_fingerprint_is_canonical_for_mapping_order():
    assert component_fingerprint({"b": 2, "a": 1}) == component_fingerprint(
        {"a": 1, "b": 2}
    )


def test_extraction_sections_are_estimated_without_double_counting():
    prompt = """PAPER_ID:\npaper-1\n\nCHUNK_ID:\nchunk-1\n\nASSET_CONTEXT:\nasset words\n\nVOCABULARY_CONTEXT:\nmetric registry\n\nLEFT_CONTEXT:\nleft words\n\nCORE_TEXT:\nsource words\n\nRIGHT_CONTEXT:\nright words"""
    components, estimated_sum = estimate_prompt_components(
        system_prompt="system",
        user_prompt=prompt,
        response_schema={"type": "object"},
    )

    assert {"source", "vocabulary", "asset_context", "left_context", "right_context"} <= set(
        components
    )
    assert estimated_sum == sum(
        components[name].estimated_tokens
        for name in ("system", "user_prompt", "schema")
    )
    assert estimated_sum > components["source"].estimated_tokens


def test_prompt_context_infers_extraction_identifiers():
    prompt = "PAPER_ID:\npaper-1\n\nCHUNK_ID:\nchunk-9\n\nCORE_TEXT:\ntext"
    assert infer_prompt_context(prompt) == {
        "paper_id": "paper-1",
        "chunk_id": "chunk-9",
        "pipeline": "extraction",
    }


def test_provider_usage_is_ground_truth_and_estimate_gap_is_diagnostic():
    event = build_usage_event(
        requested_model="requested-model",
        completion=_completion(prompt_tokens=777, completion_tokens=33),
        system_prompt="system",
        user_prompt="CORE_TEXT:\nsource",
        response_schema={"type": "object"},
        context={"pipeline": "test", "stage": "generation"},
    )

    assert event.provider_input_tokens == 777
    assert event.provider_output_tokens == 33
    assert event.provider_total_tokens == 810
    assert event.token_estimate_gap == 777 - event.estimated_sum
    assert event.pipeline == "test"
    assert event.stage == "generation"



def test_provider_billing_and_cache_metadata_are_observed():
    event = build_usage_event(
        requested_model="requested-model",
        completion=_completion(
            prompt_tokens=777,
            completion_tokens=33,
            cost=0.125,
            cached_tokens=512,
            cache_write_tokens=64,
        ),
        system_prompt="system",
        user_prompt="CORE_TEXT:\nsource",
        response_schema={"type": "object"},
        context={"pipeline": "hypothesis_maker", "stage": "generation"},
    )

    assert event.provider_cost_credits == 0.125
    assert event.provider_cached_input_tokens == 512
    assert event.provider_cache_write_tokens == 64





def test_append_usage_event_writes_no_prompt_text(tmp_path):
    event = build_usage_event(
        requested_model="requested-model",
        completion=_completion(),
        system_prompt="SECRET SYSTEM TEXT",
        user_prompt="CORE_TEXT:\nSECRET SOURCE TEXT",
        response_schema={"type": "object"},
    )
    path = tmp_path / "telemetry.jsonl"
    append_usage_event(path, event)

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["schema_version"] == "llm-call-usage-v1.1"
    assert payload["record_type"] == "call"
    assert payload["call_outcome"] == "success"
    assert "SECRET SYSTEM TEXT" not in text
    assert "SECRET SOURCE TEXT" not in text
    assert payload["estimated_components"]["system"]["fingerprint"]
    assert payload["estimated_components"]["source"]["estimated_tokens"] > 0


def test_instructor_helper_prefers_raw_completion_api_without_second_call():
    class FakeCompletions:
        def __init__(self):
            self.with_completion_calls = 0
            self.create_calls = 0

        def create_with_completion(self, **kwargs):
            self.with_completion_calls += 1
            return "draft", _completion()

        def create(self, **kwargs):
            self.create_calls += 1
            return "unexpected"

    fake = FakeCompletions()
    draft, completion = instructor_create_with_completion(fake, model="m")
    assert draft == "draft"
    assert completion.id == "resp-1"
    assert fake.with_completion_calls == 1
    assert fake.create_calls == 0


def test_instructor_helper_falls_back_to_create():
    class FakeCompletions:
        def __init__(self):
            self.create_calls = 0

        def create(self, **kwargs):
            self.create_calls += 1
            return "draft"

    fake = FakeCompletions()
    draft, completion = instructor_create_with_completion(fake, model="m")
    assert draft == "draft"
    assert completion is None
    assert fake.create_calls == 1




def test_run_instructor_structured_call_records_raw_usage(tmp_path):
    from pipeline_core.llm_telemetry import run_instructor_structured_call

    class ResponseModel:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object", "properties": {"value": {"type": "string"}}}

    class FakeCompletions:
        def create_with_completion(self, **kwargs):
            return {"value": "ok"}, _completion(prompt_tokens=321, completion_tokens=12)

    path = tmp_path / "calls.jsonl"
    draft, event = run_instructor_structured_call(
        FakeCompletions(),
        model="requested-model",
        response_model=ResponseModel,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "premises"},
        ],
        temperature=0.0,
        max_retries=1,
        telemetry_path=path,
        telemetry_context={"pipeline": "hypothesis_maker", "stage": "generation"},
        semantic_components={"premises": "premises"},
    )

    assert draft == {"value": "ok"}
    assert event.provider_input_tokens == 321
    assert event.provider_total_tokens == 333
    assert event.provider_usage_scope == "returned_completion"
    assert event.configured_max_retries == 1
    assert event.estimated_components["premises"].fingerprint
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["stage"] == "generation"
    assert persisted["provider_input_tokens"] == 321


def test_prompt_context_accepts_inline_identifiers():
    prompt = "HYPOTHESIS_ID: hyp-7\nAXIS_ID: axis-3\ntext"
    assert infer_prompt_context(prompt) == {
        "hypothesis_id": "hyp-7",
        "axis_id": "axis-3",
    }


def test_telemetry_sink_failure_is_nonfatal(monkeypatch, tmp_path):
    event = build_usage_event(
        requested_model="m",
        completion=_completion(),
        system_prompt="system",
        user_prompt="prompt",
    )

    def fail_open(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr("dac_her.llm_telemetry.os.open", fail_open)
    with __import__("pytest").warns(RuntimeWarning, match="telemetry append failed"):
        assert append_usage_event(tmp_path / "calls.jsonl", event) is False


def test_ig_section_headings_produce_premise_and_blueprint_diagnostics():
    prompt = """ASSIGNED AXIS
axis_id: axis-1

ELIGIBLE POSITIVE PREMISES
[{\"statement_id\": \"s1\", \"text\": \"evidence\"}]

VALIDATED AXIS-EVIDENCE AUDIT
{\"axis_id\": \"axis-1\"}

BLUEPRINT
{\"axis_id\": \"axis-1\"}"""
    components, _ = estimate_prompt_components(
        system_prompt="system",
        user_prompt=prompt,
        response_schema={"type": "object"},
    )
    assert components["premises"].estimated_tokens > 0
    assert components["axis_audit"].estimated_tokens > 0
    assert components["blueprint"].estimated_tokens > 0
    assert all(
        not components[name].counted_in_estimated_sum
        for name in ("premises", "axis_audit", "blueprint")
    )


def test_prompt_context_infers_axis_from_mapping_serialization():
    prompt = "AXIS\n{'axis_id': 'axis-bridge-4', 'label': 'x'}"
    assert infer_prompt_context(prompt)["axis_id"] == "axis-bridge-4"


def test_instructor_structured_call_preserves_extra_request_kwargs(tmp_path):
    from pipeline_core.llm_telemetry import run_instructor_structured_call

    class ResponseModel:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object"}

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create_with_completion(self, **kwargs):
            self.kwargs = kwargs
            return {"ok": True}, _completion()

    fake = FakeCompletions()
    run_instructor_structured_call(
        fake,
        model="m",
        response_model=ResponseModel,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        temperature=0.0,
        max_retries=2,
        telemetry_path=tmp_path / "calls.jsonl",
        request_kwargs={"max_tokens": 2048},
    )
    assert fake.kwargs["max_tokens"] == 2048
    assert fake.kwargs["max_retries"] == 2


def test_semantic_review_json_is_split_into_diagnostic_components():
    payload = {
        "task": {"task_id": "t1"},
        "eligible_positive_premises": [{"statement_id": "s1", "text": "evidence"}],
        "research_gaps": [{"statement_id": "g1", "text": "gap"}],
        "restricted_nonpremise_statements": [],
        "mechanism_routes": [{"route_id": "r1"}],
        "deterministic_diagnostics": [{"code": "ok"}],
        "hypothesis_portfolio": {"hypotheses": [{"hypothesis_id": "h1"}]},
    }
    prompt = (
        "SEMANTIC REVIEW INPUT\n"
        "=====================\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nOUTPUT REQUIREMENTS\n===================\n- review"
    )
    components, _ = estimate_prompt_components(
        system_prompt="system",
        user_prompt=prompt,
        response_schema={"type": "object"},
    )
    for name in (
        "premises",
        "gaps",
        "mechanism_routes",
        "deterministic_diagnostics",
        "hypothesis_portfolio",
        "task_context",
    ):
        assert components[name].estimated_tokens > 0
        assert components[name].counted_in_estimated_sum is False












