from __future__ import annotations

import json
from types import SimpleNamespace

from dac_her.llm_telemetry import (
    append_usage_event,
    build_usage_event,
    component_fingerprint,
    estimate_prompt_components,
    infer_prompt_context,
    instructor_create_with_completion,
)
from dac_her.llm_telemetry_report import (
    summarize_usage_events,
    summarize_usage_file,
)


def _completion(*, prompt_tokens=100, completion_tokens=20, model="served-model"):
    return SimpleNamespace(
        id="resp-1",
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
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


def test_report_summarizes_stage_cost_and_repeated_serialization():
    first = build_usage_event(
        requested_model="m",
        completion=_completion(prompt_tokens=100, completion_tokens=10),
        system_prompt="same system",
        user_prompt="first prompt",
        response_schema={"type": "object"},
        context={"pipeline": "hypothesis_maker", "stage": "generation"},
    ).to_dict()
    second = build_usage_event(
        requested_model="m",
        completion=_completion(prompt_tokens=120, completion_tokens=15),
        system_prompt="same system",
        user_prompt="second prompt",
        response_schema={"type": "object"},
        context={"pipeline": "hypothesis_maker", "stage": "generation"},
    ).to_dict()

    summary = summarize_usage_events([first, second])
    assert summary["calls"] == 2
    assert summary["provider_input_tokens"] == 220
    assert summary["provider_total_tokens"] == 245
    stage = summary["by_stage"]["hypothesis_maker:generation"]
    assert stage["calls"] == 2
    assert stage["input_tokens"] == 220
    assert stage["output_tokens"] == 25
    assert stage["total_tokens"] == 245
    assert stage["source_bearing_calls"] == 0
    assert stage["provider_input_to_estimated_source_ratio"] is None
    repeated_names = {row["component"] for row in summary["repeated_serialization"]}
    assert "system" in repeated_names
    assert "schema" in repeated_names


def test_run_instructor_structured_call_records_raw_usage(tmp_path):
    from dac_her.llm_telemetry import run_instructor_structured_call

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
    from dac_her.llm_telemetry import run_instructor_structured_call

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


def test_source_overhead_ratio_ignores_non_source_calls():
    events = [
        {
            "pipeline": "extraction",
            "stage": "graph_generation",
            "provider_input_tokens": 100,
            "provider_output_tokens": 5,
            "provider_total_tokens": 105,
            "estimated_components": {
                "source": {
                    "estimated_tokens": 20,
                    "fingerprint": "source-fp",
                    "counted_in_estimated_sum": False,
                }
            },
        },
        {
            "pipeline": "hypothesis_validation",
            "stage": "semantic_critic",
            "provider_input_tokens": 900,
            "provider_output_tokens": 10,
            "provider_total_tokens": 910,
            "estimated_components": {},
        },
    ]
    summary = summarize_usage_events(events)
    assert summary["provider_input_to_estimated_source_ratio"] == 5.0
    assert summary["source_scoped_overhead"] == {
        "calls": 1,
        "provider_input_tokens": 100,
        "estimated_source_tokens": 20,
    }


def test_v11_summary_normalizes_patch_stage_and_preserves_legacy_outcome():
    event = {
        "schema_version": "llm-call-usage-v1",
        "call_id": "call-old",
        "pipeline": "extraction",
        "stage": "KnowledgeGraphPatch",
        "outcome": "success",
        "provider_input_tokens": 400,
        "provider_output_tokens": 20,
        "provider_total_tokens": 420,
        "estimated_components": {
            "source": {
                "estimated_tokens": 100,
                "fingerprint": "src",
                "counted_in_estimated_sum": False,
            }
        },
    }
    summary = summarize_usage_events([event])
    assert summary["schema_version"] == "llm-telemetry-summary-v1.1"
    assert summary["call_outcome_counts"] == {"success": 1}
    assert "extraction:semantic_patch" in summary["by_stage"]
    assert summary["source_overhead_by_stage"][
        "extraction:semantic_patch"
    ]["provider_input_to_estimated_source_ratio"] == 4.0


def test_artifact_resolution_join_separates_call_and_artifact_outcome(tmp_path):
    from dac_her.llm_telemetry import append_extraction_artifact_resolutions

    call1 = build_usage_event(
        requested_model="m",
        completion=_completion(prompt_tokens=100, completion_tokens=10),
        system_prompt="system",
        user_prompt="CORE_TEXT:\nsource",
        response_schema={"type": "object"},
        context={"pipeline": "extraction", "stage": "graph_generation"},
    )
    call2 = build_usage_event(
        requested_model="m",
        completion=_completion(prompt_tokens=50, completion_tokens=5),
        system_prompt="patch",
        user_prompt="CORE_TEXT:\nsource",
        response_schema={"type": "object"},
        context={"pipeline": "extraction", "stage": "semantic_patch"},
    )
    path = tmp_path / "calls.jsonl"
    append_usage_event(path, call1)
    append_usage_event(path, call2)
    record = {
        "status": "success",
        "chunk_id": "chunk-1",
        "patch_attempts": 1,
        "attempt_usages": [
            {"telemetry_event": call1.to_dict()},
            {"telemetry_event": call2.to_dict()},
        ],
    }
    written = append_extraction_artifact_resolutions(
        path,
        run_id="run-1",
        paper_id="paper-1",
        materialization_status="complete",
        active_records=[record],
        quarantined_records=[],
        failed_records=[],
    )
    assert written == 2
    summary = summarize_usage_file(path)
    assert summary["calls"] == 2
    assert summary["artifact_resolution_records"] == 2
    assert summary["resolved_calls"] == 2
    assert summary["artifact_outcome_counts"] == {
        "accepted_after_repair": 2
    }
    assert summary["terminal_contribution_counts"] == {
        "non_terminal": 1,
        "terminal": 1,
    }
    assert summary["call_outcome_counts"] == {"success": 2}


def test_rejected_artifact_marks_successful_provider_calls_discarded(tmp_path):
    from dac_her.llm_telemetry import append_extraction_artifact_resolutions

    call = build_usage_event(
        requested_model="m",
        completion=_completion(prompt_tokens=100, completion_tokens=10),
        system_prompt="system",
        user_prompt="CORE_TEXT:\nsource",
        response_schema={"type": "object"},
        context={"pipeline": "extraction", "stage": "graph_generation"},
    )
    path = tmp_path / "calls.jsonl"
    append_usage_event(path, call)
    record = {
        "status": "quarantined",
        "chunk_id": "chunk-1",
        "attempt_usages": [{"telemetry_event": call.to_dict()}],
    }
    append_extraction_artifact_resolutions(
        path,
        run_id="run-1",
        paper_id="paper-1",
        materialization_status="rejected",
        active_records=[],
        quarantined_records=[record],
        failed_records=[],
    )
    summary = summarize_usage_file(path)
    assert summary["call_outcome_counts"] == {"success": 1}
    assert summary["artifact_outcome_counts"] == {"rejected": 1}
    assert summary["terminal_contribution_counts"] == {"discarded": 1}


def test_graph_generation_source_overhead_is_stage_specific():
    rows = [
        {
            "call_id": "g1",
            "pipeline": "extraction",
            "stage": "graph_generation",
            "provider_input_tokens": 750,
            "provider_total_tokens": 800,
            "estimated_components": {
                "source": {
                    "estimated_tokens": 30,
                    "fingerprint": "s1",
                    "counted_in_estimated_sum": False,
                }
            },
        },
        {
            "call_id": "p1",
            "pipeline": "extraction",
            "stage": "KnowledgeGraphPatch",
            "provider_input_tokens": 300,
            "provider_total_tokens": 320,
            "estimated_components": {
                "source": {
                    "estimated_tokens": 30,
                    "fingerprint": "s1",
                    "counted_in_estimated_sum": False,
                }
            },
        },
    ]
    summary = summarize_usage_events(rows)
    assert summary["graph_generation_source_overhead"] == {
        "calls": 1,
        "provider_input_tokens": 750,
        "estimated_source_tokens": 30,
        "provider_input_to_estimated_source_ratio": 25.0,
    }


def test_backfill_resolves_existing_v1_telemetry_without_rerun(tmp_path):
    from dac_her.llm_telemetry_backfill import backfill_extraction_resolutions

    call = build_usage_event(
        requested_model="m",
        completion=_completion(prompt_tokens=100, completion_tokens=10),
        system_prompt="system",
        user_prompt="CORE_TEXT:\nsource",
        response_schema={"type": "object"},
        context={"pipeline": "extraction", "stage": "graph_generation"},
    )
    telemetry = tmp_path / "telemetry.jsonl"
    append_usage_event(telemetry, call)

    data_root = tmp_path / "data_broad"
    paper_root = data_root / "extracted" / "paper-1"
    run_dir = paper_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (paper_root / "latest_run.json").write_text(
        json.dumps({"run_id": "run-1", "run_directory": str(run_dir)}),
        encoding="utf-8",
    )
    (run_dir / "active_chunks.json").write_text(
        json.dumps(
            {
                "paper_id": "paper-1",
                "run_id": "run-1",
                "graph_materialization_status": "complete",
                "chunks": [
                    {
                        "status": "success",
                        "chunk_id": "chunk-1",
                        "attempt_usages": [
                            {"telemetry_event": call.to_dict()}
                        ],
                    }
                ],
                "quarantined_chunks": [],
                "failed_chunks": [],
            }
        ),
        encoding="utf-8",
    )

    report = backfill_extraction_resolutions(
        data_root=data_root,
        paper_ids=["paper-1"],
        telemetry_path=telemetry,
    )
    assert report["resolution_records_appended"] == 1
    second = backfill_extraction_resolutions(
        data_root=data_root,
        paper_ids=["paper-1"],
        telemetry_path=telemetry,
    )
    assert second["resolution_records_appended"] == 0
    assert second["already_resolved_call_ids"] == 1
    summary = summarize_usage_file(telemetry)
    assert summary["resolved_calls"] == 1
    assert summary["artifact_outcome_counts"] == {"accepted": 1}
