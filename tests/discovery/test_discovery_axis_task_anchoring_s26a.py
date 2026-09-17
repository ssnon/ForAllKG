from __future__ import annotations

import pytest

from pipeline_core.discovery.discovery_axis_prompt import (
    _s26a_task_generation_guidance,
)
from pipeline_core.discovery.discovery_axis_runtime import (
    DiscoveryAxisSynthesisRuntime,
)


def test_s26a_guidance_preserves_original_endpoints_and_axis_subordination():
    text = _s26a_task_generation_guidance(
        task_source="nanostructure shape",
        task_target="electromagnetic hotspot location and intensity",
        task_question="Which structural factors determine hotspot behavior?",
    )

    assert "ORIGINAL-TASK PRESERVATION" in text
    assert "task_source: nanostructure shape" in text
    assert (
        "task_target: electromagnetic hotspot location and intensity"
        in text
    )
    assert "endpoint replacement is not" in text
    assert "ABSTAIN" in text
    assert "NOT evidence" in text


def test_s26a_no_task_metadata_preserves_legacy_prompt_behavior():
    assert (
        _s26a_task_generation_guidance(
            task_source=None,
            task_target=None,
            task_question=None,
        )
        == ""
    )


def test_s26a_partial_task_metadata_fails_closed():
    with pytest.raises(
        ValueError,
        match="requires both task_source and task_target",
    ):
        _s26a_task_generation_guidance(
            task_source="A",
            task_target=None,
            task_question=None,
        )


def test_s26a_runtime_rejects_partial_task_metadata():
    with pytest.raises(
        ValueError,
        match="requires both task_source and task_target",
    ):
        DiscoveryAxisSynthesisRuntime(
            object(),
            object(),
            task_source="A",
            task_target=None,
        )


def test_s26a_runtime_accepts_complete_task_metadata():
    runtime = DiscoveryAxisSynthesisRuntime(
        object(),
        object(),
        task_source="A",
        task_target="B",
        task_question="How does A relate to B?",
    )

    assert runtime.task_source == "A"
    assert runtime.task_target == "B"
    assert runtime.task_question == "How does A relate to B?"
