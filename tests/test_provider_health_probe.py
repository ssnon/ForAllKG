from __future__ import annotations

from dac_her.provider_health_probe import (
    _query_sha,
    _telemetry_delta,
)


def test_query_sha_changes_when_text_changes():
    left = _query_sha(
        query_id="q1",
        hypothesis_id="h1",
        query_kind="hypothesis_composite",
        query_text="alpha",
    )
    right = _query_sha(
        query_id="q1",
        hypothesis_id="h1",
        query_kind="hypothesis_composite",
        query_text="beta",
    )
    assert left != right


def test_telemetry_delta():
    value = _telemetry_delta(
        {
            "attempts": 2,
            "http_429_events": 1,
        },
        {
            "attempts": 5,
            "http_429_events": 2,
        },
    )
    assert value["attempts"] == 3
    assert (
        value["http_429_events"]
        == 1
    )
