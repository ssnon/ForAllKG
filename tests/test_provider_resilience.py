from __future__ import annotations

from datetime import datetime, timezone
from email.message import Message
from urllib.error import HTTPError

import pytest

from pipeline_core.literature.acquisition.provider_resilience import (
    ProviderRequestPacer,
    RequestTelemetry,
    parse_retry_after_seconds,
    resilient_request_json,
)


def test_retry_after_delta_seconds():
    assert (
        parse_retry_after_seconds(
            "3"
        )
        == 3.0
    )


def test_retry_after_http_date():
    reference = datetime(
        2026,
        8,
        16,
        8,
        0,
        0,
        tzinfo=timezone.utc,
    )
    value = (
        "Sun, 16 Aug 2026 08:00:05 GMT"
    )
    assert (
        parse_retry_after_seconds(
            value,
            now_utc=reference,
        )
        == 5.0
    )


def test_pacer_spacing_and_defer_without_real_sleep():
    clock = [0.0]
    sleeps = []

    def monotonic():
        return clock[0]

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    pacer = ProviderRequestPacer(
        1.05,
        monotonic_fn=monotonic,
        sleep_fn=sleep,
    )
    assert pacer.wait() == 0.0
    waited = pacer.wait()
    assert waited == pytest.approx(
        1.05
    )

    pacer.defer(4.0)
    waited = pacer.wait()
    assert waited == pytest.approx(
        4.0
    )


class _Response:
    def __init__(self, payload=b'{"ok":true}'):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        return False

    def read(self):
        return self.payload


def _http_error(
    status: int,
    *,
    retry_after: str | None = None,
):
    headers = Message()
    if retry_after is not None:
        headers[
            "Retry-After"
        ] = retry_after
    return HTTPError(
        url="https://example.test",
        code=status,
        msg="error",
        hdrs=headers,
        fp=None,
    )


def test_retry_after_is_honored_before_retry():
    outcomes = [
        _http_error(
            429,
            retry_after="7",
        ),
        _Response(),
    ]
    sleeps = []

    def urlopen_fn(
        request,
        timeout,
    ):
        outcome = outcomes.pop(0)
        if isinstance(
            outcome,
            Exception,
        ):
            raise outcome
        return outcome

    value = resilient_request_json(
        "https://example.test",
        retries=1,
        retry_backoff=1.0,
        telemetry=RequestTelemetry(),
        urlopen_fn=urlopen_fn,
        sleep_fn=sleeps.append,
    )
    assert value == {"ok": True}
    assert sleeps == [7.0]


def test_403_is_not_retried():
    calls = [0]

    def urlopen_fn(
        request,
        timeout,
    ):
        calls[0] += 1
        raise _http_error(403)

    telemetry = RequestTelemetry()
    with pytest.raises(HTTPError):
        resilient_request_json(
            "https://example.test",
            retries=2,
            telemetry=telemetry,
            urlopen_fn=urlopen_fn,
            sleep_fn=lambda _: None,
        )
    assert calls[0] == 1
    assert (
        telemetry.nonretryable_http_events
        == 1
    )


def test_terminal_429_defers_next_logical_request():
    clock = [0.0]

    def monotonic():
        return clock[0]

    def sleep(seconds):
        clock[0] += seconds

    pacer = ProviderRequestPacer(
        1.05,
        monotonic_fn=monotonic,
        sleep_fn=sleep,
    )
    telemetry = RequestTelemetry()

    def urlopen_fn(
        request,
        timeout,
    ):
        raise _http_error(429)

    with pytest.raises(HTTPError):
        resilient_request_json(
            "https://example.test",
            retries=2,
            retry_backoff=1.0,
            pacer=pacer,
            telemetry=telemetry,
            urlopen_fn=urlopen_fn,
        )

    assert telemetry.attempts == 3
    assert telemetry.http_429_events == 3
    assert telemetry.retries_scheduled == 2
    assert (
        telemetry.cooldowns_scheduled_after_terminal_failure
        == 1
    )
    # Existing exponential sequence is 1, 2, then the terminal
    # cooldown preserves the next request slot at >= 4 seconds.
    assert (
        pacer.seconds_until_allowed()
        >= 4.0
    )


def test_telemetry_snapshot_is_non_sensitive():
    telemetry = RequestTelemetry(
        attempts=2,
        http_429_events=1,
    )
    value = telemetry.snapshot()
    assert value[
        "attempts"
    ] == 2
    assert value[
        "http_429_events"
    ] == 1
    assert "api_key" not in value
