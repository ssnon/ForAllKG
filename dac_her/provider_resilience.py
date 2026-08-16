from __future__ import annotations

import email.utils
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RETRYABLE_HTTP_STATUS = frozenset(
    {429, 500, 502, 503, 504}
)


def parse_retry_after_seconds(
    value: str | None,
    *,
    now_utc: datetime | None = None,
) -> float | None:
    """Parse HTTP Retry-After as delta-seconds or HTTP-date.

    Returns None when absent/unparseable. Negative HTTP-date deltas collapse
    to zero. This function performs no sleeping and no network I/O.
    """
    text = str(value or "").strip()
    if not text:
        return None

    try:
        seconds = float(text)
    except ValueError:
        seconds = None
    if seconds is not None:
        return max(0.0, seconds)

    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )
    reference = (
        now_utc
        if now_utc is not None
        else datetime.now(timezone.utc)
    )
    return max(
        0.0,
        (parsed - reference).total_seconds(),
    )


class ProviderRequestPacer:
    """Monotonic request-start pacer with externally deferable cooldown."""

    def __init__(
        self,
        minimum_interval_seconds: float,
        *,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        interval = float(
            minimum_interval_seconds
        )
        if interval < 0:
            raise ValueError(
                "minimum_interval_seconds must be >= 0"
            )
        self.minimum_interval_seconds = (
            interval
        )
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self._next_allowed = 0.0

    def wait(self) -> float:
        now = self._monotonic()
        delay = max(
            0.0,
            self._next_allowed - now,
        )
        if delay > 0:
            self._sleep(delay)
        started = self._monotonic()
        self._next_allowed = max(
            self._next_allowed,
            started
            + self.minimum_interval_seconds,
        )
        return delay

    def defer(self, seconds: float) -> None:
        delay = max(
            0.0,
            float(seconds),
        )
        now = self._monotonic()
        self._next_allowed = max(
            self._next_allowed,
            now + delay,
        )

    def seconds_until_allowed(self) -> float:
        return max(
            0.0,
            self._next_allowed
            - self._monotonic(),
        )


@dataclass
class RequestTelemetry:
    attempts: int = 0
    successes: int = 0
    http_429_events: int = 0
    retryable_http_events: int = 0
    nonretryable_http_events: int = 0
    transport_events: int = 0
    retries_scheduled: int = 0
    retry_after_honored: int = 0
    terminal_retryable_failures: int = 0
    cooldowns_scheduled_after_terminal_failure: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "attempts":
                self.attempts,
            "successes":
                self.successes,
            "http_429_events":
                self.http_429_events,
            "retryable_http_events":
                self.retryable_http_events,
            "nonretryable_http_events":
                self.nonretryable_http_events,
            "transport_events":
                self.transport_events,
            "retries_scheduled":
                self.retries_scheduled,
            "retry_after_honored":
                self.retry_after_honored,
            "terminal_retryable_failures":
                self.terminal_retryable_failures,
            "cooldowns_scheduled_after_terminal_failure":
                self.cooldowns_scheduled_after_terminal_failure,
        }


def _retry_delay(
    *,
    attempt: int,
    retry_backoff: float,
    retry_after_header: str | None,
) -> tuple[float, bool]:
    from_header = (
        parse_retry_after_seconds(
            retry_after_header
        )
    )
    if from_header is not None:
        return from_header, True
    return (
        max(
            0.0,
            float(retry_backoff)
            * (2 ** int(attempt)),
        ),
        False,
    )


def resilient_request_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 2,
    retry_backoff: float = 1.0,
    pacer: ProviderRequestPacer | None = None,
    telemetry: RequestTelemetry | None = None,
    request_factory: Callable[..., Request] = Request,
    urlopen_fn: Callable[..., Any] = urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """GET JSON with bounded retry, Retry-After support, and request pacing.

    The function preserves the historical retryable HTTP set. It does not
    retry authentication/authorization or arbitrary 4xx responses. When the
    final retryable failure is exhausted, a pacer (if supplied) is deferred
    using the same bounded backoff rule so the next logical request does not
    immediately stampede the provider.
    """
    if retries < 0:
        raise ValueError(
            "retries must be >= 0"
        )
    if retry_backoff < 0:
        raise ValueError(
            "retry_backoff must be >= 0"
        )

    last: Exception | None = None
    for attempt in range(
        int(retries) + 1
    ):
        if pacer is not None:
            pacer.wait()
        if telemetry is not None:
            telemetry.attempts += 1

        try:
            request = request_factory(
                url,
                headers=dict(
                    headers or {}
                ),
            )
            with urlopen_fn(
                request,
                timeout=timeout,
            ) as response:
                payload = response.read()
            value = json.loads(
                payload.decode("utf-8")
            )
            if telemetry is not None:
                telemetry.successes += 1
            return value

        except HTTPError as exc:
            last = exc
            status = int(exc.code)
            retryable = (
                status
                in RETRYABLE_HTTP_STATUS
            )
            if telemetry is not None:
                if status == 429:
                    telemetry.http_429_events += 1
                if retryable:
                    telemetry.retryable_http_events += 1
                else:
                    telemetry.nonretryable_http_events += 1

            if not retryable:
                raise

            retry_after = (
                exc.headers.get(
                    "Retry-After"
                )
                if exc.headers is not None
                else None
            )
            delay, honored = _retry_delay(
                attempt=attempt,
                retry_backoff=
                    retry_backoff,
                retry_after_header=
                    retry_after,
            )
            if telemetry is not None and honored:
                telemetry.retry_after_honored += 1

            if attempt >= retries:
                if telemetry is not None:
                    telemetry.terminal_retryable_failures += 1
                # Preserve the next exponential slot as provider-level
                # cooldown after exhausting request-local retries.
                terminal_delay = delay
                if retry_after is None:
                    terminal_delay = max(
                        terminal_delay,
                        float(
                            retry_backoff
                        )
                        * (
                            2
                            ** (
                                int(attempt)
                            )
                        ),
                    )
                if (
                    pacer is not None
                    and terminal_delay > 0
                ):
                    pacer.defer(
                        terminal_delay
                    )
                    if telemetry is not None:
                        telemetry.cooldowns_scheduled_after_terminal_failure += 1
                raise

            if telemetry is not None:
                telemetry.retries_scheduled += 1
            if pacer is not None:
                pacer.defer(delay)
            elif delay > 0:
                sleep_fn(delay)

        except URLError as exc:
            last = exc
            if telemetry is not None:
                telemetry.transport_events += 1

            delay = max(
                0.0,
                float(retry_backoff)
                * (2 ** int(attempt)),
            )
            if attempt >= retries:
                if telemetry is not None:
                    telemetry.terminal_retryable_failures += 1
                if (
                    pacer is not None
                    and delay > 0
                ):
                    pacer.defer(delay)
                    if telemetry is not None:
                        telemetry.cooldowns_scheduled_after_terminal_failure += 1
                raise

            if telemetry is not None:
                telemetry.retries_scheduled += 1
            if pacer is not None:
                pacer.defer(delay)
            elif delay > 0:
                sleep_fn(delay)

    if last is not None:
        raise last
    raise RuntimeError(
        "request failed without an exception"
    )
