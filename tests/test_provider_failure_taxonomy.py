from __future__ import annotations

from dac_her.provider_failure_taxonomy import (
    classify_failure,
    extract_http_status,
)


def test_extract_http_error_429():
    assert (
        extract_http_status(
            "HTTPError: HTTP Error 429: Too Many Requests"
        )
        == 429
    )


def test_classify_rate_limit():
    row = classify_failure(
        "HTTPError: HTTP Error 429: Too Many Requests"
    )
    assert (
        row["category"]
        == "HTTP_429_RATE_LIMIT"
    )


def test_classify_403():
    row = classify_failure(
        "HTTPError: HTTP Error 403: Forbidden"
    )
    assert (
        row["category"]
        == "HTTP_403_AUTHORIZATION"
    )


def test_classify_5xx():
    row = classify_failure(
        "HTTPError: HTTP Error 503: Service Unavailable"
    )
    assert (
        row["category"]
        == "HTTP_5XX_UPSTREAM"
    )


def test_classify_timeout():
    row = classify_failure(
        "TimeoutError: timed out"
    )
    assert row["category"] == "TIMEOUT"


def test_classify_url_transport():
    row = classify_failure(
        "URLError: <urlopen error [Errno 111] Connection refused>"
    )
    assert (
        row["category"]
        == "TRANSPORT_ERROR"
    )


def test_unknown_error_is_not_relabelled_as_rate_limit():
    row = classify_failure(
        "ValueError: unexpected payload"
    )
    assert (
        row["category"]
        == "OTHER_EXCEPTION"
    )
