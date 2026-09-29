from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSpanGrounding:
    grounded_spans: list[str]
    invalid_spans: list[str]
    match_kinds: list[str]


def _normalize(value: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    ).lower()
    text = re.sub(
        r"[‐‑‒–—−]",
        "-",
        text,
    )
    text = re.sub(
        r"\s+",
        " ",
        text,
    )
    return text.strip()


def _ground_one(
    span: str,
    abstract: str,
) -> str | None:
    if not span:
        return None

    if span in abstract:
        return "EXACT"

    normalized_span = _normalize(
        span
    )
    normalized_abstract = _normalize(
        abstract
    )

    if (
        normalized_span
        and normalized_span
        in normalized_abstract
    ):
        return "NORMALIZED"

    for suffix in (
        "…",
        "...",
    ):
        if span.endswith(
            suffix
        ):
            prefix = (
                span[
                    : -len(suffix)
                ]
                .rstrip()
            )
            normalized_prefix = (
                _normalize(
                    prefix
                )
            )
            if (
                normalized_prefix
                and normalized_prefix
                in normalized_abstract
            ):
                return "TRUNCATED_PREFIX"

    return None


def ground_evidence_spans(
    spans: list[str],
    abstract: str,
) -> EvidenceSpanGrounding:
    grounded = []
    invalid = []
    kinds = []

    for raw in spans:
        span = str(
            raw or ""
        )
        kind = _ground_one(
            span,
            abstract,
        )
        if kind is None:
            invalid.append(
                span
            )
            continue

        grounded.append(
            span
        )
        kinds.append(
            kind
        )

    return EvidenceSpanGrounding(
        grounded_spans=grounded,
        invalid_spans=invalid,
        match_kinds=kinds,
    )


__all__ = [
    "EvidenceSpanGrounding",
    "ground_evidence_spans",
]
