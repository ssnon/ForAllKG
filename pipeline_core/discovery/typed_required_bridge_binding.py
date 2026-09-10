from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any


TYPED_REQUIRED_BRIDGE_PROVENANCE = "TYPED_RELATION_SOURCE_BINDING"


_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "as", "is", "are", "be", "by", "from", "with", "that", "this", "these",
    "those", "among", "under", "when", "where", "having", "have", "has",
    "same", "similar", "comparable", "overall",
}


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9*+\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9*]+", _normalize(value))
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_sentence_spans(
    source_bridge: str,
) -> list[tuple[int, int, str]]:
    """Return exact deterministic sentence spans from inferential_bridge.

    Typed bridge bindings are defined over one existing source sentence,
    never a synthetic or multi-sentence contiguous span. The splitter is
    deliberately lexical and fail-closed.
    """

    spans: list[tuple[int, int, str]] = []
    start = 0

    for match in re.finditer(
        r"[.!?](?=\s+|$)",
        source_bridge,
    ):
        end = match.end()
        raw = source_bridge[start:end]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw.rstrip())
        span_start = start + lead
        span_end = start + trail

        if span_start < span_end:
            spans.append(
                (
                    span_start,
                    span_end,
                    source_bridge[span_start:span_end],
                )
            )

        start = end

    if start < len(source_bridge):
        raw = source_bridge[start:]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw.rstrip())
        span_start = start + lead
        span_end = start + trail

        if span_start < span_end:
            spans.append(
                (
                    span_start,
                    span_end,
                    source_bridge[span_start:span_end],
                )
            )

    return spans


def _source_span_has_typed_coverage(
    *,
    source_text: str,
    claim_text: str,
    endpoints: list[object],
    scopes: list[object],
    directions: list[object],
) -> bool:
    """Check one exact source span against the frozen typed contract."""

    source_norm = _normalize(source_text)
    claim_norm = _normalize(claim_text)

    if not endpoints:
        return False

    for anchor in endpoints:
        anchor_norm = _normalize(anchor)
        if (
            not anchor_norm
            or anchor_norm not in source_norm
            or anchor_norm not in claim_norm
        ):
            return False

    for scope in scopes:
        required = _content_tokens(scope)
        if required and (
            not required.issubset(
                _content_tokens(source_text)
            )
            or not required.issubset(
                _content_tokens(claim_text)
            )
        ):
            return False

    for directional in directions:
        directional_norm = _normalize(
            directional
        )
        if directional_norm and (
            directional_norm not in source_norm
            or directional_norm not in claim_norm
        ):
            return False

    return True


def validate_typed_required_bridge_binding(
    *,
    hypothesis: object,
    claim: object,
    binding: dict[str, Any],
    contract: dict[str, Any],
) -> str:
    # Proves source identity + typed relation coverage only.
    # It is not novelty evidence and does not assess scientific truth.
    source_bridge = str(getattr(hypothesis, "inferential_bridge", "") or "")
    claim_text = str(getattr(claim, "claim_text", "") or "")
    hypothesis_id = str(getattr(hypothesis, "hypothesis_id", "") or "")
    claim_id = str(getattr(claim, "claim_id", "") or "")

    if not source_bridge or not claim_text or not hypothesis_id or not claim_id:
        raise ValueError("typed bridge binding lacks canonical source identity")

    if str(contract.get("hypothesis_id") or "") != hypothesis_id:
        raise ValueError("typed bridge binding hypothesis identity mismatch")
    if str(contract.get("claim_id") or "") != claim_id:
        raise ValueError("typed bridge binding claim identity mismatch")

    if binding.get("schema_version") != (
        "novelty-required-bridge-source-binding-diagnostic-v1"
    ):
        raise ValueError("unexpected typed bridge binding schema")
    if binding.get("binding_semantics") != (
        "TYPED_RELATION_REFERENCE_TO_EXISTING_SOURCE_SPAN"
    ):
        raise ValueError("unexpected typed bridge binding semantics")
    if binding.get("source_path") != "inferential_bridge":
        raise ValueError("typed bridge source_path must be inferential_bridge")
    if binding.get("diagnostic_only") is not True:
        raise ValueError("typed bridge source binding lost diagnostic boundary")
    if binding.get("production_authority") is not False:
        raise ValueError("typed bridge source binding claims production authority")
    if binding.get("free_text_bridge_generated") is not False:
        raise ValueError("typed bridge source binding generated free text")
    if binding.get("novelty_assessed") is not False:
        raise ValueError("typed bridge source binding claims novelty assessment")
    if binding.get("scientific_truth_assessed") is not False:
        raise ValueError("typed bridge source binding claims truth assessment")
    if binding.get("scientific_equivalence_assessed") is not False:
        raise ValueError("typed bridge source binding claims equivalence assessment")

    source_sha = _sha256_text(source_bridge)
    if str(binding.get("source_sha256") or "") != source_sha:
        raise ValueError("typed bridge source SHA mismatch")
    if str(contract.get("source_bridge_sha256") or "") != source_sha:
        raise ValueError("typed bridge contract source SHA mismatch")

    start = binding.get("start")
    end = binding.get("end")
    quote = str(binding.get("quote") or "")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("typed bridge span offsets are not integers")
    if not (0 <= start < end <= len(source_bridge)):
        raise ValueError("typed bridge span offsets out of bounds")
    if source_bridge[start:end] != quote:
        raise ValueError("typed bridge quote/offset mismatch")
    if not quote.strip():
        raise ValueError("typed bridge source quote is empty")

    endpoints = list(contract.get("relation_endpoint_anchors") or [])
    scopes = list(contract.get("scope_qualifier_spans") or [])
    directions = list(contract.get("directional_qualifier_spans") or [])

    if list(binding.get("relation_endpoint_anchors") or []) != endpoints:
        raise ValueError("typed bridge endpoint contract mismatch")
    if list(binding.get("scope_qualifier_spans") or []) != scopes:
        raise ValueError("typed bridge scope contract mismatch")
    if list(binding.get("directional_qualifier_spans") or []) != directions:
        raise ValueError("typed bridge direction contract mismatch")

    warnings = contract.get("semantic_warnings")
    if not isinstance(warnings, dict):
        raise ValueError("typed bridge semantic warning contract missing")
    expected_warning_keys = {
        "bridge_alignment_shadow_warning",
        "direction_shadow_warning",
        "endpoint_shadow_warning",
        "scope_shadow_warning",
    }
    if set(warnings) != expected_warning_keys:
        raise ValueError("typed bridge semantic warning keys drift")
    if any(bool(warnings[key]) for key in expected_warning_keys):
        raise ValueError("typed bridge semantic-fidelity warning present")

    quote_norm = _normalize(quote)
    claim_norm = _normalize(claim_text)

    if not endpoints:
        raise ValueError("typed bridge endpoint anchors empty")

    for anchor in endpoints:
        anchor_norm = _normalize(anchor)
        if not anchor_norm:
            raise ValueError("typed bridge endpoint anchor empty")
        if anchor_norm not in quote_norm:
            raise ValueError("typed bridge endpoint absent from source quote")
        if anchor_norm not in claim_norm:
            raise ValueError("typed bridge endpoint absent from atomic claim")

    for scope in scopes:
        required = _content_tokens(scope)
        if not required:
            continue
        if not required.issubset(_content_tokens(quote)):
            raise ValueError("typed bridge scope absent from source quote")
        if not required.issubset(_content_tokens(claim_text)):
            raise ValueError("typed bridge scope absent from atomic claim")

    for directional in directions:
        directional_norm = _normalize(directional)
        if not directional_norm:
            continue
        if directional_norm not in quote_norm:
            raise ValueError("typed bridge direction absent from source quote")
        if directional_norm not in claim_norm:
            raise ValueError("typed bridge direction absent from atomic claim")

    source_sentences = _source_sentence_spans(
        source_bridge
    )

    bound_sentence_matches = [
        sentence
        for sentence in source_sentences
        if (
            sentence[0] == start
            and sentence[1] == end
            and sentence[2] == quote
        )
    ]

    if len(bound_sentence_matches) != 1:
        raise ValueError(
            "typed bridge source span must be exactly one "
            "existing source sentence"
        )

    qualifying_sentences = [
        sentence
        for sentence in source_sentences
        if _source_span_has_typed_coverage(
            source_text=sentence[2],
            claim_text=claim_text,
            endpoints=endpoints,
            scopes=scopes,
            directions=directions,
        )
    ]

    if len(qualifying_sentences) != 1:
        raise ValueError(
            "typed bridge source sentence is not uniquely "
            "qualified by the typed contract"
        )

    if qualifying_sentences[0] != (
        start,
        end,
        quote,
    ):
        raise ValueError(
            "typed bridge binding does not select the unique "
            "qualifying source sentence"
        )

    return quote
