from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


BRIDGE_SOURCE_RECONCILIATION_VERSION = (
    "bridge-source-reconciliation-v1"
)


@dataclass(frozen=True)
class ReconciledSpan:
    value: str
    changed: bool
    method: str


@dataclass(frozen=True)
class SourceNormalizationOperation:
    field: str
    old_value: str
    new_value: str
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "method": self.method,
        }


_MARKDOWN_LINK = re.compile(
    r"\[([^\]]+)\]\(([^)]+)\)"
)
_HTML_TAG = re.compile(r"<[^>]+>")

# Formatting-only characters that may be inserted/removed by Markdown/LaTeX
# rendering without changing the scientific source phrase.
_IGNORABLE_FORMAT_CHARS = {
    "$",
    "_",
    "{",
    "}",
    "[",
    "]",
}

_DASHES = {
    "–": "-",
    "—": "-",
    "−": "-",
    "‑": "-",
    "‒": "-",
}


def _visible_markdown_text_with_map(
    text: str,
) -> tuple[str, list[int]]:
    """
    Remove Markdown link destinations while retaining visible link text.

    The returned index map points each visible character back to the original
    string so a normalized match can be recovered as an exact source substring.
    """
    out: list[str] = []
    mapping: list[int] = []
    cursor = 0

    for match in _MARKDOWN_LINK.finditer(text):
        start, end = match.span()
        for index in range(cursor, start):
            out.append(text[index])
            mapping.append(index)

        label_start, label_end = match.span(1)
        for index in range(label_start, label_end):
            out.append(text[index])
            mapping.append(index)

        cursor = end

    for index in range(cursor, len(text)):
        out.append(text[index])
        mapping.append(index)

    return "".join(out), mapping


def _drop_html_tags_with_map(
    text: str,
    mapping: list[int],
) -> tuple[str, list[int]]:
    out: list[str] = []
    out_map: list[int] = []
    cursor = 0

    for match in _HTML_TAG.finditer(text):
        start, end = match.span()
        out.extend(text[cursor:start])
        out_map.extend(mapping[cursor:start])
        cursor = end

    out.extend(text[cursor:])
    out_map.extend(mapping[cursor:])
    return "".join(out), out_map


def _normalized_with_map(
    text: str,
) -> tuple[str, list[int]]:
    """
    Conservative scientific-text normalization with reversible character map.

    The transformation ignores only rendering artifacts that were observed in
    Bridge failures (Markdown links/escapes, LaTeX delimiters/subscript braces,
    Unicode dash variants, and repeated whitespace). It deliberately preserves
    lexical content and punctuation so normalization cannot silently turn a
    paraphrase into a source match.
    """
    # html.unescape may change string length. Apply it only when it is a no-op;
    # otherwise character provenance would become ambiguous. Most extracted
    # Markdown already contains decoded Unicode.
    unescaped = html.unescape(text)
    if len(unescaped) == len(text):
        text = unescaped

    visible, source_map = _visible_markdown_text_with_map(text)
    visible, source_map = _drop_html_tags_with_map(visible, source_map)

    out: list[str] = []
    out_map: list[int] = []
    pending_space = False
    pending_space_index: int | None = None

    i = 0
    while i < len(visible):
        char = visible[i]
        source_index = source_map[i]

        # Markdown escaping: keep the escaped character but not the backslash.
        if char == "\\" and i + 1 < len(visible):
            i += 1
            char = visible[i]
            source_index = source_map[i]

        if char in _IGNORABLE_FORMAT_CHARS:
            i += 1
            continue

        char = _DASHES.get(char, char)

        if char.isspace():
            if out:
                pending_space = True
                if pending_space_index is None:
                    pending_space_index = source_index
            i += 1
            continue

        if pending_space:
            out.append(" ")
            out_map.append(
                pending_space_index
                if pending_space_index is not None
                else source_index
            )
            pending_space = False
            pending_space_index = None

        normalized = unicodedata.normalize("NFKC", char).casefold()
        for normalized_char in normalized:
            out.append(_DASHES.get(normalized_char, normalized_char))
            out_map.append(source_index)

        i += 1

    return "".join(out).strip(), out_map[: len("".join(out).strip())]


def normalized_scientific_text(text: str) -> str:
    return _normalized_with_map(text)[0]


def _unique_occurrence(
    haystack: str,
    needle: str,
) -> int | None:
    if not needle:
        return None

    first = haystack.find(needle)
    if first < 0:
        return None

    second = haystack.find(needle, first + 1)
    if second >= 0:
        return None

    return first


def reconcile_phrase_to_text(
    phrase: str | None,
    source_text: str,
) -> ReconciledSpan | None:
    """
    Reconcile one phrase to an exact source substring.

    Safety invariant: reconciliation succeeds only for an exact direct match or
    for a unique match under conservative formatting normalization. Ambiguous or
    paraphrased phrases remain unreconciled and must be handled by semantic
    repair/quarantine rather than fuzzy matching.
    """
    if phrase is None:
        return None

    if phrase in source_text:
        return ReconciledSpan(
            value=phrase,
            changed=False,
            method="exact",
        )

    normalized_source, source_map = _normalized_with_map(source_text)
    normalized_phrase, _ = _normalized_with_map(phrase)

    start = _unique_occurrence(
        normalized_source,
        normalized_phrase,
    )
    if start is None:
        return None

    end = start + len(normalized_phrase)
    if end <= start or end > len(source_map):
        return None

    original_start = source_map[start]
    original_end = source_map[end - 1] + 1
    exact_source_span = source_text[
        original_start:original_end
    ]

    return ReconciledSpan(
        value=exact_source_span,
        changed=exact_source_span != phrase,
        method="unique_normalized_source_span",
    )


def reconcile_concept_payload(
    payload: dict[str, Any],
    *,
    core_text: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reconcile all provenance-bearing phrase fields in one concept payload."""
    payload = dict(payload)
    operations: list[SourceNormalizationOperation] = []

    def replace_field(
        container: dict[str, Any],
        field: str,
        source: str,
        logical_field: str,
    ) -> None:
        value = container.get(field)
        if not isinstance(value, str) or not value:
            return

        reconciled = reconcile_phrase_to_text(
            value,
            source,
        )
        if reconciled is None or not reconciled.changed:
            return

        container[field] = reconciled.value
        operations.append(
            SourceNormalizationOperation(
                field=logical_field,
                old_value=value,
                new_value=reconciled.value,
                method=reconciled.method,
            )
        )

    replace_field(
        payload,
        "source_phrase",
        core_text,
        "source_phrase",
    )

    supporting = list(
        payload.get("supporting_phrases") or []
    )
    for index, value in enumerate(supporting):
        if not isinstance(value, str):
            continue
        reconciled = reconcile_phrase_to_text(
            value,
            core_text,
        )
        if reconciled is not None and reconciled.changed:
            supporting[index] = reconciled.value
            operations.append(
                SourceNormalizationOperation(
                    field=f"supporting_phrases[{index}]",
                    old_value=value,
                    new_value=reconciled.value,
                    method=reconciled.method,
                )
            )
    payload["supporting_phrases"] = supporting

    comparison_items = [
        dict(item)
        for item in (payload.get("comparison_items") or [])
    ]
    for index, item in enumerate(comparison_items):
        replace_field(
            item,
            "source_phrase",
            core_text,
            f"comparison_items[{index}].source_phrase",
        )
    payload["comparison_items"] = comparison_items

    # derived_multi_span uses supporting_phrases as the canonical set of
    # provenance spans. After source reconciliation, ensure that the exact
    # source_phrase and every exact comparison-item source span are members of
    # supporting_phrases.
    #
    # This is representation-only normalization:
    # - every inserted phrase must already resolve uniquely to CORE_TEXT;
    # - no paraphrase/fuzzy matching is allowed;
    # - never exceed the BridgeConcept schema limit of 8 supporting phrases.
    if (
        payload.get("retention_lane") == "accepted_pattern"
        and payload.get("pattern_support_mode")
        == "derived_multi_span"
    ):
        required_supports: list[str] = []

        source_phrase = payload.get("source_phrase")
        if isinstance(source_phrase, str) and source_phrase:
            grounded = reconcile_phrase_to_text(
                source_phrase,
                core_text,
            )
            if grounded is not None:
                exact_phrase = grounded.value

                # Normally replace_field() above already did this, but keep
                # the payload canonical even if it was already exact.
                payload["source_phrase"] = exact_phrase

                if exact_phrase not in required_supports:
                    required_supports.append(
                        exact_phrase
                    )

        for item in comparison_items:
            value = item.get("source_phrase")

            if not isinstance(value, str) or not value:
                continue

            grounded = reconcile_phrase_to_text(
                value,
                core_text,
            )
            if grounded is None:
                continue

            exact_phrase = grounded.value
            item["source_phrase"] = exact_phrase

            if exact_phrase not in required_supports:
                required_supports.append(
                    exact_phrase
                )

        missing_supports = [
            value
            for value in required_supports
            if value not in supporting
        ]

        # Apply only when the resulting list remains schema-valid.
        # Otherwise leave it unresolved for local repair/quarantine.
        if (
            missing_supports
            and len(supporting) + len(missing_supports) <= 8
        ):
            for value in missing_supports:
                supporting.append(value)

                operations.append(
                    SourceNormalizationOperation(
                        field="supporting_phrases",
                        old_value="",
                        new_value=value,
                        method=(
                            "derived_multi_span_required_support"
                        ),
                    )
                )

            payload["supporting_phrases"] = supporting
            
    # After supporting spans are reconciled to exact source text, evidence fields
    # are reconciled *within that supporting span* to preserve the strict
    # substring invariant used by bridge_validation.py.
    if (
        payload.get("pattern_support_mode")
        == "explicit_single_span"
        and len(supporting) == 1
    ):
        support_span = supporting[0]
        for field in (
            "subject_evidence_phrase",
            "relation_evidence_phrase",
            "object_evidence_phrase",
        ):
            replace_field(
                payload,
                field,
                support_span,
                field,
            )

    # source_phrase must be literally one of supporting_phrases for accepted
    # patterns. If they normalize to the same unique source span, use the exact
    # supporting span rather than asking the LLM to regenerate the graph.
    source_phrase = payload.get("source_phrase")
    if (
        payload.get("retention_lane") == "accepted_pattern"
        and isinstance(source_phrase, str)
        and source_phrase not in supporting
    ):
        normalized_source = normalized_scientific_text(source_phrase)
        matching = [
            item
            for item in supporting
            if normalized_scientific_text(item) == normalized_source
        ]
        if len(matching) == 1:
            payload["source_phrase"] = matching[0]
            operations.append(
                SourceNormalizationOperation(
                    field="source_phrase",
                    old_value=source_phrase,
                    new_value=matching[0],
                    method="supporting_phrase_identity",
                )
            )

    return payload, [item.as_dict() for item in operations]
