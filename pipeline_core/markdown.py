"""Shared Markdown heading and section helpers."""

from __future__ import annotations

import re
import unicodedata


_HEADING_RE = re.compile(r"(?m)^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")


def normalize_heading(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("–", "-").replace("—", "-")
    value = value.lower().strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"[.。:;]+$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def resolve_markdown_heading(markdown: str, heading: str) -> re.Match[str]:
    heading = heading.strip()
    if not heading:
        raise ValueError("Section heading must not be empty.")

    configured = re.match(r"^(?P<marks>#{1,6})\s+(?P<title>.+)$", heading)
    if configured is None:
        raise ValueError(
            "Configured section headings must begin with Markdown '#'. "
            f"Received: {heading!r}"
        )

    configured_level = len(configured.group("marks"))
    configured_normalized = normalize_heading(configured.group("title"))
    candidates = [
        match
        for match in _HEADING_RE.finditer(markdown)
        if len(match.group("marks")) == configured_level
        and normalize_heading(match.group("title")) == configured_normalized
    ]

    if not candidates:
        available = [
            f"{match.group('marks')} {match.group('title').strip()}"
            for match in _HEADING_RE.finditer(markdown)
        ]
        preview = "\n".join(f"- {item}" for item in available[:30])
        raise ValueError(
            f"Section heading not found: {heading}\n"
            f"Available headings include:\n{preview}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Section heading is ambiguous after normalization: "
            f"{heading!r}. Matches: "
            + ", ".join(match.group(0) for match in candidates)
        )
    return candidates[0]


def extract_markdown_section(markdown: str, heading: str) -> str:
    """Extract one Markdown section through the next peer/parent heading.

    Heading resolution tolerates capitalization, repeated spaces, Unicode
    dash variants, and a trailing period, but still requires a unique match.
    """
    start_match = resolve_markdown_heading(markdown, heading)
    level = len(start_match.group("marks"))
    heading_pattern = re.compile(rf"(?m)^#{{1,{level}}}\s+.+$")
    next_match = heading_pattern.search(markdown, start_match.end())
    end = next_match.start() if next_match else len(markdown)
    section = markdown[start_match.start():end].strip()
    if not section:
        raise ValueError(f"The selected section is empty: {heading}")
    return section
