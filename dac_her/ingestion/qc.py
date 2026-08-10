from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from .contracts import IngestionIssue, MarkerResult


def _normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def markdown_qc(
    result: MarkerResult,
    expected_title: str,
    min_chars: int = 3000,
    title_threshold: float = 0.35,
) -> list[IngestionIssue]:
    issues: list[IngestionIssue] = []
    if not result.succeeded or not result.normalized_markdown:
        issues.append(
            IngestionIssue(
                code="conversion_failed",
                message=result.error or "Marker conversion failed.",
                severity="error",
                file_name=Path(result.input_pdf).name,
            )
        )
        return issues
    path = Path(result.normalized_markdown)
    if not path.exists():
        issues.append(
            IngestionIssue(
                code="markdown_missing",
                message=f"Expected normalized Markdown does not exist: {path}",
                severity="error",
                file_name=Path(result.input_pdf).name,
            )
        )
        return issues
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) < min_chars:
        issues.append(
            IngestionIssue(
                code="markdown_too_short",
                message=f"Markdown has only {len(text)} characters (< {min_chars}).",
                severity="warning",
                file_name=Path(result.input_pdf).name,
            )
        )
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    if replacement_ratio > 0.002:
        issues.append(
            IngestionIssue(
                code="replacement_character_ratio",
                message=f"Unicode replacement-character ratio is {replacement_ratio:.4%}.",
                severity="warning",
                file_name=Path(result.input_pdf).name,
            )
        )
    if expected_title:
        head = _normalize_title(text[:8000])
        title = _normalize_title(expected_title)
        score = SequenceMatcher(None, title, head[: max(len(title) * 4, 100)]).ratio()
        # Marker/title layouts vary; only warn, never fail.
        if score < title_threshold and title not in head:
            issues.append(
                IngestionIssue(
                    code="title_match_weak",
                    message=f"Expected title was not strongly matched near document start (score={score:.2f}).",
                    severity="warning",
                    file_name=Path(result.input_pdf).name,
                )
            )
    return issues


def qc_status(issues: list[IngestionIssue]) -> str:
    if any(item.severity == "error" for item in issues):
        return "failed"
    if any(item.severity == "warning" for item in issues):
        return "passed_with_warnings"
    return "passed"
