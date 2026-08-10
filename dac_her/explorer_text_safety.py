from __future__ import annotations

import re


# Shared by deterministic normalization and strict validation. Keep this list
# deliberately narrow: these patterns trigger only explicit negative/absence
# assertions, not ordinary uncertainty language.
_ABSENCE_PATTERNS = (
    re.compile(r"\bnot reported\b", re.I),
    re.compile(r"\bno evidence\b", re.I),
    re.compile(r"\bno support\b", re.I),
    re.compile(r"\babsent\b", re.I),
    re.compile(r"\babsence\b", re.I),
    re.compile(r"\bnot observed\b", re.I),
    re.compile(r"\bdoes not contain\b", re.I),
)


def contains_absence_language(text: str) -> bool:
    return any(pattern.search(str(text)) for pattern in _ABSENCE_PATTERNS)
