from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


_MAIN_RE = re.compile(
    r"^(?P<owner>.+?)_(?P<number>\d+)\.pdf$",
    re.IGNORECASE,
)
_SI_RE = re.compile(
    r"^(?P<owner>.+?)_(?P<number>\d+)_SI_?(?P<si>\d+)\.pdf$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedPdfName:
    owner: str
    number: int
    role: str
    si_index: int | None = None

    @property
    def main_filename(self) -> str:
        return f"{self.owner}_{self.number}.pdf"


def parse_pdf_name(name: str) -> ParsedPdfName | None:
    base = Path(name).name
    match = _SI_RE.match(base)
    if match:
        return ParsedPdfName(
            owner=match.group("owner"),
            number=int(match.group("number")),
            role="supporting_information",
            si_index=int(match.group("si")),
        )
    match = _MAIN_RE.match(base)
    if match:
        return ParsedPdfName(
            owner=match.group("owner"),
            number=int(match.group("number")),
            role="main",
        )
    return None


def paper_id_for(
    main_filename: str,
    annotator: str,
    aliases: dict[str, str] | None = None,
) -> str:
    parsed = parse_pdf_name(main_filename)
    aliases = aliases or {}
    if parsed is not None:
        prefix = aliases.get(annotator) or aliases.get(parsed.owner)
        if prefix:
            return f"{prefix}_{parsed.number}"
        # Unicode is legal in paths/JSON and keeps the Drive convention visible.
        return f"{parsed.owner}_{parsed.number}"
    digest = hashlib.sha1(main_filename.encode("utf-8")).hexdigest()[:10]
    return f"paper_{digest}"
