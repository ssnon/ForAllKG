from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote


_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)",
    re.MULTILINE,
)
_PAGE_ANCHOR_RE = re.compile(
    r"<span\s+id=[\"']page-(?P<page>\d+)-[^\"']+[\"']\s*></span>",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"(?m)^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_FILENAME_PAGE_RE = re.compile(r"(?:^|[_-])page[_-]?(?P<page>\d+)(?:[_-]|$)", re.I)
_CAPTION_START_RE = re.compile(
    r"^(?:(?:Supplementary|Extended\s+Data)\s+)?"
    r"(?:Fig(?:ure)?\.?|Table)\s*\w+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    paper_id: str
    document_id: str
    document_role: str
    asset_type: str
    relative_path: str
    absolute_path: str
    exists: bool
    sha256: str | None
    page_id: int | None
    section: str | None
    caption: str | None
    marker_alt_text: str | None
    markdown_start: int
    markdown_end: int

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_target(target: str) -> str:
    target = target.strip()
    # Markdown may use <path> or an optional title after whitespace.
    if target.startswith("<") and ">" in target:
        target = target[1:target.index(">")]
    elif " \"" in target:
        target = target.split(" \"", 1)[0]
    elif " '" in target:
        target = target.split(" '", 1)[0]
    return unquote(target.strip())


def _is_remote_or_empty(target: str) -> bool:
    lowered = target.lower()
    return (
        not target
        or lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("data:")
        or lowered.startswith("#")
    )


def _section_before(markdown: str, position: int) -> str | None:
    section: str | None = None
    for match in _HEADING_RE.finditer(markdown, 0, position):
        section = f"{match.group('marks')} {match.group('title').strip()}"
    return section


def _page_before(markdown: str, position: int, filename: str) -> int | None:
    page: int | None = None
    for match in _PAGE_ANCHOR_RE.finditer(
        markdown,
        max(0, position - 1000),
        position,
    ):
        page = int(match.group("page"))
    if page is not None:
        return page
    filename_match = _FILENAME_PAGE_RE.search(filename)
    return int(filename_match.group("page")) if filename_match else None


def _caption_after(markdown: str, position: int) -> str | None:
    tail = markdown[position:]
    # Same-line text after the image or the next paragraph.
    tail = tail.lstrip(" \t")
    if tail.startswith("\n"):
        tail = tail.lstrip("\r\n \t")

    paragraphs = re.split(r"\n\s*\n", tail, maxsplit=2)
    if not paragraphs:
        return None

    candidate = " ".join(
        line.strip()
        for line in paragraphs[0].splitlines()
        if line.strip()
    ).strip()

    if not candidate:
        return None
    if _CAPTION_START_RE.match(candidate):
        return candidate
    return None


def _asset_type(relative_path: str, caption: str | None) -> str:
    source = f"{relative_path} {caption or ''}".lower()
    if "table" in source:
        return "table_image"
    if "equation" in source or "formula" in source:
        return "equation"
    if "fig" in source or "figure" in source or "picture" in source:
        return "figure"
    return "other"


def _asset_id(
    paper_id: str,
    document_id: str,
    relative_path: str,
) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"{paper_id}:{document_id}:asset:{digest}"


def build_asset_index(
    *,
    paper_id: str,
    document_id: str,
    document_role: str,
    package_dir: str | Path,
    markdown: str,
) -> list[AssetRecord]:
    package_dir = Path(package_dir).resolve()
    by_path: dict[str, AssetRecord] = {}

    for match in _IMAGE_RE.finditer(markdown):
        target = _clean_target(match.group("target"))
        if _is_remote_or_empty(target):
            continue

        absolute = (package_dir / target).resolve()
        try:
            relative = str(absolute.relative_to(package_dir))
        except ValueError:
            # Preserve the reference but flag it as outside the package.
            relative = target

        caption = _caption_after(markdown, match.end())
        alt = match.group("alt").strip() or None
        record = AssetRecord(
            asset_id=_asset_id(paper_id, document_id, relative),
            paper_id=paper_id,
            document_id=document_id,
            document_role=document_role,
            asset_type=_asset_type(relative, caption),
            relative_path=relative,
            absolute_path=str(absolute),
            exists=absolute.exists() and absolute.is_file(),
            sha256=(
                _sha256_file(absolute)
                if absolute.exists() and absolute.is_file()
                else None
            ),
            page_id=_page_before(markdown, match.start(), relative),
            section=_section_before(markdown, match.start()),
            caption=caption,
            marker_alt_text=alt,
            markdown_start=match.start(),
            markdown_end=match.end(),
        )

        previous = by_path.get(relative)
        if previous is None:
            by_path[relative] = record
        else:
            # Prefer the occurrence with a caption and page/section metadata.
            by_path[relative] = AssetRecord(
                asset_id=previous.asset_id,
                paper_id=previous.paper_id,
                document_id=previous.document_id,
                document_role=previous.document_role,
                asset_type=(
                    record.asset_type
                    if previous.asset_type == "other"
                    else previous.asset_type
                ),
                relative_path=previous.relative_path,
                absolute_path=previous.absolute_path,
                exists=previous.exists or record.exists,
                sha256=previous.sha256 or record.sha256,
                page_id=previous.page_id if previous.page_id is not None else record.page_id,
                section=previous.section or record.section,
                caption=previous.caption or record.caption,
                marker_alt_text=previous.marker_alt_text or record.marker_alt_text,
                markdown_start=min(previous.markdown_start, record.markdown_start),
                markdown_end=max(previous.markdown_end, record.markdown_end),
            )

    return sorted(by_path.values(), key=lambda item: item.markdown_start)


def assets_by_id(assets: Iterable[AssetRecord]) -> dict[str, AssetRecord]:
    return {asset.asset_id: asset for asset in assets}


def asset_path_to_id(assets: Iterable[AssetRecord]) -> dict[str, str]:
    return {asset.relative_path: asset.asset_id for asset in assets}


def write_assets_jsonl(
    path: str | Path,
    assets: Iterable[AssetRecord],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for asset in assets:
            file.write(json.dumps(asset.to_dict(), ensure_ascii=False) + "\n")
    return path
