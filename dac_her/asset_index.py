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
    r"^(?:(?:Supplementary|Supplemental|Extended\s+Data)\s+)?"
    r"(?:Fig(?:ure)?\.?|Table|Scheme)\s*[A-Za-z]?\d+[A-Za-z]?",
    re.IGNORECASE,
)
_CAPTION_LINE_RE = re.compile(
    r"(?mi)^\s*(?:#{1,6}\s*)?(?:\*\*|__)?"
    r"(?P<caption>"
    r"(?:(?:Supplementary|Supplemental|Extended\s+Data)\s+)?"
    r"(?:Fig(?:ure)?\.?|Table|Scheme)\s*[A-Za-z]?\d+[A-Za-z]?"
    r"[^\n]*"
    r")"
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


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
    referenced_in_markdown: bool = True
    discovery_method: str = "markdown_image"

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
    for match in _HEADING_RE.finditer(markdown, 0, max(0, position)):
        section = f"{match.group('marks')} {match.group('title').strip()}"
    return section


def _page_before(markdown: str, position: int, filename: str) -> int | None:
    page: int | None = None
    for match in _PAGE_ANCHOR_RE.finditer(
        markdown,
        max(0, position - 2000),
        position,
    ):
        page = int(match.group("page"))
    if page is not None:
        return page
    filename_match = _FILENAME_PAGE_RE.search(filename)
    return int(filename_match.group("page")) if filename_match else None


def _page_span(markdown: str, page_id: int | None) -> tuple[int, int] | None:
    if page_id is None:
        return None
    matches = list(_PAGE_ANCHOR_RE.finditer(markdown))
    for index, match in enumerate(matches):
        if int(match.group("page")) != page_id:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        return match.start(), end
    return None


def _strip_markdown_caption(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^(?:#{1,6}\s*)", "", value)
    value = value.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", value).strip()


def _captions_in_text(text: str) -> list[str]:
    captions: list[str] = []
    for match in _CAPTION_LINE_RE.finditer(text):
        caption = _strip_markdown_caption(match.group("caption"))
        if caption and caption not in captions:
            captions.append(caption)
    return captions


def _caption_for_page(
    markdown: str,
    page_id: int | None,
    filename: str,
) -> tuple[str | None, int, int]:
    span = _page_span(markdown, page_id)
    if span is None:
        return None, len(markdown), len(markdown)
    start, end = span
    captions = _captions_in_text(markdown[start:end])
    if not captions:
        return None, start, end

    lowered = filename.lower()
    if "table" in lowered:
        preferred = [item for item in captions if re.match(r"(?i)^(?:supplementary\s+)?table\b", item)]
    else:
        preferred = [item for item in captions if re.match(r"(?i)^(?:(?:supplementary|supplemental)\s+)?(?:fig|figure)\b", item)]
    return (preferred or captions)[0], start, end


def _caption_after(markdown: str, position: int) -> str | None:
    tail = markdown[position:]
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
    candidate = _strip_markdown_caption(candidate)

    if candidate and _CAPTION_START_RE.match(candidate):
        return candidate
    return None


def _asset_type(relative_path: str, caption: str | None) -> str:
    source = f"{relative_path} {caption or ''}".lower()
    if "table" in source:
        return "table_image"
    if "equation" in source or "formula" in source:
        return "equation"
    if any(token in source for token in ("fig", "figure", "picture", "image")):
        return "figure"
    return "other"


def _asset_id(paper_id: str, document_id: str, relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"{paper_id}:{document_id}:asset:{digest}"


def _record_for_file(
    *,
    paper_id: str,
    document_id: str,
    document_role: str,
    package_dir: Path,
    markdown: str,
    absolute: Path,
    relative: str,
    referenced_in_markdown: bool,
    discovery_method: str,
    marker_alt_text: str | None = None,
    markdown_start: int | None = None,
    markdown_end: int | None = None,
    explicit_caption: str | None = None,
) -> AssetRecord:
    page_id = _page_before(
        markdown,
        markdown_start if markdown_start is not None else len(markdown),
        relative,
    )
    page_caption, page_start, page_end = _caption_for_page(markdown, page_id, relative)
    start = page_start if markdown_start is None else markdown_start
    end = page_end if markdown_end is None else markdown_end
    caption = explicit_caption or page_caption
    return AssetRecord(
        asset_id=_asset_id(paper_id, document_id, relative),
        paper_id=paper_id,
        document_id=document_id,
        document_role=document_role,
        asset_type=_asset_type(relative, caption),
        relative_path=relative,
        absolute_path=str(absolute),
        exists=absolute.exists() and absolute.is_file(),
        sha256=_sha256_file(absolute) if absolute.exists() and absolute.is_file() else None,
        page_id=page_id,
        section=_section_before(markdown, start),
        caption=caption,
        marker_alt_text=marker_alt_text,
        markdown_start=start,
        markdown_end=end,
        referenced_in_markdown=referenced_in_markdown,
        discovery_method=discovery_method,
    )


def _merge_records(previous: AssetRecord, record: AssetRecord) -> AssetRecord:
    return AssetRecord(
        asset_id=previous.asset_id,
        paper_id=previous.paper_id,
        document_id=previous.document_id,
        document_role=previous.document_role,
        asset_type=(record.asset_type if previous.asset_type == "other" else previous.asset_type),
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
        referenced_in_markdown=(previous.referenced_in_markdown or record.referenced_in_markdown),
        discovery_method=(
            "markdown_image+package_scan"
            if previous.discovery_method != record.discovery_method
            else previous.discovery_method
        ),
    )


def build_asset_index(
    *,
    paper_id: str,
    document_id: str,
    document_role: str,
    package_dir: str | Path,
    markdown: str,
) -> list[AssetRecord]:
    """Index both Markdown-linked and loose Marker image files.

    Marker occasionally writes image files beside the Markdown without emitting
    a Markdown image token. Those files are still first-class source assets and
    are associated with chunks by page/caption during chunking.
    """
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
            relative = target

        record = _record_for_file(
            paper_id=paper_id,
            document_id=document_id,
            document_role=document_role,
            package_dir=package_dir,
            markdown=markdown,
            absolute=absolute,
            relative=relative,
            referenced_in_markdown=True,
            discovery_method="markdown_image",
            marker_alt_text=match.group("alt").strip() or None,
            markdown_start=match.start(),
            markdown_end=match.end(),
            explicit_caption=_caption_after(markdown, match.end()),
        )
        by_path[relative] = _merge_records(by_path[relative], record) if relative in by_path else record

    if package_dir.exists():
        for absolute in sorted(package_dir.rglob("*")):
            if not absolute.is_file() or absolute.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            relative = str(absolute.relative_to(package_dir))
            record = _record_for_file(
                paper_id=paper_id,
                document_id=document_id,
                document_role=document_role,
                package_dir=package_dir,
                markdown=markdown,
                absolute=absolute,
                relative=relative,
                referenced_in_markdown=False,
                discovery_method="package_scan",
            )
            by_path[relative] = _merge_records(by_path[relative], record) if relative in by_path else record

    return sorted(
        by_path.values(),
        key=lambda item: (
            item.page_id is None,
            item.page_id if item.page_id is not None else 10**9,
            item.markdown_start,
            item.relative_path,
        ),
    )


def assets_by_id(assets: Iterable[AssetRecord]) -> dict[str, AssetRecord]:
    return {asset.asset_id: asset for asset in assets}


def asset_path_to_id(assets: Iterable[AssetRecord]) -> dict[str, str]:
    return {asset.relative_path: asset.asset_id for asset in assets}


def write_assets_jsonl(path: str | Path, assets: Iterable[AssetRecord]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for asset in assets:
            file.write(json.dumps(asset.to_dict(), ensure_ascii=False) + "\n")
    return path
