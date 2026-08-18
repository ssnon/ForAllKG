from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pipeline_core.asset_index import AssetRecord, build_asset_index
from pipeline_core.document_config import DocumentConfig
from pipeline_core.markdown import extract_markdown_section


_SUPPLEMENTARY_REFERENCE_RE = re.compile(
    r"\b(?:Supplementary|Supplemental|Suppl\.|Supporting\s+Information)\s+"
    r"(?P<kind>Figs?\.?|Figures?|Tables?|Notes?|Methods?|Sections?|"
    r"Schemes?|Equations?|Eqs?\.?|Videos?|Data)\s*"
    r"(?P<label>"
    r"[A-Za-z]?\d+[A-Za-z]?"
    r"(?:\s*[–—-]\s*[A-Za-z]?\d+[A-Za-z]?)?"
    r"(?:\s*(?:,|and)\s*[A-Za-z]?\d+[A-Za-z]?"
    r"(?:\s*[–—-]\s*[A-Za-z]?\d+[A-Za-z]?)?)*"
    r")",
    re.IGNORECASE,
)
_STANDALONE_S_REFERENCE_RE = re.compile(
    r"\b(?P<kind>Figs?\.?|Figures?|Tables?|Schemes?|Equations?|Eqs?\.?)\s*"
    r"(?P<label>S\d+[A-Za-z]?"
    r"(?:\s*[–—-]\s*S?\d+[A-Za-z]?)?"
    r"(?:\s*(?:,|and)\s*S?\d+[A-Za-z]?"
    r"(?:\s*[–—-]\s*S?\d+[A-Za-z]?)?)*"
    r")",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class DocumentPackage:
    paper_id: str
    document_id: str
    role: str
    package_dir: Path
    markdown_path: Path
    metadata_path: Path | None
    markdown: str
    metadata: dict | None
    assets: tuple[AssetRecord, ...]


@dataclass(frozen=True)
class SelectedSource:
    paper_id: str
    document_id: str
    document_role: str
    selection_id: str
    section: str
    text: str


def load_document_package(
    *,
    paper_id: str,
    config: DocumentConfig,
) -> DocumentPackage:
    if not config.markdown_path.exists():
        raise FileNotFoundError(
            f"Markdown not found for {paper_id}/{config.document_id}: "
            f"{config.markdown_path}"
        )

    markdown = config.markdown_path.read_text(encoding="utf-8")
    metadata: dict | None = None
    if config.metadata_path is not None and config.metadata_path.exists():
        loaded = json.loads(config.metadata_path.read_text(encoding="utf-8"))
        metadata = loaded if isinstance(loaded, dict) else {"value": loaded}

    assets = tuple(
        build_asset_index(
            paper_id=paper_id,
            document_id=config.document_id,
            document_role=config.role,
            package_dir=config.package_dir,
            markdown=markdown,
        )
    )

    return DocumentPackage(
        paper_id=paper_id,
        document_id=config.document_id,
        role=config.role,
        package_dir=config.package_dir,
        markdown_path=config.markdown_path,
        metadata_path=config.metadata_path,
        markdown=markdown,
        metadata=metadata,
        assets=assets,
    )


def _singular_kind(kind: str) -> str:
    normalized = kind.lower().replace(".", "")
    if normalized.startswith("fig"):
        return "Fig"
    if normalized.startswith("table"):
        return "Table"
    if normalized.startswith("note"):
        return "Note"
    if normalized.startswith("method"):
        return "Method"
    if normalized.startswith("scheme"):
        return "Scheme"
    if normalized.startswith("equation") or normalized.startswith("eq"):
        return "Equation"
    if normalized.startswith("video"):
        return "Video"
    if normalized.startswith("data"):
        return "Data"
    return "Section"


def _expand_reference_labels(label: str) -> list[str]:
    label = re.sub(r"\s+", " ", label).strip(" .,;:")
    parts = [
        part.strip()
        for part in re.split(r"\s*(?:,|and)\s*", label)
        if part.strip()
    ]
    expanded: list[str] = []
    for part in parts:
        range_match = re.fullmatch(
            r"(?P<prefix>[A-Za-z]?)"
            r"(?P<start>\d+)\s*[–—-]\s*"
            r"(?P<end_prefix>[A-Za-z]?)"
            r"(?P<end>\d+)",
            part,
        )
        if range_match:
            prefix = range_match.group("prefix")
            end_prefix = range_match.group("end_prefix") or prefix
            start = int(range_match.group("start"))
            end = int(range_match.group("end"))
            if (
                prefix.lower() == end_prefix.lower()
                and start <= end
                and end - start <= 50
            ):
                expanded.extend(
                    f"{prefix}{value}" for value in range(start, end + 1)
                )
                continue
        expanded.append(part)
    return expanded


def extract_supplementary_references(texts: Iterable[str]) -> tuple[str, ...]:
    values: set[str] = set()
    for text in texts:
        for pattern in (
            _SUPPLEMENTARY_REFERENCE_RE,
            _STANDALONE_S_REFERENCE_RE,
        ):
            for match in pattern.finditer(text):
                kind = _singular_kind(match.group("kind"))
                for label in _expand_reference_labels(match.group("label")):
                    values.add(f"Supplementary {kind} {label}")
    return tuple(sorted(values))


def _normalized(value: str) -> str:
    value = value.lower()
    value = value.replace("supplemental", "supplementary")
    value = value.replace("suppl.", "supplementary")
    value = value.replace("figure", "fig")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _block_around_match(markdown: str, start: int, end: int) -> str:
    # Prefer a Markdown heading block when the match is in the heading itself
    # or anywhere below it. Searching only before ``start`` misses captions
    # rendered by Marker as headings such as ``## Figure S3``.
    previous_heading = None
    previous_level = None
    for match in _HEADING_RE.finditer(markdown):
        if match.start() > start:
            break
        previous_heading = match
        previous_level = len(match.group(1))
        if match.start() <= start <= match.end():
            break

    if previous_heading is not None and previous_level is not None:
        next_pattern = re.compile(rf"(?m)^#{{1,{previous_level}}}\s+.+$")
        next_heading = next_pattern.search(markdown, max(end, previous_heading.end()))
        block_end = next_heading.start() if next_heading else len(markdown)
        candidate = markdown[previous_heading.start():block_end].strip()
        if len(candidate) <= 16000:
            return candidate

    paragraph_start = markdown.rfind("\n\n", 0, start)
    paragraph_start = 0 if paragraph_start == -1 else paragraph_start + 2
    paragraph_end = markdown.find("\n\n", end)
    paragraph_end = len(markdown) if paragraph_end == -1 else paragraph_end
    return markdown[paragraph_start:paragraph_end].strip()


def _reference_variants(reference: str) -> tuple[str, ...]:
    normalized = _normalized(reference)
    variants = {normalized}
    match = re.fullmatch(
        r"supplementary (?P<kind>fig|table|note|method|section|scheme|equation|video|data) "
        r"(?P<label>[a-z0-9]+)",
        normalized,
    )
    if not match:
        return tuple(sorted(variants, key=len, reverse=True))

    kind = match.group("kind")
    label = match.group("label")
    short_label = label[1:] if label.startswith("s") and label[1:].isdigit() else label
    s_label = label if label.startswith("s") else f"s{label}"

    long_kind = {
        "fig": "figure",
        "table": "table",
        "note": "note",
        "method": "method",
        "section": "section",
        "scheme": "scheme",
        "equation": "equation",
        "video": "video",
        "data": "data",
    }[kind]

    for candidate_label in {label, short_label, s_label}:
        variants.add(f"supplementary {kind} {candidate_label}")
        variants.add(f"supplementary {long_kind} {candidate_label}")
        variants.add(f"{kind} {candidate_label}")
        variants.add(f"{long_kind} {candidate_label}")

    return tuple(sorted(variants, key=len, reverse=True))


def select_referenced_blocks(
    markdown: str,
    references: Iterable[str],
) -> list[tuple[str, str]]:
    normalized_markdown = _normalized(markdown)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Use normalized text to decide whether a reference is present, then locate
    # a shorter literal token (usually the figure/table number) in original text.
    for reference in references:
        variants = _reference_variants(reference)
        matching_variant = next(
            (variant for variant in variants if variant in normalized_markdown),
            None,
        )
        if matching_variant is None:
            continue

        tokens = matching_variant.split()
        locator_tokens = tokens[-2:] if len(tokens) >= 2 else tokens
        locator_parts = []
        for token in locator_tokens:
            if token == "fig":
                locator_parts.append(r"fig(?:ure)?\.?")
            elif token == "equation":
                locator_parts.append(r"(?:equation|eq)\.?")
            else:
                locator_parts.append(re.escape(token))
        locator_pattern = r"\W*".join(locator_parts)
        match = re.search(locator_pattern, markdown, re.IGNORECASE)
        if match is None:
            match = re.search(re.escape(reference), markdown, re.IGNORECASE)
        if match is None:
            continue

        block = _block_around_match(markdown, match.start(), match.end())
        digest = _normalized(block)
        if block and digest not in seen:
            seen.add(digest)
            results.append((reference, block))

    return results


def select_document_sources(
    *,
    package: DocumentPackage,
    config: DocumentConfig,
    supplementary_references: Iterable[str] = (),
) -> list[SelectedSource]:
    selection = config.selection

    if selection.mode == "whole_document":
        return [
            SelectedSource(
                paper_id=package.paper_id,
                document_id=package.document_id,
                document_role=package.role,
                selection_id="whole_document",
                section=f"[{package.document_id}] whole document",
                text=package.markdown.strip(),
            )
        ]

    if selection.mode == "sections":
        return [
            SelectedSource(
                paper_id=package.paper_id,
                document_id=package.document_id,
                document_role=package.role,
                selection_id=f"section:{index}",
                section=heading,
                text=extract_markdown_section(package.markdown, heading),
            )
            for index, heading in enumerate(selection.headings)
        ]

    blocks = select_referenced_blocks(
        package.markdown,
        supplementary_references,
    )
    if blocks:
        return [
            SelectedSource(
                paper_id=package.paper_id,
                document_id=package.document_id,
                document_role=package.role,
                selection_id=f"referenced:{index}",
                section=reference,
                text=block,
            )
            for index, (reference, block) in enumerate(blocks)
        ]

    if selection.fallback == "whole_document":
        return [
            SelectedSource(
                paper_id=package.paper_id,
                document_id=package.document_id,
                document_role=package.role,
                selection_id="fallback:whole_document",
                section=f"[{package.document_id}] whole document",
                text=package.markdown.strip(),
            )
        ]
    if selection.fallback == "skip":
        return []
    raise ValueError(
        f"No referenced blocks found for {package.paper_id}/{package.document_id}."
    )
