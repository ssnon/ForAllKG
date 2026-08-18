from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import tiktoken

from pipeline_core.asset_index import AssetRecord
from pipeline_core.extraction_policy import ExtractionPolicy


_ENCODER = tiktoken.get_encoding("o200k_base")
_PAGE_RE = re.compile(r"(?:page-|_page_)(?P<page>\d+)", re.IGNORECASE)
_ASSET_LOCATOR_RE = re.compile(
    r"^(?:(?:Supplementary|Extended\s+Data)\s+)?"
    r"(?:Fig(?:ure)?\.?|Table)\s*[A-Za-z0-9]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkSpec:
    paper_id: str
    section: str
    index: int

    core_text: str
    left_context: str
    right_context: str

    chunk_id: str
    document_id: str = "main"
    document_role: str = "main"
    page_ids: tuple[int, ...] = ()
    asset_ids: tuple[str, ...] = ()
    asset_paths: tuple[str, ...] = ()
    asset_pages: tuple[int | None, ...] = ()
    asset_locators: tuple[str, ...] = ()
    asset_context: str = ""
    split_depth: int = 0


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def first_tokens(text: str, count: int) -> str:
    if count <= 0:
        return ""
    tokens = _ENCODER.encode(text)
    return _ENCODER.decode(tokens[:count])


def last_tokens(text: str, count: int) -> str:
    if count <= 0:
        return ""
    tokens = _ENCODER.encode(text)
    return _ENCODER.decode(tokens[-count:])


def make_chunk_id(
    paper_id: str,
    section: str,
    core_text: str,
    *,
    document_id: str = "main",
) -> str:
    # Context and generated figure analysis are not part of chunk identity.
    value = f"{paper_id}|{document_id}|{section}|{core_text}"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{paper_id}:{document_id}:{digest}"


def split_long_unit(text: str, max_tokens: int) -> list[str]:
    tokens = _ENCODER.encode(text)
    return [
        _ENCODER.decode(tokens[start:start + max_tokens]).strip()
        for start in range(0, len(tokens), max_tokens)
        if tokens[start:start + max_tokens]
    ]


def paragraph_units(text: str, max_tokens: int) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    units: list[str] = []
    for paragraph in paragraphs:
        if count_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
        else:
            units.extend(split_long_unit(paragraph, max_tokens))
    return units


def build_core_chunks(
    section_text: str,
    policy: ExtractionPolicy,
) -> list[str]:
    units = paragraph_units(section_text, policy.max_source_tokens)
    core_chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        candidate = "\n\n".join([*current, unit])
        if current and count_tokens(candidate) > policy.target_source_tokens:
            core_chunks.append("\n\n".join(current).strip())
            current = [unit]
        else:
            current.append(unit)

    if current:
        core_chunks.append("\n\n".join(current).strip())

    if len(core_chunks) >= 2:
        tail_tokens = count_tokens(core_chunks[-1])
        merged_text = core_chunks[-2] + "\n\n" + core_chunks[-1]
        if (
            tail_tokens < policy.min_source_tokens
            and count_tokens(merged_text) <= policy.max_source_tokens
        ):
            core_chunks[-2] = merged_text
            core_chunks.pop()

    return core_chunks


def _asset_locator(asset: AssetRecord) -> str:
    if asset.caption:
        match = _ASSET_LOCATOR_RE.match(asset.caption.strip())
        if match:
            return match.group(0).strip().rstrip(".")
    return ""


def _source_metadata(
    text: str,
    assets: Iterable[AssetRecord],
    *,
    section: str = "",
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int | None, ...],
    tuple[str, ...],
]:
    pages = {int(match.group("page")) for match in _PAGE_RE.finditer(text)}
    linked: list[AssetRecord] = []
    locator_haystack = f"{section}\n{text}"

    for asset in assets:
        locator = _asset_locator(asset)
        if (
            asset.relative_path in text
            or asset.asset_id in text
            or asset.relative_path.rsplit("/", 1)[-1] in text
            or (locator and re.search(re.escape(locator), locator_haystack, re.IGNORECASE))
            or (asset.page_id is not None and asset.page_id in pages)
        ):
            linked.append(asset)
            if asset.page_id is not None:
                pages.add(asset.page_id)

    return (
        tuple(sorted(pages)),
        tuple(asset.asset_id for asset in linked),
        tuple(asset.relative_path for asset in linked),
        tuple(asset.page_id for asset in linked),
        tuple(_asset_locator(asset) for asset in linked),
    )


def _filter_parent_assets(
    text: str,
    chunk: ChunkSpec,
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int | None, ...],
    tuple[str, ...],
]:
    pages = {int(match.group("page")) for match in _PAGE_RE.finditer(text)}
    asset_ids: list[str] = []
    asset_paths: list[str] = []
    asset_pages: list[int | None] = []
    asset_locators: list[str] = []

    for asset_id, path, page, locator in zip(
        chunk.asset_ids,
        chunk.asset_paths,
        chunk.asset_pages,
        chunk.asset_locators,
    ):
        if (
            path in text
            or path.rsplit("/", 1)[-1] in text
            or (locator and re.search(re.escape(locator), text, re.IGNORECASE))
            or (page is not None and page in pages)
        ):
            asset_ids.append(asset_id)
            asset_paths.append(path)
            asset_pages.append(page)
            asset_locators.append(locator)
            if page is not None:
                pages.add(page)

    return (
        tuple(sorted(pages)),
        tuple(asset_ids),
        tuple(asset_paths),
        tuple(asset_pages),
        tuple(asset_locators),
    )


def create_chunks(
    *,
    paper_id: str,
    section: str,
    section_text: str,
    policy: ExtractionPolicy,
    document_id: str = "main",
    document_role: str = "main",
    assets: Iterable[AssetRecord] = (),
) -> list[ChunkSpec]:
    cores = build_core_chunks(section_text, policy)
    assets = tuple(assets)
    chunks: list[ChunkSpec] = []

    for index, core in enumerate(cores):
        previous_core = cores[index - 1] if index > 0 else ""
        next_core = cores[index + 1] if index + 1 < len(cores) else ""
        page_ids, asset_ids, asset_paths, asset_pages, asset_locators = _source_metadata(
            core,
            assets,
            section=section,
        )

        chunks.append(
            ChunkSpec(
                paper_id=paper_id,
                document_id=document_id,
                document_role=document_role,
                section=section,
                index=index,
                core_text=core,
                left_context=last_tokens(
                    previous_core,
                    policy.left_context_tokens,
                ),
                right_context=first_tokens(
                    next_core,
                    policy.right_context_tokens,
                ),
                page_ids=page_ids,
                asset_ids=asset_ids,
                asset_paths=asset_paths,
                asset_pages=asset_pages,
                asset_locators=asset_locators,
                chunk_id=make_chunk_id(
                    paper_id,
                    section,
                    core,
                    document_id=document_id,
                ),
            )
        )

    return chunks


def split_chunk_in_half(
    chunk: ChunkSpec,
    policy: ExtractionPolicy,
) -> list[ChunkSpec]:
    tokens = _ENCODER.encode(chunk.core_text)
    if len(tokens) < 400:
        raise RuntimeError(
            "Chunk is already too small to split safely: "
            f"{chunk.chunk_id}"
        )

    midpoint = len(tokens) // 2
    left_core = _ENCODER.decode(tokens[:midpoint]).strip()
    right_core = _ENCODER.decode(tokens[midpoint:]).strip()
    left_meta = _filter_parent_assets(left_core, chunk)
    right_meta = _filter_parent_assets(right_core, chunk)

    left = ChunkSpec(
        paper_id=chunk.paper_id,
        document_id=chunk.document_id,
        document_role=chunk.document_role,
        section=chunk.section,
        index=chunk.index * 2,
        core_text=left_core,
        left_context=chunk.left_context,
        right_context=first_tokens(
            right_core,
            policy.right_context_tokens,
        ),
        page_ids=left_meta[0],
        asset_ids=left_meta[1],
        asset_paths=left_meta[2],
        asset_pages=left_meta[3],
        asset_locators=left_meta[4],
        asset_context=chunk.asset_context,
        chunk_id=make_chunk_id(
            chunk.paper_id,
            chunk.section,
            left_core,
            document_id=chunk.document_id,
        ),
        split_depth=chunk.split_depth + 1,
    )

    right = ChunkSpec(
        paper_id=chunk.paper_id,
        document_id=chunk.document_id,
        document_role=chunk.document_role,
        section=chunk.section,
        index=chunk.index * 2 + 1,
        core_text=right_core,
        left_context=last_tokens(
            left_core,
            policy.left_context_tokens,
        ),
        right_context=chunk.right_context,
        page_ids=right_meta[0],
        asset_ids=right_meta[1],
        asset_paths=right_meta[2],
        asset_pages=right_meta[3],
        asset_locators=right_meta[4],
        asset_context=chunk.asset_context,
        chunk_id=make_chunk_id(
            chunk.paper_id,
            chunk.section,
            right_core,
            document_id=chunk.document_id,
        ),
        split_depth=chunk.split_depth + 1,
    )

    return [left, right]
