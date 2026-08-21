"""Shared document locator indexing and provenance helpers."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from pipeline_core.corpus.extraction.asset_index import AssetRecord


_PAGE_ANCHOR_RE = re.compile(
    r"<span\s+id=[\"']page-(?P<page>\d+)-[^\"']+[\"']\s*></span>",
    re.IGNORECASE,
)
_LOCATOR_RE = re.compile(
    r"\b(?:(?P<prefix>supplementary|supplemental)\s+)?"
    r"(?P<kind>figs?|figures?|tables?|schemes?)\.?\s*"
    r"(?P<label>S?\d+)(?P<panel>[A-Za-z])?\b",
    re.IGNORECASE,
)
_MARKDOWN_PREFIX_RE = re.compile(
    r"^\s*(?:<span\s+id=[\"']page-\d+-[^\"']+[\"']\s*></span>\s*)?"
    r"(?:#{1,6}\s*)?(?:[-*+]\s*)?(?:\*\*|__)?\s*",
    re.IGNORECASE,
)
_VISUAL_KINDS = {"figure", "scheme"}


@dataclass(frozen=True)
class LocatorOccurrence:
    document_id: str
    document_role: str
    locator_key: str
    base_locator_key: str
    locator_text: str
    kind: str
    label: str
    panel: str | None
    page_id: int | None
    markdown_start: int
    markdown_end: int
    caption_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocatorIndexRecord:
    document_id: str
    document_role: str
    locator_key: str
    base_locator_key: str
    locator_text: str
    kind: str
    label: str
    panel: str | None
    page_id: int | None
    asset_ids: tuple[str, ...]
    asset_paths: tuple[str, ...]
    has_markdown_block: bool
    mapping_method: str
    confidence: str
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["asset_ids"] = list(self.asset_ids)
        payload["asset_paths"] = list(self.asset_paths)
        return payload


def _kind(value: str) -> str:
    lowered = value.lower().replace(".", "")
    if lowered.startswith("fig"):
        return "figure"
    if lowered.startswith("scheme"):
        return "scheme"
    return "table"


def _canonical_label(
    *,
    label: str,
    panel: str | None,
    document_role: str,
    supplementary_prefix: bool,
) -> tuple[str, str | None]:
    label = label.upper()
    if (
        document_role == "supporting_information" or supplementary_prefix
    ) and not label.startswith("S"):
        label = f"S{label}"
    panel = panel.upper() if panel else None
    return label, panel


def locator_keys_from_text(
    value: Any,
    *,
    document_role: str = "supporting_information",
) -> tuple[str, ...]:
    keys: list[str] = []
    for match in _LOCATOR_RE.finditer(str(value or "")):
        kind = _kind(match.group("kind"))
        label, panel = _canonical_label(
            label=match.group("label"),
            panel=match.group("panel"),
            document_role=document_role,
            supplementary_prefix=bool(match.group("prefix")),
        )
        exact = f"{kind}:{label}{panel or ''}".lower()
        base = f"{kind}:{label}".lower()
        for key in (exact, base):
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def _page_at(markdown: str, position: int) -> int | None:
    page_id: int | None = None
    for match in _PAGE_ANCHOR_RE.finditer(markdown, 0, max(0, position) + 1):
        page_id = int(match.group("page"))
    return page_id


def _caption_occurrences(
    *,
    document_id: str,
    document_role: str,
    markdown: str,
) -> list[LocatorOccurrence]:
    occurrences: list[LocatorOccurrence] = []
    offset = 0
    for line in markdown.splitlines(keepends=True):
        raw_line = line.rstrip("\r\n")
        cleaned = _MARKDOWN_PREFIX_RE.sub("", raw_line)
        match = _LOCATOR_RE.match(cleaned)
        if match is None:
            offset += len(line)
            continue

        kind = _kind(match.group("kind"))
        label, panel = _canonical_label(
            label=match.group("label"),
            panel=match.group("panel"),
            document_role=document_role,
            supplementary_prefix=bool(match.group("prefix")),
        )
        locator_key = f"{kind}:{label}{panel or ''}".lower()
        base_key = f"{kind}:{label}".lower()
        locator_text = match.group(0).strip()
        occurrences.append(
            LocatorOccurrence(
                document_id=document_id,
                document_role=document_role,
                locator_key=locator_key,
                base_locator_key=base_key,
                locator_text=locator_text,
                kind=kind,
                label=label,
                panel=panel,
                page_id=_page_at(markdown, offset + match.start()),
                markdown_start=offset,
                markdown_end=offset + len(raw_line),
                caption_text=cleaned.strip(),
            )
        )
        offset += len(line)
    return occurrences


def _visual_assets(assets: Iterable[AssetRecord]) -> list[AssetRecord]:
    return [
        asset
        for asset in assets
        if asset.exists and asset.asset_type not in {"table_image", "equation"}
    ]


def _append_mapping(
    mappings: dict[str, LocatorIndexRecord],
    *,
    occurrence: LocatorOccurrence,
    assets: Iterable[AssetRecord],
    method: str,
    confidence: str,
    ambiguous: bool = False,
) -> None:
    assets = tuple(assets)
    record = LocatorIndexRecord(
        document_id=occurrence.document_id,
        document_role=occurrence.document_role,
        locator_key=occurrence.locator_key,
        base_locator_key=occurrence.base_locator_key,
        locator_text=occurrence.locator_text,
        kind=occurrence.kind,
        label=occurrence.label,
        panel=occurrence.panel,
        page_id=(
            occurrence.page_id
            if occurrence.page_id is not None
            else (assets[0].page_id if len(assets) == 1 else None)
        ),
        asset_ids=tuple(sorted({asset.asset_id for asset in assets})),
        asset_paths=tuple(sorted({asset.relative_path for asset in assets})),
        has_markdown_block=True,
        mapping_method=method,
        confidence=confidence,
        ambiguous=ambiguous,
    )

    previous = mappings.get(occurrence.locator_key)
    if previous is None:
        mappings[occurrence.locator_key] = record
    else:
        combined_ids = tuple(sorted(set(previous.asset_ids) | set(record.asset_ids)))
        combined_paths = tuple(sorted(set(previous.asset_paths) | set(record.asset_paths)))
        mappings[occurrence.locator_key] = LocatorIndexRecord(
            document_id=record.document_id,
            document_role=record.document_role,
            locator_key=record.locator_key,
            base_locator_key=record.base_locator_key,
            locator_text=previous.locator_text or record.locator_text,
            kind=record.kind,
            label=record.label,
            panel=record.panel,
            page_id=previous.page_id if previous.page_id is not None else record.page_id,
            asset_ids=combined_ids,
            asset_paths=combined_paths,
            has_markdown_block=True,
            mapping_method=(
                previous.mapping_method
                if previous.mapping_method == record.mapping_method
                else f"{previous.mapping_method}+{record.mapping_method}"
            ),
            confidence=(
                "high"
                if "high" in {previous.confidence, record.confidence}
                else "medium"
            ),
            ambiguous=previous.ambiguous or record.ambiguous or len(combined_ids) > 1,
        )

    # Panel references should be able to fall back to the parent figure.
    if occurrence.base_locator_key != occurrence.locator_key:
        base_occurrence = LocatorOccurrence(
            **{
                **occurrence.to_dict(),
                "locator_key": occurrence.base_locator_key,
                "panel": None,
            }
        )
        if occurrence.base_locator_key not in mappings:
            _append_mapping(
                mappings,
                occurrence=base_occurrence,
                assets=assets,
                method=method,
                confidence=confidence,
                ambiguous=ambiguous,
            )


def _override_records(
    *,
    override_path: Path | None,
    document_id: str,
    document_role: str,
    assets: Iterable[AssetRecord],
) -> list[LocatorIndexRecord]:
    if override_path is None or not override_path.exists():
        return []
    payload = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    documents = payload.get("documents", {}) if isinstance(payload, dict) else {}
    document_payload = documents.get(document_id, {}) if isinstance(documents, dict) else {}
    if not isinstance(document_payload, dict):
        return []

    by_path = {asset.relative_path: asset for asset in assets}
    rows: list[LocatorIndexRecord] = []
    for raw_key, raw_value in document_payload.items():
        if not isinstance(raw_value, dict):
            continue
        key = str(raw_key).strip().lower()
        parts = key.split(":", 1)
        if len(parts) != 2 or parts[0] not in {"figure", "scheme", "table"}:
            continue
        kind, raw_label = parts
        label_match = re.fullmatch(r"(?P<label>S?\d+)(?P<panel>[A-Za-z])?", raw_label, re.I)
        if label_match is None:
            continue
        label, panel = _canonical_label(
            label=label_match.group("label"),
            panel=label_match.group("panel"),
            document_role=document_role,
            supplementary_prefix=True,
        )
        canonical_key = f"{kind}:{label}{panel or ''}".lower()
        base_key = f"{kind}:{label}".lower()
        raw_paths = raw_value.get("assets", []) or []
        selected = [by_path[path] for path in raw_paths if path in by_path]
        page_id = raw_value.get("page_id")
        try:
            page_id = int(page_id) if page_id is not None else None
        except (TypeError, ValueError):
            page_id = None
        rows.append(
            LocatorIndexRecord(
                document_id=document_id,
                document_role=document_role,
                locator_key=canonical_key,
                base_locator_key=base_key,
                locator_text=str(raw_value.get("locator_text") or raw_key),
                kind=kind,
                label=label,
                panel=panel,
                page_id=page_id,
                asset_ids=tuple(sorted(asset.asset_id for asset in selected)),
                asset_paths=tuple(sorted(asset.relative_path for asset in selected)),
                has_markdown_block=bool(raw_value.get("has_markdown_block", True)),
                mapping_method="manual_override",
                confidence="high",
                ambiguous=False,
            )
        )
    return rows


def build_locator_index(
    *,
    document_id: str,
    document_role: str,
    markdown: str,
    assets: Iterable[AssetRecord],
    override_path: str | Path | None = None,
) -> list[LocatorIndexRecord]:
    assets = tuple(assets)
    occurrences = _caption_occurrences(
        document_id=document_id,
        document_role=document_role,
        markdown=markdown,
    )
    mappings: dict[str, LocatorIndexRecord] = {}
    visual_assets = _visual_assets(assets)

    # 1. Existing asset captions are the strongest automatic signal.
    for asset in visual_assets:
        keys = locator_keys_from_text(asset.caption, document_role=document_role)
        for key in keys:
            occurrence = next(
                (item for item in occurrences if item.locator_key == key),
                None,
            )
            if occurrence is None:
                kind, label = key.split(":", 1)
                panel_match = re.fullmatch(r"(?P<label>S?\d+)(?P<panel>[A-Za-z])?", label, re.I)
                if panel_match is None:
                    continue
                canonical_label, panel = _canonical_label(
                    label=panel_match.group("label"),
                    panel=panel_match.group("panel"),
                    document_role=document_role,
                    supplementary_prefix=True,
                )
                occurrence = LocatorOccurrence(
                    document_id=document_id,
                    document_role=document_role,
                    locator_key=key,
                    base_locator_key=f"{kind}:{canonical_label}".lower(),
                    locator_text=asset.caption or key,
                    kind=kind,
                    label=canonical_label,
                    panel=panel,
                    page_id=asset.page_id,
                    markdown_start=asset.markdown_start,
                    markdown_end=asset.markdown_end,
                    caption_text=asset.caption or "",
                )
            _append_mapping(
                mappings,
                occurrence=occurrence,
                assets=[asset],
                method="asset_caption_exact",
                confidence="high",
            )

    # 2. Link caption blocks to assets on the same Marker page.
    by_page_assets: dict[int, list[AssetRecord]] = {}
    by_page_occurrences: dict[int, list[LocatorOccurrence]] = {}
    for asset in visual_assets:
        if asset.page_id is not None:
            by_page_assets.setdefault(asset.page_id, []).append(asset)
    for occurrence in occurrences:
        if occurrence.kind in _VISUAL_KINDS and occurrence.page_id is not None:
            by_page_occurrences.setdefault(occurrence.page_id, []).append(occurrence)

    assigned_asset_ids = {
        asset_id
        for record in mappings.values()
        for asset_id in record.asset_ids
    }
    assigned_locator_keys = set(mappings)

    for page_id, page_occurrences in by_page_occurrences.items():
        page_assets = sorted(
            by_page_assets.get(page_id, []),
            key=lambda item: (item.markdown_start, item.relative_path),
        )
        if not page_assets:
            continue
        unresolved = [
            item for item in page_occurrences
            if item.locator_key not in assigned_locator_keys
        ]
        if not unresolved:
            continue
        available = [
            item for item in page_assets if item.asset_id not in assigned_asset_ids
        ] or page_assets

        if len(available) == 1:
            for occurrence in unresolved:
                _append_mapping(
                    mappings,
                    occurrence=occurrence,
                    assets=available,
                    method="singleton_asset_on_caption_page",
                    confidence="high",
                )
                assigned_locator_keys.add(occurrence.locator_key)
                assigned_asset_ids.add(available[0].asset_id)
        elif len(available) == len(unresolved):
            for occurrence, asset in zip(
                sorted(unresolved, key=lambda item: item.markdown_start),
                available,
            ):
                _append_mapping(
                    mappings,
                    occurrence=occurrence,
                    assets=[asset],
                    method="ordered_assets_on_caption_page",
                    confidence="medium",
                )
                assigned_locator_keys.add(occurrence.locator_key)
                assigned_asset_ids.add(asset.asset_id)

    # 3. Conservative document-order fallback when Marker omitted page anchors.
    unresolved_occurrences = [
        item
        for item in occurrences
        if item.kind in _VISUAL_KINDS and item.locator_key not in mappings
    ]
    unresolved_assets = [
        item for item in visual_assets if item.asset_id not in assigned_asset_ids
    ]
    if unresolved_occurrences and len(unresolved_occurrences) == len(unresolved_assets):
        for occurrence, asset in zip(
            sorted(unresolved_occurrences, key=lambda item: item.markdown_start),
            sorted(
                unresolved_assets,
                key=lambda item: (
                    item.page_id is None,
                    item.page_id if item.page_id is not None else 10**9,
                    item.relative_path,
                ),
            ),
        ):
            _append_mapping(
                mappings,
                occurrence=occurrence,
                assets=[asset],
                method="document_order_alignment",
                confidence="medium",
            )

    # 4. Table records preserve block/page provenance without requiring pixels.
    for occurrence in occurrences:
        if occurrence.kind != "table" or occurrence.locator_key in mappings:
            continue
        _append_mapping(
            mappings,
            occurrence=occurrence,
            assets=[],
            method="markdown_table_block",
            confidence="high",
        )

    for record in _override_records(
        override_path=Path(override_path) if override_path else None,
        document_id=document_id,
        document_role=document_role,
        assets=assets,
    ):
        mappings[record.locator_key] = record
        if record.base_locator_key != record.locator_key:
            mappings.setdefault(record.base_locator_key, record)

    return sorted(
        mappings.values(),
        key=lambda item: (
            item.kind,
            item.label,
            item.panel or "",
            item.locator_key,
        ),
    )


def locator_index_lookup(
    records: Iterable[LocatorIndexRecord | dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in records:
        record = raw.to_dict() if isinstance(raw, LocatorIndexRecord) else dict(raw)
        document_id = str(record.get("document_id", ""))
        key = str(record.get("locator_key", "")).lower()
        if document_id and key:
            lookup[(document_id, key)] = record
    return lookup


def write_locator_index_json(
    path: str | Path,
    records: Iterable[LocatorIndexRecord],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"locators": [record.to_dict() for record in records]}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_locator_index_csv(
    path: str | Path,
    records: Iterable[LocatorIndexRecord],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    fieldnames = [
        "document_id", "document_role", "locator_key", "base_locator_key",
        "locator_text", "kind", "label", "panel", "page_id",
        "asset_ids", "asset_paths", "has_markdown_block", "mapping_method",
        "confidence", "ambiguous",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["asset_ids"] = json.dumps(row.get("asset_ids", []), ensure_ascii=False)
            row["asset_paths"] = json.dumps(row.get("asset_paths", []), ensure_ascii=False)
            writer.writerow({name: row.get(name) for name in fieldnames})
    return path


def load_locator_index(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("locators", []) if isinstance(payload, dict) else []
    return [dict(row) for row in rows if isinstance(row, dict)]
