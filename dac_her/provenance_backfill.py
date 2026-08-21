from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from pipeline_core.asset_index import AssetRecord, write_assets_jsonl
from pipeline_core.document_config import PaperConfig
from pipeline_core.document_package import load_document_package
from pipeline_core.locator_index import (
    LocatorIndexRecord,
    build_locator_index,
    locator_index_lookup,
    locator_keys_from_text,
    write_locator_index_csv,
    write_locator_index_json,
)
from pipeline_core.serialization_primitives import write_json


_VISUAL_LOCATOR_RE = re.compile(
    r"\b(?:(?:supplementary|supplemental)\s+)?"
    r"(?:figs?|figures?|schemes?)\.?\s*S?\d+[A-Za-z]?\b",
    re.IGNORECASE,
)
_TABLE_LOCATOR_RE = re.compile(
    r"\b(?:(?:supplementary|supplemental)\s+)?"
    r"tables?\.?\s*S?\d+[A-Za-z]?\b",
    re.IGNORECASE,
)


def _normalized(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("supplemental", "supplementary")
    text = text.replace("figure", "fig")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def refresh_run_asset_manifest(
    *,
    paper: PaperConfig,
    run_dir: str | Path,
    locator_override_path: str | Path | None = None,
) -> dict[str, AssetRecord]:
    """Re-index source packages and build a document locator index.

    Loose Marker assets are discovered from the package directory. Figure,
    scheme, and table captions are indexed independently from edge extraction,
    allowing an existing run to recover ``Figure S# -> page -> asset`` links at
    build time without another LLM call.
    """
    run_dir = Path(run_dir)
    documents_dir = run_dir / "documents"
    all_assets: dict[str, AssetRecord] = {}
    all_locators: list[LocatorIndexRecord] = []
    document_records: list[dict[str, Any]] = []

    default_override = locator_override_path
    if default_override is None:
        # run_dir: <project>/data_dac/extracted/<paper>/runs/<run_id>
        try:
            project_root = run_dir.parents[4]
            candidate = (
                project_root
                / "configs"
                / "locator_overrides"
                / f"{paper.paper_id}.yaml"
            )
            if candidate.exists():
                default_override = candidate
        except IndexError:
            default_override = None

    for document in paper.documents:
        package = load_document_package(paper_id=paper.paper_id, config=document)
        document_dir = documents_dir / document.document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        write_assets_jsonl(document_dir / "assets.jsonl", package.assets)

        locators = build_locator_index(
            document_id=document.document_id,
            document_role=document.role,
            markdown=package.markdown,
            assets=package.assets,
            override_path=default_override,
        )
        write_locator_index_json(document_dir / "locator_index.json", locators)
        write_locator_index_csv(document_dir / "locator_index.csv", locators)
        all_locators.extend(locators)

        record = {
            "paper_id": paper.paper_id,
            "document_id": document.document_id,
            "role": document.role,
            "package_dir": str(document.package_dir),
            "markdown_path": str(document.markdown_path),
            "metadata_path": str(document.metadata_path) if document.metadata_path else None,
            "asset_count": len(package.assets),
            "loose_asset_count": sum(not item.referenced_in_markdown for item in package.assets),
            "missing_asset_count": sum(not item.exists for item in package.assets),
            "locator_count": len(locators),
            "visual_locator_count": sum(
                item.kind in {"figure", "scheme"} for item in locators
            ),
            "visual_locator_with_asset_count": sum(
                item.kind in {"figure", "scheme"} and bool(item.asset_ids)
                for item in locators
            ),
            "table_locator_count": sum(item.kind == "table" for item in locators),
        }
        document_records.append(record)
        write_json(document_dir / "document.json", record)
        for asset in package.assets:
            if asset.asset_id in all_assets:
                raise ValueError(f"Duplicate asset ID while refreshing manifest: {asset.asset_id}")
            all_assets[asset.asset_id] = asset

    write_json(run_dir / "documents.json", {"documents": document_records})
    write_json(
        run_dir / "asset_manifest.json",
        {"assets": [asset.to_dict() for asset in all_assets.values()]},
    )
    write_locator_index_json(run_dir / "locator_index.json", all_locators)
    write_locator_index_csv(run_dir / "locator_index.csv", all_locators)
    return all_assets


def _locator_record(
    *,
    document_id: str,
    pointer: dict[str, Any],
    edge_data: dict[str, Any],
    locator_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    locator_source = " | ".join(
        str(value or "")
        for value in (
            pointer.get("locator_text"),
            edge_data.get("section"),
            edge_data.get("subsection"),
            edge_data.get("evidence_text"),
        )
    )
    keys = locator_keys_from_text(
        locator_source,
        document_role=str(edge_data.get("document_role", "supporting_information")),
    )
    for key in keys:
        record = locator_lookup.get((document_id, key))
        if record is not None:
            return record
    return None


def _candidate_assets(
    *,
    document_assets: list[AssetRecord],
    pointer: dict[str, Any],
    edge_data: dict[str, Any],
    locator_record: dict[str, Any] | None = None,
) -> list[AssetRecord]:
    by_id = {asset.asset_id: asset for asset in document_assets}
    if locator_record is not None:
        indexed_ids = [str(value) for value in locator_record.get("asset_ids", [])]
        indexed = [by_id[asset_id] for asset_id in indexed_ids if asset_id in by_id]
        if indexed and not bool(locator_record.get("ambiguous", False)):
            return indexed

    locator_source = " | ".join(
        str(value or "")
        for value in (
            pointer.get("locator_text"),
            edge_data.get("section"),
            edge_data.get("subsection"),
            edge_data.get("evidence_text"),
        )
    )
    explicit_visual_locator = bool(_VISUAL_LOCATOR_RE.search(locator_source))

    page_ids = {
        int(value)
        for value in _json_list(edge_data.get("page_ids_json"))
        if str(value).isdigit()
    }
    if pointer.get("page_id") is not None:
        try:
            page_ids.add(int(pointer["page_id"]))
        except (TypeError, ValueError):
            pass

    by_page = [asset for asset in document_assets if asset.page_id in page_ids]
    locator_keys = set(
        locator_keys_from_text(
            locator_source,
            document_role=str(edge_data.get("document_role", "supporting_information")),
        )
    )
    by_locator: list[AssetRecord] = []
    if locator_keys:
        for asset in document_assets:
            asset_text = " | ".join(
                filter(None, (asset.caption, asset.section, asset.relative_path))
            )
            asset_keys = set(
                locator_keys_from_text(
                    asset_text,
                    document_role=asset.document_role,
                )
            )
            if locator_keys & asset_keys:
                by_locator.append(asset)

    if by_page and by_locator:
        intersection = [asset for asset in by_page if asset in by_locator]
        if intersection:
            return intersection
    if by_locator:
        return by_locator
    if explicit_visual_locator and len(by_page) == 1:
        return by_page
    return []


def backfill_edge_asset_provenance(
    graph: nx.MultiDiGraph,
    *,
    assets: Iterable[AssetRecord],
    locator_index: Iterable[LocatorIndexRecord | dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Backfill page and asset provenance from a locator index.

    Priority:
      1. exact/base Figure/Schema/Table locator index;
      2. legacy caption match;
      3. one visual asset on an explicitly identified page.

    Table locators receive page/block provenance but are not forced to have a
    raster asset. Existing asset pointers are never overwritten.
    """
    by_document: dict[str, list[AssetRecord]] = {}
    for asset in assets:
        by_document.setdefault(asset.document_id, []).append(asset)
    lookup = locator_index_lookup(locator_index)

    rows: list[dict[str, Any]] = []
    for source, target, key, raw_data in graph.edges(keys=True, data=True):
        data = dict(raw_data)
        document_id = str(data.get("document_id", ""))
        document_assets = by_document.get(document_id, [])
        if not document_assets and not lookup:
            continue

        pointers = _json_list(data.get("evidence_pointers_json"))
        if not pointers:
            continue
        changed = False
        added_to_edge: set[str] = set()

        for pointer in pointers:
            if not isinstance(pointer, dict):
                continue
            record = _locator_record(
                document_id=document_id,
                pointer=pointer,
                edge_data=data,
                locator_lookup=lookup,
            )
            locator_source = " | ".join(
                str(value or "")
                for value in (
                    pointer.get("locator_text"),
                    data.get("section"),
                    data.get("subsection"),
                )
            )
            is_table = bool(_TABLE_LOCATOR_RE.search(locator_source))

            page_changed = False
            if record is not None and pointer.get("page_id") is None:
                if record.get("page_id") is not None:
                    pointer["page_id"] = int(record["page_id"])
                    page_changed = True
            if record is not None:
                pointer["locator_key"] = record.get("locator_key")
                pointer["locator_mapping_method"] = record.get("mapping_method")

            asset_ids_before = [str(value) for value in pointer.get("asset_ids") or []]
            candidates: list[AssetRecord] = []
            if not asset_ids_before and not is_table:
                candidates = _candidate_assets(
                    document_assets=document_assets,
                    pointer=pointer,
                    edge_data=data,
                    locator_record=record,
                )

            asset_ids = sorted({asset.asset_id for asset in candidates})
            if asset_ids:
                pointer["asset_ids"] = asset_ids
                added_to_edge.update(asset_ids)
                changed = True
            elif page_changed or record is not None:
                changed = True

            if asset_ids or page_changed or record is not None:
                rows.append({
                    "source": str(source),
                    "relation": str(data.get("relation", "")),
                    "target": str(target),
                    "edge_key": str(key),
                    "document_id": document_id,
                    "page_id": pointer.get("page_id"),
                    "locator_text": pointer.get("locator_text"),
                    "locator_key": pointer.get("locator_key"),
                    "asset_ids": asset_ids or asset_ids_before,
                    "mapping_method": (
                        record.get("mapping_method") if record is not None else "legacy_page_or_caption"
                    ),
                    "action": (
                        "backfilled_table_block_page"
                        if is_table and not asset_ids
                        else "backfilled_locator_page_and_asset"
                    ),
                })

        if not changed:
            continue
        existing_chunk_assets = {str(value) for value in _json_list(data.get("asset_ids_json"))}
        existing_evidence_assets = {
            str(value) for value in _json_list(data.get("evidence_asset_ids_json"))
        }
        data["evidence_pointers_json"] = json.dumps(pointers, ensure_ascii=False)
        data["asset_ids_json"] = json.dumps(
            sorted(existing_chunk_assets | added_to_edge), ensure_ascii=False
        )
        data["evidence_asset_ids_json"] = json.dumps(
            sorted(existing_evidence_assets | added_to_edge), ensure_ascii=False
        )
        graph.edges[source, target, key].update(data)

    return rows
