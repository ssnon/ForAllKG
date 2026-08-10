from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INPUT_SCHEMA = "graphagentsdac-ingestion-corpus-v01"
OUTPUT_SCHEMA = "graphagentsdac-frozen-corpus-v01"


class CorpusFreezeError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(raw: str, project_root: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def _main_fingerprint(doc: dict[str, Any]) -> str | None:
    fp = doc.get("source_fingerprint") or {}
    value = fp.get("main")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _si_fingerprints(doc: dict[str, Any]) -> list[str]:
    fp = doc.get("source_fingerprint") or {}
    values = fp.get("si") or []
    return [str(v).strip() for v in values if str(v).strip()]


def _qc_rank(doc: dict[str, Any]) -> int:
    return {
        "passed": 0,
        "passed_with_warnings": 1,
    }.get(str(doc.get("qc_status") or ""), 9)


def _canonical_sort_key(doc: dict[str, Any]) -> tuple[Any, ...]:
    # Prefer clean QC, then the record carrying more SI, then a stable paper id.
    return (
        _qc_rank(doc),
        -len(doc.get("supporting_markdown") or []),
        str(doc.get("paper_id") or ""),
    )


def _merge_exact_duplicate_group(
    group: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = sorted(group, key=_canonical_sort_key)
    canonical = copy.deepcopy(ordered[0])
    canonical_id = str(canonical["paper_id"])

    merged_supporting: list[str] = []
    merged_si_fp: list[str] = []
    seen_supporting_keys: set[str] = set()
    seen_si_fp: set[str] = set()
    for doc in ordered:
        paths = [str(v) for v in (doc.get("supporting_markdown") or [])]
        fps = _si_fingerprints(doc)
        for index, path in enumerate(paths):
            fp = fps[index] if index < len(fps) else None
            # Prefer a content/source fingerprint when available. Two copies of
            # the same SI often live under different annotator paths.
            key = f"fp:{fp}" if fp else f"path:{path}"
            if key in seen_supporting_keys:
                continue
            seen_supporting_keys.add(key)
            merged_supporting.append(path)
            if fp and fp not in seen_si_fp:
                seen_si_fp.add(fp)
                merged_si_fp.append(fp)

    # Preserve any extra fingerprints even if a malformed input manifest has
    # more fingerprint entries than supporting paths.
    for doc in ordered:
        for value in _si_fingerprints(doc):
            if value not in seen_si_fp:
                seen_si_fp.add(value)
                merged_si_fp.append(value)
    canonical["supporting_markdown"] = merged_supporting
    source_fp = dict(canonical.get("source_fingerprint") or {})
    source_fp["si"] = merged_si_fp
    canonical["source_fingerprint"] = source_fp

    alias_ids = [str(d["paper_id"]) for d in ordered[1:]]
    canonical["alias_paper_ids"] = alias_ids
    canonical["duplicate_source_records"] = [
        {
            "paper_id": str(d.get("paper_id")),
            "title": d.get("title"),
            "annotator": d.get("annotator"),
            "source_file_name": d.get("source_file_name"),
            "main_markdown": d.get("main_markdown"),
            "supporting_markdown": list(d.get("supporting_markdown") or []),
            "qc_status": d.get("qc_status"),
        }
        for d in ordered[1:]
    ]

    report = {
        "main_fingerprint": _main_fingerprint(canonical),
        "canonical_paper_id": canonical_id,
        "duplicate_paper_ids": alias_ids,
        "all_paper_ids": [str(d["paper_id"]) for d in ordered],
        "merged_supporting_markdown_count": len(merged_supporting),
    }
    return canonical, report


def _review_groups_by_title(
    docs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        norm = _normalize_title(str(doc.get("title") or ""))
        if norm:
            groups[norm].append(doc)
    out: list[dict[str, Any]] = []
    for norm, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        out.append(
            {
                "normalized_title": norm,
                "paper_ids": [str(row["paper_id"]) for row in rows],
                "titles": [str(row.get("title") or "") for row in rows],
                "main_fingerprints": [_main_fingerprint(row) for row in rows],
                "review_required": len({_main_fingerprint(row) for row in rows}) > 1,
            }
        )
    return out


def _review_groups_by_si_fingerprint(
    docs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        pid = str(doc["paper_id"])
        for fp in _si_fingerprints(doc):
            groups[fp].append(pid)
    out = []
    for fp, paper_ids in sorted(groups.items()):
        unique = sorted(set(paper_ids))
        if len(unique) > 1:
            out.append(
                {
                    "si_fingerprint": fp,
                    "paper_ids": unique,
                    "review_required": True,
                }
            )
    return out


def _attach_content_hashes(
    doc: dict[str, Any],
    project_root: Path,
    verify_paths: bool,
) -> dict[str, Any]:
    value = copy.deepcopy(doc)
    main_raw = str(value.get("main_markdown") or "")
    if not main_raw:
        raise CorpusFreezeError(f"{value.get('paper_id')}: missing main_markdown")

    main_path = _resolve_path(main_raw, project_root)
    if verify_paths and not main_path.is_file():
        raise CorpusFreezeError(
            f"{value.get('paper_id')}: main Markdown not found: {main_path}"
        )

    supporting_hashes = []
    for raw in value.get("supporting_markdown") or []:
        path = _resolve_path(str(raw), project_root)
        if verify_paths and not path.is_file():
            raise CorpusFreezeError(
                f"{value.get('paper_id')}: supporting Markdown not found: {path}"
            )
        supporting_hashes.append(
            {
                "path": str(raw),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
        )

    value["content_fingerprint"] = {
        "main_markdown_sha256": _sha256_file(main_path) if main_path.is_file() else None,
        "supporting_markdown": supporting_hashes,
    }
    return value


@dataclass(frozen=True)
class FreezeOptions:
    include_warnings: bool = True
    verify_paths: bool = True


def freeze_ingestion_manifest(
    manifest: dict[str, Any],
    *,
    source_manifest_path: str,
    project_root: str | Path = ".",
    options: FreezeOptions | None = None,
) -> dict[str, Any]:
    options = options or FreezeOptions()
    if manifest.get("schema_version") != INPUT_SCHEMA:
        raise CorpusFreezeError(
            f"Expected schema_version={INPUT_SCHEMA!r}, got {manifest.get('schema_version')!r}"
        )

    root = Path(project_root).resolve()
    source_docs = list(manifest.get("documents") or [])
    accepted_qc = {"passed"}
    if options.include_warnings:
        accepted_qc.add("passed_with_warnings")
    eligible = [d for d in source_docs if str(d.get("qc_status")) in accepted_qc]
    excluded_qc = [d for d in source_docs if d not in eligible]

    by_main_fp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_fp: list[dict[str, Any]] = []
    for doc in eligible:
        fp = _main_fingerprint(doc)
        if fp:
            by_main_fp[fp].append(doc)
        else:
            no_fp.append(doc)

    frozen_docs: list[dict[str, Any]] = []
    exact_duplicate_groups: list[dict[str, Any]] = []
    for fp in sorted(by_main_fp):
        rows = by_main_fp[fp]
        if len(rows) == 1:
            row = copy.deepcopy(rows[0])
            row.setdefault("alias_paper_ids", [])
            row.setdefault("duplicate_source_records", [])
            frozen_docs.append(row)
        else:
            canonical, report = _merge_exact_duplicate_group(rows)
            frozen_docs.append(canonical)
            exact_duplicate_groups.append(report)
    for row in no_fp:
        copied = copy.deepcopy(row)
        copied.setdefault("alias_paper_ids", [])
        copied.setdefault("duplicate_source_records", [])
        frozen_docs.append(copied)

    frozen_docs.sort(key=lambda d: str(d.get("paper_id") or ""))
    title_review_groups = _review_groups_by_title(frozen_docs)
    si_review_groups = _review_groups_by_si_fingerprint(frozen_docs)
    frozen_docs = [
        _attach_content_hashes(
            doc,
            root,
            verify_paths=options.verify_paths,
        )
        for doc in frozen_docs
    ]

    source_count = int(manifest.get("document_count", len(source_docs)))
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "corpus_id": str(manifest.get("corpus_id") or ""),
        "created_at": _utc_now(),
        "source_manifest": source_manifest_path,
        "source_schema_version": manifest.get("schema_version"),
        "source_document_count": source_count,
        "eligible_document_count": len(eligible),
        "document_count": len(frozen_docs),
        "deduplicated_document_count": len(eligible) - len(frozen_docs),
        "excluded_qc_count": len(excluded_qc),
        "exact_duplicate_groups": exact_duplicate_groups,
        "title_review_groups": title_review_groups,
        "si_fingerprint_review_groups": si_review_groups,
        "documents": frozen_docs,
    }
    return payload


def load_and_freeze(
    input_manifest: str | Path,
    output_manifest: str | Path,
    *,
    project_root: str | Path = ".",
    include_warnings: bool = True,
    verify_paths: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_manifest)
    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    payload = freeze_ingestion_manifest(
        manifest,
        source_manifest_path=str(input_manifest),
        project_root=project_root,
        options=FreezeOptions(
            include_warnings=include_warnings,
            verify_paths=verify_paths,
        ),
    )
    out = Path(output_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
