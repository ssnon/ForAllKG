from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .registry import PaperRegistry


def build_corpus_manifest(
    registry: PaperRegistry,
    output: str | Path,
    corpus_id: str,
    include_warnings: bool = True,
) -> dict:
    allowed = {"passed"}
    if include_warnings:
        allowed.add("passed_with_warnings")
    documents = []
    for paper_id, entry in sorted(registry.entries.items()):
        if entry.qc_status not in allowed or not entry.main_markdown:
            continue
        documents.append(
            {
                "paper_id": paper_id,
                "title": entry.title,
                "annotator": entry.annotator,
                "main_markdown": entry.main_markdown,
                "supporting_markdown": list(entry.si_markdown),
                "source_file_name": entry.source_file_name,
                "source_fingerprint": entry.source_fingerprint,
                "marker_version": entry.marker_version,
                "qc_status": entry.qc_status,
            }
        )
    payload = {
        "schema_version": "graphagentsdac-ingestion-corpus-v01",
        "corpus_id": corpus_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "documents": documents,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
