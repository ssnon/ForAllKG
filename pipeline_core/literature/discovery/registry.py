from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .contracts import LiteratureRecord, merge_literature_records


class LiteratureRegistry:
    SCHEMA_VERSION = "graphagentsdac-literature-registry-v01"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, LiteratureRecord] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = {}
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported literature registry schema: {version}")
        self.entries = {
            paper_id: LiteratureRecord.from_dict(record)
            for paper_id, record in payload.get("papers", {}).items()
        }

    def get(self, paper_id: str) -> LiteratureRecord | None:
        return self.entries.get(paper_id)

    def upsert(self, record: LiteratureRecord) -> LiteratureRecord:
        current = self.entries.get(record.paper_id)
        merged = record if current is None else merge_literature_records(current, record)
        self.entries[record.paper_id] = merged
        return merged

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "papers": {
                paper_id: self.entries[paper_id].to_dict()
                for paper_id in sorted(self.entries)
            },
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
