from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .contracts import PaperRegistryEntry


class PaperRegistry:
    SCHEMA_VERSION = "graphagentsdac-paper-registry-v01"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, PaperRegistryEntry] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = {}
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported registry schema: {version}")
        self.entries = {
            key: PaperRegistryEntry.from_dict(value)
            for key, value in payload.get("papers", {}).items()
        }

    def get(self, paper_id: str) -> PaperRegistryEntry | None:
        return self.entries.get(paper_id)

    def put(self, entry: PaperRegistryEntry) -> None:
        self.entries[entry.paper_id] = entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "papers": {
                key: self.entries[key].to_dict()
                for key in sorted(self.entries)
            },
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
