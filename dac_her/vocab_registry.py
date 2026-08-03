from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def normalize_vocab_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9%+._\-\sηδ]+", " ", text)
    text = re.sub(r"[\s_\-/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: Any) -> str:
    normalized = normalize_vocab_text(value)
    normalized = normalized.replace("η", "eta").replace("δ", "delta")
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "unknown"


@dataclass(frozen=True)
class VocabularyEntry:
    entry_id: str
    label: str
    aliases: tuple[str, ...]
    metadata: Mapping[str, Any]


class VocabularyRegistry:
    def __init__(self, *, kind: str, version: str, entries: Mapping[str, VocabularyEntry]):
        self.kind = kind
        self.version = version
        self.entries = dict(entries)
        alias_map: dict[str, str] = {}
        for entry_id, entry in self.entries.items():
            candidates = (entry_id, entry.label, *entry.aliases)
            for candidate in candidates:
                key = normalize_vocab_text(candidate)
                prior = alias_map.get(key)
                if prior is not None and prior != entry_id:
                    raise ValueError(
                        f"Vocabulary alias collision in {kind}: {candidate!r} -> "
                        f"{prior!r}/{entry_id!r}"
                    )
                alias_map[key] = entry_id
        self.alias_map = alias_map

    @classmethod
    def from_yaml(cls, path: str | Path, *, root_key: str) -> "VocabularyRegistry":
        path = Path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Vocabulary must be a mapping: {path}")
        raw_entries = payload.get(root_key)
        if not isinstance(raw_entries, dict):
            raise ValueError(f"Vocabulary missing {root_key!r}: {path}")
        entries: dict[str, VocabularyEntry] = {}
        for entry_id, raw in raw_entries.items():
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid vocabulary entry {entry_id!r}")
            label = str(raw.get("label") or raw.get("preferred_label") or entry_id)
            aliases = tuple(str(item) for item in (raw.get("aliases") or []))
            entries[str(entry_id)] = VocabularyEntry(
                entry_id=str(entry_id),
                label=label,
                aliases=aliases,
                metadata={
                    key: value
                    for key, value in raw.items()
                    if key not in {"label", "preferred_label", "aliases"}
                },
            )
        return cls(
            kind=root_key,
            version=str(payload.get("version", "unversioned")),
            entries=entries,
        )

    def resolve(self, entry_id: str | None, label: str | None = None) -> VocabularyEntry | None:
        for value in (entry_id, label):
            if not value:
                continue
            if value in self.entries:
                return self.entries[value]
            resolved = self.alias_map.get(normalize_vocab_text(value))
            if resolved is not None:
                return self.entries[resolved]
        return None

    def canonical_or_unregistered(
        self,
        *,
        entry_id: str | None,
        label: str | None,
    ) -> tuple[str, str, bool]:
        entry = self.resolve(entry_id, label)
        if entry is not None:
            return entry.entry_id, entry.label, True
        source = label or entry_id or "unknown"
        return f"unregistered_{slugify(source)}", str(source), False

    def prompt_lines(self, *, metadata_keys: tuple[str, ...] = ()) -> list[str]:
        lines: list[str] = []
        for entry_id, entry in sorted(self.entries.items()):
            suffix = ""
            if metadata_keys:
                parts = [
                    f"{key}={entry.metadata.get(key)}"
                    for key in metadata_keys
                    if entry.metadata.get(key) is not None
                ]
                if parts:
                    suffix = " (" + ", ".join(parts) + ")"
            lines.append(f"- {entry_id}: {entry.label}{suffix}")
        return lines


def load_default_registries(project_root: str | Path) -> tuple[VocabularyRegistry, VocabularyRegistry]:
    root = Path(project_root)
    vocab_dir = root / "configs" / "vocabularies"
    experiments = VocabularyRegistry.from_yaml(
        vocab_dir / "experiment_methods.yaml",
        root_key="methods",
    )
    metrics = VocabularyRegistry.from_yaml(
        vocab_dir / "metrics.yaml",
        root_key="metrics",
    )
    return experiments, metrics
