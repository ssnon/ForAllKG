from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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


@dataclass(frozen=True)
class ParameterizedVocabularyMatch:
    entry: VocabularyEntry | None
    parameters: Mapping[str, str]
    matched_pattern: str | None
    matched_text: str

    @property
    def registered(self) -> bool:
        return self.entry is not None


class VocabularyRegistry:
    def __init__(
        self,
        *,
        kind: str,
        version: str,
        entries: Mapping[str, VocabularyEntry],
    ):
        self.kind = kind
        self.version = version
        self.entries = dict(entries)
        alias_map: dict[str, str] = {}
        compiled_patterns: list[tuple[str, re.Pattern[str]]] = []

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

            raw_patterns = entry.metadata.get("match_patterns") or []
            if not isinstance(raw_patterns, list):
                raise ValueError(
                    f"match_patterns must be a list for {kind}:{entry_id}"
                )
            for raw_pattern in raw_patterns:
                try:
                    compiled = re.compile(str(raw_pattern))
                except re.error as error:
                    raise ValueError(
                        f"Invalid regex for {kind}:{entry_id}: {raw_pattern!r}"
                    ) from error
                compiled_patterns.append((entry_id, compiled))

        self.alias_map = alias_map
        self.compiled_patterns = tuple(compiled_patterns)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        root_key: str,
    ) -> "VocabularyRegistry":
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

    def resolve(
        self,
        entry_id: str | None,
        label: str | None = None,
    ) -> VocabularyEntry | None:
        for value in (entry_id, label):
            if not value:
                continue
            if value in self.entries:
                return self.entries[value]
            resolved = self.alias_map.get(normalize_vocab_text(value))
            if resolved is not None:
                return self.entries[resolved]
        return None

    def resolve_parameterized(
        self,
        *,
        entry_id: str | None,
        label: str | None,
        source_texts: Sequence[str | None] = (),
    ) -> ParameterizedVocabularyMatch:
        """Resolve exact aliases first, then data-driven regex patterns.

        Named regex groups become structured parameters. For metrics this is
        used to retain analyte, orbital, site, isotope, or component context
        while keeping the metric ID generic across papers.
        """
        exact = self.resolve(entry_id, label)
        texts = [
            str(value).strip()
            for value in (entry_id, label, *source_texts)
            if value is not None and str(value).strip()
        ]
        joined = " | ".join(texts)

        if exact is not None:
            parameters: dict[str, str] = {}
            matched_pattern: str | None = None
            for candidate_id, pattern in self.compiled_patterns:
                if candidate_id != exact.entry_id:
                    continue
                match = pattern.search(joined)
                if match:
                    parameters.update(
                        {
                            key: str(value).strip()
                            for key, value in match.groupdict().items()
                            if value is not None and str(value).strip()
                        }
                    )
                    matched_pattern = pattern.pattern
                    break
            return ParameterizedVocabularyMatch(
                entry=exact,
                parameters=parameters,
                matched_pattern=matched_pattern,
                matched_text=joined,
            )

        matches: list[
            tuple[int, int, str, re.Match[str], re.Pattern[str]]
        ] = []
        for candidate_id, pattern in self.compiled_patterns:
            match = pattern.search(joined)
            if match is None:
                continue
            span_length = match.end() - match.start()
            parameter_count = sum(
                value is not None and str(value).strip() != ""
                for value in match.groupdict().values()
            )
            matches.append(
                (
                    parameter_count,
                    span_length,
                    candidate_id,
                    match,
                    pattern,
                )
            )

        if not matches:
            return ParameterizedVocabularyMatch(
                entry=None,
                parameters={},
                matched_pattern=None,
                matched_text=joined,
            )

        _, _, candidate_id, match, pattern = max(
            matches,
            key=lambda item: (item[0], item[1], item[2]),
        )
        parameters = {
            key: str(value).strip()
            for key, value in match.groupdict().items()
            if value is not None and str(value).strip()
        }
        return ParameterizedVocabularyMatch(
            entry=self.entries[candidate_id],
            parameters=parameters,
            matched_pattern=pattern.pattern,
            matched_text=joined,
        )

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

    def prompt_lines(
        self,
        *,
        metadata_keys: tuple[str, ...] = (),
    ) -> list[str]:
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
