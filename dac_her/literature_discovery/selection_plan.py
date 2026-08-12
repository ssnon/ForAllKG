from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BucketSelectionRule:
    bucket_id: str
    keywords: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.bucket_id.strip():
            raise ValueError("bucket_id must not be empty")
        if not self.keywords:
            raise ValueError(f"bucket {self.bucket_id!r} requires keywords")
        if any(not item.strip() for item in self.keywords):
            raise ValueError(f"bucket {self.bucket_id!r} contains an empty keyword")


@dataclass(frozen=True)
class LiteratureSelectionPlan:
    schema_version: str
    plan_id: str
    query_plan_id: str
    target_count: int
    min_abstract_chars: int
    min_total_score: float
    allowed_languages: tuple[str, ...]
    global_context_terms: tuple[str, ...]
    global_mechanism_terms: tuple[str, ...]
    bucket_rules: tuple[BucketSelectionRule, ...]
    max_abstract_chars: int | None = None

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")
        if not self.plan_id.strip():
            raise ValueError("plan_id must not be empty")
        if not self.query_plan_id.strip():
            raise ValueError("query_plan_id must not be empty")
        if self.target_count <= 0:
            raise ValueError("target_count must be positive")
        if self.min_abstract_chars < 0:
            raise ValueError("min_abstract_chars must be non-negative")
        if self.min_total_score < 0:
            raise ValueError("min_total_score must be non-negative")
        if self.max_abstract_chars is not None:
            if self.max_abstract_chars <= 0:
                raise ValueError("max_abstract_chars must be positive when set")
            if self.max_abstract_chars < self.min_abstract_chars:
                raise ValueError("max_abstract_chars must be >= min_abstract_chars")
        ids = [item.bucket_id for item in self.bucket_rules]
        if len(ids) != len(set(ids)):
            raise ValueError("selection plan contains duplicate bucket rules")

    @property
    def bucket_rule_map(self) -> dict[str, BucketSelectionRule]:
        return {item.bucket_id: item for item in self.bucket_rules}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "query_plan_id": self.query_plan_id,
            "target_count": self.target_count,
            "min_abstract_chars": self.min_abstract_chars,
            "max_abstract_chars": self.max_abstract_chars,
            "min_total_score": self.min_total_score,
            "allowed_languages": list(self.allowed_languages),
            "global_context_terms": list(self.global_context_terms),
            "global_mechanism_terms": list(self.global_mechanism_terms),
            "bucket_rules": {
                item.bucket_id: {"keywords": list(item.keywords)}
                for item in self.bucket_rules
            },
        }


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    rows = tuple(str(item).strip() for item in value if str(item).strip())
    if not rows:
        raise ValueError(f"{field} must not be empty")
    return rows


def load_selection_plan(path: str | Path) -> LiteratureSelectionPlan:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"selection plan must be a mapping: {config_path}")

    raw_rules = payload.get("bucket_rules")
    if not isinstance(raw_rules, dict) or not raw_rules:
        raise ValueError("selection plan requires a non-empty bucket_rules mapping")

    rules: list[BucketSelectionRule] = []
    for bucket_id, raw in raw_rules.items():
        if not isinstance(raw, dict):
            raise ValueError(f"bucket rule {bucket_id!r} must be a mapping")
        rules.append(
            BucketSelectionRule(
                bucket_id=str(bucket_id).strip(),
                keywords=_string_tuple(
                    raw.get("keywords"),
                    field=f"bucket_rules.{bucket_id}.keywords",
                ),
            )
        )

    allowed = payload.get("allowed_languages", ["en"])
    if not isinstance(allowed, list):
        raise ValueError("allowed_languages must be a list")

    return LiteratureSelectionPlan(
        schema_version=str(payload.get("schema_version") or "").strip(),
        plan_id=str(payload.get("plan_id") or "").strip(),
        query_plan_id=str(payload.get("query_plan_id") or "").strip(),
        target_count=int(payload.get("target_count", 0)),
        min_abstract_chars=int(payload.get("min_abstract_chars", 0)),
        min_total_score=float(payload.get("min_total_score", 0.0)),
        max_abstract_chars=(
            int(payload["max_abstract_chars"])
            if payload.get("max_abstract_chars") not in (None, "")
            else None
        ),
        allowed_languages=tuple(str(item).strip().lower() for item in allowed if str(item).strip()),
        global_context_terms=_string_tuple(
            payload.get("global_context_terms"), field="global_context_terms"
        ),
        global_mechanism_terms=_string_tuple(
            payload.get("global_mechanism_terms"), field="global_mechanism_terms"
        ),
        bucket_rules=tuple(rules),
    )
