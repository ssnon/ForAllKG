from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class QueryBucket:
    bucket_id: str
    label: str
    target: int
    queries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.bucket_id.strip():
            raise ValueError("bucket_id must not be empty")
        if not self.label.strip():
            raise ValueError(f"bucket {self.bucket_id!r} requires a label")
        if self.target <= 0:
            raise ValueError(f"bucket {self.bucket_id!r} target must be positive")
        if not self.queries:
            raise ValueError(f"bucket {self.bucket_id!r} requires at least one query")
        if any(not query.strip() for query in self.queries):
            raise ValueError(f"bucket {self.bucket_id!r} contains an empty query")


@dataclass(frozen=True)
class LiteratureQueryPlan:
    schema_version: str
    plan_id: str
    description: str
    buckets: tuple[QueryBucket, ...]

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")
        if not self.plan_id.strip():
            raise ValueError("plan_id must not be empty")
        if not self.buckets:
            raise ValueError("query plan requires at least one bucket")
        bucket_ids = [bucket.bucket_id for bucket in self.buckets]
        if len(bucket_ids) != len(set(bucket_ids)):
            raise ValueError("query plan contains duplicate bucket IDs")

    @property
    def target_paper_count(self) -> int:
        return sum(bucket.target for bucket in self.buckets)

    @property
    def query_count(self) -> int:
        return sum(len(bucket.queries) for bucket in self.buckets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "description": self.description,
            "target_paper_count": self.target_paper_count,
            "query_count": self.query_count,
            "buckets": [
                {
                    "bucket_id": bucket.bucket_id,
                    "label": bucket.label,
                    "target": bucket.target,
                    "queries": list(bucket.queries),
                }
                for bucket in self.buckets
            ],
        }


def load_query_plan(path: str | Path) -> LiteratureQueryPlan:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"query plan must be a mapping: {config_path}")

    raw_buckets = payload.get("buckets")
    if not isinstance(raw_buckets, dict) or not raw_buckets:
        raise ValueError("query plan requires a non-empty 'buckets' mapping")

    buckets: list[QueryBucket] = []
    for bucket_id, raw in raw_buckets.items():
        if not isinstance(raw, dict):
            raise ValueError(f"bucket {bucket_id!r} must be a mapping")
        raw_queries = raw.get("queries")
        if not isinstance(raw_queries, list):
            raise ValueError(f"bucket {bucket_id!r} queries must be a list")
        try:
            target = int(raw.get("target", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bucket {bucket_id!r} target must be an integer") from exc
        buckets.append(
            QueryBucket(
                bucket_id=str(bucket_id).strip(),
                label=str(raw.get("label") or bucket_id).strip(),
                target=target,
                queries=tuple(str(query).strip() for query in raw_queries),
            )
        )

    return LiteratureQueryPlan(
        schema_version=str(payload.get("schema_version") or "").strip(),
        plan_id=str(payload.get("plan_id") or "").strip(),
        description=str(payload.get("description") or "").strip(),
        buckets=tuple(buckets),
    )
