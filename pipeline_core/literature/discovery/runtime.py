from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .contracts import LiteratureRecord, merge_literature_records
from .providers.base import LiteratureProvider, LiteratureSearchRequest
from .query_plan import LiteratureQueryPlan
from .registry import LiteratureRegistry


@dataclass(frozen=True)
class DiscoveryRunArtifacts:
    output_dir: Path
    candidates_path: Path
    run_path: Path
    unique_candidates: int
    raw_candidates: int


def select_pilot_requests(
    plan: LiteratureQueryPlan,
    *,
    query_count: int,
    per_query_limit: int,
) -> list[LiteratureSearchRequest]:
    """Round-robin buckets so a small pilot covers multiple mechanism families."""
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    if per_query_limit <= 0:
        raise ValueError("per_query_limit must be positive")

    queues = [list(bucket.queries) for bucket in plan.buckets]
    requests: list[LiteratureSearchRequest] = []
    round_index = 0

    while len(requests) < query_count:
        added = False
        for bucket_index, bucket in enumerate(plan.buckets):
            if len(requests) >= query_count:
                break
            if round_index >= len(queues[bucket_index]):
                continue
            requests.append(
                LiteratureSearchRequest(
                    query=queues[bucket_index][round_index],
                    mechanism_bucket=bucket.bucket_id,
                    limit=per_query_limit,
                )
            )
            added = True
        if not added:
            break
        round_index += 1

    return requests


def all_query_requests(
    plan: LiteratureQueryPlan,
    *,
    per_query_limit: int,
) -> list[LiteratureSearchRequest]:
    if per_query_limit <= 0:
        raise ValueError("per_query_limit must be positive")
    return [
        LiteratureSearchRequest(
            query=query,
            mechanism_bucket=bucket.bucket_id,
            limit=per_query_limit,
        )
        for bucket in plan.buckets
        for query in bucket.queries
    ]


def run_discovery(
    *,
    provider: LiteratureProvider,
    plan: LiteratureQueryPlan,
    requests: Iterable[LiteratureSearchRequest],
    registry: LiteratureRegistry,
    output_dir: str | Path,
) -> DiscoveryRunArtifacts:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidates.jsonl"
    run_path = output / "run.json"

    started_at = datetime.now(timezone.utc)
    selected_requests = list(requests)
    run_candidates: dict[str, LiteratureRecord] = {}
    query_rows: list[dict] = []
    raw_candidate_count = 0

    try:
        for request in selected_requests:
            records = provider.search(request)
            raw_candidate_count += len(records)
            abstract_count = sum(record.abstract is not None for record in records)
            before = len(run_candidates)
            for record in records:
                existing = run_candidates.get(record.paper_id)
                run_candidates[record.paper_id] = (
                    record
                    if existing is None
                    else merge_literature_records(existing, record)
                )
                registry.upsert(record)
            query_rows.append(
                {
                    "mechanism_bucket": request.mechanism_bucket,
                    "query": request.query,
                    "requested_limit": request.limit,
                    "returned_count": len(records),
                    "abstract_count": abstract_count,
                    "new_unique_count": len(run_candidates) - before,
                }
            )

        registry.save()
        _write_candidates(candidates_path, run_candidates.values())
        completed_at = datetime.now(timezone.utc)
        run_payload = _run_payload(
            status="complete",
            provider=provider.provider_name,
            plan=plan,
            started_at=started_at,
            completed_at=completed_at,
            query_rows=query_rows,
            raw_candidate_count=raw_candidate_count,
            unique_candidate_count=len(run_candidates),
            candidates_path=candidates_path,
            registry_path=registry.path,
        )
        _write_json(run_path, run_payload)
    except Exception as exc:
        completed_at = datetime.now(timezone.utc)
        run_payload = _run_payload(
            status="failed",
            provider=provider.provider_name,
            plan=plan,
            started_at=started_at,
            completed_at=completed_at,
            query_rows=query_rows,
            raw_candidate_count=raw_candidate_count,
            unique_candidate_count=len(run_candidates),
            candidates_path=candidates_path,
            registry_path=registry.path,
        )
        run_payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(run_path, run_payload)
        raise

    return DiscoveryRunArtifacts(
        output_dir=output,
        candidates_path=candidates_path,
        run_path=run_path,
        unique_candidates=len(run_candidates),
        raw_candidates=raw_candidate_count,
    )


def _write_candidates(path: Path, records: Iterable[LiteratureRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.paper_id):
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _run_payload(
    *,
    status: str,
    provider: str,
    plan: LiteratureQueryPlan,
    started_at: datetime,
    completed_at: datetime,
    query_rows: list[dict],
    raw_candidate_count: int,
    unique_candidate_count: int,
    candidates_path: Path,
    registry_path: Path,
) -> dict:
    abstract_total = sum(row.get("abstract_count", 0) for row in query_rows)
    return {
        "schema_version": "graphagentsdac-literature-discovery-run-v01",
        "status": status,
        "provider": provider,
        "plan_id": plan.plan_id,
        "query_plan_schema_version": plan.schema_version,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "query_count": len(query_rows),
        "raw_candidate_count": raw_candidate_count,
        "unique_candidate_count": unique_candidate_count,
        "raw_abstract_count": abstract_total,
        "duplicate_or_repeat_count": raw_candidate_count - unique_candidate_count,
        "queries": query_rows,
        "candidates_path": str(candidates_path),
        "registry_path": str(registry_path),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
