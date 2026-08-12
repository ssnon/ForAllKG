from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .contracts import LiteratureRecord
from .query_plan import LiteratureQueryPlan
from .relevance import CandidateAssessment, assess_candidate
from .selection_plan import LiteratureSelectionPlan


@dataclass(frozen=True)
class SelectedLiterature:
    record: LiteratureRecord
    assessment: CandidateAssessment
    assigned_bucket: str
    selection_mode: str

    def to_dict(self) -> dict:
        return {
            "paper": self.record.to_dict(),
            "selection": {
                **self.assessment.to_dict(),
                "assigned_bucket": self.assigned_bucket,
                "selection_mode": self.selection_mode,
            },
        }


@dataclass(frozen=True)
class RejectedLiterature:
    record: LiteratureRecord
    assessment: CandidateAssessment
    rejection_reason: str

    def to_dict(self) -> dict:
        return {
            "paper": self.record.to_dict(),
            "selection": {
                **self.assessment.to_dict(),
                "rejection_reason": self.rejection_reason,
            },
        }


@dataclass(frozen=True)
class LiteratureSelectionResult:
    selected: tuple[SelectedLiterature, ...]
    rejected: tuple[RejectedLiterature, ...]
    quotas: dict[str, int]
    target_count: int

    @property
    def selected_count(self) -> int:
        return len(self.selected)


def scaled_bucket_quotas(
    query_plan: LiteratureQueryPlan,
    *,
    target_count: int,
) -> dict[str, int]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    total_weight = sum(bucket.target for bucket in query_plan.buckets)
    if total_weight <= 0:
        raise ValueError("query plan targets must sum to a positive value")

    raw = {
        bucket.bucket_id: target_count * bucket.target / total_weight
        for bucket in query_plan.buckets
    }
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = target_count - sum(quotas.values())
    order = sorted(
        query_plan.buckets,
        key=lambda bucket: (
            -(raw[bucket.bucket_id] - quotas[bucket.bucket_id]),
            list(query_plan.buckets).index(bucket),
        ),
    )
    for bucket in order[:remaining]:
        quotas[bucket.bucket_id] += 1
    return quotas


def select_literature(
    records: Iterable[LiteratureRecord],
    *,
    query_plan: LiteratureQueryPlan,
    selection_plan: LiteratureSelectionPlan,
) -> LiteratureSelectionResult:
    if selection_plan.query_plan_id != query_plan.plan_id:
        raise ValueError(
            "selection/query plan mismatch: "
            f"{selection_plan.query_plan_id!r} != {query_plan.plan_id!r}"
        )

    unique: dict[str, LiteratureRecord] = {}
    for record in records:
        if record.paper_id in unique:
            raise ValueError(f"candidate input contains duplicate paper_id: {record.paper_id}")
        unique[record.paper_id] = record

    assessments = {
        paper_id: assess_candidate(record, selection_plan)
        for paper_id, record in unique.items()
    }
    quotas = scaled_bucket_quotas(
        query_plan,
        target_count=selection_plan.target_count,
    )

    selected: dict[str, SelectedLiterature] = {}

    def bucket_rank(record: LiteratureRecord, bucket_id: str) -> tuple:
        assessment = assessments[record.paper_id]
        return (
            -assessment.bucket_scores.get(bucket_id, 0.0),
            -assessment.total_score,
            record.paper_id,
        )

    eligible = [
        record for record in unique.values() if assessments[record.paper_id].eligible
    ]

    for bucket in query_plan.buckets:
        bucket_id = bucket.bucket_id
        candidates = [
            record
            for record in eligible
            if record.paper_id not in selected
            and assessments[record.paper_id].bucket_scores.get(bucket_id, 0.0) > 0.0
        ]
        candidates.sort(key=lambda record: bucket_rank(record, bucket_id))
        for record in candidates[: quotas.get(bucket_id, 0)]:
            selected[record.paper_id] = SelectedLiterature(
                record=record,
                assessment=assessments[record.paper_id],
                assigned_bucket=bucket_id,
                selection_mode="bucket_quota",
            )

    remaining_slots = selection_plan.target_count - len(selected)
    if remaining_slots > 0:
        fallback = [record for record in eligible if record.paper_id not in selected]
        fallback.sort(
            key=lambda record: (
                -assessments[record.paper_id].total_score,
                -max(assessments[record.paper_id].bucket_scores.values(), default=0.0),
                record.paper_id,
            )
        )
        for record in fallback[:remaining_slots]:
            assessment = assessments[record.paper_id]
            assigned_bucket = assessment.best_bucket or query_plan.buckets[0].bucket_id
            selected[record.paper_id] = SelectedLiterature(
                record=record,
                assessment=assessment,
                assigned_bucket=assigned_bucket,
                selection_mode="global_fallback",
            )

    rejected: list[RejectedLiterature] = []
    for paper_id, record in sorted(unique.items()):
        if paper_id in selected:
            continue
        assessment = assessments[paper_id]
        reason = (
            "hard_or_score_exclusion"
            if not assessment.eligible
            else "selection_capacity"
        )
        rejected.append(
            RejectedLiterature(
                record=record,
                assessment=assessment,
                rejection_reason=reason,
            )
        )

    selected_rows = tuple(sorted(selected.values(), key=lambda item: item.record.paper_id))
    return LiteratureSelectionResult(
        selected=selected_rows,
        rejected=tuple(rejected),
        quotas=quotas,
        target_count=selection_plan.target_count,
    )


def read_candidates_jsonl(path: str | Path) -> list[LiteratureRecord]:
    rows: list[LiteratureRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(LiteratureRecord.from_dict(payload))
    return rows


def write_selection_artifacts(
    result: LiteratureSelectionResult,
    *,
    output_dir: str | Path,
    candidates_path: str | Path,
    query_plan: LiteratureQueryPlan,
    selection_plan: LiteratureSelectionPlan,
) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected.jsonl"
    rejected_path = output / "rejected.jsonl"
    report_path = output / "selection_report.json"

    _write_jsonl(selected_path, (item.to_dict() for item in result.selected))
    _write_jsonl(rejected_path, (item.to_dict() for item in result.rejected))

    selected_buckets = Counter(item.assigned_bucket for item in result.selected)
    selection_modes = Counter(item.selection_mode for item in result.selected)
    exclusion_reasons = Counter(
        reason
        for item in result.rejected
        for reason in item.assessment.exclusion_reasons
    )
    eligible_not_selected = sum(
        item.assessment.eligible for item in result.rejected
    )
    report = {
        "schema_version": "graphagentsdac-literature-selection-run-v01",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_plan_id": query_plan.plan_id,
        "selection_plan_id": selection_plan.plan_id,
        "query_plan": query_plan.to_dict(),
        "selection_plan": selection_plan.to_dict(),
        "target_count": result.target_count,
        "input_candidate_count": result.selected_count + len(result.rejected),
        "selected_count": result.selected_count,
        "selection_deficit": max(0, result.target_count - result.selected_count),
        "rejected_count": len(result.rejected),
        "eligible_not_selected_count": eligible_not_selected,
        "bucket_quotas": result.quotas,
        "selected_by_bucket": dict(sorted(selected_buckets.items())),
        "selected_by_mode": dict(sorted(selection_modes.items())),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "candidates_path": str(Path(candidates_path)),
        "candidates_sha256": _sha256_file(Path(candidates_path)),
    }
    _write_json(report_path, report)
    return selected_path, rejected_path, report_path


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
