from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CoverageStatus = Literal[
    "complete_against_explicit_expectation",
    "shortfall_against_explicit_expectation",
    "no_explicit_expectation",
]


class BenchmarkTaskMaterialization(StrictModel):
    schema_version: Literal[
        "scientific-reframe-benchmark-task-materialization-v1"
    ] = "scientific-reframe-benchmark-task-materialization-v1"

    task_key: str = Field(min_length=1)
    case_key: str = Field(min_length=1)
    replicate_key: str = Field(min_length=1)
    canonical_dir: str = Field(min_length=1)

    canonical_dir_exists: bool
    explorer_packet_present: bool
    explorer_report_present: bool
    hypothesis_context_present: bool
    audit_discoverable: bool
    fully_auditable: bool
    missing_required_artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_materialization(self) -> "BenchmarkTaskMaterialization":
        expected_missing: list[str] = []
        if not self.explorer_packet_present:
            expected_missing.append("explorer.packet.json")
        if not self.explorer_report_present:
            expected_missing.append("explorer.report.json")
        if not self.hypothesis_context_present:
            expected_missing.append("hypothesis.context.json")
        if self.missing_required_artifacts != expected_missing:
            raise ValueError("missing_required_artifacts is not canonical")
        if self.audit_discoverable != self.explorer_packet_present:
            raise ValueError(
                "0014 audit discovery is defined by explorer.packet.json presence"
            )
        if self.fully_auditable != (not expected_missing):
            raise ValueError("fully_auditable does not match required artifact matrix")
        return self


class ScientificReframeBenchmarkCoverageReport(StrictModel):
    schema_version: Literal[
        "scientific-reframe-benchmark-coverage-v1"
    ] = "scientific-reframe-benchmark-coverage-v1"

    coverage_id: str = Field(min_length=1)
    benchmark_root: str = Field(min_length=1)
    benchmark_name: str = Field(min_length=1)
    benchmark_name_numeric_tokens: list[int] = Field(default_factory=list)
    benchmark_name_numbers_are_expectations: Literal[False] = False

    top_level_directory_names: list[str] = Field(default_factory=list)
    top_level_case_like_directory_names: list[str] = Field(default_factory=list)
    replicate_directory_count: int = Field(ge=0)
    canonical_directory_count: int = Field(ge=0)
    task_materializations: list[BenchmarkTaskMaterialization]

    materialization_candidate_count: int = Field(ge=0)
    audit_discoverable_task_count: int = Field(ge=0)
    fully_auditable_task_count: int = Field(ge=0)
    incomplete_materialization_count: int = Field(ge=0)
    partially_materialized_canonical_count: int = Field(ge=0)
    missing_canonical_count: int = Field(ge=0)
    unique_materialized_case_count: int = Field(ge=0)

    prior_audit_present: bool = False
    prior_audit_path: str | None = None
    prior_audited_task_count: int = Field(ge=0, default=0)
    filesystem_discoverable_missing_from_prior_audit: list[str] = Field(
        default_factory=list
    )
    prior_audit_tasks_missing_from_filesystem: list[str] = Field(
        default_factory=list
    )

    expected_task_count: int | None = Field(default=None, ge=0)
    expected_task_count_source: Literal["cli", "none"] = "none"
    task_count_shortfall: int | None = Field(default=None, ge=0)
    coverage_status: CoverageStatus

    metadata_json_candidates: list[str] = Field(default_factory=list)

    llm_calls_performed: Literal[0] = 0
    deterministic_filesystem_audit: Literal[True] = True
    benchmark_name_not_used_as_count_authority: Literal[True] = True
    scientific_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeBenchmarkCoverageReport":
        rows = self.task_materializations
        keys = [row.task_key for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("task materialization keys must be unique")
        if self.materialization_candidate_count != len(rows):
            raise ValueError("materialization_candidate_count mismatch")
        if self.audit_discoverable_task_count != sum(
            row.audit_discoverable for row in rows
        ):
            raise ValueError("audit_discoverable_task_count mismatch")
        if self.fully_auditable_task_count != sum(row.fully_auditable for row in rows):
            raise ValueError("fully_auditable_task_count mismatch")
        incomplete = sum(not row.fully_auditable for row in rows)
        if self.incomplete_materialization_count != incomplete:
            raise ValueError("incomplete_materialization_count mismatch")
        partial = sum(
            row.canonical_dir_exists and not row.fully_auditable for row in rows
        )
        if self.partially_materialized_canonical_count != partial:
            raise ValueError("partially_materialized_canonical_count mismatch")
        missing_canonical = sum(not row.canonical_dir_exists for row in rows)
        if self.missing_canonical_count != missing_canonical:
            raise ValueError("missing_canonical_count mismatch")
        if self.expected_task_count is None:
            if self.expected_task_count_source != "none":
                raise ValueError("missing expected count requires source=none")
            if self.task_count_shortfall is not None:
                raise ValueError("shortfall is undefined without explicit expectation")
            if self.coverage_status != "no_explicit_expectation":
                raise ValueError("coverage status cannot claim completeness without expectation")
        else:
            if self.expected_task_count_source != "cli":
                raise ValueError("explicit expected count must come from cli")
            expected_shortfall = max(
                self.expected_task_count - self.audit_discoverable_task_count,
                0,
            )
            if self.task_count_shortfall != expected_shortfall:
                raise ValueError("task_count_shortfall mismatch")
            expected_status = (
                "complete_against_explicit_expectation"
                if expected_shortfall == 0
                else "shortfall_against_explicit_expectation"
            )
            if self.coverage_status != expected_status:
                raise ValueError("coverage_status mismatch")
        return self


def _numeric_tokens(name: str) -> list[int]:
    return [int(value) for value in re.findall(r"(?<![A-Za-z])\d+|\d+", name)]


def _case_like(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+\d+(?:[_-].*)?", name))


def _task_keys(benchmark_root: Path, canonical_dir: Path) -> tuple[str, str, str]:
    relative = canonical_dir.relative_to(benchmark_root)
    task_relative = relative.parent
    parts = task_relative.parts
    if not parts:
        raise ValueError(
            f"canonical directory is not nested under benchmark root: {canonical_dir}"
        )
    task_key = task_relative.as_posix()
    case_key = parts[0]
    replicate_key = parts[-1] if len(parts) > 1 else "default"
    return task_key, case_key, replicate_key


def _candidate_canonical_dirs(benchmark_root: Path) -> list[Path]:
    found: set[Path] = set()

    # Existing canonical directories matter even when explorer.packet.json is absent.
    for path in benchmark_root.rglob("canonical"):
        if path.is_dir():
            found.add(path.resolve())

    # If a known canonical artifact survives but the directory was reached through
    # an unusual layout, its parent is still a useful task materialization witness.
    for filename in (
        "explorer.packet.json",
        "explorer.report.json",
        "hypothesis.context.json",
    ):
        for path in benchmark_root.rglob(filename):
            if path.is_file() and path.parent.name == "canonical":
                found.add(path.parent.resolve())

    # Replicate directories without a canonical child are represented as an
    # expected local materialization location, but this is only a filesystem
    # diagnostic. It does not assert that every replicate_* must be a benchmark task.
    for replicate in benchmark_root.rglob("replicate_*"):
        if replicate.is_dir():
            found.add((replicate / "canonical").resolve())

    return sorted(found, key=lambda path: path.as_posix())


def _materialization(
    benchmark_root: Path,
    canonical_dir: Path,
) -> BenchmarkTaskMaterialization:
    task_key, case_key, replicate_key = _task_keys(benchmark_root, canonical_dir)
    packet = (canonical_dir / "explorer.packet.json").is_file()
    report = (canonical_dir / "explorer.report.json").is_file()
    context = (canonical_dir / "hypothesis.context.json").is_file()
    missing: list[str] = []
    if not packet:
        missing.append("explorer.packet.json")
    if not report:
        missing.append("explorer.report.json")
    if not context:
        missing.append("hypothesis.context.json")
    return BenchmarkTaskMaterialization(
        task_key=task_key,
        case_key=case_key,
        replicate_key=replicate_key,
        canonical_dir=str(canonical_dir),
        canonical_dir_exists=canonical_dir.is_dir(),
        explorer_packet_present=packet,
        explorer_report_present=report,
        hypothesis_context_present=context,
        audit_discoverable=packet,
        fully_auditable=not missing,
        missing_required_artifacts=missing,
    )


def _load_prior_audit(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    rows = payload.get("task_audits", [])
    if not isinstance(rows, list):
        raise ValueError("prior audit task_audits must be a list")
    return {
        str(row.get("task_key", "")).strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("task_key", "")).strip()
    }


def _metadata_candidates(benchmark_root: Path) -> list[str]:
    candidates: list[str] = []
    keywords = ("manifest", "benchmark", "case", "task", "summary", "index")
    for path in benchmark_root.glob("*.json"):
        lower = path.name.lower()
        if any(keyword in lower for keyword in keywords):
            candidates.append(path.name)
    return sorted(set(candidates))


def _coverage_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"scientific_reframe_benchmark_coverage:{digest}"


def inspect_benchmark_coverage(
    *,
    benchmark_root: str | Path,
    expected_task_count: int | None = None,
    prior_audit_path: str | Path | None = None,
) -> ScientificReframeBenchmarkCoverageReport:
    root = Path(benchmark_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark root does not exist: {root}")
    if expected_task_count is not None and expected_task_count < 0:
        raise ValueError("expected_task_count must be >= 0")

    top_level_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    case_like = sorted(name for name in top_level_dirs if _case_like(name))
    replicate_count = sum(
        1 for path in root.rglob("replicate_*") if path.is_dir()
    )
    canonical_dirs = _candidate_canonical_dirs(root)
    rows = [_materialization(root, path) for path in canonical_dirs]

    discoverable = {row.task_key for row in rows if row.audit_discoverable}
    prior_path = (
        Path(prior_audit_path).resolve()
        if prior_audit_path is not None
        else (root / "scientific_reframing_trigger_audit.json")
    )
    prior_present = prior_path.is_file()
    prior_tasks = _load_prior_audit(prior_path) if prior_present else set()

    if expected_task_count is None:
        shortfall = None
        status: CoverageStatus = "no_explicit_expectation"
        expected_source: Literal["cli", "none"] = "none"
    else:
        shortfall = max(expected_task_count - len(discoverable), 0)
        status = (
            "complete_against_explicit_expectation"
            if shortfall == 0
            else "shortfall_against_explicit_expectation"
        )
        expected_source = "cli"

    materialized_cases = {row.case_key for row in rows}
    body = {
        "benchmark_root": str(root),
        "benchmark_name": root.name,
        "benchmark_name_numeric_tokens": _numeric_tokens(root.name),
        "top_level_directory_names": top_level_dirs,
        "top_level_case_like_directory_names": case_like,
        "replicate_directory_count": replicate_count,
        "canonical_directory_count": sum(row.canonical_dir_exists for row in rows),
        "task_materializations": [row.model_dump(mode="json") for row in rows],
        "materialization_candidate_count": len(rows),
        "audit_discoverable_task_count": len(discoverable),
        "fully_auditable_task_count": sum(row.fully_auditable for row in rows),
        "incomplete_materialization_count": sum(
            not row.fully_auditable for row in rows
        ),
        "partially_materialized_canonical_count": sum(
            row.canonical_dir_exists and not row.fully_auditable for row in rows
        ),
        "missing_canonical_count": sum(
            not row.canonical_dir_exists for row in rows
        ),
        "unique_materialized_case_count": len(materialized_cases),
        "prior_audit_present": prior_present,
        "prior_audit_path": str(prior_path) if prior_present else None,
        "prior_audited_task_count": len(prior_tasks),
        "filesystem_discoverable_missing_from_prior_audit": sorted(
            discoverable - prior_tasks
        ),
        "prior_audit_tasks_missing_from_filesystem": sorted(
            prior_tasks - discoverable
        ),
        "expected_task_count": expected_task_count,
        "expected_task_count_source": expected_source,
        "task_count_shortfall": shortfall,
        "coverage_status": status,
        "metadata_json_candidates": _metadata_candidates(root),
    }
    return ScientificReframeBenchmarkCoverageReport(
        coverage_id=_coverage_id(body),
        **body,
    )
