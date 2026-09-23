from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ValidationPreflightStatus = Literal[
    "READY",
    "EXCLUDED_BY_CALLER",
    "INCOMPLETE_INPUTS",
    "LINEAGE_MISMATCH",
    "EMPTY_ATOMIC_HYPOTHESES",
    "EXTERNAL_NOVELTY_NOT_RESOLVED",
]


_SOURCE_CANDIDATE_PORTFOLIO = "scientific_pre_n10_candidate_portfolio.json"
_ATOMIC_REPORT = "scientific_atomic_cross_lane_v2.report.json"
_ATOMIC_HYPOTHESIS_PORTFOLIO = "scientific_atomic_cross_lane_v2.portfolio.json"
_GROUNDED_ANNOTATION = "scientific_atomic_grounded_identity_v2.annotation.json"
_PREFERRED_EXTERNAL_REPORT = "scientific_atomic_n10_external.report.json"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict_payload(path: Path) -> dict[str, object] | None:
    try:
        payload = _load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _hypothesis_ids_from_portfolio(payload: dict[str, object]) -> set[str]:
    rows = payload.get("hypotheses", [])
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get("hypothesis_id"))
        for row in rows
        if isinstance(row, dict) and row.get("hypothesis_id")
    }


def _hypothesis_ids_from_atomic_report(payload: dict[str, object]) -> set[str]:
    rows = payload.get("hypotheses", [])
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get("hypothesis_id"))
        for row in rows
        if isinstance(row, dict) and row.get("hypothesis_id")
    }


def _external_card_ids(payload: dict[str, object]) -> set[str]:
    rows = payload.get("cards", [])
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get("hypothesis_id"))
        for row in rows
        if isinstance(row, dict) and row.get("hypothesis_id")
    }


def _discover_external_reports(
    *,
    canonical_dir: Path,
    source_portfolio_id: str,
    target_hypothesis_ids: set[str],
) -> tuple[list[Path], Path | None]:
    matches: list[Path] = []
    for path in canonical_dir.rglob("*.json"):
        payload = _dict_payload(path)
        if payload is None:
            continue
        if payload.get("schema_version") != "external-novelty-report-v1":
            continue
        if str(payload.get("source_portfolio_id") or "") != source_portfolio_id:
            continue
        if _external_card_ids(payload) != target_hypothesis_ids:
            continue
        matches.append(path)

    matches = sorted(set(path.resolve() for path in matches))
    preferred = (canonical_dir / _PREFERRED_EXTERNAL_REPORT).resolve()
    if preferred in matches:
        return matches, preferred
    if len(matches) == 1:
        return matches, matches[0]
    return matches, None


class ScientificVerifierValidationTaskPreflight(StrictModel):
    task_key: str
    case_key: str
    replicate_key: str
    canonical_dir: str

    status: ValidationPreflightStatus
    ready_for_full_shadow_validation: bool

    caller_declared_excluded: bool
    exclusion_reason: str | None = None

    domain_profile_id: str | None = None
    atomic_hypothesis_count: int = Field(ge=0)
    atomic_hypothesis_ids: list[str] = Field(default_factory=list)

    source_candidate_portfolio_path: str | None = None
    atomic_report_path: str | None = None
    atomic_hypothesis_portfolio_path: str | None = None
    grounded_identity_annotation_path: str | None = None

    source_candidate_lineage_match: bool | None = None
    atomic_hypothesis_lineage_match: bool | None = None
    grounded_annotation_lineage_match: bool | None = None

    matching_external_novelty_report_count: int = Field(default=0, ge=0)
    matching_external_novelty_report_paths: list[str] = Field(default_factory=list)
    selected_external_novelty_report_path: str | None = None

    missing_required_files: list[str] = Field(default_factory=list)
    lineage_problem_codes: list[str] = Field(default_factory=list)
    diagnostic_notes: list[str] = Field(default_factory=list)

    validation_outcome_assessed: Literal[False] = False
    scientific_quality_judgment_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False


class ScientificVerifierUnseenValidationPreflight(StrictModel):
    schema_version: Literal[
        "scientific-verifier-unseen-validation-preflight-v1"
    ] = "scientific-verifier-unseen-validation-preflight-v1"

    benchmark_root: str
    caller_declared_excluded_case_keys: list[str]
    task_count: int = Field(ge=0)
    ready_task_count: int = Field(ge=0)
    excluded_task_count: int = Field(ge=0)
    blocked_task_count: int = Field(ge=0)
    status_counts: dict[str, int]
    tasks: list[ScientificVerifierValidationTaskPreflight]

    unseen_status_is_caller_declared_only: Literal[True] = True
    contamination_audit_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    retrieval_performed: Literal[False] = False
    validation_outcome_assessed: Literal[False] = False
    scientific_quality_judgment_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificVerifierUnseenValidationPreflight":
        if self.task_count != len(self.tasks):
            raise ValueError("task_count mismatch")
        if self.ready_task_count != sum(row.ready_for_full_shadow_validation for row in self.tasks):
            raise ValueError("ready_task_count mismatch")
        if self.excluded_task_count != sum(row.caller_declared_excluded for row in self.tasks):
            raise ValueError("excluded_task_count mismatch")
        if self.blocked_task_count != (
            self.task_count - self.ready_task_count - self.excluded_task_count
        ):
            raise ValueError("blocked_task_count mismatch")
        observed: dict[str, int] = {}
        for row in self.tasks:
            observed[row.status] = observed.get(row.status, 0) + 1
        if dict(sorted(observed.items())) != dict(sorted(self.status_counts.items())):
            raise ValueError("status_counts mismatch")
        return self


def inspect_validation_task(
    *,
    canonical_dir: Path,
    case_key: str,
    replicate_key: str,
    excluded_case_keys: set[str],
) -> ScientificVerifierValidationTaskPreflight:
    canonical_dir = canonical_dir.resolve()
    task_key = f"{case_key}/{replicate_key}"

    if case_key in excluded_case_keys:
        return ScientificVerifierValidationTaskPreflight(
            task_key=task_key,
            case_key=case_key,
            replicate_key=replicate_key,
            canonical_dir=str(canonical_dir),
            status="EXCLUDED_BY_CALLER",
            ready_for_full_shadow_validation=False,
            caller_declared_excluded=True,
            exclusion_reason="caller_declared_calibration_or_development_exclusion",
            atomic_hypothesis_count=0,
        )

    required = {
        "source_candidate_portfolio": canonical_dir / _SOURCE_CANDIDATE_PORTFOLIO,
        "atomic_report": canonical_dir / _ATOMIC_REPORT,
        "atomic_hypothesis_portfolio": canonical_dir / _ATOMIC_HYPOTHESIS_PORTFOLIO,
        "grounded_identity_annotation": canonical_dir / _GROUNDED_ANNOTATION,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return ScientificVerifierValidationTaskPreflight(
            task_key=task_key,
            case_key=case_key,
            replicate_key=replicate_key,
            canonical_dir=str(canonical_dir),
            status="INCOMPLETE_INPUTS",
            ready_for_full_shadow_validation=False,
            caller_declared_excluded=False,
            atomic_hypothesis_count=0,
            missing_required_files=missing,
            diagnostic_notes=[
                "preflight_does_not_materialize_missing_shadow_inputs"
            ],
        )

    source_candidates = _dict_payload(required["source_candidate_portfolio"])
    atomic_report = _dict_payload(required["atomic_report"])
    atomic_portfolio = _dict_payload(required["atomic_hypothesis_portfolio"])
    annotation = _dict_payload(required["grounded_identity_annotation"])

    malformed = [
        name
        for name, payload in (
            ("source_candidate_portfolio", source_candidates),
            ("atomic_report", atomic_report),
            ("atomic_hypothesis_portfolio", atomic_portfolio),
            ("grounded_identity_annotation", annotation),
        )
        if payload is None
    ]
    if malformed:
        return ScientificVerifierValidationTaskPreflight(
            task_key=task_key,
            case_key=case_key,
            replicate_key=replicate_key,
            canonical_dir=str(canonical_dir),
            status="INCOMPLETE_INPUTS",
            ready_for_full_shadow_validation=False,
            caller_declared_excluded=False,
            atomic_hypothesis_count=0,
            missing_required_files=[f"malformed_json:{name}" for name in malformed],
        )

    assert source_candidates is not None
    assert atomic_report is not None
    assert atomic_portfolio is not None
    assert annotation is not None

    atomic_ids = _hypothesis_ids_from_atomic_report(atomic_report)
    portfolio_ids = _hypothesis_ids_from_portfolio(atomic_portfolio)
    atomic_count = len(atomic_ids)
    domain_profile_id = str(atomic_portfolio.get("domain_profile_id") or "") or None

    if not atomic_ids:
        return ScientificVerifierValidationTaskPreflight(
            task_key=task_key,
            case_key=case_key,
            replicate_key=replicate_key,
            canonical_dir=str(canonical_dir),
            status="EMPTY_ATOMIC_HYPOTHESES",
            ready_for_full_shadow_validation=False,
            caller_declared_excluded=False,
            domain_profile_id=domain_profile_id,
            atomic_hypothesis_count=0,
            source_candidate_portfolio_path=str(required["source_candidate_portfolio"]),
            atomic_report_path=str(required["atomic_report"]),
            atomic_hypothesis_portfolio_path=str(required["atomic_hypothesis_portfolio"]),
            grounded_identity_annotation_path=str(required["grounded_identity_annotation"]),
            diagnostic_notes=["atomic_report_contains_no_hypotheses"],
        )

    source_candidate_match = (
        str(source_candidates.get("portfolio_id") or "")
        == str(atomic_report.get("source_candidate_portfolio_id") or "")
    )
    atomic_hypothesis_match = atomic_ids == portfolio_ids
    annotation_match = (
        str(annotation.get("source_atomic_synthesis_report_id") or "")
        == str(atomic_report.get("report_id") or "")
    ) and (
        str(annotation.get("source_candidate_portfolio_id") or "")
        == str(source_candidates.get("portfolio_id") or "")
    )

    lineage_problems: list[str] = []
    if not source_candidate_match:
        lineage_problems.append("source_candidate_portfolio_id_mismatch")
    if not atomic_hypothesis_match:
        lineage_problems.append("atomic_report_hypothesis_set_mismatch")
    if not annotation_match:
        lineage_problems.append("grounded_identity_annotation_lineage_mismatch")

    source_portfolio_id = str(atomic_portfolio.get("portfolio_id") or "")
    matching_external, selected_external = _discover_external_reports(
        canonical_dir=canonical_dir,
        source_portfolio_id=source_portfolio_id,
        target_hypothesis_ids=portfolio_ids,
    )

    if lineage_problems:
        status: ValidationPreflightStatus = "LINEAGE_MISMATCH"
        ready = False
    elif selected_external is None:
        status = "EXTERNAL_NOVELTY_NOT_RESOLVED"
        ready = False
    else:
        status = "READY"
        ready = True

    notes: list[str] = []
    if len(matching_external) > 1 and selected_external is not None:
        notes.append("preferred_canonical_external_report_selected_among_multiple_full_matches")
    elif len(matching_external) > 1:
        notes.append("multiple_full_match_external_reports_without_preferred_canonical_path")
    elif not matching_external:
        notes.append("no_external_report_matches_source_portfolio_and_exact_hypothesis_set")

    return ScientificVerifierValidationTaskPreflight(
        task_key=task_key,
        case_key=case_key,
        replicate_key=replicate_key,
        canonical_dir=str(canonical_dir),
        status=status,
        ready_for_full_shadow_validation=ready,
        caller_declared_excluded=False,
        domain_profile_id=domain_profile_id,
        atomic_hypothesis_count=atomic_count,
        atomic_hypothesis_ids=sorted(atomic_ids),
        source_candidate_portfolio_path=str(required["source_candidate_portfolio"]),
        atomic_report_path=str(required["atomic_report"]),
        atomic_hypothesis_portfolio_path=str(required["atomic_hypothesis_portfolio"]),
        grounded_identity_annotation_path=str(required["grounded_identity_annotation"]),
        source_candidate_lineage_match=source_candidate_match,
        atomic_hypothesis_lineage_match=atomic_hypothesis_match,
        grounded_annotation_lineage_match=annotation_match,
        matching_external_novelty_report_count=len(matching_external),
        matching_external_novelty_report_paths=[str(path) for path in matching_external],
        selected_external_novelty_report_path=(
            str(selected_external) if selected_external is not None else None
        ),
        lineage_problem_codes=lineage_problems,
        diagnostic_notes=notes,
    )


def build_unseen_validation_preflight(
    *,
    benchmark_root: Path,
    excluded_case_keys: set[str] | None = None,
    included_case_keys: set[str] | None = None,
) -> ScientificVerifierUnseenValidationPreflight:
    benchmark_root = benchmark_root.expanduser().resolve()
    excluded = set(excluded_case_keys or set())
    included = set(included_case_keys or set())

    tasks: list[ScientificVerifierValidationTaskPreflight] = []
    if benchmark_root.is_dir():
        for case_dir in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
            case_key = case_dir.name
            if included and case_key not in included:
                continue
            replicate_dirs = sorted(
                path for path in case_dir.iterdir()
                if path.is_dir() and path.name.startswith("replicate_")
            )
            for replicate_dir in replicate_dirs:
                canonical_dir = replicate_dir / "canonical"
                if not canonical_dir.is_dir():
                    continue
                tasks.append(
                    inspect_validation_task(
                        canonical_dir=canonical_dir,
                        case_key=case_key,
                        replicate_key=replicate_dir.name,
                        excluded_case_keys=excluded,
                    )
                )

    status_counts: dict[str, int] = {}
    for row in tasks:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    ready = sum(row.ready_for_full_shadow_validation for row in tasks)
    excluded_count = sum(row.caller_declared_excluded for row in tasks)

    return ScientificVerifierUnseenValidationPreflight(
        benchmark_root=str(benchmark_root),
        caller_declared_excluded_case_keys=sorted(excluded),
        task_count=len(tasks),
        ready_task_count=ready,
        excluded_task_count=excluded_count,
        blocked_task_count=len(tasks) - ready - excluded_count,
        status_counts=dict(sorted(status_counts.items())),
        tasks=tasks,
    )


__all__ = [
    "ScientificVerifierUnseenValidationPreflight",
    "ScientificVerifierValidationTaskPreflight",
    "build_unseen_validation_preflight",
    "inspect_validation_task",
]
