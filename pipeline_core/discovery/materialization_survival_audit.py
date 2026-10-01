from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AuditedArm = Literal[
    "FRONTIER_BALANCED",
    "EVOLUTION_BALANCED",
    "PORTFOLIO_SELECTED",
]


AUDITED_ARMS: tuple[AuditedArm, ...] = (
    "FRONTIER_BALANCED",
    "EVOLUTION_BALANCED",
    "PORTFOLIO_SELECTED",
)


class MaterializationTransitionCell(StrictModel):
    arm: AuditedArm
    origin: str
    idea_form: str
    operator_or_form: str
    selection_profile: str
    selected_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    verification_ready_count: int = Field(ge=0)
    verification_ready_fraction: float = Field(ge=0.0, le=1.0)
    downstream_complete_count: int = Field(ge=0)
    issue_code_counts: dict[str, int] = Field(default_factory=dict)


class MaterializationIssueTaxonomyEntry(StrictModel):
    issue_code: str
    count: int = Field(ge=1)
    status_counts: dict[str, int] = Field(default_factory=dict)
    arm_counts: dict[str, int] = Field(default_factory=dict)
    origin_counts: dict[str, int] = Field(default_factory=dict)
    operator_or_form_counts: dict[str, int] = Field(default_factory=dict)
    profile_counts: dict[str, int] = Field(default_factory=dict)


class MaterializationCaseArmSurvivalCell(StrictModel):
    arm: AuditedArm
    selected_candidate_count: int = Field(ge=0)
    verification_ready_count: int = Field(ge=0)
    verification_ready_fraction: float = Field(ge=0.0, le=1.0)
    materialization_observed: bool
    lineage_resolution_complete: bool
    terminal_status_counts: dict[str, int] = Field(default_factory=dict)


class MaterializationCaseSurvivalRow(StrictModel):
    case_id: str
    artifact_observed: bool = True
    execution_observed: bool = False
    execution_status: str | None = None
    arms: list[MaterializationCaseArmSurvivalCell] = Field(default_factory=list)


class MaterializationPairedComparison(StrictModel):
    left_arm: AuditedArm
    right_arm: AuditedArm
    case_count: int = Field(ge=0)
    case_ids: list[str] = Field(default_factory=list)
    median_left_yield: float | None = None
    median_right_yield: float | None = None
    median_delta_right_minus_left: float | None = None
    right_higher_case_count: int = Field(ge=0)
    equal_case_count: int = Field(ge=0)
    right_lower_case_count: int = Field(ge=0)


class MaterializationSurvivalArmAudit(StrictModel):
    schema_version: Literal[
        "materialization-survival-arm-audit-v1.1"
    ] = "materialization-survival-arm-audit-v1.1"
    case_id: str
    arm: AuditedArm
    candidate_pool_path: str | None = None
    selection_path: str | None = None
    materialization_report_path: str | None = None
    generation_path: str | None = None
    verification_summary_path: str | None = None

    selected_candidate_count: int = Field(ge=0)
    selected_unique_family_count: int = Field(ge=0)
    materialization_observed: bool
    terminal_status_counts: dict[str, int] = Field(default_factory=dict)
    verification_ready_count: int = Field(ge=0)
    verification_ready_fraction: float = Field(ge=0.0, le=1.0)

    downstream_operational_status: str
    downstream_complete: bool
    downstream_complete_materialized_count: int = Field(ge=0)

    selected_count_by_origin: dict[str, int] = Field(default_factory=dict)
    selected_count_by_form: dict[str, int] = Field(default_factory=dict)
    selected_count_by_operator_or_form: dict[str, int] = Field(default_factory=dict)
    selected_count_by_profile: dict[str, int] = Field(default_factory=dict)
    transition_cells: list[MaterializationTransitionCell] = Field(default_factory=list)
    issue_taxonomy: list[MaterializationIssueTaxonomyEntry] = Field(default_factory=list)
    raw_issue_code_counts: dict[str, int] = Field(default_factory=dict)
    generation_error_counts: dict[str, int] = Field(default_factory=dict)
    unresolved_candidate_ids: list[str] = Field(default_factory=list)
    lineage_resolution_complete: bool = True
    non_materialized_count: int = Field(ge=0)
    taxonomy_classified_failure_count: int = Field(ge=0)
    taxonomy_unclassified_failure_count: int = Field(ge=0)
    taxonomy_exhaustive: bool = True

    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False
    materialization_changed: Literal[False] = False
    downstream_verification_changed: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class MaterializationSurvivalCaseAudit(StrictModel):
    schema_version: Literal[
        "materialization-survival-case-audit-v1.1"
    ] = "materialization-survival-case-audit-v1.1"
    audit_id: str
    audit_sha256: str
    case_id: str
    run_dir: str | None = None
    prospective_case_dir: str
    observed_arm_count: int = Field(ge=0)
    arms: list[MaterializationSurvivalArmAudit] = Field(default_factory=list)
    partial_case: bool
    execution_observed: bool = False
    execution_status: str | None = None
    lineage_resolution_complete: bool = True

    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False
    materialization_changed: Literal[False] = False
    downstream_verification_changed: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class MaterializationSurvivalCohortAudit(StrictModel):
    schema_version: Literal[
        "materialization-survival-cohort-audit-v1.1"
    ] = "materialization-survival-cohort-audit-v1.1"
    audit_id: str
    audit_sha256: str
    prospective_output_root: str
    planned_case_count: int = Field(ge=0)
    observed_case_count: int = Field(ge=0)
    artifact_observed_case_count: int = Field(ge=0)
    execution_observed_case_count: int = Field(ge=0)
    artifact_only_case_count: int = Field(ge=0)
    partial_case_count: int = Field(ge=0)
    cases: list[MaterializationSurvivalCaseAudit] = Field(default_factory=list)
    per_case_survival_matrix: list[MaterializationCaseSurvivalRow] = Field(default_factory=list)
    common_materialization_case_count: int = Field(ge=0)
    common_materialization_case_ids: list[str] = Field(default_factory=list)
    paired_comparisons: list[MaterializationPairedComparison] = Field(default_factory=list)

    arm_selected_candidate_counts: dict[str, int] = Field(default_factory=dict)
    arm_verification_ready_counts: dict[str, int] = Field(default_factory=dict)
    arm_verification_ready_fraction_medians: dict[str, float | None] = Field(
        default_factory=dict
    )
    arm_downstream_complete_case_counts: dict[str, int] = Field(default_factory=dict)
    arm_terminal_status_counts: dict[str, dict[str, int]] = Field(default_factory=dict)

    transition_cells: list[MaterializationTransitionCell] = Field(default_factory=list)
    issue_taxonomy: list[MaterializationIssueTaxonomyEntry] = Field(default_factory=list)
    raw_issue_code_counts: dict[str, int] = Field(default_factory=dict)
    generation_error_counts: dict[str, int] = Field(default_factory=dict)
    non_materialized_count: int = Field(ge=0)
    taxonomy_classified_failure_count: int = Field(ge=0)
    taxonomy_unclassified_failure_count: int = Field(ge=0)
    taxonomy_exhaustive: bool = True

    partial_run_compatible: Literal[True] = True
    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False
    materialization_changed: Literal[False] = False
    downstream_verification_changed: Literal[False] = False
    scientific_superiority_established: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(k), int(v)) for k, v in counter.items() if int(v) > 0))


def _median_optional(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _effective_status(
    record: Mapping[str, Any] | None,
    *,
    generation_error: Mapping[str, Any] | None,
) -> str:
    if record is None:
        return "MISSING_RECORD"

    status = str(record.get("status") or "UNKNOWN").strip() or "UNKNOWN"
    if status == "GENERATION_FAILED":
        return status

    issue_codes = [
        str(value or "").upper()
        for value in (record.get("issue_codes") or [])
    ]
    issues = "\n".join(
        str(value or "")
        for value in (record.get("issues") or [])
    ).lower()

    explicit_generation_failure = (
        generation_error is not None
        or any(
            "GENERATION" in code
            or "STRUCTURED_CALL" in code
            or "INSTRUCTOR" in code
            for code in issue_codes
        )
        or "structured call failed" in issues
        or "instructorretryexception" in issues
    )
    if status in {"ABSTAINED", "MISSING_DRAFT"} and explicit_generation_failure:
        return "GENERATION_FAILED"
    return status


def _selection_source_paths(
    *,
    case_dir: Path,
    run_dir: Path | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    rebuilt = case_dir / "PORTFOLIO_SELECTED" / "selection_source"
    if (rebuilt / "candidate_pool.json").is_file():
        return (
            rebuilt / "candidate_pool.json",
            rebuilt / "selection.json",
            rebuilt / "materialization.report.json",
            rebuilt / "generation.json",
        )

    if run_dir is None:
        return (None, None, None, None)

    source = run_dir / "frontier_augmented" / "scientific_portfolio"
    return (
        source / "candidate_pool.json",
        source / "selection.json",
        source / "materialization.report.json",
        source / "generation.json",
    )


def _candidate_lookup(pool_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in pool_payload.get("candidates", []) or []:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            result[candidate_id] = row
    return result


def _profile_lookup(
    arm: AuditedArm,
    selection_payload: Mapping[str, Any],
) -> dict[str, str]:
    if arm != "PORTFOLIO_SELECTED":
        ids = selection_payload.get("retained_candidate_ids") or []
        return {
            str(candidate_id): "UNPROFILED_FAMILY_BALANCED"
            for candidate_id in ids
        }

    profiles: dict[str, str] = {}
    for row in selection_payload.get("entries", []) or []:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        profiles[candidate_id] = str(
            row.get("assigned_profile") or "UNASSIGNED_PROFILE"
        )
    return profiles


def _selected_ids(selection_payload: Mapping[str, Any]) -> list[str]:
    values = selection_payload.get("retained_candidate_ids") or []
    return [str(value) for value in values if str(value).strip()]


def _selected_family_count(selection_payload: Mapping[str, Any]) -> int:
    value = selection_payload.get("selected_unique_family_count")
    if value is None:
        value = selection_payload.get("retained_unique_family_count")
    if value is not None:
        try:
            return max(0, int(value))
        except Exception:
            pass
    values = selection_payload.get("selected_family_signatures") or []
    if not values:
        values = [
            row.get("conceptual_family_signature")
            for row in (selection_payload.get("entries") or [])
            if isinstance(row, dict)
        ]
    return len({str(value) for value in values if str(value).strip()})


def _generation_error_payload(path: Path | None) -> dict[str, Any] | None:
    payload = _read_json(path)
    value = payload.get("generation_error") if payload else None
    return value if isinstance(value, dict) and value else None


def _primary_failure_taxonomy_code(row: Mapping[str, Any]) -> str | None:
    status = str(row.get("effective_status") or "UNKNOWN")
    if status == "MATERIALIZED":
        return None
    if status == "GENERATION_FAILED":
        return "GENERATION_FAILED"

    issue_codes = sorted(
        {
            str(code or "").strip()
            for code in (row.get("issue_codes") or [])
            if str(code or "").strip()
        }
    )
    if status == "ABSTAINED" and "MODEL_ABSTAINED" in issue_codes:
        return "MODEL_ABSTAINED"
    if issue_codes:
        return issue_codes[0]

    fallback = {
        "ABSTAINED": "UNCLASSIFIED_ABSTENTION",
        "HARD_GATE_REJECTED": "UNCLASSIFIED_HARD_GATE_REJECTION",
        "COMPILE_REJECTED": "UNCLASSIFIED_COMPILE_REJECTION",
        "MISSING_DRAFT": "UNCLASSIFIED_MISSING_DRAFT",
        "MISSING_RECORD": "UNCLASSIFIED_MISSING_RECORD",
        "NOT_OBSERVED": "UNCLASSIFIED_NOT_OBSERVED",
    }
    return fallback.get(status, f"UNCLASSIFIED_{status}")


def _issue_taxonomy(
    rows: list[dict[str, Any]],
) -> list[MaterializationIssueTaxonomyEntry]:
    """Build an exhaustive, mutually-exclusive failure taxonomy.

    Each non-materialized candidate contributes to exactly one primary
    taxonomy bucket. Raw issue-code multiplicity is retained separately on
    the arm/cohort audit and therefore does not inflate failure counts here.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        code = _primary_failure_taxonomy_code(row)
        if code is not None:
            grouped[code].append(row)

    output: list[MaterializationIssueTaxonomyEntry] = []
    for code, items in sorted(grouped.items()):
        output.append(
            MaterializationIssueTaxonomyEntry(
                issue_code=code,
                count=len(items),
                status_counts=_sorted_counter(
                    Counter(str(x["effective_status"]) for x in items)
                ),
                arm_counts=_sorted_counter(Counter(str(x["arm"]) for x in items)),
                origin_counts=_sorted_counter(
                    Counter(str(x["origin"]) for x in items)
                ),
                operator_or_form_counts=_sorted_counter(
                    Counter(str(x["operator_or_form"]) for x in items)
                ),
                profile_counts=_sorted_counter(
                    Counter(str(x["selection_profile"]) for x in items)
                ),
            )
        )
    return output


def _taxonomy_is_unclassified(code: str) -> bool:
    return str(code).startswith("UNCLASSIFIED_")


def build_arm_survival_audit(
    *,
    case_id: str,
    arm: AuditedArm,
    candidate_pool_path: Path | None,
    selection_path: Path | None,
    materialization_report_path: Path | None,
    generation_path: Path | None,
    verification_summary_path: Path | None,
) -> MaterializationSurvivalArmAudit | None:
    selection = _read_json(selection_path)
    if not selection:
        return None

    pool = _read_json(candidate_pool_path)
    candidate_by_id = _candidate_lookup(pool)
    profiles = _profile_lookup(arm, selection)
    selected_ids = _selected_ids(selection)

    materialization = _read_json(materialization_report_path)
    records_by_id = {
        str(row.get("candidate_id")): row
        for row in (materialization.get("records") or [])
        if isinstance(row, dict) and str(row.get("candidate_id") or "").strip()
    }
    materialization_observed = bool(materialization)
    generation_error = _generation_error_payload(generation_path)

    verification = _read_json(verification_summary_path)
    downstream_operational_status = str(
        verification.get("operational_status")
        or verification.get("status")
        or "NOT_OBSERVED"
    )
    downstream_complete = downstream_operational_status in {
        "COMPLETE",
        "COMPLETE_SHADOW_VERIFICATION",
    }

    rows: list[dict[str, Any]] = []
    unresolved_candidate_ids: list[str] = []
    for candidate_id in selected_ids:
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            unresolved_candidate_ids.append(candidate_id)
            candidate = {}
        origin = str(candidate.get("origin") or "UNKNOWN_ORIGIN")
        idea_form = str(candidate.get("idea_form") or "UNKNOWN_FORM")
        operator_id = str(candidate.get("operator_id") or "").strip()
        operator_or_form = operator_id or idea_form
        profile = profiles.get(candidate_id, "UNASSIGNED_PROFILE")
        record = records_by_id.get(candidate_id) if materialization_observed else None
        effective_status = (
            _effective_status(record, generation_error=generation_error)
            if materialization_observed
            else "NOT_OBSERVED"
        )
        issue_codes = list(record.get("issue_codes") or []) if record else []
        rows.append(
            {
                "candidate_id": candidate_id,
                "arm": arm,
                "origin": origin,
                "idea_form": idea_form,
                "operator_or_form": operator_or_form,
                "selection_profile": profile,
                "effective_status": effective_status,
                "issue_codes": [str(x) for x in issue_codes],
            }
        )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["origin"],
                row["idea_form"],
                row["operator_or_form"],
                row["selection_profile"],
            )
        ].append(row)

    cells: list[MaterializationTransitionCell] = []
    for (origin, idea_form, operator_or_form, profile), items in sorted(grouped.items()):
        statuses = Counter(str(x["effective_status"]) for x in items)
        ready = statuses.get("MATERIALIZED", 0)
        issues = Counter(
            str(code)
            for item in items
            for code in item.get("issue_codes", [])
            if str(code).strip()
        )
        cells.append(
            MaterializationTransitionCell(
                arm=arm,
                origin=origin,
                idea_form=idea_form,
                operator_or_form=operator_or_form,
                selection_profile=profile,
                selected_count=len(items),
                status_counts=_sorted_counter(statuses),
                verification_ready_count=ready,
                verification_ready_fraction=(ready / len(items) if items else 0.0),
                downstream_complete_count=(ready if downstream_complete else 0),
                issue_code_counts=_sorted_counter(issues),
            )
        )

    statuses = Counter(str(x["effective_status"]) for x in rows)
    ready_count = statuses.get("MATERIALIZED", 0)
    non_materialized_count = len(rows) - ready_count
    taxonomy = _issue_taxonomy(rows)
    taxonomy_classified = sum(x.count for x in taxonomy)
    taxonomy_unclassified = sum(
        x.count for x in taxonomy if _taxonomy_is_unclassified(x.issue_code)
    )
    raw_issue_codes = Counter(
        str(code)
        for row in rows
        for code in row.get("issue_codes", [])
        if str(code).strip()
    )

    error_counts: Counter[str] = Counter()
    if generation_error is not None:
        error_counts[
            str(generation_error.get("error_type") or "UNKNOWN_GENERATION_ERROR")
        ] += 1

    return MaterializationSurvivalArmAudit(
        case_id=case_id,
        arm=arm,
        candidate_pool_path=(str(candidate_pool_path) if candidate_pool_path else None),
        selection_path=(str(selection_path) if selection_path else None),
        materialization_report_path=(
            str(materialization_report_path) if materialization_report_path else None
        ),
        generation_path=(
            str(generation_path)
            if generation_path and generation_path.is_file()
            else None
        ),
        verification_summary_path=(
            str(verification_summary_path) if verification_summary_path else None
        ),
        selected_candidate_count=len(selected_ids),
        selected_unique_family_count=_selected_family_count(selection),
        materialization_observed=materialization_observed,
        terminal_status_counts=_sorted_counter(statuses),
        verification_ready_count=ready_count,
        verification_ready_fraction=(ready_count / len(selected_ids) if selected_ids else 0.0),
        downstream_operational_status=downstream_operational_status,
        downstream_complete=downstream_complete,
        downstream_complete_materialized_count=(ready_count if downstream_complete else 0),
        selected_count_by_origin=_sorted_counter(Counter(str(x["origin"]) for x in rows)),
        selected_count_by_form=_sorted_counter(Counter(str(x["idea_form"]) for x in rows)),
        selected_count_by_operator_or_form=_sorted_counter(
            Counter(str(x["operator_or_form"]) for x in rows)
        ),
        selected_count_by_profile=_sorted_counter(
            Counter(str(x["selection_profile"]) for x in rows)
        ),
        transition_cells=cells,
        issue_taxonomy=taxonomy,
        raw_issue_code_counts=_sorted_counter(raw_issue_codes),
        generation_error_counts=_sorted_counter(error_counts),
        unresolved_candidate_ids=sorted(unresolved_candidate_ids),
        lineage_resolution_complete=not unresolved_candidate_ids,
        non_materialized_count=non_materialized_count,
        taxonomy_classified_failure_count=taxonomy_classified,
        taxonomy_unclassified_failure_count=taxonomy_unclassified,
        taxonomy_exhaustive=(taxonomy_classified == non_materialized_count),
    )


def _execution_case_lookup(
    execution: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in execution.get("cases", []) or []:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or "").strip()
        if case_id:
            result[case_id] = row
    return result


def _execution_row_observed(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    status = str(row.get("status") or "").strip()
    return bool(status) and not status.startswith("NOT_STARTED")


def _run_dir_from_case(
    case_dir: Path,
    *,
    execution_row: Mapping[str, Any] | None = None,
) -> Path | None:
    case_audit = _read_json(case_dir / "case.audit.json")
    value = str(case_audit.get("run_dir") or "").strip()
    if value:
        return Path(value).expanduser().resolve()

    if execution_row is not None:
        value = str(execution_row.get("run_dir") or "").strip()
        if value:
            return Path(value).expanduser().resolve()
    return None


def build_case_survival_audit(
    *,
    case_dir: Path,
    run_dir: Path | None = None,
    execution_row: Mapping[str, Any] | None = None,
) -> MaterializationSurvivalCaseAudit:
    case_dir = case_dir.expanduser().resolve()
    case_id = case_dir.name
    if run_dir is None:
        run_dir = _run_dir_from_case(case_dir, execution_row=execution_row)

    pool_path, selected_selection, selected_materialization, selected_generation = (
        _selection_source_paths(case_dir=case_dir, run_dir=run_dir)
    )

    arms: list[MaterializationSurvivalArmAudit] = []
    for arm in AUDITED_ARMS:
        arm_dir = case_dir / arm
        if arm == "PORTFOLIO_SELECTED":
            selection_path = selected_selection
            materialization_path = selected_materialization
            generation_path = selected_generation
        else:
            selection_path = arm_dir / "selection.json"
            materialization_path = arm_dir / "materialization.report.json"
            generation_path = arm_dir / "generation.json"

        audit = build_arm_survival_audit(
            case_id=case_id,
            arm=arm,
            candidate_pool_path=pool_path,
            selection_path=selection_path,
            materialization_report_path=materialization_path,
            generation_path=generation_path,
            verification_summary_path=arm_dir / "verification" / "verification.summary.json",
        )
        if audit is not None:
            arms.append(audit)

    execution_status = (
        str(execution_row.get("status") or "").strip()
        if execution_row is not None
        else ""
    ) or None
    execution_observed = _execution_row_observed(execution_row)
    lineage_resolution_complete = all(
        arm.lineage_resolution_complete for arm in arms
    )

    provisional = MaterializationSurvivalCaseAudit(
        audit_id="pending",
        audit_sha256="pending",
        case_id=case_id,
        run_dir=(str(run_dir) if run_dir is not None else None),
        prospective_case_dir=str(case_dir),
        observed_arm_count=len(arms),
        arms=arms,
        partial_case=len(arms) < len(AUDITED_ARMS),
        execution_observed=execution_observed,
        execution_status=execution_status,
        lineage_resolution_complete=lineage_resolution_complete,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("audit_id", None)
    payload.pop("audit_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "audit_id": f"materialization_survival_case:{digest[:20]}",
            "audit_sha256": digest,
        }
    )


def _merge_transition_cells(
    cases: list[MaterializationSurvivalCaseAudit],
) -> list[MaterializationTransitionCell]:
    grouped: dict[tuple[str, str, str, str, str], list[MaterializationTransitionCell]] = defaultdict(list)
    for case in cases:
        for arm in case.arms:
            for cell in arm.transition_cells:
                grouped[
                    (
                        cell.arm,
                        cell.origin,
                        cell.idea_form,
                        cell.operator_or_form,
                        cell.selection_profile,
                    )
                ].append(cell)

    output: list[MaterializationTransitionCell] = []
    for (arm, origin, idea_form, operator_or_form, profile), rows in sorted(grouped.items()):
        selected = sum(x.selected_count for x in rows)
        ready = sum(x.verification_ready_count for x in rows)
        downstream = sum(x.downstream_complete_count for x in rows)
        statuses: Counter[str] = Counter()
        issues: Counter[str] = Counter()
        for row in rows:
            statuses.update(row.status_counts)
            issues.update(row.issue_code_counts)
        output.append(
            MaterializationTransitionCell(
                arm=arm,
                origin=origin,
                idea_form=idea_form,
                operator_or_form=operator_or_form,
                selection_profile=profile,
                selected_count=selected,
                status_counts=_sorted_counter(statuses),
                verification_ready_count=ready,
                verification_ready_fraction=(ready / selected if selected else 0.0),
                downstream_complete_count=downstream,
                issue_code_counts=_sorted_counter(issues),
            )
        )
    return output


def _merge_issue_taxonomy(
    cases: list[MaterializationSurvivalCaseAudit],
) -> list[MaterializationIssueTaxonomyEntry]:
    grouped: dict[str, dict[str, Counter[str] | int]] = {}
    for case in cases:
        for arm in case.arms:
            for entry in arm.issue_taxonomy:
                slot = grouped.setdefault(
                    entry.issue_code,
                    {
                        "count": 0,
                        "status": Counter(),
                        "arm": Counter(),
                        "origin": Counter(),
                        "operator": Counter(),
                        "profile": Counter(),
                    },
                )
                slot["count"] = int(slot["count"]) + entry.count
                assert isinstance(slot["status"], Counter)
                assert isinstance(slot["arm"], Counter)
                assert isinstance(slot["origin"], Counter)
                assert isinstance(slot["operator"], Counter)
                assert isinstance(slot["profile"], Counter)
                slot["status"].update(entry.status_counts)
                slot["arm"].update(entry.arm_counts)
                slot["origin"].update(entry.origin_counts)
                slot["operator"].update(entry.operator_or_form_counts)
                slot["profile"].update(entry.profile_counts)

    output: list[MaterializationIssueTaxonomyEntry] = []
    for code, slot in sorted(grouped.items()):
        output.append(
            MaterializationIssueTaxonomyEntry(
                issue_code=code,
                count=int(slot["count"]),
                status_counts=_sorted_counter(slot["status"]),
                arm_counts=_sorted_counter(slot["arm"]),
                origin_counts=_sorted_counter(slot["origin"]),
                operator_or_form_counts=_sorted_counter(slot["operator"]),
                profile_counts=_sorted_counter(slot["profile"]),
            )
        )
    return output


def _per_case_survival_matrix(
    cases: list[MaterializationSurvivalCaseAudit],
) -> list[MaterializationCaseSurvivalRow]:
    output: list[MaterializationCaseSurvivalRow] = []
    for case in cases:
        output.append(
            MaterializationCaseSurvivalRow(
                case_id=case.case_id,
                execution_observed=case.execution_observed,
                execution_status=case.execution_status,
                arms=[
                    MaterializationCaseArmSurvivalCell(
                        arm=arm.arm,
                        selected_candidate_count=arm.selected_candidate_count,
                        verification_ready_count=arm.verification_ready_count,
                        verification_ready_fraction=arm.verification_ready_fraction,
                        materialization_observed=arm.materialization_observed,
                        lineage_resolution_complete=arm.lineage_resolution_complete,
                        terminal_status_counts=arm.terminal_status_counts,
                    )
                    for arm in case.arms
                ],
            )
        )
    return output


def _common_materialization_cases(
    cases: list[MaterializationSurvivalCaseAudit],
) -> list[MaterializationSurvivalCaseAudit]:
    output: list[MaterializationSurvivalCaseAudit] = []
    required = set(AUDITED_ARMS)
    for case in cases:
        by_arm = {arm.arm: arm for arm in case.arms}
        if set(by_arm) != required:
            continue
        if not all(by_arm[arm].materialization_observed for arm in AUDITED_ARMS):
            continue
        if not all(by_arm[arm].lineage_resolution_complete for arm in AUDITED_ARMS):
            continue
        output.append(case)
    return output


def _paired_comparisons(
    common_cases: list[MaterializationSurvivalCaseAudit],
) -> list[MaterializationPairedComparison]:
    pairs: tuple[tuple[AuditedArm, AuditedArm], ...] = (
        ("FRONTIER_BALANCED", "EVOLUTION_BALANCED"),
        ("EVOLUTION_BALANCED", "PORTFOLIO_SELECTED"),
        ("FRONTIER_BALANCED", "PORTFOLIO_SELECTED"),
    )
    output: list[MaterializationPairedComparison] = []
    for left_arm, right_arm in pairs:
        case_ids: list[str] = []
        left_values: list[float] = []
        right_values: list[float] = []
        deltas: list[float] = []
        higher = equal = lower = 0
        for case in common_cases:
            by_arm = {arm.arm: arm for arm in case.arms}
            left = by_arm[left_arm].verification_ready_fraction
            right = by_arm[right_arm].verification_ready_fraction
            delta = right - left
            case_ids.append(case.case_id)
            left_values.append(left)
            right_values.append(right)
            deltas.append(delta)
            if abs(delta) <= 1e-12:
                equal += 1
            elif delta > 0:
                higher += 1
            else:
                lower += 1
        output.append(
            MaterializationPairedComparison(
                left_arm=left_arm,
                right_arm=right_arm,
                case_count=len(case_ids),
                case_ids=case_ids,
                median_left_yield=_median_optional(left_values),
                median_right_yield=_median_optional(right_values),
                median_delta_right_minus_left=_median_optional(deltas),
                right_higher_case_count=higher,
                equal_case_count=equal,
                right_lower_case_count=lower,
            )
        )
    return output


def build_cohort_survival_audit(
    *,
    prospective_output_root: Path,
) -> MaterializationSurvivalCohortAudit:
    root = prospective_output_root.expanduser().resolve()
    execution = _read_json(root / "execution_status.json")
    planned = int(execution.get("planned_case_count") or 0)
    execution_by_case = _execution_case_lookup(execution)

    case_dirs: list[Path] = []
    for path in sorted(root.iterdir() if root.is_dir() else []):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        if (path / "case.audit.json").is_file() or any(
            (path / arm).is_dir() for arm in AUDITED_ARMS
        ):
            case_dirs.append(path)

    cases = [
        build_case_survival_audit(
            case_dir=path,
            execution_row=execution_by_case.get(path.name),
        )
        for path in case_dirs
    ]
    if planned <= 0:
        planned = len(execution_by_case) or len(cases)

    by_arm: dict[str, list[MaterializationSurvivalArmAudit]] = defaultdict(list)
    generation_errors: Counter[str] = Counter()
    raw_issue_codes: Counter[str] = Counter()
    terminal_by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        for arm in case.arms:
            by_arm[arm.arm].append(arm)
            generation_errors.update(arm.generation_error_counts)
            raw_issue_codes.update(arm.raw_issue_code_counts)
            terminal_by_arm[arm.arm].update(arm.terminal_status_counts)

    execution_observed_case_count = sum(
        _execution_row_observed(row)
        for row in execution_by_case.values()
    )
    artifact_observed_case_count = len(cases)
    artifact_only_case_count = sum(
        not case.execution_observed for case in cases
    )

    common_cases = _common_materialization_cases(cases)
    taxonomy = _merge_issue_taxonomy(cases)
    non_materialized_count = sum(
        arm.non_materialized_count
        for case in cases
        for arm in case.arms
    )
    taxonomy_classified = sum(x.count for x in taxonomy)
    taxonomy_unclassified = sum(
        x.count for x in taxonomy
        if _taxonomy_is_unclassified(x.issue_code)
    )

    provisional = MaterializationSurvivalCohortAudit(
        audit_id="pending",
        audit_sha256="pending",
        prospective_output_root=str(root),
        planned_case_count=planned,
        observed_case_count=artifact_observed_case_count,
        artifact_observed_case_count=artifact_observed_case_count,
        execution_observed_case_count=execution_observed_case_count,
        artifact_only_case_count=artifact_only_case_count,
        partial_case_count=sum(case.partial_case for case in cases),
        cases=cases,
        per_case_survival_matrix=_per_case_survival_matrix(cases),
        common_materialization_case_count=len(common_cases),
        common_materialization_case_ids=[case.case_id for case in common_cases],
        paired_comparisons=_paired_comparisons(common_cases),
        arm_selected_candidate_counts={
            arm: sum(x.selected_candidate_count for x in by_arm.get(arm, []))
            for arm in AUDITED_ARMS
        },
        arm_verification_ready_counts={
            arm: sum(x.verification_ready_count for x in by_arm.get(arm, []))
            for arm in AUDITED_ARMS
        },
        arm_verification_ready_fraction_medians={
            arm: _median_optional(
                [x.verification_ready_fraction for x in by_arm.get(arm, [])]
            )
            for arm in AUDITED_ARMS
        },
        arm_downstream_complete_case_counts={
            arm: sum(x.downstream_complete for x in by_arm.get(arm, []))
            for arm in AUDITED_ARMS
        },
        arm_terminal_status_counts={
            arm: _sorted_counter(terminal_by_arm.get(arm, Counter()))
            for arm in AUDITED_ARMS
        },
        transition_cells=_merge_transition_cells(cases),
        issue_taxonomy=taxonomy,
        raw_issue_code_counts=_sorted_counter(raw_issue_codes),
        generation_error_counts=_sorted_counter(generation_errors),
        non_materialized_count=non_materialized_count,
        taxonomy_classified_failure_count=taxonomy_classified,
        taxonomy_unclassified_failure_count=taxonomy_unclassified,
        taxonomy_exhaustive=(taxonomy_classified == non_materialized_count),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("audit_id", None)
    payload.pop("audit_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "audit_id": f"materialization_survival_cohort:{digest[:20]}",
            "audit_sha256": digest,
        }
    )


__all__ = [
    "AUDITED_ARMS",
    "MaterializationTransitionCell",
    "MaterializationIssueTaxonomyEntry",
    "MaterializationCaseArmSurvivalCell",
    "MaterializationCaseSurvivalRow",
    "MaterializationPairedComparison",
    "MaterializationSurvivalArmAudit",
    "MaterializationSurvivalCaseAudit",
    "MaterializationSurvivalCohortAudit",
    "build_arm_survival_audit",
    "build_case_survival_audit",
    "build_cohort_survival_audit",
]
