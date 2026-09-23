from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


StageState = Literal[
    "REUSE_MATCHED",
    "NEEDS_DETERMINISTIC_MATERIALIZATION",
    "NEEDS_LLM_MATERIALIZATION",
    "NEEDS_RETRIEVAL_AND_REVIEW",
    "BLOCKED_UPSTREAM",
    "AMBIGUOUS_LINEAGE",
    "NOT_APPLICABLE",
]

TaskStatus = Literal[
    "READY_FOR_VERIFIER",
    "MATERIALIZATION_REQUIRED",
    "BLOCKED_UPSTREAM",
    "AMBIGUOUS_LINEAGE",
    "EXCLUDED_BY_CALLER",
]


class ArtifactRef(StrictModel):
    artifact_kind: str
    path: str
    schema_version: str | None = None
    object_id: str | None = None
    source_object_id: str | None = None
    hypothesis_ids: list[str] = Field(default_factory=list)


class MaterializationStagePlan(StrictModel):
    stage_id: str
    state: StageState
    selected_artifacts: list[ArtifactRef] = Field(default_factory=list)
    ignored_stale_or_mismatched_artifacts: list[str] = Field(default_factory=list)
    missing_prerequisites: list[str] = Field(default_factory=list)
    next_action: str | None = None
    expected_llm_calls: int | None = Field(default=None, ge=0)
    expected_retrieval_runs: int = Field(default=0, ge=0)
    runtime_dependent_llm_calls: bool = False
    notes: list[str] = Field(default_factory=list)


class ScientificVerifierMaterializationTask(StrictModel):
    task_key: str
    case_key: str
    replicate_key: str
    canonical_dir: str
    status: TaskStatus
    stages: list[MaterializationStagePlan]
    verifier_required_artifacts_present: bool
    planned_minimum_llm_calls: int = Field(ge=0)
    planned_retrieval_runs: int = Field(ge=0)
    runtime_dependent_llm_stage_ids: list[str] = Field(default_factory=list)
    diagnostic_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_stage_ids(self) -> "ScientificVerifierMaterializationTask":
        ids = [row.stage_id for row in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("materialization stage IDs must be unique")
        if self.status == "READY_FOR_VERIFIER" and not self.verifier_required_artifacts_present:
            raise ValueError("READY_FOR_VERIFIER requires all verifier inputs")
        if self.status == "EXCLUDED_BY_CALLER" and self.verifier_required_artifacts_present:
            raise ValueError("excluded tasks are not assessed for verifier readiness")
        return self


class ScientificVerifierMaterializationPlan(StrictModel):
    schema_version: Literal[
        "scientific-verifier-holdout-materialization-plan-v1"
    ] = "scientific-verifier-holdout-materialization-plan-v1"

    plan_id: str
    benchmark_root: str
    requested_case_keys: list[str]
    excluded_case_keys: list[str]
    selected_case_keys: list[str]
    tasks: list[ScientificVerifierMaterializationTask]
    task_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)

    llm_calls_performed: Literal[0] = 0
    retrieval_performed: Literal[False] = False
    filesystem_inventory_only: Literal[True] = True
    cohort_selected_before_artifact_outcomes: Literal[True] = True
    unseen_status_is_caller_declared_only: Literal[True] = True
    contamination_audit_performed: Literal[False] = False
    scientific_quality_judgment_performed: Literal[False] = False
    validation_outcome_assessed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    rerun_after_materialization_to_bind_new_lineage: Literal[True] = True

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificVerifierMaterializationPlan":
        if self.task_count != len(self.tasks):
            raise ValueError("task_count mismatch")
        keys = [row.task_key for row in self.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate task_key in materialization plan")
        counts = Counter(row.status for row in self.tasks)
        if dict(sorted(counts.items())) != dict(sorted(self.status_counts.items())):
            raise ValueError("status_counts mismatch")
        return self


@dataclass(frozen=True)
class _JsonArtifact:
    path: Path
    payload: dict[str, object]


_SCHEMA_CONTEXT = "hypothesis-context-v1"
_SCHEMA_CROSS_LANE = "cross-lane-scientific-reasoning-shadow-portfolio-v1"
_SCHEMA_REFRAME = "scientific-reframing-shadow-report-v1"
_SCHEMA_PROXY = "scientific-proxy-challenge-shadow-report-v1"
_SCHEMA_CONTRADICTION = "scientific-contradiction-resolution-shadow-report-v1"
_SCHEMA_CANDIDATE = "production-facing-scientific-candidate-portfolio-v1"
_SCHEMA_ATOMIC_REPORT = "atomic-cross-lane-scientific-synthesis-report-v1"
_SCHEMA_HYPOTHESIS_PORTFOLIO = "hypothesis-portfolio-v1"
_SCHEMA_GROUNDED_IDENTITY = "grounded-identity-constituent-annotation-report-v1"
_SCHEMA_EXTERNAL_NOVELTY = "external-novelty-report-v1"

_RELATIONAL_PORTFOLIO_FILENAMES = (
    "novelty_refinement_a6.portfolio.json",
    "hypothesis_axis_a4.portfolio.json",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _inventory(canonical_dir: Path) -> list[_JsonArtifact]:
    rows: list[_JsonArtifact] = []
    if not canonical_dir.is_dir():
        return rows
    for path in sorted(canonical_dir.glob("*.json")):
        payload = _load_json(path)
        if payload is not None:
            rows.append(_JsonArtifact(path=path, payload=payload))
    return rows


def _schema(row: _JsonArtifact) -> str | None:
    value = row.payload.get("schema_version")
    return str(value) if isinstance(value, str) else None


def _string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if isinstance(value, str) and value else None


def _hypothesis_ids_from_portfolio(payload: dict[str, object]) -> list[str]:
    rows = payload.get("hypotheses")
    if not isinstance(rows, list):
        return []
    result: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("hypothesis_id")
        if isinstance(value, str) and value:
            result.append(value)
    return sorted(dict.fromkeys(result))


def _hypothesis_ids_from_atomic_report(payload: dict[str, object]) -> list[str]:
    return _hypothesis_ids_from_portfolio(payload)


def _hypothesis_ids_from_external_report(payload: dict[str, object]) -> list[str]:
    rows = payload.get("cards")
    if not isinstance(rows, list):
        return []
    result: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("hypothesis_id")
        if isinstance(value, str) and value:
            result.append(value)
    return sorted(dict.fromkeys(result))


def _artifact_ref(
    row: _JsonArtifact,
    *,
    artifact_kind: str,
    object_id_key: str | None = None,
    source_object_id_key: str | None = None,
    hypothesis_ids: list[str] | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_kind=artifact_kind,
        path=str(row.path),
        schema_version=_schema(row),
        object_id=(
            _string(row.payload, object_id_key)
            if object_id_key is not None
            else None
        ),
        source_object_id=(
            _string(row.payload, source_object_id_key)
            if source_object_id_key is not None
            else None
        ),
        hypothesis_ids=list(hypothesis_ids or []),
    )


def _rows_with_schema(rows: list[_JsonArtifact], schema_version: str) -> list[_JsonArtifact]:
    return [row for row in rows if _schema(row) == schema_version]


def _distinct_by_id(
    rows: list[_JsonArtifact],
    *,
    id_key: str,
) -> tuple[list[_JsonArtifact], list[str]]:
    by_id: dict[str, _JsonArtifact] = {}
    duplicate_paths: list[str] = []
    anonymous: list[_JsonArtifact] = []
    for row in rows:
        object_id = _string(row.payload, id_key)
        if object_id is None:
            anonymous.append(row)
            continue
        if object_id in by_id:
            duplicate_paths.append(str(row.path))
            continue
        by_id[object_id] = row
    return list(by_id.values()) + anonymous, duplicate_paths


def _single_context(rows: list[_JsonArtifact]) -> tuple[_JsonArtifact | None, bool]:
    contexts = _rows_with_schema(rows, _SCHEMA_CONTEXT)
    distinct, _ = _distinct_by_id(contexts, id_key="context_id")
    if len(distinct) == 1:
        return distinct[0], False
    return None, len(distinct) > 1


def _matching_cross_lane_rows(
    rows: list[_JsonArtifact],
    *,
    context_id: str | None,
) -> tuple[list[_JsonArtifact], list[str]]:
    portfolios = _rows_with_schema(rows, _SCHEMA_CROSS_LANE)
    exact = [
        row
        for row in portfolios
        if context_id is None or _string(row.payload, "source_context_id") == context_id
    ]
    mismatched = [str(row.path) for row in portfolios if row not in exact]
    distinct, duplicates = _distinct_by_id(exact, id_key="portfolio_id")
    return distinct, sorted(mismatched + duplicates)


def _matching_candidate_rows(
    rows: list[_JsonArtifact],
    *,
    context_id: str | None,
    allowed_cross_lane_portfolio_ids: set[str] | None = None,
) -> tuple[list[_JsonArtifact], list[str]]:
    candidates = _rows_with_schema(rows, _SCHEMA_CANDIDATE)
    exact: list[_JsonArtifact] = []
    for row in candidates:
        context_matches = (
            context_id is None
            or _string(row.payload, "source_context_id") == context_id
        )
        if not context_matches:
            continue
        if allowed_cross_lane_portfolio_ids:
            source_cross_lane = _string(
                row.payload,
                "source_cross_lane_portfolio_id",
            )
            if source_cross_lane not in allowed_cross_lane_portfolio_ids:
                continue
        exact.append(row)
    mismatched = [str(row.path) for row in candidates if row not in exact]
    distinct, duplicates = _distinct_by_id(exact, id_key="portfolio_id")
    return distinct, sorted(mismatched + duplicates)


def _string_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return sorted(
        dict.fromkeys(
            str(item)
            for item in value
            if isinstance(item, str) and item
        )
    )


def _report_candidate_ids(row: _JsonArtifact) -> list[str]:
    schema = _schema(row)
    if schema in {_SCHEMA_PROXY, _SCHEMA_CONTRADICTION}:
        return _string_list(row.payload, "candidate_ids")
    if schema != _SCHEMA_REFRAME:
        return []
    candidates = row.payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    result: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("reframe_id")
        if isinstance(value, str) and value:
            result.append(value)
    return sorted(dict.fromkeys(result))


def _cross_lane_reframing_source_ids(row: _JsonArtifact) -> list[str]:
    entries = row.payload.get("entries")
    if not isinstance(entries, list):
        return []
    result: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("lane_id") != "SCIENTIFIC_REFRAMING":
            continue
        value = entry.get("source_object_id")
        if isinstance(value, str) and value:
            result.append(value)
    return sorted(dict.fromkeys(result))


def _cross_lane_relational_portfolio_id(row: _JsonArtifact) -> str | None:
    lane = row.payload.get("relational_lane")
    if not isinstance(lane, dict):
        return None
    value = lane.get("source_portfolio_id")
    return str(value) if isinstance(value, str) and value else None


def _matching_reframing_source_bundles(
    rows: list[_JsonArtifact],
    *,
    context_id: str | None,
    target_candidate_ids: list[str],
) -> list[tuple[_JsonArtifact, _JsonArtifact | None, _JsonArtifact | None]]:
    target = set(target_candidate_ids)
    reframes = [
        row
        for row in _rows_with_schema(rows, _SCHEMA_REFRAME)
        if context_id is None or _string(row.payload, "source_context_id") == context_id
    ]
    proxies = [
        row
        for row in _rows_with_schema(rows, _SCHEMA_PROXY)
        if context_id is None or _string(row.payload, "source_context_id") == context_id
    ]
    contradictions = [
        row
        for row in _rows_with_schema(rows, _SCHEMA_CONTRADICTION)
        if context_id is None or _string(row.payload, "source_context_id") == context_id
    ]

    bundles: list[tuple[_JsonArtifact, _JsonArtifact | None, _JsonArtifact | None]] = []
    for reframe in reframes:
        for proxy in [None, *proxies]:
            for contradiction in [None, *contradictions]:
                ids = set(_report_candidate_ids(reframe))
                if proxy is not None:
                    ids.update(_report_candidate_ids(proxy))
                if contradiction is not None:
                    ids.update(_report_candidate_ids(contradiction))
                if ids == target:
                    bundles.append((reframe, proxy, contradiction))

    # Collapse byte/ID-equivalent combinations that only differ by duplicate files.
    unique: dict[tuple[str, str | None, str | None], tuple[_JsonArtifact, _JsonArtifact | None, _JsonArtifact | None]] = {}
    for reframe, proxy, contradiction in bundles:
        key = (
            _string(reframe.payload, "report_id") or str(reframe.path),
            (_string(proxy.payload, "report_id") if proxy is not None else None),
            (
                _string(contradiction.payload, "report_id")
                if contradiction is not None
                else None
            ),
        )
        unique.setdefault(key, (reframe, proxy, contradiction))
    return list(unique.values())


def _matching_atomic_reports(
    rows: list[_JsonArtifact],
    *,
    candidate_portfolio_id: str,
) -> tuple[list[_JsonArtifact], list[str]]:
    all_reports = _rows_with_schema(rows, _SCHEMA_ATOMIC_REPORT)
    exact = [
        row
        for row in all_reports
        if _string(row.payload, "source_candidate_portfolio_id") == candidate_portfolio_id
    ]
    mismatched = [str(row.path) for row in all_reports if row not in exact]
    distinct, duplicates = _distinct_by_id(exact, id_key="report_id")
    return distinct, sorted(mismatched + duplicates)


def _matching_atomic_portfolios(
    rows: list[_JsonArtifact],
    *,
    hypothesis_ids: list[str],
    source_context_id: str | None,
) -> tuple[list[_JsonArtifact], list[str]]:
    portfolios = _rows_with_schema(rows, _SCHEMA_HYPOTHESIS_PORTFOLIO)
    exact: list[_JsonArtifact] = []
    mismatched: list[str] = []
    expected = sorted(hypothesis_ids)
    for row in portfolios:
        ids = _hypothesis_ids_from_portfolio(row.payload)
        same_ids = ids == expected
        same_context = (
            source_context_id is None
            or _string(row.payload, "source_context_id") == source_context_id
        )
        if same_ids and same_context:
            exact.append(row)
        elif "atomic" in row.path.name or "scientific_synthesis" in row.path.name:
            mismatched.append(str(row.path))
    distinct, duplicates = _distinct_by_id(exact, id_key="portfolio_id")
    return distinct, sorted(mismatched + duplicates)


def _matching_grounded_identity(
    rows: list[_JsonArtifact],
    *,
    candidate_portfolio_id: str,
    atomic_report_id: str,
) -> tuple[list[_JsonArtifact], list[str]]:
    annotations = _rows_with_schema(rows, _SCHEMA_GROUNDED_IDENTITY)
    exact = [
        row
        for row in annotations
        if _string(row.payload, "source_candidate_portfolio_id") == candidate_portfolio_id
        and _string(row.payload, "source_atomic_synthesis_report_id") == atomic_report_id
    ]
    mismatched = [str(row.path) for row in annotations if row not in exact]
    # Annotation reports currently have no report_id. Treat byte-identical lineage copies
    # as harmless duplicates but refuse multiple distinct matching payloads.
    by_digest: dict[str, _JsonArtifact] = {}
    duplicates: list[str] = []
    for row in exact:
        digest = hashlib.sha256(_canonical_json(row.payload).encode("utf-8")).hexdigest()
        if digest in by_digest:
            duplicates.append(str(row.path))
        else:
            by_digest[digest] = row
    return list(by_digest.values()), sorted(mismatched + duplicates)


def _matching_external_reports(
    rows: list[_JsonArtifact],
    *,
    atomic_portfolio_id: str,
    hypothesis_ids: list[str],
) -> tuple[list[_JsonArtifact], list[str]]:
    reports = _rows_with_schema(rows, _SCHEMA_EXTERNAL_NOVELTY)
    expected = sorted(hypothesis_ids)
    exact = [
        row
        for row in reports
        if _string(row.payload, "source_portfolio_id") == atomic_portfolio_id
        and _hypothesis_ids_from_external_report(row.payload) == expected
    ]
    mismatched = [str(row.path) for row in reports if row not in exact]
    distinct, duplicates = _distinct_by_id(exact, id_key="report_id")
    return distinct, sorted(mismatched + duplicates)


def _upstream_candidate_prerequisites(
    rows: list[_JsonArtifact],
    *,
    context_id: str | None,
    cross_lane: _JsonArtifact | None = None,
) -> tuple[list[ArtifactRef], list[str]]:
    selected: list[ArtifactRef] = []
    missing: list[str] = []

    expected_relational_portfolio_id = (
        _cross_lane_relational_portfolio_id(cross_lane)
        if cross_lane is not None
        else None
    )
    relational: _JsonArtifact | None = None
    if expected_relational_portfolio_id is not None:
        relational = next(
            (
                row
                for row in _rows_with_schema(rows, _SCHEMA_HYPOTHESIS_PORTFOLIO)
                if _string(row.payload, "portfolio_id")
                == expected_relational_portfolio_id
            ),
            None,
        )
    else:
        for name in _RELATIONAL_PORTFOLIO_FILENAMES:
            found = next((row for row in rows if row.path.name == name), None)
            if found is not None:
                relational = found
                break

    if relational is None:
        missing.append("relational_or_pre_n10_hypothesis_portfolio")
    else:
        selected.append(
            _artifact_ref(
                relational,
                artifact_kind="relational_or_pre_n10_hypothesis_portfolio",
                object_id_key="portfolio_id",
                hypothesis_ids=_hypothesis_ids_from_portfolio(relational.payload),
            )
        )

    if cross_lane is None:
        cross_lane_rows, _ = _matching_cross_lane_rows(
            rows,
            context_id=context_id,
        )
        if len(cross_lane_rows) == 1:
            cross_lane = cross_lane_rows[0]
        elif not cross_lane_rows:
            missing.append("cross_lane_reasoning_portfolio")
        else:
            missing.append("unique_cross_lane_reasoning_portfolio")

    if cross_lane is not None:
        selected.append(
            _artifact_ref(
                cross_lane,
                artifact_kind="cross_lane_reasoning_portfolio",
                object_id_key="portfolio_id",
            )
        )
        target_reframing_ids = _cross_lane_reframing_source_ids(cross_lane)
        bundles = _matching_reframing_source_bundles(
            rows,
            context_id=context_id,
            target_candidate_ids=target_reframing_ids,
        )
        if len(bundles) == 1:
            reframe, proxy, contradiction = bundles[0]
            selected.append(
                _artifact_ref(
                    reframe,
                    artifact_kind="scientific_reframing_shadow",
                    object_id_key="report_id",
                )
            )
            if proxy is not None:
                selected.append(
                    _artifact_ref(
                        proxy,
                        artifact_kind="scientific_proxy_challenge_shadow",
                        object_id_key="report_id",
                    )
                )
            if contradiction is not None:
                selected.append(
                    _artifact_ref(
                        contradiction,
                        artifact_kind="scientific_contradiction_resolution_shadow",
                        object_id_key="report_id",
                    )
                )
        elif not bundles:
            reframe_rows = [
                row
                for row in _rows_with_schema(rows, _SCHEMA_REFRAME)
                if context_id is None
                or _string(row.payload, "source_context_id") == context_id
            ]
            if not reframe_rows:
                missing.append("scientific_reframing_shadow")
            else:
                missing.append("lineage_complete_reframing_source_bundle")
        else:
            missing.append("unique_reframing_source_bundle")

    return selected, missing


def _excluded_task(
    *,
    root: Path,
    case_key: str,
) -> ScientificVerifierMaterializationTask:
    canonical = root / case_key / "replicate_01" / "canonical"
    return ScientificVerifierMaterializationTask(
        task_key=f"{case_key}/replicate_01",
        case_key=case_key,
        replicate_key="replicate_01",
        canonical_dir=str(canonical),
        status="EXCLUDED_BY_CALLER",
        stages=[
            MaterializationStagePlan(
                stage_id="caller_exclusion",
                state="NOT_APPLICABLE",
                next_action="none",
                expected_llm_calls=0,
                notes=["Case excluded before artifact outcomes were inspected."],
            )
        ],
        verifier_required_artifacts_present=False,
        planned_minimum_llm_calls=0,
        planned_retrieval_runs=0,
        diagnostic_notes=["No validation or scientific-quality conclusion was drawn."],
    )


def _plan_task(
    *,
    root: Path,
    case_key: str,
    canonical_dir: Path,
) -> ScientificVerifierMaterializationTask:
    rows = _inventory(canonical_dir)
    replicate_key = canonical_dir.parent.name
    task_key = f"{case_key}/{replicate_key}"
    stages: list[MaterializationStagePlan] = []
    diagnostic_notes: list[str] = []

    context, context_ambiguous = _single_context(rows)
    context_id = _string(context.payload, "context_id") if context is not None else None
    if context_ambiguous:
        diagnostic_notes.append("Multiple hypothesis contexts were found; lineage cannot be selected safely.")

    cross_lane_rows, cross_lane_ignored = _matching_cross_lane_rows(
        rows,
        context_id=context_id,
    )
    cross_lane_ids = {
        value
        for row in cross_lane_rows
        if (value := _string(row.payload, "portfolio_id")) is not None
    }
    candidate_rows, candidate_ignored = _matching_candidate_rows(
        rows,
        context_id=context_id,
        allowed_cross_lane_portfolio_ids=(cross_lane_ids or None),
    )
    candidate_ignored = sorted(
        dict.fromkeys(candidate_ignored + cross_lane_ignored)
    )
    candidate: _JsonArtifact | None = candidate_rows[0] if len(candidate_rows) == 1 else None
    selected_cross_lane: _JsonArtifact | None = None
    if candidate is not None:
        source_cross_lane_id = _string(
            candidate.payload,
            "source_cross_lane_portfolio_id",
        )
        if source_cross_lane_id is not None:
            selected_cross_lane = next(
                (
                    row
                    for row in cross_lane_rows
                    if _string(row.payload, "portfolio_id") == source_cross_lane_id
                ),
                None,
            )
    if selected_cross_lane is None and len(cross_lane_rows) == 1:
        selected_cross_lane = cross_lane_rows[0]
    if len(candidate_rows) > 1 or context_ambiguous:
        stages.append(
            MaterializationStagePlan(
                stage_id="production_facing_candidate_portfolio",
                state="AMBIGUOUS_LINEAGE",
                selected_artifacts=[
                    _artifact_ref(row, artifact_kind="production_facing_candidate_portfolio", object_id_key="portfolio_id")
                    for row in candidate_rows
                ],
                ignored_stale_or_mismatched_artifacts=candidate_ignored,
                next_action="resolve_candidate_lineage_before_generation",
                expected_llm_calls=0,
            )
        )
    elif candidate is not None:
        stages.append(
            MaterializationStagePlan(
                stage_id="production_facing_candidate_portfolio",
                state="REUSE_MATCHED",
                selected_artifacts=[
                    _artifact_ref(
                        candidate,
                        artifact_kind="production_facing_candidate_portfolio",
                        object_id_key="portfolio_id",
                        source_object_id_key="source_cross_lane_portfolio_id",
                    )
                ],
                ignored_stale_or_mismatched_artifacts=candidate_ignored,
                next_action="reuse_exact_lineage_candidate_portfolio",
                expected_llm_calls=0,
            )
        )
    else:
        prereqs, missing = _upstream_candidate_prerequisites(
            rows,
            context_id=context_id,
            cross_lane=selected_cross_lane,
        )
        if context is None:
            missing = ["hypothesis_context", *missing]
        if missing:
            stages.append(
                MaterializationStagePlan(
                    stage_id="production_facing_candidate_portfolio",
                    state="BLOCKED_UPSTREAM",
                    selected_artifacts=prereqs,
                    ignored_stale_or_mismatched_artifacts=candidate_ignored,
                    missing_prerequisites=sorted(dict.fromkeys(missing)),
                    next_action="materialize_missing_upstream_reframing_or_cross_lane_artifacts",
                    expected_llm_calls=0,
                    notes=[
                        "Do not synthesize a replacement candidate merely to satisfy verifier wiring."
                    ],
                )
            )
        else:
            stages.append(
                MaterializationStagePlan(
                    stage_id="production_facing_candidate_portfolio",
                    state="NEEDS_DETERMINISTIC_MATERIALIZATION",
                    selected_artifacts=prereqs,
                    ignored_stale_or_mismatched_artifacts=candidate_ignored,
                    next_action="run_build_production_facing_scientific_candidate_portfolio",
                    expected_llm_calls=0,
                    notes=["This stage is deterministic and must not rank or prune candidates."],
                )
            )

    atomic_report: _JsonArtifact | None = None
    atomic_portfolio: _JsonArtifact | None = None
    if candidate is None:
        stages.append(
            MaterializationStagePlan(
                stage_id="atomic_cross_lane_synthesis",
                state="BLOCKED_UPSTREAM",
                missing_prerequisites=["production_facing_candidate_portfolio"],
                next_action="wait_for_candidate_portfolio",
                expected_llm_calls=0,
            )
        )
    else:
        candidate_id = _string(candidate.payload, "portfolio_id") or ""
        report_rows, report_ignored = _matching_atomic_reports(
            rows,
            candidate_portfolio_id=candidate_id,
        )
        if len(report_rows) > 1:
            stages.append(
                MaterializationStagePlan(
                    stage_id="atomic_cross_lane_synthesis",
                    state="AMBIGUOUS_LINEAGE",
                    selected_artifacts=[
                        _artifact_ref(
                            row,
                            artifact_kind="atomic_synthesis_report",
                            object_id_key="report_id",
                            source_object_id_key="source_candidate_portfolio_id",
                            hypothesis_ids=_hypothesis_ids_from_atomic_report(row.payload),
                        )
                        for row in report_rows
                    ],
                    ignored_stale_or_mismatched_artifacts=report_ignored,
                    next_action="resolve_atomic_report_lineage_before_reuse",
                    expected_llm_calls=0,
                )
            )
        elif len(report_rows) == 1:
            atomic_report = report_rows[0]
            hypothesis_ids = _hypothesis_ids_from_atomic_report(atomic_report.payload)
            source_context_id = _string(atomic_report.payload, "source_context_id")
            portfolio_rows, portfolio_ignored = _matching_atomic_portfolios(
                rows,
                hypothesis_ids=hypothesis_ids,
                source_context_id=source_context_id,
            )
            if len(portfolio_rows) == 1:
                atomic_portfolio = portfolio_rows[0]
                stages.append(
                    MaterializationStagePlan(
                        stage_id="atomic_cross_lane_synthesis",
                        state="REUSE_MATCHED",
                        selected_artifacts=[
                            _artifact_ref(
                                atomic_report,
                                artifact_kind="atomic_synthesis_report",
                                object_id_key="report_id",
                                source_object_id_key="source_candidate_portfolio_id",
                                hypothesis_ids=hypothesis_ids,
                            ),
                            _artifact_ref(
                                atomic_portfolio,
                                artifact_kind="atomic_hypothesis_portfolio",
                                object_id_key="portfolio_id",
                                hypothesis_ids=hypothesis_ids,
                            ),
                        ],
                        ignored_stale_or_mismatched_artifacts=sorted(report_ignored + portfolio_ignored),
                        next_action="reuse_exact_atomic_report_and_portfolio_pair",
                        expected_llm_calls=0,
                    )
                )
            elif len(portfolio_rows) > 1:
                stages.append(
                    MaterializationStagePlan(
                        stage_id="atomic_cross_lane_synthesis",
                        state="AMBIGUOUS_LINEAGE",
                        selected_artifacts=[
                            _artifact_ref(
                                atomic_report,
                                artifact_kind="atomic_synthesis_report",
                                object_id_key="report_id",
                                hypothesis_ids=hypothesis_ids,
                            ),
                            *[
                                _artifact_ref(
                                    row,
                                    artifact_kind="atomic_hypothesis_portfolio",
                                    object_id_key="portfolio_id",
                                    hypothesis_ids=hypothesis_ids,
                                )
                                for row in portfolio_rows
                            ],
                        ],
                        ignored_stale_or_mismatched_artifacts=sorted(report_ignored + portfolio_ignored),
                        next_action="resolve_atomic_portfolio_lineage_before_reuse",
                        expected_llm_calls=0,
                    )
                )
            else:
                stages.append(
                    MaterializationStagePlan(
                        stage_id="atomic_cross_lane_synthesis",
                        state="NEEDS_LLM_MATERIALIZATION",
                        selected_artifacts=[
                            _artifact_ref(
                                atomic_report,
                                artifact_kind="orphaned_atomic_synthesis_report",
                                object_id_key="report_id",
                                hypothesis_ids=hypothesis_ids,
                            )
                        ],
                        ignored_stale_or_mismatched_artifacts=sorted(report_ignored + portfolio_ignored),
                        missing_prerequisites=["matching_atomic_hypothesis_portfolio"],
                        next_action="restore_matching_atomic_portfolio_or_rerun_atomic_synthesis",
                        expected_llm_calls=1,
                        notes=["Existing report alone is insufficient verifier lineage authority."],
                    )
                )
        else:
            missing = [] if context is not None else ["hypothesis_context"]
            stages.append(
                MaterializationStagePlan(
                    stage_id="atomic_cross_lane_synthesis",
                    state="BLOCKED_UPSTREAM" if missing else "NEEDS_LLM_MATERIALIZATION",
                    selected_artifacts=[
                        _artifact_ref(candidate, artifact_kind="production_facing_candidate_portfolio", object_id_key="portfolio_id")
                    ],
                    ignored_stale_or_mismatched_artifacts=report_ignored,
                    missing_prerequisites=missing,
                    next_action=(
                        "restore_hypothesis_context"
                        if missing
                        else "run_atomic_cross_lane_scientific_synthesis"
                    ),
                    expected_llm_calls=0 if missing else 1,
                )
            )

    if candidate is None or atomic_report is None:
        stages.append(
            MaterializationStagePlan(
                stage_id="grounded_identity_annotation",
                state="BLOCKED_UPSTREAM",
                missing_prerequisites=[
                    name
                    for name, present in (
                        ("production_facing_candidate_portfolio", candidate is not None),
                        ("atomic_synthesis_report", atomic_report is not None),
                    )
                    if not present
                ],
                next_action="wait_for_atomic_lineage",
                expected_llm_calls=0,
            )
        )
    else:
        candidate_id = _string(candidate.payload, "portfolio_id") or ""
        report_id = _string(atomic_report.payload, "report_id") or ""
        annotation_rows, annotation_ignored = _matching_grounded_identity(
            rows,
            candidate_portfolio_id=candidate_id,
            atomic_report_id=report_id,
        )
        if len(annotation_rows) == 1:
            stages.append(
                MaterializationStagePlan(
                    stage_id="grounded_identity_annotation",
                    state="REUSE_MATCHED",
                    selected_artifacts=[
                        _artifact_ref(
                            annotation_rows[0],
                            artifact_kind="grounded_identity_annotation",
                            source_object_id_key="source_atomic_synthesis_report_id",
                        )
                    ],
                    ignored_stale_or_mismatched_artifacts=annotation_ignored,
                    next_action="reuse_exact_grounded_identity_annotation",
                    expected_llm_calls=0,
                )
            )
        elif len(annotation_rows) > 1:
            stages.append(
                MaterializationStagePlan(
                    stage_id="grounded_identity_annotation",
                    state="AMBIGUOUS_LINEAGE",
                    selected_artifacts=[
                        _artifact_ref(row, artifact_kind="grounded_identity_annotation", source_object_id_key="source_atomic_synthesis_report_id")
                        for row in annotation_rows
                    ],
                    ignored_stale_or_mismatched_artifacts=annotation_ignored,
                    next_action="resolve_grounded_identity_lineage_before_reuse",
                    expected_llm_calls=0,
                )
            )
        else:
            stages.append(
                MaterializationStagePlan(
                    stage_id="grounded_identity_annotation",
                    state="NEEDS_LLM_MATERIALIZATION",
                    selected_artifacts=[
                        _artifact_ref(candidate, artifact_kind="production_facing_candidate_portfolio", object_id_key="portfolio_id"),
                        _artifact_ref(atomic_report, artifact_kind="atomic_synthesis_report", object_id_key="report_id"),
                    ],
                    ignored_stale_or_mismatched_artifacts=annotation_ignored,
                    next_action="materialize_grounded_identity_annotation_from_exact_atomic_lineage",
                    expected_llm_calls=1,
                    notes=["Grounded annotation is provenance support, not novelty authority."],
                )
            )

    if atomic_portfolio is None:
        stages.append(
            MaterializationStagePlan(
                stage_id="external_novelty_report",
                state="BLOCKED_UPSTREAM",
                missing_prerequisites=["matching_atomic_hypothesis_portfolio"],
                next_action="wait_for_atomic_portfolio",
                expected_llm_calls=0,
            )
        )
    else:
        atomic_portfolio_id = _string(atomic_portfolio.payload, "portfolio_id") or ""
        hypothesis_ids = _hypothesis_ids_from_portfolio(atomic_portfolio.payload)
        external_rows, external_ignored = _matching_external_reports(
            rows,
            atomic_portfolio_id=atomic_portfolio_id,
            hypothesis_ids=hypothesis_ids,
        )
        if len(external_rows) == 1:
            stages.append(
                MaterializationStagePlan(
                    stage_id="external_novelty_report",
                    state="REUSE_MATCHED",
                    selected_artifacts=[
                        _artifact_ref(
                            external_rows[0],
                            artifact_kind="external_novelty_report",
                            object_id_key="report_id",
                            source_object_id_key="source_portfolio_id",
                            hypothesis_ids=hypothesis_ids,
                        )
                    ],
                    ignored_stale_or_mismatched_artifacts=external_ignored,
                    next_action="reuse_exact_external_report",
                    expected_llm_calls=0,
                    expected_retrieval_runs=0,
                )
            )
        elif len(external_rows) > 1:
            stages.append(
                MaterializationStagePlan(
                    stage_id="external_novelty_report",
                    state="AMBIGUOUS_LINEAGE",
                    selected_artifacts=[
                        _artifact_ref(
                            row,
                            artifact_kind="external_novelty_report",
                            object_id_key="report_id",
                            source_object_id_key="source_portfolio_id",
                            hypothesis_ids=hypothesis_ids,
                        )
                        for row in external_rows
                    ],
                    ignored_stale_or_mismatched_artifacts=external_ignored,
                    next_action="resolve_external_report_lineage_before_reuse",
                    expected_llm_calls=0,
                )
            )
        else:
            stages.append(
                MaterializationStagePlan(
                    stage_id="external_novelty_report",
                    state="NEEDS_RETRIEVAL_AND_REVIEW",
                    selected_artifacts=[
                        _artifact_ref(
                            atomic_portfolio,
                            artifact_kind="atomic_hypothesis_portfolio",
                            object_id_key="portfolio_id",
                            hypothesis_ids=hypothesis_ids,
                        )
                    ],
                    ignored_stale_or_mismatched_artifacts=external_ignored,
                    next_action="run_fresh_external_novelty_for_exact_atomic_portfolio",
                    expected_llm_calls=None,
                    expected_retrieval_runs=1,
                    runtime_dependent_llm_calls=True,
                    notes=[
                        "External review LLM call count is claim/review dependent; the planner does not invent an exact count.",
                        "A report is reusable only when source_portfolio_id and hypothesis ID set both match exactly.",
                    ],
                )
            )

    ordered_stage_ids = (
        "production_facing_candidate_portfolio",
        "atomic_cross_lane_synthesis",
        "grounded_identity_annotation",
        "external_novelty_report",
    )
    ordered_stages = [
        next(stage for stage in stages if stage.stage_id == stage_id)
        for stage_id in ordered_stage_ids
    ]
    ready = all(stage.state == "REUSE_MATCHED" for stage in ordered_stages)

    if ready:
        status: TaskStatus = "READY_FOR_VERIFIER"
    else:
        # Only the first unresolved lineage stage is authoritative for the task-level
        # disposition. Later stages may be BLOCKED_UPSTREAM merely because their exact
        # generated IDs do not exist yet; that is expected in an iterative planner.
        first_unresolved = next(
            stage for stage in ordered_stages if stage.state != "REUSE_MATCHED"
        )
        if first_unresolved.state == "AMBIGUOUS_LINEAGE":
            status = "AMBIGUOUS_LINEAGE"
        elif first_unresolved.state == "BLOCKED_UPSTREAM":
            status = "BLOCKED_UPSTREAM"
        else:
            status = "MATERIALIZATION_REQUIRED"

    exact_llm = sum(stage.expected_llm_calls or 0 for stage in stages)
    retrieval_runs = sum(stage.expected_retrieval_runs for stage in stages)
    runtime_ids = [stage.stage_id for stage in stages if stage.runtime_dependent_llm_calls]

    return ScientificVerifierMaterializationTask(
        task_key=task_key,
        case_key=case_key,
        replicate_key=replicate_key,
        canonical_dir=str(canonical_dir),
        status=status,
        stages=stages,
        verifier_required_artifacts_present=ready,
        planned_minimum_llm_calls=exact_llm,
        planned_retrieval_runs=retrieval_runs,
        runtime_dependent_llm_stage_ids=runtime_ids,
        diagnostic_notes=diagnostic_notes,
    )


def build_scientific_verifier_materialization_plan(
    *,
    benchmark_root: Path,
    case_keys: list[str],
    exclude_case_keys: list[str] | None = None,
) -> ScientificVerifierMaterializationPlan:
    root = benchmark_root.expanduser().resolve()
    requested = list(dict.fromkeys(case_keys))
    excluded = set(exclude_case_keys or [])
    unknown_exclusions = sorted(excluded - set(requested))
    if unknown_exclusions:
        raise ValueError(
            "excluded cases must be part of requested cohort: "
            + ", ".join(unknown_exclusions)
        )

    selected = [case for case in requested if case not in excluded]
    tasks: list[ScientificVerifierMaterializationTask] = []

    for case_key in requested:
        if case_key in excluded:
            tasks.append(_excluded_task(root=root, case_key=case_key))
            continue
        case_root = root / case_key
        canonical_dirs = sorted(case_root.glob("replicate_*/canonical"))
        if not canonical_dirs:
            canonical_dirs = [case_root / "replicate_01" / "canonical"]
        for canonical_dir in canonical_dirs:
            tasks.append(
                _plan_task(
                    root=root,
                    case_key=case_key,
                    canonical_dir=canonical_dir,
                )
            )

    tasks.sort(key=lambda row: row.task_key)
    counts = Counter(row.status for row in tasks)
    id_payload = {
        "benchmark_root": str(root),
        "requested_case_keys": requested,
        "excluded_case_keys": sorted(excluded),
        "selected_case_keys": selected,
        "tasks": [row.model_dump(mode="json") for row in tasks],
    }
    return ScientificVerifierMaterializationPlan(
        plan_id=_stable_id("scientific_verifier_materialization_plan", id_payload),
        benchmark_root=str(root),
        requested_case_keys=requested,
        excluded_case_keys=sorted(excluded),
        selected_case_keys=selected,
        tasks=tasks,
        task_count=len(tasks),
        status_counts=dict(sorted(counts.items())),
    )


__all__ = [
    "ArtifactRef",
    "MaterializationStagePlan",
    "ScientificVerifierMaterializationPlan",
    "ScientificVerifierMaterializationTask",
    "build_scientific_verifier_materialization_plan",
]
