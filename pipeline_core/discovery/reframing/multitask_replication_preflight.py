from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ArtifactState = Literal["present", "absent"]
ReplicationStatus = Literal[
    "blocked_missing_base",
    "ready_for_reframe_generation",
    "ready_to_extend_reframing",
    "ready_for_cross_lane_assembly",
    "ready_for_ablation_build",
    "ready_for_blind_evaluation",
    "replication_evaluation_materialized",
]


class ReplicationArtifact(StrictModel):
    artifact_key: str
    state: ArtifactState
    path: str | None = None
    schema_version: str | None = None
    object_count: int | None = Field(default=None, ge=0)
    decision: str | None = None


class ReplicationTaskPreflight(StrictModel):
    task_key: str
    case_key: str
    replicate_key: str
    canonical_dir: str
    status: ReplicationStatus
    artifacts: list[ReplicationArtifact]
    missing_base_artifacts: list[str] = Field(default_factory=list)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    next_actions: list[str] = Field(default_factory=list)
    diagnostic_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_artifacts(self) -> "ReplicationTaskPreflight":
        keys = [row.artifact_key for row in self.artifacts]
        if len(keys) != len(set(keys)):
            raise ValueError("replication preflight artifact keys must be unique")
        if self.status == "blocked_missing_base" and not self.missing_base_artifacts:
            raise ValueError("blocked tasks require missing_base_artifacts")
        return self


class ScientificReasoningMultitaskReplicationPreflight(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-multitask-replication-preflight-v1"
    ] = "scientific-reasoning-multitask-replication-preflight-v1"

    preflight_id: str
    benchmark_root: str
    requested_case_keys: list[str]
    tasks: list[ReplicationTaskPreflight]
    task_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)

    llm_calls_performed: Literal[0] = 0
    filesystem_inventory_only: Literal[True] = True
    scientific_quality_judgment_performed: Literal[False] = False
    scientific_task_ranking_performed: Literal[False] = False
    replication_success_established: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReasoningMultitaskReplicationPreflight":
        if self.task_count != len(self.tasks):
            raise ValueError("task_count mismatch")
        keys = [row.task_key for row in self.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate task_key in replication preflight")
        counts = Counter(row.status for row in self.tasks)
        if dict(sorted(counts.items())) != dict(sorted(self.status_counts.items())):
            raise ValueError("status_counts mismatch")
        return self


_ARTIFACT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("explorer_packet", ("explorer.packet.json",)),
    ("hypothesis_context", ("hypothesis.context.json",)),
    ("relational_portfolio", ("hypothesis_axis_a4.portfolio.json",)),
    ("reframing_trigger", ("scientific_reframing_triggers.json",)),
    ("reframing_evidence", ("scientific_reframing_evidence.json",)),
    ("reframing_tensions", ("scientific_reframing_evidence_tensions.json",)),
    ("operator_readiness", ("reframing_operator_readiness.json",)),
    (
        "reframing_shadow",
        (
            "scientific_reframing_shadow.json",
            "scientific_reframing_shadow_s38.json",
        ),
    ),
    ("proxy_enrichment_report", ("proxy_semantic_enrichment_report_v1_2.json",)),
    ("proxy_annotations", ("annotations/proxy_semantics_v1_2.jsonl",)),
    ("proxy_shadow", ("scientific_proxy_challenge_shadow.json",)),
    (
        "contradiction_shadow",
        ("scientific_contradiction_resolution_shadow.json",),
    ),
    ("mode_contrast", ("scientific_reframing_mode_contrast.json",)),
    (
        "reframing_portfolio",
        ("scientific_reframing_reasoning_portfolio.json",),
    ),
    (
        "cross_lane_portfolio",
        ("scientific_cross_lane_reasoning_portfolio.json",),
    ),
    ("ablation_packet", ("scientific_reasoning_ablation_packet.json",)),
    (
        "schema_normalized_packet",
        ("scientific_reasoning_ablation_schema_normalized_packet.json",),
    ),
    (
        "matched_count_packet",
        (
            "scientific_reasoning_ablation_generalized_matched_packet.json",
            "scientific_reasoning_ablation_matched_count_packet.json",
        ),
    ),
    (
        "matched_count_blind_evaluation",
        (
            "scientific_reasoning_ablation_generalized_matched_blind_evaluation.json",
            "scientific_reasoning_ablation_matched_count_blind_evaluation.json",
        ),
    ),
    (
        "matched_count_unblinded_evaluation",
        (
            "scientific_reasoning_ablation_generalized_matched_unblinded_evaluation.json",
            "scientific_reasoning_ablation_matched_count_unblinded_evaluation.json",
        ),
    ),
)

_BASE_REQUIRED = (
    "explorer_packet",
    "hypothesis_context",
    "relational_portfolio",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _candidate_count(payload: dict[str, object] | None) -> int | None:
    if payload is None:
        return None
    for key in ("candidate_count", "entry_count", "hypothesis_count"):
        value = payload.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    for key in ("candidates", "hypotheses", "entries"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    candidate_ids = payload.get("candidate_ids")
    if isinstance(candidate_ids, list):
        return len(candidate_ids)
    return None


def _resolve_artifact(canonical_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = canonical_dir / name
        if path.is_file():
            return path
    if names and names[0] == "scientific_reframing_shadow.json":
        matches = sorted(canonical_dir.glob("scientific_reframing_shadow*.json"))
        if matches:
            return matches[-1]
    return None


def _artifact(canonical_dir: Path, key: str, names: tuple[str, ...]) -> ReplicationArtifact:
    path = _resolve_artifact(canonical_dir, names)
    if path is None:
        return ReplicationArtifact(artifact_key=key, state="absent")
    payload = _load_json(path) if path.suffix == ".json" else None
    schema_version = None
    decision = None
    if payload is not None:
        if isinstance(payload.get("schema_version"), str):
            schema_version = str(payload["schema_version"])
        if isinstance(payload.get("decision"), str):
            decision = str(payload["decision"])
    return ReplicationArtifact(
        artifact_key=key,
        state="present",
        path=str(path),
        schema_version=schema_version,
        object_count=_candidate_count(payload),
        decision=decision,
    )


def _status_and_actions(
    artifacts: dict[str, ReplicationArtifact],
) -> tuple[ReplicationStatus, list[str], list[str], list[str]]:
    missing_base = [
        key for key in _BASE_REQUIRED if artifacts[key].state != "present"
    ]
    actions: list[str] = []
    notes: list[str] = []
    if missing_base:
        return (
            "blocked_missing_base",
            ["restore_or_regenerate_missing_base_artifacts"],
            missing_base,
            ["No replication inference should be run until base lineage artifacts are present."],
        )

    if artifacts["reframing_shadow"].state != "present":
        actions.append("run_scientific_reframing_shadow_if_trigger_eligible")
        if artifacts["reframing_trigger"].state != "present":
            actions.append("materialize_or_regenerate_reframing_trigger")
        return "ready_for_reframe_generation", actions, [], notes

    optional_modes = []
    if artifacts["proxy_shadow"].state == "present":
        optional_modes.append("proxy")
    if artifacts["contradiction_shadow"].state == "present":
        optional_modes.append("contradiction")
    if artifacts["mode_contrast"].state != "present":
        actions.append("build_reasoning_mode_contrast_from_available_operator_outputs")
    if artifacts["reframing_portfolio"].state != "present":
        actions.append("build_unified_reframing_portfolio")

    if artifacts["proxy_shadow"].state != "present":
        notes.append(
            "PROXY_CHALLENGE is optional for replication; do not backfill or generate it merely to satisfy downstream wiring."
        )
    if artifacts["contradiction_shadow"].state != "present":
        notes.append(
            "CONTRADICTION_RESOLUTION is optional for replication; absence is a valid sparse-operator portfolio state."
        )
    if optional_modes:
        notes.append("Available optional operator outputs: " + ", ".join(optional_modes))

    if artifacts["reframing_portfolio"].state != "present":
        return "ready_to_extend_reframing", actions, [], notes

    if artifacts["cross_lane_portfolio"].state != "present":
        actions.append("build_cross_lane_reasoning_portfolio")
        return "ready_for_cross_lane_assembly", actions, [], notes

    if artifacts["schema_normalized_packet"].state != "present":
        if artifacts["ablation_packet"].state != "present":
            actions.append("build_scientific_reasoning_ablation_packet")
        actions.append("build_schema_normalized_ablation_packet")
        return "ready_for_ablation_build", actions, [], notes

    if artifacts["matched_count_packet"].state != "present":
        actions.append("build_generalized_matched_count_ablation")
        notes.append(
            "Generalized matching uses min(relational_count, reframing_count) and deterministic subset scheduling; no K03-specific cardinality is required."
        )
        return "ready_for_blind_evaluation", actions, [], notes

    if artifacts["matched_count_unblinded_evaluation"].state == "present":
        return "replication_evaluation_materialized", [], [], notes

    actions.append("run_blind_evaluator_then_unblind")
    return "ready_for_blind_evaluation", actions, [], notes


def _task_key(benchmark_root: Path, canonical_dir: Path) -> tuple[str, str, str]:
    rel = canonical_dir.relative_to(benchmark_root)
    parts = rel.parts
    if len(parts) < 3 or parts[-1] != "canonical":
        raise ValueError(f"unexpected canonical path below benchmark root: {canonical_dir}")
    case_key = parts[0]
    replicate_key = parts[1]
    return f"{case_key}/{replicate_key}", case_key, replicate_key


def build_multitask_replication_preflight(
    *,
    benchmark_root: Path,
    case_keys: list[str],
) -> ScientificReasoningMultitaskReplicationPreflight:
    root = benchmark_root.resolve()
    requested = list(dict.fromkeys(case_keys))
    tasks: list[ReplicationTaskPreflight] = []

    for case_key in requested:
        case_root = root / case_key
        canonical_dirs = sorted(case_root.glob("replicate_*/canonical"))
        if not canonical_dirs:
            canonical_dirs = [case_root / "replicate_01" / "canonical"]
        for canonical_dir in canonical_dirs:
            task_key, resolved_case, replicate_key = _task_key(root, canonical_dir)
            rows = [
                _artifact(canonical_dir, key, names)
                for key, names in _ARTIFACT_SPECS
            ]
            by_key = {row.artifact_key: row for row in rows}
            status, actions, missing_base, notes = _status_and_actions(by_key)
            candidate_counts = {
                key: row.object_count
                for key, row in by_key.items()
                if row.object_count is not None
                and key
                in {
                    "relational_portfolio",
                    "reframing_shadow",
                    "proxy_shadow",
                    "contradiction_shadow",
                    "reframing_portfolio",
                    "cross_lane_portfolio",
                }
            }
            tasks.append(
                ReplicationTaskPreflight(
                    task_key=task_key,
                    case_key=resolved_case,
                    replicate_key=replicate_key,
                    canonical_dir=str(canonical_dir),
                    status=status,
                    artifacts=rows,
                    missing_base_artifacts=missing_base,
                    candidate_counts=candidate_counts,
                    next_actions=actions,
                    diagnostic_notes=notes,
                )
            )

    tasks.sort(key=lambda row: row.task_key)
    status_counts = Counter(row.status for row in tasks)
    id_payload = {
        "benchmark_root": str(root),
        "requested_case_keys": requested,
        "tasks": [row.model_dump(mode="json") for row in tasks],
    }
    return ScientificReasoningMultitaskReplicationPreflight(
        preflight_id=_stable_id("scientific_reasoning_replication_preflight", id_payload),
        benchmark_root=str(root),
        requested_case_keys=requested,
        tasks=tasks,
        task_count=len(tasks),
        status_counts=dict(sorted(status_counts.items())),
    )


__all__ = [
    "ReplicationArtifact",
    "ReplicationTaskPreflight",
    "ScientificReasoningMultitaskReplicationPreflight",
    "build_multitask_replication_preflight",
]
