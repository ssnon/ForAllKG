from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


class ClosedGenerationCycleReport(StrictModel):
    schema_version: Literal[
        "sis-v3-0-closed-generation-cycle-shadow-v1"
    ] = "sis-v3-0-closed-generation-cycle-shadow-v1"
    report_id: str
    report_sha256: str

    current_generation_index: int = Field(ge=2)
    source_generation_execution_report_id: str
    source_context_id: str
    source_context_sha256: str

    population_idea_count: int = Field(ge=0)
    realization_target_count: int = Field(ge=0)
    realization_attempt_count: int = Field(ge=0)
    materialized_hypothesis_count: int = Field(ge=0)
    usable_grounded_realization_count: int = Field(ge=0)
    same_idea_rescue_count: int = Field(ge=0)
    realization_llm_call_count: int = Field(ge=0)
    prospective_audit_llm_call_count: int = Field(ge=0)

    epistemic_realization_count: int = Field(ge=0)
    epistemic_maturity_counts: dict[str, int] = Field(default_factory=dict)
    retained_archive_entry_count: int = Field(ge=0)
    active_parallel_member_count: int = Field(ge=0)
    active_idea_count: int = Field(ge=0)
    epistemic_debt_count: int = Field(ge=0)
    next_generation_handoff_count: int = Field(ge=0)

    next_generation_index: int = Field(ge=3)
    next_generation_plan_id: str | None = None
    next_generation_selected_parent_count: int = Field(ge=0)
    next_generation_execution_report_id: str | None = None
    next_generation_raw_offspring_count: int = Field(ge=0)
    next_generation_population_count: int = Field(ge=0)
    next_generation_genuine_child_count: int = Field(ge=0)

    same_cycle_runner_can_consume_next_execution: bool
    feedback_loop_closed: Literal[True] = True
    current_generation_grounding_does_not_control_reproduction: Literal[True] = True
    evidence_acquisition_is_not_blocking_inner_loop: Literal[True] = True
    strict_hypothesis_card_contract_preserved: Literal[True] = True
    generated_next_ideas_remain_inspiration_only: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_generation_step(self) -> "ClosedGenerationCycleReport":
        if self.next_generation_index != self.current_generation_index + 1:
            raise ValueError("next_generation_index must equal current_generation_index + 1")
        expected_ready = bool(
            self.next_generation_execution_report_id
            and self.next_generation_population_count > 0
        )
        if self.same_cycle_runner_can_consume_next_execution != expected_ready:
            raise ValueError("next-execution readiness / non-empty population mismatch")
        return self


@dataclass(frozen=True)
class ClosedGenerationCycleArtifacts:
    lifecycle: Any
    portfolio: Any
    decomposition: Any
    archive: Any
    parallel: Any
    next_plan: Any | None
    next_execution: Any | None
    next_prompts: tuple[Any, ...]
    report: ClosedGenerationCycleReport


def population_nodes(execution: Any) -> list[Any]:
    """Return retained generation nodes from either v2.9 or later generalized output."""
    rows = list(_get(execution, "g4_population_nodes", []) or [])
    if rows:
        return rows
    semantic_by_id = {
        str(_get(row, "idea_id")): row
        for row in (_get(execution, "semantic_records", []) or [])
    }
    return [
        row
        for row in (_get(execution, "offspring_nodes", []) or [])
        if bool(_get(semantic_by_id.get(str(_get(row, "idea_id"))), "retained_in_g4_population", False))
    ]


def build_legacy_offspring_execution_payload(execution: Any) -> dict[str, Any]:
    """Project a v2.9+ epistemic generation into the stable realization lifecycle contract.

    This adapter is intentionally one-way. It does not grant the generated ideas any
    new evidence authority; it only lets the already-existing strict realization
    lifecycle attempt grounded HypothesisCards for retained generation members.
    """
    generation_index = int(_get(execution, "generation_index", 4))
    nodes = population_nodes(execution)
    retained_ids = {str(_get(row, "idea_id")) for row in nodes}
    source_semantics = [
        row
        for row in (_get(execution, "semantic_records", []) or [])
        if str(_get(row, "idea_id")) in retained_ids
    ]
    source_report_id = str(_get(execution, "report_id"))

    semantic_rows: list[dict[str, Any]] = []
    for row in source_semantics:
        disposition = str(_get(row, "disposition"))
        legacy_disposition = {
            "GENUINE_CHILD": "ACCEPTED_CHILD",
            "INDETERMINATE_PROBE": "ACCEPTED_INDETERMINATE_PROBE",
        }.get(disposition)
        if legacy_disposition is None:
            continue
        semantic_rows.append(
            {
                "idea_id": str(_get(row, "idea_id")),
                "task_id": str(_get(row, "task_id")),
                "channel": str(_get(row, "channel")),
                "chosen_operator_id": str(_get(row, "chosen_operator_id")),
                "parent_idea_ids": list(_get(row, "parent_idea_ids", []) or []),
                "transition": _dump(_get(row, "transition", {})),
                "disposition": legacy_disposition,
                "accepted_for_realization": True,
                "conceptual_change_summary": str(
                    _get(row, "conceptual_change_summary", "")
                ),
                "diagnostic_codes": list(_get(row, "diagnostic_codes", []) or []),
            }
        )

    # SIS-v3.1 persistence may carry an ACTIVE ResearchIdea forward without
    # generating a child. Such a node has no current-generation offspring semantic
    # record, but it is still a legitimate inspiration-only ResearchIdea that may be
    # re-realized under the strict HypothesisCard boundary. Synthesize only the
    # realization-facing EXPLOIT adapter record; this does not alter idea identity or
    # grant positive-premise authority.
    accepted_ids = {row["idea_id"] for row in semantic_rows}
    carried_ids = set(str(value) for value in (_get(execution, "carried_forward_idea_ids", []) or []))
    for node in nodes:
        idea_id = str(_get(node, "idea_id"))
        if idea_id in accepted_ids or idea_id not in carried_ids:
            continue
        semantic_rows.append(
            {
                "idea_id": idea_id,
                "task_id": _stable_id(
                    f"g{generation_index}_persistent_realization_task",
                    source_report_id,
                    idea_id,
                ),
                "channel": "EXPLOIT",
                "chosen_operator_id": "SAME_PREMISE_SHARPEN",
                "parent_idea_ids": [idea_id],
                "transition": {
                    "schema_version": "idea-transition-assessment-v1",
                    "proposed_idea_id": idea_id,
                    "parent_idea_ids": [idea_id],
                    "parent_comparisons": [
                        {
                            "parent_idea_id": idea_id,
                            "identity_relation": "SAME_IDEA",
                            "facet_assessments": [
                                {
                                    "facet": facet,
                                    "relation": "PRESERVED",
                                    "similarity": 1.0,
                                    "rationale": (
                                        "Persistence adapter compares the carried ResearchIdea "
                                        "with itself only to enter the existing realization lane."
                                    ),
                                }
                                for facet in (
                                    "CORE_COMMITMENTS",
                                    "SCOPE",
                                    "CONTRAST",
                                    "QUESTION",
                                )
                            ],
                            "family_assessment": {
                                "idea_id_a": idea_id,
                                "idea_id_b": idea_id,
                                "relation": "SAME_FAMILY",
                                "similarity": 1.0,
                                "confidence": 1.0,
                            },
                        }
                    ],
                    "identity_relation": "SAME_IDEA",
                    "genealogy_relation": "REFINEMENT_OF",
                    "operator_id": "SAME_PREMISE_SHARPEN",
                    "operator_expectation_consistent": True,
                    "diagnostic_codes": [
                        "PERSISTENT_RESEARCH_IDEA_CARRY_FORWARD_REALIZATION_ADAPTER"
                    ],
                },
                "disposition": "ACCEPTED_REFINEMENT",
                "accepted_for_realization": True,
                "conceptual_change_summary": (
                    "Persistent ACTIVE ResearchIdea carried forward without child "
                    "generation; strict grounded realization may be attempted again."
                ),
                "diagnostic_codes": [
                    "ACTIVE_IDEA_PERSISTED_WITHOUT_REPRODUCTION",
                    "POSITIVE_PREMISE_AUTHORITY_REMAINS_FALSE",
                ],
            }
        )
        accepted_ids.add(idea_id)

    nodes = [row for row in nodes if str(_get(row, "idea_id")) in accepted_ids]
    source_identity_by_id = {
        str(_get(row, "idea_id")): str(_get(row, "identity_relation"))
        for row in source_semantics
    }
    identity_counts = Counter(
        source_identity_by_id.get(row["idea_id"], "SAME_IDEA")
        for row in semantic_rows
    )
    disposition_counts = Counter(row["disposition"] for row in semantic_rows)
    channel_counts = Counter(row["channel"] for row in semantic_rows)
    operator_counts = Counter(row["chosen_operator_id"] for row in semantic_rows)
    mutation_attempt_count = len(semantic_rows)
    distinct_child_count = identity_counts.get("DIFFERENT_IDEA", 0)

    source_parallel_id = str(_get(execution, "source_parallel_report_id", source_report_id))
    return {
        "report_id": _stable_id(
            f"g{generation_index}_closed_cycle_realization_adapter",
            source_report_id,
            sorted(retained_ids),
        ),
        "report_sha256": _sha(
            {
                "source_generation_execution_report_id": source_report_id,
                "retained_idea_ids": sorted(retained_ids),
            }
        ),
        "source_plan_id": str(_get(execution, "source_plan_id", source_report_id)),
        "source_v2_2_report_id": source_parallel_id,
        "source_generational_report_id": source_parallel_id,
        "generation_index": generation_index,
        "generation_tasks": [],
        "run_records": [],
        "offspring_nodes": [_dump(row) for row in nodes],
        "semantic_records": semantic_rows,
        "raw_offspring_count": len(nodes),
        "accepted_for_realization_count": len(semantic_rows),
        "identity_relation_counts": dict(sorted(identity_counts.items())),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "generated_count_by_channel": dict(sorted(channel_counts.items())),
        "generated_count_by_operator": dict(sorted(operator_counts.items())),
        "semantic_noop_count": 0,
        "mutation_attempt_count": mutation_attempt_count,
        "distinct_child_count": distinct_child_count,
        "mutation_semantic_yield_fraction": (
            distinct_child_count / mutation_attempt_count if mutation_attempt_count else 0.0
        ),
        "indeterminate_probe_count": identity_counts.get("INDETERMINATE", 0),
        "channel_drift_child_count": 0,
        "exact_kernel_duplicate_suppressed_count": 0,
        "llm_call_count": int(_get(execution, "llm_call_count", 0) or 0),
        "input_tokens": int(_get(execution, "input_tokens", 0) or 0),
        "output_tokens": int(_get(execution, "output_tokens", 0) or 0),
        "semantic_retry_count": int(_get(execution, "semantic_retry_count", 0) or 0),
        "offspring_generation_executed": True,
        "conceptual_generation_is_inspiration_only": True,
        "semantic_identity_is_posthoc_diagnostic": True,
        "indeterminate_identity_is_not_automatic_rejection": True,
        "external_prior_art_as_positive_premise": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
        "positive_premise_authority": False,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "stage8_input_changed": False,
        "canonical_graph_mutated": False,
    }


def adapt_epistemic_execution_for_realization(execution: Any) -> Any:
    from pipeline_core.discovery.research_idea_offspring_execution import (
        OffspringExecutionReport,
    )

    return OffspringExecutionReport.model_validate(
        build_legacy_offspring_execution_payload(execution)
    )


def feedback_report_from_lifecycle(lifecycle: Any) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in (_get(lifecycle, "observations", []) or []):
        hypothesis_id = _get(row, "hypothesis_id")
        if hypothesis_id is None:
            continue
        records.append(
            {
                "hypothesis_id": str(hypothesis_id),
                "idea_id": str(_get(row, "idea_id")),
                "prospective_evaluation_state": (
                    "EVALUATED"
                    if _get(row, "prospective_status") == "COMPLETE"
                    else "NOT_EVALUATED"
                ),
                "residual_evaluation_state": "NOT_EVALUATED",
                "current_evidence_status": _get(row, "current_evidence_status"),
                "prospective_identifiability": _get(row, "prospective_identifiability"),
                "directionality_mode": _get(row, "directionality_mode"),
                "measurement_compatibility_mode": _get(
                    row, "measurement_compatibility_mode"
                ),
                "residual_epistemic_state": _get(row, "residual_epistemic_state"),
                "residual_state_reason": _get(row, "residual_state_reason"),
            }
        )
    return {"records": records}


def build_cycle_report(
    *,
    execution: Any,
    lifecycle: Any,
    decomposition: Any,
    archive: Any,
    parallel: Any,
    next_plan: Any | None = None,
    next_execution: Any | None = None,
) -> ClosedGenerationCycleReport:
    current_generation = int(_get(execution, "generation_index", 4))
    next_generation = current_generation + 1
    next_execution_report_id = (
        str(_get(next_execution, "report_id")) if next_execution is not None else None
    )
    provisional = ClosedGenerationCycleReport(
        report_id="pending",
        report_sha256="pending",
        current_generation_index=current_generation,
        source_generation_execution_report_id=str(_get(execution, "report_id")),
        source_context_id=str(_get(lifecycle, "source_context_id")),
        source_context_sha256=str(_get(lifecycle, "source_context_sha256")),
        population_idea_count=len(population_nodes(execution)),
        realization_target_count=len(_get(lifecycle, "target_idea_ids", []) or []),
        realization_attempt_count=len(_get(lifecycle, "links", []) or []),
        materialized_hypothesis_count=int(
            _get(lifecycle, "materialized_hypothesis_count", 0) or 0
        ),
        usable_grounded_realization_count=int(
            _get(lifecycle, "usable_grounded_realization_count", 0) or 0
        ),
        same_idea_rescue_count=int(
            _get(lifecycle, "rescued_within_same_idea_count", 0) or 0
        ),
        realization_llm_call_count=int(
            _get(lifecycle, "realization_llm_call_count", 0) or 0
        ),
        prospective_audit_llm_call_count=int(
            _get(lifecycle, "prospective_audit_llm_call_count", 0) or 0
        ),
        epistemic_realization_count=int(_get(decomposition, "record_count", 0) or 0),
        epistemic_maturity_counts=dict(_get(decomposition, "maturity_counts", {}) or {}),
        retained_archive_entry_count=int(
            _get(archive, "retained_entry_count", 0) or 0
        ),
        active_parallel_member_count=int(
            _get(parallel, "active_member_count", 0) or 0
        ),
        active_idea_count=int(_get(parallel, "active_idea_count", 0) or 0),
        epistemic_debt_count=int(_get(parallel, "epistemic_debt_count", 0) or 0),
        next_generation_handoff_count=int(
            _get(parallel, "evolution_handoff_count", 0) or 0
        ),
        next_generation_index=next_generation,
        next_generation_plan_id=(
            str(_get(next_plan, "plan_id")) if next_plan is not None else None
        ),
        next_generation_selected_parent_count=(
            int(_get(next_plan, "task_count", 0) or 0) if next_plan is not None else 0
        ),
        next_generation_execution_report_id=next_execution_report_id,
        next_generation_raw_offspring_count=(
            int(_get(next_execution, "raw_offspring_count", 0) or 0)
            if next_execution is not None
            else 0
        ),
        next_generation_population_count=(
            int(_get(next_execution, "g4_population_count", 0) or 0)
            if next_execution is not None
            else 0
        ),
        next_generation_genuine_child_count=(
            int(_get(next_execution, "genuine_child_count", 0) or 0)
            if next_execution is not None
            else 0
        ),
        same_cycle_runner_can_consume_next_execution=bool(
            next_execution_report_id
            and int(_get(next_execution, "g4_population_count", 0) or 0) > 0
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"g{current_generation}_closed_cycle:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def build_cycle_epistemic_artifacts(
    *,
    execution: Any,
    lifecycle: Any,
    portfolio: Any,
    acquisition_persistence_threshold: int = 2,
    decomposition_builder: Any | None = None,
    archive_builder: Any | None = None,
    parallel_builder: Any | None = None,
) -> tuple[Any, Any, Any]:
    if decomposition_builder is None or archive_builder is None or parallel_builder is None:
        from pipeline_core.discovery.research_idea_epistemic_archive import (
            build_epistemic_decomposition_report,
            build_multi_realization_archive,
        )
        from pipeline_core.discovery.research_idea_parallel_partial_search import (
            build_parallel_partial_search_report,
        )

        decomposition_builder = decomposition_builder or build_epistemic_decomposition_report
        archive_builder = archive_builder or build_multi_realization_archive
        parallel_builder = parallel_builder or build_parallel_partial_search_report

    feedback = feedback_report_from_lifecycle(lifecycle)
    decomposition = decomposition_builder(
        research_ideas=population_nodes(execution),
        lifecycles=[lifecycle],
        portfolios=[portfolio],
        feedback_reports=[feedback],
    )
    archive = archive_builder(decomposition)
    parallel = parallel_builder(
        decomposition=decomposition,
        archive=archive,
        acquisition_persistence_threshold=acquisition_persistence_threshold,
    )
    return decomposition, archive, parallel


__all__ = [
    "ClosedGenerationCycleArtifacts",
    "ClosedGenerationCycleReport",
    "adapt_epistemic_execution_for_realization",
    "build_cycle_epistemic_artifacts",
    "build_cycle_report",
    "build_legacy_offspring_execution_payload",
    "feedback_report_from_lifecycle",
    "population_nodes",
]
