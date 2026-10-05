from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AdaptiveYieldRecordKind = Literal["EFFECTIVE", "SEARCH_ATTEMPT"]


AdaptiveYieldCondition = Literal[
    "CLOSED_LOOP_775",
    "LOCAL_ADAPTIVE_776",
    "GRAPH_ADAPTIVE_777",
]

CONDITION_ORDER: tuple[AdaptiveYieldCondition, ...] = (
    "CLOSED_LOOP_775",
    "LOCAL_ADAPTIVE_776",
    "GRAPH_ADAPTIVE_777",
)

BASELINE_COMMIT = "e2c4a734b97fb36ec92bf907f09ef57197cc6666"


class AdaptiveYieldPrediction(StrictModel):
    observable: str
    expected_direction: str
    rationale: str


class AdaptiveYieldFalsifier(StrictModel):
    observable: str
    falsifying_outcome: str


class AdaptiveYieldCandidateRecord(StrictModel):
    record_id: str
    scientific_fingerprint: str
    condition: AdaptiveYieldCondition
    hypothesis_id: str
    source_context_id: str
    source_context_sha256: str
    portfolio_path: str
    epoch_index: int | None = None
    record_kind: AdaptiveYieldRecordKind = "EFFECTIVE"
    title: str
    hypothesis_statement: str
    hypothesis_type: str
    inferential_bridge: str
    premise_statement_ids: list[str] = Field(default_factory=list)
    premise_texts: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    predictions: list[AdaptiveYieldPrediction] = Field(default_factory=list)
    falsifiers: list[AdaptiveYieldFalsifier] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    marginal_from_previous_condition: bool = False
    production_selection_authority: Literal[False] = False
    novelty_authority_created: Literal[False] = False


class AdaptiveYieldStageMetrics(StrictModel):
    condition: AdaptiveYieldCondition
    status: str
    cumulative_effective_count: int = Field(ge=0)
    marginal_effective_count: int = Field(ge=0)
    search_attempt_unique_count: int = Field(default=0, ge=0)
    unresolved_count: int = Field(ge=0)
    graph_handoff_count: int = Field(ge=0)
    graph_retraversal_count: int = Field(ge=0)
    new_context_count: int = Field(ge=0)
    structurally_new_premise_count: int = Field(ge=0)
    action_counts: dict[str, int] = Field(default_factory=dict)
    llm_call_artifact_count: int = Field(ge=0)
    retrieval_report_artifact_count: int = Field(ge=0)
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    candidate_count_is_quality_signal: Literal[False] = False
    composite_quality_score_computed: Literal[False] = False


class AdaptiveYieldCaseAudit(StrictModel):
    schema_version: Literal["adaptive-scientific-yield-case-audit-v1"] = (
        "adaptive-scientific-yield-case-audit-v1"
    )
    case_id: str
    run_dir: str
    baseline_commit: str
    question: str
    domain_profile_id: str
    stages: list[AdaptiveYieldStageMetrics]
    candidates: list[AdaptiveYieldCandidateRecord]
    condition_order: list[AdaptiveYieldCondition] = Field(
        default_factory=lambda: list(CONDITION_ORDER)
    )
    production_selection_changed: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    scientific_authority_created: Literal[False] = False
    candidate_count_is_quality_signal: Literal[False] = False
    composite_quality_score_computed: Literal[False] = False

    @model_validator(mode="after")
    def validate_stages(self) -> "AdaptiveYieldCaseAudit":
        observed = [x.condition for x in self.stages]
        if observed != list(CONDITION_ORDER):
            raise ValueError(
                f"adaptive yield stages must be in canonical order: {CONDITION_ORDER}"
            )
        return self


class BlindYieldCandidate(StrictModel):
    candidate_alias: str
    title: str
    hypothesis_statement: str
    inferential_bridge: str
    premise_texts: list[str] = Field(default_factory=list)
    predictions: list[AdaptiveYieldPrediction] = Field(default_factory=list)
    falsifiers: list[AdaptiveYieldFalsifier] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class BlindYieldArm(StrictModel):
    arm_alias: Literal["ARM_A", "ARM_B"]
    candidates: list[BlindYieldCandidate] = Field(default_factory=list)
    candidate_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> "BlindYieldArm":
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count mismatch")
        return self


class BlindYieldComparison(StrictModel):
    comparison_alias: str
    arm_a: BlindYieldArm
    arm_b: BlindYieldArm
    candidate_count_is_not_quality_signal: Literal[True] = True


class AdaptiveYieldBlindPacket(StrictModel):
    schema_version: Literal["adaptive-yield-blind-packet-v1"] = (
        "adaptive-yield-blind-packet-v1"
    )
    packet_id: str
    task_alias: str
    question: str
    comparisons: list[BlindYieldComparison]
    dimensions: list[str]
    source_condition_labels_hidden: Literal[True] = True
    source_candidate_ids_hidden: Literal[True] = True
    candidate_count_is_not_quality_signal: Literal[True] = True
    overall_score_requested: Literal[False] = False
    overall_winner_requested: Literal[False] = False
    production_selection_changed: Literal[False] = False


class BlindYieldKeyEntry(StrictModel):
    comparison_alias: str
    arm_a_condition: AdaptiveYieldCondition
    arm_b_condition: AdaptiveYieldCondition
    arm_a_alias_to_record_id: dict[str, str] = Field(default_factory=dict)
    arm_b_alias_to_record_id: dict[str, str] = Field(default_factory=dict)


class AdaptiveYieldBlindKey(StrictModel):
    schema_version: Literal["adaptive-yield-blind-key-v1"] = (
        "adaptive-yield-blind-key-v1"
    )
    packet_id: str
    case_id: str
    comparisons: list[BlindYieldKeyEntry]
    evaluator_must_not_load_key: Literal[True] = True
    production_selection_changed: Literal[False] = False


YIELD_DIMENSIONS = [
    "task_relevance_and_coverage",
    "evidence_discipline",
    "mechanistic_explanatory_gain",
    "prediction_specificity_and_differentiation",
    "falsifiability",
    "discriminating_experiment_quality",
    "focus_and_redundancy_control",
    "scientific_research_usefulness",
]


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(
        canonical_json(parts).encode("utf-8")
    ).hexdigest()[:20]
    return f"{prefix}:{digest}"


def scientific_fingerprint(card: dict[str, Any]) -> str:
    payload = {
        "title": str(card.get("title") or "").strip(),
        "statement": str(
            card.get("hypothesis_statement") or card.get("statement") or ""
        ).strip(),
        "bridge": str(card.get("inferential_bridge") or "").strip(),
        "premises": sorted(
            str(x) for x in card.get("premise_statement_ids", []) or []
        ),
        "predictions": [
            {
                "observable": str(x.get("observable") or ""),
                "expected_direction": str(x.get("expected_direction") or ""),
                "rationale": str(x.get("rationale") or ""),
            }
            for x in card.get("predicted_observations", []) or []
            if isinstance(x, dict)
        ],
        "falsifiers": [
            {
                "observable": str(x.get("observable") or ""),
                "falsifying_outcome": str(x.get("falsifying_outcome") or ""),
            }
            for x in card.get("falsification_criteria", []) or []
            if isinstance(x, dict)
        ],
    }
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def normalize_card(
    *,
    condition: AdaptiveYieldCondition,
    card: dict[str, Any],
    portfolio_path: str,
    premise_text_by_id: dict[str, str],
    epoch_index: int | None = None,
    previous_fingerprints: set[str] | None = None,
    record_kind: AdaptiveYieldRecordKind = "EFFECTIVE",
) -> AdaptiveYieldCandidateRecord:
    fingerprint = scientific_fingerprint(card)
    premise_ids = [
        str(x) for x in card.get("premise_statement_ids", []) or []
    ]
    return AdaptiveYieldCandidateRecord(
        record_id=stable_id(
            "adaptive_yield_candidate",
            condition,
            card.get("hypothesis_id"),
            fingerprint,
            portfolio_path,
            record_kind,
        ),
        scientific_fingerprint=fingerprint,
        condition=condition,
        hypothesis_id=str(card.get("hypothesis_id") or ""),
        source_context_id=str(card.get("source_context_id") or ""),
        source_context_sha256=str(card.get("source_context_sha256") or ""),
        portfolio_path=str(portfolio_path),
        epoch_index=epoch_index,
        record_kind=record_kind,
        title=str(card.get("title") or ""),
        hypothesis_statement=str(card.get("hypothesis_statement") or ""),
        hypothesis_type=str(card.get("hypothesis_type") or ""),
        inferential_bridge=str(card.get("inferential_bridge") or ""),
        premise_statement_ids=premise_ids,
        premise_texts=[
            premise_text_by_id.get(statement_id, "")
            for statement_id in premise_ids
        ],
        source_paper_ids=[
            str(x) for x in card.get("source_paper_ids", []) or []
        ],
        predictions=[
            AdaptiveYieldPrediction(
                observable=str(x.get("observable") or ""),
                expected_direction=str(x.get("expected_direction") or ""),
                rationale=str(x.get("rationale") or ""),
            )
            for x in card.get("predicted_observations", []) or []
            if isinstance(x, dict)
        ],
        falsifiers=[
            AdaptiveYieldFalsifier(
                observable=str(x.get("observable") or ""),
                falsifying_outcome=str(x.get("falsifying_outcome") or ""),
            )
            for x in card.get("falsification_criteria", []) or []
            if isinstance(x, dict)
        ],
        assumptions=[str(x) for x in card.get("assumptions", []) or []],
        marginal_from_previous_condition=(
            fingerprint not in (previous_fingerprints or set())
        ),
    )


def build_blind_packet(
    audit: AdaptiveYieldCaseAudit,
) -> tuple[AdaptiveYieldBlindPacket, AdaptiveYieldBlindKey]:
    by_condition: dict[str, list[AdaptiveYieldCandidateRecord]] = {
        condition: [] for condition in CONDITION_ORDER
    }
    for row in audit.candidates:
        if row.record_kind == "EFFECTIVE":
            by_condition[row.condition].append(row)

    pairings = [
        ("CLOSED_LOOP_775", "LOCAL_ADAPTIVE_776"),
        ("LOCAL_ADAPTIVE_776", "GRAPH_ADAPTIVE_777"),
        ("CLOSED_LOOP_775", "GRAPH_ADAPTIVE_777"),
    ]
    comparisons: list[BlindYieldComparison] = []
    keys: list[BlindYieldKeyEntry] = []

    for index, (left, right) in enumerate(pairings, start=1):
        comparison_alias = f"COMPARISON_{index:02d}"
        left_rows = sorted(
            by_condition[left], key=lambda x: x.scientific_fingerprint
        )
        right_rows = sorted(
            by_condition[right], key=lambda x: x.scientific_fingerprint
        )
        swap = int(
            hashlib.sha256(
                f"{audit.case_id}|{comparison_alias}".encode("utf-8")
            ).hexdigest()[:2],
            16,
        ) % 2 == 1
        a_rows, b_rows = (
            (right_rows, left_rows) if swap else (left_rows, right_rows)
        )
        a_condition, b_condition = (
            (right, left) if swap else (left, right)
        )

        def blind(rows: list[AdaptiveYieldCandidateRecord]):
            output = []
            mapping = {}
            for i, row in enumerate(rows, start=1):
                alias = f"C{i:02d}"
                mapping[alias] = row.record_id
                output.append(
                    BlindYieldCandidate(
                        candidate_alias=alias,
                        title=row.title,
                        hypothesis_statement=row.hypothesis_statement,
                        inferential_bridge=row.inferential_bridge,
                        premise_texts=row.premise_texts,
                        predictions=row.predictions,
                        falsifiers=row.falsifiers,
                        assumptions=row.assumptions,
                    )
                )
            return output, mapping

        a_candidates, a_map = blind(a_rows)
        b_candidates, b_map = blind(b_rows)
        comparisons.append(
            BlindYieldComparison(
                comparison_alias=comparison_alias,
                arm_a=BlindYieldArm(
                    arm_alias="ARM_A",
                    candidates=a_candidates,
                    candidate_count=len(a_candidates),
                ),
                arm_b=BlindYieldArm(
                    arm_alias="ARM_B",
                    candidates=b_candidates,
                    candidate_count=len(b_candidates),
                ),
            )
        )
        keys.append(
            BlindYieldKeyEntry(
                comparison_alias=comparison_alias,
                arm_a_condition=a_condition,
                arm_b_condition=b_condition,
                arm_a_alias_to_record_id=a_map,
                arm_b_alias_to_record_id=b_map,
            )
        )

    packet_id = stable_id(
        "adaptive_yield_blind_packet",
        audit.case_id,
        [x.scientific_fingerprint for x in audit.candidates],
    )
    return (
        AdaptiveYieldBlindPacket(
            packet_id=packet_id,
            task_alias=stable_id("task_alias", audit.case_id),
            question=audit.question,
            comparisons=comparisons,
            dimensions=list(YIELD_DIMENSIONS),
        ),
        AdaptiveYieldBlindKey(
            packet_id=packet_id,
            case_id=audit.case_id,
            comparisons=keys,
        ),
    )


def count_llm_call_artifacts(root: Path) -> int:
    count = 0
    if not root.exists():
        return 0
    for path in root.rglob("*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if '"record_type": "call"' in line or '"record_type":"call"' in line:
                    count += 1
        except (OSError, UnicodeDecodeError):
            continue
    return count


def count_retrieval_report_artifacts(root: Path) -> int:
    if not root.exists():
        return 0
    names = {"external.report.json", "lower.report.json", "retraversal.summary.json"}
    return sum(1 for path in root.rglob("*.json") if path.name in names)


def elapsed_from_stage_rows(
    stage_rows: list[dict[str, Any]], stage_prefix: str
) -> float | None:
    import datetime as dt
    rows = [
        row for row in stage_rows
        if str(row.get("name") or "").startswith(stage_prefix)
    ]
    if not rows:
        return None
    start = rows[0].get("started_at_utc")
    end = rows[0].get("finished_at_utc")
    if not start or not end:
        return None
    try:
        return max(
            0.0,
            (dt.datetime.fromisoformat(str(end)) - dt.datetime.fromisoformat(str(start))).total_seconds(),
        )
    except ValueError:
        return None


def summarize_cohort(cases: list[AdaptiveYieldCaseAudit]) -> dict[str, Any]:
    per_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITION_ORDER:
        rows = [
            stage
            for case in cases
            for stage in case.stages
            if stage.condition == condition
        ]
        per_condition[condition] = {
            "case_count": len(rows),
            "cumulative_effective_count_total": sum(x.cumulative_effective_count for x in rows),
            "marginal_effective_count_total": sum(x.marginal_effective_count for x in rows),
            "unresolved_count_total": sum(x.unresolved_count for x in rows),
            "graph_retraversal_count_total": sum(x.graph_retraversal_count for x in rows),
            "llm_call_artifact_count_total": sum(x.llm_call_artifact_count for x in rows),
            "retrieval_report_artifact_count_total": sum(x.retrieval_report_artifact_count for x in rows),
        }
    return {
        "schema_version": "adaptive-scientific-yield-cohort-audit-v1",
        "baseline_commit": BASELINE_COMMIT,
        "case_count": len(cases),
        "conditions": per_condition,
        "scientific_superiority_established": False,
        "overall_winner_selected": False,
        "composite_quality_score_computed": False,
        "production_selection_authority": False,
    }
