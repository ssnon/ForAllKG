from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.grounded_scope import (
    GroundedSemanticTaskScopeResult,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReframeEvidenceStatement(StrictModel):
    statement_id: str
    text: str
    epistemic_role: str
    claim_kind: str
    paper_ids: list[str] = Field(default_factory=list)
    requires_verification: bool = False
    scientific_support_node_ids: list[str] = Field(default_factory=list)
    scientific_support_edge_ids: list[str] = Field(default_factory=list)


class ConditionEvidenceExample(StrictModel):
    object_kind: Literal["measurement", "experiment", "calculation"]
    paper_id: str
    chunk_id: str
    object_id: str
    label: str
    conditions: list[dict[str, Any]] = Field(min_length=1)
    value_summary: str | None = None
    direct_grounded_object: bool = False


class ScientificReframeEvidencePacket(StrictModel):
    schema_version: Literal[
        "scientific-reframe-evidence-packet-v1"
    ] = "scientific-reframe-evidence-packet-v1"

    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    premise_statements: list[ReframeEvidenceStatement] = Field(min_length=1)
    gap_statements: list[ReframeEvidenceStatement] = Field(default_factory=list)
    condition_examples: list[ConditionEvidenceExample] = Field(default_factory=list)
    grounded_source_chunk_count: int = Field(ge=0)
    unresolved_grounded_node_count: int = Field(ge=0)

    source_is_grounded_hypothesis_context: Literal[True] = True
    packet_catalog_wholesale_included: Literal[False] = False
    missing_condition_is_negative_evidence: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_statement_ids(self) -> "ScientificReframeEvidencePacket":
        premise_ids = [row.statement_id for row in self.premise_statements]
        gap_ids = [row.statement_id for row in self.gap_statements]
        if len(premise_ids) != len(set(premise_ids)):
            raise ValueError("premise statement IDs must be unique")
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("gap statement IDs must be unique")
        if set(premise_ids) & set(gap_ids):
            raise ValueError("statement cannot be both premise and gap")
        return self


def _record_label(record: SemanticIRRecord) -> str:
    payload = record.payload
    for key in ("label", "metric", "name", "statement", "source_expression"):
        value = str(payload.get(key, "")).strip()
        if value:
            return value
    return record.ref.object_id


def _value_summary(record: SemanticIRRecord) -> str | None:
    if record.ref.object_kind != "measurement":
        return None
    payload = record.payload
    value = payload.get("value_numeric")
    if value is None or value == "":
        value = payload.get("value_text")
    if value is None or value == "":
        return None
    unit = str(payload.get("unit", "")).strip()
    qualifier = str(payload.get("qualifier", "")).strip()
    parts = [str(value)]
    if unit:
        parts.append(unit)
    if qualifier:
        parts.append(f"({qualifier})")
    return " ".join(parts)


def _selected_records(
    *,
    bundles: dict[str, SemanticIRBundle],
    grounded: GroundedSemanticTaskScopeResult,
) -> list[SemanticIRRecord]:
    chunk_keys = {
        ref.identity_key()
        for ref in grounded.resolution.source_chunks
    }
    rows: list[SemanticIRRecord] = []
    for bundle in bundles.values():
        for record in bundle.records:
            source = record.source_chunk_ref
            if source is None or source.identity_key() not in chunk_keys:
                continue
            rows.append(record)
    return rows


def _direct_semantic_ref_keys(
    grounded: GroundedSemanticTaskScopeResult,
) -> set[tuple[str, str | None, str, str]]:
    return {
        ref.identity_key()
        for resolved in grounded.resolution.resolved_objects
        for ref in resolved.semantic_refs
    }


def _condition_examples(
    *,
    bundles: dict[str, SemanticIRBundle],
    grounded: GroundedSemanticTaskScopeResult,
    max_condition_examples: int,
) -> list[ConditionEvidenceExample]:
    if max_condition_examples <= 0:
        return []

    direct_keys = _direct_semantic_ref_keys(grounded)
    rows: list[tuple[int, str, str, str, str, ConditionEvidenceExample]] = []
    for record in _selected_records(bundles=bundles, grounded=grounded):
        if record.ref.object_kind not in {"measurement", "experiment", "calculation"}:
            continue
        conditions = record.payload.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            continue
        chunk_id = str(record.ref.chunk_id or "").strip()
        if not chunk_id:
            continue
        direct = record.ref.identity_key() in direct_keys
        example = ConditionEvidenceExample(
            object_kind=record.ref.object_kind,
            paper_id=record.ref.paper_id,
            chunk_id=chunk_id,
            object_id=record.ref.object_id,
            label=_record_label(record),
            conditions=[row for row in conditions if isinstance(row, dict)],
            value_summary=_value_summary(record),
            direct_grounded_object=direct,
        )
        if not example.conditions:
            continue
        rows.append(
            (
                0 if direct else 1,
                example.object_kind,
                example.paper_id,
                example.chunk_id,
                example.object_id,
                example,
            )
        )

    rows.sort(key=lambda row: row[:-1])
    return [row[-1] for row in rows[:max_condition_examples]]


def build_scientific_reframe_evidence_packet(
    *,
    context: HypothesisContext,
    grounded: GroundedSemanticTaskScopeResult,
    bundles: dict[str, SemanticIRBundle],
    max_condition_examples: int = 30,
) -> ScientificReframeEvidencePacket:
    if context.task_id != grounded.selection.task_id:
        raise ValueError("HypothesisContext task_id does not match grounded scope")
    by_id = {row.statement_id: row for row in context.evidence_statements}

    def statement(statement_id: str) -> ReframeEvidenceStatement:
        row = by_id.get(statement_id)
        if row is None:
            raise ValueError(
                f"grounded scope references missing context statement: {statement_id}"
            )
        return ReframeEvidenceStatement(
            statement_id=row.statement_id,
            text=row.text,
            epistemic_role=row.epistemic_role,
            claim_kind=row.claim_kind,
            paper_ids=list(row.paper_ids),
            requires_verification=row.requires_verification,
            scientific_support_node_ids=list(row.scientific_support_node_ids),
            scientific_support_edge_ids=list(row.scientific_support_edge_ids),
        )

    premises = [
        statement(statement_id)
        for statement_id in grounded.selection.selected_premise_statement_ids
    ]
    gaps = [
        statement(statement_id)
        for statement_id in grounded.selection.selected_gap_statement_ids
    ]
    if not premises:
        raise ValueError("scientific reframing requires at least one grounded premise")

    return ScientificReframeEvidencePacket(
        task_id=context.task_id,
        question=context.question,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        premise_statements=premises,
        gap_statements=gaps,
        condition_examples=_condition_examples(
            bundles=bundles,
            grounded=grounded,
            max_condition_examples=max_condition_examples,
        ),
        grounded_source_chunk_count=grounded.source_chunk_count,
        unresolved_grounded_node_count=len(
            grounded.resolution.unresolved_paper_local_node_ids
        ),
    )
