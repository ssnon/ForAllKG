from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict

from pipeline_core.chunking import ChunkSpec
from pipeline_core.draft_schema import KnowledgeGraphDraft
from pipeline_core.extraction_domain import ExtractionDomainAdapter
from pipeline_core.llm_telemetry import estimate_tokens
from pipeline_core.corpus.strict_validation import (
    ValidationContext,
    finalize_draft,
    validate_draft,
)
from pipeline_core.corpus.vocab_registry import VocabularyRegistry


REPLAY_FIXTURE_SCHEMA_VERSION = "domain-gate-recovery-replay-fixture-v1"
REPLAY_SUMMARY_SCHEMA_VERSION = "domain-gate-recovery-replay-summary-v1"


def _stable_hash(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DomainGateReplayFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = REPLAY_FIXTURE_SCHEMA_VERSION
    fixture_id: str

    adapter_id: str
    domain_profile_id: str
    prompt_version: str

    paper_id: str
    chunk_id: str
    section: str
    document_id: str
    document_role: str
    page_ids: list[int]
    asset_ids: list[str]

    core_text: str
    left_context: str
    right_context: str
    asset_context: str

    rejected_graph_payload: dict[str, Any]
    domain_error: str

    system_prompt: str
    user_prompt: str
    system_prompt_sha256: str
    user_prompt_sha256: str
    rejected_graph_sha256: str

    full_response_model: str
    compact_response_model: str
    full_schema_sha256: str
    compact_schema_sha256: str
    full_schema_estimated_tokens: int
    compact_schema_estimated_tokens: int

    captured_model: str
    captured_provider: str | None
    max_completion_tokens: int
    temperature: float = 0.0


def build_domain_gate_replay_fixture(
    *,
    extraction_adapter: ExtractionDomainAdapter,
    chunk: ChunkSpec,
    rejected_draft: KnowledgeGraphDraft,
    domain_error: Exception,
    system_prompt: str,
    user_prompt: str,
    captured_model: str,
    captured_provider: str | None,
    max_completion_tokens: int,
) -> DomainGateReplayFixture:
    full_model = extraction_adapter.domain_gate_recovery_response_model(
        compact=False
    )
    compact_model = extraction_adapter.domain_gate_recovery_response_model(
        compact=True
    )
    full_schema = full_model.model_json_schema()
    compact_schema = compact_model.model_json_schema()
    full_tokens, _ = estimate_tokens(full_schema)
    compact_tokens, _ = estimate_tokens(compact_schema)

    rejected_payload = rejected_draft.model_dump(mode="json")
    base = {
        "adapter_id": extraction_adapter.adapter_id,
        "domain_profile_id": extraction_adapter.domain_profile_id,
        "prompt_version": extraction_adapter.prompt_version,
        "paper_id": chunk.paper_id,
        "chunk_id": chunk.chunk_id,
        "section": chunk.section,
        "document_id": chunk.document_id,
        "document_role": chunk.document_role,
        "page_ids": list(chunk.page_ids),
        "asset_ids": list(chunk.asset_ids),
        "core_text": chunk.core_text,
        "left_context": chunk.left_context,
        "right_context": chunk.right_context,
        "asset_context": chunk.asset_context,
        "rejected_graph_payload": rejected_payload,
        "domain_error": str(domain_error),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(
            user_prompt.encode("utf-8")
        ).hexdigest(),
        "rejected_graph_sha256": _stable_hash(rejected_payload),
        "full_response_model": full_model.__name__,
        "compact_response_model": compact_model.__name__,
        "full_schema_sha256": _stable_hash(full_schema),
        "compact_schema_sha256": _stable_hash(compact_schema),
        "full_schema_estimated_tokens": full_tokens,
        "compact_schema_estimated_tokens": compact_tokens,
        "captured_model": captured_model,
        "captured_provider": captured_provider,
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0.0,
    }
    fixture_id = "dgr:" + _stable_hash(base)[:24]
    return DomainGateReplayFixture(
        fixture_id=fixture_id,
        **base,
    )


def verify_fixture_contract(
    fixture: DomainGateReplayFixture,
    extraction_adapter: ExtractionDomainAdapter,
) -> list[str]:
    problems: list[str] = []
    if fixture.adapter_id != extraction_adapter.adapter_id:
        problems.append(
            "adapter_id drift: "
            f"{fixture.adapter_id!r} != {extraction_adapter.adapter_id!r}"
        )
    if fixture.domain_profile_id != extraction_adapter.domain_profile_id:
        problems.append(
            "domain_profile_id drift: "
            f"{fixture.domain_profile_id!r} != "
            f"{extraction_adapter.domain_profile_id!r}"
        )
    if fixture.prompt_version != extraction_adapter.prompt_version:
        problems.append(
            "prompt_version drift: "
            f"{fixture.prompt_version!r} != {extraction_adapter.prompt_version!r}"
        )

    full_model = extraction_adapter.domain_gate_recovery_response_model(
        compact=False
    )
    compact_model = extraction_adapter.domain_gate_recovery_response_model(
        compact=True
    )
    if fixture.full_response_model != full_model.__name__:
        problems.append(
            "full response-model drift: "
            f"{fixture.full_response_model!r} != {full_model.__name__!r}"
        )
    if fixture.compact_response_model != compact_model.__name__:
        problems.append(
            "compact response-model drift: "
            f"{fixture.compact_response_model!r} != {compact_model.__name__!r}"
        )
    if fixture.full_schema_sha256 != _stable_hash(full_model.model_json_schema()):
        problems.append("full response schema fingerprint drift")
    if fixture.compact_schema_sha256 != _stable_hash(
        compact_model.model_json_schema()
    ):
        problems.append("compact response schema fingerprint drift")
    if fixture.system_prompt_sha256 != hashlib.sha256(
        fixture.system_prompt.encode("utf-8")
    ).hexdigest():
        problems.append("frozen system prompt fingerprint mismatch")
    if fixture.user_prompt_sha256 != hashlib.sha256(
        fixture.user_prompt.encode("utf-8")
    ).hexdigest():
        problems.append("frozen user prompt fingerprint mismatch")
    if fixture.rejected_graph_sha256 != _stable_hash(
        fixture.rejected_graph_payload
    ):
        problems.append("rejected graph fingerprint mismatch")
    return problems


def enforce_fixture_metadata(
    draft: KnowledgeGraphDraft,
    fixture: DomainGateReplayFixture,
) -> KnowledgeGraphDraft:
    payload = draft.model_dump(mode="python")
    payload.update(
        {
            "paper_id": fixture.paper_id,
            "chunk_id": fixture.chunk_id,
            "section": fixture.section,
            "document_id": fixture.document_id,
            "document_role": fixture.document_role,
            "page_ids": list(fixture.page_ids),
            "asset_ids": list(fixture.asset_ids),
        }
    )
    return KnowledgeGraphDraft.model_validate(payload)


def _issue_counts(report) -> dict[str, int]:
    counts = Counter(
        getattr(item.code, "value", str(item.code))
        for item in report.issues
    )
    return dict(sorted(counts.items()))


def _measurement_issue_total(issue_counts: dict[str, int]) -> int:
    return sum(
        int(count)
        for code, count in issue_counts.items()
        if "MEASUREMENT" in str(code)
    )


def evaluate_replay_draft(
    *,
    generated: BaseModel,
    fixture: DomainGateReplayFixture,
    extraction_adapter: ExtractionDomainAdapter,
    experiment_registry: VocabularyRegistry,
    metric_registry: VocabularyRegistry,
) -> dict[str, Any]:
    canonical = extraction_adapter.canonicalize_generation_output(generated)
    canonical = enforce_fixture_metadata(canonical, fixture)

    domain_gate_pass = True
    domain_gate_error: str | None = None
    try:
        canonical = extraction_adapter.normalize_draft_vocabulary(canonical)
        extraction_adapter.validate_draft_vocabulary(canonical)
    except Exception as error:
        domain_gate_pass = False
        domain_gate_error = f"{type(error).__name__}: {error}"

    report = validate_draft(
        canonical,
        relation_constraints=extraction_adapter.strict_relation_constraints,
    )
    issue_counts = _issue_counts(report)

    finalization_success = False
    finalization_issue_counts: dict[str, int] = {}
    vocabulary_issue_count = 0
    finalized_node_count = 0
    finalized_edge_count = 0

    if domain_gate_pass:
        finalized = finalize_draft(
            draft=canonical,
            context=ValidationContext(
                paper_id=fixture.paper_id,
                chunk_id=fixture.chunk_id,
                section=fixture.section,
                document_id=fixture.document_id,
                document_role=fixture.document_role,
                page_ids=list(fixture.page_ids),
                asset_ids=list(fixture.asset_ids),
            ),
            experiment_registry=experiment_registry,
            metric_registry=metric_registry,
            relation_constraints=extraction_adapter.strict_relation_constraints,
        )
        finalization_success = finalized.graph is not None
        finalization_issue_counts = _issue_counts(finalized.report)
        vocabulary_issue_count = len(finalized.vocabulary_issues)
        if finalized.graph is not None:
            finalized_node_count = len(finalized.graph.all_node_ids())
            finalized_edge_count = len(finalized.graph.edges)

    mechanism_ids = {node.id for node in canonical.mechanism_claims}
    mechanism_incident_edges = sum(
        edge.source in mechanism_ids or edge.target in mechanism_ids
        for edge in canonical.edges
    )
    mechanism_connected = bool(
        mechanism_ids and mechanism_incident_edges > 0
    )

    return {
        "canonical_draft": canonical,
        "domain_gate_pass": domain_gate_pass,
        "domain_gate_error": domain_gate_error,
        "strict_valid": bool(report.valid),
        "issue_counts": issue_counts,
        "measurement_issue_count": _measurement_issue_total(issue_counts),
        "finalization_success": finalization_success,
        "finalization_issue_counts": finalization_issue_counts,
        "vocabulary_issue_count": vocabulary_issue_count,
        "node_count": len(canonical.all_node_ids()),
        "edge_count": len(canonical.edges),
        "entity_count": len(canonical.entities),
        "experiment_count": len(canonical.experiments),
        "calculation_count": len(canonical.calculations),
        "measurement_count": len(canonical.measurements),
        "measurement_group_count": len(canonical.measurement_groups),
        "observation_claim_count": len(canonical.observation_claims),
        "mechanism_claim_count": len(canonical.mechanism_claims),
        "mechanism_incident_edge_count": mechanism_incident_edges,
        "mechanism_connected": mechanism_connected,
        "finalized_node_count": finalized_node_count,
        "finalized_edge_count": finalized_edge_count,
        "canonical_output_sha256": _stable_hash(
            canonical.model_dump(mode="json")
        ),
    }


def _aggregate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    issue_counts: Counter[str] = Counter()
    finalization_issue_counts: Counter[str] = Counter()
    for row in rows:
        issue_counts.update(row.get("issue_counts") or {})
        finalization_issue_counts.update(
            row.get("finalization_issue_counts") or {}
        )

    def isum(key: str) -> int:
        return sum(int(row.get(key) or 0) for row in rows)

    return {
        "calls": len(rows),
        "llm_success": sum(bool(row.get("llm_success")) for row in rows),
        "domain_gate_pass": sum(
            bool(row.get("domain_gate_pass")) for row in rows
        ),
        "strict_valid": sum(bool(row.get("strict_valid")) for row in rows),
        "finalization_success": sum(
            bool(row.get("finalization_success")) for row in rows
        ),
        "mechanism_connected": sum(
            bool(row.get("mechanism_connected")) for row in rows
        ),
        "provider_input_tokens": isum("provider_input_tokens"),
        "provider_output_tokens": isum("provider_output_tokens"),
        "provider_total_tokens": isum("provider_total_tokens"),
        "measurement_issue_count": isum("measurement_issue_count"),
        "node_count": isum("node_count"),
        "edge_count": isum("edge_count"),
        "mechanism_claim_count": isum("mechanism_claim_count"),
        "mechanism_incident_edge_count": isum(
            "mechanism_incident_edge_count"
        ),
        "finalized_node_count": isum("finalized_node_count"),
        "finalized_edge_count": isum("finalized_edge_count"),
        "issue_counts": dict(sorted(issue_counts.items())),
        "finalization_issue_counts": dict(
            sorted(finalization_issue_counts.items())
        ),
    }


def build_zero_loss_summary(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    fixtures = sorted({str(row["fixture_id"]) for row in rows})
    comparisons: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    manual_holds: list[str] = []

    hard_non_decrease = (
        "llm_success",
        "domain_gate_pass",
        "strict_valid",
        "finalization_success",
        "mechanism_connected",
        "mechanism_claim_count",
        "mechanism_incident_edge_count",
    )

    for fixture_id in fixtures:
        fixture_rows = [
            row for row in rows if str(row["fixture_id"]) == fixture_id
        ]
        full = _aggregate_rows(
            row for row in fixture_rows if row.get("condition") == "full"
        )
        compact = _aggregate_rows(
            row for row in fixture_rows if row.get("condition") == "compact"
        )
        fixture_failures: list[str] = []
        fixture_holds: list[str] = []

        if full["calls"] != compact["calls"] or full["calls"] == 0:
            fixture_failures.append(
                "matched call counts are missing or unequal"
            )

        for key in hard_non_decrease:
            if compact[key] < full[key]:
                fixture_failures.append(
                    f"compact {key} decreased: "
                    f"{compact[key]} < {full[key]}"
                )

        if compact["measurement_issue_count"] > full[
            "measurement_issue_count"
        ]:
            fixture_failures.append(
                "compact measurement-related validation issues increased: "
                f"{compact['measurement_issue_count']} > "
                f"{full['measurement_issue_count']}"
            )

        new_compact_issue_codes = sorted(
            set(compact["issue_counts"]) - set(full["issue_counts"])
        )
        if new_compact_issue_codes:
            fixture_failures.append(
                "compact introduced validation issue families not observed "
                "under full: " + ", ".join(new_compact_issue_codes)
            )

        for key in (
            "node_count",
            "edge_count",
            "finalized_node_count",
            "finalized_edge_count",
        ):
            if compact[key] < full[key]:
                fixture_holds.append(
                    f"compact {key} is lower and requires semantic review: "
                    f"{compact[key]} < {full[key]}"
                )

        comparisons.append(
            {
                "fixture_id": fixture_id,
                "full": full,
                "compact": compact,
                "provider_input_delta": (
                    compact["provider_input_tokens"]
                    - full["provider_input_tokens"]
                ),
                "hard_failures": fixture_failures,
                "manual_holds": fixture_holds,
            }
        )
        hard_failures.extend(
            f"{fixture_id}: {message}" for message in fixture_failures
        )
        manual_holds.extend(
            f"{fixture_id}: {message}" for message in fixture_holds
        )

    if hard_failures:
        verdict = "DO_NOT_ADOPT_OBSERVED_QUALITY_LOSS_SIGNAL"
    elif manual_holds:
        verdict = "HOLD_MANUAL_SEMANTIC_REVIEW_REQUIRED"
    else:
        verdict = (
            "PASS_AUTOMATED_ZERO_LOSS_GATE_"
            "MANUAL_SEMANTIC_REVIEW_STILL_REQUIRED"
        )

    return {
        "schema_version": REPLAY_SUMMARY_SCHEMA_VERSION,
        "quality_policy": (
            "Token savings never justify any observed scientific-quality "
            "degradation. Automated PASS is not permission to adopt; manual "
            "semantic review remains mandatory."
        ),
        "fixture_count": len(fixtures),
        "row_count": len(rows),
        "verdict": verdict,
        "hard_failures": hard_failures,
        "manual_holds": manual_holds,
        "comparisons": comparisons,
    }
