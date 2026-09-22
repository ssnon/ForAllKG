from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.nonobviousness_shadow import (
    build_nonobviousness_shadow,
)
from pipeline_core.discovery.typed_required_bridge_binding import (
    validate_typed_required_bridge_binding,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AtomicBridgeBindingDecision(StrictModel):
    hypothesis_id: str
    claim_id: str
    status: Literal[
        "BOUND_EXACT_SOURCE",
        "SKIPPED_BRIDGE_ALREADY_PRESENT",
        "SKIPPED_NO_AUDIT_RECORD",
        "SKIPPED_EXACT_SOURCE_NOT_AUTHORIZED",
        "SKIPPED_NON_INFERENTIAL_BRIDGE_SOURCE",
        "SKIPPED_SOURCE_SENTENCE_NOT_UNIQUE",
        "SKIPPED_TYPED_CONTRACT_REJECTED",
    ]
    source_path: str | None = None
    exact_source_text: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class AtomicSpecificationBridgeShadowReport(StrictModel):
    schema_version: Literal[
        "atomic-scientific-specification-bridge-shadow-report-v1"
    ] = "atomic-scientific-specification-bridge-shadow-report-v1"

    source_portfolio_id: str
    source_query_plan_id: str
    source_external_report_id: str

    claim_count: int = Field(ge=0)
    exact_source_binding_count: int = Field(ge=0)
    baseline_ready_for_closure_count: int = Field(ge=0)
    bridged_ready_for_closure_count: int = Field(ge=0)
    newly_ready_for_closure_count: int = Field(ge=0)
    newly_ready_claim_ids: list[str]
    state_transitions: dict[str, int]
    decisions: list[AtomicBridgeBindingDecision]

    diagnostic_only: Literal[True] = True
    exact_source_only: Literal[True] = True
    inferential_bridge_source_only: Literal[True] = True
    free_text_bridge_generated: Literal[False] = False
    assumptions_used_as_bridge_source: Literal[False] = False
    synonym_matching_used: Literal[False] = False
    embedding_matching_used: Literal[False] = False
    prediction_or_falsifier_repaired: Literal[False] = False
    scientific_selection_changed: Literal[False] = False
    production_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


def _normalized_space(value: object) -> str:
    return " ".join(str(value or "").split())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_sentence_spans(
    source: str,
) -> list[tuple[int, int, str]]:
    """Mirror the typed-binding sentence boundary without importing internals."""

    spans: list[tuple[int, int, str]] = []
    start = 0

    for match in re.finditer(
        r"[.!?](?=\s+|$)",
        source,
    ):
        end = match.end()
        raw = source[start:end]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw.rstrip())
        span_start = start + lead
        span_end = start + trail
        if span_start < span_end:
            spans.append(
                (
                    span_start,
                    span_end,
                    source[span_start:span_end],
                )
            )
        start = end

    if start < len(source):
        raw = source[start:]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw.rstrip())
        span_start = start + lead
        span_end = start + trail
        if span_start < span_end:
            spans.append(
                (
                    span_start,
                    span_end,
                    source[span_start:span_end],
                )
            )

    return spans


def _claim_index(
    plan: LiteratureQueryPlan,
) -> dict[str, NoveltyClaim]:
    result: dict[str, NoveltyClaim] = {}
    for group in plan.claims:
        for claim in group.claims:
            if claim.claim_id in result:
                raise ValueError(
                    "duplicate claim_id in literature query plan: "
                    + claim.claim_id
                )
            result[claim.claim_id] = claim
    return result


def _audit_index(
    audit: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if audit.get("schema_version") != (
        "novelty-specification-sanitization-audit-v1"
    ):
        raise ValueError("unexpected specification-sanitization audit schema")
    if audit.get("diagnostic_only") is not True:
        raise ValueError("specification-sanitization audit lost diagnostic boundary")

    result: dict[str, Mapping[str, Any]] = {}
    rows = audit.get("records")
    if not isinstance(rows, list):
        raise ValueError("specification-sanitization audit records must be a list")

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("specification-sanitization record must be an object")
        claim_id = str(row.get("claim_id") or "").strip()
        if not claim_id:
            raise ValueError("specification-sanitization record lacks claim_id")
        if claim_id in result:
            raise ValueError(
                "duplicate claim_id in specification-sanitization audit: "
                + claim_id
            )
        result[claim_id] = row

    return result


def _binding_from_record(
    *,
    hypothesis: object,
    claim: NoveltyClaim,
    record: Mapping[str, Any],
) -> tuple[
    dict[str, object] | None,
    dict[str, object] | None,
    AtomicBridgeBindingDecision,
]:
    if str(claim.required_bridge or "").strip():
        return (
            None,
            None,
            AtomicBridgeBindingDecision(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                status="SKIPPED_BRIDGE_ALREADY_PRESENT",
            ),
        )

    authority = record.get("exact_source_recompile_authority_shadow")
    plan = record.get("exact_source_recompile_shadow")

    if not isinstance(authority, Mapping) or not isinstance(plan, Mapping):
        return (
            None,
            None,
            AtomicBridgeBindingDecision(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                status="SKIPPED_EXACT_SOURCE_NOT_AUTHORIZED",
                reason_codes=["exact_source_shadow_missing"],
            ),
        )

    reason_codes = [
        str(value)
        for value in (authority.get("authority_reason_codes") or [])
        if str(value).strip()
    ]

    if (
        authority.get("diagnostic_only") is not True
        or authority.get("production_authority") is not False
        or authority.get("production_recompile_enabled") is not False
        or authority.get("recompile_performed") is not False
        or authority.get("authority_status") != "AUTHORIZED_SHADOW"
        or authority.get("bounded_recompile_contract_satisfied") is not True
        or reason_codes
    ):
        return (
            None,
            None,
            AtomicBridgeBindingDecision(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                status="SKIPPED_EXACT_SOURCE_NOT_AUTHORIZED",
                reason_codes=reason_codes or [
                    str(authority.get("authority_status") or "not_authorized")
                ],
            ),
        )

    source_path = str(
        authority.get("candidate_source_path") or ""
    ).strip()
    if not source_path.startswith("inferential_bridge.unit["):
        return (
            None,
            None,
            AtomicBridgeBindingDecision(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                status="SKIPPED_NON_INFERENTIAL_BRIDGE_SOURCE",
                source_path=source_path or None,
            ),
        )

    exact_source = _normalized_space(
        authority.get("exact_source_text")
    )
    if not exact_source:
        return (
            None,
            None,
            AtomicBridgeBindingDecision(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                status="SKIPPED_EXACT_SOURCE_NOT_AUTHORIZED",
                source_path=source_path,
                reason_codes=["authorized_exact_source_empty"],
            ),
        )

    source_bridge = str(
        getattr(hypothesis, "inferential_bridge", "")
        or ""
    )
    matches = [
        row
        for row in _source_sentence_spans(source_bridge)
        if _normalized_space(row[2]) == exact_source
    ]
    if len(matches) != 1:
        return (
            None,
            None,
            AtomicBridgeBindingDecision(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                status="SKIPPED_SOURCE_SENTENCE_NOT_UNIQUE",
                source_path=source_path,
                exact_source_text=exact_source,
                reason_codes=[
                    f"qualifying_sentence_count={len(matches)}"
                ],
            ),
        )

    start, end, quote = matches[0]
    endpoints = [
        str(value)
        for value in (
            plan.get("relation_endpoint_anchors")
            or []
        )
        if str(value).strip()
    ]
    scopes = [
        str(value)
        for value in (
            plan.get("scope_qualifier_spans")
            or []
        )
        if str(value).strip()
    ]
    directions = [
        str(value)
        for value in (
            plan.get("directional_qualifier_spans")
            or []
        )
        if str(value).strip()
    ]

    source_sha = _sha256_text(source_bridge)
    binding: dict[str, object] = {
        "schema_version":
            "novelty-required-bridge-source-binding-diagnostic-v1",
        "binding_semantics":
            "TYPED_RELATION_REFERENCE_TO_EXISTING_SOURCE_SPAN",
        "source_path": "inferential_bridge",
        "source_sha256": source_sha,
        "start": start,
        "end": end,
        "quote": quote,
        "relation_endpoint_anchors": endpoints,
        "scope_qualifier_spans": scopes,
        "directional_qualifier_spans": directions,
        "diagnostic_only": True,
        "production_authority": False,
        "free_text_bridge_generated": False,
        "novelty_assessed": False,
        "scientific_truth_assessed": False,
        "scientific_equivalence_assessed": False,
    }
    contract: dict[str, object] = {
        "hypothesis_id": claim.hypothesis_id,
        "claim_id": claim.claim_id,
        "source_bridge_sha256": source_sha,
        "relation_endpoint_anchors": endpoints,
        "scope_qualifier_spans": scopes,
        "directional_qualifier_spans": directions,
        "semantic_warnings": {
            "bridge_alignment_shadow_warning": False,
            "direction_shadow_warning": False,
            "endpoint_shadow_warning": False,
            "scope_shadow_warning": False,
        },
    }

    # Validate the exact contract before exposing it to N9.
    proxy_claim = SimpleNamespace(
        claim_id=claim.claim_id,
        claim_text=claim.text,
    )
    try:
        recovered = validate_typed_required_bridge_binding(
            hypothesis=hypothesis,
            claim=proxy_claim,
            binding=binding,
            contract=contract,
        )
    except ValueError as exc:
        return (
            None,
            None,
            AtomicBridgeBindingDecision(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                status="SKIPPED_TYPED_CONTRACT_REJECTED",
                source_path=source_path,
                exact_source_text=exact_source,
                reason_codes=[str(exc)],
            ),
        )

    if recovered != quote:
        raise RuntimeError(
            "typed required-bridge validation returned unexpected source text"
        )

    return (
        binding,
        contract,
        AtomicBridgeBindingDecision(
            hypothesis_id=claim.hypothesis_id,
            claim_id=claim.claim_id,
            status="BOUND_EXACT_SOURCE",
            source_path="inferential_bridge",
            exact_source_text=quote,
        ),
    )


def build_atomic_specification_bridge_bindings(
    *,
    portfolio: HypothesisPortfolio,
    query_plan: LiteratureQueryPlan,
    specification_sanitization_audit: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, object]],
    list[AtomicBridgeBindingDecision],
]:
    if query_plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("atomic bridge query-plan/portfolio mismatch")

    audit_source = str(
        specification_sanitization_audit.get("source_portfolio_id")
        or ""
    )
    if audit_source != portfolio.portfolio_id:
        raise ValueError("atomic bridge audit/portfolio mismatch")

    claim_by_id = _claim_index(query_plan)
    audit_by_id = _audit_index(specification_sanitization_audit)
    hypothesis_by_id = {
        row.hypothesis_id: row
        for row in portfolio.hypotheses
    }

    bindings: dict[str, dict[str, object]] = {}
    decisions: list[AtomicBridgeBindingDecision] = []

    for claim_id, claim in claim_by_id.items():
        hypothesis = hypothesis_by_id.get(claim.hypothesis_id)
        if hypothesis is None:
            raise ValueError(
                "atomic bridge claim references absent hypothesis: "
                + claim.hypothesis_id
            )

        record = audit_by_id.get(claim_id)
        if record is None:
            decisions.append(
                AtomicBridgeBindingDecision(
                    hypothesis_id=claim.hypothesis_id,
                    claim_id=claim.claim_id,
                    status="SKIPPED_NO_AUDIT_RECORD",
                )
            )
            continue

        if str(record.get("hypothesis_id") or "") != claim.hypothesis_id:
            raise ValueError("atomic bridge audit hypothesis identity mismatch")

        binding, contract, decision = _binding_from_record(
            hypothesis=hypothesis,
            claim=claim,
            record=record,
        )
        decisions.append(decision)
        if binding is not None and contract is not None:
            bindings[claim_id] = {
                "binding": binding,
                "contract": contract,
            }

    return bindings, decisions


def _state_by_claim(
    intake: Mapping[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for hypothesis in intake.get("hypotheses", []):
        if not isinstance(hypothesis, Mapping):
            continue
        for decision in hypothesis.get("claims", []):
            if not isinstance(decision, Mapping):
                continue
            claim = decision.get("claim")
            if not isinstance(claim, Mapping):
                continue
            claim_id = str(claim.get("claim_id") or "").strip()
            state = str(decision.get("shadow_state") or "").strip()
            if claim_id:
                result[claim_id] = state
    return result


def run_atomic_specification_bridge_shadow(
    *,
    portfolio: HypothesisPortfolio,
    query_plan: LiteratureQueryPlan,
    external_report: ExternalNoveltyReport,
    specification_sanitization_audit: Mapping[str, Any],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    AtomicSpecificationBridgeShadowReport,
]:
    if external_report.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("atomic bridge external-report/portfolio mismatch")
    if query_plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("atomic bridge query-plan/portfolio mismatch")

    bindings, decisions = build_atomic_specification_bridge_bindings(
        portfolio=portfolio,
        query_plan=query_plan,
        specification_sanitization_audit=specification_sanitization_audit,
    )

    baseline = build_nonobviousness_shadow(
        plan=query_plan,
        report=external_report,
        source_portfolio=portfolio,
    )
    bridged = build_nonobviousness_shadow(
        plan=query_plan,
        report=external_report,
        source_portfolio=portfolio,
        required_bridge_bindings=bindings,
    )

    if baseline.get("scientific_selection_changed") is not False:
        raise ValueError("baseline N9 unexpectedly changed scientific selection")
    if bridged.get("scientific_selection_changed") is not False:
        raise ValueError("bridged N9 unexpectedly changed scientific selection")

    baseline_states = _state_by_claim(baseline)
    bridged_states = _state_by_claim(bridged)
    if set(baseline_states) != set(bridged_states):
        raise ValueError("atomic bridge N9 claim membership drift")

    transitions: Counter[str] = Counter()
    newly_ready: list[str] = []

    for claim_id in sorted(baseline_states):
        before = baseline_states[claim_id]
        after = bridged_states[claim_id]
        transitions[f"{before}->{after}"] += 1
        if before != "READY_FOR_CLOSURE" and after == "READY_FOR_CLOSURE":
            newly_ready.append(claim_id)

    baseline_ready = sum(
        state == "READY_FOR_CLOSURE"
        for state in baseline_states.values()
    )
    bridged_ready = sum(
        state == "READY_FOR_CLOSURE"
        for state in bridged_states.values()
    )

    report = AtomicSpecificationBridgeShadowReport(
        source_portfolio_id=portfolio.portfolio_id,
        source_query_plan_id=query_plan.plan_id,
        source_external_report_id=external_report.report_id,
        claim_count=len(baseline_states),
        exact_source_binding_count=len(bindings),
        baseline_ready_for_closure_count=baseline_ready,
        bridged_ready_for_closure_count=bridged_ready,
        newly_ready_for_closure_count=len(newly_ready),
        newly_ready_claim_ids=newly_ready,
        state_transitions=dict(sorted(transitions.items())),
        decisions=decisions,
    )

    binding_artifact: dict[str, object] = {
        "schema_version": "typed-required-bridge-live-ablation-v1",
        "source_portfolio_id": portfolio.portfolio_id,
        "source_query_plan_id": query_plan.plan_id,
        "diagnostic_only": True,
        "production_authority": False,
        "free_text_bridge_generated": False,
        "exact_source_only": True,
        "bindings_by_claim": bindings,
    }

    return binding_artifact, baseline, bridged, report


__all__ = [
    "AtomicBridgeBindingDecision",
    "AtomicSpecificationBridgeShadowReport",
    "build_atomic_specification_bridge_bindings",
    "run_atomic_specification_bridge_shadow",
]
