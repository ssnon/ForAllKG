from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    raise TypeError(f"Unsupported artifact type: {type(value).__name__}")


def _rows(value: Any, key: str) -> list[dict[str, Any]]:
    payload = _dump(value)
    raw = payload.get(key) or []
    return [dict(row) for row in raw if isinstance(row, Mapping)]


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _tokens(value: Any) -> list[str]:
    text = str(value or "")
    return _dedupe(re.findall(r"[A-Za-z0-9α-ωΑ-Ω가-힣][A-Za-z0-9α-ωΑ-Ω가-힣_-]*", text))


def _compact_query(value: str, *, max_terms: int = 10) -> str:
    stop = {
        "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with",
        "by", "through", "under", "via", "from", "into", "between", "within",
        "that", "this", "effect", "effects", "role", "state", "states",
    }
    terms = [token for token in _tokens(value) if token.casefold() not in stop]
    return " ".join(terms[:max_terms])


EpistemicMaturity = Literal[
    "IDEA_ONLY",
    "PARTIALLY_GROUNDED",
    "STRICT_GROUNDED",
    "OPERATIONAL_GROUNDED",
    "EVIDENCE_SEEKING",
    "SPECULATIVE_BUT_FALSIFIABLE",
]

GroundingCoverage = Literal["NONE", "PARTIAL", "STRICT"]
EvidenceRequirementKind = Literal[
    "GROUNDING_PREMISE_COVERAGE",
    "RELATION_OR_MECHANISM_EVIDENCE",
    "MEASUREMENT_OR_OBSERVABLE_SUPPORT",
    "COMPARISON_CONTEXT_SUPPORT",
    "DIRECTIONALITY_SUPPORT",
    "GRAPH_TOPOLOGY_COVERAGE",
    "REPRESENTATION_EXTENSION",
]
EvidenceRecoveryRoute = Literal[
    "LITERATURE_ACQUISITION",
    "KG_RETRAVERSAL",
    "CONTEXT_REBUILD",
    "OPERATIONALIZATION_REVIEW",
    "REPRESENTATION_REVIEW",
]


class MissingEvidenceRequirement(StrictModel):
    schema_version: Literal[
        "research-idea-missing-evidence-requirement-v1"
    ] = "research-idea-missing-evidence-requirement-v1"
    requirement_id: str
    idea_id: str
    generation_index: int = Field(ge=0)
    source_realization_ids: list[str] = Field(default_factory=list)
    source_hypothesis_ids: list[str] = Field(default_factory=list)
    kind: EvidenceRequirementKind
    description: str = Field(min_length=1)
    search_terms: list[str] = Field(default_factory=list)
    proposed_queries: list[str] = Field(default_factory=list)
    recovery_routes: list[EvidenceRecoveryRoute] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence_gap_is_not_idea_failure: Literal[True] = True
    external_search_result_is_not_positive_premise: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _normalize(self) -> "MissingEvidenceRequirement":
        self.source_realization_ids = _dedupe(self.source_realization_ids)
        self.source_hypothesis_ids = _dedupe(self.source_hypothesis_ids)
        self.search_terms = _dedupe(self.search_terms)
        self.proposed_queries = _dedupe(self.proposed_queries)
        self.recovery_routes = list(dict.fromkeys(self.recovery_routes))
        self.reason_codes = _dedupe(self.reason_codes)
        return self


class EpistemicRealizationRecord(StrictModel):
    schema_version: Literal[
        "research-idea-epistemic-partial-realization-v1"
    ] = "research-idea-epistemic-partial-realization-v1"
    epistemic_realization_id: str
    idea_id: str
    generation_index: int = Field(ge=0)
    realization_id: str | None = None
    hypothesis_id: str | None = None
    source_context_id: str | None = None
    attempt_index: int | None = Field(default=None, ge=1)
    realization_kind: str | None = None
    materialization_status: str

    canonical_intent: str
    scientific_commitments: list[str] = Field(default_factory=list)
    scope_commitments: list[str] = Field(default_factory=list)
    contrastive_commitments: list[str] = Field(default_factory=list)

    grounded_premise_statement_ids: list[str] = Field(default_factory=list)
    grounded_source_paper_ids: list[str] = Field(default_factory=list)
    hypothetical_bridge: str | None = None
    prediction_texts: list[str] = Field(default_factory=list)
    falsifier_texts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    prospective_identifiability: str | None = None
    current_evidence_status: str | None = None
    directionality_mode: str | None = None
    measurement_compatibility_mode: str | None = None
    residual_epistemic_state: str | None = None
    residual_state_reason: str | None = None

    grounding_coverage: GroundingCoverage
    epistemic_maturity: EpistemicMaturity
    missing_evidence_requirements: list[MissingEvidenceRequirement] = Field(default_factory=list)
    operationalization_gap_codes: list[str] = Field(default_factory=list)
    issue_codes: list[str] = Field(default_factory=list)

    research_idea_preserved_when_not_groundable: Literal[True] = True
    hypothesis_card_contract_relaxed: Literal[False] = False
    speculative_components_are_not_positive_premises: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class EpistemicDecompositionReport(StrictModel):
    schema_version: Literal[
        "research-idea-epistemic-decomposition-shadow-v1"
    ] = "research-idea-epistemic-decomposition-shadow-v1"
    report_id: str
    report_sha256: str
    records: list[EpistemicRealizationRecord] = Field(default_factory=list)
    record_count: int = Field(ge=0)
    idea_count: int = Field(ge=0)
    maturity_counts: dict[str, int] = Field(default_factory=dict)
    requirement_kind_counts: dict[str, int] = Field(default_factory=dict)
    literature_recovery_requirement_count: int = Field(ge=0)
    kg_retraversal_requirement_count: int = Field(ge=0)
    strict_hypothesis_card_contract_preserved: Literal[True] = True
    ungrounded_idea_deletion_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "EpistemicDecompositionReport":
        if self.record_count != len(self.records):
            raise ValueError("record_count mismatch")
        if self.idea_count != len({row.idea_id for row in self.records}):
            raise ValueError("idea_count mismatch")
        return self


ArchiveDisposition = Literal[
    "RETAIN_PARETO",
    "RETAIN_EPISTEMIC_DIVERSITY",
    "RETAIN_ONLY_REALIZATION",
    "DIAGNOSTIC_DOMINATED",
]


class RealizationArchiveEntry(StrictModel):
    idea_id: str
    generation_index: int = Field(ge=0)
    epistemic_realization_id: str
    realization_id: str | None = None
    hypothesis_id: str | None = None
    materialization_status: str
    grounding_coverage: GroundingCoverage
    epistemic_maturity: EpistemicMaturity
    prospective_identifiability: str | None = None
    residual_epistemic_state: str | None = None
    missing_evidence_requirement_count: int = Field(ge=0)
    prediction_present: bool
    falsifier_present: bool
    materialization_rank: int = Field(ge=0, le=2)
    operationalization_rank: int = Field(ge=0, le=2)
    evaluation_completeness_rank: int = Field(ge=0, le=2)
    falsifiability_rank: int = Field(ge=0, le=1)
    epistemic_slice_key: str
    archive_disposition: ArchiveDisposition
    archive_retained: bool
    retained_reason_codes: list[str] = Field(default_factory=list)
    scalar_fitness_used: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class IdeaRealizationArchive(StrictModel):
    idea_id: str
    generation_index: int = Field(ge=0)
    entries: list[RealizationArchiveEntry] = Field(default_factory=list)
    retained_epistemic_realization_ids: list[str] = Field(default_factory=list)
    retained_realization_ids: list[str] = Field(default_factory=list)
    retained_hypothesis_ids: list[str] = Field(default_factory=list)
    residual_state_diversity: list[str] = Field(default_factory=list)
    maturity_diversity: list[str] = Field(default_factory=list)
    destructive_replacement_applied: Literal[False] = False
    all_attempts_preserved_for_audit: Literal[True] = True
    archive_is_not_truth_authority: Literal[True] = True


class MultiRealizationArchiveReport(StrictModel):
    schema_version: Literal[
        "research-idea-multi-realization-archive-shadow-v1"
    ] = "research-idea-multi-realization-archive-shadow-v1"
    report_id: str
    report_sha256: str
    archives: list[IdeaRealizationArchive] = Field(default_factory=list)
    idea_count: int = Field(ge=0)
    realization_entry_count: int = Field(ge=0)
    retained_entry_count: int = Field(ge=0)
    multi_realization_idea_count: int = Field(ge=0)
    epistemic_diversity_retention_count: int = Field(ge=0)
    scalar_fitness_used: Literal[False] = False
    destructive_replacement_applied: Literal[False] = False
    production_selection_authority: Literal[False] = False


EvidenceRecoveryStatus = Literal[
    "REQUESTED",
    "DISCOVERY_READY",
    "AWAITING_POSITIVE_EVIDENCE_PROMOTION",
    "CONTEXT_REBUILD_REQUIRED",
    "RETRY_READY",
]


class EvidenceRecoveryRequest(StrictModel):
    request_id: str
    requirement_id: str
    requirement_kind: EvidenceRequirementKind | None = None
    idea_id: str
    generation_index: int = Field(ge=0)
    route: EvidenceRecoveryRoute
    query_strings: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)
    source_realization_ids: list[str] = Field(default_factory=list)
    status: EvidenceRecoveryStatus
    reason_codes: list[str] = Field(default_factory=list)
    positive_evidence_promotion_required_before_retry: bool
    external_metadata_is_not_positive_premise: Literal[True] = True
    idea_mutation_not_required_by_this_request: Literal[True] = True
    scientific_truth_authority: Literal[False] = False


class EvidenceRecoveryPlan(StrictModel):
    schema_version: Literal[
        "research-idea-evidence-recovery-plan-v1"
    ] = "research-idea-evidence-recovery-plan-v1"
    plan_id: str
    plan_sha256: str
    requests: list[EvidenceRecoveryRequest] = Field(default_factory=list)
    request_count: int = Field(ge=0)
    route_counts: dict[str, int] = Field(default_factory=dict)
    literature_request_count: int = Field(ge=0)
    literature_cluster_count: int = Field(ge=0, default=0)
    literature_requests_collapsed_by_clustering_count: int = Field(ge=0, default=0)
    targeted_literature_axis_count: int = Field(ge=0, default=0)
    targeted_literature_representative_request_ids: list[str] = Field(default_factory=list)
    literature_clusters_suppressed_by_budget_count: int = Field(ge=0, default=0)
    recovery_target_budget: int = Field(ge=1, default=12)
    kg_retraversal_request_count: int = Field(ge=0)
    operationalization_review_request_count: int = Field(ge=0)
    representation_review_request_count: int = Field(ge=0)
    targeted_acquisition_profile: dict[str, Any] | None = None
    retry_candidate_idea_ids: list[str] = Field(default_factory=list)
    retry_requires_rebuilt_grounded_context: Literal[True] = True
    literature_routes_require_positive_evidence_promotion_before_retry: Literal[True] = True
    existing_literature_pipeline_reused: Literal[True] = True
    external_literature_direct_premise_injection: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "EvidenceRecoveryPlan":
        if self.request_count != len(self.requests):
            raise ValueError("request_count mismatch")
        if self.targeted_literature_axis_count != len(self.targeted_literature_representative_request_ids):
            raise ValueError("targeted_literature_axis_count mismatch")
        if self.targeted_literature_axis_count > self.recovery_target_budget:
            raise ValueError("targeted literature requests exceed recovery_target_budget")
        if self.literature_cluster_count > self.literature_request_count:
            raise ValueError("literature_cluster_count cannot exceed literature_request_count")
        return self


class EvidenceRecoveryCommandPlan(StrictModel):
    schema_version: Literal[
        "research-idea-evidence-recovery-command-plan-v1"
    ] = "research-idea-evidence-recovery-command-plan-v1"
    profile_path: str
    output_root: str
    discovery_command: list[str] = Field(default_factory=list)
    selection_command: list[str] = Field(default_factory=list)
    relevance_gate_report_path: str
    strict_selected_works_path: str
    strict_selection_report_path: str
    adjacent_candidate_archive_path: str
    downstream_promotion_note: str
    retry_note: str
    automatic_positive_evidence_promotion_executed: Literal[False] = False
    automatic_canonical_graph_mutation_executed: Literal[False] = False


RecoveryRelevanceClass = Literal[
    "STRICT_DOMAIN_RELEVANT",
    "ADJACENT_METHOD_OR_MECHANISM",
    "OFF_DOMAIN_REJECT",
]


class RecoveryCandidateRelevanceRecord(StrictModel):
    work_id: str
    title: str
    doi: str | None = None
    year: int | None = None
    original_eligibility_status: str
    original_total_score: float = 0.0
    matched_axes: list[str] = Field(default_factory=list)
    relevance_class: RecoveryRelevanceClass
    domain_anchor_hits: list[str] = Field(default_factory=list)
    specific_axis_phrase_hits: list[str] = Field(default_factory=list)
    matched_recovery_request_ids: list[str] = Field(default_factory=list)
    matched_requirement_kinds: list[str] = Field(default_factory=list)
    request_specific_gap_hits: list[str] = Field(default_factory=list)
    scientific_role_hits: list[str] = Field(default_factory=list)
    max_request_gap_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    request_conditioned_strict_match: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    acquisition_authority: bool = False
    positive_premise_authority: Literal[False] = False


class RecoveryRelevanceGateReport(StrictModel):
    schema_version: Literal[
        "sis-v2-8-recovery-relevance-gate-v1"
    ] = "sis-v2-8-recovery-relevance-gate-v1"
    report_id: str
    report_sha256: str
    profile_id: str
    source_catalog_id: str
    max_strict_acquisition_total: int = Field(ge=1)
    original_candidate_count: int = Field(ge=0)
    assessed_relevance_candidate_count: int = Field(ge=0)
    strict_candidate_count: int = Field(ge=0)
    adjacent_candidate_count: int = Field(ge=0)
    off_domain_reject_count: int = Field(ge=0)
    strict_selected_count: int = Field(ge=0)
    strict_selected_work_ids: list[str] = Field(default_factory=list)
    adjacent_candidate_work_ids: list[str] = Field(default_factory=list)
    off_domain_reject_work_ids: list[str] = Field(default_factory=list)
    derived_domain_anchors: list[str] = Field(default_factory=list)
    records: list[RecoveryCandidateRelevanceRecord] = Field(default_factory=list)
    strict_selection_is_max_budget_not_fill_target: Literal[True] = True
    adjacent_candidates_are_preserved_but_not_acquired: Literal[True] = True
    off_domain_candidates_are_not_acquired: Literal[True] = True
    metadata_is_not_positive_premise: Literal[True] = True
    request_conditioning_applied: bool = False
    scientific_role_match_is_confidence_booster_not_strict_gate: Literal[True] = True
    request_conditioned_strict_requires_role_match: Literal[False] = False
    strict_requires_domain_and_request_gap_match: Literal[True] = True
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "RecoveryRelevanceGateReport":
        if self.assessed_relevance_candidate_count != len(self.records):
            raise ValueError("assessed_relevance_candidate_count mismatch")
        if self.strict_selected_count != len(self.strict_selected_work_ids):
            raise ValueError("strict_selected_count mismatch")
        if self.strict_selected_count > self.max_strict_acquisition_total:
            raise ValueError("strict selection exceeds max budget")
        if self.strict_candidate_count + self.adjacent_candidate_count + self.off_domain_reject_count != len(self.records):
            raise ValueError("relevance class counts mismatch")
        return self


def _idea_payloads(*collections: Sequence[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for collection in collections:
        for value in collection:
            row = _dump(value)
            idea_id = str(row.get("idea_id") or "").strip()
            if idea_id:
                out[idea_id] = row
    return out


def _kernel_parts(node: Mapping[str, Any]) -> tuple[str, list[str], list[str], list[str]]:
    kernel = node.get("kernel") or {}
    if not isinstance(kernel, Mapping):
        kernel = {}
    return (
        str(kernel.get("canonical_intent") or node.get("canonical_intent") or "Unspecified research intent"),
        [str(x) for x in kernel.get("core_scientific_commitments") or []],
        [str(x) for x in kernel.get("scope_commitments") or []],
        [str(x) for x in kernel.get("contrastive_commitments") or []],
    )


def _feedback_by_hypothesis(feedback: Any | None) -> dict[str, dict[str, Any]]:
    if feedback is None:
        return {}
    return {
        str(row.get("hypothesis_id")): row
        for row in _rows(feedback, "records")
        if row.get("hypothesis_id")
    }


def _portfolio_by_hypothesis(portfolio: Any | None) -> dict[str, dict[str, Any]]:
    if portfolio is None:
        return {}
    return {
        str(row.get("hypothesis_id")): row
        for row in _rows(portfolio, "hypotheses")
        if row.get("hypothesis_id")
    }


def _evolution_requests_by_idea(report: Any | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if report is None:
        return out
    for row in _rows(report, "requests"):
        idea_id = str(row.get("source_idea_id") or "")
        if idea_id:
            out[idea_id].append(row)
    return out


def _requirement(
    *,
    idea_id: str,
    generation_index: int,
    kind: EvidenceRequirementKind,
    description: str,
    intent: str,
    commitments: Sequence[str],
    realization_id: str | None,
    hypothesis_id: str | None,
    routes: Sequence[EvidenceRecoveryRoute],
    reason_codes: Sequence[str],
) -> MissingEvidenceRequirement:
    search_terms = _dedupe([
        *_tokens(intent),
        *[token for value in commitments for token in _tokens(value)],
    ])[:16]
    base = _compact_query(" ".join([intent, *commitments]))
    proposed = []
    if base:
        proposed.append(base)
    if kind == "MEASUREMENT_OR_OBSERVABLE_SUPPORT" and base:
        proposed.append(_compact_query(base + " measurement observable proxy"))
    elif kind == "RELATION_OR_MECHANISM_EVIDENCE" and base:
        proposed.append(_compact_query(base + " mechanism evidence relation"))
    elif kind == "COMPARISON_CONTEXT_SUPPORT" and base:
        proposed.append(_compact_query(base + " matched comparison context"))
    elif kind == "DIRECTIONALITY_SUPPORT" and base:
        proposed.append(_compact_query(base + " direction causal dependency"))
    rid = _stable_id(
        "missing_evidence_requirement",
        idea_id,
        generation_index,
        kind,
        description,
        realization_id,
        hypothesis_id,
    )
    return MissingEvidenceRequirement(
        requirement_id=rid,
        idea_id=idea_id,
        generation_index=generation_index,
        source_realization_ids=[realization_id] if realization_id else [],
        source_hypothesis_ids=[hypothesis_id] if hypothesis_id else [],
        kind=kind,
        description=description,
        search_terms=search_terms,
        proposed_queries=proposed,
        recovery_routes=list(routes),
        reason_codes=list(reason_codes),
    )


def _derive_requirements(
    *,
    idea_id: str,
    generation_index: int,
    intent: str,
    commitments: Sequence[str],
    realization_id: str | None,
    hypothesis_id: str | None,
    materialization_status: str,
    issue_codes: Sequence[str],
    feedback: Mapping[str, Any] | None,
    evolution_requests: Sequence[Mapping[str, Any]],
) -> tuple[list[MissingEvidenceRequirement], list[str]]:
    rows: list[MissingEvidenceRequirement] = []
    operational: list[str] = []
    issue_set = {str(code) for code in issue_codes}

    grounding_codes = {
        "UNKNOWN_PREMISE_STATEMENT",
        "INELIGIBLE_POSITIVE_PREMISE",
        "GROUNDING_DRIFT_REJECTED",
        "NO_ELIGIBLE_POSITIVE_PREMISE",
        "MISSING_SEED_HYPOTHESIS_CARD",
    }
    grounding_failure_statuses = {
        "ABSTAINED",
        "COMPILE_REJECTED",
        "VALIDATION_REJECTED",
        "GROUNDING_DRIFT_REJECTED",
        "NOT_ATTEMPTED",
    }
    if materialization_status in grounding_failure_statuses and (
        not issue_set or issue_set & grounding_codes
    ):
        rows.append(
            _requirement(
                idea_id=idea_id,
                generation_index=generation_index,
                kind="GROUNDING_PREMISE_COVERAGE",
                description=(
                    "The ResearchIdea does not currently have a strict grounded HypothesisCard; "
                    "recover eligible positive-premise coverage without changing the idea identity."
                ),
                intent=intent,
                commitments=commitments,
                realization_id=realization_id,
                hypothesis_id=hypothesis_id,
                routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
                reason_codes=["STRICT_GROUNDED_REALIZATION_MISSING"],
            )
        )

    fb = dict(feedback or {})
    residual = str(fb.get("residual_epistemic_state") or "")
    evidence_status = str(fb.get("current_evidence_status") or "")
    identifiability = str(fb.get("prospective_identifiability") or "")
    directionality = str(fb.get("directionality_mode") or "")
    measurement = str(fb.get("measurement_compatibility_mode") or "")

    if residual == "UNRESOLVED_EVIDENCE_GAP":
        rows.append(
            _requirement(
                idea_id=idea_id,
                generation_index=generation_index,
                kind="RELATION_OR_MECHANISM_EVIDENCE",
                description="Fresh residual verification reports an unresolved evidence gap.",
                intent=intent,
                commitments=commitments,
                realization_id=realization_id,
                hypothesis_id=hypothesis_id,
                routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
                reason_codes=["RESIDUAL_UNRESOLVED_EVIDENCE_GAP"],
            )
        )
    if residual == "UNRESOLVED_TOPOLOGY_GAP":
        rows.append(
            _requirement(
                idea_id=idea_id,
                generation_index=generation_index,
                kind="GRAPH_TOPOLOGY_COVERAGE",
                description="Fresh residual verification reports a topology/context coverage gap.",
                intent=intent,
                commitments=commitments,
                realization_id=realization_id,
                hypothesis_id=hypothesis_id,
                routes=["KG_RETRAVERSAL", "CONTEXT_REBUILD"],
                reason_codes=["RESIDUAL_UNRESOLVED_TOPOLOGY_GAP"],
            )
        )

    evidence_upper = evidence_status.upper()
    if any(token in evidence_upper for token in ("INSUFFICIENT", "MISSING", "UNSUPPORTED", "PARTIAL")):
        rows.append(
            _requirement(
                idea_id=idea_id,
                generation_index=generation_index,
                kind="RELATION_OR_MECHANISM_EVIDENCE",
                description=f"Prospective evidence status remains incomplete: {evidence_status}.",
                intent=intent,
                commitments=commitments,
                realization_id=realization_id,
                hypothesis_id=hypothesis_id,
                routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
                reason_codes=["PROSPECTIVE_EVIDENCE_STATUS_INCOMPLETE"],
            )
        )

    if identifiability == "NOT_OPERATIONALIZABLE":
        operational.append("NOT_OPERATIONALIZABLE")
        routes: list[EvidenceRecoveryRoute] = ["OPERATIONALIZATION_REVIEW"]
        if measurement and any(token in measurement.upper() for token in ("MISSING", "INCOMPATIBLE", "UNSUPPORTED")):
            routes = ["LITERATURE_ACQUISITION", "OPERATIONALIZATION_REVIEW", "CONTEXT_REBUILD"]
        rows.append(
            _requirement(
                idea_id=idea_id,
                generation_index=generation_index,
                kind="MEASUREMENT_OR_OBSERVABLE_SUPPORT",
                description=(
                    "The current realization is not operationalizable. Preserve the ResearchIdea and "
                    "seek measurement/proxy support or a different realization before idea mutation."
                ),
                intent=intent,
                commitments=commitments,
                realization_id=realization_id,
                hypothesis_id=hypothesis_id,
                routes=routes,
                reason_codes=["CURRENT_REALIZATION_NOT_OPERATIONALIZABLE"],
            )
        )

    if directionality and any(token in directionality.upper() for token in ("UNSUPPORTED", "AMBIGUOUS", "UNKNOWN")):
        rows.append(
            _requirement(
                idea_id=idea_id,
                generation_index=generation_index,
                kind="DIRECTIONALITY_SUPPORT",
                description=f"Directionality remains unresolved: {directionality}.",
                intent=intent,
                commitments=commitments,
                realization_id=realization_id,
                hypothesis_id=hypothesis_id,
                routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
                reason_codes=["DIRECTIONALITY_SUPPORT_GAP"],
            )
        )

    for request in evolution_requests:
        scope = str(request.get("requested_mutation_scope") or "")
        if scope == "EVIDENCE_GAP_ESCAPE":
            rows.append(
                _requirement(
                    idea_id=idea_id,
                    generation_index=generation_index,
                    kind="RELATION_OR_MECHANISM_EVIDENCE",
                    description="v2.7 local search exhausted with an evidence-gap escalation request.",
                    intent=intent,
                    commitments=commitments,
                    realization_id=realization_id,
                    hypothesis_id=hypothesis_id,
                    routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
                    reason_codes=["V2_7_EVIDENCE_GAP_ESCAPE"],
                )
            )
        elif scope == "TOPOLOGY_ESCAPE":
            rows.append(
                _requirement(
                    idea_id=idea_id,
                    generation_index=generation_index,
                    kind="GRAPH_TOPOLOGY_COVERAGE",
                    description="v2.7 local search exhausted with a topology-gap escalation request.",
                    intent=intent,
                    commitments=commitments,
                    realization_id=realization_id,
                    hypothesis_id=hypothesis_id,
                    routes=["KG_RETRAVERSAL", "CONTEXT_REBUILD"],
                    reason_codes=["V2_7_TOPOLOGY_ESCAPE"],
                )
            )
        elif scope == "OPERATIONALIZATION_ESCAPE":
            operational.append("V2_7_OPERATIONALIZATION_ESCAPE")
            rows.append(
                _requirement(
                    idea_id=idea_id,
                    generation_index=generation_index,
                    kind="MEASUREMENT_OR_OBSERVABLE_SUPPORT",
                    description=(
                        "v2.7 exhausted local realization search at the operationalization boundary. "
                        "Preserve the ResearchIdea and seek a new observable/proxy realization before idea mutation."
                    ),
                    intent=intent,
                    commitments=commitments,
                    realization_id=realization_id,
                    hypothesis_id=hypothesis_id,
                    routes=["OPERATIONALIZATION_REVIEW"],
                    reason_codes=["V2_7_OPERATIONALIZATION_ESCAPE"],
                )
            )

    dedup: dict[tuple[str, tuple[str, ...]], MissingEvidenceRequirement] = {}
    for row in rows:
        key = (row.kind, tuple(row.recovery_routes))
        prior = dedup.get(key)
        if prior is None:
            dedup[key] = row
            continue
        dedup[key] = prior.model_copy(
            update={
                "source_realization_ids": _dedupe([*prior.source_realization_ids, *row.source_realization_ids]),
                "source_hypothesis_ids": _dedupe([*prior.source_hypothesis_ids, *row.source_hypothesis_ids]),
                "search_terms": _dedupe([*prior.search_terms, *row.search_terms]),
                "proposed_queries": _dedupe([*prior.proposed_queries, *row.proposed_queries]),
                "reason_codes": _dedupe([*prior.reason_codes, *row.reason_codes]),
            }
        )
    return list(dedup.values()), _dedupe(operational)


def _maturity(
    *,
    materialization_status: str,
    premise_count: int,
    has_prediction: bool,
    has_falsifier: bool,
    identifiability: str | None,
    requirements: Sequence[MissingEvidenceRequirement],
) -> tuple[GroundingCoverage, EpistemicMaturity]:
    strict = materialization_status == "MATERIALIZED" and premise_count > 0
    if strict:
        coverage: GroundingCoverage = "STRICT"
    elif premise_count > 0:
        coverage = "PARTIAL"
    else:
        coverage = "NONE"

    if strict and identifiability not in {None, "NOT_OPERATIONALIZABLE"}:
        return coverage, "OPERATIONAL_GROUNDED"
    if strict:
        return coverage, "STRICT_GROUNDED"
    if any("LITERATURE_ACQUISITION" in row.recovery_routes for row in requirements):
        return coverage, "EVIDENCE_SEEKING"
    if has_prediction and has_falsifier:
        return coverage, "SPECULATIVE_BUT_FALSIFIABLE"
    if coverage == "PARTIAL":
        return coverage, "PARTIALLY_GROUNDED"
    return coverage, "IDEA_ONLY"


def build_epistemic_decomposition_report(
    *,
    research_ideas: Sequence[Any],
    lifecycles: Sequence[Any],
    portfolios: Sequence[Any],
    feedback_reports: Sequence[Any] = (),
    evolution_request_report: Any | None = None,
) -> EpistemicDecompositionReport:
    ideas = _idea_payloads(research_ideas)
    cards: dict[str, dict[str, Any]] = {}
    for portfolio in portfolios:
        cards.update(_portfolio_by_hypothesis(portfolio))
    feedback_by_h: dict[str, dict[str, Any]] = {}
    for feedback in feedback_reports:
        feedback_by_h.update(_feedback_by_hypothesis(feedback))
    evolution_by_idea = _evolution_requests_by_idea(evolution_request_report)

    links: list[dict[str, Any]] = []
    for lifecycle in lifecycles:
        links.extend(_rows(lifecycle, "links"))
    links_by_idea: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        idea_id = str(link.get("idea_id") or "")
        if idea_id:
            links_by_idea[idea_id].append(link)

    records: list[EpistemicRealizationRecord] = []
    all_idea_ids = sorted(set(ideas) | set(links_by_idea))
    for idea_id in all_idea_ids:
        node = ideas.get(idea_id, {})
        generation_index = int(node.get("generation_index") or 0)
        intent, commitments, scope, contrast = _kernel_parts(node)
        idea_links = sorted(
            links_by_idea.get(idea_id, []),
            key=lambda row: (int(row.get("attempt_index") or 10**9), str(row.get("realization_id") or "")),
        )
        if not idea_links:
            idea_links = [
                {
                    "realization_id": None,
                    "hypothesis_id": None,
                    "source_context_id": node.get("source_context_id"),
                    "realization_kind": None,
                    "attempt_index": None,
                    "materialization_status": "NOT_ATTEMPTED",
                    "issue_codes": [],
                }
            ]

        for link in idea_links:
            realization_id = str(link.get("realization_id") or "") or None
            hypothesis_id = str(link.get("hypothesis_id") or "") or None
            card = cards.get(hypothesis_id or "", {})
            feedback = feedback_by_h.get(hypothesis_id or "")
            status = str(link.get("materialization_status") or "NOT_ATTEMPTED")
            issue_codes = [str(x) for x in link.get("issue_codes") or []]

            premises = [str(x) for x in card.get("premise_statement_ids") or []]
            papers = [str(x) for x in card.get("source_paper_ids") or []]
            bridge = str(card.get("inferential_bridge") or "").strip() or None
            predictions = [
                str(row.get("observable") or row.get("rationale") or "").strip()
                for row in card.get("predicted_observations") or []
                if isinstance(row, Mapping)
            ]
            falsifiers = [
                str(row.get("falsifying_outcome") or row.get("observable") or "").strip()
                for row in card.get("falsification_criteria") or []
                if isinstance(row, Mapping)
            ]
            if not predictions:
                prediction = str(node.get("differential_prediction") or node.get("discriminating_observation") or "").strip()
                if prediction:
                    predictions = [prediction]
            if not falsifiers:
                falsifier = str(node.get("falsification_condition") or "").strip()
                if falsifier:
                    falsifiers = [falsifier]

            requirements, operational_codes = _derive_requirements(
                idea_id=idea_id,
                generation_index=generation_index,
                intent=intent,
                commitments=commitments,
                realization_id=realization_id,
                hypothesis_id=hypothesis_id,
                materialization_status=status,
                issue_codes=issue_codes,
                feedback=feedback,
                evolution_requests=evolution_by_idea.get(idea_id, []),
            )
            identifiability = (
                str(feedback.get("prospective_identifiability"))
                if feedback and feedback.get("prospective_identifiability") is not None
                else None
            )
            coverage, maturity = _maturity(
                materialization_status=status,
                premise_count=len(premises),
                has_prediction=bool(predictions),
                has_falsifier=bool(falsifiers),
                identifiability=identifiability,
                requirements=requirements,
            )
            rid = _stable_id(
                "epistemic_realization",
                idea_id,
                realization_id,
                hypothesis_id,
                status,
                premises,
                [row.requirement_id for row in requirements],
            )
            records.append(
                EpistemicRealizationRecord(
                    epistemic_realization_id=rid,
                    idea_id=idea_id,
                    generation_index=generation_index,
                    realization_id=realization_id,
                    hypothesis_id=hypothesis_id,
                    source_context_id=(str(link.get("source_context_id")) if link.get("source_context_id") else None),
                    attempt_index=(int(link["attempt_index"]) if link.get("attempt_index") is not None else None),
                    realization_kind=(str(link.get("realization_kind")) if link.get("realization_kind") else None),
                    materialization_status=status,
                    canonical_intent=intent,
                    scientific_commitments=commitments,
                    scope_commitments=scope,
                    contrastive_commitments=contrast,
                    grounded_premise_statement_ids=premises,
                    grounded_source_paper_ids=papers,
                    hypothetical_bridge=bridge,
                    prediction_texts=_dedupe(predictions),
                    falsifier_texts=_dedupe(falsifiers),
                    assumptions=[str(x) for x in card.get("assumptions") or []],
                    prospective_identifiability=identifiability,
                    current_evidence_status=(
                        str(feedback.get("current_evidence_status"))
                        if feedback and feedback.get("current_evidence_status") is not None else None
                    ),
                    directionality_mode=(
                        str(feedback.get("directionality_mode"))
                        if feedback and feedback.get("directionality_mode") is not None else None
                    ),
                    measurement_compatibility_mode=(
                        str(feedback.get("measurement_compatibility_mode"))
                        if feedback and feedback.get("measurement_compatibility_mode") is not None else None
                    ),
                    residual_epistemic_state=(
                        str(feedback.get("residual_epistemic_state"))
                        if feedback and feedback.get("residual_epistemic_state") is not None else None
                    ),
                    residual_state_reason=(
                        str(feedback.get("residual_state_reason"))
                        if feedback and feedback.get("residual_state_reason") is not None else None
                    ),
                    grounding_coverage=coverage,
                    epistemic_maturity=maturity,
                    missing_evidence_requirements=requirements,
                    operationalization_gap_codes=operational_codes,
                    issue_codes=issue_codes,
                )
            )

    maturity_counts = Counter(row.epistemic_maturity for row in records)
    requirements = [req for row in records for req in row.missing_evidence_requirements]
    kind_counts = Counter(row.kind for row in requirements)
    literature_count = sum("LITERATURE_ACQUISITION" in row.recovery_routes for row in requirements)
    kg_count = sum("KG_RETRAVERSAL" in row.recovery_routes for row in requirements)
    provisional = EpistemicDecompositionReport(
        report_id="pending",
        report_sha256="pending",
        records=records,
        record_count=len(records),
        idea_count=len({row.idea_id for row in records}),
        maturity_counts=dict(sorted(maturity_counts.items())),
        requirement_kind_counts=dict(sorted(kind_counts.items())),
        literature_recovery_requirement_count=literature_count,
        kg_retraversal_requirement_count=kg_count,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"epistemic_decomposition:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def _archive_ranks(row: EpistemicRealizationRecord) -> tuple[int, int, int, int]:
    materialization = 2 if row.materialization_status == "MATERIALIZED" else 1 if row.grounding_coverage == "PARTIAL" else 0
    if row.prospective_identifiability is None:
        operational = 1
    elif row.prospective_identifiability == "NOT_OPERATIONALIZABLE":
        operational = 0
    else:
        operational = 2
    evaluated = int(row.prospective_identifiability is not None) + int(row.residual_epistemic_state is not None)
    falsifiability = int(bool(row.prediction_texts and row.falsifier_texts))
    return materialization, operational, min(2, evaluated), falsifiability


def _dominates(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return all(a >= b for a, b in zip(left, right)) and any(a > b for a, b in zip(left, right))


def _slice_key(row: EpistemicRealizationRecord) -> str:
    return "|".join(
        [
            row.residual_epistemic_state or "NO_RESIDUAL",
            row.current_evidence_status or "NO_EVIDENCE_STATUS",
        ]
    )


def build_multi_realization_archive(
    decomposition: EpistemicDecompositionReport,
) -> MultiRealizationArchiveReport:
    by_idea: dict[str, list[EpistemicRealizationRecord]] = defaultdict(list)
    for row in decomposition.records:
        by_idea[row.idea_id].append(row)

    archives: list[IdeaRealizationArchive] = []
    diversity_retained = 0
    for idea_id, rows in sorted(by_idea.items()):
        rows = sorted(rows, key=lambda row: (row.attempt_index or 10**9, row.epistemic_realization_id))
        ranks = {row.epistemic_realization_id: _archive_ranks(row) for row in rows}
        pareto: set[str] = set()
        for row in rows:
            key = row.epistemic_realization_id
            if not any(
                other.epistemic_realization_id != key
                and _dominates(ranks[other.epistemic_realization_id], ranks[key])
                for other in rows
            ):
                pareto.add(key)

        representative_by_slice: dict[str, EpistemicRealizationRecord] = {}
        for row in rows:
            key = _slice_key(row)
            prior = representative_by_slice.get(key)
            if prior is None or ranks[row.epistemic_realization_id] > ranks[prior.epistemic_realization_id]:
                representative_by_slice[key] = row
        diversity = {row.epistemic_realization_id for row in representative_by_slice.values()}
        retained = pareto | diversity
        if len(rows) == 1:
            retained = {rows[0].epistemic_realization_id}

        entries: list[RealizationArchiveEntry] = []
        for row in rows:
            eid = row.epistemic_realization_id
            reasons: list[str] = []
            if len(rows) == 1:
                disposition: ArchiveDisposition = "RETAIN_ONLY_REALIZATION"
                reasons.append("ONLY_REALIZATION_FOR_IDEA")
            elif eid in pareto:
                disposition = "RETAIN_PARETO"
                reasons.append("NONDOMINATED_ON_GROUNDING_OPERATIONALIZATION_EVALUATION_FALSIFIABILITY")
            elif eid in diversity:
                disposition = "RETAIN_EPISTEMIC_DIVERSITY"
                reasons.append("PRESERVES_DISTINCT_RESIDUAL_OR_EVIDENCE_STATE")
                diversity_retained += 1
            else:
                disposition = "DIAGNOSTIC_DOMINATED"
                reasons.append("DOMINATED_WITHIN_ARCHIVE_BUT_PRESERVED_FOR_AUDIT")
            materialization, operational, evaluated, falsifiability = ranks[eid]
            entries.append(
                RealizationArchiveEntry(
                    idea_id=idea_id,
                    generation_index=row.generation_index,
                    epistemic_realization_id=eid,
                    realization_id=row.realization_id,
                    hypothesis_id=row.hypothesis_id,
                    materialization_status=row.materialization_status,
                    grounding_coverage=row.grounding_coverage,
                    epistemic_maturity=row.epistemic_maturity,
                    prospective_identifiability=row.prospective_identifiability,
                    residual_epistemic_state=row.residual_epistemic_state,
                    missing_evidence_requirement_count=len(row.missing_evidence_requirements),
                    prediction_present=bool(row.prediction_texts),
                    falsifier_present=bool(row.falsifier_texts),
                    materialization_rank=materialization,
                    operationalization_rank=operational,
                    evaluation_completeness_rank=evaluated,
                    falsifiability_rank=falsifiability,
                    epistemic_slice_key=_slice_key(row),
                    archive_disposition=disposition,
                    archive_retained=eid in retained,
                    retained_reason_codes=reasons,
                )
            )
        archives.append(
            IdeaRealizationArchive(
                idea_id=idea_id,
                generation_index=rows[0].generation_index,
                entries=entries,
                retained_epistemic_realization_ids=sorted(retained),
                retained_realization_ids=_dedupe([
                    row.realization_id or "" for row in rows if row.epistemic_realization_id in retained
                ]),
                retained_hypothesis_ids=_dedupe([
                    row.hypothesis_id or "" for row in rows if row.epistemic_realization_id in retained
                ]),
                residual_state_diversity=sorted({row.residual_epistemic_state or "NOT_EVALUATED" for row in rows}),
                maturity_diversity=sorted({row.epistemic_maturity for row in rows}),
            )
        )

    provisional = MultiRealizationArchiveReport(
        report_id="pending",
        report_sha256="pending",
        archives=archives,
        idea_count=len(archives),
        realization_entry_count=sum(len(row.entries) for row in archives),
        retained_entry_count=sum(sum(entry.archive_retained for entry in row.entries) for row in archives),
        multi_realization_idea_count=sum(len(row.entries) > 1 for row in archives),
        epistemic_diversity_retention_count=diversity_retained,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"multi_realization_archive:{digest[:20]}",
            "report_sha256": digest,
        }
    )


_RECOVERY_QUERY_STOP = {
    "determine", "whether", "test", "evaluate", "assess", "candidate",
    "association", "effect", "effects", "influence", "influences", "role",
    "controls", "control", "changes", "change", "measured", "measurement",
    "observable", "observables", "evidence", "relation", "mechanism",
    "scientific", "research", "hypothesis", "idea", "using", "based",
    "through", "relative", "primarily", "also", "jointly", "which", "what",
    "how", "its", "their", "this", "that", "from", "with", "into", "under",
}


_RECOVERY_KIND_PRIORITY: dict[EvidenceRequirementKind, int] = {
    # Operationalization/evidence-specific gaps are more informative than the
    # generic fact that a strict positive-premise realization is currently absent.
    "MEASUREMENT_OR_OBSERVABLE_SUPPORT": 6,
    "RELATION_OR_MECHANISM_EVIDENCE": 5,
    "DIRECTIONALITY_SUPPORT": 4,
    "COMPARISON_CONTEXT_SUPPORT": 4,
    "GROUNDING_PREMISE_COVERAGE": 2,
    "REPRESENTATION_EXTENSION": 1,
    "GRAPH_TOPOLOGY_COVERAGE": 0,
}


def _recovery_terms(row: EvidenceRecoveryRequest) -> list[str]:
    values: list[str] = []
    values.extend(row.search_terms)
    for query in row.query_strings:
        values.extend(_tokens(query))
    return _dedupe(
        token.casefold()
        for token in values
        if len(token) >= 4 and token.casefold() not in _RECOVERY_QUERY_STOP
    )


def _recovery_similarity(a: EvidenceRecoveryRequest, b: EvidenceRecoveryRequest) -> float:
    if a.requirement_kind != b.requirement_kind:
        return 0.0
    left = set(_recovery_terms(a))
    right = set(_recovery_terms(b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _request_priority(row: EvidenceRecoveryRequest) -> tuple[int, int, int, str]:
    kind = row.requirement_kind or "GROUNDING_PREMISE_COVERAGE"
    reason_bonus = 0
    reasons = set(row.reason_codes)
    if "V2_7_EVIDENCE_GAP_ESCAPE" in reasons:
        reason_bonus += 3
    if "RESIDUAL_UNRESOLVED_EVIDENCE_GAP" in reasons:
        reason_bonus += 3
    if "CURRENT_REALIZATION_NOT_OPERATIONALIZABLE" in reasons:
        reason_bonus += 2
    if "PROSPECTIVE_EVIDENCE_STATUS_INCOMPLETE" in reasons:
        reason_bonus += 1
    return (
        _RECOVERY_KIND_PRIORITY.get(kind, 0),
        reason_bonus,
        len(set(row.source_realization_ids)),
        row.request_id,
    )


def _cluster_literature_requests(
    requests: Sequence[EvidenceRecoveryRequest],
    *,
    similarity_threshold: float = 0.52,
) -> list[list[EvidenceRecoveryRequest]]:
    """Greedy representative clustering; avoids transitive bridge chaining."""
    ordered = sorted(requests, key=_request_priority, reverse=True)
    clusters: list[list[EvidenceRecoveryRequest]] = []
    for row in ordered:
        best_index = None
        best_similarity = 0.0
        for index, cluster in enumerate(clusters):
            similarity = _recovery_similarity(row, cluster[0])
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_index = index
                best_similarity = similarity
        if best_index is None:
            clusters.append([row])
        else:
            clusters[best_index].append(row)
    return clusters


def _cluster_priority(cluster: Sequence[EvidenceRecoveryRequest]) -> tuple[int, int, int, int, str]:
    representative = max(cluster, key=_request_priority)
    unique_ideas = len({row.idea_id for row in cluster})
    unique_realizations = len({rid for row in cluster for rid in row.source_realization_ids})
    base = _request_priority(representative)
    return (base[0], base[1], unique_ideas, unique_realizations, representative.request_id)


def _select_target_clusters(
    clusters: Sequence[Sequence[EvidenceRecoveryRequest]],
    *,
    target_total: int,
) -> list[list[EvidenceRecoveryRequest]]:
    if target_total < 1:
        raise ValueError("target_total must be >= 1")
    ranked = sorted((list(cluster) for cluster in clusters), key=_cluster_priority, reverse=True)
    if len(ranked) <= target_total:
        return ranked

    # Preserve gap-type diversity first, then fill remaining budget by priority.
    selected: list[list[EvidenceRecoveryRequest]] = []
    used: set[int] = set()
    kinds = [
        "MEASUREMENT_OR_OBSERVABLE_SUPPORT",
        "RELATION_OR_MECHANISM_EVIDENCE",
        "DIRECTIONALITY_SUPPORT",
        "COMPARISON_CONTEXT_SUPPORT",
        "GROUNDING_PREMISE_COVERAGE",
        "REPRESENTATION_EXTENSION",
    ]
    for kind in kinds:
        for index, cluster in enumerate(ranked):
            if index in used:
                continue
            if cluster[0].requirement_kind == kind:
                selected.append(cluster)
                used.add(index)
                break
        if len(selected) >= target_total:
            return selected
    for index, cluster in enumerate(ranked):
        if index in used:
            continue
        selected.append(cluster)
        if len(selected) >= target_total:
            break
    return selected


def _axis_indicators(cluster: Sequence[EvidenceRecoveryRequest]) -> list[str]:
    # M2 require_axis_match is satisfied by any indicator, so prefer specific
    # multi-token phrases over generic singleton terms.
    phrases: list[str] = []
    singletons: list[str] = []
    for row in cluster:
        for query in row.query_strings[:2]:
            terms = [
                token.casefold()
                for token in _tokens(query)
                if len(token) >= 4 and token.casefold() not in _RECOVERY_QUERY_STOP
            ]
            for width in (3, 2):
                for index in range(max(0, len(terms) - width + 1)):
                    phrases.append(" ".join(terms[index : index + width]))
        singletons.extend(_recovery_terms(row))
    specific_phrases = [value for value in _dedupe(phrases) if len(value.split()) >= 2]
    if specific_phrases:
        return specific_phrases[:12]
    return _dedupe(singletons)[:12]


def _targeted_profile(
    *,
    requests: Sequence[EvidenceRecoveryRequest],
    domain_profile_id: str,
    results_per_query: int,
    target_total: int,
) -> tuple[dict[str, Any] | None, list[str], int]:
    literature = [row for row in requests if row.route == "LITERATURE_ACQUISITION"]
    if not literature:
        return None, [], 0
    clusters = _cluster_literature_requests(literature)
    selected_clusters = _select_target_clusters(clusters, target_total=target_total)
    axes = []
    targeted_request_ids: list[str] = []
    for index, cluster in enumerate(selected_clusters, start=1):
        representative = max(cluster, key=_request_priority)
        targeted_request_ids.append(representative.request_id)
        queries = _dedupe(
            query
            for row in cluster
            for query in row.query_strings
        )
        if not queries:
            terms = _dedupe(term for row in cluster for term in row.search_terms)
            queries = [" ".join(terms[:10]) or representative.idea_id]
        axes.append(
            {
                "axis_id": f"idea_gap_{index:03d}_{representative.requirement_id.split(':')[-1][:8]}",
                "target_selected": 1,
                "queries": queries[:3],
                "indicators": _axis_indicators(cluster),
                "weight": 1.0,
            }
        )
    resolved_target = min(max(1, int(target_total)), len(axes))
    profile_id = _stable_id(
        "idea_evidence_recovery_profile",
        domain_profile_id,
        targeted_request_ids,
    )
    return {
        "schema_version": "corpus-acquisition-profile-v1",
        "profile_id": profile_id,
        "domain_profile_id": domain_profile_id,
        "description": (
            "SIS-v2.8 targeted evidence-recovery profile. Retrieved metadata remains candidate-only; "
            "positive-premise eligibility requires the existing acquisition/materialization/extraction pipeline."
        ),
        "discovery": {
            "results_per_query": max(1, min(100, int(results_per_query))),
            "default_providers": ["semantic_scholar", "crossref"],
        },
        "scope": {
            "required_term_groups": [],
            "excluded_title_terms": [],
            "excluded_publication_types": [],
            "min_year": None,
            "max_year": None,
            "require_abstract": False,
            "manual_review_if_no_abstract": True,
            "require_axis_match": True,
        },
        "scoring": {
            "open_access_bonus": 0.5,
            "abstract_available_bonus": 0.5,
            "retrieval_axis_bonus": 1.0,
            "axis_indicator_bonus": 0.5,
            "max_axis_bonus": 2.0,
            "signals": [],
            "citation_bonuses": [],
        },
        "selection": {
            "target_total": resolved_target,
            "include_manual_review": True,
        },
        "axes": axes,
    }, targeted_request_ids, len(clusters)



_RECOVERY_AXIS_GENERIC = {
    "spatial", "distribution", "hotspot", "hotspots", "number", "intensity",
    "relative", "orientation", "geometry", "measured", "measure", "control",
    "controls", "shift", "shifts", "ratio", "effect", "effects", "regime",
    "length", "size", "structure", "dependent", "change", "changes", "signal",
    "response", "activity", "local", "field", "fields", "candidate", "association",
    "determine", "whether", "test", "evaluate", "influence", "influences",
}

_RECOVERY_DOMAIN_GENERIC = {
    *_RECOVERY_AXIS_GENERIC,
    "excitation", "polarization", "molecular", "imaging", "modal", "propagation",
    "wavelength", "frequency", "optical", "spectral", "spectroscopy", "measurement",
    "observable", "mechanism", "coupling", "material", "materials",
}


def _norm_relevance_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


def _profile_axes(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in profile.get("axes") or [] if isinstance(row, Mapping)]


def _profile_domain_anchors(profile: Mapping[str, Any]) -> list[str]:
    axes = _profile_axes(profile)
    axis_token_sets: list[set[str]] = []
    axis_phrase_sets: list[set[str]] = []
    for axis in axes:
        tokens: set[str] = set()
        phrases: set[str] = set()
        values = [*(axis.get("queries") or []), *(axis.get("indicators") or [])]
        for value in values:
            terms = [
                token.casefold()
                for token in _tokens(value)
                if len(token) >= 4
                and token.casefold() not in _RECOVERY_QUERY_STOP
                and token.casefold() not in _RECOVERY_DOMAIN_GENERIC
            ]
            tokens.update(terms)
            for width in (3, 2):
                for index in range(max(0, len(terms) - width + 1)):
                    phrases.add(" ".join(terms[index : index + width]))
        axis_token_sets.append(tokens)
        axis_phrase_sets.append(phrases)

    token_df: Counter[str] = Counter()
    phrase_df: Counter[str] = Counter()
    for values in axis_token_sets:
        token_df.update(values)
    for values in axis_phrase_sets:
        phrase_df.update(values)

    minimum_axes = 2 if len(axes) >= 2 else 1
    anchors = [value for value, count in phrase_df.items() if count >= minimum_axes]
    anchors.extend(value for value, count in token_df.items() if count >= minimum_axes)

    # Domain-profile IDs are weak fallback anchors only.  Short chemistry tokens
    # such as Au/Ag are intentionally ignored because they are too ambiguous.
    domain_profile_id = str(profile.get("domain_profile_id") or "")
    for value in re.split(r"[^A-Za-z0-9]+", domain_profile_id):
        value = value.casefold().strip()
        if len(value) >= 4 and value not in _RECOVERY_DOMAIN_GENERIC:
            anchors.append(value)
    return sorted(_dedupe(anchors), key=lambda value: (-len(value.split()), -len(value), value))


def _axis_specific_phrases(profile: Mapping[str, Any], axis_id: str) -> list[str]:
    axis = next((row for row in _profile_axes(profile) if str(row.get("axis_id")) == axis_id), None)
    if axis is None:
        return []
    candidates: list[str] = []
    for value in [*(axis.get("indicators") or []), *(axis.get("queries") or [])]:
        terms = [
            token.casefold()
            for token in _tokens(value)
            if len(token) >= 4 and token.casefold() not in _RECOVERY_QUERY_STOP
        ]
        for width in (3, 2):
            for index in range(max(0, len(terms) - width + 1)):
                phrase_terms = terms[index : index + width]
                if all(term in _RECOVERY_AXIS_GENERIC for term in phrase_terms):
                    continue
                candidates.append(" ".join(phrase_terms))
    return _dedupe(candidates)[:24]


def _axis_recovery_request_map(
    *,
    acquisition_profile: Mapping[str, Any],
    recovery_plan: EvidenceRecoveryPlan | None,
) -> dict[str, EvidenceRecoveryRequest]:
    if recovery_plan is None:
        return {}
    axes = _profile_axes(acquisition_profile)
    request_ids = list(recovery_plan.targeted_literature_representative_request_ids)
    if len(axes) != len(request_ids):
        return {}
    by_id = {row.request_id: row for row in recovery_plan.requests}
    mapped: dict[str, EvidenceRecoveryRequest] = {}
    for axis, request_id in zip(axes, request_ids):
        row = by_id.get(request_id)
        axis_id = str(axis.get("axis_id") or "")
        if axis_id and row is not None:
            mapped[axis_id] = row
    return mapped


_RECOVERY_REQUEST_SETUP_TERMS = {
    # Experimental/setup vocabulary can identify a neighboring paper without
    # establishing the missing scientific relation itself.  Keep these terms
    # available to literature retrieval, but do not let them alone satisfy the
    # request-specific gap test used for automatic M3 acquisition.
    "excitation", "wavelength", "frequency", "optical", "spectral",
    "measurement", "observable", "observables",
}

_RECOVERY_WEAK_GAP_TERMS = {
    # These broad endpoint words are useful ranking signals but are too weak to
    # establish request-specific evidence on their own.
    "intensity", "signal", "response", "activity", "yield", "ratio",
}


def _relevance_root(value: str) -> str:
    token = str(value or "").casefold().strip("-_ ")
    if len(token) <= 4:
        return token
    irregular = {
        "hotspots": "hotspot",
        "geometries": "geometry",
        "nanostructures": "nanostructure",
        "measurements": "measurement",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith("ion") and len(token) - 3 >= 4:
        return token[:-3]
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _relevance_roots(value: str) -> set[str]:
    return {_relevance_root(token) for token in _tokens(value) if _relevance_root(token)}


_RECOVERY_ROLE_CUES: dict[str, tuple[str, ...]] = {
    "MEASUREMENT_OR_OBSERVABLE_SUPPORT": (
        "measure", "measurement", "measured", "detect", "detection", "sensor",
        "sensing", "spectroscopy", "imaging", "intensity", "ratio", "yield",
        "polarization", "observable",
    ),
    "RELATION_OR_MECHANISM_EVIDENCE": (
        "mechanism", "coupling", "mediate", "mediated", "dependent", "dependence",
        "control", "controls", "influence", "effect", "relationship", "correlation",
        "govern", "modulate", "enhance", "redistribution",
    ),
    "DIRECTIONALITY_SUPPORT": (
        "increase", "decrease", "enhance", "suppress", "promote", "reduce",
        "control", "influence", "dependent", "correlation",
    ),
    "COMPARISON_CONTEXT_SUPPORT": (
        "compare", "compared", "comparison", "versus", "relative", "ratio",
        "matched", "contrast", "different", "difference",
    ),
    "GROUNDING_PREMISE_COVERAGE": (
        "reported", "observed", "measured", "associated", "dependent", "relationship",
        "effect", "activity", "response", "detection",
    ),
}


def _request_gap_terms(
    request: EvidenceRecoveryRequest,
    *,
    domain_anchors: Sequence[str],
) -> list[str]:
    domain_tokens = {
        token.casefold()
        for anchor in domain_anchors
        for token in _tokens(anchor)
        if len(token) >= 4
    }
    values = [*request.search_terms, *request.query_strings]
    terms: list[str] = []
    for value in values:
        terms.extend(
            token.casefold()
            for token in _tokens(value)
            if len(token) >= 4
            and token.casefold() not in _RECOVERY_QUERY_STOP
            and token.casefold() not in domain_tokens
            and token.casefold() not in _RECOVERY_REQUEST_SETUP_TERMS
        )
    return _dedupe(terms)[:32]


def _request_gap_evidence(
    combined: str,
    request: EvidenceRecoveryRequest,
    *,
    domain_anchors: Sequence[str],
) -> tuple[list[str], float, bool]:
    terms = _request_gap_terms(request, domain_anchors=domain_anchors)
    if not terms:
        return [], 0.0, False
    combined_roots = _relevance_roots(combined)
    term_roots = {term: _relevance_root(term) for term in terms}
    singleton_hits = [
        term for term, root in term_roots.items()
        if root and root in combined_roots
    ]
    coverage = len(set(singleton_hits)) / len(set(terms)) if terms else 0.0
    multi_hits: list[str] = []
    # Search only phrases composed of request-specific, non-domain/setup terms.
    # Exact phrase matches are useful confidence evidence, but are not required
    # when several scientifically specific singleton concepts match.
    for query in request.query_strings:
        qterms = [
            token.casefold()
            for token in _tokens(query)
            if token.casefold() in set(terms)
        ]
        for width in (3, 2):
            for index in range(max(0, len(qterms) - width + 1)):
                phrase = " ".join(qterms[index : index + width])
                if phrase and _norm_relevance_text(phrase) in combined:
                    multi_hits.append(phrase)
    hits = _dedupe([*multi_hits, *singleton_hits])
    meaningful_singletons = {
        term for term in singleton_hits
        if term not in _RECOVERY_WEAK_GAP_TERMS
    }
    phrase_has_meaningful_term = any(
        any(term not in _RECOVERY_WEAK_GAP_TERMS for term in phrase.split())
        for phrase in multi_hits
    )
    # Calibration boundary: setup-only matches (for example merely
    # ``excitation wavelength``) and generic endpoints (for example merely
    # ``intensity``) stay adjacent.  Two request-specific concepts, or a
    # meaningful request phrase, are enough to count as strong gap evidence.
    strong = (
        len(meaningful_singletons) >= 2
        or phrase_has_meaningful_term
        or (coverage >= 0.50 and bool(meaningful_singletons))
    )
    return hits, round(coverage, 6), strong


def _scientific_role_hits(
    combined: str,
    request: EvidenceRecoveryRequest,
) -> list[str]:
    kind = str(request.requirement_kind or "GROUNDING_PREMISE_COVERAGE")
    cues = _RECOVERY_ROLE_CUES.get(kind, _RECOVERY_ROLE_CUES["GROUNDING_PREMISE_COVERAGE"])
    return [cue for cue in cues if cue in combined]


def build_recovery_relevance_gate(
    *,
    acquisition_profile: Mapping[str, Any],
    catalog: Mapping[str, Any],
    assessments: Sequence[Mapping[str, Any]],
    max_strict_total: int,
    recovery_plan: EvidenceRecoveryPlan | None = None,
) -> RecoveryRelevanceGateReport:
    """Classify M2-eligible recovery literature before source acquisition.

    With a recovery plan, STRICT is request-conditioned: shared domain context
    and meaningful request-specific gap evidence must coincide for at least one
    matched acquisition axis.  Scientific-role compatibility is retained as a
    confidence/ranking booster, not a mandatory gate.  Domain-only neighbors are
    preserved as ADJACENT instead of being acquired merely because they contain
    broad words such as ``plasmonic`` or ``nanostructure``.
    """
    if max_strict_total < 1:
        raise ValueError("max_strict_total must be >= 1")
    works = {
        str(row.get("work_id")): dict(row)
        for row in catalog.get("works") or []
        if isinstance(row, Mapping) and row.get("work_id")
    }
    include_manual = bool((acquisition_profile.get("selection") or {}).get("include_manual_review"))
    allowed_status = {"eligible"}
    if include_manual:
        allowed_status.add("manual_review")
    anchors = _profile_domain_anchors(acquisition_profile)
    request_by_axis = _axis_recovery_request_map(
        acquisition_profile=acquisition_profile,
        recovery_plan=recovery_plan,
    )
    request_conditioning = bool(request_by_axis)
    records: list[RecoveryCandidateRelevanceRecord] = []

    for assessment in assessments:
        status = str(assessment.get("eligibility_status") or "")
        if status not in allowed_status:
            continue
        work_id = str(assessment.get("work_id") or "")
        work = works.get(work_id)
        if work is None:
            continue
        combined = _norm_relevance_text(
            " ".join(
                value
                for value in [work.get("title"), work.get("abstract")]
                if value
            )
        )
        matched_axes = [str(value) for value in assessment.get("matched_axes") or []]
        domain_hits = [anchor for anchor in anchors if _norm_relevance_text(anchor) in combined]
        axis_phrase_hits: list[str] = []
        for axis_id in matched_axes:
            for phrase in _axis_specific_phrases(acquisition_profile, axis_id):
                if _norm_relevance_text(phrase) in combined:
                    axis_phrase_hits.append(phrase)
        axis_phrase_hits = _dedupe(axis_phrase_hits)

        matched_request_ids: list[str] = []
        matched_kinds: list[str] = []
        request_gap_hits: list[str] = []
        role_hits: list[str] = []
        max_coverage = 0.0
        request_conditioned_strict = False
        for axis_id in matched_axes:
            request = request_by_axis.get(axis_id)
            if request is None:
                continue
            matched_request_ids.append(request.request_id)
            if request.requirement_kind is not None:
                matched_kinds.append(str(request.requirement_kind))
            gap_hits, coverage, strong_gap = _request_gap_evidence(
                combined,
                request,
                domain_anchors=anchors,
            )
            request_gap_hits.extend(gap_hits)
            max_coverage = max(max_coverage, coverage)
            current_role_hits = _scientific_role_hits(combined, request)
            role_hits.extend(current_role_hits)
            meaningful_gap_hits = [
                hit for hit in gap_hits
                if " " in hit or hit not in _RECOVERY_WEAK_GAP_TERMS
            ]
            role_boosted_gap = bool(meaningful_gap_hits and current_role_hits)
            domain_reinforced_gap = bool(
                meaningful_gap_hits
                and len(set(domain_hits)) >= 2
            )
            if domain_hits and (strong_gap or role_boosted_gap or domain_reinforced_gap):
                request_conditioned_strict = True

        if request_conditioning:
            if request_conditioned_strict:
                relevance: RecoveryRelevanceClass = "STRICT_DOMAIN_RELEVANT"
                reasons = [
                    "RECOVERY_AXIS_MATCH",
                    "SHARED_DOMAIN_CONTEXT_MATCH",
                    "REQUEST_SPECIFIC_GAP_MATCH",
                    *( ["SCIENTIFIC_ROLE_MATCH_CONFIDENCE_BOOST"] if role_hits else [] ),
                ]
            elif matched_axes and (domain_hits or (request_gap_hits and role_hits) or axis_phrase_hits):
                relevance = "ADJACENT_METHOD_OR_MECHANISM"
                reasons = ["RECOVERY_AXIS_MATCH", "INSUFFICIENT_REQUEST_CONDITIONED_STRICT_EVIDENCE"]
            else:
                relevance = "OFF_DOMAIN_REJECT"
                reasons = ["GENERIC_OR_OFF_DOMAIN_LEXICAL_COLLISION"]
        else:
            # Backward-compatible diagnostic fallback for old artifacts/tests that
            # do not carry targeted request lineage.  Production v2.8 runner passes
            # the recovery plan and therefore uses request-conditioned semantics.
            if matched_axes and domain_hits:
                relevance = "STRICT_DOMAIN_RELEVANT"
                reasons = ["RECOVERY_AXIS_MATCH", "SHARED_DOMAIN_CONTEXT_MATCH", "REQUEST_CONDITIONING_UNAVAILABLE_FALLBACK"]
            elif matched_axes and axis_phrase_hits:
                relevance = "ADJACENT_METHOD_OR_MECHANISM"
                reasons = ["RECOVERY_AXIS_MATCH", "SPECIFIC_AXIS_PHRASE_WITHOUT_SHARED_DOMAIN_CONTEXT", "REQUEST_CONDITIONING_UNAVAILABLE_FALLBACK"]
            else:
                relevance = "OFF_DOMAIN_REJECT"
                reasons = ["GENERIC_OR_OFF_DOMAIN_LEXICAL_COLLISION", "REQUEST_CONDITIONING_UNAVAILABLE_FALLBACK"]

        records.append(
            RecoveryCandidateRelevanceRecord(
                work_id=work_id,
                title=str(work.get("title") or assessment.get("title") or ""),
                doi=(str(work.get("doi")) if work.get("doi") else None),
                year=(int(work["year"]) if work.get("year") is not None else None),
                original_eligibility_status=status,
                original_total_score=float(assessment.get("total_score") or 0.0),
                matched_axes=matched_axes,
                relevance_class=relevance,
                domain_anchor_hits=_dedupe(domain_hits),
                specific_axis_phrase_hits=axis_phrase_hits,
                matched_recovery_request_ids=_dedupe(matched_request_ids),
                matched_requirement_kinds=_dedupe(matched_kinds),
                request_specific_gap_hits=_dedupe(request_gap_hits),
                scientific_role_hits=_dedupe(role_hits),
                max_request_gap_coverage=max_coverage,
                request_conditioned_strict_match=request_conditioned_strict,
                reason_codes=reasons,
                acquisition_authority=(relevance == "STRICT_DOMAIN_RELEVANT"),
            )
        )

    strict_rows = [row for row in records if row.relevance_class == "STRICT_DOMAIN_RELEVANT"]
    strict_rows = sorted(
        strict_rows,
        key=lambda row: (
            -int(row.request_conditioned_strict_match),
            -row.max_request_gap_coverage,
            -len(row.request_specific_gap_hits),
            -len(row.scientific_role_hits),
            -len(row.domain_anchor_hits),
            -row.original_total_score,
            -(int(works.get(row.work_id, {}).get("citation_count") or 0)),
            row.title.casefold(),
            row.work_id,
        ),
    )
    strict_selected = strict_rows[:max_strict_total]
    adjacent_ids = sorted(
        row.work_id for row in records if row.relevance_class == "ADJACENT_METHOD_OR_MECHANISM"
    )
    rejected_ids = sorted(
        row.work_id for row in records if row.relevance_class == "OFF_DOMAIN_REJECT"
    )
    provisional = RecoveryRelevanceGateReport(
        report_id="pending",
        report_sha256="pending",
        profile_id=str(acquisition_profile.get("profile_id") or ""),
        source_catalog_id=str(catalog.get("catalog_id") or ""),
        max_strict_acquisition_total=max_strict_total,
        original_candidate_count=len(works),
        assessed_relevance_candidate_count=len(records),
        strict_candidate_count=len(strict_rows),
        adjacent_candidate_count=len(adjacent_ids),
        off_domain_reject_count=len(rejected_ids),
        strict_selected_count=len(strict_selected),
        strict_selected_work_ids=[row.work_id for row in strict_selected],
        adjacent_candidate_work_ids=adjacent_ids,
        off_domain_reject_work_ids=rejected_ids,
        derived_domain_anchors=anchors,
        records=records,
        request_conditioning_applied=request_conditioning,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"recovery_relevance_gate:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def build_evidence_recovery_plan(
    decomposition: EpistemicDecompositionReport,
    *,
    domain_profile_id: str,
    results_per_query: int = 30,
    target_total: int = 12,
) -> EvidenceRecoveryPlan:
    requests: list[EvidenceRecoveryRequest] = []
    retry_ids: set[str] = set()
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in decomposition.records:
        for requirement in record.missing_evidence_requirements:
            for route in requirement.recovery_routes:
                key = (requirement.idea_id, requirement.kind, route)
                bucket = grouped.setdefault(
                    key,
                    {
                        "requirement_ids": [],
                        "generation_index": requirement.generation_index,
                        "query_strings": [],
                        "search_terms": [],
                        "source_realization_ids": [],
                        "reason_codes": [],
                    },
                )
                bucket["requirement_ids"].append(requirement.requirement_id)
                bucket["query_strings"].extend(requirement.proposed_queries)
                bucket["search_terms"].extend(requirement.search_terms)
                bucket["source_realization_ids"].extend(requirement.source_realization_ids)
                bucket["reason_codes"].extend(requirement.reason_codes)

    for (idea_id, requirement_kind, route), bucket in sorted(grouped.items()):
        promotion = route == "LITERATURE_ACQUISITION"
        status: EvidenceRecoveryStatus
        if route == "LITERATURE_ACQUISITION":
            status = "DISCOVERY_READY"
            retry_ids.add(idea_id)
        elif route in {"KG_RETRAVERSAL", "CONTEXT_REBUILD"}:
            status = "CONTEXT_REBUILD_REQUIRED"
            retry_ids.add(idea_id)
        else:
            status = "REQUESTED"
        merged_requirement_id = _stable_id(
            "merged_evidence_requirement", idea_id, requirement_kind, route, sorted(bucket["requirement_ids"])
        )
        requests.append(
            EvidenceRecoveryRequest(
                request_id=_stable_id("evidence_recovery_request", merged_requirement_id, route),
                requirement_id=merged_requirement_id,
                requirement_kind=requirement_kind,
                idea_id=idea_id,
                generation_index=int(bucket["generation_index"]),
                route=route,
                query_strings=_dedupe(bucket["query_strings"]),
                search_terms=_dedupe(bucket["search_terms"]),
                source_realization_ids=_dedupe(bucket["source_realization_ids"]),
                status=status,
                reason_codes=_dedupe(bucket["reason_codes"]),
                positive_evidence_promotion_required_before_retry=promotion,
            )
        )
    counts = Counter(row.route for row in requests)
    profile, targeted_request_ids, cluster_count = _targeted_profile(
        requests=requests,
        domain_profile_id=domain_profile_id,
        results_per_query=results_per_query,
        target_total=target_total,
    )
    provisional = EvidenceRecoveryPlan(
        plan_id="pending",
        plan_sha256="pending",
        requests=requests,
        request_count=len(requests),
        route_counts=dict(sorted(counts.items())),
        literature_request_count=counts.get("LITERATURE_ACQUISITION", 0),
        literature_cluster_count=cluster_count,
        literature_requests_collapsed_by_clustering_count=max(
            0,
            counts.get("LITERATURE_ACQUISITION", 0) - cluster_count,
        ),
        targeted_literature_axis_count=len(targeted_request_ids),
        targeted_literature_representative_request_ids=targeted_request_ids,
        literature_clusters_suppressed_by_budget_count=max(
            0,
            cluster_count - len(targeted_request_ids),
        ),
        recovery_target_budget=max(1, int(target_total)),
        kg_retraversal_request_count=counts.get("KG_RETRAVERSAL", 0),
        operationalization_review_request_count=counts.get("OPERATIONALIZATION_REVIEW", 0),
        representation_review_request_count=counts.get("REPRESENTATION_REVIEW", 0),
        targeted_acquisition_profile=profile,
        retry_candidate_idea_ids=sorted(retry_ids),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("plan_id", None)
    payload.pop("plan_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "plan_id": f"evidence_recovery_plan:{digest[:20]}",
            "plan_sha256": digest,
        }
    )


def build_evidence_recovery_command_plan(
    *,
    profile_path: str,
    output_root: str,
    providers: str = "semantic_scholar,crossref",
    results_per_query: int = 30,
    python_executable: str = "python",
) -> EvidenceRecoveryCommandPlan:
    catalog_dir = f"{output_root.rstrip('/')}/m1_catalog"
    selection_dir = f"{output_root.rstrip('/')}/m2_selection"
    relevance_dir = f"{output_root.rstrip('/')}/m2_5_relevance"
    return EvidenceRecoveryCommandPlan(
        profile_path=profile_path,
        output_root=output_root,
        discovery_command=[
            python_executable, "-m", "scripts.literature.discover_literature_catalog",
            "--profile", profile_path,
            "--output-dir", catalog_dir,
            "--providers", providers,
            "--results-per-query", str(results_per_query),
        ],
        selection_command=[
            python_executable, "-m", "scripts.literature.select_corpus_candidates",
            "--profile", profile_path,
            "--catalog", f"{catalog_dir}/catalog.json",
            "--output-dir", selection_dir,
        ],
        relevance_gate_report_path=f"{relevance_dir}/relevance_gate_report.json",
        strict_selected_works_path=f"{relevance_dir}/strict_selected_works.jsonl",
        strict_selection_report_path=f"{relevance_dir}/strict_selection_report.json",
        adjacent_candidate_archive_path=f"{relevance_dir}/adjacent_candidates.jsonl",
        downstream_promotion_note=(
            "Continue with the repository's existing M3/M4 extraction/publication or knowledge-backfill pipeline "
            "using m2_5_relevance/strict_selected_works.jsonl and strict_selection_report.json, not the raw M2 "
            "selection. ADJACENT candidates remain archived for method/mechanism inspiration only. Metadata "
            "discovery/selection alone MUST NOT be injected into HypothesisContext. A rebuilt context is eligible "
            "for retry only after provenance-preserving positive-evidence promotion."
        ),
        retry_note=(
            "After the trusted corpus/KG promotion path rebuilds HypothesisContext, rerun the SIS-v2.8 runner with "
            "--recovered-context pointing at that new context. The retry remains the same ResearchIdea unless the "
            "existing semantic comparator says otherwise."
        ),
    )


__all__ = [
    "EpistemicDecompositionReport",
    "EpistemicRealizationRecord",
    "EvidenceRecoveryCommandPlan",
    "EvidenceRecoveryPlan",
    "EvidenceRecoveryRequest",
    "IdeaRealizationArchive",
    "MissingEvidenceRequirement",
    "MultiRealizationArchiveReport",
    "RealizationArchiveEntry",
    "RecoveryCandidateRelevanceRecord",
    "RecoveryRelevanceGateReport",
    "build_epistemic_decomposition_report",
    "build_recovery_relevance_gate",
    "build_evidence_recovery_command_plan",
    "build_evidence_recovery_plan",
    "build_multi_realization_archive",
]
