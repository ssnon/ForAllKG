from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
)
from dac_her.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolio,
)
from dac_her.premise_role_necessity import (
    CriticPrompt,
    InstructorPremiseCriticBackend,
    PremiseRoleNecessityReport,
)


DECOMPOSITION_PROMPT_VERSION = (
    "hypothesis-clause-decomposer-v2.9.1-ps3"
)
COVERAGE_PROMPT_VERSION = (
    "hypothesis-clause-coverage-auditor-v2.9.1-ps3"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ClauseType = Literal[
    "central_relation",
    "mechanistic_mediator",
    "comparative_or_conditional",
    "consequence_or_prediction",
    "design_implication",
    "scope_or_qualifier",
]

ClauseMateriality = Literal["core", "supporting"]

CoverageStatus = Literal[
    "directly_grounded",
    "synthesis_grounded",
    "hypothesized_bridge",
    "unsupported_extension",
    "contradicted_or_limited",
    "uncertain",
]

Confidence = Literal["low", "medium", "high"]

OverallGroundingVerdict = Literal[
    "grounded_extension",
    "testable_but_under_grounded_extension",
    "unsupported_inferential_leap",
    "scope_conflicted",
    "uncertain",
]


class HypothesisClauseUnitDraft(StrictModel):
    local_id: str
    text: str = Field(min_length=1)
    clause_type: ClauseType
    materiality: ClauseMateriality


class BridgeUnitDraft(StrictModel):
    local_id: str
    text: str = Field(min_length=1)
    materiality: ClauseMateriality


class HypothesisClauseDecompositionDraft(StrictModel):
    hypothesis_id: str
    hypothesis_clauses: list[HypothesisClauseUnitDraft] = Field(
        min_length=1
    )
    bridge_units: list[BridgeUnitDraft] = Field(default_factory=list)


class ClauseCoverageReviewDraft(StrictModel):
    local_id: str
    status: CoverageStatus

    supporting_premise_statement_ids: list[str] = Field(
        default_factory=list
    )
    limiting_premise_statement_ids: list[str] = Field(
        default_factory=list
    )

    support_explanation: str = Field(min_length=1)
    missing_relation_or_scope: str | None = None
    confidence: Confidence


class HypothesisClauseCoverageDraft(StrictModel):
    hypothesis_id: str
    hypothesis_clause_reviews: list[ClauseCoverageReviewDraft] = Field(
        min_length=1
    )
    bridge_unit_reviews: list[ClauseCoverageReviewDraft] = Field(
        default_factory=list
    )
    critical_missing_links: list[str] = Field(default_factory=list)


class ExactClauseAudit(StrictModel):
    local_id: str
    text: str
    source_field: Literal[
        "hypothesis_statement",
        "inferential_bridge",
    ]
    exact_substring_match: bool


class AuditedHypothesisClause(StrictModel):
    local_id: str
    text: str
    clause_type: ClauseType
    materiality: ClauseMateriality
    quote_audit: ExactClauseAudit


class AuditedBridgeUnit(StrictModel):
    local_id: str
    text: str
    materiality: ClauseMateriality
    quote_audit: ExactClauseAudit


class AuditedClauseCoverageReview(StrictModel):
    local_id: str
    status: CoverageStatus

    supporting_premise_statement_ids: list[str] = Field(
        default_factory=list
    )
    limiting_premise_statement_ids: list[str] = Field(
        default_factory=list
    )

    support_explanation: str
    missing_relation_or_scope: str | None = None
    confidence: Confidence


class PS2HypothesisSnapshot(StrictModel):
    premise_verdicts: dict[str, str] = Field(default_factory=dict)
    role_labels: dict[str, str] = Field(default_factory=dict)
    ablation_statuses: dict[str, str] = Field(default_factory=dict)


class HypothesisClauseCoverageCard(StrictModel):
    hypothesis_id: str
    title: str
    axis_id: str
    axis_label: str

    selected_premise_statement_ids: list[str] = Field(default_factory=list)

    hypothesis_clauses: list[AuditedHypothesisClause] = Field(
        default_factory=list
    )
    bridge_units: list[AuditedBridgeUnit] = Field(default_factory=list)

    hypothesis_clause_reviews: list[
        AuditedClauseCoverageReview
    ] = Field(default_factory=list)
    bridge_unit_reviews: list[
        AuditedClauseCoverageReview
    ] = Field(default_factory=list)

    overall_verdict: OverallGroundingVerdict
    verdict_reason: str

    directly_grounded_count: int = 0
    synthesis_grounded_count: int = 0
    hypothesized_bridge_count: int = 0
    unsupported_extension_count: int = 0
    contradicted_or_limited_count: int = 0
    uncertain_count: int = 0

    critical_missing_links: list[str] = Field(default_factory=list)

    ps2_snapshot: PS2HypothesisSnapshot | None = None

    decomposition_prompt_sha256: str
    coverage_prompt_sha256: str


class HypothesisClauseCoveragePolicy(StrictModel):
    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False

    decomposition_is_evidence_blind: Literal[True] = True
    coverage_uses_selected_premise_full_set: Literal[True] = True
    discovery_axis_is_scientific_evidence: Literal[False] = False
    external_knowledge_allowed: Literal[False] = False

    clause_quotes_must_be_exact_substrings: Literal[True] = True
    support_ids_must_be_selected_premises: Literal[True] = True

    hypothesized_bridge_is_scientific_truth: Literal[False] = False
    critic_verdict_is_scientific_truth: Literal[False] = False


class HypothesisClauseCoverageReport(StrictModel):
    schema_version: Literal[
        "hypothesis-clause-coverage-report-v1"
    ] = "hypothesis-clause-coverage-report-v1"

    report_id: str
    report_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_axis_plan_id: str
    source_axis_report_id: str
    source_ps2_report_sha256: str | None = None

    domain_profile_id: str
    corpus_id: str

    critic_model: str
    decomposition_prompt_version: str
    coverage_prompt_version: str

    evaluated_hypothesis_ids: list[str] = Field(default_factory=list)
    hypothesis_count: int
    llm_call_count: int

    overall_verdict_counts: dict[str, int] = Field(default_factory=dict)
    clause_status_counts: dict[str, int] = Field(default_factory=dict)
    bridge_status_counts: dict[str, int] = Field(default_factory=dict)

    invalid_clause_quote_count: int = 0
    invalid_support_reference_count: int = 0

    cards: list[HypothesisClauseCoverageCard] = Field(default_factory=list)

    policy: HypothesisClauseCoveragePolicy = Field(
        default_factory=HypothesisClauseCoveragePolicy
    )


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _make_prompt(
    version: str,
    system_prompt: str,
    user_prompt: str,
) -> CriticPrompt:
    payload = {
        "prompt_version": version,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    return CriticPrompt(
        **payload,
        prompt_sha256=_sha256_json(payload),
    )


def build_decomposition_prompt(
    hypothesis: HypothesisCard,
) -> CriticPrompt:
    system = """You are an evidence-blind scientific claim decomposer.

You will receive ONE accepted hypothesis and its inferential bridge. You will
NOT receive any evidence. Decompose the hypothesis statement into the smallest
scientifically meaningful material claim units needed to audit grounding later.

Rules:
- Every hypothesis_clauses[i].text MUST be a short contiguous VERBATIM
  substring of hypothesis_statement.
- Every bridge_units[i].text MUST be a short contiguous VERBATIM substring of
  inferential_bridge.
- Do not paraphrase, normalize punctuation, or invent words.
- Do not tailor the decomposition to what evidence might exist; no evidence is
  available to you.
- Split distinct causal/mechanistic relations, consequences, and design
  implications when they can fail independently.
- Mark a clause as core when failure of that clause would materially change the
  scientific hypothesis. Mark descriptive qualifiers as supporting when
  appropriate.
- Use unique local IDs.
"""
    user = f"""HYPOTHESIS
ID: {hypothesis.hypothesis_id}
Title: {hypothesis.title}

hypothesis_statement:
{hypothesis.hypothesis_statement}

inferential_bridge:
{hypothesis.inferential_bridge}

Return the structured evidence-blind decomposition only.
"""
    return _make_prompt(
        DECOMPOSITION_PROMPT_VERSION,
        system,
        user,
    )


def build_coverage_prompt(
    *,
    hypothesis: HypothesisCard,
    axis: DiscoveryAxis,
    hypothesis_clauses: list[AuditedHypothesisClause],
    bridge_units: list[AuditedBridgeUnit],
    premises: list[HypothesisEvidenceStatement],
) -> CriticPrompt:
    system = """You are a full-set scientific grounding and inferential-leap auditor.

The hypothesis was decomposed BEFORE evidence was shown. Audit each fixed claim
unit against the COMPLETE SELECTED PREMISE SET.

Use ONLY the supplied premises. External knowledge is forbidden. The discovery
axis is inspiration-only and is NOT scientific evidence.

Coverage statuses:
- directly_grounded:
  At least one selected premise directly supports the clause as written with
  compatible scope. Do not use this for mere topical similarity.
- synthesis_grounded:
  Multiple selected premises jointly support the clause as written WITHOUT
  introducing a new unobserved causal/mechanistic relation.
- hypothesized_bridge:
  The premises support relevant endpoints, observations, or contextual pieces,
  but the specific relation/mechanism asserted by the clause is not directly
  established. This can be a legitimate testable hypothesis, but must be marked
  as a hypothesis rather than corpus-supported fact.
- unsupported_extension:
  The clause introduces a material entity, mechanism, direction, comparison,
  outcome, or scope that is not adequately grounded even as a synthesis of the
  selected premises.
- contradicted_or_limited:
  A selected premise explicitly limits, cautions against, or conflicts with the
  clause or the scope transfer required to support it.
- uncertain:
  The supplied text is insufficient for a reliable classification.

Important distinctions:
- "Removing a premise breaks the chain" does NOT imply that the full chain is
  grounded.
- A premise that supports one endpoint of a causal link does not automatically
  support the link itself.
- If a premise explicitly says a direct structural/reaction relation is not
  established, respect that limitation.
- Do not reward a hypothesis for being plausible; audit only supplied support.
- supporting_premise_statement_ids and limiting_premise_statement_ids may use
  ONLY IDs from SELECTED PREMISES.
- Return exactly one review for every supplied hypothesis clause and bridge
  unit, preserving local IDs exactly.
"""
    clause_payload = [
        {
            "local_id": row.local_id,
            "text": row.text,
            "clause_type": row.clause_type,
            "materiality": row.materiality,
        }
        for row in hypothesis_clauses
    ]
    bridge_payload = [
        {
            "local_id": row.local_id,
            "text": row.text,
            "materiality": row.materiality,
        }
        for row in bridge_units
    ]
    premise_payload = [
        {
            "statement_id": row.statement_id,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
            "text": row.text,
        }
        for row in premises
    ]

    user = f"""HYPOTHESIS
ID: {hypothesis.hypothesis_id}
Title: {hypothesis.title}

hypothesis_statement:
{hypothesis.hypothesis_statement}

inferential_bridge:
{hypothesis.inferential_bridge}

ASSIGNED DISCOVERY AXIS — INSPIRATION ONLY
ID: {axis.axis_id}
Label: {axis.label}
Subject: {axis.proposed_subject}
Relation: {axis.proposed_relation}
Object: {axis.proposed_object}

FIXED EVIDENCE-BLIND HYPOTHESIS CLAUSES
{json.dumps(clause_payload, ensure_ascii=False, indent=2)}

FIXED EVIDENCE-BLIND BRIDGE UNITS
{json.dumps(bridge_payload, ensure_ascii=False, indent=2)}

SELECTED PREMISES — THIS IS THE COMPLETE SCIENTIFIC EVIDENCE SET FOR THIS AUDIT
{json.dumps(premise_payload, ensure_ascii=False, indent=2)}

Audit every fixed unit. Distinguish grounded observations from newly proposed
relations, mechanisms, and scope transfers.
"""
    return _make_prompt(
        COVERAGE_PROMPT_VERSION,
        system,
        user,
    )


def _audit_decomposition(
    draft: HypothesisClauseDecompositionDraft,
    hypothesis: HypothesisCard,
) -> tuple[
    list[AuditedHypothesisClause],
    list[AuditedBridgeUnit],
    int,
]:
    clause_ids = [row.local_id for row in draft.hypothesis_clauses]
    bridge_ids = [row.local_id for row in draft.bridge_units]
    if len(clause_ids) != len(set(clause_ids)):
        raise ValueError("PS3 duplicate hypothesis clause local_id")
    if len(bridge_ids) != len(set(bridge_ids)):
        raise ValueError("PS3 duplicate bridge local_id")
    if set(clause_ids) & set(bridge_ids):
        raise ValueError(
            "PS3 local IDs collide across hypothesis and bridge units"
        )

    invalid = 0
    clauses: list[AuditedHypothesisClause] = []
    for row in draft.hypothesis_clauses:
        valid = row.text in hypothesis.hypothesis_statement
        invalid += int(not valid)
        clauses.append(
            AuditedHypothesisClause(
                local_id=row.local_id,
                text=row.text,
                clause_type=row.clause_type,
                materiality=row.materiality,
                quote_audit=ExactClauseAudit(
                    local_id=row.local_id,
                    text=row.text,
                    source_field="hypothesis_statement",
                    exact_substring_match=valid,
                ),
            )
        )

    bridges: list[AuditedBridgeUnit] = []
    for row in draft.bridge_units:
        valid = row.text in hypothesis.inferential_bridge
        invalid += int(not valid)
        bridges.append(
            AuditedBridgeUnit(
                local_id=row.local_id,
                text=row.text,
                materiality=row.materiality,
                quote_audit=ExactClauseAudit(
                    local_id=row.local_id,
                    text=row.text,
                    source_field="inferential_bridge",
                    exact_substring_match=valid,
                ),
            )
        )

    if invalid:
        raise ValueError(
            f"PS3 decomposition returned {invalid} non-verbatim clause(s)"
        )
    return clauses, bridges, invalid


def _validate_coverage_reviews(
    *,
    draft: HypothesisClauseCoverageDraft,
    hypothesis: HypothesisCard,
    clauses: list[AuditedHypothesisClause],
    bridges: list[AuditedBridgeUnit],
    selected_premise_ids: set[str],
) -> tuple[
    list[AuditedClauseCoverageReview],
    list[AuditedClauseCoverageReview],
    int,
]:
    if draft.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError(
            "PS3 coverage critic returned wrong hypothesis ID"
        )

    expected_clause_ids = {row.local_id for row in clauses}
    expected_bridge_ids = {row.local_id for row in bridges}

    actual_clause_ids = {
        row.local_id for row in draft.hypothesis_clause_reviews
    }
    actual_bridge_ids = {
        row.local_id for row in draft.bridge_unit_reviews
    }

    if actual_clause_ids != expected_clause_ids:
        raise ValueError(
            "PS3 coverage clause ID set mismatch: "
            f"expected={sorted(expected_clause_ids)}, "
            f"actual={sorted(actual_clause_ids)}"
        )
    if actual_bridge_ids != expected_bridge_ids:
        raise ValueError(
            "PS3 coverage bridge ID set mismatch: "
            f"expected={sorted(expected_bridge_ids)}, "
            f"actual={sorted(actual_bridge_ids)}"
        )

    invalid_refs = 0

    def audit_row(
        row: ClauseCoverageReviewDraft,
    ) -> AuditedClauseCoverageReview:
        nonlocal invalid_refs

        support_ids = list(dict.fromkeys(
            row.supporting_premise_statement_ids
        ))
        limiting_ids = list(dict.fromkeys(
            row.limiting_premise_statement_ids
        ))

        invalid = (
            set(support_ids) | set(limiting_ids)
        ) - selected_premise_ids
        invalid_refs += len(invalid)
        if invalid:
            raise ValueError(
                "PS3 critic referenced non-selected premise IDs: "
                f"{sorted(invalid)}"
            )

        return AuditedClauseCoverageReview(
            local_id=row.local_id,
            status=row.status,
            supporting_premise_statement_ids=support_ids,
            limiting_premise_statement_ids=limiting_ids,
            support_explanation=row.support_explanation,
            missing_relation_or_scope=row.missing_relation_or_scope,
            confidence=row.confidence,
        )

    claim_reviews = [
        audit_row(row)
        for row in draft.hypothesis_clause_reviews
    ]
    bridge_reviews = [
        audit_row(row)
        for row in draft.bridge_unit_reviews
    ]

    claim_reviews.sort(
        key=lambda row: next(
            index
            for index, clause in enumerate(clauses)
            if clause.local_id == row.local_id
        )
    )
    bridge_reviews.sort(
        key=lambda row: next(
            index
            for index, bridge in enumerate(bridges)
            if bridge.local_id == row.local_id
        )
    )

    return claim_reviews, bridge_reviews, invalid_refs


def derive_overall_verdict(
    *,
    clauses: list[AuditedHypothesisClause],
    claim_reviews: list[AuditedClauseCoverageReview],
    bridge_reviews: list[AuditedClauseCoverageReview],
) -> tuple[OverallGroundingVerdict, str]:
    materiality_by_id = {
        row.local_id: row.materiality
        for row in clauses
    }
    core_reviews = [
        row
        for row in claim_reviews
        if materiality_by_id[row.local_id] == "core"
    ]
    if not core_reviews:
        core_reviews = list(claim_reviews)

    core_statuses = {row.status for row in core_reviews}
    bridge_statuses = {row.status for row in bridge_reviews}

    if (
        "contradicted_or_limited" in core_statuses
        or "contradicted_or_limited" in bridge_statuses
    ):
        return (
            "scope_conflicted",
            "At least one core hypothesis clause or inferential bridge unit "
            "is explicitly contradicted or limited by the selected evidence.",
        )

    if (
        "unsupported_extension" in core_statuses
        or "unsupported_extension" in bridge_statuses
    ):
        return (
            "unsupported_inferential_leap",
            "At least one core hypothesis clause or inferential bridge unit "
            "contains a material extension not grounded by the selected "
            "premises.",
        )

    if (
        "hypothesized_bridge" in core_statuses
        or "hypothesized_bridge" in bridge_statuses
    ):
        return (
            "testable_but_under_grounded_extension",
            "The evidence supports relevant endpoints or contextual pieces, "
            "but at least one material relation remains an explicit "
            "hypothesized bridge rather than a grounded corpus relation.",
        )

    if (
        "uncertain" in core_statuses
        or "uncertain" in bridge_statuses
    ):
        return (
            "uncertain",
            "At least one material grounding judgment remains uncertain.",
        )

    allowed = {"directly_grounded", "synthesis_grounded"}
    if core_statuses <= allowed and bridge_statuses <= allowed:
        return (
            "grounded_extension",
            "All audited core clauses and inferential bridge units are "
            "directly or synthesis-grounded by the selected premises.",
        )

    return (
        "uncertain",
        "The clause-status pattern does not support a stronger deterministic "
        "grounding verdict.",
    )


def _ps2_snapshot(
    ps2: PremiseRoleNecessityReport | None,
    hypothesis_id: str,
) -> PS2HypothesisSnapshot | None:
    if ps2 is None:
        return None

    rows = [
        row
        for row in ps2.cards
        if row.hypothesis_id == hypothesis_id
    ]
    if not rows:
        return None

    return PS2HypothesisSnapshot(
        premise_verdicts={
            row.premise_statement_id: row.necessity_verdict
            for row in rows
        },
        role_labels={
            row.premise_statement_id: row.role_review.role
            for row in rows
        },
        ablation_statuses={
            row.premise_statement_id: (
                row.ablation_review.remaining_grounding_status
            )
            for row in rows
        },
    )


class HypothesisClauseCoverageRuntime:
    def __init__(
        self,
        backend: InstructorPremiseCriticBackend,
    ) -> None:
        self.backend = backend

    def run(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
        axis_plan: DiscoveryAxisPlan,
        axis_report: DiscoveryAxisSynthesisReport,
        *,
        ps2_report: PremiseRoleNecessityReport | None = None,
        hypothesis_ids: set[str] | None = None,
    ) -> tuple[
        HypothesisClauseCoverageReport,
        list[tuple[str, str, CriticPrompt]],
    ]:
        if portfolio.source_context_id != context.context_id:
            raise ValueError("PS3 portfolio/context ID mismatch")
        if portfolio.source_context_sha256 != context.context_sha256:
            raise ValueError("PS3 portfolio/context SHA mismatch")
        if axis_report.final_portfolio_id != portfolio.portfolio_id:
            raise ValueError("PS3 axis report/portfolio ID mismatch")
        if axis_report.axis_plan_id != axis_plan.plan_id:
            raise ValueError("PS3 axis report/plan ID mismatch")

        if ps2_report is not None:
            if ps2_report.source_context_id != context.context_id:
                raise ValueError("PS3 PS2/context ID mismatch")
            if ps2_report.source_context_sha256 != context.context_sha256:
                raise ValueError("PS3 PS2/context SHA mismatch")
            if ps2_report.source_portfolio_id != portfolio.portfolio_id:
                raise ValueError("PS3 PS2/portfolio ID mismatch")

        eligible_by_id = {
            row.statement_id: row
            for row in context.evidence_statements
            if row.eligible_as_premise
        }
        axis_by_id = {
            row.axis_id: row
            for row in axis_plan.axes
        }
        lineage_by_hypothesis = {
            row.hypothesis_id: row
            for row in axis_report.lineages
        }

        cards: list[HypothesisClauseCoverageCard] = []
        prompts: list[tuple[str, str, CriticPrompt]] = []

        selected_hypotheses = [
            row
            for row in portfolio.hypotheses
            if (
                hypothesis_ids is None
                or row.hypothesis_id in hypothesis_ids
            )
        ]
        unknown_requested = (
            set(hypothesis_ids or set())
            - {row.hypothesis_id for row in portfolio.hypotheses}
        )
        if unknown_requested:
            raise ValueError(
                "PS3 requested unknown hypothesis IDs: "
                f"{sorted(unknown_requested)}"
            )

        invalid_quote_count = 0
        invalid_support_reference_count = 0
        llm_calls = 0

        for hypothesis in selected_hypotheses:
            lineage = lineage_by_hypothesis.get(hypothesis.hypothesis_id)
            if lineage is None:
                raise ValueError(
                    "PS3 missing discovery lineage for hypothesis "
                    f"{hypothesis.hypothesis_id}"
                )
            axis = axis_by_id.get(lineage.axis_id)
            if axis is None:
                raise ValueError(
                    f"PS3 missing axis {lineage.axis_id}"
                )

            premises: list[HypothesisEvidenceStatement] = []
            for sid in hypothesis.premise_statement_ids:
                premise = eligible_by_id.get(sid)
                if premise is None:
                    raise ValueError(
                        "PS3 hypothesis uses non-eligible premise: "
                        f"{sid}"
                    )
                premises.append(premise)

            decomposition_prompt = build_decomposition_prompt(
                hypothesis
            )
            prompts.append(
                (
                    hypothesis.hypothesis_id,
                    "decomposition",
                    decomposition_prompt,
                )
            )
            generation = self.backend.call(
                decomposition_prompt,
                HypothesisClauseDecompositionDraft,
            )
            llm_calls += 1
            decomposition = generation.value
            assert isinstance(
                decomposition,
                HypothesisClauseDecompositionDraft,
            )
            if decomposition.hypothesis_id != hypothesis.hypothesis_id:
                raise ValueError(
                    "PS3 decomposer returned wrong hypothesis ID"
                )

            clauses, bridges, invalid_quotes = _audit_decomposition(
                decomposition,
                hypothesis,
            )
            invalid_quote_count += invalid_quotes

            coverage_prompt = build_coverage_prompt(
                hypothesis=hypothesis,
                axis=axis,
                hypothesis_clauses=clauses,
                bridge_units=bridges,
                premises=premises,
            )
            prompts.append(
                (
                    hypothesis.hypothesis_id,
                    "coverage",
                    coverage_prompt,
                )
            )
            coverage_generation = self.backend.call(
                coverage_prompt,
                HypothesisClauseCoverageDraft,
            )
            llm_calls += 1
            coverage_draft = coverage_generation.value
            assert isinstance(
                coverage_draft,
                HypothesisClauseCoverageDraft,
            )

            (
                claim_reviews,
                bridge_reviews,
                invalid_refs,
            ) = _validate_coverage_reviews(
                draft=coverage_draft,
                hypothesis=hypothesis,
                clauses=clauses,
                bridges=bridges,
                selected_premise_ids=set(
                    hypothesis.premise_statement_ids
                ),
            )
            invalid_support_reference_count += invalid_refs

            verdict, reason = derive_overall_verdict(
                clauses=clauses,
                claim_reviews=claim_reviews,
                bridge_reviews=bridge_reviews,
            )

            all_reviews = claim_reviews + bridge_reviews
            status_counts = Counter(
                row.status for row in all_reviews
            )

            cards.append(
                HypothesisClauseCoverageCard(
                    hypothesis_id=hypothesis.hypothesis_id,
                    title=hypothesis.title,
                    axis_id=axis.axis_id,
                    axis_label=axis.label,
                    selected_premise_statement_ids=list(
                        hypothesis.premise_statement_ids
                    ),
                    hypothesis_clauses=clauses,
                    bridge_units=bridges,
                    hypothesis_clause_reviews=claim_reviews,
                    bridge_unit_reviews=bridge_reviews,
                    overall_verdict=verdict,
                    verdict_reason=reason,
                    directly_grounded_count=status_counts[
                        "directly_grounded"
                    ],
                    synthesis_grounded_count=status_counts[
                        "synthesis_grounded"
                    ],
                    hypothesized_bridge_count=status_counts[
                        "hypothesized_bridge"
                    ],
                    unsupported_extension_count=status_counts[
                        "unsupported_extension"
                    ],
                    contradicted_or_limited_count=status_counts[
                        "contradicted_or_limited"
                    ],
                    uncertain_count=status_counts["uncertain"],
                    critical_missing_links=list(
                        coverage_draft.critical_missing_links
                    ),
                    ps2_snapshot=_ps2_snapshot(
                        ps2_report,
                        hypothesis.hypothesis_id,
                    ),
                    decomposition_prompt_sha256=(
                        decomposition_prompt.prompt_sha256
                    ),
                    coverage_prompt_sha256=(
                        coverage_prompt.prompt_sha256
                    ),
                )
            )

        overall_counts = Counter(
            row.overall_verdict for row in cards
        )
        clause_counts = Counter(
            review.status
            for card in cards
            for review in card.hypothesis_clause_reviews
        )
        bridge_counts = Counter(
            review.status
            for card in cards
            for review in card.bridge_unit_reviews
        )

        evaluated_ids = sorted(
            row.hypothesis_id for row in cards
        )
        scope_hash = _sha256_json(evaluated_ids)

        payload = {
            "schema_version": "hypothesis-clause-coverage-report-v1",
            "report_id": _stable_id(
                "hypothesis_clause_coverage_report",
                context.context_sha256,
                portfolio.portfolio_id,
                axis_plan.plan_id,
                axis_report.report_id,
                (
                    ps2_report.report_sha256
                    if ps2_report is not None
                    else "-"
                ),
                self.backend.model_name,
                DECOMPOSITION_PROMPT_VERSION,
                COVERAGE_PROMPT_VERSION,
                scope_hash,
            ),
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_axis_plan_id": axis_plan.plan_id,
            "source_axis_report_id": axis_report.report_id,
            "source_ps2_report_sha256": (
                ps2_report.report_sha256
                if ps2_report is not None
                else None
            ),
            "domain_profile_id": context.domain_profile_id,
            "corpus_id": context.corpus_id,
            "critic_model": self.backend.model_name,
            "decomposition_prompt_version": (
                DECOMPOSITION_PROMPT_VERSION
            ),
            "coverage_prompt_version": COVERAGE_PROMPT_VERSION,
            "evaluated_hypothesis_ids": evaluated_ids,
            "hypothesis_count": len(cards),
            "llm_call_count": llm_calls,
            "overall_verdict_counts": dict(overall_counts),
            "clause_status_counts": dict(clause_counts),
            "bridge_status_counts": dict(bridge_counts),
            "invalid_clause_quote_count": invalid_quote_count,
            "invalid_support_reference_count": (
                invalid_support_reference_count
            ),
            "cards": [
                row.model_dump(mode="json")
                for row in cards
            ],
            "policy": HypothesisClauseCoveragePolicy().model_dump(
                mode="json"
            ),
        }

        report = HypothesisClauseCoverageReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
        return report, prompts
