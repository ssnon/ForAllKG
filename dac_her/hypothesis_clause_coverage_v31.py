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
from dac_her.hypothesis_clause_coverage import (
    AuditedBridgeUnit,
    AuditedHypothesisClause,
    HypothesisClauseCoverageReport,
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
)


COVERAGE_PROMPT_VERSION = (
    "hypothesis-clause-coverage-auditor-v2.9.1-ps3.1"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CoverageStatusV31 = Literal[
    "directly_grounded",
    "synthesis_grounded",
    "hypothesized_bridge",
    "unsupported_extension",
    "evidence_scope_limitation",
    "scope_mismatch",
    "contradicted_by_evidence",
    "uncertain",
]

Confidence = Literal["low", "medium", "high"]

OverallGroundingVerdictV31 = Literal[
    "grounded_extension",
    "testable_but_under_grounded_extension",
    "unsupported_inferential_leap",
    "scope_conflicted",
    "uncertain",
]


class ClauseCoverageReviewDraftV31(StrictModel):
    local_id: str
    status: CoverageStatusV31

    supporting_premise_statement_ids: list[str] = Field(
        default_factory=list
    )
    limiting_premise_statement_ids: list[str] = Field(
        default_factory=list
    )

    support_explanation: str = Field(min_length=1)
    missing_relation_or_scope: str | None = None
    confidence: Confidence


class HypothesisClauseCoverageDraftV31(StrictModel):
    hypothesis_id: str
    hypothesis_clause_reviews: list[
        ClauseCoverageReviewDraftV31
    ] = Field(min_length=1)
    bridge_unit_reviews: list[
        ClauseCoverageReviewDraftV31
    ] = Field(default_factory=list)
    critical_missing_links: list[str] = Field(default_factory=list)


class AuditedClauseCoverageReviewV31(StrictModel):
    local_id: str
    status: CoverageStatusV31

    supporting_premise_statement_ids: list[str] = Field(
        default_factory=list
    )
    limiting_premise_statement_ids: list[str] = Field(
        default_factory=list
    )

    support_explanation: str
    missing_relation_or_scope: str | None = None
    confidence: Confidence

    source_ps3_status: str | None = None
    status_changed_from_ps3: bool = False


class HypothesisClauseCoverageCardV31(StrictModel):
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
        AuditedClauseCoverageReviewV31
    ] = Field(default_factory=list)
    bridge_unit_reviews: list[
        AuditedClauseCoverageReviewV31
    ] = Field(default_factory=list)

    source_ps3_overall_verdict: str
    overall_verdict: OverallGroundingVerdictV31
    overall_verdict_changed_from_ps3: bool
    verdict_reason: str

    directly_grounded_count: int = 0
    synthesis_grounded_count: int = 0
    hypothesized_bridge_count: int = 0
    unsupported_extension_count: int = 0
    evidence_scope_limitation_count: int = 0
    scope_mismatch_count: int = 0
    contradicted_by_evidence_count: int = 0
    uncertain_count: int = 0

    critical_missing_links: list[str] = Field(default_factory=list)
    coverage_prompt_sha256: str


class PS31Policy(StrictModel):
    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False

    source_ps3_decomposition_reused_exactly: Literal[True] = True
    decomposition_llm_calls: Literal[0] = 0
    coverage_uses_selected_premise_full_set: Literal[True] = True

    discovery_axis_is_scientific_evidence: Literal[False] = False
    external_knowledge_allowed: Literal[False] = False

    evidence_non_establishment_is_contradiction: Literal[False] = False
    scope_transfer_is_contradiction: Literal[False] = False

    critic_verdict_is_scientific_truth: Literal[False] = False


class HypothesisClauseCoverageReportV31(StrictModel):
    schema_version: Literal[
        "hypothesis-clause-coverage-report-v1.1"
    ] = "hypothesis-clause-coverage-report-v1.1"

    report_id: str
    report_sha256: str

    source_ps3_report_id: str
    source_ps3_report_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_axis_plan_id: str
    source_axis_report_id: str

    domain_profile_id: str
    corpus_id: str

    critic_model: str
    coverage_prompt_version: str

    evaluated_hypothesis_ids: list[str] = Field(default_factory=list)
    hypothesis_count: int
    decomposition_llm_call_count: Literal[0] = 0
    coverage_llm_call_count: int
    llm_call_count: int

    overall_verdict_counts: dict[str, int] = Field(default_factory=dict)
    clause_status_counts: dict[str, int] = Field(default_factory=dict)
    bridge_status_counts: dict[str, int] = Field(default_factory=dict)

    clause_status_transition_counts: dict[str, int] = Field(
        default_factory=dict
    )
    bridge_status_transition_counts: dict[str, int] = Field(
        default_factory=dict
    )
    overall_verdict_transition_counts: dict[str, int] = Field(
        default_factory=dict
    )

    invalid_source_clause_quote_count: int = 0
    invalid_support_reference_count: int = 0

    cards: list[HypothesisClauseCoverageCardV31] = Field(
        default_factory=list
    )

    policy: PS31Policy = Field(default_factory=PS31Policy)


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


def build_coverage_prompt_v31(
    *,
    hypothesis: HypothesisCard,
    axis: DiscoveryAxis,
    hypothesis_clauses: list[AuditedHypothesisClause],
    bridge_units: list[AuditedBridgeUnit],
    premises: list[HypothesisEvidenceStatement],
) -> CriticPrompt:
    system = """You are a full-set scientific grounding and inferential-boundary
auditor.

The claim units were decomposed in an earlier evidence-blind stage. You MUST
audit the supplied fixed units as-is against the COMPLETE SELECTED PREMISE SET.

Use ONLY the supplied premises. External knowledge is forbidden. The discovery
axis is inspiration-only and is NOT scientific evidence.

Classify every unit with exactly one status:

1. directly_grounded
   At least one selected premise directly supports the material relation or
   claim as written, with compatible scientific scope.

2. synthesis_grounded
   Multiple selected premises jointly support the claim as written without
   adding a new unobserved causal/mechanistic relation or unsupported scope
   transfer.

3. hypothesized_bridge
   The premises support relevant endpoints, observations, or contextual pieces,
   but the specific proposed relation/mechanism is not established. There is no
   explicit evidence statement that specifically warns that this relation is
   outside its demonstrated scope. This is a legitimate testable bridge, not a
   corpus-supported fact.

4. unsupported_extension
   The unit introduces a material mechanism, entity, direction, outcome,
   comparison, or design conclusion that is not adequately supported even as a
   synthesis of the selected premises. Use this when the proposed content is
   not merely a bridge between otherwise grounded endpoints.

5. evidence_scope_limitation
   The selected evidence explicitly states that the relevant relation is NOT
   established, NOT demonstrated, NOT provided, or otherwise remains outside
   the evidence boundary. This is NOT counterevidence. It means "not established
   by these premises", not "false".

   Example pattern:
   "These claims do not provide a direct structural relation between X and Y."
   If the audited clause asserts that X-Y relation, prefer
   evidence_scope_limitation rather than contradicted_by_evidence unless an
   actual opposing result is reported.

6. scope_mismatch
   Support would require transferring a result across a materially different
   metal pair, catalyst/support system, coordination motif, reaction context,
   observable, or condition without supplied evidence justifying that transfer.
   Scope mismatch is not the same as a generic lack of evidence.

7. contradicted_by_evidence
   A selected premise reports an explicitly opposite or scientifically
   incompatible result for the relevant scope. Mere non-establishment,
   caveating, or absence of a demonstrated relation MUST NOT be classified as
   contradiction.

8. uncertain
   The supplied text does not permit a reliable classification.

Critical epistemic rules:
- "Not established" is NOT "contradicted".
- "Different system" is NOT automatically "contradicted"; it is scope_mismatch
  only when the clause actually relies on transferring that result.
- A premise supporting one endpoint does not automatically support the relation
  connecting two endpoints.
- "Removing a premise breaks the chain" does NOT imply that the full chain is
  grounded.
- Do not reward physical plausibility.
- Discovery-axis content cannot serve as evidence.
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

FIXED HYPOTHESIS CLAUSES
These were produced by the prior evidence-blind PS3 decomposition and are NOT
being regenerated in PS3.1.
{json.dumps(clause_payload, ensure_ascii=False, indent=2)}

FIXED BRIDGE UNITS
{json.dumps(bridge_payload, ensure_ascii=False, indent=2)}

SELECTED PREMISES — COMPLETE SCIENTIFIC EVIDENCE SET FOR THIS AUDIT
{json.dumps(premise_payload, ensure_ascii=False, indent=2)}

Audit every fixed unit under the refined epistemic taxonomy. In particular,
distinguish evidence_scope_limitation from scope_mismatch and from genuine
contradicted_by_evidence.
"""
    return _make_prompt(
        COVERAGE_PROMPT_VERSION,
        system,
        user,
    )


def _validate_source_ps3_card(
    *,
    hypothesis: HypothesisCard,
    source_card: Any,
) -> int:
    if source_card.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("PS3.1 source PS3 hypothesis ID mismatch")

    invalid = 0

    for row in source_card.hypothesis_clauses:
        valid = (
            row.quote_audit.exact_substring_match
            and row.text in hypothesis.hypothesis_statement
        )
        invalid += int(not valid)

    for row in source_card.bridge_units:
        valid = (
            row.quote_audit.exact_substring_match
            and row.text in hypothesis.inferential_bridge
        )
        invalid += int(not valid)

    if invalid:
        raise ValueError(
            "PS3.1 source PS3 contains invalid/non-verbatim fixed "
            f"clause(s): {invalid}"
        )

    return invalid


def _validate_reviews_v31(
    *,
    draft: HypothesisClauseCoverageDraftV31,
    hypothesis: HypothesisCard,
    source_card: Any,
    selected_premise_ids: set[str],
) -> tuple[
    list[AuditedClauseCoverageReviewV31],
    list[AuditedClauseCoverageReviewV31],
    int,
]:
    if draft.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError(
            "PS3.1 critic returned wrong hypothesis ID"
        )

    expected_clause_ids = {
        row.local_id
        for row in source_card.hypothesis_clauses
    }
    expected_bridge_ids = {
        row.local_id
        for row in source_card.bridge_units
    }

    actual_clause_ids = {
        row.local_id
        for row in draft.hypothesis_clause_reviews
    }
    actual_bridge_ids = {
        row.local_id
        for row in draft.bridge_unit_reviews
    }

    if actual_clause_ids != expected_clause_ids:
        raise ValueError(
            "PS3.1 hypothesis clause ID set mismatch: "
            f"expected={sorted(expected_clause_ids)}, "
            f"actual={sorted(actual_clause_ids)}"
        )
    if actual_bridge_ids != expected_bridge_ids:
        raise ValueError(
            "PS3.1 bridge ID set mismatch: "
            f"expected={sorted(expected_bridge_ids)}, "
            f"actual={sorted(actual_bridge_ids)}"
        )

    source_claim_status = {
        row.local_id: row.status
        for row in source_card.hypothesis_clause_reviews
    }
    source_bridge_status = {
        row.local_id: row.status
        for row in source_card.bridge_unit_reviews
    }

    invalid_refs = 0

    def audit_row(
        row: ClauseCoverageReviewDraftV31,
        *,
        source_status: str | None,
    ) -> AuditedClauseCoverageReviewV31:
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
                "PS3.1 critic referenced non-selected premise IDs: "
                f"{sorted(invalid)}"
            )

        return AuditedClauseCoverageReviewV31(
            local_id=row.local_id,
            status=row.status,
            supporting_premise_statement_ids=support_ids,
            limiting_premise_statement_ids=limiting_ids,
            support_explanation=row.support_explanation,
            missing_relation_or_scope=row.missing_relation_or_scope,
            confidence=row.confidence,
            source_ps3_status=source_status,
            status_changed_from_ps3=(
                source_status is not None
                and source_status != row.status
            ),
        )

    claim_order = {
        row.local_id: i
        for i, row in enumerate(source_card.hypothesis_clauses)
    }
    bridge_order = {
        row.local_id: i
        for i, row in enumerate(source_card.bridge_units)
    }

    claim_reviews = [
        audit_row(
            row,
            source_status=source_claim_status.get(row.local_id),
        )
        for row in draft.hypothesis_clause_reviews
    ]
    bridge_reviews = [
        audit_row(
            row,
            source_status=source_bridge_status.get(row.local_id),
        )
        for row in draft.bridge_unit_reviews
    ]

    claim_reviews.sort(
        key=lambda row: claim_order[row.local_id]
    )
    bridge_reviews.sort(
        key=lambda row: bridge_order[row.local_id]
    )

    return claim_reviews, bridge_reviews, invalid_refs


def derive_overall_verdict_v31(
    *,
    clauses: list[AuditedHypothesisClause],
    claim_reviews: list[AuditedClauseCoverageReviewV31],
    bridge_reviews: list[AuditedClauseCoverageReviewV31],
) -> tuple[OverallGroundingVerdictV31, str]:
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

    core_statuses = {
        row.status
        for row in core_reviews
    }
    bridge_statuses = {
        row.status
        for row in bridge_reviews
    }
    material_statuses = core_statuses | bridge_statuses

    if material_statuses & {
        "contradicted_by_evidence",
        "scope_mismatch",
    }:
        return (
            "scope_conflicted",
            "At least one core hypothesis clause or inferential bridge unit "
            "is contradicted by supplied evidence or requires an unsupported "
            "material scope transfer.",
        )

    if "unsupported_extension" in material_statuses:
        return (
            "unsupported_inferential_leap",
            "At least one core hypothesis clause or inferential bridge unit "
            "introduces material unsupported scientific content beyond a "
            "testable bridge between grounded endpoints.",
        )

    if material_statuses & {
        "hypothesized_bridge",
        "evidence_scope_limitation",
    }:
        return (
            "testable_but_under_grounded_extension",
            "The hypothesis contains at least one explicit proposed relation "
            "or relation outside the demonstrated evidence boundary, but no "
            "material contradiction or unsupported scope transfer was found.",
        )

    if "uncertain" in material_statuses:
        return (
            "uncertain",
            "At least one material grounding judgment remains uncertain.",
        )

    allowed = {
        "directly_grounded",
        "synthesis_grounded",
    }
    if (
        core_statuses <= allowed
        and bridge_statuses <= allowed
    ):
        return (
            "grounded_extension",
            "All audited core clauses and inferential bridge units are "
            "directly or synthesis-grounded by the selected premises.",
        )

    return (
        "uncertain",
        "The refined clause-status pattern does not support a stronger "
        "deterministic verdict.",
    )


class HypothesisClauseCoverageRuntimeV31:
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
        source_ps3: HypothesisClauseCoverageReport,
        *,
        hypothesis_ids: set[str] | None = None,
    ) -> tuple[
        HypothesisClauseCoverageReportV31,
        list[tuple[str, CriticPrompt]],
    ]:
        if portfolio.source_context_id != context.context_id:
            raise ValueError("PS3.1 portfolio/context ID mismatch")
        if portfolio.source_context_sha256 != context.context_sha256:
            raise ValueError("PS3.1 portfolio/context SHA mismatch")
        if axis_report.final_portfolio_id != portfolio.portfolio_id:
            raise ValueError("PS3.1 axis report/portfolio ID mismatch")
        if axis_report.axis_plan_id != axis_plan.plan_id:
            raise ValueError("PS3.1 axis report/plan ID mismatch")

        if source_ps3.source_context_id != context.context_id:
            raise ValueError("PS3.1 source PS3/context ID mismatch")
        if source_ps3.source_context_sha256 != context.context_sha256:
            raise ValueError("PS3.1 source PS3/context SHA mismatch")
        if source_ps3.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("PS3.1 source PS3/portfolio mismatch")
        if source_ps3.source_axis_plan_id != axis_plan.plan_id:
            raise ValueError("PS3.1 source PS3/axis-plan mismatch")
        if source_ps3.source_axis_report_id != axis_report.report_id:
            raise ValueError("PS3.1 source PS3/axis-report mismatch")

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
        source_card_by_id = {
            row.hypothesis_id: row
            for row in source_ps3.cards
        }
        hypothesis_by_id = {
            row.hypothesis_id: row
            for row in portfolio.hypotheses
        }

        requested_ids = (
            set(hypothesis_ids)
            if hypothesis_ids is not None
            else set(source_ps3.evaluated_hypothesis_ids)
        )

        unknown = requested_ids - set(hypothesis_by_id)
        if unknown:
            raise ValueError(
                "PS3.1 requested unknown hypothesis IDs: "
                f"{sorted(unknown)}"
            )

        unavailable = requested_ids - set(source_card_by_id)
        if unavailable:
            raise ValueError(
                "PS3.1 requested hypotheses absent from source PS3: "
                f"{sorted(unavailable)}"
            )

        cards: list[HypothesisClauseCoverageCardV31] = []
        prompts: list[tuple[str, CriticPrompt]] = []
        invalid_quotes = 0
        invalid_refs = 0
        coverage_calls = 0

        for hypothesis in portfolio.hypotheses:
            if hypothesis.hypothesis_id not in requested_ids:
                continue

            source_card = source_card_by_id[hypothesis.hypothesis_id]
            invalid_quotes += _validate_source_ps3_card(
                hypothesis=hypothesis,
                source_card=source_card,
            )

            lineage = lineage_by_hypothesis.get(
                hypothesis.hypothesis_id
            )
            if lineage is None:
                raise ValueError(
                    "PS3.1 missing discovery lineage for hypothesis "
                    f"{hypothesis.hypothesis_id}"
                )

            axis = axis_by_id.get(lineage.axis_id)
            if axis is None:
                raise ValueError(
                    f"PS3.1 missing axis {lineage.axis_id}"
                )

            premises: list[HypothesisEvidenceStatement] = []
            for sid in hypothesis.premise_statement_ids:
                premise = eligible_by_id.get(sid)
                if premise is None:
                    raise ValueError(
                        "PS3.1 hypothesis uses non-eligible premise: "
                        f"{sid}"
                    )
                premises.append(premise)

            if list(hypothesis.premise_statement_ids) != list(
                source_card.selected_premise_statement_ids
            ):
                raise ValueError(
                    "PS3.1 source PS3 selected-premise set/order differs "
                    f"for {hypothesis.hypothesis_id}"
                )

            prompt = build_coverage_prompt_v31(
                hypothesis=hypothesis,
                axis=axis,
                hypothesis_clauses=list(
                    source_card.hypothesis_clauses
                ),
                bridge_units=list(source_card.bridge_units),
                premises=premises,
            )
            prompts.append(
                (hypothesis.hypothesis_id, prompt)
            )

            generation = self.backend.call(
                prompt,
                HypothesisClauseCoverageDraftV31,
            )
            coverage_calls += 1
            draft = generation.value
            assert isinstance(
                draft,
                HypothesisClauseCoverageDraftV31,
            )

            (
                claim_reviews,
                bridge_reviews,
                bad_refs,
            ) = _validate_reviews_v31(
                draft=draft,
                hypothesis=hypothesis,
                source_card=source_card,
                selected_premise_ids=set(
                    hypothesis.premise_statement_ids
                ),
            )
            invalid_refs += bad_refs

            verdict, reason = derive_overall_verdict_v31(
                clauses=list(source_card.hypothesis_clauses),
                claim_reviews=claim_reviews,
                bridge_reviews=bridge_reviews,
            )

            all_reviews = claim_reviews + bridge_reviews
            counts = Counter(row.status for row in all_reviews)

            cards.append(
                HypothesisClauseCoverageCardV31(
                    hypothesis_id=hypothesis.hypothesis_id,
                    title=hypothesis.title,
                    axis_id=axis.axis_id,
                    axis_label=axis.label,
                    selected_premise_statement_ids=list(
                        hypothesis.premise_statement_ids
                    ),
                    hypothesis_clauses=list(
                        source_card.hypothesis_clauses
                    ),
                    bridge_units=list(source_card.bridge_units),
                    hypothesis_clause_reviews=claim_reviews,
                    bridge_unit_reviews=bridge_reviews,
                    source_ps3_overall_verdict=(
                        source_card.overall_verdict
                    ),
                    overall_verdict=verdict,
                    overall_verdict_changed_from_ps3=(
                        verdict != source_card.overall_verdict
                    ),
                    verdict_reason=reason,
                    directly_grounded_count=counts[
                        "directly_grounded"
                    ],
                    synthesis_grounded_count=counts[
                        "synthesis_grounded"
                    ],
                    hypothesized_bridge_count=counts[
                        "hypothesized_bridge"
                    ],
                    unsupported_extension_count=counts[
                        "unsupported_extension"
                    ],
                    evidence_scope_limitation_count=counts[
                        "evidence_scope_limitation"
                    ],
                    scope_mismatch_count=counts[
                        "scope_mismatch"
                    ],
                    contradicted_by_evidence_count=counts[
                        "contradicted_by_evidence"
                    ],
                    uncertain_count=counts["uncertain"],
                    critical_missing_links=list(
                        draft.critical_missing_links
                    ),
                    coverage_prompt_sha256=prompt.prompt_sha256,
                )
            )

        overall_counts = Counter(
            row.overall_verdict
            for row in cards
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

        clause_transitions = Counter(
            f"{review.source_ps3_status} -> {review.status}"
            for card in cards
            for review in card.hypothesis_clause_reviews
        )
        bridge_transitions = Counter(
            f"{review.source_ps3_status} -> {review.status}"
            for card in cards
            for review in card.bridge_unit_reviews
        )
        overall_transitions = Counter(
            f"{card.source_ps3_overall_verdict} -> "
            f"{card.overall_verdict}"
            for card in cards
        )

        evaluated_ids = sorted(
            row.hypothesis_id
            for row in cards
        )
        scope_hash = _sha256_json(evaluated_ids)

        payload = {
            "schema_version": (
                "hypothesis-clause-coverage-report-v1.1"
            ),
            "report_id": _stable_id(
                "hypothesis_clause_coverage_report_v31",
                source_ps3.report_sha256,
                context.context_sha256,
                portfolio.portfolio_id,
                axis_plan.plan_id,
                axis_report.report_id,
                self.backend.model_name,
                COVERAGE_PROMPT_VERSION,
                scope_hash,
            ),
            "source_ps3_report_id": source_ps3.report_id,
            "source_ps3_report_sha256": source_ps3.report_sha256,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_axis_plan_id": axis_plan.plan_id,
            "source_axis_report_id": axis_report.report_id,
            "domain_profile_id": context.domain_profile_id,
            "corpus_id": context.corpus_id,
            "critic_model": self.backend.model_name,
            "coverage_prompt_version": COVERAGE_PROMPT_VERSION,
            "evaluated_hypothesis_ids": evaluated_ids,
            "hypothesis_count": len(cards),
            "decomposition_llm_call_count": 0,
            "coverage_llm_call_count": coverage_calls,
            "llm_call_count": coverage_calls,
            "overall_verdict_counts": dict(overall_counts),
            "clause_status_counts": dict(clause_counts),
            "bridge_status_counts": dict(bridge_counts),
            "clause_status_transition_counts": dict(
                clause_transitions
            ),
            "bridge_status_transition_counts": dict(
                bridge_transitions
            ),
            "overall_verdict_transition_counts": dict(
                overall_transitions
            ),
            "invalid_source_clause_quote_count": invalid_quotes,
            "invalid_support_reference_count": invalid_refs,
            "cards": [
                row.model_dump(mode="json")
                for row in cards
            ],
            "policy": PS31Policy().model_dump(mode="json"),
        }

        report = HypothesisClauseCoverageReportV31(
            **payload,
            report_sha256=_sha256_json(payload),
        )
        return report, prompts
