from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.discovery_axis_contracts import (
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
)
from dac_her.hypothesis_clause_coverage_v31 import (
    HypothesisClauseCoverageCardV31,
    HypothesisClauseCoverageReportV31,
)
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from dac_her.ig11_endpoint_scope import (
    IG11PlanningReport,
    IG11StructuredGenerator,
)


TR1_SCHEMA_VERSION = "tr1-targeted-semantic-repair-v1"
TR1_PROMPT_VERSION = "tr1-unsupported-descriptor-repair-v2.9.1"
TR1_AUDIT_PROMPT_VERSION = "tr1-semantic-gate-v2.9.1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExpectedDirection = Literal[
    "increase",
    "decrease",
    "shift",
    "non_monotonic",
    "qualitative_change",
    "unspecified",
]

TR1Action = Literal["repair", "abstain"]

TR1FinalStatus = Literal[
    "kept",
    "repaired",
    "abstained",
]

TR1RelationStatus = Literal[
    "genuinely_unestablished_relation",
    "already_directly_grounded",
    "already_synthesis_grounded",
    "unsupported_operand",
    "scope_transfer_required",
    "multiple_material_relations",
    "axis_fidelity_lost",
    "uncertain",
]

TR1OperandGroundingStatus = Literal[
    "directly_grounded",
    "synthesis_grounded",
    "unsupported",
]


class TR1GroundingSupport(StrictModel):
    statement_id: str
    excerpt: str = Field(min_length=1)


class TR1GroundedOperand(StrictModel):
    operand_text: str = Field(min_length=1)
    scientific_role: str = Field(min_length=1)
    supports: list[TR1GroundingSupport] = Field(min_length=1)


class TR1RepairPlan(StrictModel):
    schema_version: Literal[
        "tr1-targeted-semantic-repair-v1"
    ] = TR1_SCHEMA_VERSION

    source_hypothesis_id: str
    axis_id: str
    action: TR1Action

    abstention_reason: str | None = None

    title: str | None = None
    hypothesis_statement: str | None = None
    inferential_bridge: str | None = None

    premise_statement_ids: list[str] = Field(default_factory=list)
    grounded_operands: list[TR1GroundedOperand] = Field(default_factory=list)

    removed_unsupported_material: list[str] = Field(default_factory=list)

    predicted_observable: str | None = None
    expected_direction: ExpectedDirection | None = None
    prediction_rationale: str | None = None
    falsifying_outcome: str | None = None

    axis_fidelity_preserved: bool | None = None
    repair_reason: str | None = None

    @model_validator(mode="after")
    def _consistency(self) -> "TR1RepairPlan":
        scientific = (
            self.title,
            self.hypothesis_statement,
            self.inferential_bridge,
            self.predicted_observable,
            self.expected_direction,
            self.prediction_rationale,
            self.falsifying_outcome,
            self.axis_fidelity_preserved,
            self.repair_reason,
        )
        if self.action == "abstain":
            if not (self.abstention_reason or "").strip():
                raise ValueError(
                    "abstention_reason required when TR1 abstains"
                )
            if any(row is not None for row in scientific):
                raise ValueError(
                    "abstaining TR1 plan must not contain repaired science"
                )
            if self.grounded_operands:
                raise ValueError(
                    "abstaining TR1 plan must not contain grounded_operands"
                )
            return self

        if self.abstention_reason is not None:
            raise ValueError(
                "abstention_reason must be null for repair action"
            )
        if any(row is None for row in scientific):
            raise ValueError(
                "repair action requires title, statement, bridge, test, "
                "axis-fidelity flag, and repair reason"
            )
        if len(self.grounded_operands) < 2:
            raise ValueError(
                "repair action requires at least two grounded material operands"
            )
        if self.axis_fidelity_preserved is not True:
            raise ValueError(
                "repair action requires axis_fidelity_preserved=true; "
                "otherwise abstain"
            )
        return self


class TR1OperandReview(StrictModel):
    operand_text: str
    status: TR1OperandGroundingStatus
    supporting_statement_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class TR1SemanticAudit(StrictModel):
    schema_version: Literal[
        "tr1-semantic-gate-v1"
    ] = "tr1-semantic-gate-v1"

    source_hypothesis_id: str
    axis_id: str
    repaired_hypothesis_statement: str

    operand_reviews: list[TR1OperandReview] = Field(min_length=1)
    unlisted_material_operand_texts: list[str] = Field(default_factory=list)

    relation_status: TR1RelationStatus
    one_material_relation_only: bool
    scope_compatible: bool
    axis_fidelity_preserved: bool

    explanation: str = Field(min_length=1)


class TR1ValidationIssue(StrictModel):
    code: str
    detail: str


class TR1DecisionCard(StrictModel):
    source_hypothesis_id: str
    source_title: str
    axis_id: str
    axis_label: str
    source_ps31_overall_verdict: str

    unsupported_material_units: list[str] = Field(default_factory=list)
    source_premise_statement_ids: list[str] = Field(default_factory=list)

    final_status: TR1FinalStatus
    final_hypothesis_id: str | None = None
    final_hypothesis_statement: str | None = None

    repair_generation_count: int = 0
    semantic_audit_count: int = 0
    semantic_repair_count: int = 0

    plan: TR1RepairPlan | None = None
    semantic_audit: TR1SemanticAudit | None = None
    validation_issues: list[TR1ValidationIssue] = Field(default_factory=list)
    disposition_reason: str = Field(min_length=1)


class TR1Policy(StrictModel):
    source_ps31_required: Literal[True] = True
    retrieval_changed: Literal[False] = False
    selected_premise_set_changed: Literal[False] = False
    external_knowledge_allowed: Literal[False] = False

    keep_grounded_or_testable: Literal[True] = True
    repair_only_unsupported_inferential_leap: Literal[True] = True
    scope_conflicted_repair_allowed: Literal[False] = False
    contradicted_repair_allowed: Literal[False] = False

    material_operands_must_be_grounded: Literal[True] = True
    novelty_may_reside_only_in_one_relation: Literal[True] = True
    scope_guard_locked: Literal[True] = True
    axis_fidelity_required: Literal[True] = True
    failed_repair_becomes_abstention: Literal[True] = True

    max_semantic_repair_count: Literal[1] = 1


class TR1Report(StrictModel):
    schema_version: Literal[
        "tr1-targeted-semantic-repair-report-v1"
    ] = "tr1-targeted-semantic-repair-report-v1"

    report_id: str
    report_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_axis_plan_id: str
    source_axis_report_id: str
    source_planning_report_id: str
    source_ps31_report_id: str
    source_ps31_report_sha256: str

    model: str

    source_hypothesis_count: int
    kept_count: int
    targeted_count: int
    repaired_count: int
    abstained_count: int

    repair_generation_count: int
    semantic_audit_count: int
    semantic_repair_count: int

    candidate_portfolio_id: str
    candidate_axis_report_id: str

    cards: list[TR1DecisionCard] = Field(default_factory=list)
    policy: TR1Policy = Field(default_factory=TR1Policy)


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


def _eligible_statement_map(
    context: HypothesisContext,
) -> dict[str, Any]:
    return {
        row.statement_id: row
        for row in context.evidence_statements
        if row.eligible_as_premise
    }


def _ps31_review_statuses(
    card: HypothesisClauseCoverageCardV31,
) -> list[str]:
    return [
        row.status
        for row in (
            list(card.hypothesis_clause_reviews)
            + list(card.bridge_unit_reviews)
        )
    ]


def tr1_target_eligibility(
    card: HypothesisClauseCoverageCardV31,
) -> tuple[bool, str]:
    if card.overall_verdict != "unsupported_inferential_leap":
        return (
            False,
            "overall verdict is not unsupported_inferential_leap",
        )

    statuses = _ps31_review_statuses(card)
    if "unsupported_extension" not in statuses:
        return (
            False,
            "unsupported_inferential_leap has no unsupported_extension unit",
        )

    disallowed = {
        "scope_mismatch",
        "contradicted_by_evidence",
        "uncertain",
    }
    present = sorted(disallowed & set(statuses))
    if present:
        return (
            False,
            "unsafe targeted repair because source audit contains: "
            + ", ".join(present),
        )

    return True, "eligible for bounded targeted semantic repair"


def _unsupported_units(
    card: HypothesisClauseCoverageCardV31,
) -> list[dict[str, str]]:
    clauses = {
        row.local_id: row
        for row in card.hypothesis_clauses
    }
    bridges = {
        row.local_id: row
        for row in card.bridge_units
    }

    rows: list[dict[str, str]] = []
    for review in card.hypothesis_clause_reviews:
        if review.status != "unsupported_extension":
            continue
        clause = clauses[review.local_id]
        rows.append(
            {
                "source": "hypothesis_statement",
                "local_id": review.local_id,
                "text": clause.text,
                "missing": review.missing_relation_or_scope or "",
            }
        )
    for review in card.bridge_unit_reviews:
        if review.status != "unsupported_extension":
            continue
        bridge = bridges[review.local_id]
        rows.append(
            {
                "source": "inferential_bridge",
                "local_id": review.local_id,
                "text": bridge.text,
                "missing": review.missing_relation_or_scope or "",
            }
        )
    return rows


def _planning_blueprint_by_axis(
    planning: IG11PlanningReport,
) -> dict[str, Any]:
    return {
        row.axis_id: row.blueprint
        for row in planning.blueprint_records
    }


def build_repair_messages(
    *,
    hypothesis: Any,
    ps31_card: HypothesisClauseCoverageCardV31,
    axis: Any,
    scope_guard_phrase: str,
    premises: list[Any],
    prior_failure: TR1SemanticAudit | None = None,
) -> list[dict[str, str]]:
    unsupported = _unsupported_units(ps31_card)

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

    system = """You are TR1, a targeted scientific-hypothesis semantic repair
planner.

You receive ONE previously accepted hypothesis that PS3.1 classified as an
unsupported_inferential_leap. Repair ONLY the localized unsupported scientific
material identified by PS3.1.

Use ONLY the supplied selected positive premises. External knowledge and the
discovery axis are not evidence.

TR1 EPISTEMIC CONTRACT

1. KEEP THE PREMISE SET FIXED.
   You may not add, remove, or replace selected premise IDs.

2. NOVELTY MAY RESIDE IN EXACTLY ONE RELATION.
   The repaired hypothesis may propose ONE unestablished relation connecting
   grounded scientific operands.

3. DO NOT INVENT A NEW MATERIAL OPERAND.
   Every material scientific variable, descriptor, entity, mechanism,
   observable, or outcome appearing as an operand in the repaired relation
   must be grounded by the selected premises.
   A new descriptor plus a new relation is two inferential burdens and is not
   allowed.

4. AXIS FIDELITY IS REQUIRED.
   A grounded substitute is allowed only if the repaired hypothesis still
   genuinely tests the assigned discovery-axis concept.
   If removing the unsupported descriptor erases the axis identity, ABSTAIN.
   Do not relabel a different mechanism as the original axis.

5. SCOPE IS LOCKED.
   The exact scope_guard_phrase must occur verbatim in the repaired
   hypothesis_statement and inferential_bridge.
   Do not broaden material/system/structural scope.

6. RELATION ONLY.
   The hypothesis_statement must contain one material scientific relation.
   Do not append a second mechanism, consequence, design recommendation, or
   causal chain.

7. GROUNDING PROVENANCE.
   For every grounded operand, provide one or more exact contiguous VERBATIM
   excerpts from the selected premises that ground that operand.

8. REPAIR IS OPTIONAL.
   If a valid one-relation repair cannot preserve axis fidelity and scope using
   grounded operands only, return action=abstain. Principled abstention is
   preferable to semantic substitution.

9. PREDICTION/FALSIFIER.
   If repairing, provide exactly one observable, one expected direction, and
   one falsifying outcome that test only the repaired one-edge relation.
"""

    user_payload = {
        "source_hypothesis": {
            "hypothesis_id": hypothesis.hypothesis_id,
            "title": hypothesis.title,
            "hypothesis_statement": hypothesis.hypothesis_statement,
            "inferential_bridge": hypothesis.inferential_bridge,
            "premise_statement_ids": hypothesis.premise_statement_ids,
        },
        "ps31": {
            "overall_verdict": ps31_card.overall_verdict,
            "verdict_reason": ps31_card.verdict_reason,
            "unsupported_units": unsupported,
            "critical_missing_links": ps31_card.critical_missing_links,
        },
        "axis_inspiration_only": {
            "axis_id": axis.axis_id,
            "label": axis.label,
            "subject": axis.proposed_subject,
            "relation": axis.proposed_relation,
            "object": axis.proposed_object,
        },
        "locked_scope_guard_phrase": scope_guard_phrase,
        "selected_premises": premise_payload,
    }
    if prior_failure is not None:
        user_payload["previous_failed_repair_audit"] = prior_failure.model_dump(
            mode="json"
        )
        user_payload["repair_instruction"] = (
            "The previous repair failed the TR1 semantic gate. Produce one "
            "complete replacement plan. Fix the audit failure without adding "
            "premises, operands, scope, or a second relation. If impossible, "
            "abstain."
        )
    else:
        user_payload["repair_instruction"] = (
            "Return one TR1RepairPlan. Remove or legitimately replace the "
            "unsupported material identified by PS3.1. If that cannot be done "
            "while preserving the assigned axis, abstain."
        )

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                user_payload,
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


_CHAIN_MARKERS = (
    " and then ",
    " followed by ",
    " which then ",
    " subsequently ",
    " thereby causing ",
    " leading in turn to ",
)


def validate_repair_plan(
    plan: TR1RepairPlan,
    *,
    hypothesis: Any,
    ps31_card: HypothesisClauseCoverageCardV31,
    scope_guard_phrase: str,
    context: HypothesisContext,
) -> list[TR1ValidationIssue]:
    issues: list[TR1ValidationIssue] = []

    if plan.source_hypothesis_id != hypothesis.hypothesis_id:
        issues.append(
            TR1ValidationIssue(
                code="source_hypothesis_id_mismatch",
                detail=(
                    f"expected={hypothesis.hypothesis_id}; "
                    f"actual={plan.source_hypothesis_id}"
                ),
            )
        )
    if plan.axis_id != ps31_card.axis_id:
        issues.append(
            TR1ValidationIssue(
                code="axis_id_mismatch",
                detail=(
                    f"expected={ps31_card.axis_id}; actual={plan.axis_id}"
                ),
            )
        )

    if plan.action == "abstain":
        return issues

    expected_premises = set(hypothesis.premise_statement_ids)
    actual_premises = set(plan.premise_statement_ids)
    if actual_premises != expected_premises:
        issues.append(
            TR1ValidationIssue(
                code="premise_set_changed",
                detail=(
                    f"expected={sorted(expected_premises)}; "
                    f"actual={sorted(actual_premises)}"
                ),
            )
        )

    assert plan.hypothesis_statement is not None
    assert plan.inferential_bridge is not None

    if scope_guard_phrase not in plan.hypothesis_statement:
        issues.append(
            TR1ValidationIssue(
                code="scope_guard_missing_from_statement",
                detail=scope_guard_phrase,
            )
        )
    if scope_guard_phrase not in plan.inferential_bridge:
        issues.append(
            TR1ValidationIssue(
                code="scope_guard_missing_from_bridge",
                detail=scope_guard_phrase,
            )
        )

    lowered = f" {plan.hypothesis_statement.lower()} "
    for marker in _CHAIN_MARKERS:
        if marker in lowered:
            issues.append(
                TR1ValidationIssue(
                    code="multi_hop_chain_marker",
                    detail=marker.strip(),
                )
            )

    eligible = _eligible_statement_map(context)

    operand_texts = [row.operand_text for row in plan.grounded_operands]
    if len(operand_texts) != len(set(operand_texts)):
        issues.append(
            TR1ValidationIssue(
                code="duplicate_grounded_operand",
                detail=str(operand_texts),
            )
        )

    for operand in plan.grounded_operands:
        for support in operand.supports:
            if support.statement_id not in expected_premises:
                issues.append(
                    TR1ValidationIssue(
                        code="operand_support_outside_locked_premises",
                        detail=(
                            f"{operand.operand_text}:"
                            f"{support.statement_id}"
                        ),
                    )
                )
                continue
            statement = eligible.get(support.statement_id)
            if statement is None:
                issues.append(
                    TR1ValidationIssue(
                        code="operand_support_not_eligible",
                        detail=support.statement_id,
                    )
                )
                continue
            if support.excerpt not in statement.text:
                issues.append(
                    TR1ValidationIssue(
                        code="nonverbatim_operand_support_excerpt",
                        detail=(
                            f"{operand.operand_text}:"
                            f"{support.statement_id}"
                        ),
                    )
                )

    unsupported_texts = {
        row["text"]
        for row in _unsupported_units(ps31_card)
    }
    grounded_texts = set(operand_texts)
    for unsupported in unsupported_texts:
        if (
            unsupported in plan.hypothesis_statement
            or unsupported in plan.inferential_bridge
        ) and unsupported not in grounded_texts:
            issues.append(
                TR1ValidationIssue(
                    code="unsupported_material_preserved_verbatim",
                    detail=unsupported,
                )
            )

    return issues


def build_semantic_audit_messages(
    *,
    plan: TR1RepairPlan,
    hypothesis: Any,
    axis: Any,
    scope_guard_phrase: str,
    premises: list[Any],
) -> list[dict[str, str]]:
    assert plan.action == "repair"
    assert plan.hypothesis_statement is not None
    assert plan.inferential_bridge is not None

    premise_payload = [
        {
            "statement_id": row.statement_id,
            "text": row.text,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
        }
        for row in premises
    ]

    system = """You are the TR1 semantic gate.

Audit a proposed targeted repair against ONLY the locked selected premises.
External knowledge and the discovery axis are not evidence.

The repair is acceptable only when ALL of the following are true:

- every material scientific operand/descriptor/entity/mechanism/observable in
  the repaired hypothesis is directly or synthesis grounded by the selected
  premises;
- there are no unlisted material operands hidden in the wording;
- the repaired hypothesis contains exactly ONE unestablished material relation;
- that relation is genuinely unestablished, not a restatement of selected
  evidence;
- no unsupported scope transfer is required;
- the repaired relation still preserves the assigned discovery-axis concept;
- novelty resides in the relation, not simultaneously in a new descriptor and
  a new relation.

Do not treat ordinary relational words such as "associated with", "modulates",
or "depends on" as material operands. Audit their scientific operands.

If removing an unsupported axis-defining descriptor changes the hypothesis into
a different scientific axis, set relation_status=axis_fidelity_lost and
axis_fidelity_preserved=false.

If any material operand is unsupported, use
relation_status=unsupported_operand.
"""

    payload = {
        "source_hypothesis_id": hypothesis.hypothesis_id,
        "axis_inspiration_only": {
            "axis_id": axis.axis_id,
            "label": axis.label,
            "subject": axis.proposed_subject,
            "relation": axis.proposed_relation,
            "object": axis.proposed_object,
        },
        "locked_scope_guard_phrase": scope_guard_phrase,
        "repair_plan": plan.model_dump(mode="json"),
        "selected_premises": premise_payload,
    }

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


def validate_semantic_audit(
    audit: TR1SemanticAudit,
    *,
    plan: TR1RepairPlan,
    hypothesis: Any,
) -> list[TR1ValidationIssue]:
    issues: list[TR1ValidationIssue] = []

    if audit.source_hypothesis_id != hypothesis.hypothesis_id:
        issues.append(
            TR1ValidationIssue(
                code="semantic_audit_hypothesis_id_mismatch",
                detail=audit.source_hypothesis_id,
            )
        )
    if audit.axis_id != plan.axis_id:
        issues.append(
            TR1ValidationIssue(
                code="semantic_audit_axis_id_mismatch",
                detail=audit.axis_id,
            )
        )
    if audit.repaired_hypothesis_statement != plan.hypothesis_statement:
        issues.append(
            TR1ValidationIssue(
                code="semantic_audit_statement_mismatch",
                detail="audit changed repaired hypothesis statement",
            )
        )

    expected_operands = {
        row.operand_text
        for row in plan.grounded_operands
    }
    reviewed_operands = {
        row.operand_text
        for row in audit.operand_reviews
    }
    if reviewed_operands != expected_operands:
        issues.append(
            TR1ValidationIssue(
                code="semantic_audit_operand_set_mismatch",
                detail=(
                    f"expected={sorted(expected_operands)}; "
                    f"actual={sorted(reviewed_operands)}"
                ),
            )
        )

    locked = set(plan.premise_statement_ids)
    for review in audit.operand_reviews:
        support_ids = set(review.supporting_statement_ids)
        if not support_ids <= locked:
            issues.append(
                TR1ValidationIssue(
                    code="semantic_audit_support_outside_locked_premises",
                    detail=(
                        f"{review.operand_text}:"
                        f"{sorted(support_ids - locked)}"
                    ),
                )
            )
        if (
            review.status in {
                "directly_grounded",
                "synthesis_grounded",
            }
            and not review.supporting_statement_ids
        ):
            issues.append(
                TR1ValidationIssue(
                    code="grounded_operand_without_support",
                    detail=review.operand_text,
                )
            )
        if (
            review.status == "unsupported"
            and review.supporting_statement_ids
        ):
            issues.append(
                TR1ValidationIssue(
                    code="unsupported_operand_with_support_ids",
                    detail=review.operand_text,
                )
            )

    return issues


def semantic_audit_passes(
    audit: TR1SemanticAudit,
) -> bool:
    return (
        audit.relation_status == "genuinely_unestablished_relation"
        and audit.one_material_relation_only
        and audit.scope_compatible
        and audit.axis_fidelity_preserved
        and not audit.unlisted_material_operand_texts
        and all(
            row.status in {
                "directly_grounded",
                "synthesis_grounded",
            }
            for row in audit.operand_reviews
        )
    )


def _update_first_prediction(
    rows: list[Any],
    *,
    observable: str,
    expected_direction: ExpectedDirection,
    rationale: str,
) -> list[dict[str, Any]]:
    if len(rows) != 1:
        raise ValueError(
            "TR1 requires exactly one existing predicted observation"
        )
    data = rows[0].model_dump(mode="json")
    if "observable" in data:
        data["observable"] = observable
    if "expected_direction" in data:
        data["expected_direction"] = expected_direction
    if "rationale" in data:
        data["rationale"] = rationale
    return [data]


def _update_first_falsifier(
    rows: list[Any],
    *,
    observable: str,
    falsifying_outcome: str,
) -> list[dict[str, Any]]:
    if len(rows) != 1:
        raise ValueError(
            "TR1 requires exactly one existing falsification criterion"
        )
    data = rows[0].model_dump(mode="json")
    if "observable" in data:
        data["observable"] = observable
    if "falsifying_outcome" in data:
        data["falsifying_outcome"] = falsifying_outcome
    return [data]


def _make_repaired_hypothesis(
    hypothesis: Any,
    plan: TR1RepairPlan,
) -> Any:
    assert plan.action == "repair"
    assert plan.title is not None
    assert plan.hypothesis_statement is not None
    assert plan.inferential_bridge is not None
    assert plan.predicted_observable is not None
    assert plan.expected_direction is not None
    assert plan.prediction_rationale is not None
    assert plan.falsifying_outcome is not None

    data = hypothesis.model_dump(mode="json")
    new_id = _stable_id(
        "hypothesis_tr1",
        hypothesis.hypothesis_id,
        plan.hypothesis_statement,
        plan.inferential_bridge,
        sorted(plan.premise_statement_ids),
    )

    data["hypothesis_id"] = new_id
    data["title"] = plan.title
    data["hypothesis_statement"] = plan.hypothesis_statement
    data["inferential_bridge"] = plan.inferential_bridge
    data["premise_statement_ids"] = list(plan.premise_statement_ids)

    data["predicted_observations"] = _update_first_prediction(
        hypothesis.predicted_observations,
        observable=plan.predicted_observable,
        expected_direction=plan.expected_direction,
        rationale=plan.prediction_rationale,
    )
    data["falsification_criteria"] = _update_first_falsifier(
        hypothesis.falsification_criteria,
        observable=plan.predicted_observable,
        falsifying_outcome=plan.falsifying_outcome,
    )

    # A repaired claim has not yet been re-assessed by the canonical internal
    # novelty critic. Prefer explicit not_assessed when the contract accepts it.
    if "novelty_status" in data:
        prior = data["novelty_status"]
        data["novelty_status"] = "not_assessed"
        try:
            return type(hypothesis).model_validate(data)
        except Exception:
            data["novelty_status"] = prior

    return type(hypothesis).model_validate(data)


def _build_candidate_portfolio(
    source: HypothesisPortfolio,
    hypotheses: list[Any],
) -> HypothesisPortfolio:
    data = source.model_dump(mode="json")
    data["hypotheses"] = [
        row.model_dump(mode="json")
        for row in hypotheses
    ]
    data["portfolio_id"] = _stable_id(
        "hypothesis_portfolio_tr1",
        source.portfolio_id,
        [
            row.hypothesis_id
            for row in hypotheses
        ],
        [
            row.hypothesis_statement
            for row in hypotheses
        ],
    )
    return HypothesisPortfolio.model_validate(data)


def _build_candidate_axis_report(
    source: DiscoveryAxisSynthesisReport,
    *,
    candidate_portfolio: HypothesisPortfolio,
    id_map: dict[str, str],
    abstained_ids: set[str],
) -> DiscoveryAxisSynthesisReport:
    data = source.model_dump(mode="json")

    lineages = []
    for lineage in source.lineages:
        if lineage.hypothesis_id in abstained_ids:
            continue
        row = lineage.model_dump(mode="json")
        row["hypothesis_id"] = id_map.get(
            lineage.hypothesis_id,
            lineage.hypothesis_id,
        )
        lineages.append(row)

    data["lineages"] = lineages
    if "accepted_hypothesis_count" in data:
        data["accepted_hypothesis_count"] = len(
            candidate_portfolio.hypotheses
        )
    if "final_portfolio_id" in data:
        data["final_portfolio_id"] = candidate_portfolio.portfolio_id

    portfolio_sha = _sha256_json(candidate_portfolio)
    for key in tuple(data):
        if (
            "final_portfolio" in key
            and "sha" in key.lower()
        ):
            data[key] = portfolio_sha

    data["report_id"] = _stable_id(
        "discovery_axis_synthesis_report_tr1",
        source.report_id,
        candidate_portfolio.portfolio_id,
        [row["hypothesis_id"] for row in lineages],
    )
    if "report_sha256" in data:
        tmp = dict(data)
        tmp.pop("report_sha256", None)
        data["report_sha256"] = _sha256_json(tmp)

    return DiscoveryAxisSynthesisReport.model_validate(data)


def run_tr1(
    *,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    axis_plan: DiscoveryAxisPlan,
    axis_report: DiscoveryAxisSynthesisReport,
    planning_report: IG11PlanningReport,
    ps31_report: HypothesisClauseCoverageReportV31,
    generator: IG11StructuredGenerator,
    max_semantic_repairs: int = 1,
) -> tuple[
    HypothesisPortfolio,
    DiscoveryAxisSynthesisReport,
    TR1Report,
    list[tuple[str, str, list[dict[str, str]]]],
]:
    if max_semantic_repairs not in {0, 1}:
        raise ValueError("TR1 max_semantic_repairs must be 0 or 1")

    if portfolio.source_context_id != context.context_id:
        raise ValueError("TR1 portfolio/context ID mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("TR1 portfolio/context SHA mismatch")
    if axis_report.final_portfolio_id != portfolio.portfolio_id:
        raise ValueError("TR1 axis report/portfolio mismatch")
    if axis_report.axis_plan_id != axis_plan.plan_id:
        raise ValueError("TR1 axis report/axis-plan mismatch")
    if ps31_report.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("TR1 PS3.1/portfolio mismatch")
    if ps31_report.source_axis_report_id != axis_report.report_id:
        raise ValueError("TR1 PS3.1/axis-report mismatch")
    if planning_report.source_context_id != context.context_id:
        raise ValueError("TR1 planning/context mismatch")
    if planning_report.source_axis_plan_id != axis_plan.plan_id:
        raise ValueError("TR1 planning/axis-plan mismatch")

    hypothesis_by_id = {
        row.hypothesis_id: row
        for row in portfolio.hypotheses
    }
    ps31_by_id = {
        row.hypothesis_id: row
        for row in ps31_report.cards
    }
    lineage_by_id = {
        row.hypothesis_id: row
        for row in axis_report.lineages
    }
    axis_by_id = {
        row.axis_id: row
        for row in axis_plan.axes
    }
    blueprint_by_axis = _planning_blueprint_by_axis(
        planning_report
    )
    premise_by_id = _eligible_statement_map(context)

    decisions: list[TR1DecisionCard] = []
    final_hypotheses: list[Any] = []
    id_map: dict[str, str] = {}
    abstained_ids: set[str] = set()
    saved_prompts: list[
        tuple[str, str, list[dict[str, str]]]
    ] = []

    for hypothesis in portfolio.hypotheses:
        ps31 = ps31_by_id.get(hypothesis.hypothesis_id)
        if ps31 is None:
            raise ValueError(
                "TR1 missing PS3.1 card for "
                f"{hypothesis.hypothesis_id}"
            )
        lineage = lineage_by_id.get(hypothesis.hypothesis_id)
        if lineage is None:
            raise ValueError(
                "TR1 missing discovery lineage for "
                f"{hypothesis.hypothesis_id}"
            )
        axis = axis_by_id.get(lineage.axis_id)
        if axis is None:
            raise ValueError(
                f"TR1 missing axis {lineage.axis_id}"
            )

        unsupported_rows = _unsupported_units(ps31)
        unsupported_texts = sorted(
            {
                row["text"]
                for row in unsupported_rows
            }
        )

        if ps31.overall_verdict in {
            "grounded_extension",
            "testable_but_under_grounded_extension",
        }:
            final_hypotheses.append(hypothesis)
            decisions.append(
                TR1DecisionCard(
                    source_hypothesis_id=hypothesis.hypothesis_id,
                    source_title=hypothesis.title,
                    axis_id=axis.axis_id,
                    axis_label=axis.label,
                    source_ps31_overall_verdict=ps31.overall_verdict,
                    unsupported_material_units=unsupported_texts,
                    source_premise_statement_ids=list(
                        hypothesis.premise_statement_ids
                    ),
                    final_status="kept",
                    final_hypothesis_id=hypothesis.hypothesis_id,
                    final_hypothesis_statement=(
                        hypothesis.hypothesis_statement
                    ),
                    disposition_reason=(
                        "PS3.1 already classifies this hypothesis as "
                        "grounded/testable; TR1 leaves it unchanged."
                    ),
                )
            )
            continue

        eligible, eligibility_reason = tr1_target_eligibility(
            ps31
        )
        if not eligible:
            abstained_ids.add(hypothesis.hypothesis_id)
            decisions.append(
                TR1DecisionCard(
                    source_hypothesis_id=hypothesis.hypothesis_id,
                    source_title=hypothesis.title,
                    axis_id=axis.axis_id,
                    axis_label=axis.label,
                    source_ps31_overall_verdict=ps31.overall_verdict,
                    unsupported_material_units=unsupported_texts,
                    source_premise_statement_ids=list(
                        hypothesis.premise_statement_ids
                    ),
                    final_status="abstained",
                    disposition_reason=eligibility_reason,
                )
            )
            continue

        blueprint = blueprint_by_axis.get(axis.axis_id)
        if (
            blueprint is None
            or blueprint.abstain
            or blueprint.scope_envelope is None
        ):
            raise ValueError(
                "TR1 targeted hypothesis lacks active IG1.2a scope blueprint"
            )
        scope_guard = (
            blueprint.scope_envelope.scope_guard_phrase
        )

        premises = []
        for sid in hypothesis.premise_statement_ids:
            row = premise_by_id.get(sid)
            if row is None:
                raise ValueError(
                    f"TR1 locked premise is not eligible: {sid}"
                )
            premises.append(row)

        repair_generation_count = 0
        semantic_audit_count = 0
        semantic_repair_count = 0
        validation_issues: list[TR1ValidationIssue] = []

        repair_messages = build_repair_messages(
            hypothesis=hypothesis,
            ps31_card=ps31,
            axis=axis,
            scope_guard_phrase=scope_guard,
            premises=premises,
        )
        saved_prompts.append(
            (
                hypothesis.hypothesis_id,
                "repair_initial",
                repair_messages,
            )
        )
        call = generator.call(
            repair_messages,
            TR1RepairPlan,
        )
        plan = call.value
        assert isinstance(plan, TR1RepairPlan)
        repair_generation_count += 1

        issues = validate_repair_plan(
            plan,
            hypothesis=hypothesis,
            ps31_card=ps31,
            scope_guard_phrase=scope_guard,
            context=context,
        )
        validation_issues.extend(issues)

        if issues:
            plan = TR1RepairPlan(
                source_hypothesis_id=hypothesis.hypothesis_id,
                axis_id=axis.axis_id,
                action="abstain",
                abstention_reason=(
                    "TR1 repair failed deterministic provenance/scope "
                    "validation: "
                    + "; ".join(
                        f"{row.code}={row.detail}"
                        for row in issues
                    )
                ),
            )

        semantic_audit: TR1SemanticAudit | None = None

        if plan.action == "repair":
            audit_messages = build_semantic_audit_messages(
                plan=plan,
                hypothesis=hypothesis,
                axis=axis,
                scope_guard_phrase=scope_guard,
                premises=premises,
            )
            saved_prompts.append(
                (
                    hypothesis.hypothesis_id,
                    "semantic_audit_initial",
                    audit_messages,
                )
            )
            audit_call = generator.call(
                audit_messages,
                TR1SemanticAudit,
            )
            semantic_audit = audit_call.value
            assert isinstance(
                semantic_audit,
                TR1SemanticAudit,
            )
            semantic_audit_count += 1

            audit_issues = validate_semantic_audit(
                semantic_audit,
                plan=plan,
                hypothesis=hypothesis,
            )
            validation_issues.extend(audit_issues)

            passes = (
                not audit_issues
                and semantic_audit_passes(
                    semantic_audit
                )
            )

            if (
                not passes
                and max_semantic_repairs
            ):
                retry_messages = build_repair_messages(
                    hypothesis=hypothesis,
                    ps31_card=ps31,
                    axis=axis,
                    scope_guard_phrase=scope_guard,
                    premises=premises,
                    prior_failure=semantic_audit,
                )
                saved_prompts.append(
                    (
                        hypothesis.hypothesis_id,
                        "repair_after_semantic_gate",
                        retry_messages,
                    )
                )
                retry_call = generator.call(
                    retry_messages,
                    TR1RepairPlan,
                )
                plan = retry_call.value
                assert isinstance(plan, TR1RepairPlan)
                repair_generation_count += 1
                semantic_repair_count = 1

                retry_issues = validate_repair_plan(
                    plan,
                    hypothesis=hypothesis,
                    ps31_card=ps31,
                    scope_guard_phrase=scope_guard,
                    context=context,
                )
                validation_issues.extend(retry_issues)

                if not retry_issues and plan.action == "repair":
                    audit_messages = build_semantic_audit_messages(
                        plan=plan,
                        hypothesis=hypothesis,
                        axis=axis,
                        scope_guard_phrase=scope_guard,
                        premises=premises,
                    )
                    saved_prompts.append(
                        (
                            hypothesis.hypothesis_id,
                            "semantic_audit_after_repair",
                            audit_messages,
                        )
                    )
                    audit_call = generator.call(
                        audit_messages,
                        TR1SemanticAudit,
                    )
                    semantic_audit = audit_call.value
                    assert isinstance(
                        semantic_audit,
                        TR1SemanticAudit,
                    )
                    semantic_audit_count += 1

                    retry_audit_issues = (
                        validate_semantic_audit(
                            semantic_audit,
                            plan=plan,
                            hypothesis=hypothesis,
                        )
                    )
                    validation_issues.extend(
                        retry_audit_issues
                    )
                    passes = (
                        not retry_audit_issues
                        and semantic_audit_passes(
                            semantic_audit
                        )
                    )
                elif plan.action == "abstain":
                    passes = False
                else:
                    passes = False

            if (
                plan.action == "repair"
                and semantic_audit is not None
                and passes
            ):
                repaired = _make_repaired_hypothesis(
                    hypothesis,
                    plan,
                )
                final_hypotheses.append(repaired)
                id_map[
                    hypothesis.hypothesis_id
                ] = repaired.hypothesis_id

                decisions.append(
                    TR1DecisionCard(
                        source_hypothesis_id=hypothesis.hypothesis_id,
                        source_title=hypothesis.title,
                        axis_id=axis.axis_id,
                        axis_label=axis.label,
                        source_ps31_overall_verdict=ps31.overall_verdict,
                        unsupported_material_units=unsupported_texts,
                        source_premise_statement_ids=list(
                            hypothesis.premise_statement_ids
                        ),
                        final_status="repaired",
                        final_hypothesis_id=repaired.hypothesis_id,
                        final_hypothesis_statement=(
                            repaired.hypothesis_statement
                        ),
                        repair_generation_count=(
                            repair_generation_count
                        ),
                        semantic_audit_count=(
                            semantic_audit_count
                        ),
                        semantic_repair_count=(
                            semantic_repair_count
                        ),
                        plan=plan,
                        semantic_audit=semantic_audit,
                        validation_issues=validation_issues,
                        disposition_reason=(
                            "TR1 repair passed deterministic provenance/"
                            "scope checks and the semantic operand/relation "
                            "gate."
                        ),
                    )
                )
                continue

        abstained_ids.add(hypothesis.hypothesis_id)
        if plan.action == "abstain":
            reason = plan.abstention_reason or (
                "TR1 planner abstained."
            )
        elif semantic_audit is not None:
            reason = (
                "TR1 could not obtain a one-relation repair using only "
                "grounded material operands while preserving scope and "
                "axis fidelity. Final semantic-gate status: "
                f"{semantic_audit.relation_status}."
            )
        else:
            reason = (
                "TR1 deterministic validation rejected the repair."
            )

        decisions.append(
            TR1DecisionCard(
                source_hypothesis_id=hypothesis.hypothesis_id,
                source_title=hypothesis.title,
                axis_id=axis.axis_id,
                axis_label=axis.label,
                source_ps31_overall_verdict=ps31.overall_verdict,
                unsupported_material_units=unsupported_texts,
                source_premise_statement_ids=list(
                    hypothesis.premise_statement_ids
                ),
                final_status="abstained",
                repair_generation_count=repair_generation_count,
                semantic_audit_count=semantic_audit_count,
                semantic_repair_count=semantic_repair_count,
                plan=plan,
                semantic_audit=semantic_audit,
                validation_issues=validation_issues,
                disposition_reason=reason,
            )
        )

    candidate_portfolio = _build_candidate_portfolio(
        portfolio,
        final_hypotheses,
    )
    candidate_axis_report = _build_candidate_axis_report(
        axis_report,
        candidate_portfolio=candidate_portfolio,
        id_map=id_map,
        abstained_ids=abstained_ids,
    )

    counts = Counter(row.final_status for row in decisions)
    targeted = sum(
        row.source_ps31_overall_verdict
        == "unsupported_inferential_leap"
        for row in decisions
    )

    payload = {
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_axis_plan_id": axis_plan.plan_id,
        "source_axis_report_id": axis_report.report_id,
        "source_planning_report_id": planning_report.report_id,
        "source_ps31_report_id": ps31_report.report_id,
        "source_ps31_report_sha256": ps31_report.report_sha256,
        "model": generator.model_name,
        "source_hypothesis_count": len(portfolio.hypotheses),
        "kept_count": counts["kept"],
        "targeted_count": targeted,
        "repaired_count": counts["repaired"],
        "abstained_count": counts["abstained"],
        "repair_generation_count": sum(
            row.repair_generation_count
            for row in decisions
        ),
        "semantic_audit_count": sum(
            row.semantic_audit_count
            for row in decisions
        ),
        "semantic_repair_count": sum(
            row.semantic_repair_count
            for row in decisions
        ),
        "candidate_portfolio_id": candidate_portfolio.portfolio_id,
        "candidate_axis_report_id": candidate_axis_report.report_id,
        "cards": [
            row.model_dump(mode="json")
            for row in decisions
        ],
        "policy": TR1Policy().model_dump(mode="json"),
    }
    payload["report_id"] = _stable_id(
        "tr1_report",
        portfolio.portfolio_id,
        ps31_report.report_id,
        candidate_portfolio.portfolio_id,
        _sha256_json(payload["cards"]),
    )
    report = TR1Report(
        **payload,
        report_sha256=_sha256_json(payload),
    )

    return (
        candidate_portfolio,
        candidate_axis_report,
        report,
        saved_prompts,
    )
