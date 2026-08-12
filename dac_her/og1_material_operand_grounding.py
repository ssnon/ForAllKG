from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.discovery_axis_contracts import (
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
)
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.ig11_endpoint_scope import (
    IG11Blueprint,
    IG11PlanningReport,
    IG11StructuredGenerator,
)


OG1_SCHEMA_VERSION = "og1-material-operand-grounding-audit-v1"
OG1_PROMPT_VERSION = "og1-material-operand-grounding-v2.9.1"

OG1GroundingStatus = Literal[
    "directly_grounded",
    "synthesis_grounded",
    "axis_inspiration_only",
    "unsupported",
    "uncertain",
]
OG1OperandKind = Literal[
    "material_or_system_identity",
    "structural_or_coordination_variable",
    "electronic_descriptor_or_mechanism",
    "energetic_or_thermodynamic_descriptor",
    "adsorption_or_reaction_variable",
    "activity_or_performance_outcome",
    "perturbation_or_condition",
    "other_scientific_operand",
]
_ACCEPTED = {"directly_grounded", "synthesis_grounded"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OG1OperandSupport(StrictModel):
    statement_id: str
    excerpt: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class OG1MaterialOperandReview(StrictModel):
    operand_text: str = Field(min_length=1)
    operand_kind: OG1OperandKind
    grounding_status: OG1GroundingStatus
    supports: list[OG1OperandSupport] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class OG1MaterialOperandAudit(StrictModel):
    schema_version: Literal[
        "og1-material-operand-grounding-audit-v1"
    ] = OG1_SCHEMA_VERSION
    hypothesis_id: str
    axis_id: str
    relation_text: str
    operand_reviews: list[OG1MaterialOperandReview] = Field(min_length=1)
    unlisted_material_operand_texts: list[str] = Field(default_factory=list)
    coverage_complete: bool
    all_material_operands_grounded: bool
    explanation: str = Field(min_length=1)


class OG1ValidationIssue(StrictModel):
    code: str
    detail: str


class OG1AuditRecord(StrictModel):
    hypothesis_id: str
    title: str
    axis_id: str
    axis_label: str
    blueprint_sha256: str
    audit_sha256: str
    audit: OG1MaterialOperandAudit
    valid: bool
    passes_gate: bool
    generation_attempts: int = 1
    repair_count: int = 0
    validation_issues: list[OG1ValidationIssue] = Field(default_factory=list)
    unsupported_operand_texts: list[str] = Field(default_factory=list)
    axis_inspiration_only_operand_texts: list[str] = Field(default_factory=list)
    uncertain_operand_texts: list[str] = Field(default_factory=list)
    disposition_reason: str = Field(min_length=1)


class OG1Policy(StrictModel):
    selected_positive_premises_only: Literal[True] = True
    external_knowledge_allowed: Literal[False] = False
    discovery_axis_is_evidence: Literal[False] = False
    common_knowledge_can_ground_operand: Literal[False] = False
    scope_guard_can_ground_operand: Literal[False] = False
    relation_novelty_can_ground_operand: Literal[False] = False
    every_material_operand_must_be_grounded: Literal[True] = True
    novelty_may_reside_in_relation_not_new_operand: Literal[True] = True
    accepted_grounding_statuses: list[str] = Field(
        default_factory=lambda: sorted(_ACCEPTED)
    )
    rejected_grounding_statuses: list[str] = Field(
        default_factory=lambda: [
            "axis_inspiration_only", "unsupported", "uncertain"
        ]
    )
    max_audit_repairs: Literal[1] = 1
    rejected_hypotheses_are_filtered: Literal[True] = True


class OG1Report(StrictModel):
    schema_version: Literal[
        "og1-material-operand-grounding-report-v1"
    ] = "og1-material-operand-grounding-report-v1"
    report_id: str
    report_sha256: str
    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_axis_plan_id: str
    source_axis_report_id: str
    source_planning_report_id: str
    model: str
    source_hypothesis_count: int
    passed_count: int
    filtered_count: int
    valid_audit_count: int
    audit_repair_count: int
    grounding_status_counts: dict[str, int] = Field(default_factory=dict)
    candidate_portfolio_id: str
    candidate_axis_report_id: str
    records: list[OG1AuditRecord] = Field(default_factory=list)
    policy: OG1Policy = Field(default_factory=OG1Policy)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode()
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _eligible_map(context: HypothesisContext) -> dict[str, Any]:
    return {
        row.statement_id: row
        for row in context.evidence_statements
        if row.eligible_as_premise
    }


def _blueprint_by_axis(
    planning: IG11PlanningReport,
) -> dict[str, tuple[str, IG11Blueprint]]:
    return {
        row.axis_id: (row.blueprint_sha256, row.blueprint)
        for row in planning.blueprint_records
        if row.valid and not row.blueprint.abstain
    }


def _blueprint_premise_ids(blueprint: IG11Blueprint) -> list[str]:
    if blueprint.abstain:
        return []
    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    return sorted(
        set(
            blueprint.endpoint_a.supporting_statement_ids
            + blueprint.endpoint_b.supporting_statement_ids
        )
    )


def build_operand_audit_messages(
    *,
    hypothesis: Any,
    axis: Any,
    blueprint: IG11Blueprint,
    context: HypothesisContext,
    previous_audit: OG1MaterialOperandAudit | None = None,
    validation_issues: list[OG1ValidationIssue] | None = None,
) -> list[dict[str, str]]:
    if blueprint.abstain:
        raise ValueError("OG1 cannot audit an abstaining blueprint")
    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    assert blueprint.novel_bridge is not None
    assert blueprint.scope_envelope is not None

    eligible = _eligible_map(context)
    premises = []
    for sid in hypothesis.premise_statement_ids:
        row = eligible.get(sid)
        if row is None:
            raise ValueError(f"OG1 selected premise is not eligible: {sid}")
        premises.append(
            {
                "statement_id": sid,
                "text": row.text,
                "epistemic_role": row.epistemic_role,
                "claim_kind": row.claim_kind,
                "paper_ids": row.paper_ids,
            }
        )

    system = """You are OG1, the Material Operand Grounding Gate.

Audit ONLY the scientific operands used by ONE proposed hypothesis relation.
Use ONLY selected positive premises. External knowledge is forbidden.
The discovery axis is inspiration only and is NOT evidence.

VALID EPISTEMIC SHAPE:
GROUNDED OPERAND A -- one possibly-unestablished relation -- GROUNDED OPERAND B.

A material operand is a scientific variable, descriptor, entity, mechanism,
condition, perturbation, observable, or outcome whose identity/state/value is
used by relation_text. Examples include formation energy, local geometry,
metal-pair identity, nitrogen coordination, charge redistribution, adsorption
energetics, activity, poisoning-agent identity, or poisoning response.

Do NOT list pure relational grammar ("promotes", "varies with", "depends on",
"is associated with", "is predicted to") or scope-only phrases as operands.

Statuses:
- directly_grounded: selected premise text directly supports the operand.
- synthesis_grounded: an eligible evidence_synthesis premise or multiple
  selected premises jointly ground the operand at the required scope.
- axis_inspiration_only: operand comes from the discovery axis but selected
  positive premises do not ground it.
- unsupported: operand is not grounded by selected positive premises.
- uncertain: deciding would require guessing.

Hard rules:
1. Common knowledge is not positive evidence. A familiar variable such as
   formation energy is unsupported if selected premises do not ground it.
2. Discovery inspiration is not evidence.
3. Relation novelty does not ground a missing operand.
4. Prediction language does not exempt an operand.
5. Scope guards do not create grounding.
6. Enumerate every material operand in relation_text exactly once.
7. operand_text must be an exact contiguous substring of relation_text.
8. Every support excerpt must be an exact contiguous substring of its cited
   selected premise.
9. coverage_complete=true only if all material operands are listed.
10. all_material_operands_grounded=true only if every operand is directly or
    synthesis grounded and no material operand is unlisted.
11. Do not change or repair the hypothesis. Audit only.
"""

    payload = {
        "hypothesis": {
            "hypothesis_id": hypothesis.hypothesis_id,
            "title": hypothesis.title,
            "relation_text": hypothesis.hypothesis_statement,
            "inferential_bridge": hypothesis.inferential_bridge,
            "premise_statement_ids": hypothesis.premise_statement_ids,
        },
        "axis_inspiration_only": {
            "axis_id": axis.axis_id,
            "label": axis.label,
            "subject": axis.proposed_subject,
            "relation": axis.proposed_relation,
            "object": axis.proposed_object,
        },
        "fixed_blueprint": {
            "endpoint_a": blueprint.endpoint_a.model_dump(mode="json"),
            "endpoint_b": blueprint.endpoint_b.model_dump(mode="json"),
            "scope_envelope": blueprint.scope_envelope.model_dump(mode="json"),
            "novel_bridge": blueprint.novel_bridge.model_dump(mode="json"),
        },
        "selected_positive_premises": premises,
    }

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nReturn exactly one OG1MaterialOperandAudit.",
        },
    ]

    if previous_audit is not None:
        issues = validation_issues or []
        messages += [
            {
                "role": "assistant",
                "content": previous_audit.model_dump_json(indent=2),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "OG1 AUDIT REPAIR",
                        "================",
                        "Repair ONLY deterministic identity/provenance/"
                        "self-consistency failures.",
                        "Do not upgrade unsupported or axis-only operands just "
                        "to make the gate pass.",
                        "Do not change relation_text.",
                        "",
                        "Validation issues:",
                        *[f"- {x.code}: {x.detail}" for x in issues],
                        "",
                        "Return a complete replacement OG1MaterialOperandAudit.",
                    ]
                ),
            },
        ]
    return messages


def validate_operand_audit(
    result: OG1MaterialOperandAudit,
    *,
    hypothesis: Any,
    axis_id: str,
    blueprint: IG11Blueprint,
    context: HypothesisContext,
) -> list[OG1ValidationIssue]:
    issues: list[OG1ValidationIssue] = []

    if result.hypothesis_id != hypothesis.hypothesis_id:
        issues.append(OG1ValidationIssue(
            code="hypothesis_id_mismatch",
            detail=f"expected={hypothesis.hypothesis_id}; actual={result.hypothesis_id}",
        ))
    if result.axis_id != axis_id:
        issues.append(OG1ValidationIssue(
            code="axis_id_mismatch",
            detail=f"expected={axis_id}; actual={result.axis_id}",
        ))
    if result.relation_text != hypothesis.hypothesis_statement:
        issues.append(OG1ValidationIssue(
            code="relation_text_mismatch",
            detail="OG1 audit changed hypothesis_statement",
        ))

    if blueprint.abstain or blueprint.novel_bridge is None:
        issues.append(OG1ValidationIssue(
            code="missing_active_blueprint", detail=axis_id
        ))
        return issues

    if hypothesis.hypothesis_statement != blueprint.novel_bridge.relation:
        issues.append(OG1ValidationIssue(
            code="hypothesis_blueprint_relation_mismatch",
            detail="hypothesis_statement != immutable IG1.2a relation",
        ))

    expected_premises = set(_blueprint_premise_ids(blueprint))
    actual_premises = set(hypothesis.premise_statement_ids)
    if actual_premises != expected_premises:
        issues.append(OG1ValidationIssue(
            code="premise_set_mismatch",
            detail=f"blueprint={sorted(expected_premises)}; hypothesis={sorted(actual_premises)}",
        ))

    eligible = _eligible_map(context)
    relation = hypothesis.hypothesis_statement
    operand_texts = [x.operand_text for x in result.operand_reviews]
    if len(operand_texts) != len(set(operand_texts)):
        issues.append(OG1ValidationIssue(
            code="duplicate_operand_text", detail=str(operand_texts)
        ))

    for operand in result.operand_reviews:
        if operand.operand_text not in relation:
            issues.append(OG1ValidationIssue(
                code="operand_not_exact_relation_substring",
                detail=operand.operand_text,
            ))

        support_ids = [x.statement_id for x in operand.supports]
        if len(support_ids) != len(set(support_ids)):
            issues.append(OG1ValidationIssue(
                code="duplicate_operand_support_id",
                detail=operand.operand_text,
            ))

        for support in operand.supports:
            if support.statement_id not in actual_premises:
                issues.append(OG1ValidationIssue(
                    code="operand_support_outside_selected_premises",
                    detail=f"{operand.operand_text}:{support.statement_id}",
                ))
                continue
            statement = eligible.get(support.statement_id)
            if statement is None:
                issues.append(OG1ValidationIssue(
                    code="operand_support_not_eligible",
                    detail=support.statement_id,
                ))
                continue
            if support.excerpt not in statement.text:
                issues.append(OG1ValidationIssue(
                    code="nonverbatim_operand_support",
                    detail=f"{operand.operand_text}:{support.statement_id}",
                ))

        if operand.grounding_status in _ACCEPTED and not operand.supports:
            issues.append(OG1ValidationIssue(
                code="grounded_operand_without_support",
                detail=operand.operand_text,
            ))

        if (
            operand.grounding_status in {"axis_inspiration_only", "unsupported"}
            and operand.supports
        ):
            issues.append(OG1ValidationIssue(
                code="ungrounded_operand_with_positive_support",
                detail=operand.operand_text,
            ))

        if operand.grounding_status == "synthesis_grounded":
            ids = set(support_ids)
            if len(ids) == 1:
                sid = next(iter(ids))
                statement = eligible.get(sid)
                if statement is not None and statement.epistemic_role != "evidence_synthesis":
                    issues.append(OG1ValidationIssue(
                        code="synthesis_without_synthesis_basis",
                        detail=f"{operand.operand_text}: {sid}",
                    ))

    for text in result.unlisted_material_operand_texts:
        if text not in relation:
            issues.append(OG1ValidationIssue(
                code="unlisted_operand_not_relation_substring",
                detail=text,
            ))

    computed = (
        result.coverage_complete
        and not result.unlisted_material_operand_texts
        and all(x.grounding_status in _ACCEPTED for x in result.operand_reviews)
    )
    if result.all_material_operands_grounded != computed:
        issues.append(OG1ValidationIssue(
            code="all_operands_grounded_self_audit_mismatch",
            detail=f"declared={result.all_material_operands_grounded}; computed={computed}",
        ))
    return issues


def operand_audit_passes(audit: OG1MaterialOperandAudit) -> bool:
    return (
        audit.coverage_complete
        and audit.all_material_operands_grounded
        and not audit.unlisted_material_operand_texts
        and all(x.grounding_status in _ACCEPTED for x in audit.operand_reviews)
    )


def _build_candidate_portfolio(
    source: HypothesisPortfolio,
    hypotheses: list[Any],
) -> HypothesisPortfolio:
    data = source.model_dump(mode="json")
    data["hypotheses"] = [x.model_dump(mode="json") for x in hypotheses]
    data["portfolio_id"] = _stable_id(
        "hypothesis_portfolio_og1",
        source.portfolio_id,
        [x.hypothesis_id for x in hypotheses],
    )
    data["abstention_reason"] = (
        None if hypotheses else
        "OG1 filtered all hypotheses because each proposed relation contained "
        "at least one material scientific operand not grounded by selected "
        "positive premises."
    )
    return HypothesisPortfolio.model_validate(data)


def _build_candidate_axis_report(
    source: DiscoveryAxisSynthesisReport,
    *,
    candidate_portfolio: HypothesisPortfolio,
    filtered_ids: set[str],
) -> DiscoveryAxisSynthesisReport:
    data = source.model_dump(mode="json")
    lineages = [
        x.model_dump(mode="json")
        for x in source.lineages
        if x.hypothesis_id not in filtered_ids
    ]
    data["lineages"] = lineages
    data["accepted_hypothesis_count"] = len(candidate_portfolio.hypotheses)
    data["final_portfolio_id"] = candidate_portfolio.portfolio_id
    data["final_portfolio_sha256"] = _sha256_json(candidate_portfolio)
    data["report_id"] = _stable_id(
        "discovery_axis_synthesis_report_og1",
        source.report_id,
        candidate_portfolio.portfolio_id,
        [x["hypothesis_id"] for x in lineages],
    )
    tmp = dict(data)
    tmp.pop("report_sha256", None)
    data["report_sha256"] = _sha256_json(tmp)
    return DiscoveryAxisSynthesisReport.model_validate(data)


def run_og1(
    *,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    axis_plan: DiscoveryAxisPlan,
    axis_report: DiscoveryAxisSynthesisReport,
    planning_report: IG11PlanningReport,
    generator: IG11StructuredGenerator,
    max_audit_repairs: int = 1,
) -> tuple[
    HypothesisPortfolio,
    DiscoveryAxisSynthesisReport,
    OG1Report,
    list[tuple[str, str, list[dict[str, str]]]],
]:
    if max_audit_repairs not in {0, 1}:
        raise ValueError("OG1 max_audit_repairs must be 0 or 1")
    if portfolio.source_context_id != context.context_id:
        raise ValueError("OG1 portfolio/context ID mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("OG1 portfolio/context SHA mismatch")
    if axis_report.final_portfolio_id != portfolio.portfolio_id:
        raise ValueError("OG1 axis report/portfolio mismatch")
    if axis_report.axis_plan_id != axis_plan.plan_id:
        raise ValueError("OG1 axis report/axis-plan mismatch")
    if planning_report.source_context_id != context.context_id:
        raise ValueError("OG1 planning/context mismatch")
    if planning_report.source_axis_plan_id != axis_plan.plan_id:
        raise ValueError("OG1 planning/axis-plan mismatch")

    lineage_by_hypothesis = {
        x.hypothesis_id: x for x in axis_report.lineages
    }
    axis_by_id = {x.axis_id: x for x in axis_plan.axes}
    blueprint_by_axis = _blueprint_by_axis(planning_report)

    records = []
    kept = []
    filtered_ids: set[str] = set()
    prompts = []

    for hypothesis in portfolio.hypotheses:
        lineage = lineage_by_hypothesis.get(hypothesis.hypothesis_id)
        if lineage is None:
            raise ValueError(f"OG1 missing lineage for {hypothesis.hypothesis_id}")
        axis = axis_by_id.get(lineage.axis_id)
        if axis is None:
            raise ValueError(f"OG1 missing axis {lineage.axis_id}")
        item = blueprint_by_axis.get(axis.axis_id)
        if item is None:
            raise ValueError(f"OG1 missing active blueprint {axis.axis_id}")
        blueprint_sha, blueprint = item

        messages = build_operand_audit_messages(
            hypothesis=hypothesis,
            axis=axis,
            blueprint=blueprint,
            context=context,
        )
        prompts.append((hypothesis.hypothesis_id, "operand_audit_initial", messages))
        audit = generator.call(messages, OG1MaterialOperandAudit).value
        assert isinstance(audit, OG1MaterialOperandAudit)
        attempts, repairs = 1, 0

        issues = validate_operand_audit(
            audit,
            hypothesis=hypothesis,
            axis_id=axis.axis_id,
            blueprint=blueprint,
            context=context,
        )
        if issues and max_audit_repairs:
            messages = build_operand_audit_messages(
                hypothesis=hypothesis,
                axis=axis,
                blueprint=blueprint,
                context=context,
                previous_audit=audit,
                validation_issues=issues,
            )
            prompts.append((hypothesis.hypothesis_id, "operand_audit_repair", messages))
            audit = generator.call(messages, OG1MaterialOperandAudit).value
            assert isinstance(audit, OG1MaterialOperandAudit)
            attempts, repairs = 2, 1
            issues = validate_operand_audit(
                audit,
                hypothesis=hypothesis,
                axis_id=axis.axis_id,
                blueprint=blueprint,
                context=context,
            )

        valid = not issues
        passes = valid and operand_audit_passes(audit)
        unsupported = sorted(
            x.operand_text for x in audit.operand_reviews
            if x.grounding_status == "unsupported"
        )
        axis_only = sorted(
            x.operand_text for x in audit.operand_reviews
            if x.grounding_status == "axis_inspiration_only"
        )
        uncertain = sorted(
            x.operand_text for x in audit.operand_reviews
            if x.grounding_status == "uncertain"
        )

        if passes:
            kept.append(hypothesis)
            reason = (
                "All material operands are grounded by selected positive "
                "premises; relation novelty remains for downstream critics."
            )
        else:
            filtered_ids.add(hypothesis.hypothesis_id)
            if not valid:
                reason = (
                    "OG1 audit remained invalid and failed closed: "
                    + "; ".join(f"{x.code}={x.detail}" for x in issues)
                )
            else:
                bits = []
                if axis_only:
                    bits.append(f"axis-only={axis_only}")
                if unsupported:
                    bits.append(f"unsupported={unsupported}")
                if uncertain:
                    bits.append(f"uncertain={uncertain}")
                if audit.unlisted_material_operand_texts:
                    bits.append(f"unlisted={audit.unlisted_material_operand_texts}")
                if not audit.coverage_complete:
                    bits.append("coverage_complete=false")
                reason = (
                    "OG1 filtered the hypothesis because novelty would require "
                    "an ungrounded material operand, not only a new relation"
                )
                if bits:
                    reason += ": " + "; ".join(bits)

        records.append(OG1AuditRecord(
            hypothesis_id=hypothesis.hypothesis_id,
            title=hypothesis.title,
            axis_id=axis.axis_id,
            axis_label=axis.label,
            blueprint_sha256=blueprint_sha,
            audit_sha256=_sha256_json(audit),
            audit=audit,
            valid=valid,
            passes_gate=passes,
            generation_attempts=attempts,
            repair_count=repairs,
            validation_issues=issues,
            unsupported_operand_texts=unsupported,
            axis_inspiration_only_operand_texts=axis_only,
            uncertain_operand_texts=uncertain,
            disposition_reason=reason,
        ))

    candidate_portfolio = _build_candidate_portfolio(portfolio, kept)
    candidate_axis_report = _build_candidate_axis_report(
        axis_report,
        candidate_portfolio=candidate_portfolio,
        filtered_ids=filtered_ids,
    )

    counts: Counter[str] = Counter()
    for record in records:
        for operand in record.audit.operand_reviews:
            counts[operand.grounding_status] += 1

    payload = {
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_axis_plan_id": axis_plan.plan_id,
        "source_axis_report_id": axis_report.report_id,
        "source_planning_report_id": planning_report.report_id,
        "model": generator.model_name,
        "source_hypothesis_count": len(portfolio.hypotheses),
        "passed_count": len(kept),
        "filtered_count": len(filtered_ids),
        "valid_audit_count": sum(x.valid for x in records),
        "audit_repair_count": sum(x.repair_count for x in records),
        "grounding_status_counts": dict(sorted(counts.items())),
        "candidate_portfolio_id": candidate_portfolio.portfolio_id,
        "candidate_axis_report_id": candidate_axis_report.report_id,
        "records": [x.model_dump(mode="json") for x in records],
        "policy": OG1Policy().model_dump(mode="json"),
    }
    payload["report_id"] = _stable_id(
        "og1_report",
        portfolio.portfolio_id,
        axis_report.report_id,
        planning_report.report_id,
        candidate_portfolio.portfolio_id,
        _sha256_json(payload["records"]),
    )
    report = OG1Report(
        **payload,
        report_sha256=_sha256_json(payload),
    )
    return candidate_portfolio, candidate_axis_report, report, prompts
