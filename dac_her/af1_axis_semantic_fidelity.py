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
    HypothesisContext,
    HypothesisPortfolio,
)
from dac_her.ig11_endpoint_scope import IG11StructuredGenerator


AF1_SCHEMA_VERSION = "af1-axis-semantic-fidelity-audit-v1"
AF1_PROMPT_VERSION = "af1-axis-semantic-fidelity-v2.9.1"

AF1ComponentName = Literal[
    "subject",
    "relation",
    "object",
    "label_semantics",
]

AF1ComponentStatus = Literal[
    "preserved",
    "scientifically_equivalent",
    "narrowed_but_faithful",
    "epistemically_weakened_but_faithful",
    "substituted",
    "omitted",
    "unsupported_axis_component",
    "uncertain",
]

AF1AxisEvidenceStatus = Literal[
    "grounded_in_selected_premises",
    "partially_grounded_in_selected_premises",
    "not_grounded_in_selected_premises",
    "uncertain",
]

_PASS_STATUSES = {
    "preserved",
    "scientifically_equivalent",
    "narrowed_but_faithful",
    "epistemically_weakened_but_faithful",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AF1AxisComponentReview(StrictModel):
    component: AF1ComponentName
    axis_text: str = Field(min_length=1)

    status: AF1ComponentStatus

    hypothesis_expression: str | None = None
    axis_evidence_status: AF1AxisEvidenceStatus

    explanation: str = Field(min_length=1)


class AF1AxisSemanticFidelityAudit(StrictModel):
    schema_version: Literal[
        "af1-axis-semantic-fidelity-audit-v1"
    ] = AF1_SCHEMA_VERSION

    hypothesis_id: str
    axis_id: str
    hypothesis_statement: str

    component_reviews: list[AF1AxisComponentReview] = Field(min_length=1)

    component_coverage_complete: bool
    all_essential_axis_semantics_preserved: bool

    overall_status: Literal[
        "faithful",
        "faithful_with_narrowing_or_epistemic_weakening",
        "axis_fidelity_lost",
        "uncertain",
    ]

    explanation: str = Field(min_length=1)


class AF1ValidationIssue(StrictModel):
    code: str
    detail: str


class AF1AuditRecord(StrictModel):
    hypothesis_id: str
    title: str
    axis_id: str
    axis_label: str

    audit_sha256: str
    audit: AF1AxisSemanticFidelityAudit

    valid: bool
    passes_gate: bool

    generation_attempts: int = 1
    repair_count: int = 0
    validation_issues: list[AF1ValidationIssue] = Field(default_factory=list)

    failed_components: list[str] = Field(default_factory=list)
    disposition_reason: str = Field(min_length=1)


class AF1Policy(StrictModel):
    discovery_axis_is_inspiration_not_evidence: Literal[True] = True
    external_knowledge_allowed: Literal[False] = False

    lexical_identity_required: Literal[False] = False
    semantic_construct_identity_required: Literal[True] = True

    subject_substitution_allowed: Literal[False] = False
    object_substitution_allowed: Literal[False] = False
    relation_construct_substitution_allowed: Literal[False] = False

    scope_narrowing_allowed_when_same_construct: Literal[True] = True
    epistemic_relation_weakening_allowed_when_same_relation_question: Literal[True] = True

    unsupported_axis_component_can_be_replaced_by_adjacent_construct: Literal[False] = False

    max_audit_repairs: Literal[1] = 1
    rejected_hypotheses_are_filtered: Literal[True] = True


class AF1Report(StrictModel):
    schema_version: Literal[
        "af1-axis-semantic-fidelity-report-v1"
    ] = "af1-axis-semantic-fidelity-report-v1"

    report_id: str
    report_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_axis_plan_id: str
    source_axis_report_id: str

    model: str

    source_hypothesis_count: int
    passed_count: int
    filtered_count: int
    valid_audit_count: int
    audit_repair_count: int

    component_status_counts: dict[str, int] = Field(default_factory=dict)

    candidate_portfolio_id: str
    candidate_axis_report_id: str

    records: list[AF1AuditRecord] = Field(default_factory=list)
    policy: AF1Policy = Field(default_factory=AF1Policy)


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


def _eligible_map(
    context: HypothesisContext,
) -> dict[str, Any]:
    return {
        row.statement_id: row
        for row in context.evidence_statements
        if row.eligible_as_premise
    }


def _axis_components(
    axis: DiscoveryAxis,
) -> list[tuple[AF1ComponentName, str]]:
    structured: list[tuple[AF1ComponentName, str]] = []
    for name, value in [
        ("subject", axis.proposed_subject),
        ("relation", axis.proposed_relation),
        ("object", axis.proposed_object),
    ]:
        text = str(value or "").strip()
        if text:
            structured.append((name, text))

    if structured:
        return structured

    label = str(axis.label or "").strip()
    if not label:
        raise ValueError(
            f"AF1 axis has neither structured semantics nor label: {axis.axis_id}"
        )
    return [("label_semantics", label)]


def _expected_component_map(
    axis: DiscoveryAxis,
) -> dict[str, str]:
    return dict(_axis_components(axis))


def build_af1_messages(
    *,
    hypothesis: Any,
    axis: DiscoveryAxis,
    context: HypothesisContext,
    previous_audit: AF1AxisSemanticFidelityAudit | None = None,
    validation_issues: list[AF1ValidationIssue] | None = None,
) -> list[dict[str, str]]:
    eligible = _eligible_map(context)

    premises = []
    for sid in hypothesis.premise_statement_ids:
        row = eligible.get(sid)
        if row is None:
            raise ValueError(
                f"AF1 selected premise is not eligible: {sid}"
            )
        premises.append(
            {
                "statement_id": sid,
                "text": row.text,
                "epistemic_role": row.epistemic_role,
                "claim_kind": row.claim_kind,
                "paper_ids": row.paper_ids,
            }
        )

    components = [
        {
            "component": name,
            "axis_text": text,
        }
        for name, text in _axis_components(axis)
    ]

    system = """You are AF1, the Axis Semantic Fidelity Gate.

A discovery axis inspired a generated scientific hypothesis. Your task is to
decide whether the FINAL HYPOTHESIS still investigates the SAME SCIENTIFIC
CONSTRUCTS and RELATION assigned by that axis.

This is NOT a lexical-overlap test.
This is NOT a scientific-truth or novelty test.
This is NOT an operand-grounding gate; OG1 handles hypothesis operands.

The discovery axis is inspiration only and is NOT positive evidence.
Use selected positive premises only to diagnose whether an axis construct was
groundable or whether the generator silently replaced it with an adjacent
grounded construct. Do not use external knowledge beyond scientific semantic
interpretation necessary to compare the supplied phrases.

COMPONENT STATUSES

preserved
- the hypothesis explicitly retains the same scientific construct and semantic
  role as the axis component.

scientifically_equivalent
- wording differs, but the scientific construct is materially the same.
  Examples may include conventional synonymous notation or equivalent
  descriptor phrasing. Shared keywords alone are not enough.

narrowed_but_faithful
- the hypothesis uses a stricter system/scope/instance of the SAME construct.
- This may apply to subject/object/label semantics.
- It does NOT permit replacing one descriptor with a merely correlated,
  mechanistically adjacent, or broader/narrower DIFFERENT construct.

epistemically_weakened_but_faithful
- RELATION COMPONENT ONLY.
- The hypothesis tests the same directional/conditional scientific relation
  but deliberately weakens epistemic strength, e.g. causal language to a
  bounded association/conditional prediction, without changing the subject or
  object constructs.
- This does not permit reversing direction or replacing the relation with a
  different mechanism.

substituted
- the hypothesis replaces the axis construct with a scientifically adjacent
  but materially different construct.
- Example: "bonding-antibonding balance" -> "strong bonding to neighboring N".
- Example: "adsorption-desorption compatibility" -> "differentiated H*/OH*
  adsorption roles".
- Semantic relatedness, shared words, or plausible mechanistic linkage does
  not make these equivalent.

omitted
- an essential axis component does not appear in the hypothesis semantics.

unsupported_axis_component
- the selected positive premises do not ground the axis component, and the
  hypothesis avoids it rather than faithfully investigating it.
- Do NOT reward substitution merely because the substitute is grounded.

uncertain
- the supplied axis/hypothesis semantics are insufficient to decide without
  guessing.

HARD RULES

1. Review exactly every supplied axis component once.
2. axis_text must exactly equal the supplied axis component text.
3. For preserved/equivalent/narrowed/epistemically-weakened/substituted,
   hypothesis_expression must quote an exact contiguous substring from
   hypothesis_statement that realizes or replaces the axis component.
4. For omitted or unsupported_axis_component, hypothesis_expression may be
   null.
5. "epistemically_weakened_but_faithful" is allowed ONLY for component=relation.
6. Lexical similarity cannot rescue scientific construct substitution.
7. A grounded adjacent construct cannot substitute for an unsupported axis
   construct and still count as faithful.
8. Narrowing is faithful only when scientific variable/descriptor identity is
   preserved. A different observable is substitution, not narrowing.
9. Relation weakening is faithful only when the same scientific question is
   retained. Association does not preserve an axis if subject/object were
   themselves replaced.
10. all_essential_axis_semantics_preserved=true only if every reviewed component
    is preserved, scientifically_equivalent, narrowed_but_faithful, or (for
    relation only) epistemically_weakened_but_faithful.
11. overall_status:
    - faithful: every component is preserved/scientifically_equivalent.
    - faithful_with_narrowing_or_epistemic_weakening: all pass, with at least
      one narrowed or epistemically weakened component.
    - axis_fidelity_lost: any substituted/omitted/unsupported_axis_component.
    - uncertain: no hard failure but at least one uncertain component.
12. Do not repair or rewrite the hypothesis. Audit only.
"""

    payload = {
        "axis": {
            "axis_id": axis.axis_id,
            "label": axis.label,
            "components_to_review": components,
            "rendered_path": axis.rendered_path,
        },
        "hypothesis": {
            "hypothesis_id": hypothesis.hypothesis_id,
            "title": hypothesis.title,
            "hypothesis_statement": hypothesis.hypothesis_statement,
            "inferential_bridge": hypothesis.inferential_bridge,
        },
        "selected_positive_premises_for_axis_support_diagnosis_only": premises,
    }

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nReturn exactly one AF1AxisSemanticFidelityAudit.",
        },
    ]

    if previous_audit is not None:
        issues = validation_issues or []
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": previous_audit.model_dump_json(indent=2),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "AF1 AUDIT REPAIR",
                            "================",
                            "Repair ONLY deterministic identity, exact-quote, "
                            "component-coverage, or self-consistency failures.",
                            "Do not change a scientifically supported "
                            "substituted/omitted/unsupported classification "
                            "merely to make the gate pass.",
                            "",
                            "Validation issues:",
                            *[
                                f"- {row.code}: {row.detail}"
                                for row in issues
                            ],
                            "",
                            "Return a complete replacement "
                            "AF1AxisSemanticFidelityAudit.",
                        ]
                    ),
                },
            ]
        )

    return messages


def _computed_overall(
    reviews: list[AF1AxisComponentReview],
    *,
    coverage_complete: bool,
) -> tuple[bool, str]:
    if not coverage_complete:
        return False, "axis_fidelity_lost"

    statuses = [row.status for row in reviews]
    if any(
        status in {
            "substituted",
            "omitted",
            "unsupported_axis_component",
        }
        for status in statuses
    ):
        return False, "axis_fidelity_lost"

    if any(status == "uncertain" for status in statuses):
        return False, "uncertain"

    all_pass = all(status in _PASS_STATUSES for status in statuses)
    if not all_pass:
        return False, "axis_fidelity_lost"

    if any(
        status in {
            "narrowed_but_faithful",
            "epistemically_weakened_but_faithful",
        }
        for status in statuses
    ):
        return True, "faithful_with_narrowing_or_epistemic_weakening"

    return True, "faithful"


def validate_af1_audit(
    result: AF1AxisSemanticFidelityAudit,
    *,
    hypothesis: Any,
    axis: DiscoveryAxis,
) -> list[AF1ValidationIssue]:
    issues: list[AF1ValidationIssue] = []

    if result.hypothesis_id != hypothesis.hypothesis_id:
        issues.append(
            AF1ValidationIssue(
                code="hypothesis_id_mismatch",
                detail=(
                    f"expected={hypothesis.hypothesis_id}; "
                    f"actual={result.hypothesis_id}"
                ),
            )
        )

    if result.axis_id != axis.axis_id:
        issues.append(
            AF1ValidationIssue(
                code="axis_id_mismatch",
                detail=f"expected={axis.axis_id}; actual={result.axis_id}",
            )
        )

    if result.hypothesis_statement != hypothesis.hypothesis_statement:
        issues.append(
            AF1ValidationIssue(
                code="hypothesis_statement_mismatch",
                detail="AF1 audit changed hypothesis_statement",
            )
        )

    expected = _expected_component_map(axis)
    actual_names = [row.component for row in result.component_reviews]

    if len(actual_names) != len(set(actual_names)):
        issues.append(
            AF1ValidationIssue(
                code="duplicate_component_review",
                detail=str(actual_names),
            )
        )

    if set(actual_names) != set(expected):
        issues.append(
            AF1ValidationIssue(
                code="component_set_mismatch",
                detail=(
                    f"expected={sorted(expected)}; "
                    f"actual={sorted(set(actual_names))}"
                ),
            )
        )

    hypothesis_text = hypothesis.hypothesis_statement

    for row in result.component_reviews:
        expected_text = expected.get(row.component)
        if expected_text is not None and row.axis_text != expected_text:
            issues.append(
                AF1ValidationIssue(
                    code="axis_component_text_mismatch",
                    detail=(
                        f"{row.component}: expected={expected_text!r}; "
                        f"actual={row.axis_text!r}"
                    ),
                )
            )

        if (
            row.status == "epistemically_weakened_but_faithful"
            and row.component != "relation"
        ):
            issues.append(
                AF1ValidationIssue(
                    code="epistemic_weakening_nonrelation_component",
                    detail=row.component,
                )
            )

        needs_expression = row.status in {
            "preserved",
            "scientifically_equivalent",
            "narrowed_but_faithful",
            "epistemically_weakened_but_faithful",
            "substituted",
        }
        if needs_expression and not row.hypothesis_expression:
            issues.append(
                AF1ValidationIssue(
                    code="missing_hypothesis_expression",
                    detail=row.component,
                )
            )
        elif (
            row.hypothesis_expression is not None
            and row.hypothesis_expression not in hypothesis_text
        ):
            issues.append(
                AF1ValidationIssue(
                    code="hypothesis_expression_not_exact_substring",
                    detail=(
                        f"{row.component}: {row.hypothesis_expression!r}"
                    ),
                )
            )

    coverage = (
        len(actual_names) == len(expected)
        and len(actual_names) == len(set(actual_names))
        and set(actual_names) == set(expected)
    )
    if result.component_coverage_complete != coverage:
        issues.append(
            AF1ValidationIssue(
                code="component_coverage_self_audit_mismatch",
                detail=(
                    f"declared={result.component_coverage_complete}; "
                    f"computed={coverage}"
                ),
            )
        )

    computed_pass, computed_overall = _computed_overall(
        result.component_reviews,
        coverage_complete=coverage,
    )

    if result.all_essential_axis_semantics_preserved != computed_pass:
        issues.append(
            AF1ValidationIssue(
                code="all_semantics_preserved_self_audit_mismatch",
                detail=(
                    f"declared={result.all_essential_axis_semantics_preserved}; "
                    f"computed={computed_pass}"
                ),
            )
        )

    if result.overall_status != computed_overall:
        issues.append(
            AF1ValidationIssue(
                code="overall_status_self_audit_mismatch",
                detail=(
                    f"declared={result.overall_status}; "
                    f"computed={computed_overall}"
                ),
            )
        )

    return issues


def af1_audit_passes(
    audit: AF1AxisSemanticFidelityAudit,
) -> bool:
    computed_pass, _ = _computed_overall(
        audit.component_reviews,
        coverage_complete=audit.component_coverage_complete,
    )
    return (
        audit.all_essential_axis_semantics_preserved
        and computed_pass
    )


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
        "hypothesis_portfolio_af1",
        source.portfolio_id,
        [row.hypothesis_id for row in hypotheses],
    )
    if hypotheses:
        data["abstention_reason"] = None
    else:
        data["abstention_reason"] = (
            "AF1 filtered all hypotheses because the generated scientific "
            "relation did not preserve the essential semantics of its "
            "assigned discovery axis."
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
        row.model_dump(mode="json")
        for row in source.lineages
        if row.hypothesis_id not in filtered_ids
    ]
    data["lineages"] = lineages
    data["accepted_hypothesis_count"] = len(
        candidate_portfolio.hypotheses
    )
    data["final_portfolio_id"] = candidate_portfolio.portfolio_id
    data["final_portfolio_sha256"] = _sha256_json(candidate_portfolio)

    data["report_id"] = _stable_id(
        "discovery_axis_synthesis_report_af1",
        source.report_id,
        candidate_portfolio.portfolio_id,
        [row["hypothesis_id"] for row in lineages],
    )

    tmp = dict(data)
    tmp.pop("report_sha256", None)
    data["report_sha256"] = _sha256_json(tmp)

    return DiscoveryAxisSynthesisReport.model_validate(data)


def run_af1(
    *,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    axis_plan: DiscoveryAxisPlan,
    axis_report: DiscoveryAxisSynthesisReport,
    generator: IG11StructuredGenerator,
    max_audit_repairs: int = 1,
) -> tuple[
    HypothesisPortfolio,
    DiscoveryAxisSynthesisReport,
    AF1Report,
    list[tuple[str, str, list[dict[str, str]]]],
]:
    if max_audit_repairs not in {0, 1}:
        raise ValueError("AF1 max_audit_repairs must be 0 or 1")

    if portfolio.source_context_id != context.context_id:
        raise ValueError("AF1 portfolio/context ID mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("AF1 portfolio/context SHA mismatch")
    if axis_report.final_portfolio_id != portfolio.portfolio_id:
        raise ValueError("AF1 axis report/portfolio mismatch")
    if axis_report.axis_plan_id != axis_plan.plan_id:
        raise ValueError("AF1 axis report/axis-plan mismatch")

    lineage_by_hypothesis = {
        row.hypothesis_id: row
        for row in axis_report.lineages
    }
    axis_by_id = {
        row.axis_id: row
        for row in axis_plan.axes
    }

    records: list[AF1AuditRecord] = []
    kept_hypotheses: list[Any] = []
    filtered_ids: set[str] = set()
    prompts: list[
        tuple[str, str, list[dict[str, str]]]
    ] = []

    for hypothesis in portfolio.hypotheses:
        lineage = lineage_by_hypothesis.get(
            hypothesis.hypothesis_id
        )
        if lineage is None:
            raise ValueError(
                "AF1 missing lineage for "
                f"{hypothesis.hypothesis_id}"
            )

        axis = axis_by_id.get(lineage.axis_id)
        if axis is None:
            raise ValueError(
                f"AF1 missing axis {lineage.axis_id}"
            )

        messages = build_af1_messages(
            hypothesis=hypothesis,
            axis=axis,
            context=context,
        )
        prompts.append(
            (
                hypothesis.hypothesis_id,
                "axis_semantic_fidelity_initial",
                messages,
            )
        )

        audit = generator.call(
            messages,
            AF1AxisSemanticFidelityAudit,
        ).value
        assert isinstance(
            audit,
            AF1AxisSemanticFidelityAudit,
        )

        attempts = 1
        repairs = 0

        issues = validate_af1_audit(
            audit,
            hypothesis=hypothesis,
            axis=axis,
        )

        if issues and max_audit_repairs:
            messages = build_af1_messages(
                hypothesis=hypothesis,
                axis=axis,
                context=context,
                previous_audit=audit,
                validation_issues=issues,
            )
            prompts.append(
                (
                    hypothesis.hypothesis_id,
                    "axis_semantic_fidelity_repair",
                    messages,
                )
            )
            audit = generator.call(
                messages,
                AF1AxisSemanticFidelityAudit,
            ).value
            assert isinstance(
                audit,
                AF1AxisSemanticFidelityAudit,
            )
            attempts = 2
            repairs = 1

            issues = validate_af1_audit(
                audit,
                hypothesis=hypothesis,
                axis=axis,
            )

        valid = not issues
        passes = valid and af1_audit_passes(audit)

        failed_components = sorted(
            row.component
            for row in audit.component_reviews
            if row.status not in _PASS_STATUSES
        )

        if passes:
            kept_hypotheses.append(hypothesis)
            reason = (
                "The hypothesis preserves the essential scientific "
                "constructs and relation semantics of its assigned axis."
            )
        else:
            filtered_ids.add(hypothesis.hypothesis_id)
            if not valid:
                reason = (
                    "AF1 structured audit remained invalid after bounded "
                    "repair and failed closed: "
                    + "; ".join(
                        f"{row.code}={row.detail}"
                        for row in issues
                    )
                )
            else:
                details = [
                    (
                        f"{row.component}={row.status}"
                        + (
                            f" ({row.axis_text!r} -> "
                            f"{row.hypothesis_expression!r})"
                            if row.hypothesis_expression
                            else f" ({row.axis_text!r})"
                        )
                    )
                    for row in audit.component_reviews
                    if row.status not in _PASS_STATUSES
                ]
                reason = (
                    "AF1 filtered the hypothesis because at least one "
                    "essential discovery-axis construct was substituted, "
                    "omitted, unsupported, or uncertain"
                )
                if details:
                    reason += ": " + "; ".join(details)

        records.append(
            AF1AuditRecord(
                hypothesis_id=hypothesis.hypothesis_id,
                title=hypothesis.title,
                axis_id=axis.axis_id,
                axis_label=axis.label,
                audit_sha256=_sha256_json(audit),
                audit=audit,
                valid=valid,
                passes_gate=passes,
                generation_attempts=attempts,
                repair_count=repairs,
                validation_issues=issues,
                failed_components=failed_components,
                disposition_reason=reason,
            )
        )

    candidate_portfolio = _build_candidate_portfolio(
        portfolio,
        kept_hypotheses,
    )
    candidate_axis_report = _build_candidate_axis_report(
        axis_report,
        candidate_portfolio=candidate_portfolio,
        filtered_ids=filtered_ids,
    )

    counts: Counter[str] = Counter()
    for record in records:
        for review in record.audit.component_reviews:
            counts[review.status] += 1

    payload = {
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_axis_plan_id": axis_plan.plan_id,
        "source_axis_report_id": axis_report.report_id,
        "model": generator.model_name,
        "source_hypothesis_count": len(portfolio.hypotheses),
        "passed_count": len(kept_hypotheses),
        "filtered_count": len(filtered_ids),
        "valid_audit_count": sum(
            row.valid for row in records
        ),
        "audit_repair_count": sum(
            row.repair_count for row in records
        ),
        "component_status_counts": dict(sorted(counts.items())),
        "candidate_portfolio_id": candidate_portfolio.portfolio_id,
        "candidate_axis_report_id": candidate_axis_report.report_id,
        "records": [
            row.model_dump(mode="json")
            for row in records
        ],
        "policy": AF1Policy().model_dump(mode="json"),
    }

    payload["report_id"] = _stable_id(
        "af1_report",
        portfolio.portfolio_id,
        axis_report.report_id,
        candidate_portfolio.portfolio_id,
        _sha256_json(payload["records"]),
    )

    report = AF1Report(
        **payload,
        report_sha256=_sha256_json(payload),
    )

    return (
        candidate_portfolio,
        candidate_axis_report,
        report,
        prompts,
    )
