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
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from dac_her.ig11_endpoint_scope import (
    IG11AxisEvidenceAudit,
    IG11Blueprint,
    IG11PlanningReport,
    IG11StructuredGenerator,
)


SC1_SCHEMA_VERSION = "sc1-endpoint-pair-scope-compatibility-v1"
SC1_PROMPT_VERSION = "sc1-endpoint-pair-scope-v2.9.1"

SC1CompatibilityStatus = Literal[
    "same_system",
    "matched_system_variant",
    "shared_explicit_family",
    "cross_system_transfer_supported",
    "cross_system_transfer_unjustified",
    "insufficient_scope_information",
]

SC1SupportKind = Literal[
    "same_system_identity",
    "matched_variant_basis",
    "shared_family_basis",
    "cross_system_transfer_basis",
    "scope_limitation",
]

_ACCEPTED_SCOPE_STATUSES = {
    "same_system",
    "matched_system_variant",
    "shared_explicit_family",
    "cross_system_transfer_supported",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SC1ScopeSupport(StrictModel):
    statement_id: str
    excerpt: str = Field(min_length=1)
    support_kind: SC1SupportKind
    explanation: str = Field(min_length=1)


class SC1EndpointScopeSummary(StrictModel):
    endpoint_id: Literal["endpoint_a", "endpoint_b"]
    anchor_statement_id: str
    demonstrated_scope_summary: str = Field(min_length=1)
    concrete_system_or_family_labels: list[str] = Field(
        default_factory=list
    )
    scope_breadth: Literal[
        "specific_system",
        "multi_system",
        "cross_paper_synthesis",
        "generic_within_premise",
        "unclear",
    ]


class SC1EndpointPairScopeAudit(StrictModel):
    schema_version: Literal[
        "sc1-endpoint-pair-scope-compatibility-v1"
    ] = SC1_SCHEMA_VERSION

    hypothesis_id: str
    axis_id: str

    endpoint_a_anchor_statement_id: str
    endpoint_b_anchor_statement_id: str
    proposed_relation: str

    endpoint_a_scope: SC1EndpointScopeSummary
    endpoint_b_scope: SC1EndpointScopeSummary

    status: SC1CompatibilityStatus
    scope_compatible: bool

    compatibility_supports: list[SC1ScopeSupport] = Field(
        default_factory=list
    )
    transfer_supports: list[SC1ScopeSupport] = Field(
        default_factory=list
    )
    limiting_supports: list[SC1ScopeSupport] = Field(
        default_factory=list
    )

    relation_requires_scope_pairing: bool
    comparison_basis: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    missing_scope_link: str | None = None


class SC1ValidationIssue(StrictModel):
    code: str
    detail: str


class SC1AuditRecord(StrictModel):
    hypothesis_id: str
    title: str
    axis_id: str
    axis_label: str

    blueprint_sha256: str
    audit_sha256: str
    audit: SC1EndpointPairScopeAudit

    valid: bool
    passes_gate: bool

    generation_attempts: int = 1
    repair_count: int = 0
    validation_issues: list[SC1ValidationIssue] = Field(
        default_factory=list
    )

    disposition_reason: str = Field(min_length=1)


class SC1Policy(StrictModel):
    source_scope_only: Literal[True] = True
    external_knowledge_allowed: Literal[False] = False
    discovery_axis_is_evidence: Literal[False] = False

    different_paper_is_automatic_mismatch: Literal[False] = False
    same_paper_is_automatic_match: Literal[False] = False
    generic_domain_umbrella_is_sufficient: Literal[False] = False
    scope_guard_can_create_compatibility: Literal[False] = False

    accepted_statuses: list[str] = Field(
        default_factory=lambda: sorted(_ACCEPTED_SCOPE_STATUSES)
    )
    rejected_statuses: list[str] = Field(
        default_factory=lambda: [
            "cross_system_transfer_unjustified",
            "insufficient_scope_information",
        ]
    )

    max_audit_repairs: Literal[1] = 1
    rejected_hypotheses_are_filtered: Literal[True] = True


class SC1Report(StrictModel):
    schema_version: Literal[
        "sc1-endpoint-pair-scope-report-v1"
    ] = "sc1-endpoint-pair-scope-report-v1"

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

    status_counts: dict[str, int] = Field(default_factory=dict)

    candidate_portfolio_id: str
    candidate_axis_report_id: str

    records: list[SC1AuditRecord] = Field(default_factory=list)
    policy: SC1Policy = Field(default_factory=SC1Policy)


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


def _axis_audit_by_axis(
    planning: IG11PlanningReport,
) -> dict[str, IG11AxisEvidenceAudit]:
    return {
        row.axis_id: row.audit
        for row in planning.audit_records
        if row.valid
    }


def _blueprint_by_axis(
    planning: IG11PlanningReport,
) -> dict[str, tuple[str, IG11Blueprint]]:
    return {
        row.axis_id: (row.blueprint_sha256, row.blueprint)
        for row in planning.blueprint_records
        if row.valid and not row.blueprint.abstain
    }


def _statement_review_for_anchor(
    audit: IG11AxisEvidenceAudit,
    anchor_id: str,
) -> Any | None:
    for row in audit.statement_reviews:
        if row.statement_id == anchor_id:
            return row
    return None


def build_scope_audit_messages(
    *,
    hypothesis: Any,
    axis: Any,
    blueprint: IG11Blueprint,
    axis_audit: IG11AxisEvidenceAudit,
    context: HypothesisContext,
    previous_audit: SC1EndpointPairScopeAudit | None = None,
    validation_issues: list[SC1ValidationIssue] | None = None,
) -> list[dict[str, str]]:
    if blueprint.abstain:
        raise ValueError("SC1 cannot audit an abstaining blueprint")
    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    assert blueprint.novel_bridge is not None

    eligible = _eligible_map(context)
    selected_ids = sorted(
        set(
            blueprint.endpoint_a.supporting_statement_ids
            + blueprint.endpoint_b.supporting_statement_ids
        )
    )

    premises = []
    for sid in selected_ids:
        row = eligible[sid]
        premises.append(
            {
                "statement_id": sid,
                "text": row.text,
                "epistemic_role": row.epistemic_role,
                "claim_kind": row.claim_kind,
                "paper_ids": row.paper_ids,
            }
        )

    endpoint_a_review = _statement_review_for_anchor(
        axis_audit,
        blueprint.endpoint_a.anchor_statement_id,
    )
    endpoint_b_review = _statement_review_for_anchor(
        axis_audit,
        blueprint.endpoint_b.anchor_statement_id,
    )

    endpoint_payload = {
        "endpoint_a": {
            **blueprint.endpoint_a.model_dump(mode="json"),
            "source_statement_review": (
                endpoint_a_review.model_dump(mode="json")
                if endpoint_a_review is not None
                else None
            ),
        },
        "endpoint_b": {
            **blueprint.endpoint_b.model_dump(mode="json"),
            "source_statement_review": (
                endpoint_b_review.model_dump(mode="json")
                if endpoint_b_review is not None
                else None
            ),
        },
    }

    system = """You are SC1, the Endpoint Pair Scope Compatibility Gate for
scientific hypothesis generation.

Your task is NOT to decide whether the proposed relation is already proven.
The relation is intentionally allowed to be novel.

Your task is narrower:

Can the TWO GROUNDED ENDPOINT CLAIMS legitimately serve as the two terminals of
this one proposed relation without importing either endpoint claim into a
material/system scope where that endpoint is not grounded?

Use ONLY:
- selected positive premises;
- endpoint grounding/scope excerpts;
- the validated axis-evidence audit.

The discovery axis is inspiration only. External knowledge is forbidden.

STATUS DEFINITIONS

same_system
- both endpoint claims are grounded in the same concrete material/model/system
  context at the granularity required by the proposed relation.

matched_system_variant
- endpoints concern explicitly matched variants of the same material/model
  family, e.g. controlled variants differing in the relevant structural
  variable, and the selected evidence itself establishes that matching.

shared_explicit_family
- the endpoint claims are each grounded at a COMMON EXPLICIT FAMILY scope
  broad enough for the proposed relation.
- Merely saying both are "dual-atom catalysts", "DACs", "modeled systems", etc.
  is NOT sufficient unless the selected premises themselves support
  family-level applicability of BOTH endpoint claims at the granularity needed
  by the proposed relation.
- Do not use this status when one endpoint is only grounded in one specific
  system and the other endpoint is grounded in a different specific system.

cross_system_transfer_supported
- the endpoints are from different concrete systems, but selected positive
  premises explicitly support the relevant cross-system transfer,
  generalization, or synthesis needed to use them as terminals of the proposed
  relation.

cross_system_transfer_unjustified
- the proposed relation would require carrying an endpoint claim from one
  concrete system/material context into another context, but selected premises
  do not support that transfer.

insufficient_scope_information
- selected evidence does not reveal enough system/family scope to determine
  compatibility without guessing.

CRITICAL DISTINCTIONS

1. DIFFERENT PAPER IDS DO NOT AUTOMATICALLY MEAN INCOMPATIBLE.
   Evaluate scientific system/material scope, not citation identity.

2. SAME PAPER ID DOES NOT AUTOMATICALLY MEAN COMPATIBLE.
   One paper can contain unrelated systems.

3. A MISSING DIRECT A→B COMPARISON IS NOT BY ITSELF A SCOPE FAILURE.
   The relation is allowed to be a hypothesis.
   Ask whether A and B are valid terminals at a shared or supported-transfer
   scope, not whether A→B has already been observed.

4. A BROAD SCOPE GUARD DOES NOT CREATE EVIDENCE.
   Phrases such as "within reported dual-atom model contexts" cannot make two
   specific incompatible systems comparable.

5. CROSS-PAPER / MULTI-SYSTEM SYNTHESIS CAN SUPPORT COMPATIBILITY ONLY WHEN THE
   PREMISE TEXT ACTUALLY SUPPORTS THE RELEVANT FAMILY-LEVEL CLAIM.
   Do not infer this solely from multiple paper_ids.

6. SPECIFIC + BROAD CAN BE COMPATIBLE ONLY WHEN THE BROAD ENDPOINT REALLY
   APPLIES TO A FAMILY THAT INCLUDES THE specific endpoint context at the
   required granularity. Do not assume inclusion from generic terminology.

7. RELATION NOVELTY AND SCOPE COMPATIBILITY ARE SEPARATE.
   A genuinely novel relation can pass SC1.
   A scientifically plausible relation can still fail SC1 if endpoint scopes
   require unsupported transfer.

PROVENANCE
Every compatibility_support, transfer_support, or limiting_support must quote a
short exact contiguous substring from its cited selected premise.
"""

    payload = {
        "hypothesis": {
            "hypothesis_id": hypothesis.hypothesis_id,
            "title": hypothesis.title,
            "hypothesis_statement": hypothesis.hypothesis_statement,
            "premise_statement_ids": hypothesis.premise_statement_ids,
        },
        "axis_inspiration_only": {
            "axis_id": axis.axis_id,
            "label": axis.label,
            "subject": axis.proposed_subject,
            "relation": axis.proposed_relation,
            "object": axis.proposed_object,
        },
        "proposed_relation": blueprint.novel_bridge.relation,
        "endpoints": endpoint_payload,
        "scope_envelope": (
            blueprint.scope_envelope.model_dump(mode="json")
            if blueprint.scope_envelope is not None
            else None
        ),
        "selected_positive_premises": premises,
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
            + "\n\nReturn exactly one SC1EndpointPairScopeAudit.",
        },
    ]

    if previous_audit is not None:
        issues = validation_issues or []
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": previous_audit.model_dump_json(
                        indent=2
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "SC1 AUDIT REPAIR",
                            "================",
                            "The previous structured scope audit failed "
                            "deterministic identity/provenance validation.",
                            "Repair ONLY those validation failures. Do not "
                            "change the scientific scope classification merely "
                            "to make validation pass.",
                            "",
                            "Validation issues:",
                            *[
                                f"- {row.code}: {row.detail}"
                                for row in issues
                            ],
                            "",
                            "Return a complete replacement "
                            "SC1EndpointPairScopeAudit.",
                        ]
                    ),
                },
            ]
        )

    return messages


def validate_scope_audit(
    result: SC1EndpointPairScopeAudit,
    *,
    hypothesis: Any,
    axis_id: str,
    blueprint: IG11Blueprint,
    context: HypothesisContext,
) -> list[SC1ValidationIssue]:
    issues: list[SC1ValidationIssue] = []
    assert not blueprint.abstain
    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    assert blueprint.novel_bridge is not None

    if result.hypothesis_id != hypothesis.hypothesis_id:
        issues.append(
            SC1ValidationIssue(
                code="hypothesis_id_mismatch",
                detail=(
                    f"expected={hypothesis.hypothesis_id}; "
                    f"actual={result.hypothesis_id}"
                ),
            )
        )
    if result.axis_id != axis_id:
        issues.append(
            SC1ValidationIssue(
                code="axis_id_mismatch",
                detail=f"expected={axis_id}; actual={result.axis_id}",
            )
        )
    if (
        result.endpoint_a_anchor_statement_id
        != blueprint.endpoint_a.anchor_statement_id
    ):
        issues.append(
            SC1ValidationIssue(
                code="endpoint_a_anchor_mismatch",
                detail=result.endpoint_a_anchor_statement_id,
            )
        )
    if (
        result.endpoint_b_anchor_statement_id
        != blueprint.endpoint_b.anchor_statement_id
    ):
        issues.append(
            SC1ValidationIssue(
                code="endpoint_b_anchor_mismatch",
                detail=result.endpoint_b_anchor_statement_id,
            )
        )
    if (
        result.endpoint_a_scope.endpoint_id != "endpoint_a"
        or result.endpoint_a_scope.anchor_statement_id
        != blueprint.endpoint_a.anchor_statement_id
    ):
        issues.append(
            SC1ValidationIssue(
                code="endpoint_a_scope_identity_mismatch",
                detail=result.endpoint_a_scope.anchor_statement_id,
            )
        )
    if (
        result.endpoint_b_scope.endpoint_id != "endpoint_b"
        or result.endpoint_b_scope.anchor_statement_id
        != blueprint.endpoint_b.anchor_statement_id
    ):
        issues.append(
            SC1ValidationIssue(
                code="endpoint_b_scope_identity_mismatch",
                detail=result.endpoint_b_scope.anchor_statement_id,
            )
        )
    if result.proposed_relation != blueprint.novel_bridge.relation:
        issues.append(
            SC1ValidationIssue(
                code="proposed_relation_mismatch",
                detail="SC1 audit changed the fixed novel relation",
            )
        )

    expected_compatible = (
        result.status in _ACCEPTED_SCOPE_STATUSES
    )
    if result.scope_compatible != expected_compatible:
        issues.append(
            SC1ValidationIssue(
                code="scope_compatible_status_mismatch",
                detail=(
                    f"status={result.status}; "
                    f"scope_compatible={result.scope_compatible}"
                ),
            )
        )

    eligible = _eligible_map(context)
    selected_ids = set(
        blueprint.endpoint_a.supporting_statement_ids
        + blueprint.endpoint_b.supporting_statement_ids
    )

    all_supports = (
        list(result.compatibility_supports)
        + list(result.transfer_supports)
        + list(result.limiting_supports)
    )
    for support in all_supports:
        if support.statement_id not in selected_ids:
            issues.append(
                SC1ValidationIssue(
                    code="scope_support_outside_selected_premises",
                    detail=support.statement_id,
                )
            )
            continue
        statement = eligible.get(support.statement_id)
        if statement is None:
            issues.append(
                SC1ValidationIssue(
                    code="scope_support_not_eligible",
                    detail=support.statement_id,
                )
            )
            continue
        if support.excerpt not in statement.text:
            issues.append(
                SC1ValidationIssue(
                    code="nonverbatim_scope_support",
                    detail=(
                        f"{support.statement_id}: {support.excerpt}"
                    ),
                )
            )

    if result.status in {
        "same_system",
        "matched_system_variant",
        "shared_explicit_family",
    } and not result.compatibility_supports:
        issues.append(
            SC1ValidationIssue(
                code="compatible_status_without_compatibility_support",
                detail=result.status,
            )
        )

    if (
        result.status == "cross_system_transfer_supported"
        and not result.transfer_supports
    ):
        issues.append(
            SC1ValidationIssue(
                code="transfer_supported_without_transfer_support",
                detail=result.status,
            )
        )

    if (
        result.status
        in {
            "cross_system_transfer_unjustified",
            "insufficient_scope_information",
        }
        and not (
            result.limiting_supports
            or result.missing_scope_link
        )
    ):
        issues.append(
            SC1ValidationIssue(
                code="rejected_status_without_scope_reason",
                detail=result.status,
            )
        )

    return issues


def scope_audit_passes(
    audit: SC1EndpointPairScopeAudit,
) -> bool:
    return (
        audit.scope_compatible
        and audit.status in _ACCEPTED_SCOPE_STATUSES
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
        "hypothesis_portfolio_sc1",
        source.portfolio_id,
        [row.hypothesis_id for row in hypotheses],
    )
    return HypothesisPortfolio.model_validate(data)


def _build_candidate_axis_report(
    source: DiscoveryAxisSynthesisReport,
    *,
    candidate_portfolio: HypothesisPortfolio,
    filtered_ids: set[str],
) -> DiscoveryAxisSynthesisReport:
    data = source.model_dump(mode="json")

    lineages = []
    for lineage in source.lineages:
        if lineage.hypothesis_id in filtered_ids:
            continue
        lineages.append(
            lineage.model_dump(mode="json")
        )

    data["lineages"] = lineages
    if "accepted_hypothesis_count" in data:
        data["accepted_hypothesis_count"] = len(
            candidate_portfolio.hypotheses
        )
    if "final_portfolio_id" in data:
        data["final_portfolio_id"] = (
            candidate_portfolio.portfolio_id
        )

    portfolio_sha = _sha256_json(candidate_portfolio)
    for key in tuple(data):
        if (
            "final_portfolio" in key
            and "sha" in key.lower()
        ):
            data[key] = portfolio_sha

    data["report_id"] = _stable_id(
        "discovery_axis_synthesis_report_sc1",
        source.report_id,
        candidate_portfolio.portfolio_id,
        [
            row["hypothesis_id"]
            for row in lineages
        ],
    )
    if "report_sha256" in data:
        tmp = dict(data)
        tmp.pop("report_sha256", None)
        data["report_sha256"] = _sha256_json(tmp)

    return DiscoveryAxisSynthesisReport.model_validate(data)


def run_sc1(
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
    SC1Report,
    list[tuple[str, str, list[dict[str, str]]]],
]:
    if max_audit_repairs not in {0, 1}:
        raise ValueError("SC1 max_audit_repairs must be 0 or 1")

    if portfolio.source_context_id != context.context_id:
        raise ValueError("SC1 portfolio/context ID mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("SC1 portfolio/context SHA mismatch")
    if axis_report.final_portfolio_id != portfolio.portfolio_id:
        raise ValueError("SC1 axis report/portfolio mismatch")
    if axis_report.axis_plan_id != axis_plan.plan_id:
        raise ValueError("SC1 axis report/axis-plan mismatch")
    if planning_report.source_context_id != context.context_id:
        raise ValueError("SC1 planning/context mismatch")
    if planning_report.source_axis_plan_id != axis_plan.plan_id:
        raise ValueError("SC1 planning/axis-plan mismatch")

    lineage_by_hypothesis = {
        row.hypothesis_id: row
        for row in axis_report.lineages
    }
    axis_by_id = {
        row.axis_id: row
        for row in axis_plan.axes
    }
    audit_by_axis = _axis_audit_by_axis(planning_report)
    blueprint_by_axis = _blueprint_by_axis(planning_report)

    records: list[SC1AuditRecord] = []
    kept_hypotheses: list[Any] = []
    filtered_ids: set[str] = set()
    saved_prompts: list[
        tuple[str, str, list[dict[str, str]]]
    ] = []

    for hypothesis in portfolio.hypotheses:
        lineage = lineage_by_hypothesis.get(
            hypothesis.hypothesis_id
        )
        if lineage is None:
            raise ValueError(
                "SC1 missing lineage for "
                f"{hypothesis.hypothesis_id}"
            )
        axis = axis_by_id.get(lineage.axis_id)
        if axis is None:
            raise ValueError(
                f"SC1 missing axis {lineage.axis_id}"
            )
        axis_audit = audit_by_axis.get(axis.axis_id)
        if axis_audit is None:
            raise ValueError(
                f"SC1 missing valid axis audit {axis.axis_id}"
            )
        blueprint_item = blueprint_by_axis.get(axis.axis_id)
        if blueprint_item is None:
            raise ValueError(
                f"SC1 missing active blueprint {axis.axis_id}"
            )
        blueprint_sha, blueprint = blueprint_item

        messages = build_scope_audit_messages(
            hypothesis=hypothesis,
            axis=axis,
            blueprint=blueprint,
            axis_audit=axis_audit,
            context=context,
        )
        saved_prompts.append(
            (
                hypothesis.hypothesis_id,
                "scope_audit_initial",
                messages,
            )
        )

        call = generator.call(
            messages,
            SC1EndpointPairScopeAudit,
        )
        scope_audit = call.value
        assert isinstance(
            scope_audit,
            SC1EndpointPairScopeAudit,
        )
        attempts = 1
        repairs = 0

        issues = validate_scope_audit(
            scope_audit,
            hypothesis=hypothesis,
            axis_id=axis.axis_id,
            blueprint=blueprint,
            context=context,
        )

        if issues and max_audit_repairs:
            messages = build_scope_audit_messages(
                hypothesis=hypothesis,
                axis=axis,
                blueprint=blueprint,
                axis_audit=axis_audit,
                context=context,
                previous_audit=scope_audit,
                validation_issues=issues,
            )
            saved_prompts.append(
                (
                    hypothesis.hypothesis_id,
                    "scope_audit_repair",
                    messages,
                )
            )
            call = generator.call(
                messages,
                SC1EndpointPairScopeAudit,
            )
            scope_audit = call.value
            assert isinstance(
                scope_audit,
                SC1EndpointPairScopeAudit,
            )
            attempts += 1
            repairs = 1

            issues = validate_scope_audit(
                scope_audit,
                hypothesis=hypothesis,
                axis_id=axis.axis_id,
                blueprint=blueprint,
                context=context,
            )

        valid = not issues
        passes = (
            valid
            and scope_audit_passes(scope_audit)
        )

        if passes:
            kept_hypotheses.append(hypothesis)
            disposition = (
                "Endpoint pair passes SC1: "
                f"{scope_audit.status}."
            )
        else:
            filtered_ids.add(hypothesis.hypothesis_id)
            if not valid:
                disposition = (
                    "SC1 structured audit remained invalid after bounded "
                    "repair and therefore failed closed: "
                    + "; ".join(
                        f"{row.code}={row.detail}"
                        for row in issues
                    )
                )
            else:
                disposition = (
                    "Endpoint pair filtered by SC1: "
                    f"{scope_audit.status}. "
                    + scope_audit.explanation
                )

        records.append(
            SC1AuditRecord(
                hypothesis_id=hypothesis.hypothesis_id,
                title=hypothesis.title,
                axis_id=axis.axis_id,
                axis_label=axis.label,
                blueprint_sha256=blueprint_sha,
                audit_sha256=_sha256_json(scope_audit),
                audit=scope_audit,
                valid=valid,
                passes_gate=passes,
                generation_attempts=attempts,
                repair_count=repairs,
                validation_issues=issues,
                disposition_reason=disposition,
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

    status_counts = Counter(
        row.audit.status
        for row in records
    )

    payload = {
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_axis_plan_id": axis_plan.plan_id,
        "source_axis_report_id": axis_report.report_id,
        "source_planning_report_id": planning_report.report_id,
        "model": generator.model_name,
        "source_hypothesis_count": len(portfolio.hypotheses),
        "passed_count": len(kept_hypotheses),
        "filtered_count": len(filtered_ids),
        "valid_audit_count": sum(
            row.valid
            for row in records
        ),
        "audit_repair_count": sum(
            row.repair_count
            for row in records
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "candidate_portfolio_id": candidate_portfolio.portfolio_id,
        "candidate_axis_report_id": candidate_axis_report.report_id,
        "records": [
            row.model_dump(mode="json")
            for row in records
        ],
        "policy": SC1Policy().model_dump(mode="json"),
    }

    payload["report_id"] = _stable_id(
        "sc1_report",
        portfolio.portfolio_id,
        axis_report.report_id,
        planning_report.report_id,
        candidate_portfolio.portfolio_id,
        _sha256_json(payload["records"]),
    )

    report = SC1Report(
        **payload,
        report_sha256=_sha256_json(payload),
    )

    return (
        candidate_portfolio,
        candidate_axis_report,
        report,
        saved_prompts,
    )
