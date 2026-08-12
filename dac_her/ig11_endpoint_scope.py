from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from dac_her.hypothesis_llm import (
    HypothesisDraftBackend,
    HypothesisDraftGeneration,
)
from dac_her.hypothesis_prompt import HypothesisPrompt
from dac_her.llm_telemetry import run_instructor_structured_call
from dac_her.ig1_grounded_bridge import (
    IG1ConformanceIssue,
    IG1DiscriminativeTest,
    IG1NovelBridge,
    draft_conformance_issues,
)


IG11_AUDIT_VERSION = "ig1.1-axis-evidence-audit-v1"
IG11_BLUEPRINT_VERSION = "ig1.1-minimal-burden-scope-v1"
IG11_PROMPT_VERSION = "hypothesis-maker-ig1.1-v2.9.1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AxisSupport = Literal[
    "direct_axis_grounding",
    "partial_axis_grounding",
    "adjacent_context",
    "scope_only",
    "none",
]

EndpointRole = Literal[
    "axis_variable",
    "structural_context",
    "mechanistic_context",
    "stability_or_outcome",
    "reaction_or_activity_outcome",
    "scope_only",
    "not_endpoint_candidate",
]

ScopeBreadth = Literal[
    "specific_system",
    "multi_system",
    "cross_paper_synthesis",
    "generic_within_premise",
    "unclear",
]

EntityKind = Literal[
    "concrete_material_or_system",
    "reaction_species_or_product",
    "descriptor_or_mechanism",
    "generic_concept",
]

EntityGroundingStatus = Literal[
    "grounded",
    "partial",
    "ungrounded",
]


BridgeTerminalRole = Literal[
    "axis_grounding_terminal",
    "bridge_terminal",
    "both",
    "not_terminal",
]

BridgeNoveltyStatus = Literal[
    "already_directly_grounded",
    "already_synthesis_grounded",
    "genuinely_unestablished_relation",
    "scope_transfer_required",
    "uncertain",
]


class IG11StatementAxisAudit(StrictModel):
    statement_id: str
    axis_support: AxisSupport
    endpoint_role: EndpointRole

    grounding_excerpt: str | None = None
    scope_basis_excerpt: str | None = None

    scope_breadth: ScopeBreadth
    scope_summary: str = Field(min_length=1)

    endpoint_candidate: bool
    bridge_terminal_role: BridgeTerminalRole = "not_terminal"
    bridge_terminal_excerpt: str | None = None
    reason: str = Field(min_length=1)


class IG11AxisEntityAudit(StrictModel):
    entity_text: str = Field(min_length=1)
    entity_kind: EntityKind
    grounding_status: EntityGroundingStatus
    grounding_statement_ids: list[str] = Field(default_factory=list)

    entity_specific_claim_required_for_axis_fidelity: bool
    reason: str = Field(min_length=1)


class IG11AxisEvidenceAudit(StrictModel):
    schema_version: Literal["ig1.1-axis-evidence-audit-v1"] = (
        "ig1.1-axis-evidence-audit-v1"
    )
    axis_id: str
    statement_reviews: list[IG11StatementAxisAudit] = Field(
        min_length=1
    )
    axis_entities: list[IG11AxisEntityAudit] = Field(default_factory=list)

    direct_axis_statement_ids: list[str] = Field(default_factory=list)
    endpoint_candidate_statement_ids: list[str] = Field(default_factory=list)

    audit_summary: str = Field(min_length=1)


class IG11EndpointScope(StrictModel):
    scope_basis_excerpt: str = Field(min_length=1)
    scope_breadth: ScopeBreadth
    scope_summary: str = Field(min_length=1)


class IG11GroundedEndpoint(StrictModel):
    endpoint_id: Literal["endpoint_a", "endpoint_b"]
    anchor_statement_id: str
    grounded_excerpt: str = Field(min_length=1)
    supporting_statement_ids: list[str] = Field(min_length=1)
    scientific_role: str = Field(min_length=1)
    scope: IG11EndpointScope

    @model_validator(mode="after")
    def _support_integrity(self) -> "IG11GroundedEndpoint":
        if self.anchor_statement_id not in self.supporting_statement_ids:
            raise ValueError(
                "anchor_statement_id must appear in supporting_statement_ids"
            )
        if len(self.supporting_statement_ids) != len(
            set(self.supporting_statement_ids)
        ):
            raise ValueError("duplicate endpoint supporting_statement_ids")
        return self


class IG11BridgeScopeEnvelope(StrictModel):
    scope_guard_phrase: str = Field(min_length=1)
    basis_statement_ids: list[str] = Field(min_length=1)

    system_or_material_scope: str = Field(min_length=1)
    entity_or_pair_scope: str = Field(min_length=1)
    structural_or_coordination_scope: str = Field(min_length=1)
    observable_or_outcome_scope: str = Field(min_length=1)

    unsupported_scope_transfer_required: Literal[False] = False


class IG11NoveltyBurdenAudit(StrictModel):
    semantic_novel_relation_count: Literal[1] = 1
    direct_axis_grounding_used_when_available: bool
    additional_unmodeled_mechanisms_introduced: Literal[False] = False
    downstream_design_rule_introduced: Literal[False] = False
    burden_summary: str = Field(min_length=1)




class IG12BridgeNonRedundancyAudit(StrictModel):
    schema_version: Literal[
        "ig1.2-bridge-nonredundancy-audit-v1"
    ] = "ig1.2-bridge-nonredundancy-audit-v1"

    axis_id: str
    bridge_relation: str
    selected_premise_statement_ids: list[str] = Field(min_length=1)

    status: BridgeNoveltyStatus
    directly_grounding_statement_ids: list[str] = Field(default_factory=list)
    synthesis_grounding_statement_ids: list[str] = Field(default_factory=list)

    scope_compatible: bool
    explanation: str = Field(min_length=1)
    missing_relation_if_unestablished: str | None = None


class IG12BridgeAuditRecord(StrictModel):
    axis_id: str
    blueprint_sha256: str
    audit_sha256: str
    audit: IG12BridgeNonRedundancyAudit
    valid: bool
    validation_issues: list["IG11ValidationIssue"] = Field(
        default_factory=list
    )
    generation_attempts: int = 1


class IG11Blueprint(StrictModel):
    schema_version: Literal["ig1.1-minimal-burden-scope-v1"] = (
        "ig1.1-minimal-burden-scope-v1"
    )

    axis_id: str
    abstain: bool = False
    abstention_reason: str | None = None

    endpoint_a: IG11GroundedEndpoint | None = None
    endpoint_b: IG11GroundedEndpoint | None = None
    scope_envelope: IG11BridgeScopeEnvelope | None = None

    novel_bridge: IG1NovelBridge | None = None
    discriminative_test: IG1DiscriminativeTest | None = None
    novelty_burden: IG11NoveltyBurdenAudit | None = None

    scope_conditions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "IG11Blueprint":
        scientific = (
            self.endpoint_a,
            self.endpoint_b,
            self.scope_envelope,
            self.novel_bridge,
            self.discriminative_test,
            self.novelty_burden,
        )
        if self.abstain:
            if not (self.abstention_reason or "").strip():
                raise ValueError(
                    "abstention_reason required when IG1.1 abstains"
                )
            if any(row is not None for row in scientific):
                raise ValueError(
                    "abstaining IG1.1 blueprint must not contain a plan"
                )
            return self

        if self.abstention_reason is not None:
            raise ValueError(
                "abstention_reason must be null for active blueprint"
            )
        if any(row is None for row in scientific):
            raise ValueError(
                "active IG1.1 blueprint requires endpoints, scope envelope, "
                "one novel bridge, one test, and burden audit"
            )

        assert self.endpoint_a is not None
        assert self.endpoint_b is not None
        assert self.novel_bridge is not None
        assert self.scope_envelope is not None

        if self.endpoint_a.endpoint_id != "endpoint_a":
            raise ValueError("endpoint_a must use endpoint_id=endpoint_a")
        if self.endpoint_b.endpoint_id != "endpoint_b":
            raise ValueError("endpoint_b must use endpoint_id=endpoint_b")
        if (
            self.endpoint_a.anchor_statement_id
            == self.endpoint_b.anchor_statement_id
            and self.endpoint_a.grounded_excerpt
            == self.endpoint_b.grounded_excerpt
        ):
            raise ValueError(
                "IG1.1 endpoints must represent distinct grounded claims"
            )
        if (
            self.novel_bridge.subject_endpoint_id
            == self.novel_bridge.object_endpoint_id
        ):
            raise ValueError(
                "IG1.1 novel bridge must connect distinct endpoints"
            )
        if (
            self.scope_envelope.scope_guard_phrase
            not in self.novel_bridge.relation
        ):
            raise ValueError(
                "scope_guard_phrase must appear verbatim in novel relation"
            )
        return self


class IG11ValidationIssue(StrictModel):
    code: str
    detail: str


class IG11EvidenceAuditRecord(StrictModel):
    axis_id: str
    source_prompt_sha256: str
    audit_sha256: str
    audit: IG11AxisEvidenceAudit
    valid: bool
    generation_attempts: int = 1
    repair_count: int = 0
    validation_issues: list[IG11ValidationIssue] = Field(
        default_factory=list
    )


class IG11BlueprintRecord(StrictModel):
    axis_id: str
    source_prompt_sha256: str
    evidence_audit_sha256: str
    blueprint_sha256: str
    blueprint: IG11Blueprint

    generation_attempts: int
    nonredundancy_replan_count: int = 0
    valid: bool
    validation_issues: list[IG11ValidationIssue] = Field(
        default_factory=list
    )

    final_generation_count: int = 0
    conformance_repair_count: int = 0


class IG11PlanningReport(StrictModel):
    schema_version: Literal["ig1.1-planning-report-v1"] = (
        "ig1.1-planning-report-v1"
    )
    report_id: str
    report_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_axis_plan_id: str

    audit_model: str
    blueprint_model: str

    axis_count: int
    audit_valid_count: int
    audit_repair_count: int
    bridge_audit_count: int
    bridge_nonredundancy_replan_count: int
    active_blueprint_count: int
    abstained_blueprint_count: int
    invalid_blueprint_count: int
    conformance_repair_count: int

    audit_records: list[IG11EvidenceAuditRecord] = Field(
        default_factory=list
    )
    bridge_audit_records: list[IG12BridgeAuditRecord] = Field(
        default_factory=list
    )
    blueprint_records: list[IG11BlueprintRecord] = Field(
        default_factory=list
    )


class IG11ConformanceCard(StrictModel):
    hypothesis_id: str
    axis_id: str
    blueprint_sha256: str
    passes: bool

    exact_hypothesis_equals_novel_relation: bool
    expected_premise_statement_ids: list[str] = Field(default_factory=list)
    actual_premise_statement_ids: list[str] = Field(default_factory=list)

    direct_axis_statement_ids: list[str] = Field(default_factory=list)
    direct_axis_grounding_used: bool

    scope_guard_phrase: str
    scope_guard_in_hypothesis: bool
    scope_guard_in_bridge: bool

    issues: list[IG1ConformanceIssue] = Field(default_factory=list)


class IG11ConformanceReport(StrictModel):
    schema_version: Literal["ig1.1-conformance-report-v1"] = (
        "ig1.1-conformance-report-v1"
    )
    report_id: str
    report_sha256: str

    source_portfolio_id: str
    source_axis_report_id: str
    source_planning_report_id: str

    hypothesis_count: int
    passing_count: int
    failing_count: int
    issue_counts: dict[str, int] = Field(default_factory=dict)
    cards: list[IG11ConformanceCard] = Field(default_factory=list)


@dataclass(frozen=True)
class _StructuredCall:
    value: BaseModel
    elapsed_seconds: float


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
    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _axis_metadata(
    prompt: HypothesisPrompt,
) -> dict[str, str]:
    def one(pattern: str, name: str) -> str:
        matches = re.findall(
            pattern,
            prompt.user_prompt,
            flags=re.MULTILINE,
        )
        if len(matches) != 1:
            raise ValueError(
                f"IG1.1 expected one {name}; found {len(matches)}"
            )
        return matches[0].strip()

    return {
        "axis_id": one(r"^axis_id:\s*(.+)$", "axis_id"),
        "label": one(r"^label:\s*(.+)$", "label"),
        "proposed_semantics": one(
            r"^proposed_semantics:\s*(.+)$",
            "proposed_semantics",
        ),
    }


def _eligible_statements(
    context: HypothesisContext,
) -> list[Any]:
    return [
        row
        for row in context.evidence_statements
        if row.eligible_as_premise
    ]


def _eligible_map(
    context: HypothesisContext,
) -> dict[str, Any]:
    return {
        row.statement_id: row
        for row in _eligible_statements(context)
    }


def _relation_chain_markers(relation: str) -> list[str]:
    text = relation.lower()
    markers = [
        " and then ",
        " followed by ",
        " which then ",
        " subsequently ",
        " thereby causing ",
        " leading in turn to ",
    ]
    return [
        marker.strip()
        for marker in markers
        if marker in text
    ]


def build_axis_audit_messages(
    prompt: HypothesisPrompt,
    context: HypothesisContext,
) -> list[dict[str, str]]:
    axis = _axis_metadata(prompt)
    evidence = [
        {
            "statement_id": row.statement_id,
            "text": row.text,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
            "premise_restrictions": row.premise_restrictions,
            "requires_verification": row.requires_verification,
        }
        for row in _eligible_statements(context)
    ]

    system = """You are the IG1.1 axis-evidence audit stage.

Audit EVERY eligible positive premise before endpoint selection.

Primary purpose:
1. identify whether the assigned discovery axis already has grounded evidence
   for its scientific variable/relation;
2. identify which statements are useful grounded endpoints;
3. identify which statements can serve as BRIDGE TERMINALS even when they do
   not directly ground the axis variable itself;
4. expose each statement's evidence scope;
5. identify concrete named axis entities/systems whose use would require
   unsupported extrapolation.

AXIS SUPPORT LABELS
- direct_axis_grounding:
  the premise directly grounds a scientific variable, observable, descriptor,
  relation endpoint, or mechanism explicitly central to the axis.
- partial_axis_grounding:
  the premise grounds only part of the axis concept.
- adjacent_context:
  scientifically relevant but does not ground the axis variable/relation.
- scope_only:
  useful mainly to delimit where another claim applies.
- none:
  no meaningful axis-grounding role.

IMPORTANT:
A premise can directly ground an axis VARIABLE without grounding the novel
relation we may later hypothesize. Example: evidence for hydrogen adsorption
energetics can ground that endpoint even if it does not ground a proposed
stability–adsorption relation.

BRIDGE-TERMINAL POLICY
A useful endpoint does NOT have to directly ground the axis variable.
A premise may be a bridge_terminal when it provides one scientifically
grounded end of a plausible one-edge extension inspired by the axis.

Examples:
- a coordination-context premise can be a bridge terminal for a proposed
  bonding-balance -> retention relation;
- a stability premise can be a bridge terminal for a proposed
  adsorption/stability relation;
- a premise that merely restates the same relation we would call novel should
  not be used to manufacture a zero-novelty hypothesis.

Set bridge_terminal_role to:
- axis_grounding_terminal: useful mainly because it directly grounds the axis;
- bridge_terminal: useful as an endpoint even without direct axis grounding;
- both: both roles apply;
- not_terminal: not a useful grounded terminal.

When bridge_terminal_role != not_terminal, bridge_terminal_excerpt MUST be a
short contiguous VERBATIM substring of that premise showing the grounded
terminal claim.

For every review with axis_support != none, grounding_excerpt must be a short
contiguous VERBATIM substring of that exact premise. scope_basis_excerpt must
also be a contiguous VERBATIM substring whenever possible; if the statement is
too synthetic for a clean scope excerpt, use null.

AXIS ENTITY TAXONOMY POLICY
Audit scientifically meaningful named entities appearing in the assigned axis,
but classify them by scientific role.

Use `concrete_material_or_system` ONLY for system-defining identities such as:
- a specific metal pair (for example Co/Mo or Pt/Ru);
- a specific catalyst, material, composition, or support system;
- another concrete modeled/experimental material identity whose unsupported
  substitution would create entity-specific extrapolation.

Use `reaction_species_or_product` for reaction participants, intermediates,
reactants, or products such as H, H2, H+, H2O, OH-, O2, CO2, and analogous
chemical species. These are NOT catalyst/material/system identities.

Use `descriptor_or_mechanism` for descriptors/mechanisms such as
bonding–antibonding balance, d-band position, charge redistribution, adsorption
free energy, or electronic coupling.

Use `generic_concept` for other non-system concepts. Do not classify ordinary
outcome labels such as "stability" as concrete materials.

If an axis would cease to be that specific MATERIAL/SYSTEM axis when a
`concrete_material_or_system` entity is removed, set
entity_specific_claim_required_for_axis_fidelity=true.

For `reaction_species_or_product`, normally set
entity_specific_claim_required_for_axis_fidelity=false. A scientific relation
involving that species may still be unsupported and may still fail ordinary
endpoint-grounding, scope, non-redundancy, or downstream coverage checks; the
species itself simply must not trigger the concrete-material identity gate.

External knowledge is forbidden. Discovery-axis content is inspiration, not
evidence.
"""
    user = f"""ASSIGNED AXIS
axis_id: {axis["axis_id"]}
label: {axis["label"]}
proposed_semantics: {axis["proposed_semantics"]}

ELIGIBLE POSITIVE PREMISES
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Return one statement_review for EVERY eligible premise, exactly once.
direct_axis_statement_ids must exactly list the statement IDs you classified
as direct_axis_grounding.
endpoint_candidate_statement_ids must exactly list all reviews where
endpoint_candidate=true.

Endpoint candidacy must include scientifically useful bridge terminals, not
only direct axis-grounding statements. A statement may therefore have
axis_support=adjacent_context while endpoint_candidate=true if it grounds one
end of a bounded one-edge extension.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_axis_audit(
    audit: IG11AxisEvidenceAudit,
    *,
    context: HypothesisContext,
    expected_axis_id: str,
) -> list[IG11ValidationIssue]:
    issues: list[IG11ValidationIssue] = []
    eligible = _eligible_map(context)
    expected_ids = set(eligible)

    if audit.axis_id != expected_axis_id:
        issues.append(
            IG11ValidationIssue(
                code="axis_id_mismatch",
                detail=(
                    f"expected={expected_axis_id}; "
                    f"actual={audit.axis_id}"
                ),
            )
        )

    actual_ids = [row.statement_id for row in audit.statement_reviews]
    if len(actual_ids) != len(set(actual_ids)):
        issues.append(
            IG11ValidationIssue(
                code="duplicate_statement_review",
                detail="statement review IDs are not unique",
            )
        )
    if set(actual_ids) != expected_ids:
        issues.append(
            IG11ValidationIssue(
                code="statement_review_set_mismatch",
                detail=(
                    f"expected={sorted(expected_ids)}; "
                    f"actual={sorted(set(actual_ids))}"
                ),
            )
        )

    direct_from_reviews: set[str] = set()
    candidates_from_reviews: set[str] = set()

    for row in audit.statement_reviews:
        statement = eligible.get(row.statement_id)
        if statement is None:
            continue

        if row.axis_support == "direct_axis_grounding":
            direct_from_reviews.add(row.statement_id)
        if row.endpoint_candidate:
            candidates_from_reviews.add(row.statement_id)

        needs_grounding_excerpt = (
            row.axis_support
            in {
                "direct_axis_grounding",
                "partial_axis_grounding",
            }
            or row.endpoint_candidate
        )
        if needs_grounding_excerpt:
            if not row.grounding_excerpt:
                issues.append(
                    IG11ValidationIssue(
                        code="missing_grounding_excerpt",
                        detail=row.statement_id,
                    )
                )
            elif row.grounding_excerpt not in statement.text:
                issues.append(
                    IG11ValidationIssue(
                        code="nonverbatim_grounding_excerpt",
                        detail=row.statement_id,
                    )
                )
        elif (
            row.grounding_excerpt is not None
            and row.grounding_excerpt not in statement.text
        ):
            # Optional excerpts are still provenance-audited when supplied.
            issues.append(
                IG11ValidationIssue(
                    code="nonverbatim_optional_grounding_excerpt",
                    detail=row.statement_id,
                )
            )

        if row.bridge_terminal_role != "not_terminal":
            if not row.bridge_terminal_excerpt:
                issues.append(
                    IG11ValidationIssue(
                        code="missing_bridge_terminal_excerpt",
                        detail=row.statement_id,
                    )
                )
            elif row.bridge_terminal_excerpt not in statement.text:
                issues.append(
                    IG11ValidationIssue(
                        code="nonverbatim_bridge_terminal_excerpt",
                        detail=row.statement_id,
                    )
                )
        elif (
            row.bridge_terminal_excerpt is not None
            and row.bridge_terminal_excerpt not in statement.text
        ):
            issues.append(
                IG11ValidationIssue(
                    code="nonverbatim_optional_bridge_terminal_excerpt",
                    detail=row.statement_id,
                )
            )

        if (
            row.scope_basis_excerpt is not None
            and row.scope_basis_excerpt not in statement.text
        ):
            issues.append(
                IG11ValidationIssue(
                    code="nonverbatim_scope_excerpt",
                    detail=row.statement_id,
                )
            )

    if set(audit.direct_axis_statement_ids) != direct_from_reviews:
        issues.append(
            IG11ValidationIssue(
                code="direct_axis_id_summary_mismatch",
                detail=(
                    f"reviews={sorted(direct_from_reviews)}; "
                    f"summary={sorted(audit.direct_axis_statement_ids)}"
                ),
            )
        )

    if (
        set(audit.endpoint_candidate_statement_ids)
        != candidates_from_reviews
    ):
        issues.append(
            IG11ValidationIssue(
                code="endpoint_candidate_summary_mismatch",
                detail=(
                    f"reviews={sorted(candidates_from_reviews)}; "
                    f"summary={sorted(audit.endpoint_candidate_statement_ids)}"
                ),
            )
        )

    for entity in audit.axis_entities:
        bad = (
            set(entity.grounding_statement_ids)
            - expected_ids
        )
        if bad:
            issues.append(
                IG11ValidationIssue(
                    code="entity_grounding_unknown_statement",
                    detail=(
                        f"{entity.entity_text}: {sorted(bad)}"
                    ),
                )
            )
        if (
            entity.grounding_status == "ungrounded"
            and entity.grounding_statement_ids
        ):
            issues.append(
                IG11ValidationIssue(
                    code="ungrounded_entity_has_grounding_ids",
                    detail=entity.entity_text,
                )
            )

    return issues


def build_axis_audit_repair_messages(
    prompt: HypothesisPrompt,
    context: HypothesisContext,
    previous_audit: IG11AxisEvidenceAudit,
    issues: list[IG11ValidationIssue],
) -> list[dict[str, str]]:
    messages = build_axis_audit_messages(
        prompt,
        context,
    )
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
                        "IG1.1a AXIS-EVIDENCE AUDIT REPAIR",
                        "=================================",
                        "The previous audit failed deterministic "
                        "self-consistency/provenance validation.",
                        "",
                        "Repair ONLY the listed validation issues. "
                        "Do not change a scientifically supported label "
                        "merely to make validation pass.",
                        "",
                        "Important repair rules:",
                        "- If grounding_status='ungrounded', "
                        "grounding_statement_ids MUST be empty.",
                        "- Classify reaction participants/products such as "
                        "H2, H+, H2O, or OH- as "
                        "reaction_species_or_product, not "
                        "concrete_material_or_system.",
                        "- If grounding_statement_ids contain real positive "
                        "support, reconsider whether grounding_status should "
                        "be grounded or partial based only on the supplied "
                        "premise text.",
                        "- direct_axis_grounding, partial_axis_grounding, "
                        "and every endpoint_candidate=true review require a "
                        "verbatim grounding_excerpt.",
                        "- adjacent_context/scope_only reviews with "
                        "endpoint_candidate=false may use "
                        "grounding_excerpt=null.",
                        "- Any supplied grounding_excerpt or "
                        "scope_basis_excerpt must remain a contiguous "
                        "verbatim substring of that exact premise.",
                        "- Preserve exactly one review per eligible premise.",
                        "- Recompute direct_axis_statement_ids and "
                        "endpoint_candidate_statement_ids so they exactly "
                        "match the repaired reviews.",
                        "- External knowledge remains forbidden.",
                        "",
                        "Validation issues:",
                        *[
                            f"- {row.code}: {row.detail}"
                            for row in issues
                        ],
                        "",
                        "Return a complete replacement "
                        "IG11AxisEvidenceAudit.",
                    ]
                ),
            },
        ]
    )
    return messages


def build_blueprint_messages(
    prompt: HypothesisPrompt,
    context: HypothesisContext,
    audit: IG11AxisEvidenceAudit,
) -> list[dict[str, str]]:
    axis = _axis_metadata(prompt)
    evidence = [
        {
            "statement_id": row.statement_id,
            "text": row.text,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
            "premise_restrictions": row.premise_restrictions,
        }
        for row in _eligible_statements(context)
    ]

    system = """You are the IG1.1 minimal-novelty-burden endpoint planner.

Produce an IG11Blueprint, not the final hypothesis.

LEXICOGRAPHIC PLANNING OBJECTIVE
Apply these priorities in order:

1. NO UNSUPPORTED SCOPE TRANSFER.
   Select endpoints whose evidence scopes permit a bounded relation. The final
   relation must not silently generalize a specific-system observation to all
   systems.

2. USE ALREADY-GROUNDED AXIS CONTENT WHEN IT HELPS FORM A REAL NOVEL EDGE.
   If direct_axis_grounding evidence exists, prefer using it as one endpoint
   when it can participate in a genuinely new one-edge extension. Do NOT use
   it merely to restate a relation already contained in that premise.

3. USE BRIDGE-TERMINAL ENDPOINTS.
   Endpoint candidates may come from adjacent_context premises when the audit
   marks them as bridge terminals. The two endpoints should ground the two
   ends of the proposed relation; the novel descriptor/mechanism itself need
   not already be a grounded endpoint.

4. MINIMIZE NOVEL SCIENTIFIC BURDEN WITHOUT COLLAPSING NOVELTY TO ZERO.
   Choose an endpoint pair that supports exactly ONE genuinely unestablished
   relation. A relation already directly or jointly established by the selected
   endpoint premises is NOT a valid hypothesis bridge.

5. PRESERVE AXIS FIDELITY WITHOUT PROMOTING INSPIRATION TO EVIDENCE.
   A descriptor/mechanism may be the one proposed relation. A concrete named
   material/system/entity may not be used as if grounded when the audit says it
   is ungrounded.

6. DO NOT OPTIMIZE PREMISE DIVERSITY.
   Reusing a premise is acceptable when it is scientifically the best endpoint.

STRICT CONCRETE-MATERIAL/SYSTEM RULE
If an entity classified as `concrete_material_or_system` is ungrounded and
entity_specific_claim_required_for_axis_fidelity=true, ABSTAIN. Do not replace
that missing material/system evidence with plausibility.

Do NOT apply this strict identity gate to `reaction_species_or_product`. A
reaction species may still participate in an unsupported relation, but that
must be handled by ordinary endpoint grounding, scope, bridge non-redundancy,
and downstream scientific coverage rather than by material/system identity
abstention.

ENDPOINT RULES
- exactly two distinct grounded endpoint CLAIMS;
- each endpoint anchor must be an eligible premise and an audited endpoint
  candidate;
- endpoint candidates may be direct axis-grounding endpoints OR bridge-terminal
  endpoints;
- grounded_excerpt and scope.scope_basis_excerpt must be VERBATIM substrings of
  the anchor statement;
- supporting_statement_ids must be the smallest sufficient positive-premise set;
- using two claims from the same premise is allowed only when the proposed
  relation between them is not already directly/synthesis grounded by that
  premise.

SCOPE ENVELOPE
- define the narrowest scientifically defensible shared scope;
- unsupported_scope_transfer_required MUST be false;
- basis_statement_ids must be drawn only from the selected endpoint supports;
- create a concise scope_guard_phrase and include it VERBATIM inside the one
  novel relation;
- the scope guard must prevent broadening a specific-system endpoint into a
  universal/general claim.

ONE NOVEL RELATION
- exactly one scientific relation;
- it must be genuinely UNESTABLISHED by the selected positive premises;
- do not select a relation that the selected premises already directly state or
  jointly establish without a new scientific edge;
- no "A -> X -> Y -> B" chain;
- do not append a second mechanism, design rule, optimization recommendation,
  or universal-rule rejection;
- if a design rule itself is the single relation, use bridge_kind=design_rule;
- reported_fact=false and evidence_boundary_acknowledged=true.

ONE TEST
Return exactly one directly discriminative prediction/falsifier target.

ABSTAIN whenever these constraints cannot be met from the supplied evidence.
External knowledge is forbidden.
"""
    user = f"""ASSIGNED AXIS
axis_id: {axis["axis_id"]}
label: {axis["label"]}
proposed_semantics: {axis["proposed_semantics"]}

VALIDATED AXIS-EVIDENCE AUDIT
{audit.model_dump_json(indent=2)}

ELIGIBLE POSITIVE PREMISES
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Build the minimal-novelty-burden blueprint.

If direct_axis_statement_ids is nonempty, at least one endpoint must use one of
those IDs.

If the audit identifies an ungrounded `concrete_material_or_system` entity that
is required for material/system axis fidelity, return abstain=true. Do not use
this identity rule for `reaction_species_or_product`.

The scope_guard_phrase must occur exactly inside novel_bridge.relation.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def blueprint_premise_ids(
    blueprint: IG11Blueprint,
) -> list[str]:
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


def validate_blueprint(
    blueprint: IG11Blueprint,
    *,
    audit: IG11AxisEvidenceAudit,
    context: HypothesisContext,
    expected_axis_id: str,
) -> list[IG11ValidationIssue]:
    issues: list[IG11ValidationIssue] = []
    eligible = _eligible_map(context)

    if blueprint.axis_id != expected_axis_id:
        issues.append(
            IG11ValidationIssue(
                code="axis_id_mismatch",
                detail=(
                    f"expected={expected_axis_id}; "
                    f"actual={blueprint.axis_id}"
                ),
            )
        )

    if blueprint.abstain:
        return issues

    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    assert blueprint.scope_envelope is not None
    assert blueprint.novel_bridge is not None
    assert blueprint.novelty_burden is not None

    selected_ids = set(blueprint_premise_ids(blueprint))

    for endpoint in (blueprint.endpoint_a, blueprint.endpoint_b):
        anchor = eligible.get(endpoint.anchor_statement_id)
        if anchor is None:
            issues.append(
                IG11ValidationIssue(
                    code="ineligible_anchor_statement",
                    detail=endpoint.anchor_statement_id,
                )
            )
            continue

        for sid in endpoint.supporting_statement_ids:
            if sid not in eligible:
                issues.append(
                    IG11ValidationIssue(
                        code="ineligible_supporting_statement",
                        detail=sid,
                    )
                )

        if endpoint.grounded_excerpt not in anchor.text:
            issues.append(
                IG11ValidationIssue(
                    code="anchor_excerpt_not_verbatim",
                    detail=endpoint.endpoint_id,
                )
            )

        if endpoint.scope.scope_basis_excerpt not in anchor.text:
            issues.append(
                IG11ValidationIssue(
                    code="scope_excerpt_not_verbatim",
                    detail=endpoint.endpoint_id,
                )
            )

        if (
            endpoint.anchor_statement_id
            not in audit.endpoint_candidate_statement_ids
        ):
            issues.append(
                IG11ValidationIssue(
                    code="endpoint_not_audited_candidate",
                    detail=endpoint.anchor_statement_id,
                )
            )

    direct_ids = set(audit.direct_axis_statement_ids)
    direct_used = bool(selected_ids & direct_ids)
    if direct_ids and not direct_used:
        issues.append(
            IG11ValidationIssue(
                code="direct_axis_evidence_omitted",
                detail=(
                    f"direct={sorted(direct_ids)}; "
                    f"selected={sorted(selected_ids)}"
                ),
            )
        )

    if (
        blueprint.novelty_burden.direct_axis_grounding_used_when_available
        != (not direct_ids or direct_used)
    ):
        issues.append(
            IG11ValidationIssue(
                code="direct_axis_usage_self_audit_mismatch",
                detail=(
                    "novelty_burden direct-axis flag does not match "
                    "deterministic selected-premise check"
                ),
            )
        )

    bad_scope_ids = (
        set(blueprint.scope_envelope.basis_statement_ids)
        - selected_ids
    )
    if bad_scope_ids:
        issues.append(
            IG11ValidationIssue(
                code="scope_basis_outside_endpoint_support",
                detail=str(sorted(bad_scope_ids)),
            )
        )

    if (
        blueprint.scope_envelope.scope_guard_phrase
        not in blueprint.novel_bridge.relation
    ):
        issues.append(
            IG11ValidationIssue(
                code="scope_guard_missing_from_relation",
                detail=blueprint.scope_envelope.scope_guard_phrase,
            )
        )

    markers = _relation_chain_markers(
        blueprint.novel_bridge.relation
    )
    if markers:
        issues.append(
            IG11ValidationIssue(
                code="multi_hop_relation_marker",
                detail=", ".join(markers),
            )
        )

    relation_lower = blueprint.novel_bridge.relation.lower()
    for entity in audit.axis_entities:
        if (
            entity.entity_kind
            == "concrete_material_or_system"
            and entity.grounding_status == "ungrounded"
            and entity.entity_text.lower() in relation_lower
        ):
            issues.append(
                IG11ValidationIssue(
                    code="ungrounded_concrete_axis_entity_in_relation",
                    detail=entity.entity_text,
                )
            )

        if (
            entity.entity_kind
            == "concrete_material_or_system"
            and entity.grounding_status == "ungrounded"
            and entity.entity_specific_claim_required_for_axis_fidelity
        ):
            issues.append(
                IG11ValidationIssue(
                    code="required_axis_entity_ungrounded",
                    detail=entity.entity_text,
                )
            )

    return issues


def build_bridge_nonredundancy_messages(
    prompt: HypothesisPrompt,
    context: HypothesisContext,
    audit: IG11AxisEvidenceAudit,
    blueprint: IG11Blueprint,
) -> list[dict[str, str]]:
    if blueprint.abstain:
        raise ValueError(
            "cannot audit bridge non-redundancy for abstaining blueprint"
        )
    assert blueprint.novel_bridge is not None
    selected_ids = blueprint_premise_ids(blueprint)
    eligible = _eligible_map(context)
    premises = [
        {
            "statement_id": sid,
            "text": eligible[sid].text,
            "epistemic_role": eligible[sid].epistemic_role,
            "claim_kind": eligible[sid].claim_kind,
            "paper_ids": eligible[sid].paper_ids,
        }
        for sid in selected_ids
    ]

    system = """You are the IG1.2 novel-bridge non-redundancy auditor.

Decide whether the ONE proposed bridge is actually novel relative to the
SELECTED positive premises.

Use ONLY the selected premises. External knowledge and discovery-axis
inspiration are NOT evidence.

Statuses:
- already_directly_grounded:
  one selected premise directly establishes the bridge relation as written
  with compatible scope.
- already_synthesis_grounded:
  multiple selected premises jointly establish the bridge relation without
  introducing a genuinely new scientific relation.
- genuinely_unestablished_relation:
  selected premises ground the endpoints/context, but the proposed relation
  itself is not established. This is the desired hypothesis case.
- scope_transfer_required:
  the relation could only be supported/tested by transferring a premise beyond
  its demonstrated system/entity/condition scope.
- uncertain:
  cannot reliably decide from supplied text.

Critical rules:
- Do not call a relation novel merely because wording differs.
- Do not call it grounded merely because both endpoints appear.
- A real hypothesis requires a NEW relation between grounded terminals.
- If one premise already states endpoint A -> endpoint B in substance, the
  bridge is not novel even if the blueprint paraphrases it.
- If the bridge adds a descriptor/mechanism that is not established, it may be
  genuinely_unestablished_relation provided it does not require unsupported
  scope transfer.
- Keep scope compatibility separate from novelty.
"""
    user = f"""AXIS
{_axis_metadata(prompt)}

SELECTED PREMISES
{json.dumps(premises, ensure_ascii=False, indent=2)}

BLUEPRINT
{blueprint.model_dump_json(indent=2)}

Return exactly one IG12BridgeNonRedundancyAudit.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_bridge_nonredundancy_audit(
    result: IG12BridgeNonRedundancyAudit,
    *,
    blueprint: IG11Blueprint,
    expected_axis_id: str,
) -> list[IG11ValidationIssue]:
    issues: list[IG11ValidationIssue] = []
    if blueprint.abstain:
        issues.append(
            IG11ValidationIssue(
                code="bridge_audit_on_abstention",
                detail=expected_axis_id,
            )
        )
        return issues

    assert blueprint.novel_bridge is not None
    selected = set(blueprint_premise_ids(blueprint))

    if result.axis_id != expected_axis_id:
        issues.append(
            IG11ValidationIssue(
                code="bridge_audit_axis_mismatch",
                detail=f"expected={expected_axis_id}; actual={result.axis_id}",
            )
        )
    if result.bridge_relation != blueprint.novel_bridge.relation:
        issues.append(
            IG11ValidationIssue(
                code="bridge_relation_mismatch",
                detail="non-redundancy audit changed the fixed relation",
            )
        )
    if set(result.selected_premise_statement_ids) != selected:
        issues.append(
            IG11ValidationIssue(
                code="bridge_audit_premise_set_mismatch",
                detail=(
                    f"expected={sorted(selected)}; "
                    f"actual={sorted(result.selected_premise_statement_ids)}"
                ),
            )
        )

    referenced = (
        set(result.directly_grounding_statement_ids)
        | set(result.synthesis_grounding_statement_ids)
    )
    if not referenced <= selected:
        issues.append(
            IG11ValidationIssue(
                code="bridge_audit_unknown_support_id",
                detail=str(sorted(referenced - selected)),
            )
        )

    if (
        result.status == "already_directly_grounded"
        and not result.directly_grounding_statement_ids
    ):
        issues.append(
            IG11ValidationIssue(
                code="directly_grounded_without_support_ids",
                detail="direct status requires direct grounding IDs",
            )
        )
    if (
        result.status == "already_synthesis_grounded"
        and len(result.synthesis_grounding_statement_ids) < 2
    ):
        issues.append(
            IG11ValidationIssue(
                code="synthesis_grounded_without_multiple_ids",
                detail="synthesis status requires at least two support IDs",
            )
        )
    return issues


def bridge_is_acceptable_novel_hypothesis(
    result: IG12BridgeNonRedundancyAudit,
) -> bool:
    return (
        result.status == "genuinely_unestablished_relation"
        and result.scope_compatible
    )


def build_nonredundancy_replan_messages(
    prompt: HypothesisPrompt,
    context: HypothesisContext,
    audit: IG11AxisEvidenceAudit,
    previous_blueprint: IG11Blueprint,
    bridge_audit: IG12BridgeNonRedundancyAudit,
) -> list[dict[str, str]]:
    messages = build_blueprint_messages(
        prompt,
        context,
        audit,
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": previous_blueprint.model_dump_json(indent=2),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "IG1.2 NON-REDUNDANCY REPLAN",
                        "===========================",
                        "The previous one-bridge blueprint failed the "
                        "novel-bridge audit.",
                        f"status: {bridge_audit.status}",
                        f"scope_compatible: {bridge_audit.scope_compatible}",
                        f"explanation: {bridge_audit.explanation}",
                        "",
                        "Choose a NEW endpoint pair and/or ONE new relation "
                        "such that:",
                        "- both endpoint claims are grounded;",
                        "- adjacent bridge-terminal premises may be used;",
                        "- the relation is genuinely unestablished by the "
                        "selected premises;",
                        "- scope transfer is not required;",
                        "- the relation remains a single scientific edge;",
                        "- ungrounded required concrete MATERIAL/SYSTEM "
                        "entities still force abstention;",
                        "- reaction species/products do not trigger the "
                        "material/system identity gate;",
                        "- if no such bounded relation exists, abstain.",
                        "",
                        "Do not merely paraphrase the old relation.",
                        "Return a complete replacement IG11Blueprint.",
                    ]
                ),
            },
        ]
    )
    return messages


def augment_prompt(
    prompt: HypothesisPrompt,
    blueprint: IG11Blueprint,
    audit: IG11AxisEvidenceAudit,
) -> HypothesisPrompt:
    if blueprint.abstain:
        raise ValueError("cannot augment from abstaining IG1.1 blueprint")

    assert blueprint.endpoint_a is not None
    assert blueprint.endpoint_b is not None
    assert blueprint.scope_envelope is not None
    assert blueprint.novel_bridge is not None
    assert blueprint.discriminative_test is not None

    exact_premises = blueprint_premise_ids(blueprint)

    appendix = f"""
IG1.1 MINIMAL-BURDEN / SCOPE-ENVELOPE CONTRACT
==============================================
This plan is immutable.

GROUNDED ENDPOINT A
anchor_statement_id: {blueprint.endpoint_a.anchor_statement_id}
supporting_statement_ids: {json.dumps(blueprint.endpoint_a.supporting_statement_ids, ensure_ascii=False)}
grounded_excerpt: {blueprint.endpoint_a.grounded_excerpt}
scope_summary: {blueprint.endpoint_a.scope.scope_summary}

GROUNDED ENDPOINT B
anchor_statement_id: {blueprint.endpoint_b.anchor_statement_id}
supporting_statement_ids: {json.dumps(blueprint.endpoint_b.supporting_statement_ids, ensure_ascii=False)}
grounded_excerpt: {blueprint.endpoint_b.grounded_excerpt}
scope_summary: {blueprint.endpoint_b.scope.scope_summary}

AXIS DIRECT-GROUNDING STATEMENTS
{json.dumps(audit.direct_axis_statement_ids, ensure_ascii=False)}

BRIDGE SCOPE ENVELOPE
scope_guard_phrase: {blueprint.scope_envelope.scope_guard_phrase}
system/material: {blueprint.scope_envelope.system_or_material_scope}
entity/pair: {blueprint.scope_envelope.entity_or_pair_scope}
structure/coordination: {blueprint.scope_envelope.structural_or_coordination_scope}
observable/outcome: {blueprint.scope_envelope.observable_or_outcome_scope}

THE ONLY NOVEL SCIENTIFIC RELATION
{blueprint.novel_bridge.relation}

THE ONE DISCRIMINATIVE TEST
observable: {blueprint.discriminative_test.observable}
expected_direction: {blueprint.discriminative_test.expected_direction}
falsifying_outcome: {blueprint.discriminative_test.falsifying_outcome}

FINAL-DRAFT HARD RULES
- Return exactly ONE hypothesis or abstain.
- hypothesis_statement MUST EQUAL the novel relation above EXACTLY. Do not add
  any prefix, suffix, consequence, mechanism, recommendation, or broader scope.
- premise_statement_ids MUST be exactly:
  {json.dumps(exact_premises, ensure_ascii=False)}
- inferential_bridge must explain only:
  grounded endpoint A + grounded endpoint B -> motivation to test the one
  proposed relation.
- inferential_bridge must contain the exact novel relation.
- inferential_bridge must contain the exact scope guard:
  {blueprint.scope_envelope.scope_guard_phrase}
- Do not generalize either endpoint beyond its stated scope.
- Do not introduce a concrete axis entity that is absent from grounded
  endpoints/evidence.
- Exactly ONE predicted observation.
- Exactly ONE falsification criterion.
- Both use this exact observable:
  {blueprint.discriminative_test.observable}
- Prediction direction must be exactly:
  {blueprint.discriminative_test.expected_direction}
- No downstream DFT initialization recommendation unless the blueprint's one
  novel relation itself is explicitly a design_rule.
- If any extra novel scientific edge is required, abstain instead.
""".strip()

    system_prompt = (
        prompt.system_prompt.rstrip()
        + "\n\n"
        + "IG1.1 OVERRIDING POLICY\n"
        + "=======================\n"
        + "Use the fixed endpoint pair, scope envelope, and exactly one novel "
        + "relation. The final hypothesis statement must be exactly that "
        + "relation and may not broaden its scientific scope.\n"
    )
    user_prompt = (
        prompt.user_prompt.rstrip()
        + "\n\n"
        + appendix
        + "\n"
    )

    canonical = _canonical_json(
        {
            "prompt_version": IG11_PROMPT_VERSION,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
    )
    return HypothesisPrompt(
        prompt_version=IG11_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_sha256=hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    )


def final_conformance_issues(
    draft: HypothesisPortfolioDraft,
    blueprint: IG11Blueprint,
    audit: IG11AxisEvidenceAudit,
) -> list[IG1ConformanceIssue]:
    base_issues = draft_conformance_issues(
        draft,
        blueprint,  # duck-typed compatible with IG1 blueprint contract
    )
    issues = list(base_issues)

    if blueprint.abstain or not draft.hypotheses:
        return issues
    if len(draft.hypotheses) != 1:
        return issues

    assert blueprint.novel_bridge is not None
    assert blueprint.scope_envelope is not None

    proposal = draft.hypotheses[0]

    if proposal.hypothesis_statement != blueprint.novel_bridge.relation:
        issues.append(
            IG1ConformanceIssue(
                code="hypothesis_not_exact_novel_relation",
                detail="hypothesis_statement must equal blueprint relation exactly",
            )
        )

    guard = blueprint.scope_envelope.scope_guard_phrase
    if guard not in proposal.hypothesis_statement:
        issues.append(
            IG1ConformanceIssue(
                code="scope_guard_missing_from_hypothesis",
                detail=guard,
            )
        )
    if guard not in proposal.inferential_bridge:
        issues.append(
            IG1ConformanceIssue(
                code="scope_guard_missing_from_bridge",
                detail=guard,
            )
        )

    selected = set(proposal.premise_statement_ids)
    direct = set(audit.direct_axis_statement_ids)
    if direct and not (selected & direct):
        issues.append(
            IG1ConformanceIssue(
                code="direct_axis_evidence_missing_in_final",
                detail=(
                    f"direct={sorted(direct)}; "
                    f"selected={sorted(selected)}"
                ),
            )
        )

    return issues


def _repair_feedback(
    blueprint: IG11Blueprint,
    issues: list[IG1ConformanceIssue],
) -> str:
    assert not blueprint.abstain
    assert blueprint.novel_bridge is not None
    assert blueprint.scope_envelope is not None
    assert blueprint.discriminative_test is not None

    return "\n".join(
        [
            "IG1.1 FINAL CONFORMANCE REPAIR",
            "==============================",
            "Do not change the fixed blueprint.",
            *[
                f"- {row.code}: {row.detail}"
                for row in issues
            ],
            "",
            "hypothesis_statement MUST equal exactly:",
            blueprint.novel_bridge.relation,
            "",
            "scope_guard_phrase:",
            blueprint.scope_envelope.scope_guard_phrase,
            "",
            "exact premise set:",
            json.dumps(
                blueprint_premise_ids(blueprint),
                ensure_ascii=False,
            ),
            "",
            "exact observable:",
            blueprint.discriminative_test.observable,
            "",
            "Return exactly one hypothesis, one prediction, and one falsifier. "
            "Do not add any second scientific relation.",
        ]
    )


class IG11StructuredGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 2,
        timeout: float | None = 180.0,
        extra_headers: dict[str, str] | None = None,
        max_output_tokens: int = 2048,
    ) -> None:
        if max_output_tokens < 256:
            raise ValueError("max_output_tokens must be >= 256")
        self.model_name = str(model)
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env)
        )
        self.api_key_env = api_key_env
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.max_output_tokens = int(max_output_tokens)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env}."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "IG1.1 requires installed 'openai' and 'instructor'."
            ) from exc

        mode = getattr(
            instructor.Mode,
            self.instructor_mode,
            None,
        )
        if mode is None:
            raise ValueError(
                f"Unknown Instructor mode: {self.instructor_mode}"
            )

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers

        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=mode,
        )
        return self._client

    def call(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
    ) -> _StructuredCall:
        planning_stages = {
            "IG11AxisEvidenceAudit": "axis_audit",
            "IG11Blueprint": "blueprint",
            "IG12BridgeNonRedundancyAudit": "bridge_audit",
        }
        validation_stages = {
            "SC1EndpointPairScopeAudit": "sc1_scope_audit",
            "OG1MaterialOperandAudit": "og1_operand_audit",
            "AF1AxisSemanticFidelityAudit": "af1_axis_fidelity",
            "TR1RepairPlan": "tr1_repair",
            "TR1SemanticAudit": "tr1_semantic_audit",
        }
        response_name = response_model.__name__
        stage = planning_stages.get(
            response_name,
            validation_stages.get(response_name, response_name),
        )
        pipeline = (
            "hypothesis_planning"
            if response_name in planning_stages
            else "hypothesis_validation"
        )
        value, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=response_model,
            messages=messages,
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_context={
                "pipeline": pipeline,
                "stage": stage,
                "call_kind": "structured",
            },
            request_kwargs={"max_tokens": self.max_output_tokens},
        )
        if not isinstance(value, response_model):
            value = response_model.model_validate(value)
        return _StructuredCall(
            value=value,
            elapsed_seconds=event.elapsed_seconds or 0.0,
        )


class IG11CappedHypothesisBackend:
    """Experimental OpenAI-compatible final-draft backend with an explicit
    output-token ceiling.

    This intentionally avoids changing the canonical hypothesis_llm.py backend.
    """

    backend_name = "ig11_capped_instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 3,
        timeout: float | None = 180.0,
        extra_headers: dict[str, str] | None = None,
        max_output_tokens: int = 2048,
    ) -> None:
        if max_output_tokens < 256:
            raise ValueError("max_output_tokens must be >= 256")
        self.model_name = str(model)
        self.api_key = (
            api_key if api_key is not None else os.getenv(api_key_env)
        )
        self.api_key_env = api_key_env
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.max_output_tokens = int(max_output_tokens)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env}."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "IG1.1 requires installed 'openai' and 'instructor'."
            ) from exc

        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            raise ValueError(
                f"Unknown Instructor mode: {self.instructor_mode}"
            )

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers

        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=mode,
        )
        return self._client

    def _call(
        self,
        messages: list[dict[str, str]],
    ) -> HypothesisDraftGeneration:
        stage = (
            "ig11_final_generation"
            if len(messages) == 2
            else "ig11_final_repair"
        )
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=HypothesisPortfolioDraft,
            messages=messages,
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_context={
                "pipeline": "hypothesis_maker",
                "stage": stage,
                "call_kind": stage,
            },
            request_kwargs={"max_tokens": self.max_output_tokens},
        )
        if not isinstance(draft, HypothesisPortfolioDraft):
            draft = HypothesisPortfolioDraft.model_validate(draft)
        return HypothesisDraftGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )

    def generate(
        self,
        prompt: HypothesisPrompt,
    ) -> HypothesisDraftGeneration:
        return self._call(
            [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ]
        )

    def repair(
        self,
        prompt: HypothesisPrompt,
        previous_draft: HypothesisPortfolioDraft,
        feedback: str,
    ) -> HypothesisDraftGeneration:
        return self._call(
            [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
                {
                    "role": "assistant",
                    "content": previous_draft.model_dump_json(indent=2),
                },
                {"role": "user", "content": feedback},
            ]
        )


class IG11HypothesisBackend:
    backend_name = "ig11_minimal_burden_scope"

    def __init__(
        self,
        base_backend: HypothesisDraftBackend,
        *,
        context: HypothesisContext,
        audit_generator: IG11StructuredGenerator,
        blueprint_generator: IG11StructuredGenerator,
        bridge_audit_generator: IG11StructuredGenerator,
        max_audit_repairs: int = 1,
        max_blueprint_repairs: int = 1,
        max_nonredundancy_replans: int = 1,
        max_conformance_repairs: int = 1,
    ) -> None:
        if max_audit_repairs not in {0, 1}:
            raise ValueError("max_audit_repairs must be 0 or 1")
        if max_blueprint_repairs not in {0, 1}:
            raise ValueError("max_blueprint_repairs must be 0 or 1")
        if max_nonredundancy_replans not in {0, 1}:
            raise ValueError("max_nonredundancy_replans must be 0 or 1")
        if max_conformance_repairs not in {0, 1}:
            raise ValueError("max_conformance_repairs must be 0 or 1")

        self.base_backend = base_backend
        self.context = context
        self.audit_generator = audit_generator
        self.blueprint_generator = blueprint_generator
        self.bridge_audit_generator = bridge_audit_generator
        self.max_audit_repairs = max_audit_repairs
        self.max_blueprint_repairs = max_blueprint_repairs
        self.max_nonredundancy_replans = max_nonredundancy_replans
        self.max_conformance_repairs = max_conformance_repairs

        self.model_name = base_backend.model_name

        self._audit_records: dict[
            str,
            IG11EvidenceAuditRecord,
        ] = {}
        self._blueprint_records: dict[
            str,
            IG11BlueprintRecord,
        ] = {}
        self._bridge_audit_records: dict[
            str,
            IG12BridgeAuditRecord,
        ] = {}
        self._augmented_prompts: dict[
            str,
            HypothesisPrompt,
        ] = {}

    def _audit(
        self,
        prompt: HypothesisPrompt,
    ) -> IG11EvidenceAuditRecord:
        cached = self._audit_records.get(
            prompt.prompt_sha256
        )
        if cached is not None:
            return cached

        axis = _axis_metadata(prompt)

        call = self.audit_generator.call(
            build_axis_audit_messages(
                prompt,
                self.context,
            ),
            IG11AxisEvidenceAudit,
        )
        audit = call.value
        assert isinstance(audit, IG11AxisEvidenceAudit)

        generation_attempts = 1
        repair_count = 0

        issues = validate_axis_audit(
            audit,
            context=self.context,
            expected_axis_id=axis["axis_id"],
        )

        if issues and self.max_audit_repairs:
            call = self.audit_generator.call(
                build_axis_audit_repair_messages(
                    prompt,
                    self.context,
                    audit,
                    issues,
                ),
                IG11AxisEvidenceAudit,
            )
            audit = call.value
            assert isinstance(
                audit,
                IG11AxisEvidenceAudit,
            )
            generation_attempts += 1
            repair_count = 1

            issues = validate_axis_audit(
                audit,
                context=self.context,
                expected_axis_id=axis["axis_id"],
            )

        record = IG11EvidenceAuditRecord(
            axis_id=axis["axis_id"],
            source_prompt_sha256=prompt.prompt_sha256,
            audit_sha256=_sha256_json(audit),
            audit=audit,
            valid=not issues,
            generation_attempts=generation_attempts,
            repair_count=repair_count,
            validation_issues=issues,
        )
        self._audit_records[prompt.prompt_sha256] = record
        return record

    def _blueprint(
        self,
        prompt: HypothesisPrompt,
    ) -> IG11BlueprintRecord:
        cached = self._blueprint_records.get(
            prompt.prompt_sha256
        )
        if cached is not None:
            return cached

        axis = _axis_metadata(prompt)
        audit_record = self._audit(prompt)

        if not audit_record.valid:
            blueprint = IG11Blueprint(
                axis_id=axis["axis_id"],
                abstain=True,
                abstention_reason=(
                    "IG1.2 evidence audit failed deterministic validation: "
                    + "; ".join(
                        f"{row.code}={row.detail}"
                        for row in audit_record.validation_issues
                    )
                ),
            )
            record = IG11BlueprintRecord(
                axis_id=axis["axis_id"],
                source_prompt_sha256=prompt.prompt_sha256,
                evidence_audit_sha256=audit_record.audit_sha256,
                blueprint_sha256=_sha256_json(blueprint),
                blueprint=blueprint,
                generation_attempts=0,
                nonredundancy_replan_count=0,
                valid=False,
                validation_issues=[
                    IG11ValidationIssue(
                        code="invalid_axis_evidence_audit",
                        detail="blueprint generation skipped",
                    )
                ],
            )
            self._blueprint_records[prompt.prompt_sha256] = record
            return record

        audit = audit_record.audit

        call = self.blueprint_generator.call(
            build_blueprint_messages(
                prompt,
                self.context,
                audit,
            ),
            IG11Blueprint,
        )
        blueprint = call.value
        assert isinstance(blueprint, IG11Blueprint)
        attempts = 1
        nonredundancy_replans = 0

        issues = validate_blueprint(
            blueprint,
            audit=audit,
            context=self.context,
            expected_axis_id=axis["axis_id"],
        )

        if issues and self.max_blueprint_repairs:
            messages = build_blueprint_messages(
                prompt,
                self.context,
                audit,
            )
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": blueprint.model_dump_json(indent=2),
                    },
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                "IG1.2 BLUEPRINT REPAIR",
                                "The previous blueprint failed deterministic "
                                "grounding/scope checks.",
                                "Repair only the listed violations. If a "
                                "required concrete axis entity is ungrounded "
                                "or scope transfer cannot be avoided, abstain.",
                                *[
                                    f"- {row.code}: {row.detail}"
                                    for row in issues
                                ],
                            ]
                        ),
                    },
                ]
            )
            call = self.blueprint_generator.call(
                messages,
                IG11Blueprint,
            )
            blueprint = call.value
            assert isinstance(blueprint, IG11Blueprint)
            attempts += 1
            issues = validate_blueprint(
                blueprint,
                audit=audit,
                context=self.context,
                expected_axis_id=axis["axis_id"],
            )

        # Only scientifically valid active blueprints reach the novelty gate.
        if not issues and not blueprint.abstain:
            bridge_call = self.bridge_audit_generator.call(
                build_bridge_nonredundancy_messages(
                    prompt,
                    self.context,
                    audit,
                    blueprint,
                ),
                IG12BridgeNonRedundancyAudit,
            )
            bridge_audit = bridge_call.value
            assert isinstance(
                bridge_audit,
                IG12BridgeNonRedundancyAudit,
            )

            bridge_issues = validate_bridge_nonredundancy_audit(
                bridge_audit,
                blueprint=blueprint,
                expected_axis_id=axis["axis_id"],
            )

            bridge_record = IG12BridgeAuditRecord(
                axis_id=axis["axis_id"],
                blueprint_sha256=_sha256_json(blueprint),
                audit_sha256=_sha256_json(bridge_audit),
                audit=bridge_audit,
                valid=not bridge_issues,
                validation_issues=bridge_issues,
                generation_attempts=1,
            )
            self._bridge_audit_records[
                prompt.prompt_sha256
            ] = bridge_record

            acceptable = (
                not bridge_issues
                and bridge_is_acceptable_novel_hypothesis(
                    bridge_audit
                )
            )

            if (
                not acceptable
                and self.max_nonredundancy_replans
            ):
                replan_call = self.blueprint_generator.call(
                    build_nonredundancy_replan_messages(
                        prompt,
                        self.context,
                        audit,
                        blueprint,
                        bridge_audit,
                    ),
                    IG11Blueprint,
                )
                blueprint = replan_call.value
                assert isinstance(blueprint, IG11Blueprint)
                attempts += 1
                nonredundancy_replans = 1

                issues = validate_blueprint(
                    blueprint,
                    audit=audit,
                    context=self.context,
                    expected_axis_id=axis["axis_id"],
                )

                if not issues and not blueprint.abstain:
                    bridge_call = self.bridge_audit_generator.call(
                        build_bridge_nonredundancy_messages(
                            prompt,
                            self.context,
                            audit,
                            blueprint,
                        ),
                        IG12BridgeNonRedundancyAudit,
                    )
                    bridge_audit = bridge_call.value
                    assert isinstance(
                        bridge_audit,
                        IG12BridgeNonRedundancyAudit,
                    )
                    bridge_issues = validate_bridge_nonredundancy_audit(
                        bridge_audit,
                        blueprint=blueprint,
                        expected_axis_id=axis["axis_id"],
                    )
                    bridge_record = IG12BridgeAuditRecord(
                        axis_id=axis["axis_id"],
                        blueprint_sha256=_sha256_json(blueprint),
                        audit_sha256=_sha256_json(bridge_audit),
                        audit=bridge_audit,
                        valid=not bridge_issues,
                        validation_issues=bridge_issues,
                        generation_attempts=2,
                    )
                    self._bridge_audit_records[
                        prompt.prompt_sha256
                    ] = bridge_record

                    acceptable = (
                        not bridge_issues
                        and bridge_is_acceptable_novel_hypothesis(
                            bridge_audit
                        )
                    )
                else:
                    acceptable = blueprint.abstain and not issues

            if (
                not blueprint.abstain
                and not acceptable
            ):
                bridge_record = self._bridge_audit_records.get(
                    prompt.prompt_sha256
                )
                detail = (
                    bridge_record.audit.status
                    if bridge_record is not None
                    else "bridge_audit_invalid"
                )
                blueprint = IG11Blueprint(
                    axis_id=axis["axis_id"],
                    abstain=True,
                    abstention_reason=(
                        "IG1.2 could not obtain a genuinely unestablished "
                        "single bridge after bounded replan. "
                        f"Final bridge audit status: {detail}."
                    ),
                )
                issues = []

        valid = not issues
        if not valid:
            blueprint = IG11Blueprint(
                axis_id=axis["axis_id"],
                abstain=True,
                abstention_reason=(
                    "IG1.2 blueprint failed deterministic checks: "
                    + "; ".join(
                        f"{row.code}={row.detail}"
                        for row in issues
                    )
                ),
            )

        record = IG11BlueprintRecord(
            axis_id=axis["axis_id"],
            source_prompt_sha256=prompt.prompt_sha256,
            evidence_audit_sha256=audit_record.audit_sha256,
            blueprint_sha256=_sha256_json(blueprint),
            blueprint=blueprint,
            generation_attempts=attempts,
            nonredundancy_replan_count=nonredundancy_replans,
            valid=valid,
            validation_issues=issues,
        )
        self._blueprint_records[prompt.prompt_sha256] = record
        return record

    def _finalize(
        self,
        *,
        source_prompt: HypothesisPrompt,
        augmented_prompt: HypothesisPrompt,
        record: IG11BlueprintRecord,
        audit: IG11AxisEvidenceAudit,
        generation: HypothesisDraftGeneration,
    ) -> HypothesisDraftGeneration:
        blueprint = record.blueprint

        if blueprint.abstain:
            return HypothesisDraftGeneration(
                draft=HypothesisPortfolioDraft(
                    hypotheses=[],
                    abstention_reason=(
                        blueprint.abstention_reason
                        or "IG1.1 blueprint abstained"
                    ),
                )
            )

        issues = final_conformance_issues(
            generation.draft,
            blueprint,
            audit,
        )
        repairs = 0

        if issues and self.max_conformance_repairs:
            generation = self.base_backend.repair(
                augmented_prompt,
                generation.draft,
                _repair_feedback(
                    blueprint,
                    issues,
                ),
            )
            repairs = 1

        record.final_generation_count += 1
        record.conformance_repair_count += repairs
        return generation

    def generate(
        self,
        prompt: HypothesisPrompt,
    ) -> HypothesisDraftGeneration:
        audit_record = self._audit(prompt)
        record = self._blueprint(prompt)
        blueprint = record.blueprint

        if blueprint.abstain:
            record.final_generation_count += 1
            return HypothesisDraftGeneration(
                draft=HypothesisPortfolioDraft(
                    hypotheses=[],
                    abstention_reason=(
                        blueprint.abstention_reason
                        or "IG1.1 blueprint abstained"
                    ),
                )
            )

        augmented = augment_prompt(
            prompt,
            blueprint,
            audit_record.audit,
        )
        self._augmented_prompts[
            prompt.prompt_sha256
        ] = augmented

        generation = self.base_backend.generate(
            augmented
        )
        return self._finalize(
            source_prompt=prompt,
            augmented_prompt=augmented,
            record=record,
            audit=audit_record.audit,
            generation=generation,
        )

    def repair(
        self,
        prompt: HypothesisPrompt,
        previous_draft: HypothesisPortfolioDraft,
        feedback: str,
    ) -> HypothesisDraftGeneration:
        audit_record = self._audit(prompt)
        record = self._blueprint(prompt)
        blueprint = record.blueprint

        if blueprint.abstain:
            return HypothesisDraftGeneration(
                draft=HypothesisPortfolioDraft(
                    hypotheses=[],
                    abstention_reason=(
                        blueprint.abstention_reason
                        or "IG1.1 blueprint abstained"
                    ),
                )
            )

        augmented = self._augmented_prompts.get(
            prompt.prompt_sha256
        )
        if augmented is None:
            augmented = augment_prompt(
                prompt,
                blueprint,
                audit_record.audit,
            )
            self._augmented_prompts[
                prompt.prompt_sha256
            ] = augmented

        generation = self.base_backend.repair(
            augmented,
            previous_draft,
            feedback
            + "\n\n"
            + _repair_feedback(
                blueprint,
                [],
            ),
        )
        return self._finalize(
            source_prompt=prompt,
            augmented_prompt=augmented,
            record=record,
            audit=audit_record.audit,
            generation=generation,
        )

    def audit_records(self) -> list[IG11EvidenceAuditRecord]:
        return sorted(
            self._audit_records.values(),
            key=lambda row: row.axis_id,
        )

    def bridge_audit_records(self) -> list[IG12BridgeAuditRecord]:
        return sorted(
            self._bridge_audit_records.values(),
            key=lambda row: row.axis_id,
        )

    def blueprint_records(self) -> list[IG11BlueprintRecord]:
        return sorted(
            self._blueprint_records.values(),
            key=lambda row: row.axis_id,
        )

    def augmented_prompt(
        self,
        source_prompt_sha256: str,
    ) -> HypothesisPrompt | None:
        return self._augmented_prompts.get(
            source_prompt_sha256
        )


def build_planning_report(
    backend: IG11HypothesisBackend,
    *,
    context: HypothesisContext,
    axis_plan_id: str,
) -> IG11PlanningReport:
    audits = backend.audit_records()
    bridge_audits = backend.bridge_audit_records()
    blueprints = backend.blueprint_records()

    payload = {
        "schema_version": "ig1.1-planning-report-v1",
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "source_axis_plan_id": axis_plan_id,
        "audit_model": backend.audit_generator.model_name,
        "blueprint_model": backend.blueprint_generator.model_name,
        "axis_count": len(blueprints),
        "audit_valid_count": sum(row.valid for row in audits),
        "audit_repair_count": sum(
            row.repair_count
            for row in audits
        ),
        "bridge_audit_count": len(bridge_audits),
        "bridge_nonredundancy_replan_count": sum(
            row.nonredundancy_replan_count
            for row in blueprints
        ),
        "active_blueprint_count": sum(
            not row.blueprint.abstain
            for row in blueprints
        ),
        "abstained_blueprint_count": sum(
            row.blueprint.abstain
            for row in blueprints
        ),
        "invalid_blueprint_count": sum(
            not row.valid
            for row in blueprints
        ),
        "conformance_repair_count": sum(
            row.conformance_repair_count
            for row in blueprints
        ),
        "audit_records": [
            row.model_dump(mode="json")
            for row in audits
        ],
        "bridge_audit_records": [
            row.model_dump(mode="json")
            for row in bridge_audits
        ],
        "blueprint_records": [
            row.model_dump(mode="json")
            for row in blueprints
        ],
    }

    payload["report_id"] = _stable_id(
        "ig11_planning_report",
        context.context_sha256,
        axis_plan_id,
        backend.audit_generator.model_name,
        backend.blueprint_generator.model_name,
        _sha256_json(payload["audit_records"]),
        _sha256_json(payload["bridge_audit_records"]),
        _sha256_json(payload["blueprint_records"]),
    )

    return IG11PlanningReport(
        **payload,
        report_sha256=_sha256_json(payload),
    )


def build_conformance_report(
    *,
    portfolio: HypothesisPortfolio,
    axis_report: Any,
    planning_report: IG11PlanningReport,
) -> IG11ConformanceReport:
    blueprint_by_axis = {
        row.axis_id: row
        for row in planning_report.blueprint_records
    }
    audit_by_axis = {
        row.axis_id: row
        for row in planning_report.audit_records
    }
    lineage_by_hypothesis = {
        row.hypothesis_id: row
        for row in axis_report.lineages
    }

    cards: list[IG11ConformanceCard] = []

    for hypothesis in portfolio.hypotheses:
        lineage = lineage_by_hypothesis.get(
            hypothesis.hypothesis_id
        )
        if lineage is None:
            raise ValueError(
                "IG1.1 conformance missing hypothesis lineage"
            )

        bp_record = blueprint_by_axis.get(lineage.axis_id)
        audit_record = audit_by_axis.get(lineage.axis_id)
        if bp_record is None or audit_record is None:
            raise ValueError(
                "IG1.1 conformance missing planning artifact"
            )

        blueprint = bp_record.blueprint
        if blueprint.abstain:
            raise ValueError(
                "accepted hypothesis maps to abstaining IG1.1 blueprint"
            )

        assert blueprint.novel_bridge is not None
        assert blueprint.scope_envelope is not None

        expected = blueprint_premise_ids(blueprint)
        actual = sorted(
            set(hypothesis.premise_statement_ids)
        )
        direct = list(
            audit_record.audit.direct_axis_statement_ids
        )
        direct_used = (
            not direct
            or bool(set(actual) & set(direct))
        )

        issues: list[IG1ConformanceIssue] = []
        if hypothesis.hypothesis_statement != blueprint.novel_bridge.relation:
            issues.append(
                IG1ConformanceIssue(
                    code="hypothesis_not_exact_novel_relation",
                    detail="portfolio hypothesis differs from blueprint",
                )
            )
        if set(expected) != set(actual):
            issues.append(
                IG1ConformanceIssue(
                    code="premise_set_mismatch",
                    detail=(
                        f"expected={expected}; actual={actual}"
                    ),
                )
            )

        guard = blueprint.scope_envelope.scope_guard_phrase
        guard_h = guard in hypothesis.hypothesis_statement
        guard_b = guard in hypothesis.inferential_bridge
        if not guard_h:
            issues.append(
                IG1ConformanceIssue(
                    code="scope_guard_missing_from_hypothesis",
                    detail=guard,
                )
            )
        if not guard_b:
            issues.append(
                IG1ConformanceIssue(
                    code="scope_guard_missing_from_bridge",
                    detail=guard,
                )
            )
        if not direct_used:
            issues.append(
                IG1ConformanceIssue(
                    code="direct_axis_evidence_missing_in_final",
                    detail=str(direct),
                )
            )

        cards.append(
            IG11ConformanceCard(
                hypothesis_id=hypothesis.hypothesis_id,
                axis_id=lineage.axis_id,
                blueprint_sha256=bp_record.blueprint_sha256,
                passes=not issues,
                exact_hypothesis_equals_novel_relation=(
                    hypothesis.hypothesis_statement
                    == blueprint.novel_bridge.relation
                ),
                expected_premise_statement_ids=expected,
                actual_premise_statement_ids=actual,
                direct_axis_statement_ids=direct,
                direct_axis_grounding_used=direct_used,
                scope_guard_phrase=guard,
                scope_guard_in_hypothesis=guard_h,
                scope_guard_in_bridge=guard_b,
                issues=issues,
            )
        )

    counts = Counter(
        issue.code
        for card in cards
        for issue in card.issues
    )

    payload = {
        "schema_version": "ig1.1-conformance-report-v1",
        "source_portfolio_id": portfolio.portfolio_id,
        "source_axis_report_id": axis_report.report_id,
        "source_planning_report_id": planning_report.report_id,
        "hypothesis_count": len(cards),
        "passing_count": sum(row.passes for row in cards),
        "failing_count": sum(not row.passes for row in cards),
        "issue_counts": dict(counts),
        "cards": [
            row.model_dump(mode="json")
            for row in cards
        ],
    }
    payload["report_id"] = _stable_id(
        "ig11_conformance_report",
        portfolio.portfolio_id,
        axis_report.report_id,
        planning_report.report_id,
    )
    return IG11ConformanceReport(
        **payload,
        report_sha256=_sha256_json(payload),
    )
