from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.evidence_family_decomposition import (
    EvidenceFamilyDecompositionReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FamilyPremiseSelectionPolicy(StrictModel):
    mode: Literal["minimally_sufficient_family_premise"] = (
        "minimally_sufficient_family_premise"
    )
    hierarchy_is_scientific_evidence: Literal[False] = False
    child_is_additional_independent_evidence: Literal[False] = False
    child_use_forced: Literal[False] = False
    parent_use_forbidden: Literal[False] = False
    prefer_specific_sufficient_child: Literal[True] = True
    parent_allowed_for_cross_family_synthesis: Literal[True] = True
    avoid_parent_plus_all_children_redundancy: Literal[True] = True


class FamilyHierarchyChild(StrictModel):
    child_statement_id: str
    family_id: str
    paper_ids: list[str] = Field(default_factory=list)
    claim_kind: str
    node_types: list[str] = Field(default_factory=list)
    edge_relations: list[str] = Field(default_factory=list)


class FamilyHierarchyGroup(StrictModel):
    parent_statement_id: str
    child_statement_ids: list[str] = Field(default_factory=list)
    children: list[FamilyHierarchyChild] = Field(default_factory=list)


class EvidenceFamilyHierarchy(StrictModel):
    schema_version: Literal["evidence-family-hierarchy-v1"] = (
        "evidence-family-hierarchy-v1"
    )
    hierarchy_id: str
    hierarchy_sha256: str
    source_decomposition_report_id: str
    source_decomposition_report_sha256: str
    source_context_id: str
    source_context_sha256: str
    domain_profile_id: str
    groups: list[FamilyHierarchyGroup] = Field(default_factory=list)
    policy: FamilyPremiseSelectionPolicy = Field(
        default_factory=FamilyPremiseSelectionPolicy
    )

    @classmethod
    def from_decomposition_report(
        cls,
        report: EvidenceFamilyDecompositionReport,
        context: HypothesisContext,
    ) -> "EvidenceFamilyHierarchy":
        if report.source_context_id != context.context_id:
            raise ValueError("EC2-C decomposition/context ID mismatch")
        if report.output_context_sha256_after != context.context_sha256:
            raise ValueError(
                "EC2-C decomposition report does not describe the supplied "
                "expanded context SHA"
            )
        if report.domain_profile_id != context.domain_profile_id:
            raise ValueError("EC2-C decomposition/context domain mismatch")

        statement_by_id = {
            row.statement_id: row
            for row in context.evidence_statements
        }
        grouped: dict[str, list[Any]] = defaultdict(list)
        for row in report.child_lineages:
            grouped[row.parent_statement_id].append(row)

        groups: list[FamilyHierarchyGroup] = []
        for parent_id in sorted(grouped):
            parent = statement_by_id.get(parent_id)
            if parent is None:
                raise ValueError(
                    f"EC2-C parent statement missing from context: {parent_id}"
                )
            if not parent.eligible_as_premise:
                raise ValueError(
                    f"EC2-C parent is not an eligible premise: {parent_id}"
                )

            children: list[FamilyHierarchyChild] = []
            for row in sorted(
                grouped[parent_id],
                key=lambda item: (
                    item.family_id,
                    item.child_statement_id,
                ),
            ):
                child = statement_by_id.get(row.child_statement_id)
                if child is None:
                    raise ValueError(
                        "EC2-C child statement missing from context: "
                        f"{row.child_statement_id}"
                    )
                if not child.eligible_as_premise:
                    raise ValueError(
                        "EC2-C child is not an eligible premise: "
                        f"{row.child_statement_id}"
                    )
                if set(child.paper_ids) != set(row.paper_ids):
                    raise ValueError(
                        "EC2-C child/decomposition paper mismatch: "
                        f"{row.child_statement_id}"
                    )
                children.append(
                    FamilyHierarchyChild(
                        child_statement_id=row.child_statement_id,
                        family_id=row.family_id,
                        paper_ids=list(row.paper_ids),
                        claim_kind=row.claim_kind,
                        node_types=list(row.node_types),
                        edge_relations=list(row.edge_relations),
                    )
                )

            groups.append(
                FamilyHierarchyGroup(
                    parent_statement_id=parent_id,
                    child_statement_ids=[
                        child.child_statement_id
                        for child in children
                    ],
                    children=children,
                )
            )

        payload = {
            "schema_version": "evidence-family-hierarchy-v1",
            "source_decomposition_report_id": report.report_id,
            "source_decomposition_report_sha256": report.report_sha256,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "domain_profile_id": context.domain_profile_id,
            "groups": [
                group.model_dump(mode="json")
                for group in groups
            ],
            "policy": FamilyPremiseSelectionPolicy().model_dump(mode="json"),
        }
        payload["hierarchy_id"] = _stable_id(
            "evidence_family_hierarchy",
            report.report_sha256,
            context.context_sha256,
        )
        return cls(
            **payload,
            hierarchy_sha256=_sha256_json(payload),
        )


class HypothesisFamilySelectionCard(StrictModel):
    hypothesis_id: str
    title: str
    premise_statement_ids: list[str] = Field(default_factory=list)
    used_parent_statement_ids: list[str] = Field(default_factory=list)
    used_child_statement_ids: list[str] = Field(default_factory=list)
    family_usage_class: Literal[
        "no_family_hierarchy_usage",
        "parent_only",
        "child_only",
        "multiple_children_without_parent",
        "parent_plus_subset_children",
        "parent_plus_all_children_potentially_redundant",
        "mixed_multiple_groups",
    ]
    potentially_redundant_parent_child_group_ids: list[str] = Field(
        default_factory=list
    )


class FamilyPremiseSelectionReport(StrictModel):
    schema_version: Literal["family-premise-selection-report-v1"] = (
        "family-premise-selection-report-v1"
    )
    report_id: str
    report_sha256: str
    source_hierarchy_id: str
    source_hierarchy_sha256: str
    source_portfolio_id: str
    source_context_id: str
    source_context_sha256: str
    hypothesis_count: int = 0
    hypotheses_using_any_child_count: int = 0
    hypotheses_using_any_parent_count: int = 0
    parent_premise_incidence_count: int = 0
    child_premise_incidence_count: int = 0
    potential_parent_all_children_redundancy_count: int = 0
    cards: list[HypothesisFamilySelectionCard] = Field(default_factory=list)
    policy: FamilyPremiseSelectionPolicy = Field(
        default_factory=FamilyPremiseSelectionPolicy
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
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def render_family_hierarchy_guidance(
    hierarchy: EvidenceFamilyHierarchy,
) -> str:
    lines = [
        "EVIDENCE FAMILY HIERARCHY (SELECTION METADATA; NOT NEW EVIDENCE)",
        "================================================================",
        "Some eligible statements are family-specific addressable views of a broader",
        "evidence-synthesis parent. A child is NOT additional independent evidence;",
        "it exposes only the scientific support belonging to one decomposed family.",
        "",
        "MINIMALLY-SUFFICIENT PREMISE PRINCIPLE",
        "--------------------------------------",
        "- Prefer the most specific family child when that child alone is sufficient",
        "  for the scientific premise needed by the hypothesis.",
        "- Use the broader parent synthesis when the inferential bridge genuinely",
        "  depends on multiple evidence families represented by that parent.",
        "- Do NOT select a child merely to increase premise diversity or premise count.",
        "- Do NOT select the parent together with all of its children merely to repeat",
        "  the same underlying evidence at multiple granularities.",
        "- A family hierarchy is representation/provenance guidance only. It does not",
        "  make any scientific relation more established than the eligible statements",
        "  themselves already make it.",
        "",
        "FAMILY GROUPS",
        "-------------",
    ]
    if not hierarchy.groups:
        lines.append("- NONE")
        return "\n".join(lines)

    for index, group in enumerate(hierarchy.groups, start=1):
        lines.append(
            f"- group_{index}: parent={group.parent_statement_id}"
        )
        for child in group.children:
            lines.append(
                "  child="
                f"{child.child_statement_id}; "
                f"claim_kind={child.claim_kind}; "
                f"papers={','.join(child.paper_ids) or '-'}; "
                f"node_types={','.join(child.node_types) or '-'}; "
                f"relations={','.join(child.edge_relations) or '-'}"
            )
    return "\n".join(lines)


def audit_family_premise_selection(
    hierarchy: EvidenceFamilyHierarchy,
    portfolio: HypothesisPortfolio,
) -> FamilyPremiseSelectionReport:
    if portfolio.source_context_id != hierarchy.source_context_id:
        raise ValueError("EC2-C hierarchy/portfolio context ID mismatch")
    if portfolio.source_context_sha256 != hierarchy.source_context_sha256:
        raise ValueError("EC2-C hierarchy/portfolio context SHA mismatch")

    parent_ids = {
        group.parent_statement_id
        for group in hierarchy.groups
    }
    child_ids = {
        child_id
        for group in hierarchy.groups
        for child_id in group.child_statement_ids
    }

    cards: list[HypothesisFamilySelectionCard] = []
    for hypothesis in portfolio.hypotheses:
        premises = set(hypothesis.premise_statement_ids)
        used_parents = sorted(premises & parent_ids)
        used_children = sorted(premises & child_ids)

        touched_groups = []
        redundant_groups = []
        for group in hierarchy.groups:
            parent_used = group.parent_statement_id in premises
            children_used = [
                child_id
                for child_id in group.child_statement_ids
                if child_id in premises
            ]
            if parent_used or children_used:
                touched_groups.append(
                    (group, parent_used, children_used)
                )
            if (
                parent_used
                and group.child_statement_ids
                and set(children_used) == set(group.child_statement_ids)
            ):
                redundant_groups.append(group.parent_statement_id)

        if not touched_groups:
            usage_class = "no_family_hierarchy_usage"
        elif len(touched_groups) > 1:
            usage_class = "mixed_multiple_groups"
        else:
            group, parent_used, children_used = touched_groups[0]
            if (
                parent_used
                and group.child_statement_ids
                and set(children_used) == set(group.child_statement_ids)
            ):
                usage_class = (
                    "parent_plus_all_children_potentially_redundant"
                )
            elif parent_used and children_used:
                usage_class = "parent_plus_subset_children"
            elif parent_used:
                usage_class = "parent_only"
            elif len(children_used) > 1:
                usage_class = "multiple_children_without_parent"
            else:
                usage_class = "child_only"

        cards.append(
            HypothesisFamilySelectionCard(
                hypothesis_id=hypothesis.hypothesis_id,
                title=hypothesis.title,
                premise_statement_ids=list(
                    hypothesis.premise_statement_ids
                ),
                used_parent_statement_ids=used_parents,
                used_child_statement_ids=used_children,
                family_usage_class=usage_class,
                potentially_redundant_parent_child_group_ids=sorted(
                    redundant_groups
                ),
            )
        )

    payload = {
        "schema_version": "family-premise-selection-report-v1",
        "source_hierarchy_id": hierarchy.hierarchy_id,
        "source_hierarchy_sha256": hierarchy.hierarchy_sha256,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_context_id": hierarchy.source_context_id,
        "source_context_sha256": hierarchy.source_context_sha256,
        "hypothesis_count": len(cards),
        "hypotheses_using_any_child_count": sum(
            bool(card.used_child_statement_ids)
            for card in cards
        ),
        "hypotheses_using_any_parent_count": sum(
            bool(card.used_parent_statement_ids)
            for card in cards
        ),
        "parent_premise_incidence_count": sum(
            len(card.used_parent_statement_ids)
            for card in cards
        ),
        "child_premise_incidence_count": sum(
            len(card.used_child_statement_ids)
            for card in cards
        ),
        "potential_parent_all_children_redundancy_count": sum(
            bool(card.potentially_redundant_parent_child_group_ids)
            for card in cards
        ),
        "cards": [
            card.model_dump(mode="json")
            for card in cards
        ],
        "policy": hierarchy.policy.model_dump(mode="json"),
    }
    payload["report_id"] = _stable_id(
        "family_premise_selection_report",
        hierarchy.hierarchy_sha256,
        portfolio.portfolio_id,
    )
    return FamilyPremiseSelectionReport(
        **payload,
        report_sha256=_sha256_json(payload),
    )
