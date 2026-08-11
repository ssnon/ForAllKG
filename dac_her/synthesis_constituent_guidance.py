from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.evidence_constituent_resolution import (
    ExistingConstituentResolutionReport,
)
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from dac_her.hypothesis_llm import (
    HypothesisDraftBackend,
    HypothesisDraftGeneration,
)
from dac_her.hypothesis_prompt import HypothesisPrompt


GUIDED_PROMPT_VERSION = (
    "hypothesis-maker-discovery-axis-prompt-v2.9.1-ec2d2"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SynthesisConstituentSelectionPolicy(StrictModel):
    mode: Literal["synthesis_constituent_minimally_sufficient"] = (
        "synthesis_constituent_minimally_sufficient"
    )
    lineage_is_scientific_evidence: Literal[False] = False
    constituent_use_forced: Literal[False] = False
    parent_use_forbidden: Literal[False] = False
    prefer_existing_covering_constituent_when_sufficient: Literal[True] = True
    parent_allowed_for_cross_family_synthesis: Literal[True] = True
    contained_relation_is_exact_equivalence: Literal[False] = False
    avoid_parent_plus_all_constituents_redundancy: Literal[True] = True


class SynthesisConstituentMember(StrictModel):
    family_id: str
    family_claim_kind: str
    family_paper_ids: list[str] = Field(default_factory=list)
    constituent_statement_id: str
    resolution_status: Literal[
        "resolved_to_existing_statement",
        "materialized_new_family_child",
    ]
    resolution_basis: Literal[
        "exact_existing_scientific_support",
        "contained_existing_scientific_support",
        "contained_existing_node_support",
        "no_existing_constituent_materialized",
    ]
    exact_equivalence: bool = False


class SynthesisConstituentGroup(StrictModel):
    parent_statement_id: str
    constituent_statement_ids: list[str] = Field(default_factory=list)
    members: list[SynthesisConstituentMember] = Field(default_factory=list)


class SynthesisConstituentHierarchy(StrictModel):
    schema_version: Literal["synthesis-constituent-hierarchy-v1"] = (
        "synthesis-constituent-hierarchy-v1"
    )
    hierarchy_id: str
    hierarchy_sha256: str

    source_resolution_report_id: str
    source_resolution_report_sha256: str
    source_context_id: str
    source_context_sha256: str
    domain_profile_id: str

    groups: list[SynthesisConstituentGroup] = Field(default_factory=list)
    policy: SynthesisConstituentSelectionPolicy = Field(
        default_factory=SynthesisConstituentSelectionPolicy
    )

    @classmethod
    def from_resolution_report(
        cls,
        report: ExistingConstituentResolutionReport,
        context: HypothesisContext,
    ) -> "SynthesisConstituentHierarchy":
        if report.source_context_id != context.context_id:
            raise ValueError(
                "EC2-D2 resolution/context ID mismatch"
            )
        if report.output_context_sha256_after != context.context_sha256:
            raise ValueError(
                "EC2-D2 resolution report does not describe the supplied "
                "grounded context SHA"
            )
        if report.domain_profile_id != context.domain_profile_id:
            raise ValueError(
                "EC2-D2 resolution/context domain mismatch"
            )

        statement_by_id = {
            row.statement_id: row
            for row in context.evidence_statements
        }

        grouped: dict[str, list[SynthesisConstituentMember]] = {}
        for row in report.family_resolutions:
            parent = statement_by_id.get(row.parent_statement_id)
            if parent is None:
                raise ValueError(
                    "EC2-D2 parent statement missing from context: "
                    f"{row.parent_statement_id}"
                )
            if not parent.eligible_as_premise:
                raise ValueError(
                    "EC2-D2 parent is not an eligible premise: "
                    f"{row.parent_statement_id}"
                )

            constituent = statement_by_id.get(
                row.resulting_constituent_statement_id
            )
            if constituent is None:
                raise ValueError(
                    "EC2-D2 constituent statement missing from context: "
                    f"{row.resulting_constituent_statement_id}"
                )
            if not constituent.eligible_as_premise:
                raise ValueError(
                    "EC2-D2 constituent is not an eligible premise: "
                    f"{row.resulting_constituent_statement_id}"
                )

            exact = (
                row.resolution_basis
                == "exact_existing_scientific_support"
            )
            member = SynthesisConstituentMember(
                family_id=row.family_id,
                family_claim_kind=row.family_claim_kind,
                family_paper_ids=list(row.family_paper_ids),
                constituent_statement_id=(
                    row.resulting_constituent_statement_id
                ),
                resolution_status=row.resolution_status,
                resolution_basis=row.resolution_basis,
                exact_equivalence=exact,
            )
            grouped.setdefault(
                row.parent_statement_id,
                [],
            ).append(member)

        groups: list[SynthesisConstituentGroup] = []
        for parent_id in sorted(grouped):
            members = sorted(
                grouped[parent_id],
                key=lambda row: (
                    row.family_id,
                    row.constituent_statement_id,
                ),
            )
            groups.append(
                SynthesisConstituentGroup(
                    parent_statement_id=parent_id,
                    constituent_statement_ids=sorted(
                        {
                            row.constituent_statement_id
                            for row in members
                        }
                    ),
                    members=members,
                )
            )

        policy = SynthesisConstituentSelectionPolicy()
        payload = {
            "schema_version": "synthesis-constituent-hierarchy-v1",
            "source_resolution_report_id": report.report_id,
            "source_resolution_report_sha256": report.report_sha256,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "domain_profile_id": context.domain_profile_id,
            "groups": [
                row.model_dump(mode="json")
                for row in groups
            ],
            "policy": policy.model_dump(mode="json"),
        }
        payload["hierarchy_id"] = _stable_id(
            "synthesis_constituent_hierarchy",
            report.report_sha256,
            context.context_sha256,
        )
        return cls(
            **payload,
            hierarchy_sha256=_sha256_json(payload),
        )


class HypothesisConstituentSelectionCard(StrictModel):
    hypothesis_id: str
    title: str
    premise_statement_ids: list[str] = Field(default_factory=list)

    used_parent_statement_ids: list[str] = Field(default_factory=list)
    used_constituent_statement_ids: list[str] = Field(default_factory=list)

    usage_class: Literal[
        "no_synthesis_constituent_usage",
        "parent_only",
        "constituent_only",
        "multiple_constituents_without_parent",
        "parent_plus_subset_constituents",
        "parent_plus_all_constituents_potentially_redundant",
        "mixed_multiple_groups",
    ]

    potentially_redundant_parent_group_ids: list[str] = Field(
        default_factory=list
    )


class SynthesisConstituentSelectionReport(StrictModel):
    schema_version: Literal[
        "synthesis-constituent-selection-report-v1"
    ] = "synthesis-constituent-selection-report-v1"

    report_id: str
    report_sha256: str

    source_hierarchy_id: str
    source_hierarchy_sha256: str
    source_portfolio_id: str
    source_context_id: str
    source_context_sha256: str

    hypothesis_count: int = 0
    hypotheses_using_any_constituent_count: int = 0
    hypotheses_using_any_parent_count: int = 0
    parent_premise_incidence_count: int = 0
    constituent_premise_incidence_count: int = 0
    potential_parent_all_constituents_redundancy_count: int = 0

    cards: list[HypothesisConstituentSelectionCard] = Field(
        default_factory=list
    )
    policy: SynthesisConstituentSelectionPolicy = Field(
        default_factory=SynthesisConstituentSelectionPolicy
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
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")
    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def render_synthesis_constituent_guidance(
    hierarchy: SynthesisConstituentHierarchy,
) -> str:
    lines = [
        "SYNTHESIS–CONSTITUENT LINEAGE (SELECTION METADATA; NOT NEW EVIDENCE)",
        "===================================================================",
        "Some broad evidence-synthesis premises combine scientific support that is",
        "already addressable through other eligible premise statements.",
        "",
        "MINIMALLY-SUFFICIENT PREMISE PRINCIPLE",
        "--------------------------------------",
        "- When an existing covering constituent statement is sufficient for the",
        "  scientific premise needed by the hypothesis, prefer that already-grounded",
        "  constituent instead of the broader synthesis parent.",
        "- Use the broader synthesis parent when the inferential bridge genuinely",
        "  depends on the cross-family synthesis represented by that parent.",
        "- Do NOT select a constituent merely to increase premise diversity or count.",
        "- Do NOT select the parent together with all of its constituents merely to",
        "  repeat the same underlying support at multiple abstraction levels.",
        "- A relation marked 'contained' means the constituent statement covers all",
        "  support of that family but may contain additional support. It is NOT exact",
        "  semantic or evidentiary equivalence to the family.",
        "- This hierarchy is provenance/representation guidance only. It does not",
        "  create new scientific evidence or strengthen any reported claim.",
        "",
        "SYNTHESIS GROUPS",
        "----------------",
    ]

    if not hierarchy.groups:
        lines.append("- NONE")
        return "\n".join(lines)

    for index, group in enumerate(
        hierarchy.groups,
        start=1,
    ):
        lines.append(
            f"- group_{index}: parent={group.parent_statement_id}"
        )
        for member in group.members:
            relation = (
                "exact"
                if member.exact_equivalence
                else "contained"
            )
            lines.append(
                "  constituent="
                f"{member.constituent_statement_id}; "
                f"family={member.family_id}; "
                f"family_kind={member.family_claim_kind}; "
                f"family_papers={','.join(member.family_paper_ids) or '-'}; "
                f"resolution={member.resolution_status}; "
                f"coverage_relation={relation}; "
                f"basis={member.resolution_basis}"
            )

    return "\n".join(lines)


class SynthesisConstituentPromptAugmenter:
    def __init__(
        self,
        hierarchy: SynthesisConstituentHierarchy,
    ) -> None:
        self.hierarchy = hierarchy

    def augment(
        self,
        prompt: HypothesisPrompt,
    ) -> HypothesisPrompt:
        guidance = render_synthesis_constituent_guidance(
            self.hierarchy
        )
        user_prompt = (
            prompt.user_prompt.rstrip()
            + "\n\n"
            + guidance
            + "\n"
        )
        canonical = _canonical_json(
            {
                "prompt_version": GUIDED_PROMPT_VERSION,
                "system_prompt": prompt.system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return HypothesisPrompt(
            prompt_version=GUIDED_PROMPT_VERSION,
            system_prompt=prompt.system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        )


class SynthesisConstituentGuidedBackend:
    """Backend adapter that augments every generate/repair prompt.

    The underlying Alpha4 planner/runtime/compiler/validator are untouched.
    """

    def __init__(
        self,
        base_backend: HypothesisDraftBackend,
        hierarchy: SynthesisConstituentHierarchy,
    ) -> None:
        self.base_backend = base_backend
        self.augmenter = SynthesisConstituentPromptAugmenter(
            hierarchy
        )
        self.backend_name = (
            f"{base_backend.backend_name}"
            "+synthesis_constituent_guidance"
        )
        self.model_name = base_backend.model_name

    def generate(
        self,
        prompt: HypothesisPrompt,
    ) -> HypothesisDraftGeneration:
        return self.base_backend.generate(
            self.augmenter.augment(prompt)
        )

    def repair(
        self,
        prompt: HypothesisPrompt,
        previous_draft,
        feedback: str,
    ) -> HypothesisDraftGeneration:
        return self.base_backend.repair(
            self.augmenter.augment(prompt),
            previous_draft,
            feedback,
        )


def audit_synthesis_constituent_selection(
    hierarchy: SynthesisConstituentHierarchy,
    portfolio: HypothesisPortfolio,
) -> SynthesisConstituentSelectionReport:
    if portfolio.source_context_id != hierarchy.source_context_id:
        raise ValueError(
            "EC2-D2 hierarchy/portfolio context ID mismatch"
        )
    if (
        portfolio.source_context_sha256
        != hierarchy.source_context_sha256
    ):
        raise ValueError(
            "EC2-D2 hierarchy/portfolio context SHA mismatch"
        )

    parent_ids = {
        group.parent_statement_id
        for group in hierarchy.groups
    }
    constituent_ids = {
        sid
        for group in hierarchy.groups
        for sid in group.constituent_statement_ids
    }

    cards: list[HypothesisConstituentSelectionCard] = []
    for hypothesis in portfolio.hypotheses:
        premises = set(hypothesis.premise_statement_ids)
        used_parents = sorted(premises & parent_ids)
        used_constituents = sorted(
            premises & constituent_ids
        )

        touched_groups = []
        redundant_groups = []

        for group in hierarchy.groups:
            parent_used = (
                group.parent_statement_id in premises
            )
            group_constituents = set(
                group.constituent_statement_ids
            )
            constituents_used = sorted(
                premises & group_constituents
            )

            if parent_used or constituents_used:
                touched_groups.append(
                    (
                        group,
                        parent_used,
                        constituents_used,
                    )
                )

            if (
                parent_used
                and group_constituent_ids_nonempty(group)
                and set(constituents_used)
                == group_constituents
            ):
                redundant_groups.append(
                    group.parent_statement_id
                )

        if not touched_groups:
            usage_class = "no_synthesis_constituent_usage"
        elif len(touched_groups) > 1:
            usage_class = "mixed_multiple_groups"
        else:
            group, parent_used, constituents_used = (
                touched_groups[0]
            )
            if (
                parent_used
                and group_constituent_ids_nonempty(group)
                and set(constituents_used)
                == set(group.constituent_statement_ids)
            ):
                usage_class = (
                    "parent_plus_all_constituents_potentially_redundant"
                )
            elif parent_used and constituents_used:
                usage_class = "parent_plus_subset_constituents"
            elif parent_used:
                usage_class = "parent_only"
            elif len(constituents_used) > 1:
                usage_class = (
                    "multiple_constituents_without_parent"
                )
            else:
                usage_class = "constituent_only"

        cards.append(
            HypothesisConstituentSelectionCard(
                hypothesis_id=hypothesis.hypothesis_id,
                title=hypothesis.title,
                premise_statement_ids=list(
                    hypothesis.premise_statement_ids
                ),
                used_parent_statement_ids=used_parents,
                used_constituent_statement_ids=used_constituents,
                usage_class=usage_class,
                potentially_redundant_parent_group_ids=sorted(
                    redundant_groups
                ),
            )
        )

    payload = {
        "schema_version": "synthesis-constituent-selection-report-v1",
        "source_hierarchy_id": hierarchy.hierarchy_id,
        "source_hierarchy_sha256": hierarchy.hierarchy_sha256,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_context_id": hierarchy.source_context_id,
        "source_context_sha256": hierarchy.source_context_sha256,
        "hypothesis_count": len(cards),
        "hypotheses_using_any_constituent_count": sum(
            bool(row.used_constituent_statement_ids)
            for row in cards
        ),
        "hypotheses_using_any_parent_count": sum(
            bool(row.used_parent_statement_ids)
            for row in cards
        ),
        "parent_premise_incidence_count": sum(
            len(row.used_parent_statement_ids)
            for row in cards
        ),
        "constituent_premise_incidence_count": sum(
            len(row.used_constituent_statement_ids)
            for row in cards
        ),
        "potential_parent_all_constituents_redundancy_count": sum(
            bool(row.potentially_redundant_parent_group_ids)
            for row in cards
        ),
        "cards": [
            row.model_dump(mode="json")
            for row in cards
        ],
        "policy": hierarchy.policy.model_dump(mode="json"),
    }
    payload["report_id"] = _stable_id(
        "synthesis_constituent_selection_report",
        hierarchy.hierarchy_sha256,
        portfolio.portfolio_id,
    )

    return SynthesisConstituentSelectionReport(
        **payload,
        report_sha256=_sha256_json(payload),
    )


def group_constituent_ids_nonempty(
    group: SynthesisConstituentGroup,
) -> bool:
    return bool(group.constituent_statement_ids)
