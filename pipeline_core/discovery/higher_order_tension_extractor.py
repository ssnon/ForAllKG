from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_hypothesis_batch import (
    HigherOrderShadowBatchOutcome,
)
from pipeline_core.discovery.higher_order_semantic_critic import (
    HigherOrderBatchSemanticCritique,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TensionType = Literal[
    "proxy_decoupling",
    "tradeoff_pareto",
    "contrasting_relations",
]


class ScientificTensionCandidate(StrictModel):
    schema_version: Literal[
        "scientific-tension-candidate-v1"
    ] = "scientific-tension-candidate-v1"

    tension_id: str
    tension_type: TensionType

    source_arm_indices: list[int] = Field(min_length=1)
    source_context_ids: list[str] = Field(min_length=1)
    basis_premise_ids: list[str] = Field(default_factory=list)
    basis_relation_texts: list[str] = Field(default_factory=list)
    trigger_codes: list[str] = Field(default_factory=list)

    requested_source: str
    requested_target: str
    tension_statement: str
    rationale: str

    modifier_text: str | None = None
    modifier_anchor_text: str | None = None

    candidate_inspiration_involved: bool = False
    requires_verification: Literal[True] = True
    epistemic_status: Literal[
        "inspiration_only"
    ] = "inspiration_only"

    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False

    competing_explanation_generation_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


class ScientificTensionCandidateSet(StrictModel):
    schema_version: Literal[
        "scientific-tension-candidate-set-v1"
    ] = "scientific-tension-candidate-set-v1"

    candidate_count: int = Field(ge=0)
    type_counts: dict[str, int] = Field(default_factory=dict)
    candidate_inspiration_candidate_count: int = Field(ge=0)
    candidates: list[ScientificTensionCandidate] = Field(
        default_factory=list
    )

    extraction_scope: Literal[
        "higher_order_shadow_post_generation"
    ] = "higher_order_shadow_post_generation"

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    competing_explanation_generation_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _authority_value(value: object) -> str:
    return str(getattr(value, "value", value)).strip()


def _relation_text(premise: object) -> str:
    return (
        f"{getattr(premise, 'subject', '')} "
        f"--{getattr(premise, 'relation', '')}--> "
        f"{getattr(premise, 'object', '')}"
    ).strip()


def _premise_by_role(context: object, role: str) -> object | None:
    for premise in getattr(context, "premises", ()):
        if getattr(premise, "premise_role", None) == role:
            return premise
    return None


def _candidate_inspiration_involved(context: object) -> bool:
    return any(
        _authority_value(getattr(premise, "authority", ""))
        == "candidate_inspiration"
        for premise in getattr(context, "premises", ())
    )


def _proxy_decoupling_candidates(
    *,
    outcome: HigherOrderShadowBatchOutcome,
    critique: HigherOrderBatchSemanticCritique,
) -> list[ScientificTensionCandidate]:
    rows = []
    critique_by_arm = {
        row.arm_index: row
        for row in critique.arms
    }

    for arm_index, arm in enumerate(outcome.arms, start=1):
        critic_row = critique_by_arm.get(arm_index)
        if critic_row is None:
            continue

        if not any(
            issue.code == "MODIFIER_ANCHOR_BRIDGE_OMITTED"
            for issue in critic_row.issues
        ):
            continue

        context = arm.higher_order_context
        modifier = _premise_by_role(context, "modifier_relation")
        if modifier is None:
            continue

        opportunity = context.structural_opportunity
        modifier_text = str(opportunity.modifier_text).strip()
        anchor_text = str(opportunity.modifier_anchor_text).strip()

        if not modifier_text or not anchor_text:
            continue

        rows.append(
            ScientificTensionCandidate(
                tension_id=_stable_id(
                    "scientific_tension",
                    "proxy_decoupling",
                    context.context_id,
                    modifier_text,
                    anchor_text,
                ),
                tension_type="proxy_decoupling",
                source_arm_indices=[arm_index],
                source_context_ids=[context.context_id],
                basis_premise_ids=[modifier.premise_id],
                basis_relation_texts=[_relation_text(modifier)],
                trigger_codes=[
                    "MODIFIER_ANCHOR_BRIDGE_OMITTED",
                ],
                requested_source=context.requested_source,
                requested_target=context.requested_target,
                tension_statement=(
                    "It remains unresolved whether "
                    f"{modifier_text} changes the requested target relation "
                    f"itself or instead changes/reflects {anchor_text} "
                    "without the supplied evidence establishing the bridge "
                    "between those levels."
                ),
                rationale=(
                    "The generated hypothesis used the modifier in the "
                    "requested source-target claim while omitting the "
                    "modifier's validated anchor from the full generated "
                    "card. This supports a proxy-decoupling tension only, "
                    "not a causal conclusion."
                ),
                modifier_text=modifier_text,
                modifier_anchor_text=anchor_text,
                candidate_inspiration_involved=(
                    _candidate_inspiration_involved(context)
                ),
            )
        )

    return rows


def _relational_signal_candidates(
    *,
    outcome: HigherOrderShadowBatchOutcome,
) -> list[ScientificTensionCandidate]:
    rows = []

    for arm_index, arm in enumerate(outcome.arms, start=1):
        context = arm.higher_order_context

        for premise in context.premises:
            relation = str(premise.relation).strip().upper()

            if relation == "IMPOSES_TRADEOFF":
                tension_type: TensionType = "tradeoff_pareto"
                trigger = "EXPLICIT_TRADEOFF_RELATION"
                statement = (
                    "The recorded relation between "
                    f"{premise.subject} and {premise.object} indicates that "
                    "the requested source-target behavior may involve a "
                    "tradeoff rather than monotonic co-optimization."
                )
                rationale = (
                    "A recorded IMPOSES_TRADEOFF predicate is an explicit "
                    "structural signal for a tradeoff/Pareto tension. The "
                    "candidate remains inspiration-only and does not assert "
                    "where the Pareto frontier lies."
                )
            elif relation == "CONTRASTS_WITH":
                tension_type = "contrasting_relations"
                trigger = "EXPLICIT_CONTRAST_RELATION"
                statement = (
                    "The recorded contrast between "
                    f"{premise.subject} and {premise.object} may distinguish "
                    "competing explanations of the requested source-target "
                    "relationship."
                )
                rationale = (
                    "A recorded CONTRASTS_WITH predicate supplies an explicit "
                    "contrast signal. This does not establish conflicting "
                    "literature or choose between explanations."
                )
            else:
                continue

            rows.append(
                ScientificTensionCandidate(
                    tension_id=_stable_id(
                        "scientific_tension",
                        tension_type,
                        context.context_id,
                        premise.premise_id,
                    ),
                    tension_type=tension_type,
                    source_arm_indices=[arm_index],
                    source_context_ids=[context.context_id],
                    basis_premise_ids=[premise.premise_id],
                    basis_relation_texts=[_relation_text(premise)],
                    trigger_codes=[trigger],
                    requested_source=context.requested_source,
                    requested_target=context.requested_target,
                    tension_statement=statement,
                    rationale=rationale,
                    candidate_inspiration_involved=(
                        _candidate_inspiration_involved(context)
                    ),
                )
            )

    return rows


def _dedupe(
    rows: list[ScientificTensionCandidate],
) -> list[ScientificTensionCandidate]:
    """
    Dedupe only exact scientific signals.

    Identical tension type + identical basis relation text + identical
    requested task collapse across multiple backbone realizations. Source
    arm/context provenance is merged losslessly.
    """
    by_key: dict[
        tuple[str, tuple[str, ...], str, str],
        ScientificTensionCandidate,
    ] = {}

    for row in rows:
        key = (
            row.tension_type,
            tuple(sorted(row.basis_relation_texts)),
            row.requested_source,
            row.requested_target,
        )
        previous = by_key.get(key)

        if previous is None:
            by_key[key] = row
            continue

        merged_arms = sorted(
            set(previous.source_arm_indices)
            | set(row.source_arm_indices)
        )
        merged_contexts = sorted(
            set(previous.source_context_ids)
            | set(row.source_context_ids)
        )
        merged_premises = sorted(
            set(previous.basis_premise_ids)
            | set(row.basis_premise_ids)
        )
        merged_triggers = sorted(
            set(previous.trigger_codes)
            | set(row.trigger_codes)
        )

        by_key[key] = previous.model_copy(
            update={
                "source_arm_indices": merged_arms,
                "source_context_ids": merged_contexts,
                "basis_premise_ids": merged_premises,
                "trigger_codes": merged_triggers,
                "candidate_inspiration_involved": (
                    previous.candidate_inspiration_involved
                    or row.candidate_inspiration_involved
                ),
            }
        )

    return sorted(
        by_key.values(),
        key=lambda row: (
            row.tension_type,
            row.tension_id,
        ),
    )


def extract_scientific_tension_candidates(
    *,
    outcome: HigherOrderShadowBatchOutcome,
    semantic_critique: HigherOrderBatchSemanticCritique,
) -> ScientificTensionCandidateSet:
    rows = [
        *_proxy_decoupling_candidates(
            outcome=outcome,
            critique=semantic_critique,
        ),
        *_relational_signal_candidates(
            outcome=outcome,
        ),
    ]
    rows = _dedupe(rows)

    counts = Counter(row.tension_type for row in rows)

    return ScientificTensionCandidateSet(
        candidate_count=len(rows),
        type_counts=dict(sorted(counts.items())),
        candidate_inspiration_candidate_count=sum(
            row.candidate_inspiration_involved
            for row in rows
        ),
        candidates=rows,
    )
