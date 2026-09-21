from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_tension_extractor import (
    ScientificTensionCandidate,
    ScientificTensionCandidateSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CompetingExplanationType = Literal[
    "target_state_change",
    "anchor_or_proxy_only_change",
    "intrinsic_tradeoff",
    "condition_or_proxy_dependent_tradeoff",
    "modifier_specific_contrast",
    "context_or_measurement_specific_contrast",
]


class CompetingExplanation(StrictModel):
    schema_version: Literal[
        "competing-explanation-v1"
    ] = "competing-explanation-v1"

    explanation_id: str
    tension_id: str
    tension_type: str
    explanation_type: CompetingExplanationType
    explanation_role: Literal["A", "B"]

    explanation_statement: str
    discriminator_requirement: str

    source_arm_indices: list[int] = Field(min_length=1)
    source_context_ids: list[str] = Field(min_length=1)
    basis_premise_ids: list[str] = Field(default_factory=list)
    basis_relation_texts: list[str] = Field(default_factory=list)

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
    discriminating_hypothesis_generation_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


class CompetingExplanationPair(StrictModel):
    schema_version: Literal[
        "competing-explanation-pair-v1"
    ] = "competing-explanation-pair-v1"

    pair_id: str
    tension_id: str
    tension_type: str
    requested_source: str
    requested_target: str
    tension_statement: str

    explanations: list[CompetingExplanation] = Field(
        min_length=2,
        max_length=2,
    )

    candidate_inspiration_involved: bool = False
    requires_verification: Literal[True] = True
    epistemic_status: Literal[
        "inspiration_only"
    ] = "inspiration_only"

    explanation_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    discriminating_hypothesis_generation_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


class CompetingExplanationSet(StrictModel):
    schema_version: Literal[
        "competing-explanation-set-v1"
    ] = "competing-explanation-set-v1"

    tension_count: int = Field(ge=0)
    pair_count: int = Field(ge=0)
    explanation_count: int = Field(ge=0)
    type_counts: dict[str, int] = Field(default_factory=dict)
    candidate_inspiration_pair_count: int = Field(ge=0)
    pairs: list[CompetingExplanationPair] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    explanation_selection_performed: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    discriminating_hypothesis_generation_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _explanation(
    *,
    tension: ScientificTensionCandidate,
    role: Literal["A", "B"],
    explanation_type: CompetingExplanationType,
    statement: str,
    discriminator: str,
) -> CompetingExplanation:
    return CompetingExplanation(
        explanation_id=_stable_id(
            "competing_explanation",
            tension.tension_id,
            role,
            explanation_type,
        ),
        tension_id=tension.tension_id,
        tension_type=tension.tension_type,
        explanation_type=explanation_type,
        explanation_role=role,
        explanation_statement=statement,
        discriminator_requirement=discriminator,
        source_arm_indices=list(tension.source_arm_indices),
        source_context_ids=list(tension.source_context_ids),
        basis_premise_ids=list(tension.basis_premise_ids),
        basis_relation_texts=list(tension.basis_relation_texts),
        candidate_inspiration_involved=(
            tension.candidate_inspiration_involved
        ),
    )


def _proxy_decoupling_pair(
    tension: ScientificTensionCandidate,
) -> CompetingExplanationPair:
    basis = (
        tension.basis_relation_texts[0]
        if tension.basis_relation_texts
        else "the recorded anchor relation"
    )

    a = _explanation(
        tension=tension,
        role="A",
        explanation_type="target_state_change",
        statement=(
            "The modifier may change the requested target state itself, "
            "with the recorded anchor relation acting as a correlated or "
            "downstream observable rather than the sole affected level."
        ),
        discriminator=(
            "Under a matched intervention on the modifier, independently "
            f"measure {tension.requested_target} and the anchor represented "
            f"by {basis}. Explanation A is supported only if the requested "
            "target changes in a reproducible way that cannot be reduced to "
            "an anchor-only measurement change."
        ),
    )

    b = _explanation(
        tension=tension,
        role="B",
        explanation_type="anchor_or_proxy_only_change",
        statement=(
            "The modifier may change only the recorded anchor/proxy response "
            "while the requested target state remains materially unchanged."
        ),
        discriminator=(
            "Under the same matched modifier intervention, independently "
            f"measure {tension.requested_target} and the anchor represented "
            f"by {basis}. Explanation B is supported if the anchor changes "
            "while the requested target remains stable within measurement "
            "uncertainty."
        ),
    )

    return _pair(tension, a, b)


def _tradeoff_pair(
    tension: ScientificTensionCandidate,
) -> CompetingExplanationPair:
    a = _explanation(
        tension=tension,
        role="A",
        explanation_type="intrinsic_tradeoff",
        statement=(
            "The recorded tradeoff may be intrinsic to the physical relation, "
            "so improving one reported quantity systematically degrades the "
            "other under otherwise matched conditions."
        ),
        discriminator=(
            "Perform a matched-condition sweep spanning both quantities in "
            "the recorded tradeoff and test whether the negative coupling "
            "persists after controlling experimental conditions and "
            "measurement definitions."
        ),
    )

    b = _explanation(
        tension=tension,
        role="B",
        explanation_type="condition_or_proxy_dependent_tradeoff",
        statement=(
            "The apparent tradeoff may depend on condition, proxy, or "
            "measurement differences and may weaken or disappear when those "
            "factors are matched."
        ),
        discriminator=(
            "Repeat the same joint sweep with matched conditions, observables, "
            "and normalization. Explanation B is supported if the apparent "
            "tradeoff substantially weakens, changes sign, or disappears."
        ),
    )

    return _pair(tension, a, b)


def _contrast_pair(
    tension: ScientificTensionCandidate,
) -> CompetingExplanationPair:
    a = _explanation(
        tension=tension,
        role="A",
        explanation_type="modifier_specific_contrast",
        statement=(
            "The recorded contrast may reflect a real modifier-specific "
            "difference that changes the requested source-target relationship."
        ),
        discriminator=(
            "Compare the contrasted conditions under matched task source, "
            "measurement, and background conditions while varying only the "
            "contrasted factor. Explanation A requires a reproducible "
            "difference in the requested target relation."
        ),
    )

    b = _explanation(
        tension=tension,
        role="B",
        explanation_type="context_or_measurement_specific_contrast",
        statement=(
            "The recorded contrast may arise from differing context or "
            "measurement conditions rather than a modifier-specific change "
            "in the requested source-target relationship."
        ),
        discriminator=(
            "Repeat the comparison after matching context and measurement "
            "conditions. Explanation B is supported if the contrast shrinks "
            "or disappears without changing the contrasted factor itself."
        ),
    )

    return _pair(tension, a, b)


def _pair(
    tension: ScientificTensionCandidate,
    a: CompetingExplanation,
    b: CompetingExplanation,
) -> CompetingExplanationPair:
    return CompetingExplanationPair(
        pair_id=_stable_id(
            "competing_explanation_pair",
            tension.tension_id,
            a.explanation_id,
            b.explanation_id,
        ),
        tension_id=tension.tension_id,
        tension_type=tension.tension_type,
        requested_source=tension.requested_source,
        requested_target=tension.requested_target,
        tension_statement=tension.tension_statement,
        explanations=[a, b],
        candidate_inspiration_involved=(
            tension.candidate_inspiration_involved
        ),
    )


def generate_competing_explanations(
    tensions: ScientificTensionCandidateSet,
) -> CompetingExplanationSet:
    pairs = []

    for tension in tensions.candidates:
        if tension.tension_type == "proxy_decoupling":
            pairs.append(_proxy_decoupling_pair(tension))
        elif tension.tension_type == "tradeoff_pareto":
            pairs.append(_tradeoff_pair(tension))
        elif tension.tension_type == "contrasting_relations":
            pairs.append(_contrast_pair(tension))
        else:
            continue

    counts = Counter(pair.tension_type for pair in pairs)

    return CompetingExplanationSet(
        tension_count=tensions.candidate_count,
        pair_count=len(pairs),
        explanation_count=sum(
            len(pair.explanations)
            for pair in pairs
        ),
        type_counts=dict(sorted(counts.items())),
        candidate_inspiration_pair_count=sum(
            pair.candidate_inspiration_involved
            for pair in pairs
        ),
        pairs=pairs,
    )
