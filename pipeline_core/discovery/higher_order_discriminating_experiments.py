from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_competing_explanations import (
    CompetingExplanationPair,
    CompetingExplanationSet,
)
from pipeline_core.discovery.higher_order_tension_extractor import (
    ScientificTensionCandidate,
    ScientificTensionCandidateSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExperimentMode = Literal[
    "matched_modifier_intervention",
    "matched_joint_tradeoff_sweep",
    "matched_contrast_test",
]


class DiscriminatingPrediction(StrictModel):
    schema_version: Literal[
        "discriminating-prediction-v1"
    ] = "discriminating-prediction-v1"

    explanation_role: Literal["A", "B"]
    explanation_id: str
    predicted_pattern: str

    requires_verification: Literal[True] = True
    evidence_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class DiscriminatingExperimentCandidate(StrictModel):
    schema_version: Literal[
        "discriminating-experiment-candidate-v1"
    ] = "discriminating-experiment-candidate-v1"

    experiment_id: str
    tension_id: str
    pair_id: str
    tension_type: str
    experiment_mode: ExperimentMode

    requested_source: str
    requested_target: str

    intervention_or_sweep: str
    held_constant_conditions: list[str] = Field(min_length=1)
    observables: list[str] = Field(min_length=1)

    predictions: list[DiscriminatingPrediction] = Field(
        min_length=2,
        max_length=2,
    )
    decision_rule: str
    inconclusive_rule: str

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
    production_selection_authority: Literal[False] = False
    external_novelty_review_bypass_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


class DiscriminatingExperimentSet(StrictModel):
    schema_version: Literal[
        "discriminating-experiment-set-v1"
    ] = "discriminating-experiment-set-v1"

    pair_count: int = Field(ge=0)
    experiment_count: int = Field(ge=0)
    type_counts: dict[str, int] = Field(default_factory=dict)
    candidate_inspiration_experiment_count: int = Field(ge=0)
    experiments: list[DiscriminatingExperimentCandidate] = Field(
        default_factory=list
    )

    deterministic_generation: Literal[True] = True
    llm_generation_used: Literal[False] = False
    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    explanation_selection_performed: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    external_novelty_review_bypass_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _parse_relation(text: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r"\s*(.*?)\s+--([A-Z0-9_]+)-->\s+(.*?)\s*",
        str(text),
    )
    if match is None:
        return None
    return (
        match.group(1).strip(),
        match.group(2).strip(),
        match.group(3).strip(),
    )


def _tension_by_id(
    tensions: ScientificTensionCandidateSet,
) -> dict[str, ScientificTensionCandidate]:
    return {
        row.tension_id: row
        for row in tensions.candidates
    }


def _explanations(
    pair: CompetingExplanationPair,
) -> tuple[object, object]:
    by_role = {
        row.explanation_role: row
        for row in pair.explanations
    }
    if set(by_role) != {"A", "B"}:
        raise ValueError(
            "competing explanation pair requires exactly A and B"
        )
    return by_role["A"], by_role["B"]


def _prediction(
    explanation: object,
    pattern: str,
) -> DiscriminatingPrediction:
    return DiscriminatingPrediction(
        explanation_role=explanation.explanation_role,
        explanation_id=explanation.explanation_id,
        predicted_pattern=pattern,
    )


def _proxy_experiment(
    *,
    pair: CompetingExplanationPair,
    tension: ScientificTensionCandidate,
) -> DiscriminatingExperimentCandidate:
    a, b = _explanations(pair)
    modifier = (
        tension.modifier_text
        or "the modifier identified by the source tension"
    )
    anchor = (
        tension.modifier_anchor_text
        or "the recorded anchor response"
    )

    return DiscriminatingExperimentCandidate(
        experiment_id=_stable_id(
            "discriminating_experiment",
            pair.pair_id,
            "proxy_decoupling",
        ),
        tension_id=pair.tension_id,
        pair_id=pair.pair_id,
        tension_type=pair.tension_type,
        experiment_mode="matched_modifier_intervention",
        requested_source=pair.requested_source,
        requested_target=pair.requested_target,
        intervention_or_sweep=(
            f"Intervene on or stratify {modifier} while holding the task "
            "source and measurement conditions matched."
        ),
        held_constant_conditions=[
            pair.requested_source,
            "excitation or measurement conditions relevant to the task",
            "background composition and sampling conditions",
        ],
        observables=[
            pair.requested_target,
            anchor,
        ],
        predictions=[
            _prediction(
                a,
                (
                    f"{modifier} is associated with a reproducible change in "
                    f"{pair.requested_target}; {anchor} may covary, but the "
                    "target-state change remains independently measurable."
                ),
            ),
            _prediction(
                b,
                (
                    f"{anchor} changes with {modifier} while "
                    f"{pair.requested_target} remains stable within "
                    "measurement uncertainty."
                ),
            ),
        ],
        decision_rule=(
            "Favor neither explanation a priori. The observation separates "
            "the explanations only if the requested target and anchor are "
            "measured independently under the same modifier intervention: "
            "target-state change supports A-like behavior; anchor-only change "
            "supports B-like behavior."
        ),
        inconclusive_rule=(
            "Treat the test as inconclusive if target and anchor cannot be "
            "measured independently, if both remain unchanged, or if "
            "condition drift is large enough to explain the difference."
        ),
        source_arm_indices=list(tension.source_arm_indices),
        source_context_ids=list(tension.source_context_ids),
        basis_premise_ids=list(tension.basis_premise_ids),
        basis_relation_texts=list(tension.basis_relation_texts),
        candidate_inspiration_involved=(
            tension.candidate_inspiration_involved
        ),
    )


def _tradeoff_experiment(
    *,
    pair: CompetingExplanationPair,
    tension: ScientificTensionCandidate,
) -> DiscriminatingExperimentCandidate:
    a, b = _explanations(pair)
    relation = (
        _parse_relation(tension.basis_relation_texts[0])
        if tension.basis_relation_texts
        else None
    )
    left = relation[0] if relation else "tradeoff quantity A"
    right = relation[2] if relation else "tradeoff quantity B"

    return DiscriminatingExperimentCandidate(
        experiment_id=_stable_id(
            "discriminating_experiment",
            pair.pair_id,
            "tradeoff_pareto",
        ),
        tension_id=pair.tension_id,
        pair_id=pair.pair_id,
        tension_type=pair.tension_type,
        experiment_mode="matched_joint_tradeoff_sweep",
        requested_source=pair.requested_source,
        requested_target=pair.requested_target,
        intervention_or_sweep=(
            f"Perform a matched sweep over {pair.requested_source} or a "
            "task-relevant structural control while jointly measuring "
            f"{left} and {right}."
        ),
        held_constant_conditions=[
            "measurement definition and normalization",
            "excitation and environmental conditions",
            "sample comparison protocol",
        ],
        observables=[left, right],
        predictions=[
            _prediction(
                a,
                (
                    f"The negative coupling between {left} and {right} "
                    "persists across matched conditions."
                ),
            ),
            _prediction(
                b,
                (
                    f"The apparent negative coupling between {left} and "
                    f"{right} weakens, changes sign, or disappears after "
                    "conditions and measurement definitions are matched."
                ),
            ),
        ],
        decision_rule=(
            "A persistent negative relation after matched-condition controls "
            "is A-like; substantial attenuation or reversal after those "
            "controls is B-like."
        ),
        inconclusive_rule=(
            "Treat the test as inconclusive if the sweep does not span enough "
            "of both observables, if the quantities are not measured under "
            "comparable conditions, or if uncertainty is too large to "
            "distinguish persistence from attenuation."
        ),
        source_arm_indices=list(tension.source_arm_indices),
        source_context_ids=list(tension.source_context_ids),
        basis_premise_ids=list(tension.basis_premise_ids),
        basis_relation_texts=list(tension.basis_relation_texts),
        candidate_inspiration_involved=(
            tension.candidate_inspiration_involved
        ),
    )


def _contrast_experiment(
    *,
    pair: CompetingExplanationPair,
    tension: ScientificTensionCandidate,
) -> DiscriminatingExperimentCandidate:
    a, b = _explanations(pair)
    relation = (
        _parse_relation(tension.basis_relation_texts[0])
        if tension.basis_relation_texts
        else None
    )
    contrasted_factor = (
        relation[2]
        if relation is not None
        else "the contrasted factor"
    )

    return DiscriminatingExperimentCandidate(
        experiment_id=_stable_id(
            "discriminating_experiment",
            pair.pair_id,
            "contrasting_relations",
        ),
        tension_id=pair.tension_id,
        pair_id=pair.pair_id,
        tension_type=pair.tension_type,
        experiment_mode="matched_contrast_test",
        requested_source=pair.requested_source,
        requested_target=pair.requested_target,
        intervention_or_sweep=(
            f"Compare matched conditions while varying {contrasted_factor} "
            "and preserving the requested source relation."
        ),
        held_constant_conditions=[
            pair.requested_source,
            "measurement and normalization protocol",
            "background and environmental conditions",
        ],
        observables=[pair.requested_target],
        predictions=[
            _prediction(
                a,
                (
                    f"The requested target relation differs reproducibly when "
                    f"{contrasted_factor} is varied under matched context."
                ),
            ),
            _prediction(
                b,
                (
                    f"The apparent contrast associated with "
                    f"{contrasted_factor} shrinks or disappears once context "
                    "and measurement conditions are matched."
                ),
            ),
        ],
        decision_rule=(
            "A reproducible residual contrast under matched conditions is "
            "A-like; disappearance of the contrast after matching is B-like."
        ),
        inconclusive_rule=(
            "Treat the test as inconclusive if the contrasted factor cannot "
            "be isolated from other condition changes or if the requested "
            "target is not measured consistently across conditions."
        ),
        source_arm_indices=list(tension.source_arm_indices),
        source_context_ids=list(tension.source_context_ids),
        basis_premise_ids=list(tension.basis_premise_ids),
        basis_relation_texts=list(tension.basis_relation_texts),
        candidate_inspiration_involved=(
            tension.candidate_inspiration_involved
        ),
    )


def generate_discriminating_experiments(
    *,
    explanations: CompetingExplanationSet,
    tensions: ScientificTensionCandidateSet,
) -> DiscriminatingExperimentSet:
    tension_lookup = _tension_by_id(tensions)
    rows = []

    for pair in explanations.pairs:
        tension = tension_lookup.get(pair.tension_id)
        if tension is None:
            raise ValueError(
                "missing tension for competing explanation pair: "
                + pair.tension_id
            )

        if pair.tension_type == "proxy_decoupling":
            rows.append(
                _proxy_experiment(pair=pair, tension=tension)
            )
        elif pair.tension_type == "tradeoff_pareto":
            rows.append(
                _tradeoff_experiment(pair=pair, tension=tension)
            )
        elif pair.tension_type == "contrasting_relations":
            rows.append(
                _contrast_experiment(pair=pair, tension=tension)
            )

    counts = Counter(row.tension_type for row in rows)

    return DiscriminatingExperimentSet(
        pair_count=explanations.pair_count,
        experiment_count=len(rows),
        type_counts=dict(sorted(counts.items())),
        candidate_inspiration_experiment_count=sum(
            row.candidate_inspiration_involved
            for row in rows
        ),
        experiments=rows,
    )
