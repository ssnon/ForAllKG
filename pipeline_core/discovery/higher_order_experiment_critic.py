from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_discriminating_experiments import (
    DiscriminatingExperimentCandidate,
    DiscriminatingExperimentSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExperimentCriticIssueCode = Literal[
    "PREDICTION_SEPARATION_UNRESOLVED",
    "OBSERVABLE_SET_DEGENERATE",
    "INTERVENTION_UNDERSPECIFIED",
    "MEASUREMENT_INDEPENDENCE_UNRESOLVED",
    "BASIS_ALIGNMENT_UNRESOLVED",
]


class ExperimentCriticIssue(StrictModel):
    code: ExperimentCriticIssueCode
    message: str
    metrics: dict[str, float] = Field(default_factory=dict)


class DiscriminatingExperimentCritique(StrictModel):
    experiment_id: str
    tension_id: str
    tension_type: str
    experiment_mode: str
    candidate_inspiration_involved: bool

    prediction_jaccard: float
    observable_pair_max_containment: float
    intervention_basis_alignment: float

    issues: list[ExperimentCriticIssue] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    diagnostic_only: Literal[True] = True
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class DiscriminatingExperimentCriticReport(StrictModel):
    schema_version: Literal[
        "discriminating-experiment-critic-v1"
    ] = "discriminating-experiment-critic-v1"

    experiment_count: int = Field(ge=0)
    flagged_experiment_count: int = Field(ge=0)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    candidate_issue_counts: dict[str, int] = Field(default_factory=dict)
    experiments: list[DiscriminatingExperimentCritique] = Field(
        default_factory=list
    )

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    experiment_selection_performed: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    external_novelty_review_bypass_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


_DASHES = "‐-‒–—−"
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can",
    "do", "does", "for", "from", "has", "have", "how", "in", "into", "is",
    "may", "of", "on", "or", "such", "than", "that", "the", "their", "to",
    "under", "with", "while",
}
_GENERIC_INTERVENTION_PHRASES = (
    "or a task-relevant structural control",
    "or a task relevant structural control",
    "or another task-relevant structural control",
    "or another task relevant structural control",
)


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    for dash in _DASHES:
        value = value.replace(dash, "-")
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: str) -> set[str]:
    raw = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        _normalize(text),
    )
    out: set[str] = set()
    for token in raw:
        parts = [part for part in token.split("-") if part]
        candidates = list(parts)
        if len(parts) > 1:
            candidates.append("".join(parts))
        for candidate in candidates:
            if len(candidate) > 1 and candidate not in _STOPWORDS:
                out.add(candidate)
    return out


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def _containment(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return float(len(a & b) / min(len(a), len(b)))


def _coverage(phrase: str, text: str) -> float:
    required = _tokens(phrase)
    if not required:
        return 0.0
    return float(len(required & _tokens(text)) / len(required))


def _max_observable_containment(observables: list[str]) -> float:
    if len(observables) < 2:
        return 0.0

    values = []
    for i in range(len(observables)):
        for j in range(i + 1, len(observables)):
            values.append(
                _containment(
                    observables[i],
                    observables[j],
                )
            )
    return max(values, default=0.0)


def _basis_terms(experiment: DiscriminatingExperimentCandidate) -> list[str]:
    terms = []
    for relation in experiment.basis_relation_texts:
        match = re.fullmatch(
            r"\s*(.*?)\s+--[A-Z0-9_]+-->\s+(.*?)\s*",
            str(relation),
        )
        if match is None:
            continue
        terms.extend(
            [
                match.group(1).strip(),
                match.group(2).strip(),
            ]
        )
    return terms


def _basis_alignment(
    experiment: DiscriminatingExperimentCandidate,
) -> float:
    """
    Maximum coverage of any basis-relation endpoint by intervention text.

    Tradeoff sweeps are allowed to align through requested_source even when
    neither recorded tradeoff endpoint is itself the swept control.
    """
    candidates = _basis_terms(experiment)

    if experiment.experiment_mode == "matched_joint_tradeoff_sweep":
        candidates.append(experiment.requested_source)

    if not candidates:
        return 0.0

    return max(
        _coverage(
            candidate,
            experiment.intervention_or_sweep,
        )
        for candidate in candidates
    )


def _prediction_separation(
    experiment: DiscriminatingExperimentCandidate,
) -> float:
    by_role = {
        row.explanation_role: row.predicted_pattern
        for row in experiment.predictions
    }
    if set(by_role) != {"A", "B"}:
        return 1.0
    return _jaccard(by_role["A"], by_role["B"])


def critique_discriminating_experiment(
    experiment: DiscriminatingExperimentCandidate,
) -> DiscriminatingExperimentCritique:
    prediction_jaccard = _prediction_separation(experiment)
    observable_pair_max_containment = _max_observable_containment(
        experiment.observables
    )
    intervention_basis_alignment = _basis_alignment(experiment)

    issues: list[ExperimentCriticIssue] = []

    roles = {
        row.explanation_role
        for row in experiment.predictions
    }

    if (
        roles != {"A", "B"}
        or len(experiment.predictions) != 2
        or prediction_jaccard >= 0.80
    ):
        issues.append(
            ExperimentCriticIssue(
                code="PREDICTION_SEPARATION_UNRESOLVED",
                message=(
                    "The A/B prediction pair is not clearly separable from "
                    "its structured form and lexical content. The experiment "
                    "should not be treated as discriminating until distinct "
                    "outcome patterns are explicit."
                ),
                metrics={
                    "prediction_jaccard": prediction_jaccard,
                },
            )
        )

    normalized_observables = {
        _normalize(value)
        for value in experiment.observables
        if _normalize(value)
    }
    required_observable_count = (
        2
        if experiment.experiment_mode
        in {
            "matched_modifier_intervention",
            "matched_joint_tradeoff_sweep",
        }
        else 1
    )

    if (
        len(normalized_observables) < required_observable_count
        or observable_pair_max_containment >= 0.90
    ):
        issues.append(
            ExperimentCriticIssue(
                code="OBSERVABLE_SET_DEGENERATE",
                message=(
                    "The observable set is too small or lexically redundant "
                    "for the intended discrimination mode."
                ),
                metrics={
                    "unique_observable_count":
                        float(len(normalized_observables)),
                    "required_observable_count":
                        float(required_observable_count),
                    "observable_pair_max_containment":
                        observable_pair_max_containment,
                },
            )
        )

    normalized_intervention = _normalize(
        experiment.intervention_or_sweep
    )
    if any(
        phrase in normalized_intervention
        for phrase in _GENERIC_INTERVENTION_PHRASES
    ):
        issues.append(
            ExperimentCriticIssue(
                code="INTERVENTION_UNDERSPECIFIED",
                message=(
                    "The intervention/sweep contains an unresolved alternative "
                    "control rather than one identified variable or explicit "
                    "set of variables."
                ),
                metrics={
                    "intervention_basis_alignment":
                        intervention_basis_alignment,
                },
            )
        )

    if (
        experiment.experiment_mode == "matched_modifier_intervention"
        and len(normalized_observables) >= 2
    ):
        issues.append(
            ExperimentCriticIssue(
                code="MEASUREMENT_INDEPENDENCE_UNRESOLVED",
                message=(
                    "The candidate requires target and anchor observables to "
                    "be measured independently, but no measurement-method or "
                    "independence witness is encoded in the experiment "
                    "artifact. This is an instrumentation/identifiability "
                    "requirement, not a rejection."
                ),
                metrics={
                    "observable_pair_max_containment":
                        observable_pair_max_containment,
                },
            )
        )

    if intervention_basis_alignment < 0.50:
        issues.append(
            ExperimentCriticIssue(
                code="BASIS_ALIGNMENT_UNRESOLVED",
                message=(
                    "The intervention/sweep is not sufficiently aligned with "
                    "an endpoint of the recorded basis relation (or the "
                    "requested source for tradeoff sweeps)."
                ),
                metrics={
                    "intervention_basis_alignment":
                        intervention_basis_alignment,
                },
            )
        )

    return DiscriminatingExperimentCritique(
        experiment_id=experiment.experiment_id,
        tension_id=experiment.tension_id,
        tension_type=experiment.tension_type,
        experiment_mode=experiment.experiment_mode,
        candidate_inspiration_involved=(
            experiment.candidate_inspiration_involved
        ),
        prediction_jaccard=prediction_jaccard,
        observable_pair_max_containment=(
            observable_pair_max_containment
        ),
        intervention_basis_alignment=intervention_basis_alignment,
        issues=issues,
    )


def critique_discriminating_experiments(
    experiments: DiscriminatingExperimentSet,
) -> DiscriminatingExperimentCriticReport:
    rows = [
        critique_discriminating_experiment(experiment)
        for experiment in experiments.experiments
    ]

    issue_counts = Counter(
        issue.code
        for row in rows
        for issue in row.issues
    )
    candidate_issue_counts = Counter(
        issue.code
        for row in rows
        if row.candidate_inspiration_involved
        for issue in row.issues
    )

    return DiscriminatingExperimentCriticReport(
        experiment_count=len(rows),
        flagged_experiment_count=sum(
            bool(row.issues)
            for row in rows
        ),
        issue_counts=dict(sorted(issue_counts.items())),
        candidate_issue_counts=dict(
            sorted(candidate_issue_counts.items())
        ),
        experiments=rows,
    )
