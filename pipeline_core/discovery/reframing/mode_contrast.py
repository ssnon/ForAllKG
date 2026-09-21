from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
    ScientificContradictionResolutionCandidate,
)
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeRunReport,
    ScientificProxyChallengeCandidate,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RepresentationTransform = Literal[
    "introduces_hidden_construct",
    "partitions_response_law",
    "challenges_measurement_equivalence",
    "reconciles_apparently_incompatible_evidence",
]

ModeRelation = Literal[
    "same_representation_family",
    "distinct_operator_surface_near_duplicate",
    "distinct_mode_shared_scientific_neighborhood",
    "distinct_mode_low_observed_overlap",
]


_GENERIC_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "can",
    "could", "for", "from", "has", "have", "if", "in", "into", "is", "it",
    "may", "might", "of", "on", "or", "should", "than", "that", "the", "their",
    "then", "this", "to", "under", "when", "where", "which", "while", "with",
    "across", "between", "through", "using", "used", "use", "measured",
    "measurement", "response", "result", "results", "reported", "model",
    "alternative", "baseline", "expected", "observation", "observations",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", text.lower())
        if len(token) >= 3 and token not in _GENERIC_STOP
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _id_jaccard(left: list[str], right: list[str]) -> float:
    return _jaccard(set(left), set(right))


def _aggregate_tokens(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.update(_tokens(value))
    return result


def _transform_for_operator(operator_id: str) -> RepresentationTransform:
    if operator_id == "LATENT_VARIABLE":
        return "introduces_hidden_construct"
    if operator_id == "REGIME_BOUNDARY":
        return "partitions_response_law"
    if operator_id == "PROXY_CHALLENGE":
        return "challenges_measurement_equivalence"
    if operator_id == "CONTRADICTION_RESOLUTION":
        return "reconciles_apparently_incompatible_evidence"
    raise ValueError(f"unsupported reasoning operator: {operator_id}")


class ReasoningModeCandidateProfile(StrictModel):
    candidate_id: str
    operator_id: str
    representation_transform: RepresentationTransform
    premise_statement_ids: list[str]
    gap_statement_ids: list[str]
    challenge_text: str
    baseline_summary: str
    alternative_summary: str
    construct_texts: list[str] = Field(default_factory=list)
    prediction_observables: list[str] = Field(default_factory=list)
    test_observables: list[str] = Field(default_factory=list)


class PairwiseReasoningModeContrast(StrictModel):
    candidate_a_id: str
    operator_a: str
    transform_a: RepresentationTransform
    candidate_b_id: str
    operator_b: str
    transform_b: RepresentationTransform

    representation_transform_distinct: bool
    shared_premise_count: int = Field(ge=0)
    premise_jaccard: float = Field(ge=0.0, le=1.0)
    gap_jaccard: float = Field(ge=0.0, le=1.0)
    challenge_token_jaccard: float = Field(ge=0.0, le=1.0)
    baseline_summary_token_jaccard: float = Field(ge=0.0, le=1.0)
    alternative_summary_token_jaccard: float = Field(ge=0.0, le=1.0)
    prediction_observable_token_jaccard: float = Field(ge=0.0, le=1.0)
    test_observable_token_jaccard: float = Field(ge=0.0, le=1.0)
    construct_token_jaccard: float = Field(ge=0.0, le=1.0)

    shared_scientific_neighborhood_signals: list[str] = Field(default_factory=list)
    surface_near_duplicate_signals: list[str] = Field(default_factory=list)
    relation: ModeRelation

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    portfolio_selection_authority: Literal[False] = False


class ReasoningModeContrastReport(StrictModel):
    schema_version: Literal[
        "scientific-reframing-mode-contrast-v1"
    ] = "scientific-reframing-mode-contrast-v1"
    report_id: str
    source_task_id: str
    candidate_profiles: list[ReasoningModeCandidateProfile]
    pairwise_contrasts: list[PairwiseReasoningModeContrast]
    relation_counts: dict[str, int]
    llm_calls_performed: Literal[0] = 0

    diagnostic_only: Literal[True] = True
    reasoning_mode_structure_compared: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    candidate_winner_selected: Literal[False] = False
    portfolio_selection_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ReasoningModeContrastReport":
        expected = Counter(row.relation for row in self.pairwise_contrasts)
        if dict(sorted(expected.items())) != dict(sorted(self.relation_counts.items())):
            raise ValueError("relation_counts must match pairwise contrasts")
        return self


def _profile_reframe(candidate: ScientificReframeCandidate) -> ReasoningModeCandidateProfile:
    constructs = list(candidate.proposed_constructs)
    if candidate.operator_id == "LATENT_VARIABLE":
        constructs.extend(candidate.latent_constructs)
    elif candidate.operator_id == "REGIME_BOUNDARY":
        constructs.extend(candidate.boundary_variables)
    return ReasoningModeCandidateProfile(
        candidate_id=candidate.reframe_id,
        operator_id=candidate.operator_id,
        representation_transform=_transform_for_operator(candidate.operator_id),
        premise_statement_ids=list(candidate.premise_statement_ids),
        gap_statement_ids=list(candidate.gap_statement_ids),
        challenge_text=candidate.challenged_assumption,
        baseline_summary=candidate.baseline_model.summary,
        alternative_summary=candidate.alternative_model.summary,
        construct_texts=constructs,
        prediction_observables=[row.observable for row in candidate.differential_predictions],
        test_observables=list(candidate.discriminating_test.primary_observables),
    )


def _profile_proxy(candidate: ScientificProxyChallengeCandidate) -> ReasoningModeCandidateProfile:
    return ReasoningModeCandidateProfile(
        candidate_id=candidate.candidate_id,
        operator_id=candidate.operator_id,
        representation_transform=_transform_for_operator(candidate.operator_id),
        premise_statement_ids=list(candidate.premise_statement_ids),
        gap_statement_ids=list(candidate.gap_statement_ids),
        challenge_text=candidate.challenged_proxy_assumption,
        baseline_summary=candidate.baseline_model.summary,
        alternative_summary=candidate.alternative_model.summary,
        construct_texts=[candidate.challenged_observable, candidate.target_construct],
        prediction_observables=[row.observable for row in candidate.differential_predictions],
        test_observables=list(candidate.discriminating_test.primary_observables),
    )


def _profile_contradiction(
    candidate: ScientificContradictionResolutionCandidate,
) -> ReasoningModeCandidateProfile:
    constructs = [candidate.resolution_principle]
    constructs.extend(candidate.distinguishing_context_variables)
    constructs.extend(candidate.proposed_resolution_constructs)
    return ReasoningModeCandidateProfile(
        candidate_id=candidate.candidate_id,
        operator_id=candidate.operator_id,
        representation_transform=_transform_for_operator(candidate.operator_id),
        premise_statement_ids=list(candidate.premise_statement_ids),
        gap_statement_ids=list(candidate.gap_statement_ids),
        challenge_text=candidate.apparent_contradiction,
        baseline_summary=candidate.baseline_model.summary,
        alternative_summary=candidate.resolution_model.summary,
        construct_texts=constructs,
        prediction_observables=[row.observable for row in candidate.differential_predictions],
        test_observables=list(candidate.discriminating_test.primary_observables),
    )


def _contrast(
    left: ReasoningModeCandidateProfile,
    right: ReasoningModeCandidateProfile,
) -> PairwiseReasoningModeContrast:
    premise_left = set(left.premise_statement_ids)
    premise_right = set(right.premise_statement_ids)
    premise_jaccard = _id_jaccard(left.premise_statement_ids, right.premise_statement_ids)
    gap_jaccard = _id_jaccard(left.gap_statement_ids, right.gap_statement_ids)
    challenge_jaccard = _jaccard(_tokens(left.challenge_text), _tokens(right.challenge_text))
    baseline_jaccard = _jaccard(_tokens(left.baseline_summary), _tokens(right.baseline_summary))
    alternative_jaccard = _jaccard(
        _tokens(left.alternative_summary), _tokens(right.alternative_summary)
    )
    prediction_jaccard = _jaccard(
        _aggregate_tokens(left.prediction_observables),
        _aggregate_tokens(right.prediction_observables),
    )
    test_jaccard = _jaccard(
        _aggregate_tokens(left.test_observables),
        _aggregate_tokens(right.test_observables),
    )
    construct_jaccard = _jaccard(
        _aggregate_tokens(left.construct_texts),
        _aggregate_tokens(right.construct_texts),
    )

    neighborhood_signals: list[str] = []
    if len(premise_left & premise_right) >= 2 or premise_jaccard >= 0.35:
        neighborhood_signals.append("shared_grounded_premise_family")
    if challenge_jaccard >= 0.25:
        neighborhood_signals.append("overlapping_challenged_assumption_semantics")
    if prediction_jaccard >= 0.25:
        neighborhood_signals.append("overlapping_prediction_observables")
    if test_jaccard >= 0.25:
        neighborhood_signals.append("overlapping_discriminating_observables")
    if alternative_jaccard >= 0.35:
        neighborhood_signals.append("overlapping_alternative_model_surface")

    duplicate_signals: list[str] = []
    if premise_jaccard >= 0.60:
        duplicate_signals.append("high_premise_overlap")
    if challenge_jaccard >= 0.65:
        duplicate_signals.append("high_challenge_overlap")
    if alternative_jaccard >= 0.70:
        duplicate_signals.append("high_alternative_summary_overlap")
    if prediction_jaccard >= 0.60:
        duplicate_signals.append("high_prediction_observable_overlap")

    transform_distinct = left.representation_transform != right.representation_transform
    if not transform_distinct:
        relation: ModeRelation = "same_representation_family"
    elif len(duplicate_signals) >= 3:
        relation = "distinct_operator_surface_near_duplicate"
    elif len(neighborhood_signals) >= 2:
        relation = "distinct_mode_shared_scientific_neighborhood"
    else:
        relation = "distinct_mode_low_observed_overlap"

    return PairwiseReasoningModeContrast(
        candidate_a_id=left.candidate_id,
        operator_a=left.operator_id,
        transform_a=left.representation_transform,
        candidate_b_id=right.candidate_id,
        operator_b=right.operator_id,
        transform_b=right.representation_transform,
        representation_transform_distinct=transform_distinct,
        shared_premise_count=len(premise_left & premise_right),
        premise_jaccard=round(premise_jaccard, 6),
        gap_jaccard=round(gap_jaccard, 6),
        challenge_token_jaccard=round(challenge_jaccard, 6),
        baseline_summary_token_jaccard=round(baseline_jaccard, 6),
        alternative_summary_token_jaccard=round(alternative_jaccard, 6),
        prediction_observable_token_jaccard=round(prediction_jaccard, 6),
        test_observable_token_jaccard=round(test_jaccard, 6),
        construct_token_jaccard=round(construct_jaccard, 6),
        shared_scientific_neighborhood_signals=neighborhood_signals,
        surface_near_duplicate_signals=duplicate_signals,
        relation=relation,
    )


def build_reasoning_mode_contrast(
    *,
    reframe_shadow: ScientificReframingShadowReport,
    proxy_shadow: ProxyChallengeRunReport | None = None,
    contradiction_shadow: ContradictionResolutionRunReport | None = None,
) -> ReasoningModeContrastReport:
    task_ids = {reframe_shadow.source_task_id}
    if proxy_shadow is not None:
        task_ids.add(proxy_shadow.source_task_id)
    if contradiction_shadow is not None:
        task_ids.add(contradiction_shadow.source_task_id)
    if len(task_ids) != 1:
        raise ValueError("reasoning-mode contrast requires one shared source task")
    profiles = [_profile_reframe(row) for row in reframe_shadow.candidates]
    if proxy_shadow is not None:
        profiles.extend(_profile_proxy(row) for row in proxy_shadow.candidates)
    if contradiction_shadow is not None:
        profiles.extend(
            _profile_contradiction(row)
            for row in contradiction_shadow.candidates
        )

    pairs: list[PairwiseReasoningModeContrast] = []
    for index, left in enumerate(profiles):
        for right in profiles[index + 1 :]:
            if left.operator_id == right.operator_id:
                continue
            pairs.append(_contrast(left, right))

    payload = {
        "task_id": reframe_shadow.source_task_id,
        "candidate_ids": [row.candidate_id for row in profiles],
        "pairs": [
            [row.candidate_a_id, row.candidate_b_id, row.relation]
            for row in pairs
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    counts = Counter(row.relation for row in pairs)
    return ReasoningModeContrastReport(
        report_id=f"scientific_reframing_mode_contrast:{digest}",
        source_task_id=reframe_shadow.source_task_id,
        candidate_profiles=profiles,
        pairwise_contrasts=pairs,
        relation_counts=dict(sorted(counts.items())),
    )


def load_reasoning_mode_inputs(
    *,
    reframe_shadow_path: str | Path,
    proxy_shadow_path: str | Path | None = None,
    contradiction_shadow_path: str | Path | None = None,
) -> tuple[
    ScientificReframingShadowReport,
    ProxyChallengeRunReport | None,
    ContradictionResolutionRunReport | None,
]:
    reframe = ScientificReframingShadowReport.model_validate_json(
        Path(reframe_shadow_path).read_text(encoding="utf-8")
    )
    proxy = None
    if proxy_shadow_path is not None:
        proxy = ProxyChallengeRunReport.model_validate_json(
            Path(proxy_shadow_path).read_text(encoding="utf-8")
        )
    contradiction = None
    if contradiction_shadow_path is not None:
        contradiction = ContradictionResolutionRunReport.model_validate_json(
            Path(contradiction_shadow_path).read_text(encoding="utf-8")
        )
    return reframe, proxy, contradiction
