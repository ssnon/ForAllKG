from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SignalLevel = Literal["HIGH", "MEDIUM", "LOW"]
RealizationViability = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
IdeaReproductiveValue = Literal[
    "ENCOURAGED",
    "PRESERVED",
    "CONDITIONAL",
    "TERMINATED",
]
EvolutionChannel = Literal["EXPLOIT", "TRANSFORM", "EXPLORE", "WILDCARD"]
PolicyUncertainty = Literal["HIGH", "MEDIUM", "LOW"]
EvolutionOperatorHint = Literal[
    "SAME_PREMISE_SHARPEN",
    "EVIDENCE_REAXIS",
    "CROSS_SOURCE_BRIDGE",
    "BACKBONE_MUTATION",
    "CANDIDATE_INTERPRETATION",
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
    "PROXY_CHALLENGE",
    "AXIS_MUTATION",
    "REQUEST_GRAPH_RETRAVERSAL",
]


_SIGNAL_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_VIABILITY_RANK = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
_REPRODUCTIVE_RANK = {
    "TERMINATED": 0,
    "CONDITIONAL": 1,
    "PRESERVED": 2,
    "ENCOURAGED": 3,
}
_UNCERTAINTY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_BURDEN_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


_HARD_IDEA_CONSTRAINT_CODES = {
    "FABRICATED_PROVENANCE",
    "EXTERNAL_LITERATURE_AS_POSITIVE_PREMISE",
    "BROKEN_SOURCE_CONTEXT_LINEAGE",
    "TASK_REPLACEMENT",
    "SCIENTIFICALLY_INCOHERENT_RELATION",
    "SAFETY_CONSTRAINT",
}


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class IdeaEvolutionPolicyState(StrictModel):
    schema_version: Literal[
        "idea-evolution-policy-state-v1"
    ] = "idea-evolution-policy-state-v1"

    idea_id: str = Field(min_length=1)
    observation_ids: list[str] = Field(default_factory=list)
    observation_scopes: list[str] = Field(default_factory=list)

    realization_viability: RealizationViability
    idea_reproductive_value: IdeaReproductiveValue
    exploit_value: SignalLevel
    transformation_pressure: SignalLevel
    exploration_value: SignalLevel
    uncertainty: PolicyUncertainty

    preferred_channels: list[EvolutionChannel] = Field(default_factory=list)
    operator_hints: list[EvolutionOperatorHint] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    hard_constraint_codes: list[str] = Field(default_factory=list)

    soft_feedback_only: bool = True
    current_realization_failure_is_not_idea_failure: Literal[True] = True
    verification_is_not_fertility_authority: Literal[True] = True
    operator_hints_are_nonbinding: Literal[True] = True
    idea_termination_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _normalize(self) -> "IdeaEvolutionPolicyState":
        self.observation_ids = list(dict.fromkeys(self.observation_ids))
        self.observation_scopes = list(dict.fromkeys(self.observation_scopes))
        self.preferred_channels = list(dict.fromkeys(self.preferred_channels))
        self.operator_hints = list(dict.fromkeys(self.operator_hints))
        self.reason_codes = list(dict.fromkeys(self.reason_codes))
        self.hard_constraint_codes = list(dict.fromkeys(self.hard_constraint_codes))
        if self.idea_reproductive_value == "TERMINATED" and not self.hard_constraint_codes:
            raise ValueError("TERMINATED requires an explicit hard idea constraint")
        return self


class EvolutionSlotBudget(StrictModel):
    schema_version: Literal[
        "idea-evolution-slot-budget-v1"
    ] = "idea-evolution-slot-budget-v1"

    max_parent_budget: int = Field(ge=1)
    exploit_slots: int = Field(ge=0)
    transform_slots: int = Field(ge=0)
    explore_slots: int = Field(ge=0)
    wildcard_slots: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_total(self) -> "EvolutionSlotBudget":
        total = (
            self.exploit_slots
            + self.transform_slots
            + self.explore_slots
            + self.wildcard_slots
        )
        if total != self.max_parent_budget:
            raise ValueError("evolution slot counts must equal max_parent_budget")
        return self


class EvolutionAllocationRecord(StrictModel):
    idea_id: str = Field(min_length=1)
    channel: EvolutionChannel
    operator_hints: list[EvolutionOperatorHint] = Field(default_factory=list)
    scientific_profile: str | None = None
    realization_viability: RealizationViability
    idea_reproductive_value: IdeaReproductiveValue
    exploit_value: SignalLevel
    transformation_pressure: SignalLevel
    exploration_value: SignalLevel
    uncertainty: PolicyUncertainty
    inherited_v2_1_1_priority_band: str = "MEDIUM"
    selection_reason_codes: list[str] = Field(default_factory=list)


class EvolutionaryIdeaSearchShadowReport(StrictModel):
    schema_version: Literal[
        "evolutionary-idea-search-shadow-v1"
    ] = "evolutionary-idea-search-shadow-v1"

    report_id: str
    report_sha256: str
    source_generational_report_id: str
    source_generational_report_sha256: str
    policy_states: list[IdeaEvolutionPolicyState] = Field(default_factory=list)
    policy_state_count: int = Field(ge=0)
    slot_budget: EvolutionSlotBudget
    allocations: list[EvolutionAllocationRecord] = Field(default_factory=list)
    selected_parent_idea_ids: list[str] = Field(default_factory=list)
    selected_parent_count: int = Field(ge=0)
    selected_parent_channel_counts: dict[str, int] = Field(default_factory=dict)

    v2_1_1_parent_idea_ids: list[str] = Field(default_factory=list)
    v2_1_1_parent_count: int = Field(ge=0)
    parent_set_changed: bool = False
    parent_overlap_count: int = Field(ge=0)
    parent_added_idea_ids: list[str] = Field(default_factory=list)
    parent_dropped_idea_ids: list[str] = Field(default_factory=list)

    realization_viability_counts: dict[str, int] = Field(default_factory=dict)
    reproductive_value_counts: dict[str, int] = Field(default_factory=dict)
    transformation_pressure_counts: dict[str, int] = Field(default_factory=dict)
    operator_hint_counts: dict[str, int] = Field(default_factory=dict)
    selected_low_viability_transform_count: int = Field(ge=0)
    selected_v2_1_1_low_priority_count: int = Field(ge=0)
    hard_constraint_termination_count: int = Field(ge=0)

    soft_feedback_cannot_terminate_idea: Literal[True] = True
    verification_is_not_fertility_authority: Literal[True] = True
    operator_hints_are_nonbinding: Literal[True] = True
    transform_lane_not_profile_gated: Literal[True] = True
    explore_lane_not_profile_gated: Literal[True] = True
    wildcard_lane_not_profile_gated: Literal[True] = True
    exact_kernel_duplicate_suppression_only: Literal[True] = True
    legacy_family_hard_gate_used: Literal[False] = False
    single_scalar_fitness_used: Literal[False] = False
    new_llm_calls: Literal[False] = False
    new_retrieval_calls: Literal[False] = False
    shadow_only: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "EvolutionaryIdeaSearchShadowReport":
        if self.policy_state_count != len(self.policy_states):
            raise ValueError("policy_state_count mismatch")
        if self.selected_parent_count != len(self.selected_parent_idea_ids):
            raise ValueError("selected_parent_count mismatch")
        if self.v2_1_1_parent_count != len(self.v2_1_1_parent_idea_ids):
            raise ValueError("v2_1_1_parent_count mismatch")
        if self.selected_parent_count > self.slot_budget.max_parent_budget:
            raise ValueError("selected parent count exceeds slot budget")
        return self


def derive_slot_budget(max_parent_budget: int) -> EvolutionSlotBudget:
    if max_parent_budget < 1:
        raise ValueError("max_parent_budget must be >= 1")
    channels = ("EXPLOIT", "TRANSFORM", "EXPLORE", "WILDCARD")
    weights = {
        "EXPLOIT": 0.25,
        "TRANSFORM": 0.375,
        "EXPLORE": 0.25,
        "WILDCARD": 0.125,
    }
    raw = {name: max_parent_budget * weights[name] for name in channels}
    counts = {name: int(raw[name]) for name in channels}
    remainder = max_parent_budget - sum(counts.values())
    order = sorted(
        channels,
        key=lambda name: (
            -(raw[name] - counts[name]),
            ("TRANSFORM", "EXPLOIT", "EXPLORE", "WILDCARD").index(name),
        ),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return EvolutionSlotBudget(
        max_parent_budget=max_parent_budget,
        exploit_slots=counts["EXPLOIT"],
        transform_slots=counts["TRANSFORM"],
        explore_slots=counts["EXPLORE"],
        wildcard_slots=counts["WILDCARD"],
    )


def _max_signal(current: SignalLevel, proposed: SignalLevel) -> SignalLevel:
    return proposed if _SIGNAL_RANK[proposed] > _SIGNAL_RANK[current] else current


def _max_viability(
    current: RealizationViability,
    proposed: RealizationViability,
) -> RealizationViability:
    return proposed if _VIABILITY_RANK[proposed] > _VIABILITY_RANK[current] else current


def _policy_for_observation(observation: Any) -> dict[str, Any]:
    viability: RealizationViability = "UNKNOWN"
    exploit: SignalLevel = "MEDIUM"
    transform: SignalLevel = "LOW"
    explore: SignalLevel = "MEDIUM"
    uncertainty: PolicyUncertainty = "MEDIUM"
    reproductive: IdeaReproductiveValue = "PRESERVED"
    channels: list[EvolutionChannel] = []
    hints: list[EvolutionOperatorHint] = []
    reasons: list[str] = []

    materialization = str(observation.materialization_status or "")
    if materialization == "MATERIALIZED":
        viability = "MEDIUM"
        channels.append("EXPLOIT")
        reasons.append("REALIZATION_MATERIALIZED")
    elif materialization:
        viability = "LOW"
        exploit = "LOW"
        transform = "MEDIUM"
        explore = "HIGH"
        uncertainty = "HIGH"
        channels.extend(["TRANSFORM", "EXPLORE", "WILDCARD"])
        hints.extend(["BACKBONE_MUTATION", "CANDIDATE_INTERPRETATION"])
        reasons.extend([
            "CURRENT_REALIZATION_FAILED_TO_MATERIALIZE",
            "MATERIALIZATION_FAILURE_IS_TRANSFORMATION_SIGNAL_NOT_IDEA_REJECTION",
        ])

    residual = str(observation.residual_epistemic_state or "")
    if residual == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW":
        viability = _max_viability(viability, "HIGH")
        exploit = _max_signal(exploit, "HIGH")
        transform = _max_signal(transform, "LOW")
        uncertainty = "LOW"
        reproductive = "ENCOURAGED"
        channels.extend(["EXPLOIT", "EXPLORE"])
        hints.append("SAME_PREMISE_SHARPEN")
        reasons.append("FRESH_RESIDUAL_SUPPORTS_EXPLOITATION")
    elif residual == "UNRESOLVED_EVIDENCE_GAP":
        viability = _max_viability(viability, "MEDIUM")
        transform = _max_signal(transform, "MEDIUM")
        explore = _max_signal(explore, "HIGH")
        uncertainty = "HIGH"
        channels.extend(["EXPLORE", "TRANSFORM", "WILDCARD"])
        hints.extend([
            "EVIDENCE_REAXIS",
            "CROSS_SOURCE_BRIDGE",
            "REQUEST_GRAPH_RETRAVERSAL",
        ])
        reasons.append("EVIDENCE_GAP_PRESERVES_EXPLORATION_VALUE")
    elif residual == "UNRESOLVED_TOPOLOGY_GAP":
        viability = _max_viability(viability, "MEDIUM")
        exploit = "LOW"
        transform = "HIGH"
        explore = "HIGH"
        reproductive = "ENCOURAGED"
        channels.extend(["TRANSFORM", "EXPLORE"])
        hints.extend([
            "BACKBONE_MUTATION",
            "LATENT_VARIABLE",
            "REGIME_BOUNDARY",
        ])
        reasons.append("TOPOLOGY_GAP_INCREASES_TRANSFORMATION_PRESSURE")
    elif residual == "PRIOR_ART_BACKED_OR_NO_RESIDUAL":
        viability = "LOW"
        exploit = "LOW"
        transform = "HIGH"
        explore = "HIGH"
        reproductive = "ENCOURAGED"
        uncertainty = "LOW"
        channels.extend(["TRANSFORM", "EXPLORE"])
        hints.extend([
            "AXIS_MUTATION",
            "REGIME_BOUNDARY",
            "PROXY_CHALLENGE",
            "CANDIDATE_INTERPRETATION",
        ])
        reasons.extend([
            "KNOWN_REGION_REDUCES_EXPLOITATION_NOT_FERTILITY",
            "PRIOR_ART_BOUNDARY_INCREASES_MUTATIONAL_PRESSURE",
        ])

    prospective = str(observation.prospective_identifiability or "")
    if prospective == "CURRENTLY_IDENTIFIED":
        viability = "HIGH"
        exploit = "HIGH"
        uncertainty = "LOW"
        reproductive = "ENCOURAGED"
        channels.append("EXPLOIT")
        hints.append("SAME_PREMISE_SHARPEN")
        reasons.append("CURRENT_REALIZATION_IDENTIFIED")
    elif prospective == "PROSPECTIVELY_IDENTIFIABLE":
        viability = "HIGH"
        exploit = "HIGH"
        transform = _max_signal(transform, "MEDIUM")
        uncertainty = "MEDIUM"
        reproductive = "ENCOURAGED"
        channels.extend(["EXPLOIT", "EXPLORE"])
        hints.extend(["SAME_PREMISE_SHARPEN", "EVIDENCE_REAXIS"])
        reasons.append("PROSPECTIVE_REALIZATION_REMAINS_TESTABLE")
    elif prospective == "NOT_OPERATIONALIZABLE":
        viability = "LOW"
        exploit = "LOW"
        transform = "HIGH"
        explore = "HIGH"
        reproductive = "ENCOURAGED"
        uncertainty = "MEDIUM"
        channels.extend(["TRANSFORM", "EXPLORE", "WILDCARD"])
        hints.extend([
            "BACKBONE_MUTATION",
            "LATENT_VARIABLE",
            "REGIME_BOUNDARY",
            "PROXY_CHALLENGE",
            "REQUEST_GRAPH_RETRAVERSAL",
        ])
        reasons.extend([
            "CURRENT_REALIZATION_NOT_OPERATIONALIZABLE",
            "NON_OPERATIONAL_REALIZATION_INCREASES_REFORMULATION_PRESSURE",
        ])

    if observation.prospective_contract_integrity_passed is False:
        viability = "LOW"
        exploit = "LOW"
        transform = _max_signal(transform, "MEDIUM")
        explore = _max_signal(explore, "HIGH")
        uncertainty = "HIGH"
        channels.extend(["TRANSFORM", "WILDCARD"])
        hints.extend(["BACKBONE_MUTATION", "EVIDENCE_REAXIS"])
        reasons.append("PROSPECTIVE_CONTRACT_FAILURE_IS_REALIZATION_SCOPED")

    if not channels:
        channels = ["EXPLORE", "WILDCARD"]
    return {
        "viability": viability,
        "exploit": exploit,
        "transform": transform,
        "explore": explore,
        "uncertainty": uncertainty,
        "reproductive": reproductive,
        "channels": channels,
        "hints": hints,
        "reasons": reasons,
    }


def compile_evolution_policy_states(
    generational_report: Any,
    *,
    hard_constraint_codes_by_idea: Mapping[str, Sequence[str]] | None = None,
) -> list[IdeaEvolutionPolicyState]:
    explicit_hard = {
        str(idea_id): [str(code) for code in codes]
        for idea_id, codes in (hard_constraint_codes_by_idea or {}).items()
    }
    observations_by_idea: dict[str, list[Any]] = defaultdict(list)
    for observation in generational_report.observations:
        observations_by_idea[str(observation.idea_id)].append(observation)

    result: list[IdeaEvolutionPolicyState] = []
    for node in generational_report.research_ideas:
        idea_id = str(node.idea_id)
        rows = observations_by_idea.get(idea_id, [])
        hard_codes = [
            code
            for code in explicit_hard.get(idea_id, [])
            if code in _HARD_IDEA_CONSTRAINT_CODES
        ]
        if hard_codes:
            result.append(
                IdeaEvolutionPolicyState(
                    idea_id=idea_id,
                    observation_ids=[str(row.observation_id) for row in rows],
                    observation_scopes=[str(getattr(row, "target_scope", "REALIZATION")) for row in rows],
                    realization_viability="UNKNOWN",
                    idea_reproductive_value="TERMINATED",
                    exploit_value="LOW",
                    transformation_pressure="LOW",
                    exploration_value="LOW",
                    uncertainty="LOW",
                    preferred_channels=[],
                    operator_hints=[],
                    reason_codes=["EXPLICIT_HARD_IDEA_CONSTRAINT"],
                    hard_constraint_codes=hard_codes,
                    soft_feedback_only=False,
                )
            )
            continue

        if not rows:
            result.append(
                IdeaEvolutionPolicyState(
                    idea_id=idea_id,
                    observation_ids=[],
                    observation_scopes=[],
                    realization_viability="UNKNOWN",
                    idea_reproductive_value="PRESERVED",
                    exploit_value="MEDIUM",
                    transformation_pressure="MEDIUM",
                    exploration_value="HIGH",
                    uncertainty="HIGH",
                    preferred_channels=["EXPLORE", "WILDCARD", "TRANSFORM"],
                    operator_hints=["CROSS_SOURCE_BRIDGE", "BACKBONE_MUTATION"],
                    reason_codes=[
                        "NO_REALIZATION_OUTCOME_OBSERVED",
                        "ABSENCE_OF_VERIFICATION_IS_NOT_NEGATIVE_FITNESS",
                    ],
                )
            )
            continue

        viability: RealizationViability = "UNKNOWN"
        exploit: SignalLevel = "LOW"
        transform: SignalLevel = "LOW"
        explore: SignalLevel = "LOW"
        reproductive: IdeaReproductiveValue = "PRESERVED"
        uncertainty: PolicyUncertainty = "LOW"
        channels: list[EvolutionChannel] = []
        hints: list[EvolutionOperatorHint] = []
        reasons: list[str] = []
        scopes: list[str] = []
        post_verification_seen = False

        for row in rows:
            one = _policy_for_observation(row)
            viability = _max_viability(viability, one["viability"])
            exploit = _max_signal(exploit, one["exploit"])
            transform = _max_signal(transform, one["transform"])
            explore = _max_signal(explore, one["explore"])
            if _REPRODUCTIVE_RANK[one["reproductive"]] > _REPRODUCTIVE_RANK[reproductive]:
                reproductive = one["reproductive"]
            if _UNCERTAINTY_RANK[one["uncertainty"]] > _UNCERTAINTY_RANK[uncertainty]:
                uncertainty = one["uncertainty"]
            channels.extend(one["channels"])
            hints.extend(one["hints"])
            reasons.extend(one["reasons"])
            scopes.append(str(getattr(row, "target_scope", "REALIZATION")))
            post_verification_seen = post_verification_seen or bool(
                row.residual_epistemic_state is not None
                or row.prospective_identifiability is not None
            )

        if not post_verification_seen:
            uncertainty = "HIGH"
            explore = _max_signal(explore, "HIGH")
            channels.extend(["EXPLORE", "WILDCARD"])
            reasons.append("NO_POST_VERIFICATION_FEEDBACK_PRESERVES_EXPLORATION")

        if transform == "HIGH" or explore == "HIGH" or exploit == "HIGH":
            reproductive = "ENCOURAGED"
        result.append(
            IdeaEvolutionPolicyState(
                idea_id=idea_id,
                observation_ids=[str(row.observation_id) for row in rows],
                observation_scopes=scopes,
                realization_viability=viability,
                idea_reproductive_value=reproductive,
                exploit_value=exploit,
                transformation_pressure=transform,
                exploration_value=explore,
                uncertainty=uncertainty,
                preferred_channels=list(dict.fromkeys(channels)),
                operator_hints=list(dict.fromkeys(hints)),
                reason_codes=list(dict.fromkeys(reasons)),
                hard_constraint_codes=[],
                soft_feedback_only=True,
            )
        )
    return result


def _best_profile(candidate: Any) -> str | None:
    layers = dict(candidate.pareto_layer_by_profile or {})
    if not layers:
        return None
    return min(layers, key=lambda profile: (layers[profile], profile))


def _best_layer(candidate: Any) -> int:
    layers = list((candidate.pareto_layer_by_profile or {}).values())
    return min(layers) if layers else 999


def _dimension(candidate: Any, name: str) -> int:
    return int((candidate.dimension_levels or {}).get(name, 0))


def _allocate_channel(
    *,
    channel: EvolutionChannel,
    quota: int,
    candidates: Sequence[Any],
    policy_by_idea: Mapping[str, IdeaEvolutionPolicyState],
    node_by_idea: Mapping[str, Any],
    selected_ids: list[str],
    selected_kernels: set[str],
) -> list[EvolutionAllocationRecord]:
    if quota <= 0:
        return []

    def eligible(candidate: Any) -> bool:
        idea_id = str(candidate.idea_id)
        policy = policy_by_idea[idea_id]
        if idea_id in selected_ids:
            return False
        if policy.idea_reproductive_value == "TERMINATED":
            return False
        if node_by_idea[idea_id].kernel_sha256 in selected_kernels:
            return False
        if channel == "EXPLOIT":
            return policy.exploit_value != "LOW" and bool(candidate.eligible_profiles)
        if channel == "TRANSFORM":
            return policy.transformation_pressure != "LOW"
        if channel == "EXPLORE":
            return policy.exploration_value != "LOW"
        return True

    def key(candidate: Any) -> tuple[Any, ...]:
        policy = policy_by_idea[str(candidate.idea_id)]
        if channel == "EXPLOIT":
            return (
                -_SIGNAL_RANK[policy.exploit_value],
                -_VIABILITY_RANK[policy.realization_viability],
                _best_layer(candidate),
                -_dimension(candidate, "task_relevance"),
                -_dimension(candidate, "falsifiability"),
                -_dimension(candidate, "operationalizability"),
                -_dimension(candidate, "information_gain"),
                _BURDEN_RANK.get(str(candidate.verification_burden), 1),
                -float(candidate.underexplored_bonus),
                str(candidate.idea_id),
            )
        if channel == "TRANSFORM":
            # Failed/known current realizations are not penalized here. They can
            # be exactly the parents from which useful conceptual mutations arise.
            return (
                -_SIGNAL_RANK[policy.transformation_pressure],
                -_REPRODUCTIVE_RANK[policy.idea_reproductive_value],
                _VIABILITY_RANK[policy.realization_viability],
                -_SIGNAL_RANK[policy.exploration_value],
                -float(candidate.underexplored_bonus),
                -_dimension(candidate, "information_gain"),
                -_dimension(candidate, "mechanistic_coherence"),
                str(candidate.idea_id),
            )
        if channel == "EXPLORE":
            return (
                -_SIGNAL_RANK[policy.exploration_value],
                -float(candidate.underexplored_bonus),
                -_UNCERTAINTY_RANK[policy.uncertainty],
                -_dimension(candidate, "information_gain"),
                -_dimension(candidate, "mechanistic_coherence"),
                str(candidate.idea_id),
            )
        return (
            -_UNCERTAINTY_RANK[policy.uncertainty],
            -float(candidate.underexplored_bonus),
            _best_layer(candidate),
            str(candidate.idea_id),
        )

    selected: list[EvolutionAllocationRecord] = []
    pool = [candidate for candidate in candidates if eligible(candidate)]
    pool.sort(key=key)
    for candidate in pool[:quota]:
        idea_id = str(candidate.idea_id)
        policy = policy_by_idea[idea_id]
        selected_ids.append(idea_id)
        selected_kernels.add(node_by_idea[idea_id].kernel_sha256)
        selected.append(
            EvolutionAllocationRecord(
                idea_id=idea_id,
                channel=channel,
                operator_hints=policy.operator_hints,
                scientific_profile=_best_profile(candidate),
                realization_viability=policy.realization_viability,
                idea_reproductive_value=policy.idea_reproductive_value,
                exploit_value=policy.exploit_value,
                transformation_pressure=policy.transformation_pressure,
                exploration_value=policy.exploration_value,
                uncertainty=policy.uncertainty,
                inherited_v2_1_1_priority_band=str(candidate.priority_band),
                selection_reason_codes=[
                    f"{channel}_SLOT",
                    "CHANNEL_SPECIFIC_NONSCALAR_ORDERING",
                    "EXACT_KERNEL_NOT_ALREADY_SELECTED",
                    (
                        "LEGACY_PROFILE_USED_ONLY_FOR_EXPLOIT"
                        if channel == "EXPLOIT"
                        else "LEGACY_PROFILE_NOT_A_HARD_GATE"
                    ),
                ],
            )
        )
    return selected


def build_evolutionary_idea_search_shadow(
    generational_report: Any,
    *,
    max_parent_budget: int | None = None,
    slot_budget: EvolutionSlotBudget | None = None,
    hard_constraint_codes_by_idea: Mapping[str, Sequence[str]] | None = None,
) -> EvolutionaryIdeaSearchShadowReport:
    max_budget = int(max_parent_budget or generational_report.g2_parent_count or 1)
    budget = slot_budget or derive_slot_budget(max_budget)
    if budget.max_parent_budget != max_budget:
        raise ValueError("slot budget / max_parent_budget mismatch")

    policies = compile_evolution_policy_states(
        generational_report,
        hard_constraint_codes_by_idea=hard_constraint_codes_by_idea,
    )
    policy_by_idea = {row.idea_id: row for row in policies}
    node_by_idea = {str(row.idea_id): row for row in generational_report.research_ideas}
    candidates = list(generational_report.candidate_states)

    selected_ids: list[str] = []
    selected_kernels: set[str] = set()
    allocations: list[EvolutionAllocationRecord] = []
    quotas = (
        ("EXPLOIT", budget.exploit_slots),
        ("TRANSFORM", budget.transform_slots),
        ("EXPLORE", budget.explore_slots),
        ("WILDCARD", budget.wildcard_slots),
    )
    for channel, quota in quotas:
        allocations.extend(
            _allocate_channel(
                channel=channel,
                quota=quota,
                candidates=candidates,
                policy_by_idea=policy_by_idea,
                node_by_idea=node_by_idea,
                selected_ids=selected_ids,
                selected_kernels=selected_kernels,
            )
        )

    # Sparse channels should not waste compute. Refill capacity using a fixed
    # lane cycle rather than collapsing every signal into one scalar fitness.
    refill_cycle: tuple[EvolutionChannel, ...] = (
        "TRANSFORM",
        "EXPLORE",
        "EXPLOIT",
        "WILDCARD",
    )
    while len(selected_ids) < budget.max_parent_budget:
        progress = False
        for channel in refill_cycle:
            rows = _allocate_channel(
                channel=channel,
                quota=1,
                candidates=candidates,
                policy_by_idea=policy_by_idea,
                node_by_idea=node_by_idea,
                selected_ids=selected_ids,
                selected_kernels=selected_kernels,
            )
            if rows:
                rows[0].selection_reason_codes.append("UNUSED_SLOT_REFILL")
                allocations.extend(rows)
                progress = True
                if len(selected_ids) >= budget.max_parent_budget:
                    break
        if not progress:
            break

    old_ids = list(map(str, generational_report.g2_parent_idea_ids))
    old_set = set(old_ids)
    new_set = set(selected_ids)
    channel_counts = Counter(row.channel for row in allocations)
    viability_counts = Counter(row.realization_viability for row in policies)
    reproductive_counts = Counter(row.idea_reproductive_value for row in policies)
    transform_counts = Counter(row.transformation_pressure for row in policies)
    hint_counts = Counter(
        hint
        for row in policies
        for hint in row.operator_hints
    )
    low_priority = {
        str(row.idea_id)
        for row in generational_report.candidate_states
        if str(row.priority_band) == "LOW"
    }

    provisional = EvolutionaryIdeaSearchShadowReport(
        report_id="pending",
        report_sha256="pending",
        source_generational_report_id=str(generational_report.report_id),
        source_generational_report_sha256=str(generational_report.report_sha256),
        policy_states=policies,
        policy_state_count=len(policies),
        slot_budget=budget,
        allocations=allocations,
        selected_parent_idea_ids=selected_ids,
        selected_parent_count=len(selected_ids),
        selected_parent_channel_counts=dict(sorted(channel_counts.items())),
        v2_1_1_parent_idea_ids=old_ids,
        v2_1_1_parent_count=len(old_ids),
        parent_set_changed=old_set != new_set,
        parent_overlap_count=len(old_set & new_set),
        parent_added_idea_ids=[idea_id for idea_id in selected_ids if idea_id not in old_set],
        parent_dropped_idea_ids=[idea_id for idea_id in old_ids if idea_id not in new_set],
        realization_viability_counts=dict(sorted(viability_counts.items())),
        reproductive_value_counts=dict(sorted(reproductive_counts.items())),
        transformation_pressure_counts=dict(sorted(transform_counts.items())),
        operator_hint_counts=dict(sorted(hint_counts.items())),
        selected_low_viability_transform_count=sum(
            row.channel == "TRANSFORM" and row.realization_viability == "LOW"
            for row in allocations
        ),
        selected_v2_1_1_low_priority_count=sum(
            row.idea_id in low_priority for row in allocations
        ),
        hard_constraint_termination_count=sum(
            row.idea_reproductive_value == "TERMINATED" for row in policies
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"evolutionary_idea_search_shadow:{digest[:20]}",
            "report_sha256": digest,
        }
    )


__all__ = [
    "EvolutionAllocationRecord",
    "EvolutionChannel",
    "EvolutionOperatorHint",
    "EvolutionSlotBudget",
    "EvolutionaryIdeaSearchShadowReport",
    "IdeaEvolutionPolicyState",
    "IdeaReproductiveValue",
    "PolicyUncertainty",
    "RealizationViability",
    "SignalLevel",
    "build_evolutionary_idea_search_shadow",
    "compile_evolution_policy_states",
    "derive_slot_budget",
]
