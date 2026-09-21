from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.mode_contrast import (
    ModeRelation,
    ReasoningModeCandidateProfile,
    ReasoningModeContrastReport,
    RepresentationTransform,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ReasoningModeSlotId = Literal[
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
    "PROXY_CHALLENGE",
    "CONTRADICTION_RESOLUTION",
]


_CANONICAL_SLOTS: tuple[ReasoningModeSlotId, ...] = (
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
    "PROXY_CHALLENGE",
    "CONTRADICTION_RESOLUTION",
)

_EXPECTED_TRANSFORM: dict[ReasoningModeSlotId, RepresentationTransform] = {
    "LATENT_VARIABLE": "introduces_hidden_construct",
    "REGIME_BOUNDARY": "partitions_response_law",
    "PROXY_CHALLENGE": "challenges_measurement_equivalence",
    "CONTRADICTION_RESOLUTION": "reconciles_apparently_incompatible_evidence",
}


class UnifiedReasoningPortfolioEntry(StrictModel):
    candidate_id: str = Field(min_length=1)
    operator_id: ReasoningModeSlotId
    representation_transform: RepresentationTransform
    premise_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    prediction_observable_count: int = Field(ge=0)
    test_observable_count: int = Field(ge=0)
    scientific_neighborhood_cluster_id: str = Field(min_length=1)

    shadow_candidate_preserved: Literal[True] = True
    retained_because_operator_is_distinct: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    redundancy_based_candidate_drop_performed: Literal[False] = False


class UnifiedReasoningModeSlot(StrictModel):
    slot_id: ReasoningModeSlotId
    representation_transform: RepresentationTransform
    candidate_ids: list[str] = Field(default_factory=list)
    empty_reason: str | None = None

    @model_validator(mode="after")
    def validate_slot(self) -> "UnifiedReasoningModeSlot":
        expected = _EXPECTED_TRANSFORM[self.slot_id]
        if self.representation_transform != expected:
            raise ValueError(
                f"{self.slot_id} must use representation transform {expected}"
            )
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("reasoning-mode slot candidate IDs must be unique")
        if self.candidate_ids and self.empty_reason is not None:
            raise ValueError("occupied reasoning-mode slots must not have empty_reason")
        if not self.candidate_ids and not (self.empty_reason or "").strip():
            raise ValueError("empty reasoning-mode slots require empty_reason")
        return self


class SharedScientificNeighborhoodCluster(StrictModel):
    cluster_id: str = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1)
    operator_ids: list[ReasoningModeSlotId] = Field(min_length=1)
    representation_transforms: list[RepresentationTransform] = Field(min_length=1)
    pair_relations: dict[str, int] = Field(default_factory=dict)
    neighborhood_signals: list[str] = Field(default_factory=list)
    surface_near_duplicate_signals: list[str] = Field(default_factory=list)

    connected_by_mode_contrast_heuristic: Literal[True] = True
    scientific_equivalence_established: Literal[False] = False
    scientific_topic_identity_established: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_cluster(self) -> "SharedScientificNeighborhoodCluster":
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("cluster candidate IDs must be unique")
        if len(self.operator_ids) != len(set(self.operator_ids)):
            raise ValueError("cluster operator IDs must be unique")
        if len(self.representation_transforms) != len(
            set(self.representation_transforms)
        ):
            raise ValueError("cluster representation transforms must be unique")
        return self


class UnifiedScientificReasoningShadowPortfolio(StrictModel):
    schema_version: Literal[
        "unified-scientific-reasoning-shadow-portfolio-v1"
    ] = "unified-scientific-reasoning-shadow-portfolio-v1"

    portfolio_id: str = Field(min_length=1)
    source_mode_contrast_report_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)

    entries: list[UnifiedReasoningPortfolioEntry]
    slots: list[UnifiedReasoningModeSlot]
    scientific_neighborhood_clusters: list[SharedScientificNeighborhoodCluster]

    candidate_count: int = Field(ge=0)
    occupied_reasoning_mode_count: int = Field(ge=0)
    empty_reasoning_mode_count: int = Field(ge=0)
    scientific_neighborhood_cluster_count: int = Field(ge=0)
    relation_counts: dict[str, int] = Field(default_factory=dict)

    llm_calls_performed: Literal[0] = 0
    shadow_only: Literal[True] = True
    reasoning_mode_portfolio_assembled: Literal[True] = True
    reasoning_mode_diversity_described: Literal[True] = True
    neighborhood_structure_described: Literal[True] = True
    relational_lane_integrated: Literal[False] = False
    critic_vectors_integrated: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False
    candidate_winner_selected: Literal[False] = False
    redundancy_based_candidate_drop_performed: Literal[False] = False
    portfolio_selection_authority: Literal[False] = False
    scientific_topic_diversity_established: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_portfolio(self) -> "UnifiedScientificReasoningShadowPortfolio":
        if tuple(row.slot_id for row in self.slots) != _CANONICAL_SLOTS:
            raise ValueError("reasoning-mode slots must use canonical operator order")

        entry_ids = [row.candidate_id for row in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("portfolio entry candidate IDs must be unique")
        if self.candidate_count != len(self.entries):
            raise ValueError("candidate_count must match entries")

        occupied = sum(bool(row.candidate_ids) for row in self.slots)
        if self.occupied_reasoning_mode_count != occupied:
            raise ValueError("occupied_reasoning_mode_count must match slots")
        if self.empty_reasoning_mode_count != len(self.slots) - occupied:
            raise ValueError("empty_reasoning_mode_count must match slots")

        slot_members = {
            candidate_id
            for slot in self.slots
            for candidate_id in slot.candidate_ids
        }
        if slot_members != set(entry_ids):
            raise ValueError("reasoning-mode slots must partition portfolio entries")

        cluster_members = {
            candidate_id
            for cluster in self.scientific_neighborhood_clusters
            for candidate_id in cluster.candidate_ids
        }
        if cluster_members != set(entry_ids):
            raise ValueError("scientific-neighborhood clusters must cover entries")
        if self.scientific_neighborhood_cluster_count != len(
            self.scientific_neighborhood_clusters
        ):
            raise ValueError("scientific_neighborhood_cluster_count must match clusters")

        membership_counts = Counter(
            candidate_id
            for cluster in self.scientific_neighborhood_clusters
            for candidate_id in cluster.candidate_ids
        )
        if any(count != 1 for count in membership_counts.values()):
            raise ValueError("each portfolio entry must belong to exactly one cluster")

        entry_cluster = {
            row.candidate_id: row.scientific_neighborhood_cluster_id
            for row in self.entries
        }
        for cluster in self.scientific_neighborhood_clusters:
            if any(entry_cluster[candidate_id] != cluster.cluster_id for candidate_id in cluster.candidate_ids):
                raise ValueError("entry cluster lineage must match cluster membership")
        return self


def _stable_cluster_id(task_id: str, candidate_ids: list[str]) -> str:
    payload = json.dumps(
        [task_id, *sorted(candidate_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"scientific_reasoning_neighborhood:{digest}"


def _stable_portfolio_id(report: ReasoningModeContrastReport) -> str:
    payload = json.dumps(
        [
            report.report_id,
            report.source_task_id,
            *sorted(row.candidate_id for row in report.candidate_profiles),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"unified_scientific_reasoning_portfolio:{digest}"


def _cluster_components(report: ReasoningModeContrastReport) -> list[list[str]]:
    candidate_ids = [row.candidate_id for row in report.candidate_profiles]
    adjacency: dict[str, set[str]] = {candidate_id: set() for candidate_id in candidate_ids}
    connect_relations: set[ModeRelation] = {
        "same_representation_family",
        "distinct_operator_surface_near_duplicate",
        "distinct_mode_shared_scientific_neighborhood",
    }
    for pair in report.pairwise_contrasts:
        if pair.relation not in connect_relations:
            continue
        adjacency[pair.candidate_a_id].add(pair.candidate_b_id)
        adjacency[pair.candidate_b_id].add(pair.candidate_a_id)

    components: list[list[str]] = []
    unseen = set(candidate_ids)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        unseen -= component
        components.append(sorted(component))
    components.sort(key=lambda values: values[0] if values else "")
    return components


def _cluster_rows(
    report: ReasoningModeContrastReport,
) -> tuple[list[SharedScientificNeighborhoodCluster], dict[str, str]]:
    profiles = {row.candidate_id: row for row in report.candidate_profiles}
    clusters: list[SharedScientificNeighborhoodCluster] = []
    membership: dict[str, str] = {}

    for candidate_ids in _cluster_components(report):
        cluster_id = _stable_cluster_id(report.source_task_id, candidate_ids)
        pairs = [
            row
            for row in report.pairwise_contrasts
            if row.candidate_a_id in candidate_ids and row.candidate_b_id in candidate_ids
        ]
        relation_counts = Counter(row.relation for row in pairs)
        neighborhood_signals = sorted({
            signal
            for row in pairs
            for signal in row.shared_scientific_neighborhood_signals
        })
        duplicate_signals = sorted({
            signal
            for row in pairs
            for signal in row.surface_near_duplicate_signals
        })
        operator_ids = sorted({profiles[candidate_id].operator_id for candidate_id in candidate_ids})
        transforms = sorted({
            profiles[candidate_id].representation_transform
            for candidate_id in candidate_ids
        })
        clusters.append(
            SharedScientificNeighborhoodCluster(
                cluster_id=cluster_id,
                candidate_ids=candidate_ids,
                operator_ids=operator_ids,
                representation_transforms=transforms,
                pair_relations=dict(sorted(relation_counts.items())),
                neighborhood_signals=neighborhood_signals,
                surface_near_duplicate_signals=duplicate_signals,
            )
        )
        for candidate_id in candidate_ids:
            membership[candidate_id] = cluster_id

    return clusters, membership


def _entry(
    profile: ReasoningModeCandidateProfile,
    *,
    cluster_id: str,
) -> UnifiedReasoningPortfolioEntry:
    if profile.operator_id not in _CANONICAL_SLOTS:
        raise ValueError(f"unsupported portfolio operator: {profile.operator_id}")
    slot_id: ReasoningModeSlotId = profile.operator_id  # type: ignore[assignment]
    expected = _EXPECTED_TRANSFORM[slot_id]
    if profile.representation_transform != expected:
        raise ValueError(
            f"operator {slot_id} has unexpected representation transform "
            f"{profile.representation_transform}"
        )
    return UnifiedReasoningPortfolioEntry(
        candidate_id=profile.candidate_id,
        operator_id=slot_id,
        representation_transform=profile.representation_transform,
        premise_count=len(profile.premise_statement_ids),
        gap_count=len(profile.gap_statement_ids),
        prediction_observable_count=len(profile.prediction_observables),
        test_observable_count=len(profile.test_observables),
        scientific_neighborhood_cluster_id=cluster_id,
    )


def build_unified_reasoning_portfolio(
    report: ReasoningModeContrastReport,
) -> UnifiedScientificReasoningShadowPortfolio:
    clusters, membership = _cluster_rows(report)
    entries = [
        _entry(profile, cluster_id=membership[profile.candidate_id])
        for profile in report.candidate_profiles
    ]

    by_operator: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        by_operator[entry.operator_id].append(entry.candidate_id)
    for values in by_operator.values():
        values.sort()

    slots = [
        UnifiedReasoningModeSlot(
            slot_id=slot_id,
            representation_transform=_EXPECTED_TRANSFORM[slot_id],
            candidate_ids=by_operator.get(slot_id, []),
            empty_reason=(
                None
                if by_operator.get(slot_id)
                else (
                    "No compiled shadow candidate for this reasoning mode was present "
                    "in the source mode-contrast report. The slot remains empty; no "
                    "candidate is manufactured to satisfy portfolio coverage."
                )
            ),
        )
        for slot_id in _CANONICAL_SLOTS
    ]
    occupied = sum(bool(row.candidate_ids) for row in slots)
    return UnifiedScientificReasoningShadowPortfolio(
        portfolio_id=_stable_portfolio_id(report),
        source_mode_contrast_report_id=report.report_id,
        source_task_id=report.source_task_id,
        entries=entries,
        slots=slots,
        scientific_neighborhood_clusters=clusters,
        candidate_count=len(entries),
        occupied_reasoning_mode_count=occupied,
        empty_reasoning_mode_count=len(slots) - occupied,
        scientific_neighborhood_cluster_count=len(clusters),
        relation_counts=dict(report.relation_counts),
    )
