from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.frontier_idea_population import (
    FrontierIdea,
    FrontierIdeaPopulation,
    FrontierSourceKind,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_SOURCE_ORDER: dict[str, int] = {
    "KG_AXIS": 0,
    "OPEN_WORLD_AXIS": 1,
    "HIGHER_ORDER": 2,
    "DIRECT_HIGHER_ORDER": 3,
    "TENSION_DERIVED": 4,
}


class FrontierPrimitiveFamilyRecord(StrictModel):
    family_id: str
    normalized_relation: str
    representative_relation: str
    provider_source_kinds: list[FrontierSourceKind]
    provider_idea_ids: list[str]
    candidate_unit_ids: list[str] = Field(default_factory=list)
    topology_consumer_source_kinds: list[FrontierSourceKind] = Field(
        default_factory=list
    )
    topology_consumer_idea_ids: list[str] = Field(default_factory=list)
    source_exclusive: bool
    reused_in_topology: bool


class FrontierPrimitiveLayerAudit(StrictModel):
    relation_axis_idea_count: int = Field(ge=0)
    unique_primitive_family_count: int = Field(ge=0)
    unique_primitive_family_count_by_source_kind: dict[str, int]
    source_exclusive_primitive_family_count_by_source_kind: dict[str, int]
    cross_source_primitive_family_count: int = Field(ge=0)
    primitive_family_reused_in_topology_count: int = Field(ge=0)
    kg_axis_primitive_reused_in_topology_count: int = Field(ge=0)
    open_world_axis_primitive_reused_in_topology_count: int = Field(ge=0)
    open_world_exclusive_primitive_family_count: int = Field(ge=0)
    open_world_uncomposed_primitive_family_count: int = Field(ge=0)
    families: list[FrontierPrimitiveFamilyRecord] = Field(default_factory=list)


class FrontierBackboneFamilyRecord(StrictModel):
    family_id: str
    task_source: str
    task_target: str
    backbone_relation_texts: list[str]
    source_kinds: list[FrontierSourceKind]
    topology_count: int = Field(ge=1)
    topology_idea_ids: list[str] = Field(min_length=1)
    unique_modifier_family_count: int = Field(ge=0)
    attachment_role_counts: dict[str, int] = Field(default_factory=dict)


class FrontierModifierFamilyRecord(StrictModel):
    family_id: str
    modifier_text: str
    source_kinds: list[FrontierSourceKind]
    topology_count: int = Field(ge=1)
    topology_idea_ids: list[str] = Field(min_length=1)
    attachment_roles: list[str] = Field(default_factory=list)
    modifier_relation_texts: list[str] = Field(default_factory=list)
    candidate_unit_ids: list[str] = Field(default_factory=list)


class FrontierTopologyLayerAudit(StrictModel):
    topology_idea_count: int = Field(ge=0)
    topology_idea_count_by_source_kind: dict[str, int]
    unique_backbone_family_count: int = Field(ge=0)
    unique_backbone_family_count_by_source_kind: dict[str, int]
    unique_modifier_family_count: int = Field(ge=0)
    unique_modifier_family_count_by_source_kind: dict[str, int]
    unique_modifier_attachment_family_count: int = Field(ge=0)
    candidate_modifier_topology_count: int = Field(ge=0)
    unverified_lineage_topology_count: int = Field(ge=0)
    backbone_family_with_multiple_modifiers_count: int = Field(ge=0)
    largest_backbone_family_size: int = Field(ge=0)
    largest_backbone_family_share: float = Field(ge=0.0, le=1.0)
    top3_backbone_family_share: float = Field(ge=0.0, le=1.0)
    median_topologies_per_backbone: float = Field(ge=0.0)
    mean_topologies_per_backbone: float = Field(ge=0.0)
    attachment_role_counts: dict[str, int]
    backbone_families: list[FrontierBackboneFamilyRecord] = Field(
        default_factory=list
    )
    modifier_families: list[FrontierModifierFamilyRecord] = Field(
        default_factory=list
    )


class FrontierInterpretiveLineageRecord(StrictModel):
    tension_idea_id: str
    tension_id: str
    tension_type: str
    source_context_ids: list[str]
    linked_higher_order_topology_ids: list[str] = Field(default_factory=list)
    linked_frontier_topology_idea_ids: list[str] = Field(default_factory=list)
    explanation_idea_ids: list[str] = Field(default_factory=list)
    exact_context_to_topology_lineage: bool
    exact_tension_to_explanation_lineage: bool
    candidate_inspiration_involved: bool
    verified_structural_derivation_depth: int = Field(ge=1, le=3)


class FrontierInterpretiveLayerAudit(StrictModel):
    tension_seed_count: int = Field(ge=0)
    tension_count_by_type: dict[str, int]
    unique_tension_family_count: int = Field(ge=0)
    unique_tension_basis_relation_family_count: int = Field(ge=0)
    competing_explanation_seed_count: int = Field(ge=0)
    competing_explanation_pair_count: int = Field(ge=0)
    candidate_inspiration_tension_count: int = Field(ge=0)
    candidate_inspiration_explanation_seed_count: int = Field(ge=0)
    exact_context_linked_tension_count: int = Field(ge=0)
    exact_tension_linked_explanation_seed_count: int = Field(ge=0)
    exact_full_chain_linked_explanation_seed_count: int = Field(ge=0)
    unresolved_tension_context_id_count: int = Field(ge=0)
    max_verified_structural_derivation_depth: int = Field(ge=0, le=3)
    lineage_records: list[FrontierInterpretiveLineageRecord] = Field(
        default_factory=list
    )


class FrontierSourcePairReuse(StrictModel):
    source_kind_a: FrontierSourceKind
    source_kind_b: FrontierSourceKind
    shared_relation_component_family_count: int = Field(ge=0)
    shared_candidate_unit_count: int = Field(ge=0)
    example_shared_relations: list[str] = Field(default_factory=list)


class FrontierCrossSourceAudit(StrictModel):
    unique_relation_component_family_count: int = Field(ge=0)
    relation_component_family_count_by_source_kind: dict[str, int]
    source_exclusive_relation_component_family_count_by_source_kind: dict[
        str, int
    ]
    cross_source_shared_relation_component_family_count: int = Field(ge=0)
    structural_overlap_pair_count: int = Field(ge=0)
    lexical_overlap_pair_count: int = Field(ge=0)
    unique_structural_shared_relation_count: int = Field(ge=0)
    unique_structural_shared_candidate_unit_count: int = Field(ge=0)
    source_pair_reuse: list[FrontierSourcePairReuse] = Field(default_factory=list)


class FrontierCapacityAudit(StrictModel):
    general_higher_order_topology_count: int = Field(ge=0)
    direct_higher_order_topology_count: int = Field(ge=0)
    higher_order_topology_cap: int | None = Field(default=None, ge=1)
    higher_order_topology_cap_source: str
    higher_order_topology_cap_reached: bool | None = None
    higher_order_topology_cap_utilization: float | None = Field(
        default=None,
        ge=0.0,
    )
    generation_report_present: bool
    strict_backbone_count: int | None = Field(default=None, ge=0)
    strict_two_component_backbone_count: int | None = Field(default=None, ge=0)
    strict_three_component_backbone_count: int | None = Field(default=None, ge=0)
    eligible_modifier_count: int | None = Field(default=None, ge=0)
    eligible_candidate_modifier_count: int | None = Field(default=None, ge=0)
    higher_order_context_count: int | None = Field(default=None, ge=0)
    generation_context_budget: int | None = Field(default=None, ge=0)
    generation_candidate_modifier_context_count: int | None = Field(
        default=None,
        ge=0,
    )
    generation_unique_candidate_modifier_count: int | None = Field(
        default=None,
        ge=0,
    )
    selected_context_count: int | None = Field(default=None, ge=0)
    proposed_count: int | None = Field(default=None, ge=0)


class FrontierExplorationAudit(StrictModel):
    schema_version: Literal[
        "frontier-exploration-audit-v1"
    ] = "frontier-exploration-audit-v1"

    audit_id: str
    audit_sha256: str
    population_id: str
    population_sha256: str
    source_context_id: str
    source_context_sha256: str
    research_question: str
    task_source: str
    task_target: str

    raw_total_idea_count: int = Field(ge=0)
    task_relation_mode_counts: dict[str, int]
    subordinate_idea_fraction: float = Field(ge=0.0, le=1.0)

    primitive_layer: FrontierPrimitiveLayerAudit
    topology_layer: FrontierTopologyLayerAudit
    interpretive_layer: FrontierInterpretiveLayerAudit
    cross_source: FrontierCrossSourceAudit
    capacity: FrontierCapacityAudit

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    generation_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class FrontierExplorationCohortCase(StrictModel):
    case_id: str
    audit_id: str
    population_id: str
    research_question: str
    raw_total_idea_count: int = Field(ge=0)
    unique_primitive_family_count: int = Field(ge=0)
    unique_backbone_family_count: int = Field(ge=0)
    unique_modifier_family_count: int = Field(ge=0)
    topology_idea_count: int = Field(ge=0)
    largest_backbone_family_share: float = Field(ge=0.0, le=1.0)
    open_world_exclusive_primitive_family_count: int = Field(ge=0)
    open_world_composed_primitive_family_count: int = Field(ge=0)
    tension_seed_count: int = Field(ge=0)
    competing_explanation_pair_count: int = Field(ge=0)
    candidate_modifier_topology_count: int = Field(ge=0)
    candidate_inspiration_tension_count: int = Field(ge=0)
    max_verified_structural_derivation_depth: int = Field(ge=0, le=3)
    higher_order_topology_cap_reached: bool | None = None
    subordinate_idea_fraction: float = Field(ge=0.0, le=1.0)


class FrontierExplorationCohortAudit(StrictModel):
    schema_version: Literal[
        "frontier-exploration-cohort-audit-v1"
    ] = "frontier-exploration-cohort-audit-v1"

    cohort_id: str
    cohort_sha256: str
    case_count: int = Field(ge=1)
    cases: list[FrontierExplorationCohortCase] = Field(min_length=1)

    total_raw_idea_count: int = Field(ge=0)
    case_count_with_topology_cap_reached: int = Field(ge=0)
    case_count_with_open_world_exclusive_primitives: int = Field(ge=0)
    case_count_with_open_world_topology_reuse: int = Field(ge=0)
    case_count_with_interpretive_branches: int = Field(ge=0)
    case_count_with_candidate_modifier_topologies: int = Field(ge=0)
    case_count_with_candidate_driven_tensions: int = Field(ge=0)
    case_count_with_verified_depth_3: int = Field(ge=0)

    median_raw_idea_count: float = Field(ge=0.0)
    median_unique_primitive_family_count: float = Field(ge=0.0)
    median_unique_backbone_family_count: float = Field(ge=0.0)
    median_unique_modifier_family_count: float = Field(ge=0.0)
    median_largest_backbone_family_share: float = Field(ge=0.0, le=1.0)

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    cross_case_winner_selected: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    generation_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False


_TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9α-ω가-힣]+")


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = _TOKEN_CLEAN_RE.sub(" ", text)
    return " ".join(text.split())


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}:{_sha256_text(raw)[:length]}"


def _relation_text(idea: FrontierIdea) -> str | None:
    signature = idea.relation_signature
    if signature is None:
        return None
    return (
        f"{signature.subject} --{signature.relation}--> "
        f"{signature.object}"
    )


def _idea_relation_components(idea: FrontierIdea) -> list[tuple[str, str, str]]:
    """Return (role, relation_text, candidate_unit_id) components."""
    rows: list[tuple[str, str, str]] = []

    if idea.relation_signature is not None:
        text = _relation_text(idea)
        if text:
            rows.append(
                (
                    "PRIMARY_RELATION",
                    text,
                    idea.relation_signature.candidate_unit_id,
                )
            )

    if idea.topology_signature is not None:
        sig = idea.topology_signature
        if sig.modifier_relation_text:
            rows.append(
                (
                    "MODIFIER_RELATION",
                    sig.modifier_relation_text,
                    sig.modifier_candidate_unit_id,
                )
            )
        rows.extend(
            (
                "BACKBONE_RELATION",
                relation_text,
                "",
            )
            for relation_text in sig.backbone_relation_texts
            if str(relation_text).strip()
        )

    if idea.tension_signature is not None:
        rows.extend(
            ("BASIS_RELATION", relation_text, "")
            for relation_text in idea.tension_signature.basis_relation_texts
            if str(relation_text).strip()
        )

    if idea.competing_explanation_signature is not None:
        rows.extend(
            ("BASIS_RELATION", relation_text, "")
            for relation_text in (
                idea.competing_explanation_signature.basis_relation_texts
            )
            if str(relation_text).strip()
        )

    return rows


def _source_sort(values: set[str] | Sequence[str]) -> list[str]:
    return sorted(
        set(values),
        key=lambda value: (
            _SOURCE_ORDER.get(str(value), 999),
            str(value),
        ),
    )


def _primitive_layer(population: FrontierIdeaPopulation) -> FrontierPrimitiveLayerAudit:
    axis_ideas = [
        idea
        for idea in population.ideas
        if idea.idea_form == "RELATION_AXIS"
        and idea.relation_signature is not None
    ]
    topology_ideas = [
        idea
        for idea in population.ideas
        if idea.idea_form == "HIGHER_ORDER_TOPOLOGY"
        and idea.topology_signature is not None
    ]

    topology_by_relation: dict[str, list[FrontierIdea]] = defaultdict(list)
    for idea in topology_ideas:
        for _, relation_text, _ in _idea_relation_components(idea):
            key = _norm(relation_text)
            if key:
                topology_by_relation[key].append(idea)

    families: dict[str, dict[str, Any]] = {}
    for idea in axis_ideas:
        relation_text = _relation_text(idea)
        if not relation_text:
            continue
        key = _norm(relation_text)
        row = families.setdefault(
            key,
            {
                "representative": relation_text,
                "providers": [],
                "provider_ids": [],
                "candidate_unit_ids": set(),
            },
        )
        row["providers"].append(idea.source_kind)
        row["provider_ids"].append(idea.idea_id)
        candidate_id = str(
            idea.relation_signature.candidate_unit_id or ""
        ).strip()
        if candidate_id:
            row["candidate_unit_ids"].add(candidate_id)

    records: list[FrontierPrimitiveFamilyRecord] = []
    for key in sorted(families):
        row = families[key]
        providers = _source_sort(set(row["providers"]))
        consumers = topology_by_relation.get(key, [])
        consumer_sources = _source_sort(
            {idea.source_kind for idea in consumers}
        )
        records.append(
            FrontierPrimitiveFamilyRecord(
                family_id=_stable_id("frontier_primitive_family", key),
                normalized_relation=key,
                representative_relation=row["representative"],
                provider_source_kinds=providers,
                provider_idea_ids=sorted(set(row["provider_ids"])),
                candidate_unit_ids=sorted(row["candidate_unit_ids"]),
                topology_consumer_source_kinds=consumer_sources,
                topology_consumer_idea_ids=sorted(
                    {idea.idea_id for idea in consumers}
                ),
                source_exclusive=len(providers) == 1,
                reused_in_topology=bool(consumers),
            )
        )

    by_source: dict[str, int] = {}
    exclusive_by_source: dict[str, int] = {}
    for source in _SOURCE_ORDER:
        if source not in {"KG_AXIS", "OPEN_WORLD_AXIS"}:
            continue
        by_source[source] = sum(
            source in row.provider_source_kinds
            for row in records
        )
        exclusive_by_source[source] = sum(
            row.provider_source_kinds == [source]
            for row in records
        )

    open_world_rows = [
        row
        for row in records
        if "OPEN_WORLD_AXIS" in row.provider_source_kinds
    ]

    return FrontierPrimitiveLayerAudit(
        relation_axis_idea_count=len(axis_ideas),
        unique_primitive_family_count=len(records),
        unique_primitive_family_count_by_source_kind=by_source,
        source_exclusive_primitive_family_count_by_source_kind=(
            exclusive_by_source
        ),
        cross_source_primitive_family_count=sum(
            len(row.provider_source_kinds) > 1
            for row in records
        ),
        primitive_family_reused_in_topology_count=sum(
            row.reused_in_topology for row in records
        ),
        kg_axis_primitive_reused_in_topology_count=sum(
            "KG_AXIS" in row.provider_source_kinds and row.reused_in_topology
            for row in records
        ),
        open_world_axis_primitive_reused_in_topology_count=sum(
            "OPEN_WORLD_AXIS" in row.provider_source_kinds
            and row.reused_in_topology
            for row in records
        ),
        open_world_exclusive_primitive_family_count=sum(
            row.provider_source_kinds == ["OPEN_WORLD_AXIS"]
            for row in records
        ),
        open_world_uncomposed_primitive_family_count=sum(
            not row.reused_in_topology
            for row in open_world_rows
        ),
        families=records,
    )


def _backbone_key(idea: FrontierIdea) -> tuple[str, str, tuple[str, ...]]:
    sig = idea.topology_signature
    assert sig is not None
    return (
        _norm(sig.task_source),
        _norm(sig.task_target),
        tuple(_norm(row) for row in sig.backbone_relation_texts),
    )


def _modifier_key(idea: FrontierIdea) -> str:
    sig = idea.topology_signature
    assert sig is not None
    value = _norm(sig.modifier_text)
    if value:
        return value
    value = _norm(sig.modifier_relation_text)
    if value:
        return value
    return "<unspecified-modifier>"


def _attachment_key(idea: FrontierIdea) -> tuple[str, str, str]:
    sig = idea.topology_signature
    assert sig is not None
    return (
        _modifier_key(idea),
        _norm(sig.modifier_anchor_role),
        _norm(sig.modifier_anchor_text),
    )


def _topology_layer(population: FrontierIdeaPopulation) -> FrontierTopologyLayerAudit:
    ideas = [
        idea
        for idea in population.ideas
        if idea.idea_form == "HIGHER_ORDER_TOPOLOGY"
        and idea.topology_signature is not None
    ]

    by_source = Counter(idea.source_kind for idea in ideas)
    backbone_rows: dict[tuple[str, str, tuple[str, ...]], list[FrontierIdea]] = defaultdict(list)
    modifier_rows: dict[str, list[FrontierIdea]] = defaultdict(list)
    attachment_keys = set()
    attachment_role_counts = Counter()

    for idea in ideas:
        backbone_rows[_backbone_key(idea)].append(idea)
        modifier_rows[_modifier_key(idea)].append(idea)
        attachment_keys.add(_attachment_key(idea))
        role = idea.topology_signature.modifier_anchor_role or "UNSPECIFIED"
        attachment_role_counts[role] += 1

    backbone_records: list[FrontierBackboneFamilyRecord] = []
    for key in sorted(backbone_rows, key=lambda row: (row[0], row[1], row[2])):
        rows = backbone_rows[key]
        first = rows[0].topology_signature
        assert first is not None
        modifier_keys = {_modifier_key(row) for row in rows}
        role_counts = Counter(
            row.topology_signature.modifier_anchor_role or "UNSPECIFIED"
            for row in rows
            if row.topology_signature is not None
        )
        backbone_records.append(
            FrontierBackboneFamilyRecord(
                family_id=_stable_id(
                    "frontier_backbone_family",
                    key[0],
                    key[1],
                    *key[2],
                ),
                task_source=first.task_source,
                task_target=first.task_target,
                backbone_relation_texts=list(first.backbone_relation_texts),
                source_kinds=_source_sort(
                    {row.source_kind for row in rows}
                ),
                topology_count=len(rows),
                topology_idea_ids=sorted(row.idea_id for row in rows),
                unique_modifier_family_count=len(modifier_keys),
                attachment_role_counts=dict(sorted(role_counts.items())),
            )
        )

    modifier_records: list[FrontierModifierFamilyRecord] = []
    for key in sorted(modifier_rows):
        rows = modifier_rows[key]
        representative = next(
            (
                row.topology_signature.modifier_text
                for row in rows
                if row.topology_signature is not None
                and row.topology_signature.modifier_text
            ),
            key,
        )
        modifier_records.append(
            FrontierModifierFamilyRecord(
                family_id=_stable_id("frontier_modifier_family", key),
                modifier_text=representative,
                source_kinds=_source_sort(
                    {row.source_kind for row in rows}
                ),
                topology_count=len(rows),
                topology_idea_ids=sorted(row.idea_id for row in rows),
                attachment_roles=sorted(
                    {
                        row.topology_signature.modifier_anchor_role
                        or "UNSPECIFIED"
                        for row in rows
                        if row.topology_signature is not None
                    }
                ),
                modifier_relation_texts=sorted(
                    {
                        row.topology_signature.modifier_relation_text
                        for row in rows
                        if row.topology_signature is not None
                        and row.topology_signature.modifier_relation_text
                    }
                ),
                candidate_unit_ids=sorted(
                    {
                        row.topology_signature.modifier_candidate_unit_id
                        for row in rows
                        if row.topology_signature is not None
                        and row.topology_signature.modifier_candidate_unit_id
                    }
                ),
            )
        )

    family_sizes = sorted(
        (len(rows) for rows in backbone_rows.values()),
        reverse=True,
    )
    total = len(ideas)
    largest = family_sizes[0] if family_sizes else 0
    top3 = sum(family_sizes[:3])

    source_backbones: dict[str, set[tuple[str, str, tuple[str, ...]]]] = defaultdict(set)
    source_modifiers: dict[str, set[str]] = defaultdict(set)
    for idea in ideas:
        source_backbones[idea.source_kind].add(_backbone_key(idea))
        source_modifiers[idea.source_kind].add(_modifier_key(idea))

    return FrontierTopologyLayerAudit(
        topology_idea_count=total,
        topology_idea_count_by_source_kind=dict(sorted(by_source.items())),
        unique_backbone_family_count=len(backbone_rows),
        unique_backbone_family_count_by_source_kind={
            source: len(keys)
            for source, keys in sorted(source_backbones.items())
        },
        unique_modifier_family_count=len(modifier_rows),
        unique_modifier_family_count_by_source_kind={
            source: len(keys)
            for source, keys in sorted(source_modifiers.items())
        },
        unique_modifier_attachment_family_count=len(attachment_keys),
        candidate_modifier_topology_count=sum(
            bool(idea.topology_signature.modifier_candidate_unit_id)
            for idea in ideas
            if idea.topology_signature is not None
        ),
        unverified_lineage_topology_count=sum(
            any(
                lineage.candidate_or_unverified_lineage
                for lineage in idea.source_lineage
            )
            for idea in ideas
        ),
        backbone_family_with_multiple_modifiers_count=sum(
            row.unique_modifier_family_count > 1
            for row in backbone_records
        ),
        largest_backbone_family_size=largest,
        largest_backbone_family_share=(largest / total if total else 0.0),
        top3_backbone_family_share=(top3 / total if total else 0.0),
        median_topologies_per_backbone=(
            float(statistics.median(family_sizes))
            if family_sizes
            else 0.0
        ),
        mean_topologies_per_backbone=(
            float(statistics.mean(family_sizes))
            if family_sizes
            else 0.0
        ),
        attachment_role_counts=dict(sorted(attachment_role_counts.items())),
        backbone_families=backbone_records,
        modifier_families=modifier_records,
    )


def _context_to_topology(
    contexts: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for index, row in enumerate(contexts, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"higher-order context row {index} is not an object"
            )
        context_id = str(row.get("context_id") or "").strip()
        topology_id = str(
            row.get("higher_order_topology_id") or ""
        ).strip()
        if not context_id or not topology_id:
            raise ValueError(
                "higher-order context requires context_id and "
                "higher_order_topology_id"
            )
        previous = mapping.get(context_id)
        if previous is not None and previous != topology_id:
            raise ValueError(
                f"context {context_id!r} maps to multiple topology ids"
            )
        mapping[context_id] = topology_id
    return mapping


def _interpretive_layer(
    population: FrontierIdeaPopulation,
    *,
    higher_order_contexts: Sequence[Mapping[str, Any]],
) -> FrontierInterpretiveLayerAudit:
    tensions = [
        idea
        for idea in population.ideas
        if idea.idea_form == "TENSION_SEED"
        and idea.tension_signature is not None
    ]
    explanations = [
        idea
        for idea in population.ideas
        if idea.idea_form == "COMPETING_EXPLANATION_SEED"
        and idea.competing_explanation_signature is not None
    ]
    topology_by_object_id = {
        lineage.source_object_id: idea
        for idea in population.ideas
        if idea.idea_form == "HIGHER_ORDER_TOPOLOGY"
        for lineage in idea.source_lineage[:1]
    }
    context_map = _context_to_topology(higher_order_contexts)

    tension_by_object_id = {
        idea.source_lineage[0].source_object_id: idea
        for idea in tensions
    }
    explanations_by_tension: dict[str, list[FrontierIdea]] = defaultdict(list)
    for idea in explanations:
        sig = idea.competing_explanation_signature
        assert sig is not None
        explanations_by_tension[sig.tension_id].append(idea)

    tension_family_keys = set()
    basis_keys = set()
    type_counts = Counter()
    lineage_records: list[FrontierInterpretiveLineageRecord] = []
    unresolved_context_ids = set()
    exact_linked_tensions = 0
    exact_linked_explanations = 0
    exact_full_chain_explanations = 0
    max_depth = 1 if (tensions or explanations) else (1 if any(
        idea.idea_form == "HIGHER_ORDER_TOPOLOGY"
        for idea in population.ideas
    ) else 0)

    for idea in tensions:
        sig = idea.tension_signature
        assert sig is not None
        basis = tuple(sorted(_norm(row) for row in sig.basis_relation_texts))
        tension_family_keys.add(
            (
                _norm(sig.tension_type),
                _norm(sig.requested_source),
                _norm(sig.requested_target),
                basis,
            )
        )
        basis_keys.update(key for key in basis if key)
        type_counts[sig.tension_type] += 1

        lineage = idea.source_lineage[0]
        linked_topology_ids = []
        linked_frontier_ids = []
        for context_id in lineage.source_context_ids:
            topology_id = context_map.get(context_id)
            if topology_id is None:
                unresolved_context_ids.add(context_id)
                continue
            linked_topology_ids.append(topology_id)
            topology_idea = topology_by_object_id.get(topology_id)
            if topology_idea is not None:
                linked_frontier_ids.append(topology_idea.idea_id)

        exact_context_link = bool(linked_frontier_ids)
        if exact_context_link:
            exact_linked_tensions += 1
            max_depth = max(max_depth, 2)

        tension_id = lineage.source_object_id
        explanation_rows = explanations_by_tension.get(tension_id, [])
        exact_explanation_link = bool(explanation_rows)
        if exact_explanation_link:
            exact_linked_explanations += len(explanation_rows)
            if exact_context_link:
                exact_full_chain_explanations += len(explanation_rows)
                max_depth = max(max_depth, 3)

        lineage_records.append(
            FrontierInterpretiveLineageRecord(
                tension_idea_id=idea.idea_id,
                tension_id=tension_id,
                tension_type=sig.tension_type,
                source_context_ids=list(lineage.source_context_ids),
                linked_higher_order_topology_ids=sorted(set(linked_topology_ids)),
                linked_frontier_topology_idea_ids=sorted(
                    set(linked_frontier_ids)
                ),
                explanation_idea_ids=sorted(
                    row.idea_id for row in explanation_rows
                ),
                exact_context_to_topology_lineage=exact_context_link,
                exact_tension_to_explanation_lineage=exact_explanation_link,
                candidate_inspiration_involved=any(
                    x.candidate_or_unverified_lineage
                    for x in idea.source_lineage
                ),
                verified_structural_derivation_depth=(
                    3
                    if exact_context_link and exact_explanation_link
                    else 2
                    if exact_context_link
                    else 1
                ),
            )
        )

    orphan_explanations = [
        idea
        for idea in explanations
        if (
            idea.competing_explanation_signature is not None
            and idea.competing_explanation_signature.tension_id
            not in tension_by_object_id
        )
    ]
    if orphan_explanations:
        # A population carrying explanation seeds without their tension seed
        # would make derivation-depth accounting misleading.
        raise ValueError(
            "competing explanation seed references a tension absent from "
            "the Frontier population"
        )

    explanation_pair_count = sum(
        1
        for rows in explanations_by_tension.values()
        if len(rows) >= 2
    )

    return FrontierInterpretiveLayerAudit(
        tension_seed_count=len(tensions),
        tension_count_by_type=dict(sorted(type_counts.items())),
        unique_tension_family_count=len(tension_family_keys),
        unique_tension_basis_relation_family_count=len(basis_keys),
        competing_explanation_seed_count=len(explanations),
        competing_explanation_pair_count=explanation_pair_count,
        candidate_inspiration_tension_count=sum(
            any(
                lineage.candidate_or_unverified_lineage
                for lineage in idea.source_lineage
            )
            for idea in tensions
        ),
        candidate_inspiration_explanation_seed_count=sum(
            any(
                lineage.candidate_or_unverified_lineage
                for lineage in idea.source_lineage
            )
            for idea in explanations
        ),
        exact_context_linked_tension_count=exact_linked_tensions,
        exact_tension_linked_explanation_seed_count=exact_linked_explanations,
        exact_full_chain_linked_explanation_seed_count=(
            exact_full_chain_explanations
        ),
        unresolved_tension_context_id_count=len(unresolved_context_ids),
        max_verified_structural_derivation_depth=max_depth,
        lineage_records=sorted(
            lineage_records,
            key=lambda row: (row.tension_type, row.tension_id),
        ),
    )


def _cross_source(population: FrontierIdeaPopulation) -> FrontierCrossSourceAudit:
    source_relation_sets: dict[str, set[str]] = defaultdict(set)
    relation_repr: dict[str, str] = {}
    source_candidate_sets: dict[str, set[str]] = defaultdict(set)

    for idea in population.ideas:
        for _, relation_text, candidate_unit_id in _idea_relation_components(idea):
            key = _norm(relation_text)
            if key:
                source_relation_sets[idea.source_kind].add(key)
                relation_repr.setdefault(key, relation_text)
            candidate_unit_id = str(candidate_unit_id or "").strip()
            if candidate_unit_id:
                source_candidate_sets[idea.source_kind].add(candidate_unit_id)

    all_relation_keys = set().union(*source_relation_sets.values()) if source_relation_sets else set()
    relation_sources: dict[str, set[str]] = defaultdict(set)
    for source, keys in source_relation_sets.items():
        for key in keys:
            relation_sources[key].add(source)

    pair_rows: list[FrontierSourcePairReuse] = []
    present_sources = _source_sort(set(source_relation_sets) | set(source_candidate_sets))
    for left, right in combinations(present_sources, 2):
        shared_relations = source_relation_sets[left] & source_relation_sets[right]
        shared_candidates = source_candidate_sets[left] & source_candidate_sets[right]
        if not shared_relations and not shared_candidates:
            continue
        pair_rows.append(
            FrontierSourcePairReuse(
                source_kind_a=left,
                source_kind_b=right,
                shared_relation_component_family_count=len(shared_relations),
                shared_candidate_unit_count=len(shared_candidates),
                example_shared_relations=[
                    relation_repr[key]
                    for key in sorted(shared_relations)[:10]
                ],
            )
        )

    structural_shared_relations = {
        _norm(text)
        for row in population.structural_overlap_diagnostics
        for text in row.shared_relation_texts
        if _norm(text)
    }
    structural_shared_candidates = {
        candidate_id
        for row in population.structural_overlap_diagnostics
        for candidate_id in row.shared_candidate_unit_ids
        if str(candidate_id).strip()
    }

    return FrontierCrossSourceAudit(
        unique_relation_component_family_count=len(all_relation_keys),
        relation_component_family_count_by_source_kind={
            source: len(keys)
            for source, keys in sorted(source_relation_sets.items())
        },
        source_exclusive_relation_component_family_count_by_source_kind={
            source: sum(
                relation_sources[key] == {source}
                for key in keys
            )
            for source, keys in sorted(source_relation_sets.items())
        },
        cross_source_shared_relation_component_family_count=sum(
            len(sources) > 1
            for sources in relation_sources.values()
        ),
        structural_overlap_pair_count=(
            population.cross_source_structural_overlap_pair_count
        ),
        lexical_overlap_pair_count=population.cross_source_overlap_pair_count,
        unique_structural_shared_relation_count=len(structural_shared_relations),
        unique_structural_shared_candidate_unit_count=len(
            structural_shared_candidates
        ),
        source_pair_reuse=pair_rows,
    )


def _optional_nonnegative_int(
    payload: Mapping[str, Any],
    key: str,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _capacity(
    population: FrontierIdeaPopulation,
    *,
    higher_order_generation_report: Mapping[str, Any] | None,
    higher_order_topology_cap: int | None,
    higher_order_topology_cap_source: str,
) -> FrontierCapacityAudit:
    general_count = sum(
        idea.source_kind == "HIGHER_ORDER"
        and idea.idea_form == "HIGHER_ORDER_TOPOLOGY"
        for idea in population.ideas
    )
    direct_count = sum(
        idea.source_kind == "DIRECT_HIGHER_ORDER"
        and idea.idea_form == "HIGHER_ORDER_TOPOLOGY"
        for idea in population.ideas
    )

    if higher_order_topology_cap is not None and higher_order_topology_cap < 1:
        raise ValueError("higher_order_topology_cap must be >= 1")

    report = higher_order_generation_report
    if report is not None:
        report_topology_count = _optional_nonnegative_int(
            report,
            "higher_order_topology_count",
        )
        if (
            report_topology_count is not None
            and report_topology_count != general_count
        ):
            raise ValueError(
                "higher-order generation report topology count does not match "
                "the Frontier HIGHER_ORDER population: "
                f"report={report_topology_count}, population={general_count}"
            )

    cap_reached = (
        None
        if higher_order_topology_cap is None
        else general_count >= higher_order_topology_cap
    )
    utilization = (
        None
        if higher_order_topology_cap is None
        else general_count / higher_order_topology_cap
    )

    return FrontierCapacityAudit(
        general_higher_order_topology_count=general_count,
        direct_higher_order_topology_count=direct_count,
        higher_order_topology_cap=higher_order_topology_cap,
        higher_order_topology_cap_source=higher_order_topology_cap_source,
        higher_order_topology_cap_reached=cap_reached,
        higher_order_topology_cap_utilization=utilization,
        generation_report_present=report is not None,
        strict_backbone_count=(
            None if report is None else _optional_nonnegative_int(report, "strict_backbone_count")
        ),
        strict_two_component_backbone_count=(
            None if report is None else _optional_nonnegative_int(report, "strict_two_component_backbone_count")
        ),
        strict_three_component_backbone_count=(
            None if report is None else _optional_nonnegative_int(report, "strict_three_component_backbone_count")
        ),
        eligible_modifier_count=(
            None if report is None else _optional_nonnegative_int(report, "eligible_modifier_count")
        ),
        eligible_candidate_modifier_count=(
            None if report is None else _optional_nonnegative_int(report, "eligible_candidate_modifier_count")
        ),
        higher_order_context_count=(
            None if report is None else _optional_nonnegative_int(report, "higher_order_context_count")
        ),
        generation_context_budget=(
            None if report is None else _optional_nonnegative_int(report, "generation_context_budget")
        ),
        generation_candidate_modifier_context_count=(
            None if report is None else _optional_nonnegative_int(report, "generation_candidate_modifier_context_count")
        ),
        generation_unique_candidate_modifier_count=(
            None if report is None else _optional_nonnegative_int(report, "generation_unique_candidate_modifier_count")
        ),
        selected_context_count=(
            None if report is None else _optional_nonnegative_int(report, "selected_context_count")
        ),
        proposed_count=(
            None if report is None else _optional_nonnegative_int(report, "proposed_count")
        ),
    )


def build_frontier_exploration_audit(
    *,
    population: FrontierIdeaPopulation,
    higher_order_contexts: Sequence[Mapping[str, Any]] = (),
    higher_order_generation_report: Mapping[str, Any] | None = None,
    higher_order_topology_cap: int | None = None,
    higher_order_topology_cap_source: str = "UNSPECIFIED",
) -> FrontierExplorationAudit:
    primitive = _primitive_layer(population)
    topology = _topology_layer(population)
    interpretive = _interpretive_layer(
        population,
        higher_order_contexts=higher_order_contexts,
    )
    cross_source = _cross_source(population)
    capacity = _capacity(
        population,
        higher_order_generation_report=higher_order_generation_report,
        higher_order_topology_cap=higher_order_topology_cap,
        higher_order_topology_cap_source=higher_order_topology_cap_source,
    )

    total = population.total_idea_count
    subordinate = int(population.task_relation_mode_counts.get("SUBORDINATE", 0))

    audit_id = _stable_id(
        "frontier_exploration_audit",
        population.population_id,
        population.population_sha256,
    )

    body = {
        "schema_version": "frontier-exploration-audit-v1",
        "audit_id": audit_id,
        "population_id": population.population_id,
        "population_sha256": population.population_sha256,
        "source_context_id": population.source_context_id,
        "source_context_sha256": population.source_context_sha256,
        "research_question": population.research_question,
        "task_source": population.task_source,
        "task_target": population.task_target,
        "raw_total_idea_count": total,
        "task_relation_mode_counts": dict(population.task_relation_mode_counts),
        "subordinate_idea_fraction": (subordinate / total if total else 0.0),
        "primitive_layer": primitive.model_dump(mode="json"),
        "topology_layer": topology.model_dump(mode="json"),
        "interpretive_layer": interpretive.model_dump(mode="json"),
        "cross_source": cross_source.model_dump(mode="json"),
        "capacity": capacity.model_dump(mode="json"),
        "diagnostic_only": True,
        "scientific_quality_ranking_performed": False,
        "candidate_deletion_authority": False,
        "positive_premise_authority_created": False,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "production_selection_authority": False,
        "stage8_input_changed": False,
        "canonical_graph_mutated": False,
    }
    audit_sha256 = _sha256_text(_canonical_json(body))
    return FrontierExplorationAudit(
        **body,
        audit_sha256=audit_sha256,
    )


def build_frontier_exploration_cohort_audit(
    *,
    audits: Mapping[str, FrontierExplorationAudit],
) -> FrontierExplorationCohortAudit:
    if not audits:
        raise ValueError("at least one exploration audit is required")

    cases: list[FrontierExplorationCohortCase] = []
    for case_id in sorted(audits):
        audit = audits[case_id]
        open_world_total = audit.primitive_layer.unique_primitive_family_count_by_source_kind.get(
            "OPEN_WORLD_AXIS", 0
        )
        open_world_uncomposed = audit.primitive_layer.open_world_uncomposed_primitive_family_count
        cases.append(
            FrontierExplorationCohortCase(
                case_id=case_id,
                audit_id=audit.audit_id,
                population_id=audit.population_id,
                research_question=audit.research_question,
                raw_total_idea_count=audit.raw_total_idea_count,
                unique_primitive_family_count=(
                    audit.primitive_layer.unique_primitive_family_count
                ),
                unique_backbone_family_count=(
                    audit.topology_layer.unique_backbone_family_count
                ),
                unique_modifier_family_count=(
                    audit.topology_layer.unique_modifier_family_count
                ),
                topology_idea_count=audit.topology_layer.topology_idea_count,
                largest_backbone_family_share=(
                    audit.topology_layer.largest_backbone_family_share
                ),
                open_world_exclusive_primitive_family_count=(
                    audit.primitive_layer.open_world_exclusive_primitive_family_count
                ),
                open_world_composed_primitive_family_count=max(
                    0,
                    open_world_total - open_world_uncomposed,
                ),
                tension_seed_count=audit.interpretive_layer.tension_seed_count,
                competing_explanation_pair_count=(
                    audit.interpretive_layer.competing_explanation_pair_count
                ),
                candidate_modifier_topology_count=(
                    audit.topology_layer.candidate_modifier_topology_count
                ),
                candidate_inspiration_tension_count=(
                    audit.interpretive_layer.candidate_inspiration_tension_count
                ),
                max_verified_structural_derivation_depth=(
                    audit.interpretive_layer.max_verified_structural_derivation_depth
                ),
                higher_order_topology_cap_reached=(
                    audit.capacity.higher_order_topology_cap_reached
                ),
                subordinate_idea_fraction=audit.subordinate_idea_fraction,
            )
        )

    cohort_id = _stable_id(
        "frontier_exploration_cohort",
        *[
            f"{row.case_id}:{row.audit_id}"
            for row in cases
        ],
    )

    body = {
        "schema_version": "frontier-exploration-cohort-audit-v1",
        "cohort_id": cohort_id,
        "case_count": len(cases),
        "cases": [row.model_dump(mode="json") for row in cases],
        "total_raw_idea_count": sum(row.raw_total_idea_count for row in cases),
        "case_count_with_topology_cap_reached": sum(
            row.higher_order_topology_cap_reached is True for row in cases
        ),
        "case_count_with_open_world_exclusive_primitives": sum(
            row.open_world_exclusive_primitive_family_count > 0 for row in cases
        ),
        "case_count_with_open_world_topology_reuse": sum(
            row.open_world_composed_primitive_family_count > 0 for row in cases
        ),
        "case_count_with_interpretive_branches": sum(
            row.tension_seed_count > 0 or row.competing_explanation_pair_count > 0
            for row in cases
        ),
        "case_count_with_candidate_modifier_topologies": sum(
            row.candidate_modifier_topology_count > 0 for row in cases
        ),
        "case_count_with_candidate_driven_tensions": sum(
            row.candidate_inspiration_tension_count > 0 for row in cases
        ),
        "case_count_with_verified_depth_3": sum(
            row.max_verified_structural_derivation_depth >= 3 for row in cases
        ),
        "median_raw_idea_count": float(
            statistics.median(row.raw_total_idea_count for row in cases)
        ),
        "median_unique_primitive_family_count": float(
            statistics.median(row.unique_primitive_family_count for row in cases)
        ),
        "median_unique_backbone_family_count": float(
            statistics.median(row.unique_backbone_family_count for row in cases)
        ),
        "median_unique_modifier_family_count": float(
            statistics.median(row.unique_modifier_family_count for row in cases)
        ),
        "median_largest_backbone_family_share": float(
            statistics.median(row.largest_backbone_family_share for row in cases)
        ),
        "diagnostic_only": True,
        "scientific_quality_ranking_performed": False,
        "cross_case_winner_selected": False,
        "positive_premise_authority_created": False,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "production_selection_authority": False,
    }
    cohort_sha256 = _sha256_text(_canonical_json(body))
    return FrontierExplorationCohortAudit(
        **body,
        cohort_sha256=cohort_sha256,
    )


__all__ = [
    "FrontierBackboneFamilyRecord",
    "FrontierCapacityAudit",
    "FrontierCrossSourceAudit",
    "FrontierExplorationAudit",
    "FrontierExplorationCohortAudit",
    "FrontierExplorationCohortCase",
    "FrontierInterpretiveLayerAudit",
    "FrontierInterpretiveLineageRecord",
    "FrontierModifierFamilyRecord",
    "FrontierPrimitiveFamilyRecord",
    "FrontierPrimitiveLayerAudit",
    "FrontierSourcePairReuse",
    "FrontierTopologyLayerAudit",
    "build_frontier_exploration_audit",
    "build_frontier_exploration_cohort_audit",
]
