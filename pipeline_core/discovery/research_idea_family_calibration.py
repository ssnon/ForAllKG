from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_semantics import assess_conceptual_family


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FamilyEdgeClass = Literal[
    "TIGHT",
    "PROGRAM",
    "ADJACENT_ONLY",
    "DISTINCT",
]


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


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _norm(value: object) -> str:
    text = str(value or "").casefold().strip()
    return re.sub(r"\s+", " ", text)


def _structured_entities(node: ResearchIdeaNode) -> set[str] | None:
    entities: set[str] = set()
    parsed_any = False
    for value in node.kernel.core_scientific_commitments:
        match = re.match(r"^\s*(.+?)\s*--(.+?)-->\s*(.+?)\s*$", str(value or ""))
        if match is None:
            continue
        parsed_any = True
        subject, _relation, obj = match.groups()
        entities.add(_norm(subject))
        entities.add(_norm(obj))
    return entities if parsed_any else None


def _entity_overlap(left: ResearchIdeaNode, right: ResearchIdeaNode) -> float | None:
    a = _structured_entities(left)
    b = _structured_entities(right)
    if a is None or b is None:
        return None
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class FamilyPairEdge(StrictModel):
    idea_id_a: str = Field(min_length=1)
    idea_id_b: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    similarity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    structured_entity_overlap: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_class: FamilyEdgeClass
    used_for_tight_neighborhood: bool
    used_for_program_family: bool
    diagnostic_only: Literal[True] = True
    family_assignment_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class FamilyGraphAssignment(StrictModel):
    idea_id: str = Field(min_length=1)
    generation_index: int = Field(ge=0)
    tight_neighborhood_key: str = Field(min_length=1)
    program_family_key: str = Field(min_length=1)
    tight_neighborhood_size: int = Field(ge=1)
    program_family_size: int = Field(ge=1)
    recomputable: Literal[True] = True
    immutable_idea_identity_field: Literal[False] = False
    production_selection_authority: Literal[False] = False


class FamilyThresholdCurvePoint(StrictModel):
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    connected_component_count: int = Field(ge=0)
    largest_component_size: int = Field(ge=0)
    singleton_component_count: int = Field(ge=0)


class FamilyCalibrationReport(StrictModel):
    schema_version: Literal[
        "scientific-program-family-calibration-shadow-v1"
    ] = "scientific-program-family-calibration-shadow-v1"
    report_id: str
    report_sha256: str
    idea_count: int = Field(ge=0)
    pair_count: int = Field(ge=0)
    tight_neighborhood_count: int = Field(ge=0)
    program_family_count: int = Field(ge=0)
    tight_singleton_count: int = Field(ge=0)
    program_singleton_count: int = Field(ge=0)
    assignments: list[FamilyGraphAssignment] = Field(default_factory=list)
    pair_edges: list[FamilyPairEdge] = Field(default_factory=list)
    relation_counts: dict[str, int] = Field(default_factory=dict)
    edge_class_counts: dict[str, int] = Field(default_factory=dict)
    generation_program_family_counts: dict[str, int] = Field(default_factory=dict)
    program_family_birth_count_by_generation: dict[str, int] = Field(default_factory=dict)
    program_family_size_histogram: dict[str, int] = Field(default_factory=dict)
    threshold_curve: list[FamilyThresholdCurvePoint] = Field(default_factory=list)
    program_similarity_floor: float = Field(ge=0.0, le=1.0)
    tight_relation_required: Literal["SAME_FAMILY"] = "SAME_FAMILY"
    adjacent_relation_can_join_program_above_floor: Literal[True] = True
    family_assignment_is_graph_based: Literal[True] = True
    family_assignment_is_soft_and_recomputable: Literal[True] = True
    family_is_not_research_idea_identity: Literal[True] = True
    family_is_not_hard_selection_gate: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "FamilyCalibrationReport":
        if self.idea_count != len(self.assignments):
            raise ValueError("idea_count mismatch")
        if self.pair_count != len(self.pair_edges):
            raise ValueError("pair_count mismatch")
        return self


class ProgramFamilySearchState(StrictModel):
    program_family_key: str = Field(min_length=1)
    member_idea_ids: list[str] = Field(default_factory=list)
    idea_count: int = Field(ge=0)
    generation_counts: dict[str, int] = Field(default_factory=dict)
    exact_kernel_unique_count: int = Field(ge=0)
    offspring_generation_llm_calls: int = Field(ge=0)
    realization_llm_calls: int = Field(ge=0)
    prospective_audit_llm_calls: int = Field(ge=0)
    total_tracked_llm_calls: int = Field(ge=0)
    compute_share: float = Field(ge=0.0, le=1.0)
    materialized_realization_count: int = Field(ge=0)
    usable_grounded_realization_count: int = Field(ge=0)
    not_operationalizable_count: int = Field(ge=0)
    soft_search_abstraction: Literal[True] = True
    hard_gate_authority: Literal[False] = False


class ProgramFamilyComputeReport(StrictModel):
    schema_version: Literal[
        "scientific-program-family-compute-shadow-v1"
    ] = "scientific-program-family-compute-shadow-v1"
    report_id: str
    report_sha256: str
    source_family_calibration_report_id: str
    states: list[ProgramFamilySearchState] = Field(default_factory=list)
    tracked_llm_calls: int = Field(ge=0)
    family_compute_hhi: float = Field(ge=0.0, le=1.0)
    largest_family_compute_share: float = Field(ge=0.0, le=1.0)
    overconcentrated_family_keys: list[str] = Field(default_factory=list)
    overconcentration_threshold: float = Field(ge=0.0, le=1.0)
    concentration_is_soft_search_pressure_only: Literal[True] = True
    family_is_not_hard_selection_gate: Literal[True] = True
    production_selection_authority: Literal[False] = False


class _UnionFind:
    def __init__(self, keys: Sequence[str]) -> None:
        self.parent = {key: key for key in keys}
        self.size = {key: 1 for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb] or (
            self.size[ra] == self.size[rb] and rb < ra
        ):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for key in sorted(self.parent):
            groups[self.find(key)].append(key)
        return sorted((sorted(rows) for rows in groups.values()), key=lambda rows: rows[0])


def _component_map(
    idea_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    prefix: str,
) -> tuple[dict[str, str], dict[str, int]]:
    uf = _UnionFind(idea_ids)
    for a, b in edges:
        uf.union(a, b)
    by_idea: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for members in uf.components():
        key = _stable_id(prefix, members)
        sizes[key] = len(members)
        for idea_id in members:
            by_idea[idea_id] = key
    return by_idea, sizes


def _curve_point(
    idea_ids: Sequence[str],
    similarities: Sequence[tuple[str, str, float]],
    threshold: float,
) -> FamilyThresholdCurvePoint:
    edges = [(a, b) for a, b, similarity in similarities if similarity >= threshold]
    mapping, sizes = _component_map(idea_ids, edges, prefix=f"family_curve_{threshold:.3f}")
    del mapping
    values = list(sizes.values())
    return FamilyThresholdCurvePoint(
        similarity_threshold=threshold,
        connected_component_count=len(values),
        largest_component_size=max(values, default=0),
        singleton_component_count=sum(value == 1 for value in values),
    )


def build_family_calibration(
    nodes: Sequence[ResearchIdeaNode],
    *,
    program_similarity_floor: float = 0.58,
    curve_thresholds: Sequence[float] = (0.42, 0.50, 0.58, 0.66, 0.78),
) -> FamilyCalibrationReport:
    """Build two soft family views from a complete pairwise similarity graph.

    Tight neighborhoods use only the existing SAME_FAMILY diagnostic. Broader
    scientific-program families additionally connect high-similarity ADJACENT_FAMILY
    pairs. Neither projection is ResearchIdea identity or a selection gate.
    """

    if not 0.0 <= program_similarity_floor <= 1.0:
        raise ValueError("program_similarity_floor must be in [0, 1]")
    ordered = sorted(nodes, key=lambda row: (row.generation_index, row.idea_id))
    ids = [row.idea_id for row in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("family calibration requires unique ResearchIdea IDs")

    tight_edges: list[tuple[str, str]] = []
    program_edges: list[tuple[str, str]] = []
    similarity_rows: list[tuple[str, str, float]] = []
    pair_rows: list[FamilyPairEdge] = []
    relation_counts: Counter[str] = Counter()
    edge_class_counts: Counter[str] = Counter()

    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            assessment = assess_conceptual_family(left, right)
            relation_counts[assessment.relation] += 1
            similarity_rows.append((left.idea_id, right.idea_id, assessment.similarity))
            tight = assessment.relation == "SAME_FAMILY"
            entity_overlap = _entity_overlap(left, right)
            structured_anchor_compatible = (
                entity_overlap is None or entity_overlap >= 0.25
            )
            program = tight or (
                assessment.relation == "ADJACENT_FAMILY"
                and assessment.similarity >= program_similarity_floor
                and structured_anchor_compatible
            )
            if tight:
                edge_class: FamilyEdgeClass = "TIGHT"
                tight_edges.append((left.idea_id, right.idea_id))
                program_edges.append((left.idea_id, right.idea_id))
            elif program:
                edge_class = "PROGRAM"
                program_edges.append((left.idea_id, right.idea_id))
            elif assessment.relation == "ADJACENT_FAMILY":
                edge_class = "ADJACENT_ONLY"
            else:
                edge_class = "DISTINCT"
            edge_class_counts[edge_class] += 1
            pair_rows.append(
                FamilyPairEdge(
                    idea_id_a=left.idea_id,
                    idea_id_b=right.idea_id,
                    relation=assessment.relation,
                    similarity=assessment.similarity,
                    confidence=assessment.confidence,
                    structured_entity_overlap=entity_overlap,
                    edge_class=edge_class,
                    used_for_tight_neighborhood=tight,
                    used_for_program_family=program,
                )
            )

    tight_map, tight_sizes = _component_map(ids, tight_edges, prefix="tight_neighborhood")
    program_map, program_sizes = _component_map(ids, program_edges, prefix="scientific_program_family")
    node_by_id = {row.idea_id: row for row in ordered}
    assignments = [
        FamilyGraphAssignment(
            idea_id=idea_id,
            generation_index=node_by_id[idea_id].generation_index,
            tight_neighborhood_key=tight_map[idea_id],
            program_family_key=program_map[idea_id],
            tight_neighborhood_size=tight_sizes[tight_map[idea_id]],
            program_family_size=program_sizes[program_map[idea_id]],
        )
        for idea_id in ids
    ]

    generation_programs: dict[str, set[str]] = defaultdict(set)
    first_generation: dict[str, int] = {}
    for row in assignments:
        generation_programs[f"G{row.generation_index}"].add(row.program_family_key)
        first_generation[row.program_family_key] = min(
            row.generation_index,
            first_generation.get(row.program_family_key, row.generation_index),
        )
    births = Counter(f"G{generation}" for generation in first_generation.values())
    size_histogram = Counter(str(value) for value in program_sizes.values())
    thresholds = sorted({float(value) for value in curve_thresholds})
    curve = [_curve_point(ids, similarity_rows, threshold) for threshold in thresholds]

    provisional = FamilyCalibrationReport(
        report_id="pending",
        report_sha256="pending",
        idea_count=len(ids),
        pair_count=len(pair_rows),
        tight_neighborhood_count=len(tight_sizes),
        program_family_count=len(program_sizes),
        tight_singleton_count=sum(size == 1 for size in tight_sizes.values()),
        program_singleton_count=sum(size == 1 for size in program_sizes.values()),
        assignments=assignments,
        pair_edges=pair_rows,
        relation_counts=dict(sorted(relation_counts.items())),
        edge_class_counts=dict(sorted(edge_class_counts.items())),
        generation_program_family_counts={
            key: len(value) for key, value in sorted(generation_programs.items())
        },
        program_family_birth_count_by_generation=dict(sorted(births.items())),
        program_family_size_histogram=dict(sorted(size_histogram.items(), key=lambda item: int(item[0]))),
        threshold_curve=curve,
        program_similarity_floor=program_similarity_floor,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"family_calibration:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def calibrate_frontier_population(
    population: Any,
    *,
    program_similarity_floor: float = 0.58,
    curve_thresholds: Sequence[float] = (0.42, 0.50, 0.58, 0.66, 0.78),
) -> FamilyCalibrationReport:
    from pipeline_core.discovery.research_idea_projection import project_frontier_idea

    nodes = [project_frontier_idea(idea, generation_index=0) for idea in population.ideas]
    return build_family_calibration(
        nodes,
        program_similarity_floor=program_similarity_floor,
        curve_thresholds=curve_thresholds,
    )


def build_program_family_compute_report(
    *,
    calibration: FamilyCalibrationReport,
    nodes: Sequence[ResearchIdeaNode],
    generation2_execution: Any,
    generation3_execution: Any,
    generation2_lifecycle: Any,
    generation3_lifecycle: Any,
    active_reports: Sequence[Any] = (),
    overconcentration_threshold: float = 0.35,
) -> ProgramFamilyComputeReport:
    if not 0.0 <= overconcentration_threshold <= 1.0:
        raise ValueError("overconcentration_threshold must be in [0, 1]")
    family_by_idea = {row.idea_id: row.program_family_key for row in calibration.assignments}
    node_by_id = {row.idea_id: row for row in nodes}
    members: dict[str, list[str]] = defaultdict(list)
    for idea_id, family_key in family_by_idea.items():
        members[family_key].append(idea_id)

    offspring_calls: Counter[str] = Counter()
    for execution in (generation2_execution, generation3_execution):
        for row in getattr(execution, "run_records", []):
            family_key = family_by_idea.get(str(row.primary_parent_idea_id))
            if family_key:
                offspring_calls[family_key] += int(row.llm_call_count or 0)

    realization_calls: Counter[str] = Counter()
    prospective_calls: Counter[str] = Counter()
    materialized: Counter[str] = Counter()
    usable: Counter[str] = Counter()
    not_op: Counter[str] = Counter()
    for lifecycle in (generation2_lifecycle, generation3_lifecycle):
        observations_by_realization = {
            str(row.realization_id): row for row in getattr(lifecycle, "observations", [])
        }
        for link in getattr(lifecycle, "links", []):
            family_key = family_by_idea.get(str(link.idea_id))
            if not family_key:
                continue
            realization_calls[family_key] += int(link.llm_call_count or 0)
            materialized[family_key] += int(str(link.materialization_status) == "MATERIALIZED")
            observation = observations_by_realization.get(str(link.realization_id))
            if observation is not None:
                if observation.prospective_status is not None:
                    prospective_calls[family_key] += 1
                is_not_op = (
                    observation.prospective_status == "COMPLETE"
                    and observation.prospective_contract_integrity_passed is True
                    and observation.prospective_identifiability == "NOT_OPERATIONALIZABLE"
                )
                not_op[family_key] += int(is_not_op)
                usable[family_key] += int(
                    str(link.materialization_status) == "MATERIALIZED" and not is_not_op
                )

    for active in active_reports:
        for record in getattr(active, "action_records", []):
            family_key = family_by_idea.get(str(record.idea_id))
            if family_key:
                realization_calls[family_key] += int(record.llm_call_count or 0)
        for observation in getattr(active, "new_observations", []):
            family_key = family_by_idea.get(str(observation.idea_id))
            if family_key and observation.prospective_status is not None:
                prospective_calls[family_key] += 1

    total_by_family: dict[str, int] = {}
    for family_key in members:
        total_by_family[family_key] = (
            offspring_calls[family_key]
            + realization_calls[family_key]
            + prospective_calls[family_key]
        )
    total_calls = sum(total_by_family.values())
    states: list[ProgramFamilySearchState] = []
    for family_key, idea_ids in sorted(members.items()):
        generations = Counter(
            f"G{node_by_id[idea_id].generation_index}"
            for idea_id in idea_ids
            if idea_id in node_by_id
        )
        kernels = {
            node_by_id[idea_id].kernel_sha256
            for idea_id in idea_ids
            if idea_id in node_by_id
        }
        tracked = total_by_family[family_key]
        share = tracked / total_calls if total_calls else 0.0
        states.append(
            ProgramFamilySearchState(
                program_family_key=family_key,
                member_idea_ids=sorted(idea_ids),
                idea_count=len(idea_ids),
                generation_counts=dict(sorted(generations.items())),
                exact_kernel_unique_count=len(kernels),
                offspring_generation_llm_calls=offspring_calls[family_key],
                realization_llm_calls=realization_calls[family_key],
                prospective_audit_llm_calls=prospective_calls[family_key],
                total_tracked_llm_calls=tracked,
                compute_share=share,
                materialized_realization_count=materialized[family_key],
                usable_grounded_realization_count=usable[family_key],
                not_operationalizable_count=not_op[family_key],
            )
        )
    hhi = sum(row.compute_share ** 2 for row in states)
    largest = max((row.compute_share for row in states), default=0.0)
    over = [
        row.program_family_key
        for row in states
        if row.compute_share > overconcentration_threshold
    ]
    provisional = ProgramFamilyComputeReport(
        report_id="pending",
        report_sha256="pending",
        source_family_calibration_report_id=calibration.report_id,
        states=states,
        tracked_llm_calls=total_calls,
        family_compute_hhi=hhi,
        largest_family_compute_share=largest,
        overconcentrated_family_keys=over,
        overconcentration_threshold=overconcentration_threshold,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"program_family_compute:{digest[:20]}",
            "report_sha256": digest,
        }
    )


__all__ = [
    "FamilyCalibrationReport",
    "FamilyGraphAssignment",
    "FamilyPairEdge",
    "FamilyThresholdCurvePoint",
    "ProgramFamilyComputeReport",
    "ProgramFamilySearchState",
    "build_family_calibration",
    "build_program_family_compute_report",
    "calibrate_frontier_population",
]
