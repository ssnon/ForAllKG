from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


ProgramRelation = Literal[
    "SAME_PROGRAM",
    "ADJACENT_PROGRAM",
    "DISTINCT_PROGRAM",
]


class ScientificProgramPairAssessment(StrictModel):
    idea_id_a: str = Field(min_length=1)
    idea_id_b: str = Field(min_length=1)
    relation: ProgramRelation
    shared_mechanistic_backbone: str = ""
    distinguishing_axis: str = ""
    rationale: str = Field(min_length=1)

    diagnostic_only: Literal[True] = True
    research_idea_identity_authority: Literal[False] = False
    truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _different_ideas(self) -> "ScientificProgramPairAssessment":
        if self.idea_id_a == self.idea_id_b:
            raise ValueError("program-pair assessment requires two different ideas")
        return self


class ScientificProgramPairBatch(StrictModel):
    schema_version: Literal[
        "sis-v3-3-1-scientific-program-pair-batch-v1"
    ] = "sis-v3-3-1-scientific-program-pair-batch-v1"
    pairs: list[ScientificProgramPairAssessment] = Field(default_factory=list)


class ScientificProgramFamilyAssignment(StrictModel):
    idea_id: str = Field(min_length=1)
    scientific_program_key: str = Field(min_length=1)
    same_program_member_ids: list[str] = Field(min_length=1)
    same_program_size: int = Field(ge=1)
    adjacent_program_neighbor_ids: list[str] = Field(default_factory=list)
    adjacent_program_neighbor_count: int = Field(ge=0)
    distinct_program_neighbor_count: int = Field(ge=0)

    semantic_family_is_not_research_idea_identity: Literal[True] = True
    semantic_family_is_recomputable: Literal[True] = True
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "ScientificProgramFamilyAssignment":
        if self.same_program_size != len(self.same_program_member_ids):
            raise ValueError("same_program_size mismatch")
        if self.idea_id not in set(self.same_program_member_ids):
            raise ValueError("program family must contain its own idea")
        if self.adjacent_program_neighbor_count != len(
            self.adjacent_program_neighbor_ids
        ):
            raise ValueError("adjacent_program_neighbor_count mismatch")
        return self


class ScientificProgramFamilyReport(StrictModel):
    schema_version: Literal[
        "sis-v3-3-1-scientific-program-family-shadow-v1"
    ] = "sis-v3-3-1-scientific-program-family-shadow-v1"
    report_id: str
    report_sha256: str
    idea_count: int = Field(ge=0)
    pair_count: int = Field(ge=0)
    program_count: int = Field(ge=0)
    singleton_program_count: int = Field(ge=0)
    same_program_pair_count: int = Field(ge=0)
    adjacent_program_pair_count: int = Field(ge=0)
    distinct_program_pair_count: int = Field(ge=0)
    transitivity_diagnostic_count: int = Field(ge=0)
    relation_counts: dict[str, int] = Field(default_factory=dict)
    program_size_histogram: dict[str, int] = Field(default_factory=dict)
    assignments: list[ScientificProgramFamilyAssignment] = Field(default_factory=list)
    pair_assessments: list[ScientificProgramPairAssessment] = Field(default_factory=list)

    semantic_pair_assessment_is_diagnostic_only: Literal[True] = True
    family_is_not_hard_survival_gate: Literal[True] = True
    family_is_not_research_idea_identity: Literal[True] = True
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "ScientificProgramFamilyReport":
        if self.idea_count != len(self.assignments):
            raise ValueError("idea_count mismatch")
        if self.pair_count != len(self.pair_assessments):
            raise ValueError("pair_count mismatch")
        return self


class _UnionFind:
    def __init__(self, keys: Sequence[str]) -> None:
        self.parent = {key: key for key in keys}
        self.size = {key: 1 for key in keys}

    def find(self, key: str) -> str:
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb] or (
            self.size[ra] == self.size[rb] and rb < ra
        ):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for key in sorted(self.parent):
            grouped[self.find(key)].append(key)
        return sorted((sorted(rows) for rows in grouped.values()), key=lambda x: x[0])


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def validate_program_pair_batch(
    *,
    idea_ids: Sequence[str],
    batch: ScientificProgramPairBatch,
) -> None:
    ids = sorted(set(str(value) for value in idea_ids))
    expected = {_pair_key(a, b) for a, b in combinations(ids, 2)}
    observed: set[tuple[str, str]] = set()
    unknown: set[str] = set()
    for row in batch.pairs:
        if row.idea_id_a not in ids:
            unknown.add(row.idea_id_a)
        if row.idea_id_b not in ids:
            unknown.add(row.idea_id_b)
        key = _pair_key(row.idea_id_a, row.idea_id_b)
        if key in observed:
            raise ValueError(f"duplicate scientific-program pair assessment: {key}")
        observed.add(key)
    if unknown:
        raise ValueError(f"program pair batch contains unknown ideas: {sorted(unknown)}")
    if observed != expected:
        raise ValueError(
            "program pair batch does not cover every unordered pair exactly once; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )


def program_family_prompt_payload(nodes: Sequence[ResearchIdeaNode]) -> dict[str, Any]:
    ordered = sorted(nodes, key=lambda row: row.idea_id)
    return {
        "policy": {
            "SAME_PROGRAM": (
                "Same central mechanistic/causal backbone AND the same main "
                "perturbation or decisive-experiment class. Minor wording, scope, "
                "or measurement-detail changes do not create a new program."
            ),
            "ADJACENT_PROGRAM": (
                "Shares a central mechanistic backbone or scientific question, but "
                "uses a genuinely different perturbation, regime, mediator, causal "
                "contrast, or decisive discriminator that merits a separate branch."
            ),
            "DISTINCT_PROGRAM": (
                "Different central mechanistic commitment or scientific program; "
                "similar vocabulary alone is insufficient for adjacency."
            ),
        },
        "ideas": [
            {
                "idea_id": row.idea_id,
                "generation_index": row.generation_index,
                "canonical_intent": row.kernel.canonical_intent,
                "core_scientific_commitments": list(
                    row.kernel.core_scientific_commitments
                ),
                "scope_commitments": list(row.kernel.scope_commitments),
                "contrastive_commitments": list(
                    row.kernel.contrastive_commitments
                ),
                "question_commitment": row.kernel.question_commitment,
                "differential_prediction": row.differential_prediction,
                "falsification_condition": row.falsification_condition,
                "discriminating_observation": row.discriminating_observation,
            }
            for row in ordered
        ],
    }


PROGRAM_FAMILY_SYSTEM_PROMPT = r"""You are performing a semantic research-program
family audit over a bounded set of ResearchIdea objects.

This is NOT idea identity, truth, novelty, publication-value, or production selection.
Judge only whether two ideas belong to the same scientific research program.

Use the following distinctions strictly:
- SAME_PROGRAM: same central mechanistic/causal backbone AND same main perturbation
  or decisive-experiment class. Rewordings, narrow scope refinements, or a different
  measurement detail do not make a new program.
- ADJACENT_PROGRAM: same central backbone/question, but a genuinely different
  perturbation, mediator, regime boundary, competing-mechanism contrast, or decisive
  discriminator that is worth preserving as a separate branch.
- DISTINCT_PROGRAM: different central mechanistic commitment or scientific program.

Important:
- Do not rely on token overlap alone.
- Do not make outside scientific claims; use only supplied idea content.
- Assess EVERY unordered pair exactly once.
- Preserve idea IDs verbatim.
- The relation is diagnostic and recomputable; it never changes ResearchIdea identity.
"""


def build_scientific_program_family_report(
    *,
    nodes: Sequence[ResearchIdeaNode],
    pair_batch: ScientificProgramPairBatch,
) -> ScientificProgramFamilyReport:
    ordered_nodes = sorted(nodes, key=lambda row: row.idea_id)
    idea_ids = [row.idea_id for row in ordered_nodes]
    if len(idea_ids) != len(set(idea_ids)):
        raise ValueError("scientific-program family input contains duplicate idea IDs")
    validate_program_pair_batch(idea_ids=idea_ids, batch=pair_batch)

    relevant = [
        row
        for row in pair_batch.pairs
        if row.idea_id_a in set(idea_ids) and row.idea_id_b in set(idea_ids)
    ]
    by_pair = {_pair_key(row.idea_id_a, row.idea_id_b): row for row in relevant}

    uf = _UnionFind(idea_ids)
    for row in relevant:
        if row.relation == "SAME_PROGRAM":
            uf.union(row.idea_id_a, row.idea_id_b)

    components = uf.components()
    component_by_id: dict[str, list[str]] = {}
    key_by_id: dict[str, str] = {}
    for members in components:
        program_key = _stable_id("scientific_program", members)
        for idea_id in members:
            component_by_id[idea_id] = members
            key_by_id[idea_id] = program_key

    adjacent_by_id: dict[str, list[str]] = defaultdict(list)
    distinct_by_id: dict[str, int] = defaultdict(int)
    for row in relevant:
        if row.relation == "ADJACENT_PROGRAM":
            adjacent_by_id[row.idea_id_a].append(row.idea_id_b)
            adjacent_by_id[row.idea_id_b].append(row.idea_id_a)
        elif row.relation == "DISTINCT_PROGRAM":
            distinct_by_id[row.idea_id_a] += 1
            distinct_by_id[row.idea_id_b] += 1

    assignments = [
        ScientificProgramFamilyAssignment(
            idea_id=idea_id,
            scientific_program_key=key_by_id[idea_id],
            same_program_member_ids=list(component_by_id[idea_id]),
            same_program_size=len(component_by_id[idea_id]),
            adjacent_program_neighbor_ids=sorted(set(adjacent_by_id.get(idea_id, []))),
            adjacent_program_neighbor_count=len(set(adjacent_by_id.get(idea_id, []))),
            distinct_program_neighbor_count=distinct_by_id.get(idea_id, 0),
        )
        for idea_id in idea_ids
    ]

    transitivity_diagnostics = 0
    for members in components:
        if len(members) < 3:
            continue
        for a, b in combinations(members, 2):
            if by_pair[_pair_key(a, b)].relation != "SAME_PROGRAM":
                transitivity_diagnostics += 1

    relation_counts = Counter(row.relation for row in relevant)
    size_hist = Counter(str(len(members)) for members in components)
    provisional = ScientificProgramFamilyReport(
        report_id="pending",
        report_sha256="pending",
        idea_count=len(idea_ids),
        pair_count=len(relevant),
        program_count=len(components),
        singleton_program_count=sum(len(members) == 1 for members in components),
        same_program_pair_count=relation_counts.get("SAME_PROGRAM", 0),
        adjacent_program_pair_count=relation_counts.get("ADJACENT_PROGRAM", 0),
        distinct_program_pair_count=relation_counts.get("DISTINCT_PROGRAM", 0),
        transitivity_diagnostic_count=transitivity_diagnostics,
        relation_counts=dict(sorted(relation_counts.items())),
        program_size_histogram=dict(sorted(size_hist.items())),
        assignments=assignments,
        pair_assessments=sorted(
            relevant,
            key=lambda row: _pair_key(row.idea_id_a, row.idea_id_b),
        ),
    )
    body = provisional.model_dump(mode="json")
    body.pop("report_id", None)
    body.pop("report_sha256", None)
    digest = _sha(body)
    return provisional.model_copy(
        update={
            "report_id": f"scientific_program_family:{digest[:20]}",
            "report_sha256": digest,
        }
    )


__all__ = [
    "ProgramRelation",
    "ScientificProgramPairAssessment",
    "ScientificProgramPairBatch",
    "ScientificProgramFamilyAssignment",
    "ScientificProgramFamilyReport",
    "PROGRAM_FAMILY_SYSTEM_PROMPT",
    "program_family_prompt_payload",
    "validate_program_pair_batch",
    "build_scientific_program_family_report",
]
