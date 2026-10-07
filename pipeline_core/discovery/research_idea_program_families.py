from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_projection import project_frontier_idea


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "for", "with",
    "by", "through", "under", "via", "from", "into", "between", "within", "that",
    "this", "effect", "effects", "role", "state", "states", "condition", "conditions",
}


def _stem_token(token: str) -> str:
    value = token.casefold().strip()
    if len(value) > 6 and value.startswith("re"):
        value = value[2:]
    for suffix in ("ization", "isation", "ation", "ition", "ment", "ing", "ed", "es", "s"):
        if len(value) > len(suffix) + 3 and value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def _tokens(value: object) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9]+", str(value or "").casefold())
    return {
        stem
        for token in raw
        if token not in _STOP
        for stem in [_stem_token(token)]
        if stem and stem not in _STOP
    }


def _trigrams(value: object) -> set[str]:
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _phrase_similarity(left: object, right: object) -> float:
    token = _jaccard(_tokens(left), _tokens(right))
    chars = _jaccard(_trigrams(left), _trigrams(right))
    return max(token, 0.65 * token + 0.35 * chars, chars * 0.8)


def _soft_set_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    a = list(left)
    b = list(right)
    left_score = sum(max(_phrase_similarity(x, y) for y in b) for x in a) / len(a)
    right_score = sum(max(_phrase_similarity(y, x) for x in a) for y in b) / len(b)
    return (left_score + right_score) / 2.0


def _relations(node: ResearchIdeaNode) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for row in node.kernel.core_scientific_commitments:
        match = re.match(r"^\s*(.+?)\s*--(.+?)-->\s*(.+?)\s*$", str(row or ""))
        if match is None:
            continue
        subject, relation, obj = match.groups()
        out.append((subject.strip(), relation.strip().upper(), obj.strip()))
    return out


class ScientificProgramKernel(StrictModel):
    schema_version: Literal[
        "scientific-program-kernel-v1"
    ] = "scientific-program-kernel-v1"
    idea_id: str = Field(min_length=1)
    endpoint_phenomena: list[str] = Field(default_factory=list)
    backbone_relation_types: list[str] = Field(default_factory=list)
    causal_orientation_pairs: list[str] = Field(default_factory=list)
    mechanism_entities: list[str] = Field(default_factory=list)
    constitutive_regime_terms: list[str] = Field(default_factory=list)
    contrast_terms: list[str] = Field(default_factory=list)
    fallback_program_terms: list[str] = Field(default_factory=list)
    projection_is_coarser_than_research_idea_identity: Literal[True] = True
    operator_id_used_as_family_authority: Literal[False] = False


class ScientificProgramPairAssessment(StrictModel):
    idea_id_a: str
    idea_id_b: str
    endpoint_similarity: float = Field(ge=0.0, le=1.0)
    relation_similarity: float = Field(ge=0.0, le=1.0)
    orientation_similarity: float = Field(ge=0.0, le=1.0)
    mechanism_similarity: float = Field(ge=0.0, le=1.0)
    regime_similarity: float = Field(ge=0.0, le=1.0)
    contrast_similarity: float = Field(ge=0.0, le=1.0)
    overall_similarity: float = Field(ge=0.0, le=1.0)
    direct_ancestry: bool
    conceptual_delta_category: str | None = None
    conceptual_delta_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    shared_program_anchor: bool
    compatible_program_edge: bool
    compatibility_reason_codes: list[str] = Field(default_factory=list)
    diagnostic_only: Literal[True] = True
    family_assignment_authority: Literal[False] = False


class ScientificProgramFamilyAssignment(StrictModel):
    idea_id: str
    generation_index: int = Field(ge=0)
    program_family_key: str
    program_family_size: int = Field(ge=1)
    kernel: ScientificProgramKernel
    assignment_method_version: Literal[
        "scientific-program-complete-link-v1"
    ] = "scientific-program-complete-link-v1"
    recomputable: Literal[True] = True
    immutable_idea_identity_field: Literal[False] = False


class ScientificProgramThresholdPoint(StrictModel):
    similarity_floor: float = Field(ge=0.0, le=1.0)
    endpoint_floor: float = Field(ge=0.0, le=1.0)
    family_count: int = Field(ge=0)
    largest_family_size: int = Field(ge=0)
    singleton_family_count: int = Field(ge=0)


class ScientificProgramFamilyReport(StrictModel):
    schema_version: Literal[
        "scientific-program-family-shadow-v1"
    ] = "scientific-program-family-shadow-v1"
    report_id: str
    report_sha256: str
    idea_count: int = Field(ge=0)
    program_family_count: int = Field(ge=0)
    singleton_family_count: int = Field(ge=0)
    largest_family_size: int = Field(ge=0)
    fragmentation_ratio: float = Field(ge=0.0, le=1.0)
    assignments: list[ScientificProgramFamilyAssignment] = Field(default_factory=list)
    pair_assessments: list[ScientificProgramPairAssessment] = Field(default_factory=list)
    family_size_histogram: dict[str, int] = Field(default_factory=dict)
    generation_family_counts: dict[str, int] = Field(default_factory=dict)
    family_birth_count_by_generation: dict[str, int] = Field(default_factory=dict)
    threshold_curve: list[ScientificProgramThresholdPoint] = Field(default_factory=list)
    similarity_floor: float = Field(ge=0.0, le=1.0)
    endpoint_floor: float = Field(ge=0.0, le=1.0)
    endpoint_anchor_guard_prevents_bridge_chaining: Literal[True] = True
    ancestry_is_soft_evidence_not_family_authority: Literal[True] = True
    conceptual_delta_is_soft_evidence_not_family_authority: Literal[True] = True
    operator_id_is_not_family_authority: Literal[True] = True
    family_is_soft_and_recomputable: Literal[True] = True
    family_is_not_hard_selection_gate: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "ScientificProgramFamilyReport":
        if self.idea_count != len(self.assignments):
            raise ValueError("idea_count mismatch")
        return self


class FrontierProgramReferenceCalibration(StrictModel):
    schema_version: Literal[
        "frontier-scientific-program-reference-calibration-v1"
    ] = "frontier-scientific-program-reference-calibration-v1"
    population_id: str
    topology_idea_count: int = Field(ge=0)
    expected_backbone_signature_count: int = Field(ge=0)
    projected_program_family_count: int = Field(ge=0)
    absolute_family_count_error: int = Field(ge=0)
    projected_fragmentation_ratio: float = Field(ge=0.0, le=1.0)
    reference_is_diagnostic_only: Literal[True] = True


def project_scientific_program_kernel(node: ResearchIdeaNode) -> ScientificProgramKernel:
    relations = _relations(node)
    endpoint_phenomena: list[str] = []
    relation_types: list[str] = []
    orientation_pairs: list[str] = []
    mechanism: list[str] = []
    if relations:
        degree: Counter[str] = Counter()
        canonical: dict[str, str] = {}
        for subject, relation, obj in relations:
            s_key = " ".join(sorted(_tokens(subject))) or subject.casefold().strip()
            o_key = " ".join(sorted(_tokens(obj))) or obj.casefold().strip()
            canonical.setdefault(s_key, subject)
            canonical.setdefault(o_key, obj)
            degree[s_key] += 1
            degree[o_key] += 1
            relation_types.append(relation)
            orientation_pairs.append(f"{subject} -> {obj}")
        endpoint_keys = [key for key, count in degree.items() if count == 1]
        if not endpoint_keys:
            endpoint_keys = list(degree)
        endpoint_phenomena = [canonical[key] for key in sorted(endpoint_keys)]
        mechanism = [canonical[key] for key, count in sorted(degree.items()) if count > 1]
    else:
        endpoint_phenomena = list(node.kernel.core_scientific_commitments)

    scope_terms = list(node.kernel.scope_commitments)
    contrast_terms = list(node.kernel.contrastive_commitments)
    fallback = [
        node.kernel.canonical_intent,
        *node.kernel.core_scientific_commitments,
        *scope_terms,
    ]
    return ScientificProgramKernel(
        idea_id=node.idea_id,
        endpoint_phenomena=list(dict.fromkeys(endpoint_phenomena)),
        backbone_relation_types=list(dict.fromkeys(relation_types)),
        causal_orientation_pairs=list(dict.fromkeys(orientation_pairs)),
        mechanism_entities=list(dict.fromkeys(mechanism)),
        constitutive_regime_terms=list(dict.fromkeys(scope_terms)),
        contrast_terms=list(dict.fromkeys(contrast_terms)),
        fallback_program_terms=list(dict.fromkeys(fallback)),
    )



def _program_anchor(kernel: ScientificProgramKernel) -> tuple[str, ...]:
    values: list[str] = []
    for phrase in kernel.endpoint_phenomena:
        tokens = sorted(_tokens(phrase))
        if tokens:
            values.append(" ".join(tokens))
    if values:
        return tuple(sorted(set(values)))
    fallback = sorted({token for phrase in kernel.fallback_program_terms for token in _tokens(phrase)})
    return tuple(fallback[:8])

def _pair_score(left: ScientificProgramKernel, right: ScientificProgramKernel) -> dict[str, float]:
    endpoint = _soft_set_similarity(left.endpoint_phenomena, right.endpoint_phenomena)
    relation = _soft_set_similarity(left.backbone_relation_types, right.backbone_relation_types)
    orientation = _soft_set_similarity(left.causal_orientation_pairs, right.causal_orientation_pairs)
    mechanism = _soft_set_similarity(left.mechanism_entities, right.mechanism_entities)
    regime = _soft_set_similarity(left.constitutive_regime_terms, right.constitutive_regime_terms)
    contrast = _soft_set_similarity(left.contrast_terms, right.contrast_terms)
    fallback = _soft_set_similarity(left.fallback_program_terms, right.fallback_program_terms)
    if left.endpoint_phenomena and right.endpoint_phenomena:
        overall = (
            0.42 * endpoint
            + 0.16 * relation
            + 0.14 * orientation
            + 0.13 * mechanism
            + 0.08 * regime
            + 0.04 * contrast
            + 0.03 * fallback
        )
    else:
        overall = 0.55 * fallback + 0.20 * relation + 0.15 * regime + 0.10 * contrast
    return {
        "endpoint": min(1.0, max(0.0, endpoint)),
        "relation": min(1.0, max(0.0, relation)),
        "orientation": min(1.0, max(0.0, orientation)),
        "mechanism": min(1.0, max(0.0, mechanism)),
        "regime": min(1.0, max(0.0, regime)),
        "contrast": min(1.0, max(0.0, contrast)),
        "overall": min(1.0, max(0.0, overall)),
    }


def _audit_map(conceptual_audits: Sequence[Any]) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for report in conceptual_audits:
        for row in getattr(report, "records", []) or []:
            child_id = str(getattr(row, "child_idea_id", "") or "")
            if not child_id:
                continue
            out[child_id] = (
                str(getattr(row, "audit_category", "") or ""),
                float(getattr(row, "audit_confidence", 0.0) or 0.0),
            )
    return out



def _program_compatible_from_values(
    *,
    overall: float,
    endpoint: float,
    relation: float,
    orientation: float,
    direct_ancestry: bool,
    category: str | None,
    confidence: float | None,
    structured_endpoints_present: bool,
    shared_program_anchor: bool,
    similarity_floor: float,
    endpoint_floor: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if direct_ancestry and category in {"SAME_CONCEPT", "REFINEMENT"} and (confidence or 0.0) >= 0.60:
        return True, ["DIRECT_LINEAGE_REFINEMENT_SUPPORTS_SAME_PROGRAM"]
    if (
        shared_program_anchor
        and overall >= similarity_floor
        and endpoint >= endpoint_floor
        and (relation >= 0.30 or orientation >= 0.30)
    ):
        return True, ["PROGRAM_FEATURES_CLEAR_SIMILARITY_FLOORS"]
    if (
        direct_ancestry
        and category == "GENUINE_MUTATION"
        and (confidence or 0.0) >= 0.60
        and endpoint >= max(0.30, endpoint_floor - 0.10)
        and overall >= max(0.0, similarity_floor - 0.08)
    ):
        return True, ["DIRECT_LINEAGE_MUTATION_PRESERVES_PROGRAM_ENDPOINTS"]
    if not structured_endpoints_present and overall >= min(1.0, similarity_floor + 0.12):
        return True, ["UNSTRUCTURED_FALLBACK_HIGH_PROGRAM_SIMILARITY"]
    return False, ["PROGRAM_COMPATIBILITY_NOT_ESTABLISHED"]


def _row_compatible_at(
    row: ScientificProgramPairAssessment,
    *,
    similarity_floor: float,
    endpoint_floor: float,
) -> bool:
    # Pair rows do not retain a structured-endpoint flag. For threshold curves,
    # the normal structured rule is intentionally used; unstructured fallbacks
    # are already diagnostic in the primary assignment and need not dominate
    # the calibration curve.
    compatible, _ = _program_compatible_from_values(
        overall=row.overall_similarity,
        endpoint=row.endpoint_similarity,
        relation=row.relation_similarity,
        orientation=row.orientation_similarity,
        direct_ancestry=row.direct_ancestry,
        category=row.conceptual_delta_category,
        confidence=row.conceptual_delta_confidence,
        structured_endpoints_present=True,
        shared_program_anchor=row.shared_program_anchor,
        similarity_floor=similarity_floor,
        endpoint_floor=endpoint_floor,
    )
    return compatible

def _pair_assessments(
    nodes: Sequence[ResearchIdeaNode],
    *,
    conceptual_audits: Sequence[Any],
    similarity_floor: float,
    endpoint_floor: float,
) -> list[ScientificProgramPairAssessment]:
    ordered = sorted(nodes, key=lambda row: row.idea_id)
    kernels = {row.idea_id: project_scientific_program_kernel(row) for row in ordered}
    node_by_id = {row.idea_id: row for row in ordered}
    audit = _audit_map(conceptual_audits)
    rows: list[ScientificProgramPairAssessment] = []
    for index, left_node in enumerate(ordered):
        for right_node in ordered[index + 1 :]:
            left = kernels[left_node.idea_id]
            right = kernels[right_node.idea_id]
            scores = _pair_score(left, right)
            shared_program_anchor = _program_anchor(left) == _program_anchor(right)
            direct_ancestry = (
                left_node.idea_id in set(right_node.parent_idea_ids)
                or right_node.idea_id in set(left_node.parent_idea_ids)
            )
            child_id = None
            if left_node.idea_id in set(right_node.parent_idea_ids):
                child_id = right_node.idea_id
            elif right_node.idea_id in set(left_node.parent_idea_ids):
                child_id = left_node.idea_id
            category, confidence = audit.get(child_id, (None, None)) if child_id else (None, None)
            compatible, reasons = _program_compatible_from_values(
                overall=scores["overall"],
                endpoint=scores["endpoint"],
                relation=scores["relation"],
                orientation=scores["orientation"],
                direct_ancestry=direct_ancestry,
                category=category,
                confidence=confidence,
                structured_endpoints_present=bool(left.endpoint_phenomena and right.endpoint_phenomena),
                shared_program_anchor=shared_program_anchor,
                similarity_floor=similarity_floor,
                endpoint_floor=endpoint_floor,
            )
            rows.append(
                ScientificProgramPairAssessment(
                    idea_id_a=left_node.idea_id,
                    idea_id_b=right_node.idea_id,
                    endpoint_similarity=scores["endpoint"],
                    relation_similarity=scores["relation"],
                    orientation_similarity=scores["orientation"],
                    mechanism_similarity=scores["mechanism"],
                    regime_similarity=scores["regime"],
                    contrast_similarity=scores["contrast"],
                    overall_similarity=scores["overall"],
                    direct_ancestry=direct_ancestry,
                    conceptual_delta_category=category,
                    conceptual_delta_confidence=confidence,
                    shared_program_anchor=shared_program_anchor,
                    compatible_program_edge=compatible,
                    compatibility_reason_codes=reasons,
                )
            )
    return rows


def _cluster_anchor_guarded(
    idea_ids: Sequence[str],
    pair_rows: Sequence[ScientificProgramPairAssessment],
) -> list[list[str]]:
    parent = {idea_id: idea_id for idea_id in idea_ids}
    size = {idea_id: 1 for idea_id in idea_ids}

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            nxt = parent[value]
            parent[value] = root
            value = nxt
        return root

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        if size[a] < size[b] or (size[a] == size[b] and b < a):
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    for row in sorted(
        (row for row in pair_rows if row.compatible_program_edge),
        key=lambda row: (-row.overall_similarity, row.idea_id_a, row.idea_id_b),
    ):
        union(row.idea_id_a, row.idea_id_b)

    groups: dict[str, list[str]] = defaultdict(list)
    for idea_id in sorted(idea_ids):
        groups[find(idea_id)].append(idea_id)
    return sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])

def _family_counts_by_generation(
    assignments: Sequence[ScientificProgramFamilyAssignment],
) -> tuple[dict[str, int], dict[str, int]]:
    by_family: dict[str, list[ScientificProgramFamilyAssignment]] = defaultdict(list)
    for row in assignments:
        by_family[row.program_family_key].append(row)
    generation_sets: dict[int, set[str]] = defaultdict(set)
    births: Counter[int] = Counter()
    for family_key, rows in by_family.items():
        generations = {row.generation_index for row in rows}
        for generation in generations:
            generation_sets[generation].add(family_key)
        births[min(generations)] += 1
    return (
        {f"G{g}": len(values) for g, values in sorted(generation_sets.items())},
        {f"G{g}": value for g, value in sorted(births.items())},
    )


def build_scientific_program_families(
    nodes: Sequence[ResearchIdeaNode],
    *,
    conceptual_audits: Sequence[Any] = (),
    similarity_floor: float = 0.46,
    endpoint_floor: float = 0.55,
    curve_floors: Sequence[float] = (0.34, 0.40, 0.46, 0.52, 0.58),
) -> ScientificProgramFamilyReport:
    if not 0.0 <= similarity_floor <= 1.0:
        raise ValueError("similarity_floor must be in [0,1]")
    if not 0.0 <= endpoint_floor <= 1.0:
        raise ValueError("endpoint_floor must be in [0,1]")
    ordered = sorted({row.idea_id: row for row in nodes}.values(), key=lambda row: row.idea_id)
    idea_ids = [row.idea_id for row in ordered]
    pair_rows = _pair_assessments(
        ordered,
        conceptual_audits=conceptual_audits,
        similarity_floor=similarity_floor,
        endpoint_floor=endpoint_floor,
    )
    clusters = _cluster_anchor_guarded(idea_ids, pair_rows)
    node_by_id = {row.idea_id: row for row in ordered}
    kernels = {idea_id: project_scientific_program_kernel(node_by_id[idea_id]) for idea_id in idea_ids}
    assignments: list[ScientificProgramFamilyAssignment] = []
    for members in clusters:
        family_key = _stable_id("scientific_program_family", members)
        for idea_id in members:
            node = node_by_id[idea_id]
            assignments.append(
                ScientificProgramFamilyAssignment(
                    idea_id=idea_id,
                    generation_index=node.generation_index,
                    program_family_key=family_key,
                    program_family_size=len(members),
                    kernel=kernels[idea_id],
                )
            )
    assignments.sort(key=lambda row: row.idea_id)
    generation_counts, births = _family_counts_by_generation(assignments)
    histogram = Counter(str(len(members)) for members in clusters)
    curve: list[ScientificProgramThresholdPoint] = []
    for floor in curve_floors:
        rows = [
            row.model_copy(
                update={
                    "compatible_program_edge": _row_compatible_at(
                        row,
                        similarity_floor=float(floor),
                        endpoint_floor=endpoint_floor,
                    )
                }
            )
            for row in pair_rows
        ]
        trial = _cluster_anchor_guarded(idea_ids, rows)
        curve.append(
            ScientificProgramThresholdPoint(
                similarity_floor=float(floor),
                endpoint_floor=endpoint_floor,
                family_count=len(trial),
                largest_family_size=max((len(members) for members in trial), default=0),
                singleton_family_count=sum(len(members) == 1 for members in trial),
            )
        )
    provisional = ScientificProgramFamilyReport(
        report_id="pending",
        report_sha256="pending",
        idea_count=len(ordered),
        program_family_count=len(clusters),
        singleton_family_count=sum(len(members) == 1 for members in clusters),
        largest_family_size=max((len(members) for members in clusters), default=0),
        fragmentation_ratio=(len(clusters) / len(ordered) if ordered else 0.0),
        assignments=assignments,
        pair_assessments=pair_rows,
        family_size_histogram=dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        generation_family_counts=generation_counts,
        family_birth_count_by_generation=births,
        threshold_curve=curve,
        similarity_floor=similarity_floor,
        endpoint_floor=endpoint_floor,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"scientific_program_families:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def _topology_backbone_signature(idea: Any) -> tuple[str, ...] | None:
    topology = getattr(idea, "topology_signature", None)
    if topology is None:
        return None
    rows = tuple(str(value).strip().casefold() for value in topology.backbone_relation_texts)
    return rows or None


def calibrate_frontier_program_reference(
    population: FrontierIdeaPopulation,
    *,
    similarity_floor: float = 0.46,
    endpoint_floor: float = 0.55,
) -> FrontierProgramReferenceCalibration:
    topology_ideas = [idea for idea in population.ideas if getattr(idea, "topology_signature", None) is not None]
    signatures = {
        signature
        for idea in topology_ideas
        for signature in [_topology_backbone_signature(idea)]
        if signature is not None
    }
    # Reference populations can contain hundreds of near-duplicate topology
    # candidates. Use the same coarse program projection (endpoint anchor +
    # relation-type backbone) without O(n^2) pairwise calibration so the
    # historical 512-candidate cohort remains practical as a regression.
    nodes = [project_frontier_idea(idea, generation_index=0) for idea in topology_ideas]
    projected_signatures = set()
    for node in nodes:
        kernel = project_scientific_program_kernel(node)
        projected_signatures.add(
            (
                _program_anchor(kernel),
                tuple(sorted(kernel.backbone_relation_types)),
            )
        )
    expected = len(signatures)
    projected = len(projected_signatures)
    return FrontierProgramReferenceCalibration(
        population_id=population.population_id,
        topology_idea_count=len(topology_ideas),
        expected_backbone_signature_count=expected,
        projected_program_family_count=projected,
        absolute_family_count_error=abs(projected - expected),
        projected_fragmentation_ratio=(projected / len(topology_ideas) if topology_ideas else 0.0),
    )


__all__ = [
    "FrontierProgramReferenceCalibration",
    "ScientificProgramFamilyAssignment",
    "ScientificProgramFamilyReport",
    "ScientificProgramKernel",
    "ScientificProgramPairAssessment",
    "build_scientific_program_families",
    "calibrate_frontier_program_reference",
    "project_scientific_program_kernel",
]
