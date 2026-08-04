from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx

from dac_her.resolution_candidates import (
    normalize_scientific_text,
    normalized_tokens,
)


_GENERIC_PATTERNS = (
    re.compile(
        r"\b(superior|enhanced|improved|excellent|high)\b.*"
        r"\b(activity|performance)\b",
        re.I,
    ),
    re.compile(
        r"\bstructure\b.*\b(improves|enhances)\b.*\bactivity\b",
        re.I,
    ),
)

# Phrase order matters: multiword polarity phrases must be tested before the
# generic higher/lower tokens.
_POSITIVE_PATTERNS = (
    re.compile(r"\blower overpotential\b", re.I),
    re.compile(r"\blower (?:charge[- ]?transfer )?resistance\b", re.I),
    re.compile(r"\bhigher (?:activity|current density|surface area|capacitance)\b", re.I),
    re.compile(r"\b(increased|improved|enhanced|favorable|superior)\b", re.I),
)
_NEGATIVE_PATTERNS = (
    re.compile(r"\bhigher overpotential\b", re.I),
    re.compile(r"\bhigher (?:charge[- ]?transfer )?resistance\b", re.I),
    re.compile(r"\blower (?:activity|current density|surface area|capacitance)\b", re.I),
    re.compile(r"\b(decreased|worse|unfavorable|inferior)\b", re.I),
)

# Lexical property fallback is deliberately conservative. Direct Measurement
# evidence is preferred; these patterns only cover claims whose evidence is an
# Experiment/Calculation or another claim.
_PROPERTY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("metric:overpotential", re.compile(r"\boverpotential\b|\bη\s*10\b|\beta\s*10\b", re.I)),
    ("metric:tafel_slope", re.compile(r"\btafel slope\b", re.I)),
    ("metric:exchange_current_density", re.compile(r"\bexchange current density\b|\bj0\b", re.I)),
    ("metric:charge_transfer_resistance", re.compile(r"\bcharge[- ]?transfer resistance\b|\brct\b", re.I)),
    ("metric:double_layer_capacitance", re.compile(r"\bdouble[- ]?layer capacitance\b|\bcdl\b", re.I)),
    ("metric:bet_surface_area", re.compile(r"\bBET surface area\b|\bspecific surface area\b", re.I)),
    ("metric:turnover_frequency", re.compile(r"\bturnover frequency\b|\bTOF\b", re.I)),
    ("metric:hydrogen_adsorption_free_energy", re.compile(r"\bhydrogen adsorption free energy\b|\bdelta g[_ ]?h\b|ΔG", re.I)),
    ("property:intrinsic_activity", re.compile(r"\bintrinsic (?:HER )?activity\b", re.I)),
    ("property:stability", re.compile(r"\bstability\b|\bdurability\b|\bretention\b", re.I)),
    ("property:electronic_structure", re.compile(r"\belectronic structure\b|\belectron(?:ic)? redistribution\b|\bcharge transfer\b", re.I)),
    ("property:active_site", re.compile(r"\bactive site\b", re.I)),
)
_BOND_PATTERN = re.compile(
    r"\b([A-Z][a-z]?)\s*[-=–—]\s*([A-Z][a-z]?)\b"
)
_COORDINATION_PATTERN = re.compile(
    r"\b([A-Z][a-z]?)\s*[-–—]?\s*coordinat(?:ed|ion)\b",
    re.I,
)


def _jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _claim_statement(graph: nx.Graph, node_id: str) -> str:
    data = graph.nodes[node_id]
    return str(data.get("statement") or data.get("label") or node_id)


def _targets(graph: nx.Graph, node_id: str) -> frozenset[str]:
    return frozenset(
        str(target)
        for _, target, data in graph.out_edges(node_id, data=True)
        if str(data.get("relation", "")) == "APPLIES_TO"
    )


def _evidence_sources(graph: nx.Graph, node_id: str) -> frozenset[str]:
    return frozenset(
        str(source)
        for source, _, data in graph.in_edges(node_id, data=True)
        if str(data.get("relation", ""))
        in {"SUPPORTS_CLAIM", "INTERPRETED_AS"}
    )


def _sections(graph: nx.Graph, node_id: str) -> frozenset[str]:
    values = set()
    for _, _, data in graph.in_edges(node_id, data=True):
        section = str(data.get("section", "")).strip()
        if section:
            values.add(section)
    for _, _, data in graph.out_edges(node_id, data=True):
        section = str(data.get("section", "")).strip()
        if section:
            values.add(section)
    return frozenset(values)


def _polarity(statement: str) -> str:
    positive = any(pattern.search(statement) for pattern in _POSITIVE_PATTERNS)
    negative = any(pattern.search(statement) for pattern in _NEGATIVE_PATTERNS)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return "neutral"


def _candidate_id(left_id: str, right_id: str) -> str:
    left, right = sorted((left_id, right_id))
    digest = hashlib.sha256(f"{left}|{right}".encode()).hexdigest()[:20]
    return "claim_overlap:" + digest


def _parse_conditions(value: Any) -> dict[str, frozenset[str]]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, list):
        return {}

    result: dict[str, set[str]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = normalize_scientific_text(item.get("name", ""))
        if not name:
            continue
        parts = [
            item.get("value_numeric"),
            item.get("value_text"),
            item.get("unit"),
            item.get("reference"),
        ]
        value_text = normalize_scientific_text(
            " ".join(str(part) for part in parts if part not in (None, ""))
        )
        if value_text:
            result.setdefault(name, set()).add(value_text)
    return {key: frozenset(values) for key, values in result.items()}


def _measurement_property(graph: nx.Graph, node_id: str) -> str | None:
    data = graph.nodes[node_id]
    if str(data.get("type", "")) != "Measurement":
        return None
    metric_id = str(data.get("metric_id", "")).strip()
    if not metric_id:
        metric_id = normalize_scientific_text(data.get("metric", ""))
    return f"metric:{metric_id}" if metric_id else None


def _lexical_property_keys(statement: str) -> set[str]:
    properties = {
        property_key
        for property_key, pattern in _PROPERTY_PATTERNS
        if pattern.search(statement)
    }
    for left, right in _BOND_PATTERN.findall(statement):
        properties.add(f"bond:{left.lower()}-{right.lower()}")
    for element in _COORDINATION_PATTERN.findall(statement):
        properties.add(f"coordination:{element.lower()}")
    return properties


def _claim_properties(
    graph: nx.Graph,
    node_id: str,
    *,
    visited: set[str] | None = None,
) -> frozenset[str]:
    visited = set(visited or set())
    if node_id in visited:
        return frozenset()
    visited.add(node_id)

    properties = _lexical_property_keys(_claim_statement(graph, node_id))
    for source in _evidence_sources(graph, node_id):
        source_type = str(graph.nodes[source].get("type", ""))
        measurement_property = _measurement_property(graph, source)
        if measurement_property:
            properties.add(measurement_property)
        elif source_type in {"ObservationClaim", "MechanismClaim"}:
            properties.update(
                _claim_properties(graph, source, visited=visited)
            )
        elif source_type == "Calculation":
            calculation_type = str(
                graph.nodes[source].get("calculation_type", "")
            ).strip()
            if calculation_type:
                properties.add(f"calculation:{calculation_type}")
    return frozenset(properties)


def _claim_conditions(
    graph: nx.Graph,
    node_id: str,
    *,
    visited: set[str] | None = None,
) -> dict[str, frozenset[str]]:
    visited = set(visited or set())
    if node_id in visited:
        return {}
    visited.add(node_id)

    collected: dict[str, set[str]] = {}
    for source in _evidence_sources(graph, node_id):
        source_type = str(graph.nodes[source].get("type", ""))
        if source_type == "Measurement":
            for name, values in _parse_conditions(
                graph.nodes[source].get("conditions_json")
            ).items():
                collected.setdefault(name, set()).update(values)
        elif source_type in {"ObservationClaim", "MechanismClaim"}:
            nested = _claim_conditions(graph, source, visited=visited)
            for name, values in nested.items():
                collected.setdefault(name, set()).update(values)
    return {key: frozenset(values) for key, values in collected.items()}


def _conditions_compatible(
    left: dict[str, frozenset[str]],
    right: dict[str, frozenset[str]],
) -> bool:
    # Missing condition information is not treated as a conflict. A conflict is
    # declared only when both claims specify the same condition dimension and
    # the reported values are disjoint.
    for name in set(left) & set(right):
        if left[name] and right[name] and not (left[name] & right[name]):
            return False
    return True


@dataclass(frozen=True)
class ClaimOverlapCandidate:
    candidate_id: str
    left_id: str
    right_id: str
    claim_node_type: str
    claim_type: str
    left_statement: str
    right_statement: str
    left_properties: str
    right_properties: str
    shared_properties: str
    condition_compatible: bool
    statement_similarity: float
    token_jaccard: float
    target_overlap: float
    evidence_overlap: float
    section_overlap: float
    total_score: float
    suggested_relation: str
    auto_merge: bool
    needs_review: bool

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ClaimCluster:
    claim_cluster_id: str
    claim_node_type: str
    suggested_relation: str
    review_status: str
    representative_claim_id: str
    supporting_claim_ids: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["supporting_claim_ids_json"] = (
            json.dumps(
                list(
                    row.pop(
                        "supporting_claim_ids"
                    )
                ),
                ensure_ascii=False,
            )
        )
        return row
    

def generate_claim_overlap_candidates(
    graph: nx.Graph,
    *,
    minimum_score: float = 0.68,
) -> list[ClaimOverlapCandidate]:
    claims = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if str(data.get("type", ""))
        in {"ObservationClaim", "MechanismClaim"}
    )
    candidates: list[ClaimOverlapCandidate] = []

    property_cache = {
        claim_id: _claim_properties(graph, claim_id)
        for claim_id in claims
    }
    condition_cache = {
        claim_id: _claim_conditions(graph, claim_id)
        for claim_id in claims
    }

    for index, left_id in enumerate(claims):
        left_data = graph.nodes[left_id]
        for right_id in claims[index + 1 :]:
            right_data = graph.nodes[right_id]
            if left_data.get("type") != right_data.get("type"):
                continue

            left_statement = _claim_statement(graph, left_id)
            right_statement = _claim_statement(graph, right_id)
            normalized_left = normalize_scientific_text(left_statement)
            normalized_right = normalize_scientific_text(right_statement)
            normalized_equal = normalized_left == normalized_right
            same_claim_type = str(left_data.get("claim_type", "")) == str(
                right_data.get("claim_type", "")
            )
            # Exact normalized statements must bypass property/target blocking.
            # This catches main-vs-SI duplicates even when extraction assigned
            # slightly different claim_type values or evidence targets.
            if not same_claim_type and not normalized_equal:
                continue
            statement_similarity = SequenceMatcher(
                None, normalized_left, normalized_right
            ).ratio()
            token_jaccard = _jaccard(
                normalized_tokens(left_statement),
                normalized_tokens(right_statement),
            )
            target_overlap = _jaccard(
                _targets(graph, left_id),
                _targets(graph, right_id),
            )
            evidence_overlap = _jaccard(
                _evidence_sources(graph, left_id),
                _evidence_sources(graph, right_id),
            )
            section_overlap = _jaccard(
                _sections(graph, left_id),
                _sections(graph, right_id),
            )
            total = (
                0.45 * statement_similarity
                + 0.20 * target_overlap
                + 0.15 * evidence_overlap
                + 0.10 * token_jaccard
                + 0.10 * section_overlap
            )

            left_properties = property_cache[left_id]
            right_properties = property_cache[right_id]
            shared_properties = left_properties & right_properties
            condition_compatible = _conditions_compatible(
                condition_cache[left_id], condition_cache[right_id]
            )
            same_evidence = bool(
                _evidence_sources(graph, left_id)
                & _evidence_sources(graph, right_id)
            )
            polarities = {
                _polarity(left_statement),
                _polarity(right_statement),
            }

            if normalized_equal:
                relation = (
                    "EXACT_DUPLICATE"
                    if same_evidence
                    else "SAME_CONCLUSION_DIFFERENT_EVIDENCE"
                )
            elif (
                polarities == {"positive", "negative"}
                and target_overlap > 0
                and bool(shared_properties)
                and condition_compatible
            ):
                # Contradiction is blocked unless the claims concern the same
                # normalized metric/property under compatible conditions.
                relation = "POSSIBLE_CONTRADICTION"
            elif (
                statement_similarity >= 0.90
                and target_overlap >= 0.5
                and not same_evidence
                and bool(shared_properties)
                and condition_compatible
            ):
                relation = "SAME_CONCLUSION_DIFFERENT_EVIDENCE"
            elif (
                total >= minimum_score
                and bool(shared_properties)
                and condition_compatible
            ):
                left_tokens = normalized_tokens(left_statement)
                right_tokens = normalized_tokens(right_statement)
                if left_tokens < right_tokens or right_tokens < left_tokens:
                    relation = "REFINES_OR_BROADENS"
                else:
                    relation = "SEMANTICALLY_OVERLAPS"
            else:
                continue

            candidates.append(
                ClaimOverlapCandidate(
                    candidate_id=_candidate_id(left_id, right_id),
                    left_id=left_id,
                    right_id=right_id,
                    claim_node_type=str(left_data.get("type", "")),
                    claim_type=(
                        str(left_data.get("claim_type", ""))
                        if same_claim_type
                        else " | ".join(sorted({
                            str(left_data.get("claim_type", "")),
                            str(right_data.get("claim_type", "")),
                        }))
                    ),
                    left_statement=left_statement,
                    right_statement=right_statement,
                    left_properties=json.dumps(
                        sorted(left_properties), ensure_ascii=False
                    ),
                    right_properties=json.dumps(
                        sorted(right_properties), ensure_ascii=False
                    ),
                    shared_properties=json.dumps(
                        sorted(shared_properties), ensure_ascii=False
                    ),
                    condition_compatible=condition_compatible,
                    statement_similarity=round(statement_similarity, 6),
                    token_jaccard=round(token_jaccard, 6),
                    target_overlap=round(target_overlap, 6),
                    evidence_overlap=round(evidence_overlap, 6),
                    section_overlap=round(section_overlap, 6),
                    total_score=round(total, 6),
                    suggested_relation=relation,
                    auto_merge=False,
                    needs_review=True,
                )
            )

    return sorted(
        candidates,
        key=lambda item: (-item.total_score, item.left_id, item.right_id),
    )

def build_claim_clusters(
    candidates: Sequence[
        ClaimOverlapCandidate
    ],
) -> list[ClaimCluster]:
    clusterable_relations = {
        "EXACT_DUPLICATE",
        "SAME_CONCLUSION_DIFFERENT_EVIDENCE",
    }

    graph = nx.Graph()

    candidate_lookup: dict[
        tuple[str, str],
        ClaimOverlapCandidate,
    ] = {}

    for candidate in candidates:
        if (
            candidate.suggested_relation
            not in clusterable_relations
        ):
            continue

        graph.add_edge(
            candidate.left_id,
            candidate.right_id,
        )
        candidate_lookup[
            tuple(
                sorted(
                    (
                        candidate.left_id,
                        candidate.right_id,
                    )
                )
            )
        ] = candidate

    clusters: list[ClaimCluster] = []

    for members_value in (
        nx.connected_components(graph)
    ):
        members = tuple(
            sorted(
                str(value)
                for value in members_value
            )
        )

        if len(members) < 2:
            continue

        digest = hashlib.sha256(
            "|".join(members).encode(
                "utf-8"
            )
        ).hexdigest()[:20]

        member_candidates = [
            candidate
            for candidate in candidates
            if (
                candidate.left_id in members
                and candidate.right_id in members
                and candidate.suggested_relation
                in clusterable_relations
            )
        ]

        node_types = sorted({
            candidate.claim_node_type
            for candidate in member_candidates
        })

        relations = {
            candidate.suggested_relation
            for candidate in member_candidates
        }

        suggested_relation = (
            "EXACT_DUPLICATE"
            if relations
            == {"EXACT_DUPLICATE"}
            else (
                "SAME_CONCLUSION_DIFFERENT_EVIDENCE"
            )
        )

        clusters.append(
            ClaimCluster(
                claim_cluster_id=(
                    "claim_cluster:"
                    + digest
                ),
                claim_node_type=(
                    node_types[0]
                    if len(node_types) == 1
                    else "mixed"
                ),
                suggested_relation=(
                    suggested_relation
                ),
                review_status=(
                    "needs_review"
                ),
                representative_claim_id=(
                    members[0]
                ),
                supporting_claim_ids=members,
            )
        )

    return sorted(
        clusters,
        key=lambda item: (
            item.claim_node_type,
            item.representative_claim_id,
        ),
    )

def generic_claim_rows(graph: nx.Graph) -> list[dict[str, Any]]:
    rows = []
    for node_id, data in graph.nodes(data=True):
        if str(data.get("type", "")) not in {
            "ObservationClaim",
            "MechanismClaim",
        }:
            continue
        statement = _claim_statement(graph, str(node_id))
        if any(pattern.search(statement) for pattern in _GENERIC_PATTERNS):
            rows.append(
                {
                    "id": str(node_id),
                    "type": str(data.get("type", "")),
                    "claim_type": str(data.get("claim_type", "")),
                    "statement": statement,
                    "reason": (
                        "generic performance/mechanism wording; compare "
                        "against more specific claims"
                    ),
                }
            )
    return rows


def write_csv(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_claim_overlap_audit(
    graph: nx.Graph,
    output_dir: str | Path,
    *,
    minimum_score: float = 0.68,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    candidates = generate_claim_overlap_candidates(
        graph, minimum_score=minimum_score
    )
    generic = generic_claim_rows(graph)
    write_csv(
        output_dir / "claim_overlap_candidates.csv",
        [item.to_row() for item in candidates],
    )
    clusters = build_claim_clusters(
        candidates
    )
    write_csv(output_dir / "generic_claims.csv", generic)
    write_csv(
        output_dir / "claim_clusters.csv",
        [
            cluster.to_row()
            for cluster in clusters
        ],
    )
    summary = {
        "claim_overlap_candidates": len(candidates),
        "exact_duplicates": sum(
            item.suggested_relation == "EXACT_DUPLICATE"
            for item in candidates
        ),
        "possible_contradictions": sum(
            item.suggested_relation == "POSSIBLE_CONTRADICTION"
            for item in candidates
        ),
        "generic_claims": len(generic),
        "review_required": len(candidates),
        "auto_merged": 0,
        "minimum_score": minimum_score,
        "property_aware_blocking": True,
        "contradiction_requires_shared_property": True,
        "contradiction_requires_compatible_conditions": True,
        "claim_clusters": len(clusters),
        "clustered_claim_nodes": len({
            claim_id
            for cluster in clusters
            for claim_id
            in cluster.supporting_claim_ids
        }),
        "claims_destructively_merged": 0,
    }
    (output_dir / "claim_overlap_summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary
