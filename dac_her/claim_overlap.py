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

from dac_her.resolution_candidates import normalize_scientific_text, normalized_tokens


_GENERIC_PATTERNS = (
    re.compile(r"\b(superior|enhanced|improved|excellent|high)\b.*\b(activity|performance)\b", re.I),
    re.compile(r"\bstructure\b.*\b(improves|enhances)\b.*\bactivity\b", re.I),
)
_POSITIVE = {"higher", "increase", "increases", "improved", "enhanced", "favorable", "lower overpotential"}
_NEGATIVE = {"lower", "decrease", "decreases", "worse", "unfavorable", "higher overpotential"}


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
        if str(data.get("relation", "")) in {"SUPPORTS_CLAIM", "INTERPRETED_AS"}
    )


def _sections(graph: nx.Graph, node_id: str) -> frozenset[str]:
    values = set()
    for source, _, data in graph.in_edges(node_id, data=True):
        section = str(data.get("section", "")).strip()
        if section:
            values.add(section)
    for _, target, data in graph.out_edges(node_id, data=True):
        section = str(data.get("section", "")).strip()
        if section:
            values.add(section)
    return frozenset(values)


def _polarity(statement: str) -> str:
    lowered = normalize_scientific_text(statement)
    positive = any(token in lowered for token in _POSITIVE)
    negative = any(token in lowered for token in _NEGATIVE)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return "neutral"


def _candidate_id(left_id: str, right_id: str) -> str:
    left, right = sorted((left_id, right_id))
    return "claim_overlap:" + hashlib.sha256(f"{left}|{right}".encode()).hexdigest()[:20]


@dataclass(frozen=True)
class ClaimOverlapCandidate:
    candidate_id: str
    left_id: str
    right_id: str
    claim_node_type: str
    claim_type: str
    left_statement: str
    right_statement: str
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


def generate_claim_overlap_candidates(
    graph: nx.Graph,
    *,
    minimum_score: float = 0.68,
) -> list[ClaimOverlapCandidate]:
    claims = [
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if str(data.get("type", "")) in {"ObservationClaim", "MechanismClaim"}
    ]
    candidates: list[ClaimOverlapCandidate] = []
    for index, left_id in enumerate(sorted(claims)):
        left_data = graph.nodes[left_id]
        for right_id in sorted(claims)[index + 1:]:
            right_data = graph.nodes[right_id]
            if left_data.get("type") != right_data.get("type"):
                continue
            if str(left_data.get("claim_type", "")) != str(right_data.get("claim_type", "")):
                continue
            left_statement, right_statement = _claim_statement(graph, left_id), _claim_statement(graph, right_id)
            statement_similarity = SequenceMatcher(
                None,
                normalize_scientific_text(left_statement),
                normalize_scientific_text(right_statement),
            ).ratio()
            token_jaccard = _jaccard(normalized_tokens(left_statement), normalized_tokens(right_statement))
            target_overlap = _jaccard(_targets(graph, left_id), _targets(graph, right_id))
            evidence_overlap = _jaccard(_evidence_sources(graph, left_id), _evidence_sources(graph, right_id))
            section_overlap = _jaccard(_sections(graph, left_id), _sections(graph, right_id))
            total = (
                0.45 * statement_similarity
                + 0.20 * target_overlap
                + 0.15 * evidence_overlap
                + 0.10 * token_jaccard
                + 0.10 * section_overlap
            )
            normalized_equal = normalize_scientific_text(left_statement) == normalize_scientific_text(right_statement)
            same_evidence = bool(_evidence_sources(graph, left_id) & _evidence_sources(graph, right_id))
            if normalized_equal and target_overlap == 1.0 and same_evidence:
                relation = "EXACT_DUPLICATE"
            elif _polarity(left_statement) != _polarity(right_statement) and {
                _polarity(left_statement), _polarity(right_statement)
            } == {"positive", "negative"} and target_overlap > 0:
                relation = "POSSIBLE_CONTRADICTION"
            elif statement_similarity >= 0.90 and target_overlap >= 0.5 and not same_evidence:
                relation = "SAME_CONCLUSION_DIFFERENT_EVIDENCE"
            elif total >= minimum_score:
                left_tokens, right_tokens = normalized_tokens(left_statement), normalized_tokens(right_statement)
                if left_tokens < right_tokens or right_tokens < left_tokens:
                    relation = "REFINES_OR_BROADENS"
                else:
                    relation = "SEMANTICALLY_OVERLAPS"
            else:
                continue
            candidates.append(ClaimOverlapCandidate(
                candidate_id=_candidate_id(left_id, right_id),
                left_id=left_id,
                right_id=right_id,
                claim_node_type=str(left_data.get("type", "")),
                claim_type=str(left_data.get("claim_type", "")),
                left_statement=left_statement,
                right_statement=right_statement,
                statement_similarity=round(statement_similarity, 6),
                token_jaccard=round(token_jaccard, 6),
                target_overlap=round(target_overlap, 6),
                evidence_overlap=round(evidence_overlap, 6),
                section_overlap=round(section_overlap, 6),
                total_score=round(total, 6),
                suggested_relation=relation,
                auto_merge=False,
                needs_review=True,
            ))
    return sorted(candidates, key=lambda item: (-item.total_score, item.left_id, item.right_id))


def generic_claim_rows(graph: nx.Graph) -> list[dict[str, Any]]:
    rows = []
    for node_id, data in graph.nodes(data=True):
        if str(data.get("type", "")) not in {"ObservationClaim", "MechanismClaim"}:
            continue
        statement = _claim_statement(graph, str(node_id))
        if any(pattern.search(statement) for pattern in _GENERIC_PATTERNS):
            rows.append({
                "id": str(node_id),
                "type": str(data.get("type", "")),
                "claim_type": str(data.get("claim_type", "")),
                "statement": statement,
                "reason": "generic performance/mechanism wording; compare against more specific claims",
            })
    return rows


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
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
    candidates = generate_claim_overlap_candidates(graph, minimum_score=minimum_score)
    generic = generic_claim_rows(graph)
    write_csv(output_dir / "claim_overlap_candidates.csv", [item.to_row() for item in candidates])
    write_csv(output_dir / "generic_claims.csv", generic)
    summary = {
        "claim_overlap_candidates": len(candidates),
        "exact_duplicates": sum(item.suggested_relation == "EXACT_DUPLICATE" for item in candidates),
        "possible_contradictions": sum(item.suggested_relation == "POSSIBLE_CONTRADICTION" for item in candidates),
        "generic_claims": len(generic),
        "review_required": len(candidates),
        "auto_merged": 0,
        "minimum_score": minimum_score,
    }
    (output_dir / "claim_overlap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
