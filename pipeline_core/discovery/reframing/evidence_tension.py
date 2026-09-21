from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EvidenceTensionType = Literal[
    "DIRECTIONAL",
    "MAGNITUDE",
    "BOUNDARY",
    "MEASUREMENT",
    "CONTEXT",
    "MECHANISTIC",
    "NULL_EFFECT",
    "QUALITATIVE",
    "POTENTIAL_CONFLICT",
]

TensionExtractionMode = Literal[
    "explorer_only",
    "packet_assisted",
]


class EvidenceLevelTensionWitness(StrictModel):
    witness_id: str
    source_tension_id: str
    source_tension_type: str
    focal_statement_id: str | None = None
    side_a_statement_ids: list[str] = Field(default_factory=list)
    side_b_statement_ids: list[str] = Field(default_factory=list)
    grounded_statement_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    claim_kinds: list[str] = Field(default_factory=list)
    tension_types: list[EvidenceTensionType] = Field(default_factory=list)
    classification_bases: list[str] = Field(default_factory=list)
    side_a_node_types: list[str] = Field(default_factory=list)
    side_b_node_types: list[str] = Field(default_factory=list)
    relevant_condition_signature_count: int = Field(ge=0, default=0)
    relevant_condition_names: list[str] = Field(default_factory=list)
    side_a_response_families: list[str] = Field(default_factory=list)
    side_b_response_families: list[str] = Field(default_factory=list)
    paired_grounded_sides: bool = False
    paired_response_signal: bool = False
    independence_basis: Literal[
        "cross_paper",
        "mixed_claim_kind",
        "single_family",
        "unknown",
    ] = "unknown"
    independent_family_signal: bool = False

    diagnostic_only: Literal[True] = True
    contrast_is_not_conflict_authority: Literal[True] = True
    boundary_is_not_regime_authority: Literal[True] = True
    scientific_conflict_authority: Literal[False] = False
    causal_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_sorted_unique_types(self) -> "EvidenceLevelTensionWitness":
        if len(self.tension_types) != len(set(self.tension_types)):
            raise ValueError("evidence tension types must be unique")
        return self


class ScientificEvidenceTensionReport(StrictModel):
    schema_version: Literal[
        "scientific-evidence-tension-report-v1"
    ] = "scientific-evidence-tension-report-v1"
    report_id: str
    source_task_id: str
    source_context_id: str
    source_context_sha256: str
    source_explorer_report_id: str | None = None
    source_explorer_report_sha256: str
    source_evidence_sha256: str
    source_packet_sha256: str | None = None
    extraction_mode: TensionExtractionMode
    witnesses: list[EvidenceLevelTensionWitness] = Field(default_factory=list)
    tension_type_counts: dict[str, int] = Field(default_factory=dict)

    llm_calls_performed: Literal[0] = 0
    deterministic_extraction: Literal[True] = True
    scientific_conflict_authority: Literal[False] = False
    regime_boundary_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_witness_ids(self) -> "ScientificEvidenceTensionReport":
        ids = [row.witness_id for row in self.witnesses]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence tension witness IDs must be unique")
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nonblank(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _condition_signature(conditions: list[dict[str, Any]]) -> str:
    normalized = [
        {str(key): row[key] for key in sorted(row)}
        for row in conditions
        if isinstance(row, dict)
    ]
    return _canonical_json(normalized)


def _statement_index(evidence: ScientificReframeEvidencePacket):
    return {
        row.statement_id: row
        for row in [*evidence.premise_statements, *evidence.gap_statements]
    }


def _packet_node_types(explorer_packet: Mapping[str, Any] | None) -> dict[str, str]:
    if explorer_packet is None:
        return {}
    catalog = explorer_packet.get("evidence_catalog", {})
    if not isinstance(catalog, dict):
        return {}
    nodes = catalog.get("nodes", {})
    if not isinstance(nodes, dict):
        return {}
    result: dict[str, str] = {}
    for node_id, raw in nodes.items():
        if not isinstance(raw, dict):
            continue
        node_type = str(raw.get("node_type", "")).strip()
        if node_type:
            result[str(node_id)] = node_type
    return result


_NULL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bno\s+(?:statistically\s+)?significant\b",
        r"\bnot\s+(?:statistically\s+)?significant\b",
        r"\bno\s+(?:detectable|measurable|clear|appreciable)\b",
        r"\bno\s+(?:effect|change|difference|association|correlation)\b",
        r"\b(?:does|did)\s+not\s+(?:affect|alter|change|influence|shift)\b",
        r"\b(?:unchanged|negligible|insensitive)\b",
    )
)

_UP_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:increase|increased|increases|increasing|higher|greater|stronger|enhanced|enhancement|improved)\b",
        r"\bred[- ]?shift(?:ed|s|ing)?\b",
    )
)

_DOWN_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:decrease|decreased|decreases|decreasing|lower|smaller|weaker|reduced|suppressed|diminished)\b",
        r"\bblue[- ]?shift(?:ed|s|ing)?\b",
    )
)

_BOUNDARY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:threshold|crossover|regime|saturat(?:e|ed|ion)|plateau|non[- ]?monotonic)\b",
        r"\b(?:only\s+when|condition[- ]dependent|context[- ]dependent)\b",
        r"\b(?:maximum|minimum|optimum|optimal|offset)\b",
    )
)

_RESPONSE_FAMILY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "performance": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bSERS\b",
            r"\bRaman\b",
            r"\b(?:signal|response|effect|enhancement|sensitivity|detection|readout)\b",
        )
    ),
    "field_hotspot": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:electric|electromagnetic|near[- ]?field)\s+(?:field\s+)?(?:intensity|enhancement|strength)\b",
            r"\bhot[- ]?spot(?:s)?\b",
        )
    ),
    "reliability": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:reproducib(?:ility|le)|reliab(?:ility|le)|accuracy|calibration|uniformity|variability)\b",
        )
    ),
    "spectral_response": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:LSPR|plasmon(?:ic)?\s+(?:resonance|band)|extinction\s+(?:peak|band)|spectral\s+(?:peak|profile))\b",
        )
    ),
}

_MAGNITUDE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:magnitude|fold|times|percent|%)\b",
        r"\b(?:higher|lower|greater|smaller|stronger|weaker|enhanced|reduced)\b",
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z%µμ]+)?",
    )
)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _side_text(statement_ids: list[str], by_statement: Mapping[str, Any]) -> str:
    return " ".join(
        by_statement[statement_id].text
        for statement_id in statement_ids
        if statement_id in by_statement
    )


def _response_families(text: str) -> list[str]:
    return sorted(
        family
        for family, patterns in _RESPONSE_FAMILY_PATTERNS.items()
        if _matches_any(text, patterns)
    )


def _direction(text: str) -> str | None:
    if _matches_any(text, _NULL_PATTERNS):
        return None
    up = _matches_any(text, _UP_PATTERNS)
    down = _matches_any(text, _DOWN_PATTERNS)
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


def _relevant_condition_summary(
    *,
    evidence: ScientificReframeEvidencePacket,
    paper_ids: set[str],
) -> tuple[int, list[str]]:
    signatures: set[str] = set()
    names: set[str] = set()
    for example in evidence.condition_examples:
        if paper_ids and example.paper_id not in paper_ids:
            continue
        signatures.add(_condition_signature(example.conditions))
        for condition in example.conditions:
            name = str(condition.get("name", "")).strip()
            if name:
                names.add(name)
    return len(signatures), sorted(names)


def _node_types_for_statements(
    statement_ids: list[str],
    *,
    by_statement: Mapping[str, Any],
    node_types: Mapping[str, str],
) -> list[str]:
    result: set[str] = set()
    for statement_id in statement_ids:
        statement = by_statement.get(statement_id)
        if statement is None:
            continue
        for node_id in statement.scientific_support_node_ids:
            node_type = node_types.get(node_id)
            if node_type:
                result.add(node_type)
    return sorted(result)


def _classify(
    *,
    source_type: str,
    side_a_ids: list[str],
    side_b_ids: list[str],
    by_statement: Mapping[str, Any],
    side_a_node_types: list[str],
    side_b_node_types: list[str],
    relevant_condition_signature_count: int,
) -> tuple[list[EvidenceTensionType], list[str], list[str], list[str], bool, bool]:
    tension_types: set[EvidenceTensionType] = set()
    bases: list[str] = []

    if source_type == "context_dependency":
        tension_types.add("CONTEXT")
        bases.append("Explorer classified the source tension as context_dependency.")
    elif source_type == "quantitative_difference":
        tension_types.add("MAGNITUDE")
        bases.append("Explorer classified the source tension as quantitative_difference.")
    elif source_type == "qualitative_difference":
        tension_types.add("QUALITATIVE")
        bases.append("Explorer classified the source tension as qualitative_difference.")
    elif source_type == "potential_conflict":
        tension_types.add("POTENTIAL_CONFLICT")
        bases.append("Explorer classified the source relation as potential_conflict only.")

    side_a_text = _side_text(side_a_ids, by_statement)
    side_b_text = _side_text(side_b_ids, by_statement)
    side_a_response_families = _response_families(side_a_text)
    side_b_response_families = _response_families(side_b_text)
    paired_grounded_sides = bool(side_a_text.strip() and side_b_text.strip())
    paired_response_signal = bool(
        paired_grounded_sides
        and side_a_response_families
        and side_b_response_families
    )
    null_a = _matches_any(side_a_text, _NULL_PATTERNS)
    null_b = _matches_any(side_b_text, _NULL_PATTERNS)
    if paired_response_signal and null_a != null_b:
        tension_types.add("NULL_EFFECT")
        bases.append("One grounded side contains an explicit null/no-effect expression and the other side does not.")

    direction_a = _direction(side_a_text)
    direction_b = _direction(side_b_text)
    if (
        paired_response_signal
        and direction_a is not None
        and direction_b is not None
        and direction_a != direction_b
    ):
        tension_types.add("DIRECTIONAL")
        bases.append("The two grounded sides express opposite directional changes.")

    if (
        source_type == "quantitative_difference"
        or paired_response_signal
        and _matches_any(side_a_text, _MAGNITUDE_PATTERNS)
        and _matches_any(side_b_text, _MAGNITUDE_PATTERNS)
    ):
        tension_types.add("MAGNITUDE")
        if source_type != "quantitative_difference":
            bases.append("Both grounded sides contain quantitative or magnitude-comparison cues.")

    claim_kinds = {
        by_statement[statement_id].claim_kind
        for statement_id in [*side_a_ids, *side_b_ids]
        if statement_id in by_statement
    }
    mechanism_node_types = {"MechanismClaim"}
    if (
        "mechanism" in claim_kinds
        or mechanism_node_types.intersection(side_a_node_types)
        or mechanism_node_types.intersection(side_b_node_types)
    ):
        tension_types.add("MECHANISTIC")
        bases.append("At least one grounded side is mechanism-bearing by claim kind or packet node type.")

    measurement_types = {"Measurement", "MeasurementGroup"}
    measurement_a = bool(measurement_types.intersection(side_a_node_types))
    measurement_b = bool(measurement_types.intersection(side_b_node_types))
    if measurement_a or measurement_b:
        tension_types.add("MEASUREMENT")
        bases.append("At least one grounded side is supported by a Measurement or MeasurementGroup node in the Explorer packet.")

    combined_text = f"{side_a_text} {side_b_text}"
    if (
        paired_response_signal
        and relevant_condition_signature_count >= 2
        and source_type in {
            "context_dependency",
            "qualitative_difference",
            "quantitative_difference",
        }
        and _matches_any(combined_text, _BOUNDARY_PATTERNS)
    ):
        tension_types.add("BOUNDARY")
        bases.append(
            "The witness has multiple relevant structured condition signatures plus an explicit boundary/context cue; this is diagnostic only, not regime-boundary authority."
        )

    return (
        sorted(tension_types),
        bases,
        side_a_response_families,
        side_b_response_families,
        paired_grounded_sides,
        paired_response_signal,
    )


def extract_evidence_level_tensions(
    *,
    explorer_report: Mapping[str, Any],
    evidence: ScientificReframeEvidencePacket,
    explorer_packet: Mapping[str, Any] | None = None,
) -> ScientificEvidenceTensionReport:
    task_id = str(explorer_report.get("task_id", "")).strip()
    if task_id and task_id != evidence.task_id:
        raise ValueError("Explorer report task_id does not match reframe evidence task_id")
    if explorer_packet is not None:
        packet_task = explorer_packet.get("task", {})
        packet_task_id = (
            str(packet_task.get("task_id", "")).strip()
            if isinstance(packet_task, dict)
            else ""
        )
        if packet_task_id and packet_task_id != evidence.task_id:
            raise ValueError("Explorer packet task_id does not match reframe evidence task_id")

    by_statement = _statement_index(evidence)
    node_types = _packet_node_types(explorer_packet)
    rows = explorer_report.get("evidence_tensions", [])
    if not isinstance(rows, list):
        rows = []

    witnesses: list[EvidenceLevelTensionWitness] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            continue
        source_type = str(raw.get("tension_type", "")).strip()
        side_a = _nonblank(raw.get("side_a_statement_ids"))
        side_b = _nonblank(raw.get("side_b_statement_ids"))
        focal = str(raw.get("statement_id", "")).strip() or None
        referenced = sorted(
            set(side_a) | set(side_b) | ({focal} if focal else set())
        )
        grounded_ids = [row for row in referenced if row in by_statement]
        if not grounded_ids:
            continue

        papers = set(_nonblank(raw.get("paper_ids")))
        claim_kinds: set[str] = set()
        for statement_id in grounded_ids:
            statement = by_statement[statement_id]
            papers.update(statement.paper_ids)
            if statement.claim_kind:
                claim_kinds.add(statement.claim_kind)

        if len(papers) >= 2:
            independence_basis = "cross_paper"
            independent = True
        elif len(claim_kinds) >= 2:
            independence_basis = "mixed_claim_kind"
            independent = True
        elif grounded_ids:
            independence_basis = "single_family"
            independent = False
        else:
            independence_basis = "unknown"
            independent = False

        side_a_node_types = _node_types_for_statements(
            side_a,
            by_statement=by_statement,
            node_types=node_types,
        )
        side_b_node_types = _node_types_for_statements(
            side_b,
            by_statement=by_statement,
            node_types=node_types,
        )
        condition_count, condition_names = _relevant_condition_summary(
            evidence=evidence,
            paper_ids=papers,
        )
        (
            tension_types,
            bases,
            side_a_response_families,
            side_b_response_families,
            paired_grounded_sides,
            paired_response_signal,
        ) = _classify(
            source_type=source_type,
            side_a_ids=side_a,
            side_b_ids=side_b,
            by_statement=by_statement,
            side_a_node_types=side_a_node_types,
            side_b_node_types=side_b_node_types,
            relevant_condition_signature_count=condition_count,
        )

        source_id = str(raw.get("tension_id", "")).strip() or f"tension:{index}"
        digest = hashlib.sha256(
            "|".join(
                [
                    source_id,
                    source_type,
                    *grounded_ids,
                    *sorted(tension_types),
                    *sorted(papers),
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]
        witnesses.append(
            EvidenceLevelTensionWitness(
                witness_id=f"evidence_tension:{digest}",
                source_tension_id=source_id,
                source_tension_type=source_type,
                focal_statement_id=focal,
                side_a_statement_ids=side_a,
                side_b_statement_ids=side_b,
                grounded_statement_ids=grounded_ids,
                paper_ids=sorted(papers),
                claim_kinds=sorted(claim_kinds),
                tension_types=tension_types,
                classification_bases=bases,
                side_a_node_types=side_a_node_types,
                side_b_node_types=side_b_node_types,
                relevant_condition_signature_count=condition_count,
                relevant_condition_names=condition_names,
                side_a_response_families=side_a_response_families,
                side_b_response_families=side_b_response_families,
                paired_grounded_sides=paired_grounded_sides,
                paired_response_signal=paired_response_signal,
                independence_basis=independence_basis,
                independent_family_signal=independent,
            )
        )

    witnesses.sort(key=lambda row: row.witness_id)
    counts: dict[str, int] = {}
    for witness in witnesses:
        for tension_type in witness.tension_types:
            counts[tension_type] = counts.get(tension_type, 0) + 1

    explorer_payload = dict(explorer_report)
    evidence_payload = evidence.model_dump(mode="json")
    packet_sha = _sha256(dict(explorer_packet)) if explorer_packet is not None else None
    payload = {
        "source_task_id": evidence.task_id,
        "source_context_id": evidence.source_context_id,
        "source_context_sha256": evidence.source_context_sha256,
        "source_explorer_report_sha256": _sha256(explorer_payload),
        "source_evidence_sha256": _sha256(evidence_payload),
        "source_packet_sha256": packet_sha,
        "extraction_mode": "packet_assisted" if explorer_packet is not None else "explorer_only",
        "witnesses": [row.model_dump(mode="json") for row in witnesses],
        "tension_type_counts": dict(sorted(counts.items())),
    }
    report_id = "scientific_evidence_tension:" + hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()[:20]
    return ScientificEvidenceTensionReport(
        report_id=report_id,
        source_task_id=evidence.task_id,
        source_context_id=evidence.source_context_id,
        source_context_sha256=evidence.source_context_sha256,
        source_explorer_report_id=(
            str(explorer_report.get("report_id", "")).strip() or None
        ),
        source_explorer_report_sha256=_sha256(explorer_payload),
        source_evidence_sha256=_sha256(evidence_payload),
        source_packet_sha256=packet_sha,
        extraction_mode=(
            "packet_assisted" if explorer_packet is not None else "explorer_only"
        ),
        witnesses=witnesses,
        tension_type_counts=dict(sorted(counts.items())),
    )
