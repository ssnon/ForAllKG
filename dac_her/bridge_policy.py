from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from dac_her.bridge_schemas import BridgeChunkGraph, BridgeConcept, BridgeLink
from dac_her.scientific_signatures import normalize_scientific_text


BRIDGE_POLICY_VERSION = "dac-her-bridge-policy-v2.2.1"

_GENERIC_LABELS = {
    "high performance",
    "excellent activity",
    "good activity",
    "important effect",
    "strong interaction",
    "this behavior",
    "experimental result",
    "calculation result",
    "high activity",
    "better performance",
    "optimized structure",
    "stable structure",
    "favorable property",
}

_METRIC_TERMS = re.compile(
    r"\b(?:distance|height|angle|bond length|coordination number|"
    r"adsorption energy|binding energy|formation energy|free energy|"
    r"overpotential|tafel slope|current density|charge density|"
    r"work function|bader charge|dos|pdos|parameter|metric|value)\b",
    re.I,
)

_RELATION_CUES = re.compile(
    r"\b(?:correlat|associat|vary|varies|depend|increase|decrease|trend|"
    r"compete|competition|select|prefer|favor|tend|contrast|modulat|mediat|"
    r"promot|suppress|trade[- ]?off|failure|reconstruct|reorganiz|"
    r"facilitat|inhibit|control|govern|determin)\w*\b",
    re.I,
)

_TABLE_FIELD_CUES = re.compile(
    r"^(?:site )?(?:height|distance|angle|parameter|value|metric)(?: parameter)?$",
    re.I,
)

_NUMERIC_OR_UNIT = re.compile(
    r"(?:\b\d+(?:\.\d+)?\b|å|angstrom|ev\b|mv\b|ma\b|cm-?2\b|cm\^?-?2)",
    re.I,
)

_RELATION_EVIDENCE_CUES: dict[str, re.Pattern[str]] = {
    "CORRELATES_WITH": re.compile(
        r"\b(?:correlat\w*|associat\w*|relationship|related to)\b", re.I
    ),
    "VARIES_WITH": re.compile(
        r"\b(?:var(?:y|ies|ied|iation)|depend\w* on|change\w* with|"
        r"different (?:for|among|across)|as .{0,80} (?:increase|decrease|change))\b",
        re.I,
    ),
    "COMPETES_WITH": re.compile(r"\b(?:compet\w*|competitive)\b", re.I),
    "COMPETES_FOR": re.compile(r"\b(?:compet\w*|competitive)\b", re.I),
    "SELECTS": re.compile(
        r"\b(?:select\w*|prefer\w*|favor\w*|favou?r\w*|tend\w* to|"
        r"occup(?:y|ies|ied))\b",
        re.I,
    ),
    "CONTRASTS_WITH": re.compile(
        r"\b(?:contrast\w*|whereas|while|compared with|different from|"
        r"in contrast)\b",
        re.I,
    ),
    "MODULATES": re.compile(r"\bmodulat\w*\b", re.I),
    "MEDIATES": re.compile(r"\bmediat\w*\b", re.I),
    "PROMOTES": re.compile(
        r"\b(?:promot\w*|enhanc\w*|facilitat\w*|accelerat\w*)\b", re.I
    ),
    "SUPPRESSES": re.compile(
        r"\b(?:suppress\w*|inhibit\w*|retard\w*|reduce\w*)\b", re.I
    ),
    "SUGGESTS_DESIGN_RULE": re.compile(
        r"\b(?:suggest\w*|design|principle|strategy|should|can be used to)\b",
        re.I,
    ),
    "IMPOSES_TRADEOFF": re.compile(
        r"\b(?:trade[- ]?off|at the expense of|balance between|compromise)\b",
        re.I,
    ),
    "IDENTIFIES_FAILURE_MODE": re.compile(
        r"\b(?:fail\w*|degrad\w*|deactivat\w*|collapse\w*|dissolv\w*|"
        r"agglomerat\w*)\b",
        re.I,
    ),
}

_COMPETITION_TARGET_TERMS = re.compile(
    r"\b(?:adsorption|binding|occupancy|reaction|activity|site selection|"
    r"intermediate|proton|hydrogen|resource|target|substrate)\b",
    re.I,
)
_COLLECTIVE_COMPETITOR_TERMS = re.compile(
    r"\b(?:motifs?|sites?|classes?|configurations?|states?|species|pathways?)\b",
    re.I,
)


@dataclass(frozen=True)
class BridgeRejection:
    paper_id: str
    chunk_id: str
    concept_id: str
    label: str
    source_phrase: str
    reason_codes: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["reason_codes"] = list(self.reason_codes)
        return row


def _strict_labels(strict_nodes: Iterable[dict[str, Any]]) -> set[str]:
    labels: set[str] = set()
    for node in strict_nodes:
        for key in ("label", "metric_id"):
            value = normalize_scientific_text(node.get(key, ""))
            if value:
                labels.add(value)
    return labels


def _qualifier_map(concept: BridgeConcept) -> dict[str, str]:
    return {
        normalize_scientific_text(qualifier.key).replace(" ", "_"): qualifier.value
        for qualifier in concept.qualifiers
    }


def _normalized_contains(parent: str, child: str | None) -> bool:
    if not child:
        return False
    return normalize_scientific_text(child) in normalize_scientific_text(parent)


def _phrase_in_core(phrase: str, core_text: str | None) -> bool:
    if core_text is None:
        return True
    return normalize_scientific_text(phrase) in normalize_scientific_text(core_text)


def _member_count(value: str) -> int:
    normalized = value.strip()
    if not normalized:
        return 0
    parts = re.split(r"\s*(?:;|\bvs\.?\b|\bversus\b|\band\b)\s*", normalized, flags=re.I)
    return len([part for part in parts if part.strip()])


def _pattern_grounding_reasons(
    concept: BridgeConcept,
    *,
    core_text: str | None,
    linked_links: list[BridgeLink],
) -> list[str]:
    reasons: list[str] = []
    relation = str(concept.pattern_relation or "")

    if concept.pattern_support_mode == "explicit_single_span":
        span = concept.supporting_phrases[0]
        evidence_phrases = (
            concept.subject_evidence_phrase,
            concept.relation_evidence_phrase,
            concept.object_evidence_phrase,
        )
        if not all(_normalized_contains(span, phrase) for phrase in evidence_phrases):
            reasons.append("PATTERN_NOT_ENTAILED")

        cue = _RELATION_EVIDENCE_CUES.get(relation)
        relation_evidence = concept.relation_evidence_phrase or ""
        if cue is None or not cue.search(relation_evidence):
            reasons.append("PATTERN_NOT_ENTAILED")

        if not _phrase_in_core(span, core_text):
            reasons.append("UNSUPPORTED_SOURCE_SPAN")

    elif concept.pattern_support_mode == "derived_multi_span":
        if relation not in {"CORRELATES_WITH", "VARIES_WITH", "CONTRASTS_WITH"}:
            reasons.append("UNSUPPORTED_DERIVED_RELATION")

        pairs = {
            (
                normalize_scientific_text(item.subject_value),
                normalize_scientific_text(item.object_value),
            )
            for item in concept.comparison_items
        }
        subject_values = {left for left, _ in pairs if left}
        object_values = {right for _, right in pairs if right}
        if len(pairs) < 2 or len(subject_values) < 2:
            reasons.append("INSUFFICIENT_COMPARISON_EVIDENCE")
        if relation in {"VARIES_WITH", "CONTRASTS_WITH"} and len(object_values) < 2:
            reasons.append("INSUFFICIENT_COMPARISON_EVIDENCE")

        if any(
            not _phrase_in_core(item.source_phrase, core_text)
            for item in concept.comparison_items
        ):
            reasons.append("UNSUPPORTED_SOURCE_SPAN")

        if any(link.evidence_strength != "indirect" for link in linked_links):
            reasons.append("DERIVED_RELATION_REQUIRES_INDIRECT_EVIDENCE")

    return reasons


def _competition_reasons(concept: BridgeConcept) -> list[str]:
    relation = concept.pattern_relation
    if relation not in {"COMPETES_WITH", "COMPETES_FOR"}:
        return []

    reasons: list[str] = []
    qualifiers = _qualifier_map(concept)
    subject = concept.pattern_subject or ""
    object_ = concept.pattern_object or ""

    if relation == "COMPETES_WITH":
        if not qualifiers.get("competition_target"):
            reasons.append("COMPETITION_TARGET_MISSING")
        # Reject the exact semantic error observed in the pilot: a collective
        # set of competitors connected directly to the process they compete for.
        if (
            _COLLECTIVE_COMPETITOR_TERMS.search(subject)
            and _COMPETITION_TARGET_TERMS.search(object_)
        ):
            reasons.append("COMPETITION_ARGUMENT_MISMATCH")

    if relation == "COMPETES_FOR":
        members = qualifiers.get("competitor_members", "")
        if _member_count(members) < 2:
            reasons.append("COMPETITOR_MEMBERS_MISSING")
        if normalize_scientific_text(subject) == normalize_scientific_text(object_):
            reasons.append("COMPETITION_ARGUMENT_MISMATCH")

    return reasons


def _relation_direction_reasons(concept: BridgeConcept) -> list[str]:
    # Conservative guard for the pilot's reversed identity/site relation. It is
    # intentionally narrow to avoid imposing a universal causal orientation.
    if concept.pattern_relation != "VARIES_WITH":
        return []
    subject = normalize_scientific_text(concept.pattern_subject or "")
    object_ = normalize_scientific_text(concept.pattern_object or "")
    if "identity" in subject and "identity" not in object_:
        return ["RELATION_ARGUMENT_DIRECTION"]
    return []


def concept_rejection_reasons(
    concept: BridgeConcept,
    *,
    strict_nodes: Iterable[dict[str, Any]],
    core_text: str | None = None,
    linked_links: list[BridgeLink] | None = None,
) -> list[str]:
    """High-precision deterministic rejection rules.

    Accepted patterns must expose auditable source support. Frontier concepts
    are filtered more aggressively because table fields and scalar metrics are
    already represented in the canonical evidence graph.
    """
    reasons: list[str] = []
    normalized_label = normalize_scientific_text(concept.label)
    normalized_phrase = normalize_scientific_text(concept.source_phrase)

    if normalized_label in _GENERIC_LABELS:
        reasons.append("GENERIC_LANGUAGE")

    if concept.retention_lane == "accepted_pattern":
        subject = normalize_scientific_text(concept.pattern_subject or "")
        object_ = normalize_scientific_text(concept.pattern_object or "")
        if not subject or not object_ or subject == object_:
            reasons.append("RELATION_MISSING")
        if (
            _METRIC_TERMS.search(subject)
            and _METRIC_TERMS.search(object_)
            and not _RELATION_CUES.search(concept.source_phrase)
            and concept.pattern_support_mode != "derived_multi_span"
        ):
            reasons.append("UNSUPPORTED_RELATION")

        reasons.extend(
            _pattern_grounding_reasons(
                concept,
                core_text=core_text,
                linked_links=linked_links or [],
            )
        )
        reasons.extend(_competition_reasons(concept))
        reasons.extend(_relation_direction_reasons(concept))
        return list(dict.fromkeys(reasons))

    if _TABLE_FIELD_CUES.fullmatch(normalized_label):
        reasons.append("TABLE_FIELD")

    if _METRIC_TERMS.search(normalized_label) and not _RELATION_CUES.search(
        " ".join((concept.label, concept.source_phrase, concept.description or ""))
    ):
        reasons.append("SCALAR_METRIC")

    strict_labels = _strict_labels(strict_nodes)
    if normalized_label in strict_labels or normalized_phrase in strict_labels:
        reasons.append("STRICT_DUPLICATE")

    if _NUMERIC_OR_UNIT.search(concept.source_phrase) and _METRIC_TERMS.search(
        concept.source_phrase
    ):
        reasons.append("INSTANCE_ONLY")

    return list(dict.fromkeys(reasons))


def filter_bridge_result(
    result: BridgeChunkGraph,
    *,
    strict_nodes: list[dict[str, Any]],
    core_text: str | None = None,
) -> tuple[BridgeChunkGraph, list[BridgeRejection]]:
    rejections: list[BridgeRejection] = []
    retained: list[BridgeConcept] = []
    retained_ids: set[str] = set()
    seen_signatures: set[tuple[str, ...]] = set()
    links_by_concept: dict[str, list[BridgeLink]] = {}
    for link in result.links:
        links_by_concept.setdefault(link.concept_id, []).append(link)

    for concept in result.concepts:
        reasons = concept_rejection_reasons(
            concept,
            strict_nodes=strict_nodes,
            core_text=core_text,
            linked_links=links_by_concept.get(concept.id, []),
        )
        signature = (
            concept.retention_lane,
            normalize_scientific_text(concept.label),
            normalize_scientific_text(concept.pattern_subject or ""),
            str(concept.pattern_relation or ""),
            normalize_scientific_text(concept.pattern_object or ""),
        )
        if signature in seen_signatures:
            reasons.append("DUPLICATE_MENTION")
        seen_signatures.add(signature)

        if reasons:
            rejections.append(
                BridgeRejection(
                    paper_id=result.paper_id,
                    chunk_id=result.chunk_id,
                    concept_id=concept.id,
                    label=concept.label,
                    source_phrase=concept.source_phrase,
                    reason_codes=tuple(dict.fromkeys(reasons)),
                    detail=(
                        "Deterministic Bridge v2.2 policy rejected a candidate "
                        "whose full relation was not sufficiently grounded, whose "
                        "competition arguments were malformed, or whose content "
                        "belongs in the canonical evidence graph."
                    ),
                )
            )
            continue
        retained.append(concept)
        retained_ids.add(concept.id)

    retained_links = [
        link for link in result.links if link.concept_id in retained_ids
    ]
    filtered = result.model_copy(
        update={"concepts": retained, "links": retained_links}
    )
    filtered = BridgeChunkGraph.model_validate(filtered.model_dump())
    return filtered, rejections
