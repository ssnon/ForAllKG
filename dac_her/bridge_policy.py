from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from dac_her.bridge_schemas import BridgeChunkGraph, BridgeConcept, BridgeLink
from dac_her.scientific_signatures import normalize_scientific_text


BRIDGE_POLICY_VERSION = "dac-her-bridge-policy-v2.3.3-calibration"

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
        r"""
        \b(?:
            correlat\w*
            | associat\w*
            | relationship
            | related\s+to
            | reflect\w*
            | correspond\w*\s+to
        )\b
        """,
        re.I | re.VERBOSE,
    ),
    "VARIES_WITH": re.compile(
        r"""
        \b(?:
            var(?:y|ies|ied|iation)
            | depend\w*\s+on
            | change\w*\s+with
            | different\s+(?:for|among|across)
            | with\s+the\s+(?:increase|decrease)\s+of
            | systematically\s+
            (?:increase|decrease|reduce)\w*
            |
            (?:increase|decrease|reduce)\w*
            \s+from\s+.{0,60}\s+to
        )\b
        """,
        re.I | re.VERBOSE,
    ),
    "COMPETES_WITH": re.compile(r"\b(?:compet\w*|competitive)\b", re.I),
    "COMPETES_FOR": re.compile(r"\b(?:compet\w*|competitive)\b", re.I),
    "SELECTS": re.compile(
        r"\b(?:select\w*|prefer\w*|favor\w*|favou?r\w*|tend\w* to|"
        r"occup(?:y|ies|ied))\b",
        re.I,
    ),
    "CONTRASTS_WITH": re.compile(
        r"""
        \b(?:
            contrast\w*
            | whereas
            | compared\s+(?:with|to)
            | different\s+from
            | in\s+contrast
            | outperform\w*
            |
            (?:
                better
                | worse
                | higher
                | lower
                | superior
                | inferior
                | more\s+favorable
                | less\s+favorable
            )
            (?:\s+[\w-]+){0,4}
            \s+(?:than|to)
        )\b
        """,
        re.I | re.VERBOSE,
    ),
    "MODULATES": re.compile(
        r"""
        \b(?:
            modulat\w*
            | affect\w*
            | redistribut\w*
            | tun(?:e|ed|ing)\s+by
            | alter\w*
        )\b
        """,
        re.I | re.VERBOSE,
    ),
    "MEDIATES": re.compile(r"\bmediat\w*\b", re.I),
    "PROMOTES": re.compile(
        r"""
        \b(?:
            promot\w*
            | enhanc\w*
            | facilitat\w*
            | accelerat\w*
            | boost\w*
            | improv\w*
            | result\w*\s+in
            | lead\w*\s+to
            | stabiliz\w*
            | guarantee\w*
            | thanks\s+to
            | provide\w*
            .{0,60}
            (?:site|sites|nucleation)
            | serve\w*
            .{0,60}
            to\s+stabiliz\w*
        )\b
        """,
        re.I | re.VERBOSE,
    ),
    "SUPPRESSES": re.compile(
        r"\b(?:suppress\w*|inhibit\w*|retard\w*|reduce\w*)\b", re.I
    ),
    "SUGGESTS_DESIGN_RULE": re.compile(
        r"\b(?:suggest\w*|design|principle|strategy|should|can be used to)\b",
        re.I,
    ),
    "IMPOSES_TRADEOFF": re.compile(
        r"""
        \b(?:
            trade[- ]?off
            | at\s+the\s+expense\s+of
            | balance\s+between
            | compromise
            | comes?\s+with
            .{0,50}
            sacrifice\s+of
        )\b
        """,
        re.I | re.VERBOSE,
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
_CANDIDATE_ONLY_CODES = frozenset({
    "RELATION_CUE_MISMATCH",
    "ARGUMENT_SCOPE_AMBIGUOUS",
    "CAUSAL_ARGUMENT_SCOPE_AMBIGUOUS",
    (
        "TABLE_DERIVED_RELATION_"
        "REQUIRES_CONTEXT"
    ),
})

_FIGURE_CAPTION = re.compile(
    r"""
    ^\s*
    (?:\*\*)?
    (?:
        fig(?:ure)?\.?\s*[\w.-]+
        |
        effects?\s+of\b
    )
    """,
    re.I | re.VERBOSE,
)

_EXPLICIT_TREND = re.compile(
    r"""
    \b(?:
        increase\w*
        | decrease\w*
        | reduc\w*
        | rise\w*
        | fall\w*
        | higher
        | lower
        | better
        | worse
        | superior
        | inferior
        | from\s+.{0,40}\s+to
        | correlat\w*
        | depend\w*
        | var(?:y|ies|ied)
    )\b
    """,
    re.I | re.VERBOSE,
)

_TABLE_ROW = re.compile(
    r"^\s*\|.*\|\s*$",
    re.S,
)

_FAILURE_EVENT = re.compile(
    r"""
    \b(?:
        fail\w*
        | degrad\w*
        | deactivat\w*
        | collaps\w*
        | dissolv\w*
        | agglomerat\w*
        | poison\w*
        | detach\w*
        | leach\w*
        | reconstruct\w*
    )\b
    """,
    re.I | re.VERBOSE,
)

_TABLE_ONLY_DERIVED_CODE = (
    "TABLE_DERIVED_RELATION_REQUIRES_CONTEXT"
)

@dataclass(frozen=True)
class BridgePolicyPartition:
    accepted: BridgeChunkGraph
    candidates: BridgeChunkGraph
    candidate_records: tuple[
        BridgeRejection,
        ...,
    ]
    rejections: tuple[
        BridgeRejection,
        ...
    ]


def _candidate_only(
    issues: list[BridgePolicyIssue],
) -> bool:
    codes = {
        issue.code
        for issue in issues
    }

    return (
        bool(codes)
        and codes.issubset(
            _CANDIDATE_ONLY_CODES
        )
    )

_COORDINATED_CAUSAL_INTERPRETATION = (
    re.compile(
        r"""
        \bsuggest\w*
        .{0,160}
        \band\s+
        (?:
            enhanc\w*
            | improv\w*
            | promot\w*
        )
        """,
        re.I | re.VERBOSE,
    )
)

@dataclass(frozen=True)
class BridgePolicyIssue:
    code: str
    field: str
    detail: str
    repairable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class BridgeRejection:
    paper_id: str
    chunk_id: str
    concept_id: str
    label: str

    retention_lane: str
    pattern_subject: str
    pattern_relation: str
    pattern_object: str
    pattern_support_mode: str

    subject_evidence_phrase: str
    relation_evidence_phrase: str
    object_evidence_phrase: str
    source_phrase: str

    reason_codes: tuple[str, ...]
    reason_details: tuple[
        dict[str, Any],
        ...
    ]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["reason_codes"] = list(
            self.reason_codes
        )
        row["reason_details"] = list(
            self.reason_details
        )
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


def _policy_issue(
    code: str,
    field: str,
    detail: str,
    *,
    repairable: bool = False,
) -> BridgePolicyIssue:
    return BridgePolicyIssue(
        code=code,
        field=field,
        detail=detail,
        repairable=repairable,
    )

def _dedupe_issues(
    issues: Iterable[BridgePolicyIssue],
) -> list[BridgePolicyIssue]:
    result: list[BridgePolicyIssue] = []
    seen: set[
        tuple[str, str, str, bool]
    ] = set()

    for issue in issues:
        signature = (
            issue.code,
            issue.field,
            issue.detail,
            issue.repairable,
        )

        if signature in seen:
            continue

        seen.add(signature)
        result.append(issue)

    return result

def _relation_cue_supported(
    concept: BridgeConcept,
    cue: re.Pattern[str],
) -> bool:
    candidates = [
        concept.relation_evidence_phrase or "",
        concept.source_phrase,
        *concept.supporting_phrases,
    ]

    return any(
        cue.search(candidate)
        for candidate in candidates
        if candidate
    )

def _has_any_relation_cue(
    text: str,
) -> bool:
    return any(
        cue.search(text)
        for cue
        in _RELATION_EVIDENCE_CUES.values()
    )

def _pattern_grounding_issues(
    concept: BridgeConcept,
    *,
    core_text: str | None,
    linked_links: list[BridgeLink],
) -> list[BridgePolicyIssue]:
    issues: list[
        BridgePolicyIssue
    ] = []

    relation = str(
        concept.pattern_relation or ""
    )

    if (
        concept.pattern_support_mode
        == "explicit_single_span"
    ):
        span = (
            concept.supporting_phrases[0]
        )

        checks = (
            (
                "SUBJECT_EVIDENCE_NOT_IN_SPAN",
                "subject_evidence_phrase",
                concept.subject_evidence_phrase,
            ),
            (
                "RELATION_EVIDENCE_NOT_IN_SPAN",
                "relation_evidence_phrase",
                concept.relation_evidence_phrase,
            ),
            (
                "OBJECT_EVIDENCE_NOT_IN_SPAN",
                "object_evidence_phrase",
                concept.object_evidence_phrase,
            ),
        )

        for code, field, phrase in checks:
            if not _normalized_contains(
                span,
                phrase,
            ):
                issues.append(
                    BridgePolicyIssue(
                        code=code,
                        field=field,
                        detail=(
                            f"{field} is not "
                            "contained in the "
                            "supporting span"
                        ),
                        repairable=True,
                    )
                )

        cue = (
            _RELATION_EVIDENCE_CUES.get(
                relation
            )
        )

        if cue is None:
            issues.append(
                BridgePolicyIssue(
                    code=(
                        "UNSUPPORTED_RELATION"
                    ),
                    field=(
                        "pattern_relation"
                    ),
                    detail=(
                        "No deterministic "
                        "cue policy exists "
                        f"for {relation!r}."
                    ),
                    repairable=False,
                )
            )

        elif not _relation_cue_supported(
            concept,
            cue,
        ):
            issues.append(
                BridgePolicyIssue(
                    code=(
                        "RELATION_CUE_MISMATCH"
                    ),
                    field=(
                        "relation_evidence_phrase"
                    ),
                    detail=(
                        "Relation evidence "
                        "does not lexically "
                        f"support {relation}."
                    ),
                    repairable=True,
                )
            )

        if not _phrase_in_core(
            span,
            core_text,
        ):
            issues.append(
                BridgePolicyIssue(
                    code=(
                        "SOURCE_SPAN_NOT_IN_CORE"
                    ),
                    field=(
                        "supporting_phrases"
                    ),
                    detail=(
                        "Supporting span is "
                        "not verbatim in "
                        "CORE_TEXT."
                    ),
                    repairable=False,
                )
            )
    elif (
        concept.pattern_support_mode
        == "derived_multi_span"
    ):
        if relation not in {
            "CORRELATES_WITH",
            "VARIES_WITH",
            "CONTRASTS_WITH",
        }:
            issues.append(
                _policy_issue(
                    "UNSUPPORTED_DERIVED_RELATION",
                    "pattern_relation",
                    (
                        "Derived multi-span support is "
                        "restricted to comparative "
                        "non-causal relations."
                    ),
                )
            )

        pairs = {
            (
                normalize_scientific_text(
                    item.subject_value
                ),
                normalize_scientific_text(
                    item.object_value
                ),
            )
            for item in concept.comparison_items
        }

        subject_values = {
            subject
            for subject, _ in pairs
            if subject
        }

        object_values = {
            object_
            for _, object_ in pairs
            if object_
        }

        if (
            len(pairs) < 2
            or len(subject_values) < 2
        ):
            issues.append(
                _policy_issue(
                    "INSUFFICIENT_COMPARISON_EVIDENCE",
                    "comparison_items",
                    (
                        "At least two distinct "
                        "comparison items are required."
                    ),
                )
            )

        if (
            relation
            in {
                "VARIES_WITH",
                "CONTRASTS_WITH",
            }
            and len(object_values) < 2
        ):
            issues.append(
                _policy_issue(
                    "INSUFFICIENT_COMPARISON_EVIDENCE",
                    "comparison_items",
                    (
                        "The compared outcomes do not "
                        "provide distinct object values."
                    ),
                )
            )

        if any(
            not _phrase_in_core(
                item.source_phrase,
                core_text,
            )
            for item in concept.comparison_items
        ):
            issues.append(
                _policy_issue(
                    "SOURCE_SPAN_NOT_IN_CORE",
                    "comparison_items",
                    (
                        "One or more comparison source "
                        "phrases are not in CORE_TEXT."
                    ),
                )
            )

        if any(
            link.evidence_strength != "indirect"
            for link in linked_links
        ):
            issues.append(
                _policy_issue(
                    (
                        "DERIVED_RELATION_REQUIRES_"
                        "INDIRECT_EVIDENCE"
                    ),
                    "links",
                    (
                        "Derived relations must use "
                        "indirect evidence links."
                    ),
                )
            )
    # 기존 derived_multi_span 검사는
    # 같은 방식으로 Issue 객체를 생성한다.

    return issues

def _competition_issues(
    concept: BridgeConcept,
) -> list[BridgePolicyIssue]:
    relation = concept.pattern_relation

    if relation not in {
        "COMPETES_WITH",
        "COMPETES_FOR",
    }:
        return []

    issues: list[BridgePolicyIssue] = []
    qualifiers = _qualifier_map(concept)

    subject = (
        concept.pattern_subject or ""
    )
    object_ = (
        concept.pattern_object or ""
    )

    if relation == "COMPETES_WITH":
        if not qualifiers.get(
            "competition_target"
        ):
            issues.append(
                _policy_issue(
                    "COMPETITION_TARGET_MISSING",
                    "qualifiers",
                    (
                        "COMPETES_WITH requires a "
                        "competition_target qualifier."
                    ),
                )
            )

        if (
            _COLLECTIVE_COMPETITOR_TERMS.search(
                subject
            )
            and _COMPETITION_TARGET_TERMS.search(
                object_
            )
        ):
            issues.append(
                _policy_issue(
                    "COMPETITION_ARGUMENT_MISMATCH",
                    "pattern_object",
                    (
                        "A collective competitor set "
                        "cannot directly use the "
                        "competition target as its peer."
                    ),
                )
            )

    if relation == "COMPETES_FOR":
        members = qualifiers.get(
            "competitor_members",
            "",
        )

        if _member_count(members) < 2:
            issues.append(
                _policy_issue(
                    "COMPETITOR_MEMBERS_MISSING",
                    "qualifiers",
                    (
                        "COMPETES_FOR requires at "
                        "least two competitor members."
                    ),
                )
            )

    return issues

def _relation_direction_issues(
    concept: BridgeConcept,
) -> list[BridgePolicyIssue]:
    """Reject the pilot's known reversed VARIES_WITH orientation.

    Preferred orientation:
        outcome/property VARIES_WITH condition/axis

    Example:
        preferred adsorption site
            VARIES_WITH
        anchored metal identity
    """
    if concept.pattern_relation != "VARIES_WITH":
        return []

    subject = normalize_scientific_text(
        concept.pattern_subject or ""
    )
    object_ = normalize_scientific_text(
        concept.pattern_object or ""
    )

    if (
        "identity" in subject
        and "identity" not in object_
    ):
        return [
            _policy_issue(
                "RELATION_ARGUMENT_DIRECTION",
                "pattern_subject",
                (
                    "VARIES_WITH should normally orient "
                    "the varying outcome/property as the "
                    "subject and the condition or axis as "
                    "the object."
                ),
                repairable=False,
            )
        ]

    return []

def _figure_caption_issues(
    concept: BridgeConcept,
) -> list[BridgePolicyIssue]:
    if (
        concept.retention_lane
        != "accepted_pattern"
    ):
        return []

    if (
        _FIGURE_CAPTION.search(
            concept.source_phrase
        )
        and not _EXPLICIT_TREND.search(
            concept.source_phrase
        )
    ):
        return [
            _policy_issue(
                "FIGURE_CAPTION_WITHOUT_TREND",
                "source_phrase",
                (
                    "A figure caption names variables "
                    "but does not state a directional "
                    "or comparative result."
                ),
            )
        ]

    return []

def _single_row_variation_issues(
    concept: BridgeConcept,
) -> list[BridgePolicyIssue]:
    if (
        concept.pattern_relation
        != "VARIES_WITH"
        or concept.pattern_support_mode
        != "explicit_single_span"
    ):
        return []

    if _TABLE_ROW.fullmatch(
        concept.source_phrase
    ):
        return [
            _policy_issue(
                "SINGLE_ROW_VARIATION_INFERENCE",
                "source_phrase",
                (
                    "One table row cannot establish "
                    "variation across an identity, "
                    "condition, or composition axis. "
                    "Use derived_multi_span with at "
                    "least two distinct comparison items."
                ),
            )
        ]

    return []

def _failure_mode_issues(
    concept: BridgeConcept,
) -> list[BridgePolicyIssue]:
    if (
        concept.pattern_relation
        != "IDENTIFIES_FAILURE_MODE"
    ):
        return []

    evidence = " ".join([
        concept.source_phrase,
        (
            concept.relation_evidence_phrase
            or ""
        ),
    ])

    if not _FAILURE_EVENT.search(
        evidence
    ):
        return [
            _policy_issue(
                "FAILURE_MODE_WITHOUT_FAILURE_EVENT",
                "pattern_relation",
                (
                    "IDENTIFIES_FAILURE_MODE "
                    "requires an actual degradation, "
                    "instability, or failure event."
                ),
            )
        ]

    return []

def _cross_clause_causal_scope_issues(
    concept: BridgeConcept,
) -> list[BridgePolicyIssue]:
    if concept.pattern_relation not in {
        "PROMOTES",
        "SUPPRESSES",
        "MODULATES",
        "MEDIATES",
    }:
        return []

    source = normalize_scientific_text(
        concept.source_phrase
    )
    subject_phrase = (
        normalize_scientific_text(
            concept.subject_evidence_phrase
            or ""
        )
    )
    relation_phrase = (
        normalize_scientific_text(
            concept.relation_evidence_phrase
            or ""
        )
    )

    boundary = source.find("in which")
    subject_pos = source.find(
        subject_phrase
    )
    relation_pos = source.find(
        relation_phrase
    )

    if (
        boundary >= 0
        and subject_pos >= 0
        and relation_pos >= 0
        and subject_pos < boundary
        and relation_pos > boundary
    ):
        return [
            _policy_issue(
                "ARGUMENT_SCOPE_AMBIGUOUS",
                "pattern_subject",
                (
                    "The extracted subject occurs "
                    "before a clause boundary, while "
                    "the causal predicate occurs in "
                    "a later clause with a potentially "
                    "different grammatical actor."
                ),
                repairable=True,
            )
        ]

    return []

def _causal_scope_issues(
    concept: BridgeConcept,
) -> list[BridgePolicyIssue]:
    if concept.pattern_relation not in {
        "PROMOTES",
        "SUPPRESSES",
        "MODULATES",
        "MEDIATES",
    }:
        return []

    if (
        _COORDINATED_CAUSAL_INTERPRETATION
        .search(concept.source_phrase)
    ):
        return [
            _policy_issue(
                "CAUSAL_ARGUMENT_SCOPE_AMBIGUOUS",
                "pattern_subject",
                (
                    "The source presents coordinated "
                    "interpretations, so the extracted "
                    "subject may not be the grammatical "
                    "cause of the object."
                ),
                repairable=True,
            )
        ]

    return []

def _table_derived_context_issues(
    concept: BridgeConcept,
) -> list[BridgePolicyIssue]:
    if (
        concept.retention_lane
        != "accepted_pattern"
        or concept.pattern_support_mode
        != "derived_multi_span"
    ):
        return []

    phrases = [
        item.source_phrase
        for item
        in concept.comparison_items
        if item.source_phrase
    ]

    if (
        phrases
        and all(
            _TABLE_ROW.fullmatch(
                phrase
            )
            for phrase in phrases
        )
    ):
        return [
            _policy_issue(
                (
                    "TABLE_DERIVED_RELATION_"
                    "REQUIRES_CONTEXT"
                ),
                "comparison_items",
                (
                    "The relation is derived only "
                    "from bare table rows. Table "
                    "header or column-semantic "
                    "context is required before "
                    "confirmed acceptance."
                ),
                repairable=True,
            )
        ]

    return []

def concept_policy_issues(
    concept: BridgeConcept,
    *,
    strict_nodes: Iterable[dict[str, Any]],
    core_text: str | None = None,
    linked_links: list[BridgeLink] | None = None,
) -> list[BridgePolicyIssue]:
    """Apply high-precision deterministic Bridge policy checks."""

    issues: list[BridgePolicyIssue] = []

    normalized_label = normalize_scientific_text(
        concept.label
    )
    normalized_phrase = normalize_scientific_text(
        concept.source_phrase
    )

    if normalized_label in _GENERIC_LABELS:
        issues.append(
            _policy_issue(
                "GENERIC_LANGUAGE",
                "label",
                (
                    "The concept label is too generic "
                    "to represent a reusable Bridge concept."
                ),
            )
        )

    if concept.retention_lane == "accepted_pattern":
        subject = normalize_scientific_text(
            concept.pattern_subject or ""
        )
        object_ = normalize_scientific_text(
            concept.pattern_object or ""
        )

        if (
            not subject
            or not object_
            or subject == object_
        ):
            issues.append(
                _policy_issue(
                    "RELATION_MISSING",
                    "pattern_subject/pattern_object",
                    (
                        "Accepted patterns require two "
                        "distinct non-empty arguments."
                    ),
                )
            )

        if (
            _METRIC_TERMS.search(subject)
            and _METRIC_TERMS.search(object_)
            and not _has_any_relation_cue(
                concept.source_phrase
            )
            and concept.pattern_support_mode
            != "derived_multi_span"
        ):
            issues.append(
                _policy_issue(
                    "UNSUPPORTED_RELATION",
                    "source_phrase",
                    (
                        "The source phrase does not expose "
                        "a relation cue supporting the two "
                        "metric-like arguments."
                    ),
                )
            )

        issues.extend(
            _pattern_grounding_issues(
                concept,
                core_text=core_text,
                linked_links=(
                    linked_links or []
                ),
            )
        )

        issues.extend(
            _competition_issues(
                concept
            )
        )

        issues.extend(
            _relation_direction_issues(
                concept
            )
        )

        issues.extend(
            _figure_caption_issues(
                concept
            )
        )

        issues.extend(
            _single_row_variation_issues(
                concept
            )
        )

        issues.extend(
            _failure_mode_issues(
                concept
            )
        )

        issues.extend(
            _causal_scope_issues(
                concept
            )
        )
        issues.extend(
            _cross_clause_causal_scope_issues(
                concept
            )
        )
        issues.extend(
            _table_derived_context_issues(
                concept
            )
        )

        return _dedupe_issues(issues)

    # paper_local_frontier checks
    if _TABLE_FIELD_CUES.fullmatch(
        normalized_label
    ):
        issues.append(
            _policy_issue(
                "TABLE_FIELD",
                "label",
                (
                    "A bare table-field label is already "
                    "represented in the strict evidence graph."
                ),
            )
        )

    combined_text = " ".join(
        (
            concept.label,
            concept.source_phrase,
            concept.description or "",
        )
    )

    if (
        _METRIC_TERMS.search(normalized_label)
        and not _has_any_relation_cue(
            combined_text
        )
    ):
        issues.append(
            _policy_issue(
                "SCALAR_METRIC",
                "label",
                (
                    "The candidate is a scalar metric "
                    "rather than a reusable frontier concept."
                ),
            )
        )

    strict_labels = _strict_labels(
        strict_nodes
    )

    if (
        normalized_label in strict_labels
        or normalized_phrase in strict_labels
    ):
        issues.append(
            _policy_issue(
                "STRICT_DUPLICATE",
                "label/source_phrase",
                (
                    "The candidate duplicates content "
                    "already represented in the strict graph."
                ),
            )
        )

    if (
        _NUMERIC_OR_UNIT.search(
            concept.source_phrase
        )
        and _METRIC_TERMS.search(
            concept.source_phrase
        )
    ):
        issues.append(
            _policy_issue(
                "INSTANCE_ONLY",
                "source_phrase",
                (
                    "The candidate is a paper-specific "
                    "numeric instance rather than a "
                    "reusable Bridge concept."
                ),
            )
        )

    return _dedupe_issues(issues)

def filter_bridge_result(
    result: BridgeChunkGraph,
    *,
    strict_nodes: list[dict[str, Any]],
    core_text: str | None = None,
) -> tuple[
    BridgeChunkGraph,
    list[BridgeRejection],
]:
    partition = partition_bridge_result(
        result,
        strict_nodes=strict_nodes,
        core_text=core_text,
    )

    return (
        partition.accepted,
        [
            *partition.candidate_records,
            *partition.rejections,
        ],
    )

def concept_rejection_reasons(
    concept: BridgeConcept,
    *,
    strict_nodes: Iterable[
        dict[str, Any]
    ],
    core_text: str | None = None,
    linked_links: list[
        BridgeLink
    ] | None = None,
) -> list[str]:
    return list(dict.fromkeys(
        issue.code
        for issue in concept_policy_issues(
            concept,
            strict_nodes=strict_nodes,
            core_text=core_text,
            linked_links=linked_links,
        )
    ))

def partition_bridge_result(
    result: BridgeChunkGraph,
    *,
    strict_nodes: list[dict[str, Any]],
    core_text: str | None = None,
) -> BridgePolicyPartition:
    accepted_ids: set[str] = set()
    candidate_ids: set[str] = set()

    accepted_concepts: list[
        BridgeConcept
    ] = []
    candidate_concepts: list[
        BridgeConcept
    ] = []

    candidate_records: list[
        BridgeRejection
    ] = []
    rejections: list[
        BridgeRejection
    ] = []

    seen_signatures: set[tuple[str, ...]] = set()
    links_by_concept: dict[str, list[BridgeLink]] = {}
    for link in result.links:
        links_by_concept.setdefault(link.concept_id, []).append(link)

    for concept in result.concepts:
        issues = concept_policy_issues(
            concept,
            strict_nodes=strict_nodes,
            core_text=core_text,
            linked_links=links_by_concept.get(
                concept.id,
                [],
            ),
        )

        signature = (
            concept.retention_lane,
            normalize_scientific_text(
                concept.label
            ),
            normalize_scientific_text(
                concept.pattern_subject or ""
            ),
            str(
                concept.pattern_relation or ""
            ),
            normalize_scientific_text(
                concept.pattern_object or ""
            ),
        )

        if signature in seen_signatures:
            issues.append(
                _policy_issue(
                    "DUPLICATE_MENTION",
                    "label",
                    (
                        "An equivalent Bridge mention "
                        "already appeared in this chunk."
                    ),
                )
            )

        seen_signatures.add(signature)

        if not issues:
            accepted_concepts.append(
                concept
            )
            accepted_ids.add(concept.id)
            continue

        record = BridgeRejection(
                paper_id=result.paper_id,
                chunk_id=result.chunk_id,
                concept_id=concept.id,
                label=concept.label,
                retention_lane=(
                    concept.retention_lane
                ),
                pattern_subject=(
                    concept.pattern_subject or ""
                ),
                pattern_relation=(
                    concept.pattern_relation or ""
                ),
                pattern_object=(
                    concept.pattern_object or ""
                ),
                pattern_support_mode=(
                    concept.pattern_support_mode
                    or ""
                ),
                subject_evidence_phrase=(
                    concept.subject_evidence_phrase
                    or ""
                ),
                relation_evidence_phrase=(
                    concept.relation_evidence_phrase
                    or ""
                ),
                object_evidence_phrase=(
                    concept.object_evidence_phrase
                    or ""
                ),
                source_phrase=(
                    concept.source_phrase
                ),
                reason_codes=tuple(
                    dict.fromkeys(
                        issue.code
                        for issue in issues
                    )
                ),
                reason_details=tuple(
                    issue.to_dict()
                    for issue in issues
                ),
            )

        if _candidate_only(issues):
            candidate_concepts.append(
                concept
            )
            candidate_ids.add(
                concept.id
            )
            candidate_records.append(
                record
            )
        else:
            rejections.append(
                record
            )

    accepted = result.model_copy(
        update={
            "concepts": accepted_concepts,
            "links": [
                link
                for link in result.links
                if link.concept_id
                in accepted_ids
            ],
        }
    )

    candidates = result.model_copy(
        update={
            "concepts": candidate_concepts,
            "links": [
                link
                for link in result.links
                if link.concept_id
                in candidate_ids
            ],
        }
    )

    return BridgePolicyPartition(
        accepted=(
            BridgeChunkGraph
            .model_validate(
                accepted.model_dump()
            )
        ),
        candidates=(
            BridgeChunkGraph
            .model_validate(
                candidates.model_dump()
            )
        ),
        candidate_records=tuple(
            candidate_records
        ),
        rejections=tuple(
            rejections
        ),
    )