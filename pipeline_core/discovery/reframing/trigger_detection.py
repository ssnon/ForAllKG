from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from pipeline_core.discovery.reframing.evidence_tension import (
    ScientificEvidenceTensionReport,
    extract_evidence_level_tensions,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ConditionDiversitySignal,
    DirectScientificTriggerSignal,
    ReframeOperatorTriggerAssessment,
    ScientificReframeTriggerReport,
    ScientificTensionWitness,
    stable_trigger_report_id,
)


_TENSION_MAP = {
    "context_dependency": "CONTEXT_DEPENDENCY",
    "qualitative_difference": "QUALITATIVE_DIFFERENCE",
    "quantitative_difference": "MAGNITUDE_DIFFERENCE",
    "potential_conflict": "POTENTIAL_CONFLICT",
    "insufficient_context": "INSUFFICIENT_CONTEXT",
}

_LEGACY_LATENT_TRIGGER_TYPES = {
    "CONTEXT_DEPENDENCY",
    "QUALITATIVE_DIFFERENCE",
    "MAGNITUDE_DIFFERENCE",
    "POTENTIAL_CONFLICT",
}

_LEGACY_REGIME_TRIGGER_TYPES = {
    "CONTEXT_DEPENDENCY",
    "QUALITATIVE_DIFFERENCE",
    "MAGNITUDE_DIFFERENCE",
}

_REFINED_LATENT_TYPES = {
    "CONTEXT",
    "MECHANISTIC",
    "MEASUREMENT",
    "DIRECTIONAL",
    "NULL_EFFECT",
    "MAGNITUDE",
    "QUALITATIVE",
    "POTENTIAL_CONFLICT",
}

_REFINED_REGIME_SECONDARY_TYPES = {
    "DIRECTIONAL",
    "NULL_EFFECT",
    "QUALITATIVE",
}

_STRONG_MEDIATION_GAP_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcompensat(?:e|es|ed|ing|ion)\b",
        r"\bequaliz(?:e|es|ed|ing|ation)\b",
        r"\bnegat(?:e|es|ed|ing|ion)\b",
        r"\bcounteract(?:s|ed|ing)?\b",
        r"\bovercome(?:s|ing)?\b",
        r"\bjoint(?:ly)?\b",
        r"\binteract(?:s|ed|ing|ion)?\b",
        r"\bmediat(?:e|es|ed|ing|ion)\b",
    )
)

_RESPONSE_CUE = re.compile(
    r"\b(?:SERS|Raman|signal|response|effect|enhancement|sensitivity|detection|readout|"
    r"reproducib(?:ility|le)|reliab(?:ility|le)|accuracy|calibration|uniformity|"
    r"electric\s+field|electromagnetic\s+field|field\s+intensity|hot[- ]?spot(?:s)?)\b",
    re.IGNORECASE,
)
_UP_CUE = re.compile(
    r"\b(?:increase|increased|increases|increasing|higher|greater|stronger|enhanced|improved|rise|rises|rose)\b",
    re.IGNORECASE,
)
_DOWN_CUE = re.compile(
    r"\b(?:decrease|decreased|decreases|decreasing|lower|smaller|weaker|reduced|decline|declined|drops?|dropped|diminished)\b",
    re.IGNORECASE,
)
_CONTRAST_CUE = re.compile(
    r"\b(?:but|however|whereas|while|although|yet|followed\s+by|at\s+the\s+smallest|at\s+the\s+largest)\b",
    re.IGNORECASE,
)
_FINITE_OPTIMUM_CUE = re.compile(
    r"\b(?:optimum|optimal|best|maximum|minimum|maximized|minimized|peaked)\b",
    re.IGNORECASE,
)
_PARAMETER_CUE = re.compile(
    r"\b(?:gap|distance|size|diameter|length|width|ratio|concentration|temperature|pH|"
    r"wavelength|frequency|time|duration|loading|coverage|porosity|pressure|potential|voltage|current)\b",
    re.IGNORECASE,
)
_SATURATION_CUE = re.compile(
    r"\b(?:saturat(?:e|ed|es|ing|ion)|plateau(?:ed|ing)?|levels?\s+off|no\s+further\s+(?:increase|improvement))\b",
    re.IGNORECASE,
)
_THRESHOLD_CUE = re.compile(
    r"\b(?:threshold|critical\s+(?:value|point|size|distance|concentration)|above\s+which|below\s+which)\b",
    re.IGNORECASE,
)
_MECHANISM_SWITCH_CUE = re.compile(
    r"\b(?:mechanism\s+switch|switch(?:es|ed|ing)?\s+(?:from|to|between)|"
    r"transition(?:s|ed|ing)?\s+(?:from|to|between)|becomes?\s+dominant|dominates?\s+instead)\b",
    re.IGNORECASE,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _condition_signature(conditions: list[dict[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for row in conditions:
        if not isinstance(row, dict):
            continue
        normalized.append({str(key): row[key] for key in sorted(row)})
    return _canonical_json(normalized)


def _condition_diversity(
    evidence: ScientificReframeEvidencePacket,
) -> ConditionDiversitySignal:
    signatures: set[str] = set()
    papers: set[str] = set()
    kinds: set[str] = set()
    names: set[str] = set()
    for example in evidence.condition_examples:
        signatures.add(_condition_signature(example.conditions))
        papers.add(example.paper_id)
        kinds.add(example.object_kind)
        for condition in example.conditions:
            name = str(condition.get("name", "")).strip()
            if name:
                names.add(name)
    return ConditionDiversitySignal(
        example_count=len(evidence.condition_examples),
        paper_count=len(papers),
        distinct_condition_signature_count=len(signatures),
        object_kinds=sorted(kinds),
        condition_names=sorted(names),
        diversity_present=(
            len(evidence.condition_examples) >= 2
            and len(signatures) >= 2
        ),
    )


def _normalized_source_type(source_type: str) -> str | None:
    return _TENSION_MAP.get(source_type)


def _trigger_witnesses_from_evidence_report(
    report: ScientificEvidenceTensionReport,
) -> list[ScientificTensionWitness]:
    witnesses: list[ScientificTensionWitness] = []
    for row in report.witnesses:
        normalized = _normalized_source_type(row.source_tension_type)
        if normalized is None:
            continue
        witnesses.append(
            ScientificTensionWitness(
                witness_id=row.witness_id,
                source_tension_id=row.source_tension_id,
                source_tension_type=row.source_tension_type,
                normalized_tension_type=normalized,
                focal_statement_id=row.focal_statement_id,
                side_a_statement_ids=list(row.side_a_statement_ids),
                side_b_statement_ids=list(row.side_b_statement_ids),
                paper_ids=list(row.paper_ids),
                grounded_statement_ids=list(row.grounded_statement_ids),
                distinct_claim_kinds=list(row.claim_kinds),
                evidence_tension_types=list(row.tension_types),
                classification_bases=list(row.classification_bases),
                relevant_condition_signature_count=(
                    row.relevant_condition_signature_count
                ),
                side_a_response_families=list(row.side_a_response_families),
                side_b_response_families=list(row.side_b_response_families),
                paired_grounded_sides=row.paired_grounded_sides,
                paired_response_signal=row.paired_response_signal,
                independence_basis=row.independence_basis,
                independent_family_signal=row.independent_family_signal,
            )
        )
    return sorted(witnesses, key=lambda row: row.witness_id)


def _stable_direct_signal_id(
    *,
    kind: str,
    operator_id: str,
    statement_ids: list[str],
    gap_statement_ids: list[str],
) -> str:
    payload = {
        "kind": kind,
        "operator_id": operator_id,
        "statement_ids": sorted(statement_ids),
        "gap_statement_ids": sorted(gap_statement_ids),
    }
    return "direct_trigger:" + hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()[:16]


def _direct_trigger_signals(
    evidence: ScientificReframeEvidencePacket,
) -> list[DirectScientificTriggerSignal]:
    signals: list[DirectScientificTriggerSignal] = []

    # LATENT_VARIABLE: a mediation/compensation gap can justify one hidden-variable
    # attempt even when the evidence families are complementary rather than in
    # explicit tension. This lane deliberately requires a strong bridge cue in a
    # grounded gap plus multi-paper, mechanism-bearing premises.
    mediation_gaps = [
        row
        for row in evidence.gap_statements
        if any(pattern.search(row.text) for pattern in _STRONG_MEDIATION_GAP_PATTERNS)
    ]
    mechanism_like = [
        row
        for row in evidence.premise_statements
        if row.claim_kind in {"mechanism", "association", "observation", "comparison"}
        and _RESPONSE_CUE.search(row.text)
    ]
    mechanism_papers = {
        paper_id
        for row in mechanism_like
        for paper_id in row.paper_ids
        if paper_id
    }
    if mediation_gaps and len(mechanism_like) >= 3 and len(mechanism_papers) >= 2:
        statement_ids = sorted({row.statement_id for row in mechanism_like})
        gap_ids = sorted({row.statement_id for row in mediation_gaps})
        signal_id = _stable_direct_signal_id(
            kind="MEDIATION_GAP",
            operator_id="LATENT_VARIABLE",
            statement_ids=statement_ids,
            gap_statement_ids=gap_ids,
        )
        signals.append(
            DirectScientificTriggerSignal(
                signal_id=signal_id,
                kind="MEDIATION_GAP",
                target_operator_id="LATENT_VARIABLE",
                strength="sufficient",
                statement_ids=statement_ids,
                gap_statement_ids=gap_ids,
                reasons=[
                    "A grounded gap asks whether one supported factor compensates for, balances, negates, or otherwise mediates another supported factor.",
                    "At least three response-bearing premises from at least two papers support a complementary-mechanism shadow attempt without requiring an explicit conflict witness.",
                ],
            )
        )

    # REGIME_BOUNDARY: direct within-statement response-law evidence outranks
    # cross-paper lexical boundary cues. Non-monotonicity, saturation, explicit
    # thresholds, and mechanism switches are sufficient; a finite optimum is
    # retained as supporting evidence but does not trigger by itself.
    for row in evidence.premise_statements:
        text = row.text
        if not _RESPONSE_CUE.search(text):
            continue
        kinds: list[tuple[str, str, str]] = []
        if (
            _UP_CUE.search(text)
            and _DOWN_CUE.search(text)
            and _CONTRAST_CUE.search(text)
        ):
            kinds.append((
                "NON_MONOTONIC_RESPONSE",
                "sufficient",
                "One grounded premise contains both increasing and decreasing response directions linked by an explicit contrast cue.",
            ))
        if _SATURATION_CUE.search(text):
            kinds.append((
                "SATURATION_OR_PLATEAU",
                "sufficient",
                "One grounded premise explicitly reports saturation, a plateau, or no further response improvement.",
            ))
        if _THRESHOLD_CUE.search(text):
            kinds.append((
                "EXPLICIT_THRESHOLD",
                "sufficient",
                "One grounded premise explicitly reports a threshold or critical response boundary.",
            ))
        if _MECHANISM_SWITCH_CUE.search(text):
            kinds.append((
                "MECHANISM_SWITCH",
                "sufficient",
                "One grounded premise explicitly reports a mechanism switch or dominance transition.",
            ))
        if _FINITE_OPTIMUM_CUE.search(text) and _PARAMETER_CUE.search(text):
            kinds.append((
                "FINITE_OPTIMUM",
                "supporting",
                "One grounded premise reports a finite optimum/best response at a parameter value; this supports but does not alone establish a regime call.",
            ))
        for kind, strength, reason in kinds:
            signal_id = _stable_direct_signal_id(
                kind=kind,
                operator_id="REGIME_BOUNDARY",
                statement_ids=[row.statement_id],
                gap_statement_ids=[],
            )
            signals.append(
                DirectScientificTriggerSignal(
                    signal_id=signal_id,
                    kind=kind,
                    target_operator_id="REGIME_BOUNDARY",
                    strength=strength,
                    statement_ids=[row.statement_id],
                    reasons=[reason],
                )
            )

    signals.sort(key=lambda row: row.signal_id)
    return signals


def _validate_tension_lineage(
    *,
    tension_report: ScientificEvidenceTensionReport,
    evidence: ScientificReframeEvidencePacket,
) -> None:
    if tension_report.source_task_id != evidence.task_id:
        raise ValueError("evidence tension task_id does not match reframe evidence")
    if tension_report.source_context_id != evidence.source_context_id:
        raise ValueError("evidence tension context_id does not match reframe evidence")
    if tension_report.source_context_sha256 != evidence.source_context_sha256:
        raise ValueError(
            "evidence tension context_sha256 does not match reframe evidence"
        )


def _legacy_assessments(
    *,
    witnesses: list[ScientificTensionWitness],
    condition: ConditionDiversitySignal,
) -> list[ReframeOperatorTriggerAssessment]:
    latent_witnesses = [
        row
        for row in witnesses
        if row.normalized_tension_type in _LEGACY_LATENT_TRIGGER_TYPES
        and row.independent_family_signal
    ]
    if latent_witnesses:
        latent = ReframeOperatorTriggerAssessment(
            operator_id="LATENT_VARIABLE",
            decision="triggered",
            witness_ids=[row.witness_id for row in latent_witnesses],
            reasons=[
                "At least one grounded contrast/tension spans independent evidence families; a hidden-variable explanation is worth a shadow attempt.",
                "The trigger is diagnostic only: contrast does not establish scientific conflict or the existence of a latent variable.",
            ],
            independent_family_required=True,
            scientific_trigger_signal=True,
        )
    elif any(
        row.normalized_tension_type in _LEGACY_LATENT_TRIGGER_TYPES
        for row in witnesses
    ):
        latent = ReframeOperatorTriggerAssessment(
            operator_id="LATENT_VARIABLE",
            decision="insufficient_trigger_evidence",
            witness_ids=[
                row.witness_id
                for row in witnesses
                if row.normalized_tension_type in _LEGACY_LATENT_TRIGGER_TYPES
            ],
            reasons=[
                "A contrast/tension is present, but the recovered witness does not span two deterministic evidence-family proxies (cross-paper or mixed claim-kind).",
            ],
            independent_family_required=True,
            scientific_trigger_signal=False,
        )
    else:
        latent = ReframeOperatorTriggerAssessment(
            operator_id="LATENT_VARIABLE",
            decision="not_triggered",
            reasons=[
                "No grounded Explorer evidence tension relevant to LATENT_VARIABLE was recovered for this task scope.",
            ],
            independent_family_required=True,
            scientific_trigger_signal=False,
        )

    regime_witnesses = [
        row
        for row in witnesses
        if row.normalized_tension_type in _LEGACY_REGIME_TRIGGER_TYPES
    ]
    if regime_witnesses and condition.diversity_present:
        regime = ReframeOperatorTriggerAssessment(
            operator_id="REGIME_BOUNDARY",
            decision="triggered",
            witness_ids=[row.witness_id for row in regime_witnesses],
            reasons=[
                "A context/qualitative/magnitude tension is present and the grounded semantic scope contains multiple structured condition signatures.",
                "Condition diversity is an availability signal only; it does not itself establish a regime boundary.",
            ],
            condition_diversity_required=True,
            scientific_trigger_signal=True,
        )
    elif regime_witnesses:
        regime = ReframeOperatorTriggerAssessment(
            operator_id="REGIME_BOUNDARY",
            decision="insufficient_trigger_evidence",
            witness_ids=[row.witness_id for row in regime_witnesses],
            reasons=[
                "A regime-relevant tension is present, but structured condition diversity is insufficient to justify a REGIME_BOUNDARY generation call.",
            ],
            condition_diversity_required=True,
            scientific_trigger_signal=False,
        )
    else:
        regime = ReframeOperatorTriggerAssessment(
            operator_id="REGIME_BOUNDARY",
            decision="not_triggered",
            reasons=[
                "No grounded context/qualitative/magnitude tension relevant to REGIME_BOUNDARY was recovered.",
            ],
            condition_diversity_required=True,
            scientific_trigger_signal=False,
        )
    return [latent, regime]


def _refined_assessments(
    *,
    witnesses: list[ScientificTensionWitness],
    direct_signals: list[DirectScientificTriggerSignal],
    condition: ConditionDiversitySignal,
) -> list[ReframeOperatorTriggerAssessment]:
    latent_witnesses = [
        row
        for row in witnesses
        if row.independent_family_signal
        and row.paired_grounded_sides
        and row.paired_response_signal
        and bool(set(row.evidence_tension_types) & _REFINED_LATENT_TYPES)
    ]
    latent_direct = [
        row
        for row in direct_signals
        if row.target_operator_id == "LATENT_VARIABLE"
        and row.strength == "sufficient"
    ]
    if latent_witnesses or latent_direct:
        reasons: list[str] = []
        if latent_witnesses:
            reasons.append(
                "At least one independent evidence-level tension has two grounded, response-bearing sides suitable for hidden-variable reasoning."
            )
        if latent_direct:
            reasons.append(
                "A grounded mediation/compensation gap links complementary supported mechanisms, so a latent bridge construct is worth one shadow attempt even without explicit tension."
            )
        reasons.append(
            "The trigger remains diagnostic only; it does not establish conflict, causality, or the existence of a latent variable."
        )
        latent = ReframeOperatorTriggerAssessment(
            operator_id="LATENT_VARIABLE",
            decision="triggered",
            witness_ids=[row.witness_id for row in latent_witnesses],
            direct_signal_ids=[row.signal_id for row in latent_direct],
            reasons=reasons,
            independent_family_required=bool(latent_witnesses),
            scientific_trigger_signal=True,
        )
    else:
        weak_latent = [
            row
            for row in witnesses
            if bool(set(row.evidence_tension_types) & _REFINED_LATENT_TYPES)
        ]
        if weak_latent:
            latent = ReframeOperatorTriggerAssessment(
                operator_id="LATENT_VARIABLE",
                decision="insufficient_trigger_evidence",
                witness_ids=[row.witness_id for row in weak_latent],
                reasons=[
                    "A nominal evidence tension exists, but no witness has both grounded response-bearing sides plus the required evidence-family independence signal.",
                    "One-sided tensions and cross-variable comparisons are retained diagnostically but do not consume a LATENT_VARIABLE generation call.",
                ],
                independent_family_required=True,
                scientific_trigger_signal=False,
            )
        else:
            latent = ReframeOperatorTriggerAssessment(
                operator_id="LATENT_VARIABLE",
                decision="not_triggered",
                reasons=[
                    "No paired response tension or grounded mediation gap satisfies the refined LATENT_VARIABLE trigger contract.",
                ],
                independent_family_required=True,
                scientific_trigger_signal=False,
            )

    regime_direct = [
        row
        for row in direct_signals
        if row.target_operator_id == "REGIME_BOUNDARY"
        and row.strength == "sufficient"
    ]
    regime_supporting = [
        row
        for row in direct_signals
        if row.target_operator_id == "REGIME_BOUNDARY"
        and row.strength == "supporting"
    ]

    # Pair-level regime evidence must compare response-bearing sides. Merely
    # observing that one parameter moved "below" a value while another response
    # increased is a cross-variable comparison, not a response-law boundary.
    regime_witnesses: list[ScientificTensionWitness] = []
    for row in witnesses:
        if not (row.paired_grounded_sides and row.paired_response_signal):
            continue
        types = set(row.evidence_tension_types)
        strong_boundary = "BOUNDARY" in types
        context_plus_response = (
            "CONTEXT" in types
            and bool(types & _REFINED_REGIME_SECONDARY_TYPES)
        )
        if (
            (strong_boundary or context_plus_response)
            and row.relevant_condition_signature_count >= 2
        ):
            regime_witnesses.append(row)

    if regime_direct or regime_witnesses:
        reasons = []
        if regime_direct:
            reasons.append(
                "At least one grounded premise directly reports a non-monotonic response, saturation/plateau, explicit threshold, or mechanism switch."
            )
        if regime_witnesses:
            reasons.append(
                "At least one pair-local tension compares response-bearing sides and carries a boundary or context-plus-response signal with local condition support."
            )
        if regime_supporting:
            reasons.append(
                "Finite-optimum statements are retained as supporting evidence; they are not sufficient for a regime call by themselves."
            )
        reasons.append(
            "These signals justify one shadow attempt only and do not establish a scientific regime boundary."
        )
        regime = ReframeOperatorTriggerAssessment(
            operator_id="REGIME_BOUNDARY",
            decision="triggered",
            witness_ids=[row.witness_id for row in regime_witnesses],
            direct_signal_ids=[
                row.signal_id for row in [*regime_direct, *regime_supporting]
            ],
            reasons=reasons,
            condition_diversity_required=bool(regime_witnesses),
            scientific_trigger_signal=True,
        )
    else:
        broad_regime_relevant = [
            row
            for row in witnesses
            if (
                "BOUNDARY" in row.evidence_tension_types
                or "CONTEXT" in row.evidence_tension_types
                or "MAGNITUDE" in row.evidence_tension_types
                or bool(
                    set(row.evidence_tension_types)
                    & _REFINED_REGIME_SECONDARY_TYPES
                )
            )
        ]
        if broad_regime_relevant or regime_supporting:
            regime = ReframeOperatorTriggerAssessment(
                operator_id="REGIME_BOUNDARY",
                decision="insufficient_trigger_evidence",
                witness_ids=[row.witness_id for row in broad_regime_relevant],
                direct_signal_ids=[row.signal_id for row in regime_supporting],
                reasons=[
                    "Boundary-like or finite-optimum evidence is present, but no direct response-law change or paired response witness satisfies the stricter REGIME_BOUNDARY contract.",
                    "Parameter movement, lexical boundary cues, and finite optima alone are not treated as regime evidence.",
                ],
                condition_diversity_required=True,
                scientific_trigger_signal=False,
            )
        else:
            regime = ReframeOperatorTriggerAssessment(
                operator_id="REGIME_BOUNDARY",
                decision="not_triggered",
                reasons=[
                    "No direct response-law signal or pair-local response tension satisfies the refined REGIME_BOUNDARY trigger contract.",
                ],
                condition_diversity_required=True,
                scientific_trigger_signal=False,
            )
    return [latent, regime]

def detect_scientific_reframe_triggers(
    *,
    explorer_report: Mapping[str, Any],
    evidence: ScientificReframeEvidencePacket,
    explorer_packet: Mapping[str, Any] | None = None,
    evidence_tension_report: ScientificEvidenceTensionReport | None = None,
) -> ScientificReframeTriggerReport:
    task_id = str(explorer_report.get("task_id", "")).strip()
    if task_id and task_id != evidence.task_id:
        raise ValueError("Explorer report task_id does not match reframe evidence task_id")

    if evidence_tension_report is None:
        evidence_tension_report = extract_evidence_level_tensions(
            explorer_report=explorer_report,
            evidence=evidence,
            explorer_packet=explorer_packet,
        )
    _validate_tension_lineage(
        tension_report=evidence_tension_report,
        evidence=evidence,
    )
    witnesses = _trigger_witnesses_from_evidence_report(
        evidence_tension_report
    )
    condition = _condition_diversity(evidence)
    direct_signals = _direct_trigger_signals(evidence)

    refined = evidence_tension_report.extraction_mode == "packet_assisted"
    assessments = (
        _refined_assessments(
            witnesses=witnesses,
            direct_signals=direct_signals,
            condition=condition,
        )
        if refined
        else _legacy_assessments(witnesses=witnesses, condition=condition)
    )

    explorer_payload = dict(explorer_report)
    explorer_sha = _sha256(explorer_payload)
    evidence_payload = evidence.model_dump(mode="json")
    evidence_sha = _sha256(evidence_payload)
    report_payload = {
        "source_task_id": evidence.task_id,
        "source_context_id": evidence.source_context_id,
        "source_context_sha256": evidence.source_context_sha256,
        "source_explorer_report_sha256": explorer_sha,
        "source_evidence_sha256": evidence_sha,
        "source_evidence_tension_report_id": evidence_tension_report.report_id,
        "trigger_resolution_mode": (
            "evidence_level_refined" if refined else "legacy_explorer_tension"
        ),
        "tension_witnesses": [row.model_dump(mode="json") for row in witnesses],
        "direct_trigger_signals": [
            row.model_dump(mode="json") for row in direct_signals
        ],
        "condition_diversity": condition.model_dump(mode="json"),
        "assessments": [row.model_dump(mode="json") for row in assessments],
    }
    return ScientificReframeTriggerReport(
        report_id=stable_trigger_report_id(report_payload),
        source_task_id=evidence.task_id,
        source_context_id=evidence.source_context_id,
        source_context_sha256=evidence.source_context_sha256,
        source_explorer_report_id=(
            str(explorer_report.get("report_id", "")).strip() or None
        ),
        source_explorer_report_sha256=explorer_sha,
        source_evidence_sha256=evidence_sha,
        source_evidence_tension_report_id=evidence_tension_report.report_id,
        trigger_resolution_mode=(
            "evidence_level_refined" if refined else "legacy_explorer_tension"
        ),
        tension_witnesses=witnesses,
        direct_trigger_signals=direct_signals,
        condition_diversity=condition,
        assessments=assessments,
    )
