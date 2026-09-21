from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ResponseMatchSource = Literal["generic_core", "domain_adapter"]
ResponseRole = Literal[
    "outcome_performance",
    "reliability",
    "field_response",
    "spectral_response",
    "measurement_observable",
]
CandidateLawSignalKind = Literal[
    "VOLCANO_NON_MONOTONIC_RESPONSE",
    "FINITE_OPTIMUM",
    "SATURATION_OR_PLATEAU",
    "EXPLICIT_THRESHOLD",
    "MECHANISM_SWITCH",
]


class SemanticCueMatch(StrictModel):
    role: str
    source: ResponseMatchSource
    trigger_eligible: bool
    matches: list[str] = Field(default_factory=list)


class CandidateResponseLawSignal(StrictModel):
    kind: CandidateLawSignalKind
    strength: Literal["sufficient", "supporting"]
    reasons: list[str] = Field(default_factory=list)
    diagnostic_only: Literal[True] = True
    trigger_authority: Literal[False] = False


class ResponseSemanticsStatementProbe(StrictModel):
    statement_id: str
    role: Literal["premise", "gap"]
    text: str
    v1_frozen_response_matched: bool
    v1_frozen_response_matches: list[str] = Field(default_factory=list)
    response_matches: list[SemanticCueMatch] = Field(default_factory=list)
    parameter_matches: list[SemanticCueMatch] = Field(default_factory=list)
    curve_shape_matches: list[SemanticCueMatch] = Field(default_factory=list)
    trigger_eligible_response: bool
    candidate_response_law_signals: list[CandidateResponseLawSignal] = Field(
        default_factory=list
    )
    diagnostic_only: Literal[True] = True


class ScientificResponseSemanticsProbeReport(StrictModel):
    schema_version: Literal[
        "scientific-response-semantics-probe-v1"
    ] = "scientific-response-semantics-probe-v1"
    probe_id: str
    source_diagnostics_pack_id: str
    source_task_id: str
    validation_domain_label: str
    semantics_profile_id: str
    statements: list[ResponseSemanticsStatementProbe] = Field(default_factory=list)
    premise_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    v1_response_match_count: int = Field(ge=0)
    candidate_response_match_count: int = Field(ge=0)
    candidate_law_signal_counts: dict[str, int] = Field(default_factory=dict)
    candidate_latent_response_support_premise_count: int = Field(ge=0)
    candidate_latent_support_paper_count: int = Field(ge=0)
    candidate_latent_current_floor_met: bool

    adaptation_probe: Literal[True] = True
    eligible_as_untouched_validation_after_design_use: Literal[False] = False
    frozen_trigger_semantics_modified: Literal[False] = False
    frozen_prompt_semantics_modified: Literal[False] = False
    actual_trigger_decision_changed: Literal[False] = False
    threshold_tuning_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    scientific_authority: Literal[False] = False
    generalization_claim_established: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificResponseSemanticsProbeReport":
        premises = sum(row.role == "premise" for row in self.statements)
        gaps = sum(row.role == "gap" for row in self.statements)
        if premises != self.premise_count:
            raise ValueError("premise_count does not match statement probes")
        if gaps != self.gap_count:
            raise ValueError("gap_count does not match statement probes")
        return self


_GENERIC_RESPONSE_PATTERNS: dict[ResponseRole, tuple[re.Pattern[str], ...]] = {
    "outcome_performance": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bactivity\b",
            r"\bperformance\b",
            r"\befficienc(?:y|ies)\b",
            r"\byield\b",
            r"\brate\b",
            r"\bselectivit(?:y|ies)\b",
            r"\bstabilit(?:y|ies)\b",
            r"\bconversion\b",
            r"\bturnover\b",
        )
    ),
    "reliability": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\breproducib(?:ility|le)\b",
            r"\breliab(?:ility|le)\b",
            r"\baccuracy\b",
            r"\bcalibration\b",
            r"\buniformity\b",
            r"\bvariability\b",
        )
    ),
    "field_response": (),
    "spectral_response": (),
    "measurement_observable": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bmeasurement\b",
            r"\bobservable\b",
        )
    ),
}

_DOMAIN_RESPONSE_PATTERNS: dict[str, dict[ResponseRole, tuple[re.Pattern[str], ...]]] = {
    "sers": {
        "outcome_performance": tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\bSERS\b",
                r"\b(?:signal|enhancement|sensitivity|detection|readout)\b",
            )
        ),
        "field_response": tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\b(?:electric|electromagnetic|near[- ]?field)\s+(?:field\s+)?(?:intensity|enhancement|strength)\b",
                r"\bhot[- ]?spot(?:s)?\b",
            )
        ),
        "spectral_response": tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\b(?:LSPR|plasmon(?:ic)?\s+(?:resonance|band)|extinction\s+(?:peak|band)|spectral\s+(?:peak|profile))\b",
            )
        ),
        # Raman is an observable/measurement modality by itself; it becomes a
        # performance response only when a task-specific outcome term is also
        # present. Keeping it non-trigger-eligible avoids the v1 false-positive
        # seen in the DAC/HER adaptation probe.
        "measurement_observable": (
            re.compile(r"\bRaman\b", re.IGNORECASE),
        ),
        "reliability": (),
    },
    "dac_her": {
        "outcome_performance": tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\bHER\s+activity\b",
                r"\bhydrogen[- ]evolution\s+activity\b",
                r"\boverpotential\b",
                r"\bcurrent\s+density\b",
                r"\bexchange\s+current\s+density\b",
                r"\bTafel\s+slope\b",
                r"\bturnover\s+frequency\b",
                r"\bTOF\b",
                r"\bFaradaic\s+efficienc(?:y|ies)\b",
            )
        ),
        "measurement_observable": tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\bRaman\s+signal(?:s)?\b",
                r"\bXAS\b",
                r"\bEXAFS\b",
                r"\bXANES\b",
            )
        ),
        "field_response": (),
        "spectral_response": (),
        "reliability": (),
    },
}

_GENERIC_PARAMETER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bfree\s+energy\b",
        r"\bbinding\s+energy\b",
        r"\badsorption\s+energy\b",
        r"\bpotential\b",
        r"\btemperature\b",
        r"\bpressure\b",
        r"\bconcentration\b",
        r"\bpH\b",
        r"\bdistance\b",
        r"\bsize\b",
        r"\bloading\b",
        r"\bcoverage\b",
        r"\btime\b",
    )
)

_DOMAIN_PARAMETER_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "sers": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bgap(?:\s+size)?\b",
            r"\bwavelength\b",
            r"\bLSPR\b",
        )
    ),
    "dac_her": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bGibbs\s+free\s+energy\s+of\s+hydrogen\s+adsorption\b",
            r"\b(?:ΔG_H|delta\s*G[_ -]?H)\b",
            r"\bhydrogen\s+(?:adsorption|binding)\s+(?:free\s+)?energy\b",
        )
    ),
}

_GENERIC_CURVE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "volcano": (
        re.compile(r"\bvolcano\s+(?:relationship|relation|curve|trend)\b", re.IGNORECASE),
    ),
    "finite_optimum": (
        re.compile(r"\b(?:optimum|optimal|best|maximum|minimum|maximized|minimized|peaked)\b", re.IGNORECASE),
    ),
    "saturation": (
        re.compile(r"\b(?:saturat(?:e|ed|es|ing|ion)|plateau(?:ed|ing)?|levels?\s+off)\b", re.IGNORECASE),
    ),
    "threshold": (
        re.compile(r"\b(?:threshold|critical\s+(?:value|point)|above\s+which|below\s+which)\b", re.IGNORECASE),
    ),
    "mechanism_switch": (
        re.compile(
            r"\b(?:mechanism\s+switch|switch(?:es|ed|ing)?\s+(?:from|to|between)|"
            r"transition(?:s|ed|ing)?\s+(?:from|to|between)|becomes?\s+dominant|dominates?\s+instead)\b",
            re.IGNORECASE,
        ),
    ),
}


_TRIGGER_ELIGIBLE_ROLES = {
    "outcome_performance",
    "reliability",
    "field_response",
    "spectral_response",
}


def _canonical_domain(domain: str) -> str:
    normalized = domain.strip().lower().replace("-", "_")
    aliases = {
        "sers_au_ag": "sers",
        "sers": "sers",
        "dac_her": "dac_her",
    }
    return aliases.get(normalized, normalized)


def _matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if value:
                found.append(value)
    return sorted(set(found), key=lambda value: (value.lower(), value))


def _semantic_matches(text: str, domain: str) -> list[SemanticCueMatch]:
    rows: list[SemanticCueMatch] = []
    for role, patterns in _GENERIC_RESPONSE_PATTERNS.items():
        matches = _matches(text, patterns)
        if matches:
            rows.append(
                SemanticCueMatch(
                    role=role,
                    source="generic_core",
                    trigger_eligible=role in _TRIGGER_ELIGIBLE_ROLES,
                    matches=matches,
                )
            )
    for role, patterns in _DOMAIN_RESPONSE_PATTERNS.get(domain, {}).items():
        matches = _matches(text, patterns)
        if matches:
            rows.append(
                SemanticCueMatch(
                    role=role,
                    source="domain_adapter",
                    trigger_eligible=role in _TRIGGER_ELIGIBLE_ROLES,
                    matches=matches,
                )
            )
    return rows


def _parameter_matches(text: str, domain: str) -> list[SemanticCueMatch]:
    rows: list[SemanticCueMatch] = []
    generic = _matches(text, _GENERIC_PARAMETER_PATTERNS)
    if generic:
        rows.append(
            SemanticCueMatch(
                role="scientific_parameter",
                source="generic_core",
                trigger_eligible=False,
                matches=generic,
            )
        )
    domain_hits = _matches(text, _DOMAIN_PARAMETER_PATTERNS.get(domain, ()))
    if domain_hits:
        rows.append(
            SemanticCueMatch(
                role="domain_parameter",
                source="domain_adapter",
                trigger_eligible=False,
                matches=domain_hits,
            )
        )
    return rows


def _curve_matches(text: str) -> list[SemanticCueMatch]:
    rows: list[SemanticCueMatch] = []
    for role, patterns in _GENERIC_CURVE_PATTERNS.items():
        matches = _matches(text, patterns)
        if matches:
            rows.append(
                SemanticCueMatch(
                    role=role,
                    source="generic_core",
                    trigger_eligible=False,
                    matches=matches,
                )
            )
    return rows


def _candidate_law_signals(
    *,
    response_matches: list[SemanticCueMatch],
    parameter_matches: list[SemanticCueMatch],
    curve_matches: list[SemanticCueMatch],
) -> list[CandidateResponseLawSignal]:
    eligible_response = any(row.trigger_eligible for row in response_matches)
    curve_roles = {row.role for row in curve_matches}
    has_parameter = bool(parameter_matches)
    if not eligible_response:
        return []

    rows: list[CandidateResponseLawSignal] = []
    if "volcano" in curve_roles:
        rows.append(
            CandidateResponseLawSignal(
                kind="VOLCANO_NON_MONOTONIC_RESPONSE",
                strength="sufficient",
                reasons=[
                    "A trigger-eligible scientific response is explicitly described by a volcano-shaped relation, which is a direct non-monotonic response-law cue.",
                ],
            )
        )
    if "finite_optimum" in curve_roles and has_parameter:
        rows.append(
            CandidateResponseLawSignal(
                kind="FINITE_OPTIMUM",
                strength="supporting",
                reasons=[
                    "A trigger-eligible response has an explicit finite optimum together with a scientific parameter cue.",
                ],
            )
        )
    if "saturation" in curve_roles:
        rows.append(
            CandidateResponseLawSignal(
                kind="SATURATION_OR_PLATEAU",
                strength="sufficient",
                reasons=[
                    "A trigger-eligible response is explicitly described as saturating or reaching a plateau.",
                ],
            )
        )
    if "threshold" in curve_roles:
        rows.append(
            CandidateResponseLawSignal(
                kind="EXPLICIT_THRESHOLD",
                strength="sufficient",
                reasons=[
                    "A trigger-eligible response is linked to an explicit threshold or critical boundary.",
                ],
            )
        )
    if "mechanism_switch" in curve_roles:
        rows.append(
            CandidateResponseLawSignal(
                kind="MECHANISM_SWITCH",
                strength="sufficient",
                reasons=[
                    "A trigger-eligible response is linked to an explicit mechanism or dominance switch.",
                ],
            )
        )
    return rows


def _stable_id(prefix: str, value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def build_response_semantics_probe(
    diagnostics_payload: dict[str, Any],
) -> ScientificResponseSemanticsProbeReport:
    tasks = diagnostics_payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("response semantics probe currently requires exactly one task")
    task = tasks[0]
    if not isinstance(task, dict):
        raise ValueError("diagnostic task payload must be an object")
    domain = _canonical_domain(str(diagnostics_payload.get("validation_domain_label", "")))
    if not domain:
        raise ValueError("validation_domain_label must be present")

    probes: list[ResponseSemanticsStatementProbe] = []
    latent_support_papers: set[str] = set()
    latent_support_count = 0
    law_counts: dict[str, int] = {}
    for raw in task.get("statement_traces", []):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", ""))
        role = str(raw.get("role", ""))
        if role not in {"premise", "gap"}:
            continue
        response_matches = _semantic_matches(text, domain)
        parameter_matches = _parameter_matches(text, domain)
        curve_matches = _curve_matches(text)
        eligible = any(row.trigger_eligible for row in response_matches)
        signals = _candidate_law_signals(
            response_matches=response_matches,
            parameter_matches=parameter_matches,
            curve_matches=curve_matches,
        )
        for signal in signals:
            law_counts[signal.kind] = law_counts.get(signal.kind, 0) + 1
        if role == "premise" and eligible:
            latent_support_count += 1
            for paper_id in raw.get("paper_ids", []):
                value = str(paper_id).strip()
                if value:
                    latent_support_papers.add(value)

        frozen = raw.get("frozen_trigger_response_cue")
        frozen_matched = bool(frozen.get("matched")) if isinstance(frozen, dict) else False
        frozen_matches = (
            [str(value) for value in frozen.get("matches", [])]
            if isinstance(frozen, dict)
            else []
        )
        probes.append(
            ResponseSemanticsStatementProbe(
                statement_id=str(raw.get("statement_id", "")),
                role=role,
                text=text,
                v1_frozen_response_matched=frozen_matched,
                v1_frozen_response_matches=frozen_matches,
                response_matches=response_matches,
                parameter_matches=parameter_matches,
                curve_shape_matches=curve_matches,
                trigger_eligible_response=eligible,
                candidate_response_law_signals=signals,
            )
        )

    source_pack_id = str(diagnostics_payload.get("pack_id", ""))
    task_id = str(task.get("task_id", ""))
    profile_id = f"scientific-response-semantics-v2-probe:{domain}"
    payload = {
        "source_pack_id": source_pack_id,
        "task_id": task_id,
        "domain": domain,
        "profile_id": profile_id,
        "statement_ids": [row.statement_id for row in probes],
    }
    return ScientificResponseSemanticsProbeReport(
        probe_id=_stable_id("scientific_response_semantics_probe", payload),
        source_diagnostics_pack_id=source_pack_id,
        source_task_id=task_id,
        validation_domain_label=domain,
        semantics_profile_id=profile_id,
        statements=probes,
        premise_count=sum(row.role == "premise" for row in probes),
        gap_count=sum(row.role == "gap" for row in probes),
        v1_response_match_count=sum(row.v1_frozen_response_matched for row in probes),
        candidate_response_match_count=sum(row.trigger_eligible_response for row in probes),
        candidate_law_signal_counts=dict(sorted(law_counts.items())),
        candidate_latent_response_support_premise_count=latent_support_count,
        candidate_latent_support_paper_count=len(latent_support_papers),
        candidate_latent_current_floor_met=(
            latent_support_count >= 3 and len(latent_support_papers) >= 2
        ),
    )


def load_response_semantics_probe(
    diagnostics_path: str | Path,
) -> ScientificResponseSemanticsProbeReport:
    payload = json.loads(Path(diagnostics_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("diagnostics JSON must contain an object")
    return build_response_semantics_probe(payload)
