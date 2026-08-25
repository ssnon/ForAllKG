from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio


class HypothesisValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Literal["error", "warning"]
    code: str
    location: str
    message: str


class HypothesisValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passes: bool
    errors: int
    warnings: int
    issues: list[HypothesisValidationIssue]


_NOVELTY_PATTERNS = (
    re.compile(r"\bnovel\b", re.I),
    re.compile(r"\bunprecedented\b", re.I),
    re.compile(r"\bpreviously\s+(?:unknown|unreported|unrecognized)\b", re.I),
    re.compile(r"\bfor\s+the\s+first\s+time\b", re.I),
    re.compile(r"\bfirst\s+(?:report|demonstration|observation|evidence)\b", re.I),
    re.compile(r"\bnever\s+before\b", re.I),
)

_PROTOCOL_PATTERNS = (
    re.compile(r"\b(?:synthesi[sz]e|anneal|calcine|pyroly[sz]e)\b.{0,50}\b(?:at|for|under)\b", re.I),
    re.compile(r"\bscan\s+rate\b", re.I),
    re.compile(r"\b(?:rpm|revolutions?\s+per\s+minute)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:mM|M)\s+(?:KOH|NaOH|H2SO4|HClO4|electrolyte)\b", re.I),
    re.compile(r"\b(?:heat|hold|stir|sonicate)\b.{0,35}\b\d+(?:\.\d+)?\s*(?:°?C|K|h|hr|hours?|min|minutes?)\b", re.I),
    re.compile(r"\b(?:working|counter|reference)\s+electrode\b.{0,80}\b(?:use|prepare|load|coat)\b", re.I),
)

_ABSENCE_RE = re.compile(
    r"\b(?:absent|absence|not\s+reported|does\s+not\s+report|did\s+not\s+report|"
    r"no\s+evidence\s+of|no\s+support\s+for|lacks?|without)\b",
    re.I,
)
_PAPER_SCOPE_RE = re.compile(r"\b(?:paper|study|article|source|Kiwook_\d+)\b", re.I)

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?:\s*(?:"
    r"(?i:eV|meV|mV|mA|nm|pm|cm|mm|ms|mol|pH)"
    r"|Å|V|A|K|°C|C|%|s|h|M"
    r"))?"
    r"(?![A-Za-z])"
)


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = re.sub(r"[\u2010-\u2015\u2212-]+", " ", value)
    value = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _numbers(text: str) -> set[str]:
    return {" ".join(match.group(0).split()).lower() for match in _NUMBER_RE.finditer(text)}


def _observable_matches(a: str, b: str) -> bool:
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return False
    return na == nb or (len(na) >= 5 and na in nb) or (len(nb) >= 5 and nb in na)


def _card_text(card: object) -> str:
    parts: list[str] = [
        str(getattr(card, "title", "")),
        str(getattr(card, "hypothesis_statement", "")),
        str(getattr(card, "inferential_bridge", "")),
    ]
    for row in getattr(card, "predicted_observations", []):
        parts.extend([str(row.observable), str(row.rationale)])
    for row in getattr(card, "falsification_criteria", []):
        parts.extend([str(row.observable), str(row.falsifying_outcome)])
    parts.extend(map(str, getattr(card, "assumptions", [])))
    return "\n".join(parts)


class HypothesisValidator:
    def validate(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
    ) -> HypothesisValidationResult:
        issues: list[HypothesisValidationIssue] = []

        def error(code: str, location: str, message: str) -> None:
            issues.append(
                HypothesisValidationIssue(
                    severity="error", code=code, location=location, message=message
                )
            )

        def warning(code: str, location: str, message: str) -> None:
            issues.append(
                HypothesisValidationIssue(
                    severity="warning", code=code, location=location, message=message
                )
            )

        if portfolio.source_context_id != context.context_id:
            error("CONTEXT_ID_MISMATCH", "portfolio.source_context_id", "source context ID mismatch")
        if portfolio.source_context_sha256 != context.context_sha256:
            error("CONTEXT_SHA_MISMATCH", "portfolio.source_context_sha256", "source context SHA mismatch")
        if portfolio.source_report_id != context.source_report_id:
            error("REPORT_ID_MISMATCH", "portfolio.source_report_id", "source report ID mismatch")
        if portfolio.source_report_sha256 != context.source_report_sha256:
            error("REPORT_SHA_MISMATCH", "portfolio.source_report_sha256", "source report SHA mismatch")

        statements = {x.statement_id: x for x in context.evidence_statements}
        hypothesis_ids = [x.hypothesis_id for x in portfolio.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            error("DUPLICATE_HYPOTHESIS_ID", "portfolio.hypotheses", "hypothesis IDs must be unique")

        for index, card in enumerate(portfolio.hypotheses):
            location = f"hypotheses[{index}]"
            premises = []
            gaps = []
            for statement_id in card.premise_statement_ids:
                statement = statements.get(statement_id)
                if statement is None:
                    error("UNKNOWN_PREMISE_STATEMENT", location + ".premise_statement_ids", statement_id)
                    continue
                premises.append(statement)
                if not statement.eligible_as_premise:
                    error(
                        "INELIGIBLE_POSITIVE_PREMISE",
                        location + ".premise_statement_ids",
                        f"{statement_id}: {statement.premise_restrictions}",
                    )
                if statement.alignment_path_ids:
                    error(
                        "ALIGNMENT_USED_AS_SCIENTIFIC_PREMISE",
                        location + ".premise_statement_ids",
                        f"{statement_id} depends on alignment path(s)",
                    )
            for statement_id in card.gap_statement_ids:
                statement = statements.get(statement_id)
                if statement is None:
                    error("UNKNOWN_GAP_STATEMENT", location + ".gap_statement_ids", statement_id)
                    continue
                gaps.append(statement)
                if not statement.eligible_as_gap:
                    error("INELIGIBLE_GAP_STATEMENT", location + ".gap_statement_ids", statement_id)

            expected_papers = sorted({p for x in premises for p in x.paper_ids})
            if card.source_paper_ids != expected_papers:
                error(
                    "SOURCE_PAPER_SCOPE_MISMATCH",
                    location + ".source_paper_ids",
                    f"expected={expected_papers}, actual={card.source_paper_ids}",
                )
            expected_gap_papers = sorted({p for x in gaps for p in x.paper_ids})
            if card.gap_paper_ids != expected_gap_papers:
                error(
                    "GAP_PAPER_SCOPE_MISMATCH",
                    location + ".gap_paper_ids",
                    f"expected={expected_gap_papers}, actual={card.gap_paper_ids}",
                )
            if card.cross_paper_synthesis != (len(expected_papers) >= 2):
                error(
                    "CROSS_PAPER_FLAG_MISMATCH",
                    location + ".cross_paper_synthesis",
                    "cross-paper flag does not match premise paper scope",
                )

            candidate_count = sum(bool(x.requires_verification) for x in premises)
            if candidate_count == 0:
                expected_candidate_dependency = "none"
            elif candidate_count == len(premises):
                expected_candidate_dependency = "essential"
            else:
                expected_candidate_dependency = "supporting"
            if card.candidate_dependency != expected_candidate_dependency:
                error(
                    "CANDIDATE_DEPENDENCY_MISMATCH",
                    location + ".candidate_dependency",
                    f"expected={expected_candidate_dependency}, actual={card.candidate_dependency}",
                )

            profile = card.evidence_profile
            expected_profile = {
                "premise_count": len(premises),
                "gap_count": len(gaps),
                "source_paper_count": len(expected_papers),
                "candidate_premise_count": candidate_count,
                "reported_premise_count": sum(x.epistemic_role == "reported" for x in premises),
                "synthesis_premise_count": sum(x.epistemic_role == "evidence_synthesis" for x in premises),
            }
            for name, expected in expected_profile.items():
                if getattr(profile, name) != expected:
                    error(
                        "EVIDENCE_PROFILE_MISMATCH",
                        location + f".evidence_profile.{name}",
                        f"expected={expected}, actual={getattr(profile, name)}",
                    )

            if not card.predicted_observations:
                error("MISSING_PREDICTION", location + ".predicted_observations", "at least one prediction is required")
            if not card.falsification_criteria:
                error("MISSING_FALSIFIER", location + ".falsification_criteria", "at least one falsifier is required")

            for f_index, falsifier in enumerate(card.falsification_criteria):
                if not any(
                    _observable_matches(falsifier.observable, prediction.observable)
                    for prediction in card.predicted_observations
                ):
                    error(
                        "FALSIFIER_OBSERVABLE_NOT_PREDICTED",
                        location + f".falsification_criteria[{f_index}].observable",
                        f"No predicted observable matches {falsifier.observable!r}",
                    )

            if card.predicted_observations and all(
                x.expected_direction == "unspecified" for x in card.predicted_observations
            ):
                warning(
                    "ALL_PREDICTION_DIRECTIONS_UNSPECIFIED",
                    location + ".predicted_observations",
                    "all predicted observations use expected_direction='unspecified'",
                )

            generated_text = _card_text(card)
            if any(pattern.search(generated_text) for pattern in _NOVELTY_PATTERNS):
                error(
                    "EXTERNAL_NOVELTY_CLAIM",
                    location,
                    "literature-wide novelty has not been assessed",
                )
            if any(pattern.search(generated_text) for pattern in _PROTOCOL_PATTERNS):
                error(
                    "EXPERIMENT_PROTOCOL_LEAKAGE",
                    location,
                    "Hypothesis Maker may state observables/falsifiers but not an experimental protocol",
                )

            blocked_scope = set(card.source_paper_ids) | set(card.gap_paper_ids)
            blocked_scope &= set(context.partial_absence_blocked_paper_ids)
            if blocked_scope and _ABSENCE_RE.search(generated_text) and _PAPER_SCOPE_RE.search(generated_text):
                error(
                    "PARTIAL_PAPER_ABSENCE_CLAIM",
                    location,
                    f"paper-level absence claim is unsafe for partial paper(s): {sorted(blocked_scope)}",
                )

            allowed_numbers: set[str] = set()
            for statement in premises:
                allowed_numbers.update(_numbers(statement.text))
            generated_numbers = _numbers(generated_text)
            unsupported_numbers = sorted(generated_numbers - allowed_numbers)
            if unsupported_numbers:
                error(
                    "UNSUPPORTED_NUMERIC_PREDICTION",
                    location,
                    f"generated numeric values not present in positive premises: {unsupported_numbers}",
                )

        errors = sum(x.severity == "error" for x in issues)
        warnings = sum(x.severity == "warning" for x in issues)
        return HypothesisValidationResult(
            passes=errors == 0,
            errors=errors,
            warnings=warnings,
            issues=issues,
        )
