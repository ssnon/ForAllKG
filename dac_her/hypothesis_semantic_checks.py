from __future__ import annotations

import re
import unicodedata
from itertools import combinations

from dac_her.hypothesis_benchmark_contracts import HypothesisBenchmarkIssue
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio


_ASSERTIVE_EVIDENCE_RE = re.compile(
    r"\b(?:proves?|establish(?:es|ed)?|confirms?|demonstrates?|"
    r"the\s+evidence\s+(?:shows|establishes|proves)|known\s+to)\b",
    re.I,
)
_ALIGNMENT_CAUSAL_RE = re.compile(
    r"\b(?:graph|alignment|registry|pattern)\b.{0,120}"
    r"\b(?:proves?|demonstrates?|establish(?:es|ed)?|causes?|drives?|mediates?)\b",
    re.I | re.S,
)
_CAUSAL_RE = re.compile(
    r"\b(?:causes?|drives?|determines?|directly\s+leads?\s+to|"
    r"is\s+responsible\s+for)\b",
    re.I,
)
_MODAL_RE = re.compile(
    r"\b(?:may|might|could|can|hypothes(?:is|ize|ized)|propos(?:e|ed)|if)\b",
    re.I,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9α-ωΑ-Ω]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "as", "is", "are", "be", "that", "this", "it", "from", "through",
    "at", "into", "when", "while", "across", "between", "than", "then",
}


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = re.sub(r"[\u2010-\u2015\u2212-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(_norm(text))
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _card_text(card: object) -> str:
    parts = [
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


def _statement_overlap(statement_text: str, generated_text: str) -> int:
    return len(_tokens(statement_text) & _tokens(generated_text))


def semantic_diagnostics(
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
) -> list[HypothesisBenchmarkIssue]:
    """Conservative, non-authoritative semantic diagnostics.

    These checks NEVER change hard-gate acceptance. They exist to surface
    review candidates for v2.6.2 and to seed the later semantic-critic layer.
    """
    issues: list[HypothesisBenchmarkIssue] = []
    statements = {row.statement_id: row for row in context.evidence_statements}
    has_alignment_route = any(route.uses_alignment for route in context.mechanism_routes)

    def warn(
        code: str,
        location: str,
        message: str,
        *,
        hypothesis_id: str | None = None,
        statement_ids: list[str] | None = None,
    ) -> None:
        issues.append(
            HypothesisBenchmarkIssue(
                severity="warning",
                layer="diagnostic",
                code=code,
                location=location,
                message=message,
                hypothesis_id=hypothesis_id,
                statement_ids=list(statement_ids or []),
                source="hypothesis_semantic_checks_v262",
            )
        )

    for index, card in enumerate(portfolio.hypotheses):
        location = f"hypotheses[{index}]"
        generated = _card_text(card)
        premises = [
            statements[sid]
            for sid in card.premise_statement_ids
            if sid in statements
        ]
        gaps = [
            statements[sid]
            for sid in card.gap_statement_ids
            if sid in statements
        ]

        if card.candidate_dependency != "none" and _ASSERTIVE_EVIDENCE_RE.search(generated):
            warn(
                "CANDIDATE_OVERCLAIM_LANGUAGE",
                location,
                "Candidate-dependent hypothesis uses language that can read as "
                "confirmed/established evidence; semantic review is required.",
                hypothesis_id=card.hypothesis_id,
                statement_ids=[
                    row.statement_id for row in premises if row.requires_verification
                ],
            )

        if has_alignment_route and _ALIGNMENT_CAUSAL_RE.search(generated):
            warn(
                "ALIGNMENT_CAUSALIZATION_LANGUAGE",
                location,
                "Generated text appears to use graph/alignment language as causal or "
                "mechanistic support. Alignment is navigation context only.",
                hypothesis_id=card.hypothesis_id,
            )

        if gaps and _ASSERTIVE_EVIDENCE_RE.search(generated):
            overlapping_gap_ids = [
                row.statement_id
                for row in gaps
                if _statement_overlap(row.text, generated) >= 2
            ]
            if overlapping_gap_ids and not _MODAL_RE.search(
                str(card.hypothesis_statement) + " " + str(card.inferential_bridge)
            ):
                warn(
                    "GAP_PROMOTION_LANGUAGE",
                    location,
                    "A research-gap concept is expressed with assertive evidence "
                    "language and without clear hypothetical/modal framing.",
                    hypothesis_id=card.hypothesis_id,
                    statement_ids=overlapping_gap_ids,
                )

        association_premises = [
            row.statement_id
            for row in premises
            if str(row.claim_kind).lower() == "association"
        ]
        if (
            association_premises
            and _CAUSAL_RE.search(
                str(card.hypothesis_statement) + " " + str(card.inferential_bridge)
            )
            and not _MODAL_RE.search(
                str(card.hypothesis_statement) + " " + str(card.inferential_bridge)
            )
        ):
            warn(
                "POSSIBLE_CAUSAL_STRENGTHENING",
                location,
                "Association premise is strengthened into causal language without an "
                "explicit hypothetical/modal qualifier.",
                hypothesis_id=card.hypothesis_id,
                statement_ids=association_premises,
            )

        premise_text = "\n".join(row.text for row in premises).lower()
        for p_index, prediction in enumerate(card.predicted_observations):
            if prediction.expected_direction != "non_monotonic":
                continue
            observable = prediction.observable.lower()
            explicit_nonmono = bool(
                re.search(r"\bnon[- ]?monotonic\b", premise_text, re.I)
            )
            volcano_activity = (
                "volcano" in premise_text
                and any(
                    key in observable
                    for key in ("activity", "exchange current", "current density")
                )
            )
            if not explicit_nonmono and not volcano_activity:
                warn(
                    "UNJUSTIFIED_NON_MONOTONIC_SPECIFICITY",
                    location + f".predicted_observations[{p_index}]",
                    "Prediction specifies a non-monotonic direction, but the selected "
                    "premise text does not explicitly support non-monotonicity for "
                    "this observable. Treat as a semantic-review warning, not a hard failure.",
                    hypothesis_id=card.hypothesis_id,
                    statement_ids=list(card.premise_statement_ids),
                )

        if (
            card.cross_paper_synthesis
            and re.search(r"\b(?:this|the|same)\s+catalyst\b", generated, re.I)
        ):
            warn(
                "CROSS_PAPER_ENTITY_CONFLATION_RISK",
                location,
                "Cross-paper synthesis refers to a singular catalyst; verify that "
                "distinct source systems were not silently conflated.",
                hypothesis_id=card.hypothesis_id,
            )

    for left, right in combinations(portfolio.hypotheses, 2):
        lt = _tokens(left.hypothesis_statement + " " + left.inferential_bridge)
        rt = _tokens(right.hypothesis_statement + " " + right.inferential_bridge)
        if not lt or not rt:
            continue
        similarity = len(lt & rt) / len(lt | rt)
        if similarity >= 0.78:
            warn(
                "HYPOTHESIS_REDUNDANCY",
                "portfolio.hypotheses",
                f"Hypotheses {left.hypothesis_id} and {right.hypothesis_id} have "
                f"high lexical-semantic overlap (Jaccard={similarity:.3f}); "
                "review whether they represent distinct scientific hypotheses.",
                statement_ids=sorted(
                    set(left.premise_statement_ids) | set(right.premise_statement_ids)
                ),
            )

    return issues
