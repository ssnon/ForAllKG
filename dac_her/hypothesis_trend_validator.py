from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dac_her.hypothesis_trend_compiler import (
    CONTEXT_USES,
    GAP_USES,
    POSITIVE_USES,
    USE_TO_LANE,
    required_companion_uses,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareHypothesisPortfolio,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)


HYPOTHESIS_TREND_VALIDATOR_SEMANTICS_ID = (
    "hypothesis_trend_validator_v1_alpha4c5c"
)


class TrendHypothesisValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Literal["error", "warning"]
    code: str
    location: str
    message: str


class TrendHypothesisValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    semantics_id: str
    passes: bool
    errors: int
    warnings: int
    issues: list[TrendHypothesisValidationIssue]


_NOVELTY_PATTERNS = (
    re.compile(r"\bnovel\b", re.I),
    re.compile(r"\bunprecedented\b", re.I),
    re.compile(
        r"\bpreviously\s+(?:unknown|unreported|unrecognized)\b",
        re.I,
    ),
    re.compile(r"\bfor\s+the\s+first\s+time\b", re.I),
    re.compile(
        r"\bfirst\s+(?:report|demonstration|observation|evidence)\b",
        re.I,
    ),
    re.compile(r"\bnever\s+before\b", re.I),
)

_PROTOCOL_PATTERNS = (
    re.compile(
        r"\b(?:synthesi[sz]e|anneal|calcine|pyroly[sz]e)\b"
        r".{0,50}\b(?:at|for|under)\b",
        re.I,
    ),
    re.compile(r"\bscan\s+rate\b", re.I),
    re.compile(
        r"\b(?:rpm|revolutions?\s+per\s+minute)\b",
        re.I,
    ),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mM|M)\s+"
        r"(?:KOH|NaOH|H2SO4|HClO4|electrolyte)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:heat|hold|stir|sonicate)\b.{0,35}"
        r"\b\d+(?:\.\d+)?\s*(?:°?C|K|h|hr|hours?|min|minutes?)\b",
        re.I,
    ),
)

_ABSENCE_RE = re.compile(
    r"\b(?:absent|absence|not\s+reported|does\s+not\s+report|"
    r"did\s+not\s+report|no\s+evidence\s+of|no\s+support\s+for|"
    r"lacks?|without)\b",
    re.I,
)
_PAPER_SCOPE_RE = re.compile(
    r"\b(?:paper|study|article|source|Kiwook_\d+)\b",
    re.I,
)

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?"
    r"(?:\s*(?:eV|meV|V|mV|A|mA|K|°C|C|%|nm|Å|pm|cm|mm|s|"
    r"ms|h|mol|M|pH))?",
    re.I,
)


def _norm(text: str) -> str:
    value = unicodedata.normalize(
        "NFKC",
        str(text),
    ).lower()
    value = re.sub(
        r"[\u2010-\u2015\u2212-]+",
        " ",
        value,
    )
    value = re.sub(
        r"[^a-z0-9α-ω가-힣]+",
        " ",
        value,
    )
    return re.sub(r"\s+", " ", value).strip()


def _numbers(text: str) -> set[str]:
    return {
        " ".join(match.group(0).split()).lower()
        for match in _NUMBER_RE.finditer(text)
    }


def _observable_matches(a: str, b: str) -> bool:
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return False
    return (
        na == nb
        or (len(na) >= 5 and na in nb)
        or (len(nb) >= 5 and nb in na)
    )


def _card_text(card: object) -> str:
    parts: list[str] = [
        str(getattr(card, "title", "")),
        str(
            getattr(
                card,
                "hypothesis_statement",
                "",
            )
        ),
        str(getattr(card, "inferential_bridge", "")),
    ]
    for row in getattr(
        card,
        "predicted_observations",
        [],
    ):
        parts.extend(
            [str(row.observable), str(row.rationale)]
        )
    for row in getattr(
        card,
        "falsification_criteria",
        [],
    ):
        parts.extend(
            [
                str(row.observable),
                str(row.falsifying_outcome),
            ]
        )
    parts.extend(
        map(str, getattr(card, "assumptions", []))
    )
    return "\n".join(parts)


class TrendAwareHypothesisValidator:
    semantics_id = HYPOTHESIS_TREND_VALIDATOR_SEMANTICS_ID

    def validate(
        self,
        source: TrendAwareHypothesisInput,
        portfolio: TrendAwareHypothesisPortfolio,
    ) -> TrendHypothesisValidationResult:
        verify_trend_aware_input_sources(source)
        context = source.grounded_context
        statements = {
            row.statement_id: row
            for row in context.evidence_statements
        }
        views = {
            row.view_id: row
            for row in source.trend_views
        }

        issues: list[
            TrendHypothesisValidationIssue
        ] = []

        def error(
            code: str,
            location: str,
            message: str,
        ) -> None:
            issues.append(
                TrendHypothesisValidationIssue(
                    severity="error",
                    code=code,
                    location=location,
                    message=message,
                )
            )

        def warning(
            code: str,
            location: str,
            message: str,
        ) -> None:
            issues.append(
                TrendHypothesisValidationIssue(
                    severity="warning",
                    code=code,
                    location=location,
                    message=message,
                )
            )

        lineage = {
            "source_context_id": context.context_id,
            "source_context_sha256":
                context.context_sha256,
            "source_report_id": context.source_report_id,
            "source_report_sha256":
                context.source_report_sha256,
            "source_trend_input_id": source.input_id,
            "source_trend_input_sha256":
                source.input_sha256,
        }
        for field, expected in lineage.items():
            if getattr(portfolio, field) != expected:
                error(
                    "PORTFOLIO_LINEAGE_MISMATCH",
                    f"portfolio.{field}",
                    (
                        f"expected={expected!r}, "
                        f"actual={getattr(portfolio, field)!r}"
                    ),
                )

        ids = [
            row.hypothesis_id
            for row in portfolio.hypotheses
        ]
        if len(ids) != len(set(ids)):
            error(
                "DUPLICATE_HYPOTHESIS_ID",
                "portfolio.hypotheses",
                "hypothesis IDs must be unique",
            )

        for h_index, card in enumerate(
            portfolio.hypotheses
        ):
            location = f"hypotheses[{h_index}]"

            for field, expected in lineage.items():
                if getattr(card, field) != expected:
                    error(
                        "CARD_LINEAGE_MISMATCH",
                        location + f".{field}",
                        (
                            f"expected={expected!r}, "
                            f"actual={getattr(card, field)!r}"
                        ),
                    )

            explorer_premises = []
            explorer_gaps = []
            for statement_id in card.premise_statement_ids:
                row = statements.get(statement_id)
                if row is None:
                    error(
                        "UNKNOWN_PREMISE_STATEMENT",
                        location
                        + ".premise_statement_ids",
                        statement_id,
                    )
                    continue
                explorer_premises.append(row)
                if not row.eligible_as_premise:
                    error(
                        "INELIGIBLE_POSITIVE_PREMISE",
                        location
                        + ".premise_statement_ids",
                        statement_id,
                    )
                if row.alignment_path_ids:
                    error(
                        "ALIGNMENT_USED_AS_SCIENTIFIC_PREMISE",
                        location
                        + ".premise_statement_ids",
                        statement_id,
                    )

            for statement_id in card.gap_statement_ids:
                row = statements.get(statement_id)
                if row is None:
                    error(
                        "UNKNOWN_GAP_STATEMENT",
                        location + ".gap_statement_ids",
                        statement_id,
                    )
                    continue
                explorer_gaps.append(row)
                if not row.eligible_as_gap:
                    error(
                        "INELIGIBLE_GAP_STATEMENT",
                        location + ".gap_statement_ids",
                        statement_id,
                    )

            resolved_refs = []
            uses_by_grounding = defaultdict(set)
            for r_index, ref in enumerate(
                card.trend_references
            ):
                rloc = (
                    location
                    + f".trend_references[{r_index}]"
                )
                view = views.get(ref.view_id)
                if view is None:
                    error(
                        "UNKNOWN_TREND_VIEW",
                        rloc + ".view_id",
                        ref.view_id,
                    )
                    continue

                expected_lane = USE_TO_LANE[
                    ref.use_role
                ]
                if ref.lane != expected_lane:
                    error(
                        "COMPILED_TREND_LANE_MISMATCH",
                        rloc + ".lane",
                        (
                            f"expected={expected_lane}, "
                            f"actual={ref.lane}"
                        ),
                    )
                if view.lane != ref.lane:
                    error(
                        "TREND_VIEW_LANE_DRIFT",
                        rloc + ".lane",
                        (
                            f"input={view.lane}, "
                            f"card={ref.lane}"
                        ),
                    )
                for field in (
                    "grounding_id",
                    "relation_id",
                    "cross_context_status",
                    "paper_ids",
                    "directions",
                    "shapes",
                    "requires_context_qualification",
                    "requires_verification",
                    "directional_cross_paper_premise_allowed",
                ):
                    if getattr(ref, field) != getattr(
                        view,
                        field,
                    ):
                        error(
                            "TREND_REFERENCE_PROVENANCE_MISMATCH",
                            rloc + f".{field}",
                            (
                                f"input={getattr(view, field)!r}, "
                                f"card={getattr(ref, field)!r}"
                            ),
                        )
                expected_association = bool(
                    view.association_only_result_ids
                )
                if ref.association_only != expected_association:
                    error(
                        "TREND_ASSOCIATION_FLAG_MISMATCH",
                        rloc + ".association_only",
                        (
                            f"expected={expected_association}, "
                            f"actual={ref.association_only}"
                        ),
                    )
                if (
                    ref.trend_causal_authorization is not False
                    or ref.trend_universal_authorization is not False
                ):
                    error(
                        "TREND_AUTHORIZATION_ESCALATION",
                        rloc,
                        (
                            "Trend reference cannot authorize causal "
                            "or universal evidence claims."
                        ),
                    )

                resolved_refs.append((ref, view))
                uses_by_grounding[
                    view.grounding_id
                ].add(ref.use_role)

            for ref, view in resolved_refs:
                if ref.use_role not in POSITIVE_USES:
                    continue
                required = required_companion_uses(
                    view
                )
                missing = (
                    required
                    - uses_by_grounding[
                        view.grounding_id
                    ]
                )
                for use in sorted(missing):
                    error(
                        "MISSING_TREND_LIMITATION_COMPANION",
                        location + ".trend_references",
                        (
                            f"{view.grounding_id} "
                            f"status={view.cross_context_status} "
                            f"requires companion {use}."
                        ),
                    )

            positive_refs = [
                ref
                for ref, _ in resolved_refs
                if ref.use_role in POSITIVE_USES
            ]
            gap_refs = [
                ref
                for ref, _ in resolved_refs
                if ref.use_role in GAP_USES
            ]
            context_refs = [
                ref
                for ref, _ in resolved_refs
                if ref.use_role in CONTEXT_USES
            ]

            if not explorer_premises and not positive_refs:
                error(
                    "MISSING_POSITIVE_SUPPORT",
                    location,
                    (
                        "Hypothesis requires at least one "
                        "Explorer or Trend positive support."
                    ),
                )

            explorer_support_papers = sorted({
                paper_id
                for row in explorer_premises
                for paper_id in row.paper_ids
            })
            trend_positive_papers = sorted({
                paper_id
                for ref in positive_refs
                for paper_id in ref.paper_ids
            })
            support_papers = sorted(
                set(explorer_support_papers)
                | set(trend_positive_papers)
            )
            explorer_gap_papers = sorted({
                paper_id
                for row in explorer_gaps
                for paper_id in row.paper_ids
            })
            trend_gap_papers = sorted({
                paper_id
                for ref in gap_refs
                for paper_id in ref.paper_ids
            })
            context_papers = sorted({
                paper_id
                for ref in context_refs
                for paper_id in ref.paper_ids
            })

            scope_checks = {
                "explorer_source_paper_ids":
                    explorer_support_papers,
                "trend_positive_source_paper_ids":
                    trend_positive_papers,
                "support_paper_ids": support_papers,
                "explorer_gap_paper_ids":
                    explorer_gap_papers,
                "trend_gap_paper_ids": trend_gap_papers,
                "context_and_counterevidence_paper_ids":
                    context_papers,
            }
            for field, expected in scope_checks.items():
                if getattr(card, field) != expected:
                    error(
                        "PAPER_SCOPE_MISMATCH",
                        location + f".{field}",
                        (
                            f"expected={expected}, "
                            f"actual={getattr(card, field)}"
                        ),
                    )

            if card.cross_paper_synthesis != (
                len(support_papers) >= 2
            ):
                error(
                    "CROSS_PAPER_FLAG_MISMATCH",
                    location + ".cross_paper_synthesis",
                    (
                        "cross-paper flag does not match "
                        "positive support paper scope"
                    ),
                )

            explorer_verification = sum(
                bool(row.requires_verification)
                for row in explorer_premises
            )
            trend_verification = sum(
                bool(ref.requires_verification)
                for ref in positive_refs
            )
            positive_source_count = (
                len(explorer_premises)
                + len(positive_refs)
            )
            verification_count = (
                explorer_verification
                + trend_verification
            )
            if verification_count == 0:
                expected_dependency = "none"
            elif verification_count == positive_source_count:
                expected_dependency = "essential"
            else:
                expected_dependency = "supporting"
            if (
                card.verification_dependency
                != expected_dependency
            ):
                error(
                    "VERIFICATION_DEPENDENCY_MISMATCH",
                    location
                    + ".verification_dependency",
                    (
                        f"expected={expected_dependency}, "
                        f"actual={card.verification_dependency}"
                    ),
                )

            profile = card.evidence_profile
            expected_profile = {
                "explorer_premise_count":
                    len(explorer_premises),
                "explorer_gap_count":
                    len(explorer_gaps),
                "trend_reference_count":
                    len(resolved_refs),
                "trend_positive_support_count":
                    len(positive_refs),
                "trend_cross_paper_support_count":
                    sum(
                        ref.use_role
                        == "cross_paper_empirical_support"
                        for ref, _ in resolved_refs
                    ),
                "trend_context_qualification_count":
                    sum(
                        ref.use_role
                        == "context_qualification"
                        for ref, _ in resolved_refs
                    ),
                "trend_counterevidence_count":
                    sum(
                        ref.use_role
                        == "counterevidence_boundary"
                        for ref, _ in resolved_refs
                    ),
                "trend_gap_count": len(gap_refs),
                "support_paper_count":
                    len(support_papers),
                "verification_required_support_count":
                    verification_count,
                "association_only_support_count":
                    sum(
                        ref.association_only
                        for ref in positive_refs
                    ),
                "reported_explorer_premise_count":
                    sum(
                        row.epistemic_role == "reported"
                        for row in explorer_premises
                    ),
                "synthesis_explorer_premise_count":
                    sum(
                        row.epistemic_role
                        == "evidence_synthesis"
                        for row in explorer_premises
                    ),
            }
            for field, expected in expected_profile.items():
                actual = getattr(profile, field)
                if actual != expected:
                    error(
                        "EVIDENCE_PROFILE_MISMATCH",
                        location
                        + f".evidence_profile.{field}",
                        (
                            f"expected={expected}, "
                            f"actual={actual}"
                        ),
                    )

            if (
                card.trend_causal_authorization is not False
                or card.trend_universal_authorization is not False
            ):
                error(
                    "CARD_TREND_AUTHORIZATION_ESCALATION",
                    location,
                    (
                        "Trend provenance cannot authorize causal "
                        "or universal evidence claims."
                    ),
                )

            if not card.predicted_observations:
                error(
                    "MISSING_PREDICTION",
                    location + ".predicted_observations",
                    "at least one prediction is required",
                )
            if not card.falsification_criteria:
                error(
                    "MISSING_FALSIFIER",
                    location + ".falsification_criteria",
                    "at least one falsifier is required",
                )

            for f_index, falsifier in enumerate(
                card.falsification_criteria
            ):
                if not any(
                    _observable_matches(
                        falsifier.observable,
                        prediction.observable,
                    )
                    for prediction
                    in card.predicted_observations
                ):
                    error(
                        "FALSIFIER_OBSERVABLE_NOT_PREDICTED",
                        (
                            location
                            + f".falsification_criteria[{f_index}]"
                            + ".observable"
                        ),
                        falsifier.observable,
                    )

            if (
                card.predicted_observations
                and all(
                    row.expected_direction == "unspecified"
                    for row
                    in card.predicted_observations
                )
            ):
                warning(
                    "ALL_PREDICTION_DIRECTIONS_UNSPECIFIED",
                    location + ".predicted_observations",
                    (
                        "all predicted observations use "
                        "expected_direction='unspecified'"
                    ),
                )

            generated_text = _card_text(card)
            if any(
                pattern.search(generated_text)
                for pattern in _NOVELTY_PATTERNS
            ):
                error(
                    "EXTERNAL_NOVELTY_CLAIM",
                    location,
                    (
                        "literature-wide novelty has not "
                        "been assessed"
                    ),
                )
            if any(
                pattern.search(generated_text)
                for pattern in _PROTOCOL_PATTERNS
            ):
                error(
                    "EXPERIMENT_PROTOCOL_LEAKAGE",
                    location,
                    (
                        "Hypothesis may state observables/"
                        "falsifiers but not a protocol"
                    ),
                )

            blocked_scope = (
                set(support_papers)
                | set(explorer_gap_papers)
                | set(trend_gap_papers)
                | set(context_papers)
            )
            blocked_scope &= set(
                context.partial_absence_blocked_paper_ids
            )
            if (
                blocked_scope
                and _ABSENCE_RE.search(generated_text)
                and _PAPER_SCOPE_RE.search(generated_text)
            ):
                error(
                    "PARTIAL_PAPER_ABSENCE_CLAIM",
                    location,
                    (
                        "paper-level absence claim is unsafe "
                        f"for partial paper(s): "
                        f"{sorted(blocked_scope)}"
                    ),
                )

            # 5c Trend input views intentionally carry no raw numeric
            # values.  Therefore numeric generation remains licensed only
            # by exact Explorer positive-premise text.
            allowed_numbers: set[str] = set()
            for statement in explorer_premises:
                allowed_numbers.update(
                    _numbers(statement.text)
                )
            unsupported = sorted(
                _numbers(generated_text)
                - allowed_numbers
            )
            if unsupported:
                error(
                    "UNSUPPORTED_NUMERIC_PREDICTION",
                    location,
                    (
                        "generated numeric values not present "
                        "in Explorer positive premises: "
                        f"{unsupported}"
                    ),
                )

        errors = sum(
            row.severity == "error" for row in issues
        )
        warnings = sum(
            row.severity == "warning"
            for row in issues
        )
        return TrendHypothesisValidationResult(
            semantics_id=self.semantics_id,
            passes=errors == 0,
            errors=errors,
            warnings=warnings,
            issues=issues,
        )
