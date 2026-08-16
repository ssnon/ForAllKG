from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from dac_her.hypothesis_trend_compiler import (
    POSITIVE_USES,
    TrendHypothesisCompileError,
    TrendHypothesisCompileIssue,
)
from dac_her.hypothesis_trend_contract_hardened_contracts import (
    ContractHardenedTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_contract_hardened_renderer import (
    HYPOTHESIS_TREND_HARDENED_RENDERER_SEMANTICS_ID,
    render_context_qualification,
    render_hypothesis_statement,
    render_inferential_bridge,
    render_prediction_rationale,
    render_view_clause,
)
from dac_her.hypothesis_trend_directional_compiler import (
    DirectionAwareTrendHypothesisCompiler,
)
from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwarePredictedObservationDraft,
    DirectionAwareTrendHypothesisPortfolio,
    DirectionAwareTrendHypothesisPortfolioDraft,
    DirectionAwareTrendHypothesisProposalDraft,
    TrendDirectionBindingDraft,
    canonical_dependent_change,
    expected_prediction_direction,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareFalsificationCriterionDraft,
)
from dac_her.hypothesis_trend_input import (
    HypothesisTrendInputView,
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)


HYPOTHESIS_TREND_HARDENED_COMPILER_SEMANTICS_ID = (
    "hypothesis_trend_contract_hardened_compiler_v1_alpha4c5i"
)

_PAPER_SCOPE_RE = re.compile(
    r"\b(?:paper|study|article|source|publication|literature)\b",
    re.I,
)
_ABSENCE_RE = re.compile(
    r"\b(?:absent|absence|not\s+reported|does\s+not\s+report|"
    r"did\s+not\s+report|no\s+evidence\s+of|no\s+support\s+for|"
    r"lacks?|without|never\s+reported|unreported)\b",
    re.I,
)
_DIRECTION_MARKER_RE = re.compile(
    r"\b(?:increas\w*|decreas\w*|reduc\w*|smaller|larger|"
    r"higher|lower|less|more|shrink\w*|grow\w*)\b",
    re.I,
)


def _key_phrase(key: str) -> str:
    return " ".join(str(key).replace("_", " ").split()).lower()


def _uses_directional_frame(
    text: str,
    independent_variable_key: str,
) -> bool:
    normalized = " ".join(str(text).lower().split())
    phrase = _key_phrase(independent_variable_key)
    if not phrase or phrase not in normalized:
        return False
    start = 0
    while True:
        index = normalized.find(phrase, start)
        if index < 0:
            return False
        left = max(0, index - 64)
        right = min(
            len(normalized),
            index + len(phrase) + 64,
        )
        if _DIRECTION_MARKER_RE.search(normalized[left:right]):
            return True
        start = index + len(phrase)


def _freeform_texts(hypothesis: object) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("title", str(getattr(hypothesis, "title", ""))),
        (
            "mechanistic_proposal",
            str(getattr(hypothesis, "mechanistic_proposal", "")),
        ),
        (
            "inferential_bridge",
            str(getattr(hypothesis, "inferential_bridge", "")),
        ),
    ]
    for index, value in enumerate(
        getattr(hypothesis, "assumptions", [])
    ):
        rows.append((f"assumptions[{index}]", str(value)))
    for index, row in enumerate(
        getattr(hypothesis, "predicted_observations", [])
    ):
        rows.append(
            (
                f"predicted_observations[{index}].mechanistic_rationale",
                str(row.mechanistic_rationale),
            )
        )
        if row.prediction_kind == "exploratory":
            rows.append(
                (
                    f"predicted_observations[{index}].exploratory_observable",
                    str(row.exploratory_observable or ""),
                )
            )
    for index, row in enumerate(
        getattr(hypothesis, "falsification_criteria", [])
    ):
        rows.append(
            (
                f"falsification_criteria[{index}].falsifying_outcome",
                str(row.falsifying_outcome),
            )
        )
    return rows


class ContractHardenedTrendHypothesisCompiler:
    semantics_id = HYPOTHESIS_TREND_HARDENED_COMPILER_SEMANTICS_ID
    renderer_semantics_id = (
        HYPOTHESIS_TREND_HARDENED_RENDERER_SEMANTICS_ID
    )

    def __init__(
        self,
        *,
        directional_compiler:
            DirectionAwareTrendHypothesisCompiler | None = None,
    ) -> None:
        self.directional_compiler = (
            directional_compiler
            or DirectionAwareTrendHypothesisCompiler()
        )

    def _authority_issues(
        self,
        source: TrendAwareHypothesisInput,
        draft: ContractHardenedTrendHypothesisPortfolioDraft,
    ) -> list[TrendHypothesisCompileIssue]:
        views = {row.view_id: row for row in source.trend_views}
        source_paper_ids = {
            paper_id
            for row in source.grounded_context.evidence_statements
            for paper_id in row.paper_ids
        }
        issues: list[TrendHypothesisCompileIssue] = []

        for h_index, hypothesis in enumerate(draft.hypotheses):
            base = f"draft.hypotheses[{h_index}]"
            refs = {row.view_id: row for row in hypothesis.trend_references}
            positive_ids = {
                row.view_id
                for row in hypothesis.trend_references
                if row.use_role in POSITIVE_USES
            }
            bound_ids: set[str] = set()

            for field, text in _freeform_texts(hypothesis):
                paper_scoped = bool(_PAPER_SCOPE_RE.search(text))
                if not paper_scoped:
                    lowered = text.lower()
                    paper_scoped = any(
                        str(paper_id).lower() in lowered
                        for paper_id in source_paper_ids
                    )
                if paper_scoped and _ABSENCE_RE.search(text):
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code=(
                                "LLM_PAPER_ABSENCE_AUTHORITY_VIOLATION"
                            ),
                            location=base + "." + field,
                            message=(
                                "alpha4c.5i does not allow LLM-authored "
                                "paper/study/literature-level absence "
                                "claims. Select grounded gap IDs instead."
                            ),
                        )
                    )

            for p_index, prediction in enumerate(
                hypothesis.predicted_observations
            ):
                ploc = (
                    base
                    + f".predicted_observations[{p_index}]"
                )
                if prediction.prediction_kind != "trend_bound":
                    continue

                selected_views: list[HypothesisTrendInputView] = []
                for view_id in prediction.trend_view_ids:
                    view = views.get(view_id)
                    if view is None:
                        issues.append(
                            TrendHypothesisCompileIssue(
                                code="UNKNOWN_HARDENED_TREND_VIEW",
                                location=ploc + ".trend_view_ids",
                                message=f"Unknown Trend view: {view_id}",
                            )
                        )
                        continue
                    if view_id not in positive_ids:
                        issues.append(
                            TrendHypothesisCompileIssue(
                                code=(
                                    "HARDENED_TREND_BINDING_NOT_"
                                    "POSITIVE_SUPPORT"
                                ),
                                location=ploc + ".trend_view_ids",
                                message=(
                                    "Trend-bound predictions may reference "
                                    "only positive Trend support selected "
                                    "in the same hypothesis."
                                ),
                            )
                        )
                        continue
                    selected_views.append(view)
                    bound_ids.add(view_id)

                if selected_views:
                    observables = {
                        row.dependent_observable_key
                        for row in selected_views
                    }
                    changes = {
                        canonical_dependent_change(row.directions)
                        for row in selected_views
                    }
                    if len(observables) != 1 or len(changes) != 1:
                        issues.append(
                            TrendHypothesisCompileIssue(
                                code=(
                                    "INCOMPATIBLE_HARDENED_TREND_"
                                    "BINDING_GROUP"
                                ),
                                location=ploc + ".trend_view_ids",
                                message=(
                                    "A single Trend-bound prediction may "
                                    "combine only views with the same "
                                    "dependent observable and same canonical "
                                    "dependent change. No averaging or "
                                    "majority direction is allowed."
                                ),
                            )
                        )

            for view_id in sorted(positive_ids - bound_ids):
                issues.append(
                    TrendHypothesisCompileIssue(
                        code="MISSING_HARDENED_TREND_BINDING",
                        location=base + ".predicted_observations",
                        message=(
                            "Every selected positive Trend support view "
                            "must be bound to a trend_bound prediction. "
                            f"Missing view: {view_id}"
                        ),
                    )
                )

            positive_iv_keys = sorted(
                {
                    views[view_id].independent_variable_key
                    for view_id in positive_ids
                    if view_id in views
                }
            )
            for field, text in _freeform_texts(hypothesis):
                for key in positive_iv_keys:
                    if _uses_directional_frame(text, key):
                        issues.append(
                            TrendHypothesisCompileIssue(
                                code=(
                                    "LLM_DIRECTIONAL_AUTHORITY_VIOLATION"
                                ),
                                location=base + "." + field,
                                message=(
                                    "alpha4c.5i reserves Trend-bound "
                                    "directional wording for the deterministic "
                                    "renderer. The LLM-owned text used a "
                                    f"direction marker near {key!r}."
                                ),
                            )
                        )
                        break

            # Unknown reference/use-role and companion checks remain delegated
            # to the frozen 5d.1 compiler/validator after deterministic
            # translation.
            _ = refs

        return issues

    def _translate(
        self,
        source: TrendAwareHypothesisInput,
        draft: ContractHardenedTrendHypothesisPortfolioDraft,
    ) -> DirectionAwareTrendHypothesisPortfolioDraft:
        views = {row.view_id: row for row in source.trend_views}
        hypotheses: list[
            DirectionAwareTrendHypothesisProposalDraft
        ] = []

        for hypothesis in draft.hypotheses:
            predictions: list[
                DirectionAwarePredictedObservationDraft
            ] = []
            prediction_observables: dict[str, str] = {}
            bound_views_for_statement: dict[
                str, HypothesisTrendInputView
            ] = {}

            for prediction in hypothesis.predicted_observations:
                if prediction.prediction_kind == "exploratory":
                    observable = str(
                        prediction.exploratory_observable
                    )
                    expected_direction = (
                        prediction.exploratory_expected_direction
                    )
                    assert expected_direction is not None
                    predictions.append(
                        DirectionAwarePredictedObservationDraft(
                            local_id=prediction.local_id,
                            observable=observable,
                            expected_direction=expected_direction,
                            rationale=prediction.mechanistic_rationale,
                            trend_direction_bindings=[],
                        )
                    )
                    prediction_observables[
                        prediction.local_id
                    ] = observable
                    continue

                selected = [
                    views[view_id]
                    for view_id in prediction.trend_view_ids
                ]
                changes = [
                    canonical_dependent_change(row.directions)
                    for row in selected
                ]
                dependent_change = changes[0]
                observable = selected[0].dependent_observable_key
                clauses = [
                    render_view_clause(
                        row,
                        dependent_change=canonical_dependent_change(
                            row.directions
                        ),
                    )
                    for row in selected
                ]
                bindings = [
                    TrendDirectionBindingDraft(
                        view_id=row.view_id,
                        independent_change="increase",
                        dependent_change=canonical_dependent_change(
                            row.directions
                        ),
                    )
                    for row in selected
                ]
                predictions.append(
                    DirectionAwarePredictedObservationDraft(
                        local_id=prediction.local_id,
                        observable=observable,
                        expected_direction=(
                            expected_prediction_direction(changes)
                        ),
                        rationale=render_prediction_rationale(
                            canonical_clauses=clauses,
                            mechanistic_rationale=(
                                prediction.mechanistic_rationale
                            ),
                        ),
                        trend_direction_bindings=bindings,
                    )
                )
                prediction_observables[prediction.local_id] = observable
                for row in selected:
                    bound_views_for_statement[row.view_id] = row

            statement_clauses = [
                render_view_clause(
                    row,
                    dependent_change=canonical_dependent_change(
                        row.directions
                    ),
                )
                for _, row in sorted(
                    bound_views_for_statement.items()
                )
            ]
            qualifications = [
                render_context_qualification(row)
                for _, row in sorted(
                    bound_views_for_statement.items()
                )
            ]
            falsifiers = [
                TrendAwareFalsificationCriterionDraft(
                    local_id=row.local_id,
                    observable=prediction_observables[
                        row.prediction_local_id
                    ],
                    falsifying_outcome=row.falsifying_outcome,
                )
                for row in hypothesis.falsification_criteria
            ]

            hypotheses.append(
                DirectionAwareTrendHypothesisProposalDraft(
                    local_id=hypothesis.local_id,
                    title=hypothesis.title,
                    hypothesis_statement=render_hypothesis_statement(
                        hypothesis.mechanistic_proposal,
                        statement_clauses,
                    ),
                    hypothesis_type=hypothesis.hypothesis_type,
                    premise_statement_ids=(
                        hypothesis.premise_statement_ids
                    ),
                    gap_statement_ids=hypothesis.gap_statement_ids,
                    trend_references=hypothesis.trend_references,
                    inferential_bridge=render_inferential_bridge(
                        hypothesis.inferential_bridge,
                        qualifications,
                    ),
                    predicted_observations=predictions,
                    falsification_criteria=falsifiers,
                    assumptions=hypothesis.assumptions,
                )
            )

        return DirectionAwareTrendHypothesisPortfolioDraft(
            hypotheses=hypotheses,
            abstention_reason=draft.abstention_reason,
        )

    def compile(
        self,
        source: TrendAwareHypothesisInput,
        draft: ContractHardenedTrendHypothesisPortfolioDraft,
    ) -> DirectionAwareTrendHypothesisPortfolio:
        verify_trend_aware_input_sources(source)
        issues = self._authority_issues(source, draft)
        if issues:
            raise TrendHypothesisCompileError(issues)

        translated = self._translate(source, draft)
        return self.directional_compiler.compile(
            source,
            translated,
        )
