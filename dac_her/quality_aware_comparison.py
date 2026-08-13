from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Iterable

from dac_her.comparison_domain import (
    ComparisonAssessment,
    ComparisonContext,
    MetricDefinitionAssessment,
)
from dac_her.metric_definition_domain import (
    MetricDefinitionContext,
    MetricDefinitionDomainAdapter,
)


QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID = (
    "quality_aware_numeric_gate_v2_alpha4b3b4c1"
)


def _stable_id(*parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _metric_context_map(
    contexts: Iterable[MetricDefinitionContext],
) -> dict[tuple[str, str], MetricDefinitionContext]:
    rows: dict[tuple[str, str], MetricDefinitionContext] = {}
    for context in contexts:
        key = (context.paper_id, context.measurement_id)
        if key in rows:
            raise ValueError(
                "MetricDefinitionContext is not unique by "
                f"(paper, measurement): {key!r}."
            )
        rows[key] = context
    return rows


def _definition_signature(
    context: MetricDefinitionContext,
) -> tuple[str, str, str, str]:
    return (
        context.definition_family,
        context.normalization_basis,
        context.reference_basis,
        context.criterion,
    )


def _definition_mismatch_reasons(
    left: MetricDefinitionContext,
    right: MetricDefinitionContext,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if left.definition_family != right.definition_family:
        reasons.append("metric_definition_family_mismatch")
    if left.normalization_basis != right.normalization_basis:
        reasons.append("metric_normalization_basis_mismatch")
    if left.reference_basis != right.reference_basis:
        reasons.append("metric_reference_basis_mismatch")
    if left.criterion != right.criterion:
        reasons.append("metric_criterion_mismatch")
    return tuple(reasons)


def compare_metric_definitions(
    *,
    comparison_assessment: ComparisonAssessment,
    left_context: ComparisonContext,
    right_context: ComparisonContext,
    left_metric: MetricDefinitionContext | None,
    right_metric: MetricDefinitionContext | None,
    adapter: MetricDefinitionDomainAdapter,
) -> MetricDefinitionAssessment:
    observable = comparison_assessment.observable_key
    if observable not in adapter.supported_observable_keys:
        return MetricDefinitionAssessment(
            assessment_id=(
                "metricgate:"
                + _stable_id(
                    QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID,
                    comparison_assessment.assessment_id,
                    "not_applicable",
                )
            ),
            quality_gate_semantics_id=(
                QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID
            ),
            metric_definition_semantics_id=adapter.semantics_id,
            comparison_assessment_id=comparison_assessment.assessment_id,
            observable_key=observable,
            left_context_id=comparison_assessment.left_context_id,
            right_context_id=comparison_assessment.right_context_id,
            left_metric_definition_context_id="",
            right_metric_definition_context_id="",
            compatibility="not_applicable",
            left_definition_status="not_applicable",
            right_definition_status="not_applicable",
            left_definition_family="",
            right_definition_family="",
            left_aggregation_scope="not_applicable",
            right_aggregation_scope="not_applicable",
            numeric_metric_definition_gate=True,
            reasons=("metric_definition_not_required_for_observable",),
        )

    if left_metric is None or right_metric is None:
        missing = []
        if left_metric is None:
            missing.append(
                f"{left_context.paper_id}:{left_context.measurement_id}"
            )
        if right_metric is None:
            missing.append(
                f"{right_context.paper_id}:{right_context.measurement_id}"
            )
        raise ValueError(
            "Missing MetricDefinitionContext for registered observable "
            f"{observable!r}: {', '.join(missing)}."
        )

    for metric, context, side in (
        (left_metric, left_context, "left"),
        (right_metric, right_context, "right"),
    ):
        if metric.observable_key != observable:
            raise ValueError(
                f"{side} metric-definition observable mismatch: "
                f"{metric.observable_key!r} != {observable!r}."
            )
        if metric.measurement_id != context.measurement_id:
            raise ValueError(
                f"{side} metric-definition measurement mismatch: "
                f"{metric.measurement_id!r} != "
                f"{context.measurement_id!r}."
            )
        if metric.paper_id != context.paper_id:
            raise ValueError(
                f"{side} metric-definition paper mismatch: "
                f"{metric.paper_id!r} != {context.paper_id!r}."
            )
        if metric.metric_definition_semantics_id != adapter.semantics_id:
            raise ValueError(
                f"{side} metric-definition semantics mismatch: "
                f"{metric.metric_definition_semantics_id!r} != "
                f"{adapter.semantics_id!r}."
            )

    reasons: list[str] = []

    if left_metric.definition_status != "known":
        reasons.append(
            "left_metric_definition_"
            f"{left_metric.definition_status}"
        )
    if right_metric.definition_status != "known":
        reasons.append(
            "right_metric_definition_"
            f"{right_metric.definition_status}"
        )

    if reasons:
        compatibility = "unknown"
        gate = False
    else:
        signature_mismatch_reasons = _definition_mismatch_reasons(
            left_metric,
            right_metric,
        )
        if signature_mismatch_reasons:
            # alpha4b.3b.4c.1 precedence rule:
            # a known definition contradiction remains a contradiction even
            # when a secondary aggregation field is unknown.
            compatibility = "different_definition"
            gate = False
            reasons.extend(signature_mismatch_reasons)
            if (
                left_metric.aggregation_scope == "unspecified"
                or right_metric.aggregation_scope == "unspecified"
            ):
                reasons.append("metric_aggregation_scope_unknown")
            elif (
                left_metric.aggregation_scope
                != right_metric.aggregation_scope
            ):
                reasons.append("metric_aggregation_scope_mismatch")
        elif (
            left_metric.aggregation_scope == "unspecified"
            or right_metric.aggregation_scope == "unspecified"
        ):
            compatibility = "unknown"
            gate = False
            reasons.append("metric_aggregation_scope_unknown")
        elif (
            left_metric.aggregation_scope
            != right_metric.aggregation_scope
        ):
            compatibility = "different_definition"
            gate = False
            reasons.append("metric_aggregation_scope_mismatch")
        else:
            compatibility = "same_definition"
            gate = True

    return MetricDefinitionAssessment(
        assessment_id=(
            "metricgate:"
            + _stable_id(
                QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID,
                adapter.semantics_id,
                comparison_assessment.assessment_id,
                left_metric.context_id,
                right_metric.context_id,
            )
        ),
        quality_gate_semantics_id=QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID,
        metric_definition_semantics_id=adapter.semantics_id,
        comparison_assessment_id=comparison_assessment.assessment_id,
        observable_key=observable,
        left_context_id=comparison_assessment.left_context_id,
        right_context_id=comparison_assessment.right_context_id,
        left_metric_definition_context_id=left_metric.context_id,
        right_metric_definition_context_id=right_metric.context_id,
        compatibility=compatibility,
        left_definition_status=left_metric.definition_status,
        right_definition_status=right_metric.definition_status,
        left_definition_family=left_metric.definition_family,
        right_definition_family=right_metric.definition_family,
        left_aggregation_scope=left_metric.aggregation_scope,
        right_aggregation_scope=right_metric.aggregation_scope,
        numeric_metric_definition_gate=gate,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def build_metric_definition_assessments(
    *,
    comparison_assessments: Iterable[ComparisonAssessment],
    comparison_contexts: Iterable[ComparisonContext],
    metric_definition_contexts: Iterable[MetricDefinitionContext],
    adapter: MetricDefinitionDomainAdapter,
) -> list[MetricDefinitionAssessment]:
    contexts = {
        item.context_id: item
        for item in comparison_contexts
    }
    metrics = _metric_context_map(metric_definition_contexts)

    rows: list[MetricDefinitionAssessment] = []
    for assessment in comparison_assessments:
        try:
            left = contexts[assessment.left_context_id]
            right = contexts[assessment.right_context_id]
        except KeyError as exc:
            raise ValueError(
                "Comparison assessment refers to a missing "
                "ComparisonContext."
            ) from exc

        if assessment.observable_key in adapter.supported_observable_keys:
            left_metric = metrics.get(
                (left.paper_id, left.measurement_id)
            )
            right_metric = metrics.get(
                (right.paper_id, right.measurement_id)
            )
        else:
            left_metric = None
            right_metric = None

        rows.append(
            compare_metric_definitions(
                comparison_assessment=assessment,
                left_context=left,
                right_context=right,
                left_metric=left_metric,
                right_metric=right_metric,
                adapter=adapter,
            )
        )

    return sorted(
        rows,
        key=lambda item: (
            item.observable_key,
            item.left_context_id,
            item.right_context_id,
            item.assessment_id,
        ),
    )


def apply_metric_definition_numeric_gate(
    assessments: Iterable[ComparisonAssessment],
    metric_definition_assessments: Iterable[
        MetricDefinitionAssessment
    ],
) -> list[ComparisonAssessment]:
    metric_by_comparison = {
        item.comparison_assessment_id: item
        for item in metric_definition_assessments
    }

    gated: list[ComparisonAssessment] = []
    for assessment in assessments:
        metric = metric_by_comparison.get(assessment.assessment_id)
        if metric is None:
            raise ValueError(
                "Missing metric-definition assessment for comparison "
                f"assessment {assessment.assessment_id!r}."
            )

        allowed = (
            assessment.numeric_ranking_allowed
            and metric.numeric_metric_definition_gate
        )
        reasons = list(assessment.reasons)
        if (
            assessment.numeric_ranking_allowed
            and not metric.numeric_metric_definition_gate
        ):
            reasons.append(
                "numeric_ranking_blocked_by_metric_definition"
            )

        gated.append(
            replace(
                assessment,
                numeric_ranking_allowed=allowed,
                reasons=tuple(dict.fromkeys(reasons)),
                metric_definition_assessment_id=metric.assessment_id,
                metric_definition_compatibility=metric.compatibility,
                metric_definition_gate=(
                    metric.numeric_metric_definition_gate
                ),
            )
        )
    return gated
