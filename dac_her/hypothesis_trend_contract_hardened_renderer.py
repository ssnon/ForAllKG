from __future__ import annotations

from typing import Iterable

from dac_her.hypothesis_trend_input import HypothesisTrendInputView


HYPOTHESIS_TREND_HARDENED_RENDERER_SEMANTICS_ID = (
    "hypothesis_trend_contract_hardened_renderer_v1_alpha4c5i"
)


def _label(key: str) -> str:
    return " ".join(str(key).replace("_", " ").split())


def render_canonical_trend_clause(
    *,
    independent_variable_key: str,
    dependent_observable_key: str,
    dependent_change: str,
    shapes: Iterable[str] = (),
) -> str:
    x = _label(independent_variable_key)
    y = _label(dependent_observable_key)
    shape_values = sorted(
        {str(value).strip() for value in shapes if str(value).strip()}
    )

    if dependent_change == "increase":
        return (
            f"As {x} increases under the canonical independent-variable "
            "frame defined by the frozen Trend relation, "
            f"{y} is predicted to increase."
        )
    if dependent_change == "decrease":
        return (
            f"As {x} increases under the canonical independent-variable "
            "frame defined by the frozen Trend relation, "
            f"{y} is predicted to decrease."
        )
    if dependent_change == "unchanged":
        return (
            f"As {x} increases under the canonical independent-variable "
            "frame defined by the frozen Trend relation, "
            f"{y} is predicted to remain unchanged."
        )
    if dependent_change == "non_monotonic":
        if shape_values == ["single_optimum"]:
            return (
                f"As {x} increases, {y} is predicted to exhibit a "
                "single-optimum response."
            )
        if shape_values == ["u_shaped"]:
            return (
                f"As {x} increases, {y} is predicted to exhibit a "
                "U-shaped response."
            )
        if shape_values == ["inverted_u"]:
            return (
                f"As {x} increases, {y} is predicted to exhibit an "
                "inverted-U response."
            )
        if shape_values == ["threshold"]:
            return (
                f"As {x} increases, {y} is predicted to exhibit a "
                "threshold-type response."
            )
        if shape_values == ["saturating"]:
            return (
                f"As {x} increases, {y} is predicted to exhibit a "
                "saturating response."
            )
        return (
            f"As {x} increases, {y} is predicted to exhibit a "
            "non-monotonic response."
        )
    return (
        f"As {x} increases, the directional response of {y} remains "
        "unspecified by the frozen Trend evidence."
    )


def render_view_clause(
    view: HypothesisTrendInputView,
    *,
    dependent_change: str,
) -> str:
    return render_canonical_trend_clause(
        independent_variable_key=view.independent_variable_key,
        dependent_observable_key=view.dependent_observable_key,
        dependent_change=dependent_change,
        shapes=view.shapes,
    )


def render_context_qualification(
    view: HypothesisTrendInputView,
) -> str:
    relation = (
        f"{_label(view.independent_variable_key)} → "
        f"{_label(view.dependent_observable_key)}"
    )
    status = str(view.cross_context_status)
    if status == "repeated":
        return (
            f"For {relation}, the supplied cross-context assessment "
            "records repeated support; this does not authorize a "
            "universal relation."
        )
    if status == "context_specific":
        return (
            f"For {relation}, the supplied assessment is "
            "context-specific; the hypothesis remains context-qualified."
        )
    if status == "reversed":
        return (
            f"For {relation}, a reversal boundary is retained; opposing "
            "contextual evidence is not collapsed by majority vote."
        )
    if status == "insufficient":
        return (
            f"For {relation}, cross-context replication remains "
            "insufficient."
        )
    return (
        f"For {relation}, cross-context status is retained as "
        f"{status or 'unspecified'} without extrapolation."
    )


def render_hypothesis_statement(
    mechanistic_proposal: str,
    bound_view_clauses: Iterable[str],
) -> str:
    parts = [str(mechanistic_proposal).strip()]
    clauses = [str(value).strip() for value in bound_view_clauses if str(value).strip()]
    if clauses:
        parts.append(
            "Canonical Trend-grounded prediction: " + " ".join(clauses)
        )
    return " ".join(value for value in parts if value)


def render_inferential_bridge(
    inferential_bridge: str,
    qualifications: Iterable[str],
) -> str:
    parts = [str(inferential_bridge).strip()]
    rows = [str(value).strip() for value in qualifications if str(value).strip()]
    if rows:
        parts.append("Epistemic qualification: " + " ".join(rows))
    return " ".join(value for value in parts if value)


def render_prediction_rationale(
    *,
    canonical_clauses: Iterable[str],
    mechanistic_rationale: str,
) -> str:
    clauses = [str(value).strip() for value in canonical_clauses if str(value).strip()]
    parts: list[str] = []
    if clauses:
        parts.append("Canonical Trend frame: " + " ".join(clauses))
    parts.append(
        "Proposed mechanistic rationale: "
        + str(mechanistic_rationale).strip()
    )
    return " ".join(parts)
