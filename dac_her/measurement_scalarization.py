from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

from dac_her.schemas import KnowledgeGraph, MeasurementNode


_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+−]?\d+(?:\.\d+)?")
_COMPOSITE_MARKERS = (
    ";",
    "respectively",
    " in acid",
    " in base",
    " acidic",
    " alkaline",
    " before ",
    " after ",
    " compared with ",
    " versus ",
    " vs. ",
)


@dataclass(frozen=True)
class ScalarizationIssue:
    measurement_id: str
    issue: str
    value_text: str
    source_expression: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def numeric_tokens(value: str) -> tuple[str, ...]:
    return tuple(_NUMBER_RE.findall(value or ""))


def measurement_scalarization_issues(
    graph: KnowledgeGraph,
) -> list[ScalarizationIssue]:
    """Detect non-scalar result payloads conservatively.

    Numeric measurements are already scalar by schema. Textual measurements
    are flagged only when they contain multiple numbers plus comparison or
    condition markers. Qualitative text without multiple values is allowed.
    """
    issues: list[ScalarizationIssue] = []
    for measurement in graph.measurements:
        if measurement.value_numeric is not None:
            continue
        text = (measurement.value_text or "").strip()
        lowered = f" {text.lower()} "
        numbers = numeric_tokens(text)
        if len(numbers) >= 2 and any(marker in lowered for marker in _COMPOSITE_MARKERS):
            issues.append(ScalarizationIssue(
                measurement_id=measurement.id,
                issue=(
                    "textual measurement appears to combine multiple scalar "
                    "values or conditions; split it into separate Measurement "
                    "nodes and optionally a MeasurementGroup"
                ),
                value_text=text,
                source_expression=measurement.source_expression,
            ))
    return issues


def format_scalarization_errors(issues: Iterable[ScalarizationIssue]) -> str:
    rows = list(issues)
    if not rows:
        return ""
    return "Measurement scalarization failed:\n" + "\n".join(
        f"- {item.measurement_id}: {item.issue}; value_text={item.value_text!r}"
        for item in rows
    )
