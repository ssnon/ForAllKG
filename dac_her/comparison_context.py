from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import replace
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping

import networkx as nx

from dac_her.comparison_domain import (
    ComparisonAssessment,
    ComparisonCompatibility,
    ComparisonContext,
    ComparisonDimensionValue,
    ComparisonDomainAdapter,
    MeasurementUnitStatus,
    MetricDefinitionAssessment,
    ObservableComparisonPolicy,
)
from dac_her.method_context import (
    MethodContext,
    ProtocolAssessment,
)



def stable_comparison_id(*parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def normalize_comparison_value(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("×", "x")
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def dimension_from_values(
    name: str,
    values: Iterable[tuple[str, str]],
    *,
    normalizer: Callable[[Any], str] = normalize_comparison_value,
) -> ComparisonDimensionValue:
    grouped: dict[str, dict[str, set[str]]] = {}
    for raw_value, source_node_id in values:
        normalized = normalizer(raw_value)
        if not normalized:
            continue
        bucket = grouped.setdefault(
            normalized,
            {"raw": set(), "nodes": set()},
        )
        bucket["raw"].add(str(raw_value).strip())
        if str(source_node_id).strip():
            bucket["nodes"].add(str(source_node_id))

    if not grouped:
        return ComparisonDimensionValue(
            name=name,
            status="unknown",
        )

    source_values = tuple(sorted({
        raw
        for bucket in grouped.values()
        for raw in bucket["raw"]
    }))
    source_node_ids = tuple(sorted({
        node_id
        for bucket in grouped.values()
        for node_id in bucket["nodes"]
    }))

    if len(grouped) == 1:
        normalized = next(iter(grouped))
        return ComparisonDimensionValue(
            name=name,
            status="known",
            normalized_value=normalized,
            source_values=source_values,
            source_node_ids=source_node_ids,
        )

    return ComparisonDimensionValue(
        name=name,
        status="ambiguous",
        source_values=source_values,
        source_node_ids=source_node_ids,
    )


def _unit_status(
    left: ComparisonContext,
    right: ComparisonContext,
) -> MeasurementUnitStatus:
    left_unit = normalize_comparison_value(left.unit)
    right_unit = normalize_comparison_value(right.unit)
    if not left_unit or not right_unit:
        return "unknown"
    if left_unit == right_unit:
        return "matched"
    return "mismatched"


def _unknown_policy_assessment(
    left: ComparisonContext,
    right: ComparisonContext,
    *,
    adapter: ComparisonDomainAdapter,
) -> ComparisonAssessment:
    left_id, right_id = sorted((left.context_id, right.context_id))
    left_paper, right_paper = (
        (left.paper_id, right.paper_id)
        if left.context_id == left_id
        else (right.paper_id, left.paper_id)
    )
    return ComparisonAssessment(
        assessment_id=(
            "comparison:"
            + stable_comparison_id(
                adapter.semantics_id,
                left.observable_key,
                left_id,
                right_id,
            )
        ),
        comparison_semantics_id=adapter.semantics_id,
        observable_key=left.observable_key,
        observable_policy_id="unregistered",
        observable_family="unregistered",
        applicable_dimensions=(),
        ranking_required_dimensions=(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
        left_context_id=left_id,
        right_context_id=right_id,
        left_paper_id=left_paper,
        right_paper_id=right_paper,
        compatibility="unknown",
        measurement_unit_status=_unit_status(left, right),
        matched_dimensions=(),
        mismatched_dimensions=(),
        unknown_dimensions=(),
        ambiguous_dimensions=(),
        reasons=("observable_policy_unregistered",),
        numeric_ranking_allowed=False,
    )


def compare_contexts(
    left: ComparisonContext,
    right: ComparisonContext,
    *,
    adapter: ComparisonDomainAdapter,
) -> ComparisonAssessment:
    if left.context_id == right.context_id:
        raise ValueError(
            "Cannot compare a ComparisonContext with itself."
        )
    if left.paper_id == right.paper_id:
        raise ValueError(
            "Comparison assessments are cross-paper only."
        )
    if left.observable_key != right.observable_key:
        raise ValueError(
            "Comparison contexts must have the same observable_key: "
            f"{left.observable_key!r} != {right.observable_key!r}"
        )
    if left.comparison_semantics_id != adapter.semantics_id:
        raise ValueError(
            "Left context does not match comparison adapter semantics."
        )
    if right.comparison_semantics_id != adapter.semantics_id:
        raise ValueError(
            "Right context does not match comparison adapter semantics."
        )

    policy = adapter.policy_for(left.observable_key)
    if policy is None:
        return _unknown_policy_assessment(
            left,
            right,
            adapter=adapter,
        )

    matched: list[str] = []
    mismatched: list[str] = []
    unknown: list[str] = []
    ambiguous: list[str] = []
    reasons: list[str] = []

    left_dimensions = left.dimension_map
    right_dimensions = right.dimension_map

    for name in policy.applicable_dimensions:
        lvalue = left_dimensions[name]
        rvalue = right_dimensions[name]

        if (
            lvalue.status == "ambiguous"
            or rvalue.status == "ambiguous"
        ):
            unknown.append(name)
            ambiguous.append(name)
            reasons.append(f"ambiguous_dimension:{name}")
            continue
        if lvalue.status != "known" or rvalue.status != "known":
            unknown.append(name)
            reasons.append(f"unknown_dimension:{name}")
            continue
        if (
            lvalue.normalized_value
            == rvalue.normalized_value
        ):
            matched.append(name)
        else:
            mismatched.append(name)
            reasons.append(f"dimension_mismatch:{name}")

    if mismatched:
        compatibility: ComparisonCompatibility = "incompatible"
    elif not policy.applicable_dimensions:
        compatibility = "unknown"
        reasons.append("no_applicable_context_contract")
    elif not unknown:
        compatibility = "compatible"
    elif matched:
        compatibility = "partially_compatible"
    else:
        compatibility = "unknown"

    unit_status = _unit_status(left, right)
    if unit_status == "mismatched":
        reasons.append("measurement_unit_mismatch")
    elif unit_status == "unknown":
        reasons.append("measurement_unit_unknown")

    required = policy.ranking_required_dimensions
    required_all_matched = required.issubset(set(matched))
    numeric_ranking_allowed = bool(
        policy.numeric_ranking_mode == "allowed_if_complete"
        and required_all_matched
        and not mismatched
        and left.value_numeric is not None
        and right.value_numeric is not None
        and unit_status == "matched"
    )

    if policy.numeric_ranking_mode == "disabled":
        reasons.append("numeric_ranking_disabled_for_observable")
    elif not numeric_ranking_allowed:
        if left.value_numeric is None or right.value_numeric is None:
            reasons.append("numeric_value_missing")
        if not required_all_matched:
            reasons.append(
                "ranking_required_context_not_fully_matched"
            )

    left_id, right_id = sorted(
        (left.context_id, right.context_id)
    )
    left_paper, right_paper = (
        (left.paper_id, right.paper_id)
        if left.context_id == left_id
        else (right.paper_id, left.paper_id)
    )

    return ComparisonAssessment(
        assessment_id=(
            "comparison:"
            + stable_comparison_id(
                adapter.semantics_id,
                left.observable_key,
                left_id,
                right_id,
            )
        ),
        comparison_semantics_id=adapter.semantics_id,
        observable_key=left.observable_key,
        observable_policy_id=policy.policy_id,
        observable_family=policy.family,
        applicable_dimensions=tuple(
            policy.applicable_dimensions
        ),
        ranking_required_dimensions=tuple(sorted(required)),
        numeric_ranking_mode=policy.numeric_ranking_mode,
        ranking_direction=policy.ranking_direction,
        left_context_id=left_id,
        right_context_id=right_id,
        left_paper_id=left_paper,
        right_paper_id=right_paper,
        compatibility=compatibility,
        measurement_unit_status=unit_status,
        matched_dimensions=tuple(matched),
        mismatched_dimensions=tuple(mismatched),
        unknown_dimensions=tuple(unknown),
        ambiguous_dimensions=tuple(ambiguous),
        reasons=tuple(dict.fromkeys(reasons)),
        numeric_ranking_allowed=numeric_ranking_allowed,
    )


def method_dimension_from_values(
    name: str,
    values: Iterable[tuple[str, str, str]],
    *,
    normalizer: Callable[[Any], str] = normalize_comparison_value,
):
    from dac_her.method_context import MethodDimensionValue

    grouped: dict[str, dict[str, set[str]]] = {}
    scopes: set[str] = set()
    for raw_value, source_node_id, provenance_scope in values:
        normalized = normalizer(raw_value)
        if not normalized:
            continue
        bucket = grouped.setdefault(
            normalized,
            {"raw": set(), "nodes": set()},
        )
        bucket["raw"].add(str(raw_value).strip())
        if str(source_node_id).strip():
            bucket["nodes"].add(str(source_node_id))
        if str(provenance_scope).strip():
            scopes.add(str(provenance_scope).strip())

    if not grouped:
        return MethodDimensionValue(name=name, status="unknown")

    source_values = tuple(sorted({
        raw
        for bucket in grouped.values()
        for raw in bucket["raw"]
    }))
    source_node_ids = tuple(sorted({
        node_id
        for bucket in grouped.values()
        for node_id in bucket["nodes"]
    }))
    provenance_scopes = tuple(sorted(scopes))

    if len(grouped) == 1:
        normalized = next(iter(grouped))
        return MethodDimensionValue(
            name=name,
            status="known",
            normalized_value=normalized,
            source_values=source_values,
            source_node_ids=source_node_ids,
            provenance_scopes=provenance_scopes,
        )

    return MethodDimensionValue(
        name=name,
        status="ambiguous",
        source_values=source_values,
        source_node_ids=source_node_ids,
        provenance_scopes=provenance_scopes,
    )


def compare_method_contexts(
    *,
    left_context: ComparisonContext,
    right_context: ComparisonContext,
    left_method: MethodContext,
    right_method: MethodContext,
    adapter: ComparisonDomainAdapter,
) -> ProtocolAssessment:
    semantics = adapter.method_semantics
    if semantics is None:
        raise ValueError(
            "Protocol comparison requires adapter.method_semantics."
        )
    if left_context.paper_id == right_context.paper_id:
        raise ValueError("Protocol assessments are cross-paper only.")

    matched: list[str] = []
    mismatched: list[str] = []
    unknown: list[str] = []
    ambiguous: list[str] = []
    reasons: list[str] = []

    left_dimensions = left_method.dimension_map
    right_dimensions = right_method.dimension_map

    for name in semantics.dimensions:
        left = left_dimensions[name]
        right = right_dimensions[name]
        if left.status == "ambiguous" or right.status == "ambiguous":
            unknown.append(name)
            ambiguous.append(name)
            reasons.append(f"ambiguous_method_dimension:{name}")
            continue
        if left.status != "known" or right.status != "known":
            unknown.append(name)
            reasons.append(f"unknown_method_dimension:{name}")
            continue
        if left.normalized_value == right.normalized_value:
            matched.append(name)
        else:
            mismatched.append(name)
            reasons.append(f"method_dimension_mismatch:{name}")

    critical_mismatches = tuple(
        name
        for name in mismatched
        if name in semantics.critical_dimensions
    )
    if critical_mismatches:
        comparability = "different_protocol"
    elif (
        len(matched) == len(semantics.dimensions)
        and not mismatched
        and not unknown
    ):
        comparability = "same_protocol"
    elif matched:
        # Partial means at least one explicit equality. Unknown or
        # non-critical mismatch may remain, but a pair with mismatch-only
        # evidence is never called "partially matched".
        comparability = "partially_matched"
    elif mismatched:
        comparability = "different_protocol"
    else:
        comparability = "unknown"

    numeric_protocol_gate = (
        comparability
        in semantics.numeric_ranking_allowed_protocols
    )
    if not numeric_protocol_gate:
        reasons.append(
            "protocol_not_sufficient_for_numeric_ranking"
        )

    left_method_id, right_method_id = sorted((
        left_method.method_context_id,
        right_method.method_context_id,
    ))
    if left_method.method_context_id == left_method_id:
        left_context_id = left_context.context_id
        right_context_id = right_context.context_id
        left_paper_id = left_context.paper_id
        right_paper_id = right_context.paper_id
    else:
        left_context_id = right_context.context_id
        right_context_id = left_context.context_id
        left_paper_id = right_context.paper_id
        right_paper_id = left_context.paper_id

    return ProtocolAssessment(
        protocol_assessment_id=(
            "protocol:"
            + stable_comparison_id(
                semantics.semantics_id,
                left_context.observable_key,
                left_method_id,
                right_method_id,
            )
        ),
        method_semantics_id=semantics.semantics_id,
        observable_key=left_context.observable_key,
        left_context_id=left_context_id,
        right_context_id=right_context_id,
        left_method_context_id=left_method_id,
        right_method_context_id=right_method_id,
        left_paper_id=left_paper_id,
        right_paper_id=right_paper_id,
        comparability=comparability,
        matched_dimensions=tuple(matched),
        mismatched_dimensions=tuple(mismatched),
        unknown_dimensions=tuple(unknown),
        ambiguous_dimensions=tuple(ambiguous),
        critical_mismatches=critical_mismatches,
        reasons=tuple(dict.fromkeys(reasons)),
        numeric_protocol_gate=numeric_protocol_gate,
    )


def build_protocol_assessments(
    contexts: Iterable[ComparisonContext],
    method_contexts: Iterable[MethodContext],
    *,
    adapter: ComparisonDomainAdapter,
) -> list[ProtocolAssessment]:
    if adapter.method_semantics is None:
        return []

    methods = {
        context.method_context_id: context
        for context in method_contexts
    }
    grouped: dict[str, list[ComparisonContext]] = defaultdict(list)
    for context in contexts:
        grouped[context.observable_key].append(context)

    assessments: list[ProtocolAssessment] = []
    for observable_key in sorted(grouped):
        rows = sorted(
            grouped[observable_key],
            key=lambda item: (item.paper_id, item.context_id),
        )
        for left, right in itertools.combinations(rows, 2):
            if left.paper_id == right.paper_id:
                continue
            if not left.method_context_id or not right.method_context_id:
                raise ValueError(
                    "SERS comparison context is missing method_context_id."
                )
            try:
                left_method = methods[left.method_context_id]
                right_method = methods[right.method_context_id]
            except KeyError as exc:
                raise ValueError(
                    "Comparison context refers to missing MethodContext."
                ) from exc
            assessments.append(
                compare_method_contexts(
                    left_context=left,
                    right_context=right,
                    left_method=left_method,
                    right_method=right_method,
                    adapter=adapter,
                )
            )

    return sorted(
        assessments,
        key=lambda item: (
            item.observable_key,
            item.left_paper_id,
            item.right_paper_id,
            item.protocol_assessment_id,
        ),
    )


def apply_protocol_numeric_gate(
    assessments: Iterable[ComparisonAssessment],
    protocol_assessments: Iterable[ProtocolAssessment],
    *,
    adapter: ComparisonDomainAdapter,
) -> list[ComparisonAssessment]:
    if adapter.method_semantics is None:
        return list(assessments)

    protocol_by_pair = {
        (
            item.observable_key,
            item.left_context_id,
            item.right_context_id,
        ): item
        for item in protocol_assessments
    }

    gated: list[ComparisonAssessment] = []
    for assessment in assessments:
        key = (
            assessment.observable_key,
            assessment.left_context_id,
            assessment.right_context_id,
        )
        protocol = protocol_by_pair.get(key)
        if protocol is None:
            reverse_key = (
                assessment.observable_key,
                assessment.right_context_id,
                assessment.left_context_id,
            )
            protocol = protocol_by_pair.get(reverse_key)
        if protocol is None:
            raise ValueError(
                "Missing protocol assessment for comparison assessment "
                f"{assessment.assessment_id!r}."
            )

        allowed = (
            assessment.numeric_ranking_allowed
            and protocol.numeric_protocol_gate
        )
        reasons = list(assessment.reasons)
        if (
            assessment.numeric_ranking_allowed
            and not protocol.numeric_protocol_gate
        ):
            reasons.append(
                "numeric_ranking_blocked_by_protocol_context"
            )

        gated.append(
            replace(
                assessment,
                numeric_ranking_allowed=allowed,
                reasons=tuple(dict.fromkeys(reasons)),
                protocol_assessment_id=(
                    protocol.protocol_assessment_id
                ),
                protocol_comparability=protocol.comparability,
            )
        )
    return gated

def build_pairwise_assessments(
    contexts: Iterable[ComparisonContext],
    *,
    adapter: ComparisonDomainAdapter,
) -> list[ComparisonAssessment]:
    grouped: dict[str, list[ComparisonContext]] = defaultdict(list)
    for context in contexts:
        grouped[context.observable_key].append(context)

    assessments: list[ComparisonAssessment] = []
    for observable_key in sorted(grouped):
        rows = sorted(
            grouped[observable_key],
            key=lambda item: (item.paper_id, item.context_id),
        )
        for left, right in itertools.combinations(rows, 2):
            if left.paper_id == right.paper_id:
                continue
            assessments.append(
                compare_contexts(left, right, adapter=adapter)
            )

    return sorted(
        assessments,
        key=lambda item: (
            item.observable_key,
            item.left_paper_id,
            item.right_paper_id,
            item.assessment_id,
        ),
    )


def audit_comparison_outputs(
    *,
    contexts: Iterable[ComparisonContext],
    assessments: Iterable[ComparisonAssessment],
    source_graphs: Mapping[str, nx.Graph],
    adapter: ComparisonDomainAdapter,
    method_contexts: Iterable[MethodContext] = (),
    protocol_assessments: Iterable[ProtocolAssessment] = (),
    metric_definition_assessments: Iterable[
        MetricDefinitionAssessment
    ] = (),
) -> dict[str, Any]:
    context_rows = list(contexts)
    assessment_rows = list(assessments)
    method_rows = list(method_contexts)
    protocol_rows = list(protocol_assessments)
    metric_definition_rows = list(metric_definition_assessments)
    issues: list[dict[str, str]] = []

    context_by_id = {item.context_id: item for item in context_rows}
    if len(context_by_id) != len(context_rows):
        issues.append({
            "code": "DUPLICATE_CONTEXT_ID",
            "message": "Comparison context IDs are not unique.",
        })

    for context in context_rows:
        graph = source_graphs.get(context.paper_id)
        if graph is None:
            issues.append({
                "code": "MISSING_SOURCE_GRAPH",
                "message": f"No source graph for {context.paper_id}.",
            })
            continue
        if context.measurement_id not in graph:
            issues.append({
                "code": "MISSING_SOURCE_MEASUREMENT",
                "message": (
                    f"{context.context_id}: source measurement "
                    f"{context.measurement_id!r} does not exist."
                ),
            })
        elif str(graph.nodes[context.measurement_id].get("type", "")) != "Measurement":
            issues.append({
                "code": "SOURCE_NOT_MEASUREMENT",
                "message": (
                    f"{context.context_id}: source node "
                    f"{context.measurement_id!r} is not Measurement."
                ),
            })
        for source_node_id in context.source_node_ids:
            if source_node_id not in graph:
                issues.append({
                    "code": "INVENTED_SOURCE_NODE",
                    "message": (
                        f"{context.context_id}: source node "
                        f"{source_node_id!r} does not exist."
                    ),
                })
        for dimension in context.dimensions:
            for source_node_id in dimension.source_node_ids:
                if source_node_id not in graph:
                    issues.append({
                        "code": "INVENTED_DIMENSION_SOURCE",
                        "message": (
                            f"{context.context_id}/{dimension.name}: "
                            f"{source_node_id!r} does not exist."
                        ),
                    })

    method_by_id = {
        item.method_context_id: item for item in method_rows
    }
    if len(method_by_id) != len(method_rows):
        issues.append({
            "code": "DUPLICATE_METHOD_CONTEXT_ID",
            "message": "MethodContext IDs are not unique.",
        })

    if adapter.method_semantics is not None:
        for context in context_rows:
            if not context.method_context_id:
                issues.append({
                    "code": "MISSING_METHOD_CONTEXT_LINK",
                    "message": context.context_id,
                })
            elif context.method_context_id not in method_by_id:
                issues.append({
                    "code": "METHOD_CONTEXT_LINK_NOT_FOUND",
                    "message": context.context_id,
                })

        for method in method_rows:
            graph = source_graphs.get(method.paper_id)
            if graph is None:
                issues.append({
                    "code": "METHOD_SOURCE_GRAPH_MISSING",
                    "message": method.method_context_id,
                })
                continue
            if method.measurement_id not in graph:
                issues.append({
                    "code": "METHOD_SOURCE_MEASUREMENT_MISSING",
                    "message": method.method_context_id,
                })
            for source_node_id in method.source_node_ids:
                if source_node_id not in graph:
                    issues.append({
                        "code": "METHOD_SOURCE_NODE_INVENTED",
                        "message": (
                            f"{method.method_context_id}:"
                            f"{source_node_id}"
                        ),
                    })
            for dimension in method.dimensions:
                if (
                    dimension.status != "unknown"
                    and not dimension.provenance_scopes
                ):
                    issues.append({
                        "code": "METHOD_PROVENANCE_SCOPE_MISSING",
                        "message": (
                            f"{method.method_context_id}:"
                            f"{dimension.name}"
                        ),
                    })
                if dimension.name == "analyte_concentration":
                    for source_node_id in dimension.source_node_ids:
                        if source_node_id not in graph:
                            continue
                        source_type = str(
                            graph.nodes[source_node_id].get("type", "")
                        )
                        if source_type in {"Analyte", "RamanReporter"}:
                            issues.append({
                                "code": "GLOBAL_ENTITY_CONCENTRATION_LEAK",
                                "message": (
                                    f"{method.method_context_id}:"
                                    f"{source_node_id}"
                                ),
                            })
                if dimension.name in {
                    "sample_preparation",
                    "preparation_medium",
                    "measurement_environment",
                    "sample_state",
                    "substrate_condition",
                }:
                    for source_node_id in dimension.source_node_ids:
                        if source_node_id not in graph:
                            continue
                        source_attrs = graph.nodes[source_node_id]
                        source_type = str(source_attrs.get("type", ""))
                        if source_type not in {"Measurement", "Experiment"}:
                            issues.append({
                                "code": "METHOD_PROTOCOL_SCOPE_LEAK",
                                "message": (
                                    f"{method.method_context_id}:"
                                    f"{dimension.name}:"
                                    f"{source_node_id}"
                                ),
                            })
                        if (
                            dimension.name
                            in {
                                "preparation_medium",
                                "measurement_environment",
                            }
                            and source_type == "Experiment"
                        ):
                            experiment_type = normalize_comparison_value(
                                source_attrs.get("experiment_type", "")
                            )
                            method_label = normalize_comparison_value(
                                source_attrs.get("method_label", "")
                            )
                            if any(
                                marker in experiment_type
                                or marker in method_label
                                for marker in (
                                    "calculation",
                                    "simulation",
                                    "approximation",
                                )
                            ):
                                issues.append({
                                    "code": "SIMULATION_MEDIUM_LEAK",
                                    "message": (
                                        f"{method.method_context_id}:"
                                        f"{source_node_id}"
                                    ),
                                })

        if len(protocol_rows) != len(assessment_rows):
            issues.append({
                "code": "PROTOCOL_ASSESSMENT_COUNT_MISMATCH",
                "message": (
                    f"protocol={len(protocol_rows)} "
                    f"comparison={len(assessment_rows)}"
                ),
            })

    metric_definition_by_comparison: dict[
        str,
        MetricDefinitionAssessment,
    ] = {}
    if metric_definition_rows:
        metric_ids: set[str] = set()
        for metric in metric_definition_rows:
            if metric.assessment_id in metric_ids:
                issues.append({
                    "code": "DUPLICATE_METRIC_DEFINITION_ASSESSMENT_ID",
                    "message": metric.assessment_id,
                })
            metric_ids.add(metric.assessment_id)
            if (
                metric.comparison_assessment_id
                in metric_definition_by_comparison
            ):
                issues.append({
                    "code": (
                        "DUPLICATE_METRIC_DEFINITION_COMPARISON_LINK"
                    ),
                    "message": metric.comparison_assessment_id,
                })
            metric_definition_by_comparison[
                metric.comparison_assessment_id
            ] = metric

        if len(metric_definition_rows) != len(assessment_rows):
            issues.append({
                "code": "METRIC_DEFINITION_ASSESSMENT_COUNT_MISMATCH",
                "message": (
                    f"metric_definition={len(metric_definition_rows)} "
                    f"comparison={len(assessment_rows)}"
                ),
            })

    assessment_ids: set[str] = set()
    for assessment in assessment_rows:
        if assessment.assessment_id in assessment_ids:
            issues.append({
                "code": "DUPLICATE_ASSESSMENT_ID",
                "message": assessment.assessment_id,
            })
        assessment_ids.add(assessment.assessment_id)

        left = context_by_id.get(assessment.left_context_id)
        right = context_by_id.get(assessment.right_context_id)
        if left is None or right is None:
            issues.append({
                "code": "ASSESSMENT_CONTEXT_MISSING",
                "message": assessment.assessment_id,
            })
            continue

        if assessment.numeric_ranking_allowed:
            if assessment.compatibility != "compatible":
                issues.append({
                    "code": "UNSAFE_NUMERIC_RANKING_COMPATIBILITY",
                    "message": assessment.assessment_id,
                })
            if assessment.measurement_unit_status != "matched":
                issues.append({
                    "code": "UNSAFE_NUMERIC_RANKING_UNIT",
                    "message": assessment.assessment_id,
                })
            if left.value_numeric is None or right.value_numeric is None:
                issues.append({
                    "code": "UNSAFE_NUMERIC_RANKING_VALUE",
                    "message": assessment.assessment_id,
                })
            required = set(assessment.ranking_required_dimensions)
            matched = set(assessment.matched_dimensions)
            if not required.issubset(matched):
                issues.append({
                    "code": "UNSAFE_NUMERIC_RANKING_REQUIRED_CONTEXT",
                    "message": assessment.assessment_id,
                })
            if assessment.mismatched_dimensions:
                issues.append({
                    "code": "UNSAFE_NUMERIC_RANKING_MISMATCH",
                    "message": assessment.assessment_id,
                })
            if assessment.numeric_ranking_mode != "allowed_if_complete":
                issues.append({
                    "code": "UNSAFE_NUMERIC_RANKING_POLICY",
                    "message": assessment.assessment_id,
                })
            if (
                adapter.method_semantics is not None
                and assessment.protocol_comparability
                not in adapter.method_semantics.numeric_ranking_allowed_protocols
            ):
                issues.append({
                    "code": "UNSAFE_NUMERIC_RANKING_PROTOCOL",
                    "message": assessment.assessment_id,
                })
            if not assessment.metric_definition_gate:
                issues.append({
                    "code": "UNSAFE_NUMERIC_RANKING_METRIC_DEFINITION",
                    "message": assessment.assessment_id,
                })
            if assessment.metric_definition_compatibility not in {
                "same_definition",
                "not_applicable",
            }:
                issues.append({
                    "code": (
                        "UNSAFE_NUMERIC_RANKING_METRIC_COMPATIBILITY"
                    ),
                    "message": assessment.assessment_id,
                })

        if metric_definition_rows:
            metric = metric_definition_by_comparison.get(
                assessment.assessment_id
            )
            if metric is None:
                issues.append({
                    "code": "MISSING_METRIC_DEFINITION_ASSESSMENT_LINK",
                    "message": assessment.assessment_id,
                })
            else:
                if (
                    assessment.metric_definition_assessment_id
                    != metric.assessment_id
                ):
                    issues.append({
                        "code": "METRIC_DEFINITION_LINK_MISMATCH",
                        "message": assessment.assessment_id,
                    })
                if (
                    assessment.metric_definition_compatibility
                    != metric.compatibility
                ):
                    issues.append({
                        "code": (
                            "METRIC_DEFINITION_COMPATIBILITY_MISMATCH"
                        ),
                        "message": assessment.assessment_id,
                    })
                if (
                    assessment.metric_definition_gate
                    != metric.numeric_metric_definition_gate
                ):
                    issues.append({
                        "code": "METRIC_DEFINITION_GATE_MISMATCH",
                        "message": assessment.assessment_id,
                    })

    compatibility_counts = Counter(
        item.compatibility for item in assessment_rows
    )
    observable_policy_counts = Counter(
        item.observable_policy_id for item in assessment_rows
    )
    observable_family_counts = Counter(
        item.observable_family for item in assessment_rows
    )
    protocol_comparability_counts = Counter(
        item.comparability for item in protocol_rows
    )
    metric_definition_compatibility_counts = Counter(
        item.compatibility for item in metric_definition_rows
    )
    unknown_dimension_counts = Counter(
        dimension.name
        for context in context_rows
        for dimension in context.dimensions
        if dimension.status == "unknown"
    )
    ambiguous_dimension_counts = Counter(
        dimension.name
        for context in context_rows
        for dimension in context.dimensions
        if dimension.status == "ambiguous"
    )
    method_dimension_status_counts: dict[str, dict[str, int]] = {}
    for method in method_rows:
        for dimension in method.dimensions:
            bucket = method_dimension_status_counts.setdefault(
                dimension.name,
                {"known": 0, "unknown": 0, "ambiguous": 0},
            )
            bucket[dimension.status] += 1

    method_provenance_scope_counts = Counter(
        scope
        for method in method_rows
        for dimension in method.dimensions
        for scope in dimension.provenance_scopes
    )
    protocol_matched_dimension_counts = Counter(
        name
        for assessment in protocol_rows
        for name in assessment.matched_dimensions
    )
    protocol_mismatched_dimension_counts = Counter(
        name
        for assessment in protocol_rows
        for name in assessment.mismatched_dimensions
    )
    protocol_pairs_with_any_match = sum(
        bool(assessment.matched_dimensions)
        for assessment in protocol_rows
    )

    assessment_by_id = {
        item.assessment_id: item
        for item in assessment_rows
    }
    ranking_relevant_metric_rows = [
        item
        for item in metric_definition_rows
        if item.compatibility != "not_applicable"
        and item.comparison_assessment_id in assessment_by_id
        and assessment_by_id[
            item.comparison_assessment_id
        ].numeric_ranking_mode != "disabled"
    ]

    return {
        "comparison_adapter_id": adapter.adapter_id,
        "comparison_semantics_id": adapter.semantics_id,
        "context_count": len(context_rows),
        "assessment_count": len(assessment_rows),
        "method_context_count": len(method_rows),
        "protocol_assessment_count": len(protocol_rows),
        "protocol_comparability_counts": dict(
            sorted(protocol_comparability_counts.items())
        ),
        "metric_definition_assessment_count": len(
            metric_definition_rows
        ),
        "metric_definition_compatibility_counts": dict(
            sorted(metric_definition_compatibility_counts.items())
        ),
        "metric_definition_applicable_assessment_count": sum(
            item.compatibility != "not_applicable"
            for item in metric_definition_rows
        ),
        "metric_definition_gate_pass_count": sum(
            item.numeric_metric_definition_gate
            and item.compatibility != "not_applicable"
            for item in metric_definition_rows
        ),
        "metric_definition_gate_blocked_applicable_count": sum(
            (not item.numeric_metric_definition_gate)
            and item.compatibility != "not_applicable"
            for item in metric_definition_rows
        ),
        "metric_definition_ranking_relevant_assessment_count": len(
            ranking_relevant_metric_rows
        ),
        "metric_definition_ranking_relevant_gate_pass_count": sum(
            item.numeric_metric_definition_gate
            for item in ranking_relevant_metric_rows
        ),
        "metric_definition_ranking_relevant_gate_blocked_count": sum(
            not item.numeric_metric_definition_gate
            for item in ranking_relevant_metric_rows
        ),
        "numeric_context_count": sum(
            item.value_numeric is not None for item in context_rows
        ),
        "numeric_ranking_allowed_count": sum(
            item.numeric_ranking_allowed for item in assessment_rows
        ),
        "compatibility_counts": dict(sorted(compatibility_counts.items())),
        "observable_policy_counts": dict(
            sorted(observable_policy_counts.items())
        ),
        "observable_family_counts": dict(
            sorted(observable_family_counts.items())
        ),
        "unknown_dimension_counts": dict(
            sorted(unknown_dimension_counts.items())
        ),
        "ambiguous_dimension_counts": dict(
            sorted(ambiguous_dimension_counts.items())
        ),
        "method_dimension_status_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(
                method_dimension_status_counts.items()
            )
        },
        "method_provenance_scope_counts": dict(
            sorted(method_provenance_scope_counts.items())
        ),
        "protocol_matched_dimension_counts": dict(
            sorted(protocol_matched_dimension_counts.items())
        ),
        "protocol_mismatched_dimension_counts": dict(
            sorted(protocol_mismatched_dimension_counts.items())
        ),
        "protocol_pairs_with_any_match": protocol_pairs_with_any_match,
        "missing_context_is_not_quarantine": True,
        "issues": issues,
        "passes_structural_gate": not issues,
    }
