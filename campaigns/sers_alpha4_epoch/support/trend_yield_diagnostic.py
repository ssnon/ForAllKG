from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx


ALPHA4C5G_DIAGNOSTIC_ID = "trend_yield_recall_diagnostic_v1_alpha4c5g"
EXPECTED_DOMAIN_PROFILE_ID = "sers_au_ag"
CLAIM_TYPES = frozenset({"ObservationClaim", "MechanismClaim"})
NUMERIC_TREND_BASES = frozenset({
    "controlled_numeric_pair",
    "controlled_numeric_series",
})
CLAIM_TREND_BASES = frozenset({
    "reported_directional_claim",
    "reported_correlation",
})

METHOD_GUARD_DIMENSIONS = (
    "analyte",
    "reporter",
    "analyte_concentration",
    "excitation_wavelength",
    "laser_power",
    "integration_time",
    "sample_preparation",
    "preparation_medium",
    "measurement_environment",
    "sample_state",
    "substrate_condition",
)


def norm(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("μ", "µ")
        .replace("×", "x")
    )
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected JSON object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def relation(attrs: Mapping[str, Any]) -> str:
    return str(attrs.get("relation", "")).strip()


def outgoing(
    graph: nx.Graph,
    node_id: str,
    rel: str,
) -> tuple[str, ...]:
    if node_id not in graph or not graph.is_directed():
        return ()
    values: set[str] = set()
    if graph.is_multigraph():
        iterator = graph.out_edges(
            node_id,
            keys=True,
            data=True,
        )
        for _left, right, _key, attrs in iterator:
            if relation(attrs) == rel:
                values.add(str(right))
    else:
        iterator = graph.out_edges(node_id, data=True)
        for _left, right, attrs in iterator:
            if relation(attrs) == rel:
                values.add(str(right))
    return tuple(sorted(values))


def incoming(
    graph: nx.Graph,
    node_id: str,
    rel: str,
) -> tuple[str, ...]:
    if node_id not in graph or not graph.is_directed():
        return ()
    values: set[str] = set()
    if graph.is_multigraph():
        iterator = graph.in_edges(
            node_id,
            keys=True,
            data=True,
        )
        for left, _right, _key, attrs in iterator:
            if relation(attrs) == rel:
                values.add(str(left))
    else:
        iterator = graph.in_edges(node_id, data=True)
        for left, _right, attrs in iterator:
            if relation(attrs) == rel:
                values.add(str(left))
    return tuple(sorted(values))


def claim_text(attrs: Mapping[str, Any]) -> str:
    for key in (
        "statement",
        "source_expression",
        "description",
        "label",
        "node_text",
    ):
        value = str(attrs.get(key, "")).strip()
        if value:
            return value
    return ""


_DIRECTION_RE = re.compile(
    r"\b(?:"
    r"increas\w*|decreas\w*|higher|lower|stronger|weaker|"
    r"enhanc\w*|improv\w*|reduc\w*|suppress\w*|"
    r"strongest|weakest|highest|lowest|best|worst|"
    r"optimal|optimum|maxim\w*|minim\w*|plateau\w*|"
    r"correlat\w*|depend\w*|saturat\w*|"
    r"similar|comparable|greater|smaller|larger|narrower|wider"
    r")\b",
    re.I,
)

_RESPONSE_RE = re.compile(
    r"\b(?:sers|serrs|raman)\b",
    re.I,
)

_CONTROL_RE = re.compile(
    r"\b(?:"
    r"thickness|gap|distance|spacing|ratio|composition|fraction|"
    r"concentration|loading|coverage|content|amount|size|diameter|"
    r"length|height|width|aspect\s+ratio|morpholog\w*|shape|"
    r"temperature|time|power|wavelength|potential|voltage|"
    r"electrodeposition|etch\w*|growth|cycle|layer|shell|core|"
    r"nanostructure|roughness"
    r")\b",
    re.I,
)


def broad_claim_candidate(
    *,
    paper_id: str,
    claim_id: str,
    attrs: Mapping[str, Any],
    admitted_claim_ids: set[str],
) -> dict[str, Any] | None:
    if str(attrs.get("type", "")) not in CLAIM_TYPES:
        return None
    text = claim_text(attrs)
    if not text:
        return None
    response_signal = bool(_RESPONSE_RE.search(text))
    direction_signal = bool(_DIRECTION_RE.search(text))
    varied_attr = str(
        attrs.get("varied_dimension")
        or attrs.get("independent_variable")
        or attrs.get("control_variable")
        or ""
    ).strip()
    control_signal = bool(
        varied_attr
        or _CONTROL_RE.search(text)
        or re.search(
            r"\b(?:with|as|versus|vs\.?|compared\s+with|"
            r"function\s+of|vary\w*|chang\w*)\b",
            text,
            re.I,
        )
    )
    if not (
        response_signal
        and direction_signal
        and control_signal
    ):
        return None
    return {
        "paper_id": paper_id,
        "claim_id": claim_id,
        "claim_type": str(attrs.get("type", "")),
        "text": text,
        "response_signal": response_signal,
        "direction_signal": direction_signal,
        "control_signal": control_signal,
        "explicit_varied_dimension": varied_attr,
        "admitted_by_current_trend": (
            claim_id in admitted_claim_ids
        ),
    }


def parse_conditions(
    attrs: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw = attrs.get("conditions_json", "")
    if isinstance(raw, list):
        parsed = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return ()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(
        dict(row)
        for row in parsed
        if isinstance(row, dict)
    )


def numeric_condition_value(
    condition: Mapping[str, Any],
) -> float | None:
    value = condition.get("value_numeric")
    if value is not None and str(value).strip():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        if math.isfinite(number):
            return number
    text = str(condition.get("value_text") or "").strip()
    match = re.search(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
        text,
    )
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def normalize_condition_name(name: Any) -> str:
    text = norm(name)
    text = re.sub(r"[^a-z0-9µ%/+:\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def condition_method_dimension(name: str) -> str:
    text = norm(name)
    if "concentration" in text:
        return "analyte_concentration"
    if (
        "laser power" in text
        or "excitation power" in text
        or text == "power"
    ):
        return "laser_power"
    if (
        "integration time" in text
        or "acquisition time" in text
        or "exposure time" in text
    ):
        return "integration_time"
    if (
        "excitation wavelength" in text
        or "laser wavelength" in text
    ):
        return "excitation_wavelength"
    return ""


def identity_by_representative(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(
            row.get("representative_measurement_id", "")
        ).strip()
        if key:
            result[key] = row
    return result


def method_by_id(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("method_context_id", "")): row
        for row in rows
        if str(row.get("method_context_id", "")).strip()
    }


def method_dimension_map(
    row: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in row.get("dimensions", []) or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            result[name] = item
    return result


def methods_compatible(
    rows: Iterable[Mapping[str, Any]],
    *,
    varied_method_dimension: str = "",
) -> tuple[bool, list[str]]:
    rows = list(rows)
    reasons: list[str] = []
    for name in METHOD_GUARD_DIMENSIONS:
        if name == varied_method_dimension:
            continue
        known: set[str] = set()
        for row in rows:
            item = method_dimension_map(row).get(name)
            if item is None:
                continue
            status = str(
                item.get("status", "unknown")
            ).strip()
            if status == "ambiguous":
                reasons.append(
                    f"ambiguous_method_dimension:{name}"
                )
                continue
            if status == "known":
                value = str(
                    item.get("normalized_value", "")
                ).strip()
                if value:
                    known.add(value)
        if len(known) > 1:
            reasons.append(
                f"conflicting_method_dimension:{name}"
            )
    return (not reasons), sorted(set(reasons))


def measurement_mentions(
    measurement_id: str,
    identity_row: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if identity_row is None:
        return (measurement_id,)
    values = tuple(
        str(value)
        for value in identity_row.get(
            "source_mention_ids",
            [],
        )
        if str(value).strip()
    )
    return values or (measurement_id,)


def measurement_lineage(
    graph: nx.Graph,
    measurement_id: str,
    identity_row: Mapping[str, Any] | None,
    method_row: Mapping[str, Any] | None,
    context_row: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]] | None:
    mentions = measurement_mentions(
        measurement_id,
        identity_row,
    )
    group_ids: set[str] = set()
    experiment_ids: set[str] = set()

    for mention_id in mentions:
        group_ids.update(
            outgoing(
                graph,
                mention_id,
                "IN_MEASUREMENT_GROUP",
            )
        )
        for producer in incoming(
            graph,
            mention_id,
            "HAS_MEASUREMENT",
        ):
            if (
                producer in graph
                and str(
                    graph.nodes[producer].get(
                        "type",
                        "",
                    )
                )
                == "Experiment"
            ):
                experiment_ids.add(producer)

    if method_row is not None:
        for node_id in method_row.get(
            "source_node_ids",
            [],
        ) or []:
            node_id = str(node_id)
            if (
                node_id in graph
                and str(
                    graph.nodes[node_id].get(
                        "type",
                        "",
                    )
                )
                == "MeasurementGroup"
            ):
                group_ids.add(node_id)
        for producer in method_row.get(
            "producer_ids",
            [],
        ) or []:
            producer = str(producer)
            if (
                producer in graph
                and str(
                    graph.nodes[producer].get(
                        "type",
                        "",
                    )
                )
                == "Experiment"
            ):
                experiment_ids.add(producer)

    for node_id in context_row.get(
        "source_node_ids",
        [],
    ) or []:
        node_id = str(node_id)
        if (
            node_id in graph
            and str(
                graph.nodes[node_id].get(
                    "type",
                    "",
                )
            )
            == "MeasurementGroup"
        ):
            group_ids.add(node_id)

    if group_ids:
        return "measurement_group", tuple(
            sorted(group_ids)
        )
    if experiment_ids:
        return "experiment", tuple(
            sorted(experiment_ids)
        )
    return None


def measurement_conditions(
    graph: nx.Graph,
    measurement_id: str,
    identity_row: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for mention_id in measurement_mentions(
        measurement_id,
        identity_row,
    ):
        if mention_id not in graph:
            continue
        for condition in parse_conditions(
            graph.nodes[mention_id]
        ):
            name = normalize_condition_name(
                condition.get("name", "")
            )
            value = numeric_condition_value(condition)
            if not name or value is None:
                continue
            result.append(
                {
                    "source_measurement_id": mention_id,
                    "condition_name": name,
                    "value_numeric": value,
                    "unit": str(
                        condition.get("unit", "")
                    ).strip(),
                    "value_text": str(
                        condition.get(
                            "value_text",
                            "",
                        )
                    ).strip(),
                }
            )
    # exact duplicate structured conditions are irrelevant to a varied series
    unique: dict[
        tuple[str, float, str],
        dict[str, Any],
    ] = {}
    for row in result:
        key = (
            row["condition_name"],
            float(row["value_numeric"]),
            row["unit"],
        )
        unique[key] = row
    return list(unique.values())


def actual_trend_maps(
    trend_rows: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[str, list[Mapping[str, Any]]],
    dict[str, set[str]],
    dict[str, list[set[str]]],
]:
    by_paper: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    claim_ids: dict[str, set[str]] = defaultdict(set)
    numeric_measurement_sets: dict[
        str,
        list[set[str]],
    ] = defaultdict(list)

    for row in trend_rows:
        paper_id = str(row.get("paper_id", ""))
        if not paper_id:
            continue
        by_paper[paper_id].append(row)
        basis = str(row.get("evidence_basis", ""))
        if basis in CLAIM_TREND_BASES:
            for claim_id in row.get(
                "source_claim_ids",
                [],
            ) or []:
                claim_ids[paper_id].add(str(claim_id))
        if basis in NUMERIC_TREND_BASES:
            numeric_measurement_sets[paper_id].append(
                {
                    str(value)
                    for value in row.get(
                        "source_measurement_ids",
                        [],
                    )
                    or []
                    if str(value).strip()
                }
            )
    return (
        dict(by_paper),
        dict(claim_ids),
        dict(numeric_measurement_sets),
    )


def build_numeric_candidates(
    *,
    paper_id: str,
    graph: nx.Graph,
    identity_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    admitted_measurement_sets: list[set[str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    identities = identity_by_representative(
        identity_rows
    )
    methods = method_by_id(method_rows)

    funnel = Counter()
    grouped: dict[
        tuple[
            str,
            str,
            tuple[str, ...],
            str,
            str,
            str,
        ],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for context in comparison_rows:
        funnel["comparison_contexts"] += 1
        measurement_id = str(
            context.get("measurement_id", "")
        ).strip()
        value = context.get("value_numeric")
        if not measurement_id or value is None:
            continue
        try:
            dependent_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dependent_value):
            continue
        funnel["numeric_response_contexts"] += 1

        identity_row = identities.get(measurement_id)
        if identity_row is None:
            funnel["blocked_missing_identity"] += 1
            continue
        funnel["identity_bound"] += 1

        method_id = str(
            context.get("method_context_id", "")
        ).strip()
        method_row = methods.get(method_id)
        if method_row is None:
            funnel["blocked_missing_method_context"] += 1
            continue
        funnel["method_bound"] += 1

        lineage = measurement_lineage(
            graph,
            measurement_id,
            identity_row,
            method_row,
            context,
        )
        if lineage is None:
            funnel["blocked_missing_lineage"] += 1
            continue
        funnel["lineage_bound"] += 1
        lineage_kind, lineage_ids = lineage

        conditions = measurement_conditions(
            graph,
            measurement_id,
            identity_row,
        )
        if not conditions:
            funnel["blocked_no_numeric_structured_condition"] += 1
            continue
        funnel["contexts_with_numeric_structured_condition"] += 1

        observable_key = str(
            context.get("observable_key", "")
        ).strip()
        dependent_unit = str(
            context.get("unit", "")
        ).strip()
        for cond in conditions:
            funnel["numeric_condition_points"] += 1
            key = (
                observable_key,
                lineage_kind,
                lineage_ids,
                dependent_unit,
                cond["condition_name"],
                cond["unit"],
            )
            grouped[key].append(
                {
                    "measurement_id": measurement_id,
                    "method_context_id": method_id,
                    "method": method_row,
                    "dependent_value": dependent_value,
                    "dependent_unit": dependent_unit,
                    "condition_name": cond[
                        "condition_name"
                    ],
                    "condition_value": float(
                        cond["value_numeric"]
                    ),
                    "condition_unit": cond["unit"],
                    "source_expression": str(
                        context.get(
                            "source_expression",
                            "",
                        )
                    ).strip(),
                }
            )

    candidates: list[dict[str, Any]] = []
    for (
        observable_key,
        lineage_kind,
        lineage_ids,
        dependent_unit,
        condition_name,
        condition_unit,
    ), rows in sorted(
        grouped.items(),
        key=lambda item: str(item[0]),
    ):
        measurement_ids = {
            row["measurement_id"]
            for row in rows
        }
        x_values = {
            float(row["condition_value"])
            for row in rows
        }
        if len(measurement_ids) < 2:
            funnel["group_block_single_measurement"] += 1
            continue
        if len(x_values) < 2:
            funnel["group_block_no_varied_x"] += 1
            continue
        funnel["broad_numeric_series_candidates"] += 1

        varied_method_dimension = (
            condition_method_dimension(
                condition_name
            )
        )
        compatible, compatibility_reasons = (
            methods_compatible(
                (row["method"] for row in rows),
                varied_method_dimension=(
                    varied_method_dimension
                ),
            )
        )
        admitted = any(
            len(measurement_ids & admitted) >= 2
            for admitted in admitted_measurement_sets
        )
        reasons: list[str] = []
        if not compatible:
            reasons.extend(compatibility_reasons)
        if not admitted:
            reasons.append(
                "not_admitted_by_current_trend_adapter"
            )
        if not reasons:
            reasons.append(
                "admitted_or_equivalent_numeric_trend"
            )

        candidates.append(
            {
                "paper_id": paper_id,
                "observable_key": observable_key,
                "lineage_kind": lineage_kind,
                "lineage_ids": list(lineage_ids),
                "dependent_unit": dependent_unit,
                "condition_name": condition_name,
                "condition_unit": condition_unit,
                "varied_method_dimension": (
                    varied_method_dimension
                ),
                "measurement_ids": sorted(
                    measurement_ids
                ),
                "point_count": len(rows),
                "distinct_x_count": len(x_values),
                "method_compatible_except_varied_dimension": (
                    compatible
                ),
                "compatibility_reasons": (
                    compatibility_reasons
                ),
                "admitted_by_current_trend": admitted,
                "diagnostic_reasons": sorted(
                    set(reasons)
                ),
            }
        )

    funnel["broad_numeric_candidates_admitted"] = sum(
        row["admitted_by_current_trend"]
        for row in candidates
    )
    funnel["broad_numeric_candidates_missed"] = sum(
        not row["admitted_by_current_trend"]
        for row in candidates
    )
    return candidates, dict(sorted(funnel.items()))


def classify_paper(
    *,
    actual_precision_count: int,
    actual_trend_count: int,
    claim_candidates: list[dict[str, Any]],
    numeric_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_misses = [
        row
        for row in claim_candidates
        if not row["admitted_by_current_trend"]
    ]
    numeric_misses = [
        row
        for row in numeric_candidates
        if not row["admitted_by_current_trend"]
    ]
    precision_filtered = (
        actual_trend_count > 0
        and actual_precision_count == 0
    )

    if actual_precision_count > 0:
        primary = "D_current_trend_yield"
    elif numeric_misses and claim_misses:
        primary = "BC_claim_and_numeric_miss"
    elif numeric_misses:
        primary = "C_numeric_pipeline_block_candidate"
    elif claim_misses:
        primary = "B_claim_adapter_miss_candidate"
    elif precision_filtered:
        primary = "P_precision_filter"
    else:
        primary = "A_no_broad_directional_candidate"

    return {
        "primary_class": primary,
        "has_current_trend": actual_precision_count > 0,
        "raw_trend_evidence_count": actual_trend_count,
        "precision_local_result_count": actual_precision_count,
        "broad_claim_candidate_count": len(
            claim_candidates
        ),
        "claim_adapter_miss_count": len(
            claim_misses
        ),
        "broad_numeric_candidate_count": len(
            numeric_candidates
        ),
        "numeric_pipeline_miss_count": len(
            numeric_misses
        ),
        "precision_filtered": precision_filtered,
    }


def diagnose_development_trend_yield(
    *,
    paper_ids: list[str],
    canonical_paths: Mapping[str, Path],
    identity_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    trend_rows: list[dict[str, Any]],
    local_result_rows: list[dict[str, Any]],
    implementation_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Diagnostic paper_ids contain duplicates.")

    identities_by_paper: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in identity_rows:
        identities_by_paper[
            str(row.get("paper_id", ""))
        ].append(row)

    methods_by_paper: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in method_rows:
        methods_by_paper[
            str(row.get("paper_id", ""))
        ].append(row)

    comparison_by_paper: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in comparison_rows:
        comparison_by_paper[
            str(row.get("paper_id", ""))
        ].append(row)

    (
        trend_by_paper,
        admitted_claim_ids,
        admitted_numeric_sets,
    ) = actual_trend_maps(trend_rows)

    local_by_paper = Counter(
        str(row.get("paper_id", ""))
        for row in local_result_rows
        if str(row.get("paper_id", "")).strip()
    )

    claim_candidate_rows: list[
        dict[str, Any]
    ] = []
    numeric_candidate_rows: list[
        dict[str, Any]
    ] = []
    paper_rows: list[dict[str, Any]] = []
    global_numeric_funnel = Counter()

    for paper_id in paper_ids:
        graph_path = canonical_paths[paper_id]
        graph = nx.read_graphml(
            graph_path,
            force_multigraph=True,
        )

        paper_claim_candidates: list[
            dict[str, Any]
        ] = []
        for claim_id, attrs in sorted(
            graph.nodes(data=True),
            key=lambda item: str(item[0]),
        ):
            row = broad_claim_candidate(
                paper_id=paper_id,
                claim_id=str(claim_id),
                attrs=attrs,
                admitted_claim_ids=(
                    admitted_claim_ids.get(
                        paper_id,
                        set(),
                    )
                ),
            )
            if row is not None:
                paper_claim_candidates.append(row)
                claim_candidate_rows.append(row)

        paper_numeric_candidates, funnel = (
            build_numeric_candidates(
                paper_id=paper_id,
                graph=graph,
                identity_rows=identities_by_paper.get(
                    paper_id,
                    [],
                ),
                method_rows=methods_by_paper.get(
                    paper_id,
                    [],
                ),
                comparison_rows=comparison_by_paper.get(
                    paper_id,
                    [],
                ),
                admitted_measurement_sets=(
                    admitted_numeric_sets.get(
                        paper_id,
                        [],
                    )
                ),
            )
        )
        numeric_candidate_rows.extend(
            paper_numeric_candidates
        )
        global_numeric_funnel.update(funnel)

        classification = classify_paper(
            actual_precision_count=local_by_paper.get(
                paper_id,
                0,
            ),
            actual_trend_count=len(
                trend_by_paper.get(
                    paper_id,
                    [],
                )
            ),
            claim_candidates=paper_claim_candidates,
            numeric_candidates=paper_numeric_candidates,
        )
        paper_rows.append(
            {
                "paper_id": paper_id,
                **classification,
                "canonical_sha256": sha256_file(
                    graph_path
                ),
            }
        )

    primary_counts = Counter(
        row["primary_class"]
        for row in paper_rows
    )
    zero_yield = [
        row
        for row in paper_rows
        if row["precision_local_result_count"] == 0
    ]
    zero_yield_count = len(zero_yield)
    a_like = sum(
        row["primary_class"]
        == "A_no_broad_directional_candidate"
        for row in zero_yield
    )
    b_flag = sum(
        row["claim_adapter_miss_count"] > 0
        for row in zero_yield
    )
    c_flag = sum(
        row["numeric_pipeline_miss_count"] > 0
        for row in zero_yield
    )
    bc_overlap = sum(
        row["claim_adapter_miss_count"] > 0
        and row["numeric_pipeline_miss_count"] > 0
        for row in zero_yield
    )

    def ratio(value: int, denominator: int) -> float:
        return (
            float(value) / float(denominator)
            if denominator
            else 0.0
        )

    summary = {
        "diagnostic_id": ALPHA4C5G_DIAGNOSTIC_ID,
        "domain_profile_id": EXPECTED_DOMAIN_PROFILE_ID,
        "paper_count": len(paper_ids),
        "paper_ids": paper_ids,
        "development_only": True,
        "reserve_a_used": False,
        "reserve_b_used": False,
        "reserve_b_remains_sealed": True,
        "scientific_semantics_modified": False,
        "acceptance_semantics_modified": False,
        "count_thresholds_used_for_acceptance": False,
        "llm_calls": 0,
        "raw_trend_evidence_count": len(trend_rows),
        "precision_local_result_count": len(
            local_result_rows
        ),
        "papers_with_precision_trend": sum(
            row["precision_local_result_count"] > 0
            for row in paper_rows
        ),
        "zero_yield_paper_count": zero_yield_count,
        "primary_class_counts": dict(
            sorted(primary_counts.items())
        ),
        "zero_yield_diagnostic_flags": {
            "A_no_broad_candidate_count": a_like,
            "A_no_broad_candidate_ratio": ratio(
                a_like,
                zero_yield_count,
            ),
            "B_claim_miss_flag_count": b_flag,
            "B_claim_miss_flag_ratio": ratio(
                b_flag,
                zero_yield_count,
            ),
            "C_numeric_block_flag_count": c_flag,
            "C_numeric_block_flag_ratio": ratio(
                c_flag,
                zero_yield_count,
            ),
            "BC_overlap_count": bc_overlap,
            "BC_overlap_ratio": ratio(
                bc_overlap,
                zero_yield_count,
            ),
            "note": (
                "B and C flags are intentionally non-exclusive. "
                "primary_class_counts provides an exclusive partition."
            ),
        },
        "claim_candidate_counts": {
            "broad_candidates": len(
                claim_candidate_rows
            ),
            "admitted_by_current_trend": sum(
                row["admitted_by_current_trend"]
                for row in claim_candidate_rows
            ),
            "candidate_misses": sum(
                not row["admitted_by_current_trend"]
                for row in claim_candidate_rows
            ),
        },
        "numeric_candidate_counts": {
            "broad_series_candidates": len(
                numeric_candidate_rows
            ),
            "admitted_by_current_trend": sum(
                row["admitted_by_current_trend"]
                for row in numeric_candidate_rows
            ),
            "candidate_misses": sum(
                not row["admitted_by_current_trend"]
                for row in numeric_candidate_rows
            ),
        },
        "numeric_funnel": dict(
            sorted(global_numeric_funnel.items())
        ),
        "implementation_sha256": dict(
            sorted(implementation_hashes.items())
        ),
        "interpretation": {
            "A": (
                "No broad directional SERS/Raman claim candidate and "
                "no broad numeric varied-condition series candidate "
                "was observed by the diagnostic census."
            ),
            "B": (
                "At least one broad directional SERS/Raman claim "
                "candidate was present but was not admitted as current "
                "TrendEvidence. This is a recall-diagnostic flag, not "
                "automatic evidence that the adapter is wrong."
            ),
            "C": (
                "At least one broad numeric varied-condition series "
                "candidate was present but was not admitted as a "
                "current numeric TrendEvidence. Reasons are recorded "
                "without weakening provenance/context gates."
            ),
            "D": "Current Trend/Precision yielded a local result.",
            "P": (
                "Raw TrendEvidence existed but did not survive the "
                "current precision stage."
            ),
        },
    }
    return {
        "summary": summary,
        "papers": paper_rows,
        "claim_candidates": claim_candidate_rows,
        "numeric_candidates": numeric_candidate_rows,
    }
