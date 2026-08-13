from __future__ import annotations

import json
import re
from typing import Any, Iterable

import networkx as nx

from dac_her.reproducibility_domain import (
    ReproducibilityDomainAdapter,
    ReproducibilityEvidence,
)
from dac_her.reproducibility_evidence import (
    stable_reproducibility_result_id,
)


SERS_REPRODUCIBILITY_SEMANTICS_ID = (
    "sers_au_ag_reproducibility_v2_alpha4b3b4a1"
)

SERS_REPRODUCIBILITY_SCOPES = frozenset({
    "spot_to_spot",
    "substrate_to_substrate",
    "batch_to_batch",
    "particle_to_particle",
    "replicate",
    "unknown",
})

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_REPRODUCIBILITY_MARKER_RE = re.compile(
    r"\b(?:reproducib\w*|repeatab\w*|batch[- ]to[- ]batch)\b",
    re.I,
)
_EXPLICIT_DISPERSION_RE = re.compile(
    r"(?:"
    r"\b(?:rsd|relative\s+standard\s+deviation|"
    r"coefficient\s+of\s+variation|cv|deviation)\b"
    r".{0,40}\b\d+(?:\.\d+)?\s*%"
    r"|"
    r"\b\d+(?:\.\d+)?\s*%\b.{0,40}"
    r"\b(?:rsd|relative\s+standard\s+deviation|"
    r"coefficient\s+of\s+variation|cv|deviation)\b"
    r")",
    re.I,
)
_SPATIAL_REPLICATE_RE = re.compile(
    r"\b(?:average|mean)\s+of\s+"
    r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:raman\s+|sers\s+)?spectra\b.{0,100}"
    r"\b(?:different|distinct)\b.{0,50}"
    r"\b(?:positions?|spots?|sites?|locations?)\b",
    re.I,
)
_POPULATION_RE = re.compile(
    r"\b(?P<count>\d+)\s+(?:individual\s+)?"
    r"(?:particles?|nanoparticles?|nps?|dips?)\b",
    re.I,
)


def _relation(attrs: dict[str, Any]) -> str:
    return str(attrs.get("relation", "")).strip()


def _incoming(graph: nx.Graph, node_id: str, relation: str) -> list[str]:
    if not graph.is_directed():
        return []
    if graph.is_multigraph():
        iterator = graph.in_edges(node_id, keys=True, data=True)
        return sorted({
            str(left)
            for left, _right, _key, attrs in iterator
            if _relation(dict(attrs)) == relation
        })
    return sorted({
        str(left)
        for left, _right, attrs in graph.in_edges(node_id, data=True)
        if _relation(dict(attrs)) == relation
    })


def _outgoing(graph: nx.Graph, node_id: str, relation: str) -> list[str]:
    if not graph.is_directed():
        return []
    if graph.is_multigraph():
        iterator = graph.out_edges(node_id, keys=True, data=True)
        return sorted({
            str(right)
            for _left, right, _key, attrs in iterator
            if _relation(dict(attrs)) == relation
        })
    return sorted({
        str(right)
        for _left, right, attrs in graph.out_edges(node_id, data=True)
        if _relation(dict(attrs)) == relation
    })


def _node_text(graph: nx.Graph, node_id: str) -> str:
    attrs = graph.nodes[node_id]
    keys = (
        "label",
        "source_expression",
        "description",
        "qualifier",
        "value_text",
        "raw_method_name",
    )
    return " ".join(
        str(attrs.get(key, "")).strip()
        for key in keys
        if str(attrs.get(key, "")).strip()
    )


def _source_text(graph: nx.Graph, node_ids: Iterable[str]) -> str:
    return " ".join(
        _node_text(graph, node_id)
        for node_id in sorted(set(map(str, node_ids)))
        if node_id in graph
    ).strip()


def _parse_conditions(graph: nx.Graph, node_id: str) -> list[dict[str, Any]]:
    if node_id not in graph:
        return []
    raw = str(graph.nodes[node_id].get("conditions_json", "")).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed conditions_json on node {node_id!r}."
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(
            f"conditions_json on node {node_id!r} must be a list."
        )
    rows: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(
                f"conditions_json item on node {node_id!r} must be an object."
            )
        rows.append(dict(item))
    return rows


def _positive_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 1 or not number.is_integer():
        return None
    return int(number)


def _count_from_conditions(
    graph: nx.Graph,
    node_ids: Iterable[str],
    names: frozenset[str],
) -> int | None:
    values: set[int] = set()
    for node_id in sorted(set(map(str, node_ids))):
        for condition in _parse_conditions(graph, node_id):
            name = str(condition.get("name", "")).strip().lower()
            if name not in names:
                continue
            number = _positive_int(condition.get("value_numeric"))
            if number is not None:
                values.add(number)
    if len(values) == 1:
        return next(iter(values))
    return None


def _word_or_int(value: str) -> int | None:
    text = value.strip().lower()
    if text.isdigit():
        return int(text)
    return _WORD_NUMBERS.get(text)


def _scope(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bbatch[- ]to[- ]batch\b|\bdifferent\s+\w*\s*batches\b|\bacross\s+\w*\s*batches\b", lowered):
        return "batch_to_batch"
    if re.search(r"\bsubstrate[- ]to[- ]substrate\b|\bdifferent\s+substrates\b|\bacross\s+substrates\b", lowered):
        return "substrate_to_substrate"
    if re.search(r"\b(?:individual\s+)?(?:particles?|nanoparticles?|nps?|dips?)\b", lowered) and (
        "distribution" in lowered
        or "population" in lowered
        or "single-particle" in lowered
        or "single particle" in lowered
    ):
        return "particle_to_particle"
    if re.search(r"\bdifferent\b.{0,50}\b(?:positions?|spots?|sites?|locations?)\b", lowered):
        return "spot_to_spot"
    if re.search(r"\breplicat\w*\b|\brepeat(?:ed|s|ing)?\b|\baverage\s+of\s+\w+\s+(?:raman\s+|sers\s+)?spectra\b", lowered):
        return "replicate"
    return "unknown"


def _counts(graph: nx.Graph, node_ids: Iterable[str], text: str) -> dict[str, int | None]:
    ids = tuple(sorted(set(map(str, node_ids))))
    result: dict[str, int | None] = {
        "n_spots": _count_from_conditions(
            graph,
            ids,
            frozenset({"number of spots", "number of positions", "number of sites", "measurement positions"}),
        ),
        "n_substrates": _count_from_conditions(
            graph,
            ids,
            frozenset({"number of substrates", "substrates measured"}),
        ),
        "n_batches": _count_from_conditions(
            graph,
            ids,
            frozenset({"number of batches", "batches measured"}),
        ),
        "n_replicates": _count_from_conditions(
            graph,
            ids,
            frozenset({"number of replicates", "replicates", "number of spectra", "spectra averaged"}),
        ),
        "n_particles": _count_from_conditions(
            graph,
            ids,
            frozenset({"measurement population", "population measured", "number of particles", "particles measured"}),
        ),
    }

    spatial = _SPATIAL_REPLICATE_RE.search(text)
    if spatial:
        number = _word_or_int(spatial.group("count"))
        if number is not None:
            if result["n_replicates"] is None:
                result["n_replicates"] = number
            if result["n_spots"] is None:
                result["n_spots"] = number

    population = _POPULATION_RE.search(text)
    if population and result["n_particles"] is None:
        result["n_particles"] = int(population.group("count"))

    batch = re.search(
        r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:different\s+)?(?:\w+\s+){0,3}batches\b",
        text,
        re.I,
    )
    if batch and result["n_batches"] is None:
        result["n_batches"] = _word_or_int(batch.group("count"))

    return result


def _explicit_text_attr(
    graph: nx.Graph,
    node_ids: Iterable[str],
    keys: tuple[str, ...],
) -> str:
    values: set[str] = set()
    for node_id in sorted(set(map(str, node_ids))):
        if node_id not in graph:
            continue
        attrs = graph.nodes[node_id]
        for key in keys:
            value = str(attrs.get(key, "")).strip()
            if value:
                values.add(value)
    if len(values) == 1:
        return next(iter(values))
    return ""


def _measurement_value(
    graph: nx.Graph,
    measurement_id: str,
) -> tuple[float | None, str, str]:
    attrs = graph.nodes[measurement_id]
    numeric_raw = str(attrs.get("value_numeric", "")).strip()
    text_raw = str(attrs.get("value_text", "")).strip()
    if numeric_raw and text_raw:
        raise ValueError(
            f"Measurement {measurement_id!r} violates numeric/text XOR."
        )
    value_numeric: float | None = None
    if numeric_raw:
        try:
            value_numeric = float(numeric_raw)
        except ValueError as exc:
            raise ValueError(
                f"Measurement {measurement_id!r} has non-numeric value_numeric."
            ) from exc
    return value_numeric, text_raw, str(attrs.get("unit", "")).strip()



def _rsd_evidence_kind(
    *,
    source_text: str,
    value_numeric: float | None,
    value_text: str,
) -> str:
    if value_numeric is not None:
        return "relative_standard_deviation"
    explicit_surface = " ".join(
        value
        for value in (value_text.strip(), source_text.strip())
        if value
    )
    if _EXPLICIT_DISPERSION_RE.search(explicit_surface):
        return "relative_standard_deviation"
    return "repeatability_statement"


def _measurement_subject_ids(
    graph: nx.Graph,
    measurement_ids: Iterable[str],
) -> set[str]:
    subjects: set[str] = set()
    for measurement_id in sorted(set(map(str, measurement_ids))):
        if measurement_id not in graph:
            continue
        subject_id = str(
            graph.nodes[measurement_id].get("subject_id", "")
        ).strip()
        if subject_id:
            subjects.add(subject_id)
        subjects.update(_outgoing(graph, measurement_id, "MEASURED_FOR"))
    return subjects


def _compatible_optional_scalar(left: Any, right: Any) -> bool:
    return left is None or right is None or left == right


def _compatible_optional_text(left: str, right: str) -> bool:
    return not left or not right or left == right


def _optional_result_metadata_compatible(
    left: ReproducibilityEvidence,
    right: ReproducibilityEvidence,
) -> bool:
    return (
        _compatible_optional_scalar(left.n_spots, right.n_spots)
        and _compatible_optional_scalar(
            left.n_substrates,
            right.n_substrates,
        )
        and _compatible_optional_scalar(left.n_batches, right.n_batches)
        and _compatible_optional_scalar(
            left.n_replicates,
            right.n_replicates,
        )
        and _compatible_optional_scalar(
            left.n_particles,
            right.n_particles,
        )
        and _compatible_optional_text(
            left.mapping_area,
            right.mapping_area,
        )
        and _compatible_optional_text(
            left.internal_standard,
            right.internal_standard,
        )
    )


def _same_result_payload(
    left: ReproducibilityEvidence,
    right: ReproducibilityEvidence,
) -> bool:
    if (
        left.paper_id != right.paper_id
        or left.evidence_kind != right.evidence_kind
        or left.reproducibility_scope != right.reproducibility_scope
    ):
        return False
    if not _optional_result_metadata_compatible(left, right):
        return False
    return (
        left.value_numeric == right.value_numeric
        and left.value_text == right.value_text
        and left.unit.strip().lower() == right.unit.strip().lower()
    )


def _shared_exact_lineage(
    left: ReproducibilityEvidence,
    right: ReproducibilityEvidence,
) -> bool:
    return bool(
        set(left.source_measurement_group_ids)
        & set(right.source_measurement_group_ids)
    ) or bool(
        set(left.source_experiment_ids)
        & set(right.source_experiment_ids)
    )


def _choose_optional_scalar(left: Any, right: Any) -> Any:
    if left is not None:
        return left
    return right


def _choose_optional_text(left: str, right: str) -> str:
    return left or right


def _merge_exact_result_mentions(
    *,
    graph: nx.Graph,
    left: ReproducibilityEvidence,
    right: ReproducibilityEvidence,
) -> ReproducibilityEvidence | None:
    if left.paper_id != right.paper_id:
        return None
    if left.reproducibility_scope != right.reproducibility_scope:
        return None
    if not _shared_exact_lineage(left, right):
        return None
    if not _optional_result_metadata_compatible(left, right):
        return None

    same_kind = _same_result_payload(left, right)
    cross_kind = {
        left.evidence_kind,
        right.evidence_kind,
    } == {"relative_standard_deviation", "repeatability_statement"}
    if not same_kind and not cross_kind:
        return None

    left_subjects = _measurement_subject_ids(
        graph,
        left.source_measurement_ids,
    )
    right_subjects = _measurement_subject_ids(
        graph,
        right.source_measurement_ids,
    )
    if left_subjects and right_subjects and left_subjects != right_subjects:
        return None

    mentions = tuple(sorted(
        set(left.source_mention_node_ids)
        | set(right.source_mention_node_ids)
    ))
    if len(mentions) < 2:
        return None

    source_expressions = tuple(sorted({
        value
        for value in (
            *left.source_expressions,
            *right.source_expressions,
            left.source_expression,
            right.source_expression,
        )
        if value.strip()
    }))
    source_node_ids = tuple(sorted(
        set(left.source_node_ids) | set(right.source_node_ids)
    ))
    source_measurements = tuple(sorted(
        set(left.source_measurement_ids)
        | set(right.source_measurement_ids)
    ))
    source_groups = tuple(sorted(
        set(left.source_measurement_group_ids)
        | set(right.source_measurement_group_ids)
    ))
    source_experiments = tuple(sorted(
        set(left.source_experiment_ids)
        | set(right.source_experiment_ids)
    ))

    primary = left
    if cross_kind and right.evidence_kind == "relative_standard_deviation":
        primary = right

    return ReproducibilityEvidence(
        evidence_id=stable_reproducibility_result_id(
            paper_id=left.paper_id,
            evidence_kind=primary.evidence_kind,
            source_mention_node_ids=mentions,
        ),
        domain_profile_id=left.domain_profile_id,
        reproducibility_semantics_id=left.reproducibility_semantics_id,
        paper_id=left.paper_id,
        evidence_kind=primary.evidence_kind,
        reproducibility_scope=left.reproducibility_scope,
        value_numeric=primary.value_numeric,
        value_text=primary.value_text,
        unit=primary.unit,
        n_spots=_choose_optional_scalar(left.n_spots, right.n_spots),
        n_substrates=_choose_optional_scalar(
            left.n_substrates,
            right.n_substrates,
        ),
        n_batches=_choose_optional_scalar(
            left.n_batches,
            right.n_batches,
        ),
        n_replicates=_choose_optional_scalar(
            left.n_replicates,
            right.n_replicates,
        ),
        n_particles=_choose_optional_scalar(
            left.n_particles,
            right.n_particles,
        ),
        mapping_area=_choose_optional_text(
            left.mapping_area,
            right.mapping_area,
        ),
        internal_standard=_choose_optional_text(
            left.internal_standard,
            right.internal_standard,
        ),
        result_identity_status="consolidated_exact",
        source_expression=(
            source_expressions[0] if source_expressions else ""
        ),
        source_expressions=source_expressions,
        source_mention_node_ids=mentions,
        source_measurement_ids=source_measurements,
        source_measurement_group_ids=source_groups,
        source_experiment_ids=source_experiments,
        source_node_ids=source_node_ids,
    )


def _consolidate_exact_results(
    graph: nx.Graph,
    evidence: list[ReproducibilityEvidence],
) -> list[ReproducibilityEvidence]:
    pending = sorted(evidence, key=lambda item: item.evidence_id)
    changed = True
    while changed:
        changed = False
        next_rows: list[ReproducibilityEvidence] = []
        consumed: set[int] = set()
        for left_index, left in enumerate(pending):
            if left_index in consumed:
                continue
            current = left
            for right_index in range(left_index + 1, len(pending)):
                if right_index in consumed:
                    continue
                merged = _merge_exact_result_mentions(
                    graph=graph,
                    left=current,
                    right=pending[right_index],
                )
                if merged is None:
                    continue
                current = merged
                consumed.add(right_index)
                changed = True
            next_rows.append(current)
        pending = sorted(next_rows, key=lambda item: item.evidence_id)
    return pending


def _typed_sources(graph: nx.Graph, node_ids: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    measurements: list[str] = []
    groups: list[str] = []
    experiments: list[str] = []
    for node_id in sorted(set(map(str, node_ids))):
        if node_id not in graph:
            continue
        node_type = str(graph.nodes[node_id].get("type", ""))
        if node_type == "Measurement":
            measurements.append(node_id)
        elif node_type == "MeasurementGroup":
            groups.append(node_id)
        elif node_type == "Experiment":
            experiments.append(node_id)
    return tuple(measurements), tuple(groups), tuple(experiments)


def _make_evidence(
    *,
    graph: nx.Graph,
    paper_id: str,
    primary_node_id: str,
    evidence_kind: str,
    source_node_ids: Iterable[str],
    value_numeric: float | None = None,
    value_text: str = "",
    unit: str = "",
) -> ReproducibilityEvidence:
    source_ids = tuple(sorted(set(map(str, source_node_ids))))
    text = _source_text(graph, source_ids)
    counts = _counts(graph, source_ids, text)
    measurements, groups, experiments = _typed_sources(graph, source_ids)
    source_expression = str(
        graph.nodes[primary_node_id].get("source_expression", "")
        or graph.nodes[primary_node_id].get("description", "")
        or graph.nodes[primary_node_id].get("label", "")
    ).strip()
    return ReproducibilityEvidence(
        evidence_id=stable_reproducibility_result_id(
            paper_id=paper_id,
            evidence_kind=evidence_kind,
            source_mention_node_ids=(primary_node_id,),
        ),
        domain_profile_id="sers_au_ag",
        reproducibility_semantics_id=SERS_REPRODUCIBILITY_SEMANTICS_ID,
        paper_id=paper_id,
        evidence_kind=evidence_kind,
        reproducibility_scope=_scope(text),
        value_numeric=value_numeric,
        value_text=value_text,
        unit=unit,
        n_spots=counts["n_spots"],
        n_substrates=counts["n_substrates"],
        n_batches=counts["n_batches"],
        n_replicates=counts["n_replicates"],
        n_particles=counts["n_particles"],
        mapping_area=_explicit_text_attr(
            graph,
            source_ids,
            ("mapping_area", "map_area", "mapping_area_text"),
        ),
        internal_standard=_explicit_text_attr(
            graph,
            source_ids,
            ("internal_standard",),
        ),
        result_identity_status="single_mention",
        source_expression=source_expression,
        source_expressions=(
            (source_expression,) if source_expression else ()
        ),
        source_mention_node_ids=(primary_node_id,),
        source_measurement_ids=measurements,
        source_measurement_group_ids=groups,
        source_experiment_ids=experiments,
        source_node_ids=source_ids,
    )


def _is_sers_quality_experiment(graph: nx.Graph, node_id: str) -> bool:
    if node_id not in graph:
        return False
    attrs = graph.nodes[node_id]
    if str(attrs.get("type", "")) != "Experiment":
        return False
    method_surface = " ".join(
        str(attrs.get(key, ""))
        for key in (
            "label",
            "experiment_type",
            "method_label",
            "raw_method_name",
        )
    ).lower()
    return bool(re.search(r"\b(?:sers|raman)\b", method_surface, re.I))


def _is_rsd_measurement(graph: nx.Graph, node_id: str) -> bool:
    attrs = graph.nodes[node_id]
    if str(attrs.get("type", "")) != "Measurement":
        return False
    metric = str(
        attrs.get("metric_id")
        or attrs.get("metric")
        or attrs.get("label")
        or ""
    ).strip().lower().replace(" ", "_")
    return metric == "relative_standard_deviation"


def extract_sers_reproducibility_evidence(
    graph: nx.Graph,
    paper_id: str,
) -> list[ReproducibilityEvidence]:
    evidence: list[ReproducibilityEvidence] = []
    rsd_producers: set[str] = set()
    rsd_groups: set[str] = set()

    # Quantitative RSD is the strongest directly represented quality evidence.
    # Enrich it only with producer/group context already connected in the graph.
    for node_id in sorted(map(str, graph.nodes)):
        if not _is_rsd_measurement(graph, node_id):
            continue
        producers = _incoming(graph, node_id, "HAS_MEASUREMENT")
        groups = _outgoing(graph, node_id, "IN_MEASUREMENT_GROUP")
        rsd_producers.update(producers)
        rsd_groups.update(groups)
        value_numeric, value_text, unit = _measurement_value(graph, node_id)
        source_text = _node_text(graph, node_id)
        evidence_kind = _rsd_evidence_kind(
            source_text=source_text,
            value_numeric=value_numeric,
            value_text=value_text,
        )
        evidence.append(
            _make_evidence(
                graph=graph,
                paper_id=paper_id,
                primary_node_id=node_id,
                evidence_kind=evidence_kind,
                source_node_ids=(node_id, *producers, *groups),
                value_numeric=(
                    value_numeric
                    if evidence_kind == "relative_standard_deviation"
                    else None
                ),
                value_text=(
                    value_text
                    if evidence_kind == "relative_standard_deviation"
                    else ""
                ),
                unit=(
                    unit
                    if evidence_kind == "relative_standard_deviation"
                    else ""
                ),
            )
        )

    # Measurement-local qualitative reproducibility is admissible when the
    # source itself says reproducibility/repeatability. No numeric value is
    # invented from words such as "good" or "high".
    for node_id in sorted(map(str, graph.nodes)):
        attrs = graph.nodes[node_id]
        if str(attrs.get("type", "")) != "Measurement":
            continue
        if _is_rsd_measurement(graph, node_id):
            continue
        text = _node_text(graph, node_id)
        if not _REPRODUCIBILITY_MARKER_RE.search(text):
            continue
        producers = _incoming(graph, node_id, "HAS_MEASUREMENT")
        groups = _outgoing(graph, node_id, "IN_MEASUREMENT_GROUP")
        evidence.append(
            _make_evidence(
                graph=graph,
                paper_id=paper_id,
                primary_node_id=node_id,
                evidence_kind="repeatability_statement",
                source_node_ids=(node_id, *producers, *groups),
            )
        )

    # Experiment-level evidence is limited to explicit reproducibility language,
    # explicit spatial replicate averaging, or an explicit sampled population.
    # Mere words such as "uniform" do not create evidence.
    for node_id in sorted(map(str, graph.nodes)):
        attrs = graph.nodes[node_id]
        if not _is_sers_quality_experiment(graph, node_id):
            continue
        text = _node_text(graph, node_id)
        spatial = _SPATIAL_REPLICATE_RE.search(text)
        population = _POPULATION_RE.search(text) if (
            "distribution" in text.lower()
            or "population" in text.lower()
            or "single-particle" in text.lower()
            or "single particle" in text.lower()
        ) else None

        if spatial:
            evidence_kind = "spatial_sampling"
        elif population:
            evidence_kind = "population_sampling"
        elif _REPRODUCIBILITY_MARKER_RE.search(text):
            # If this Experiment already provides the context for a direct RSD,
            # the RSD evidence carries the producer provenance and no duplicate
            # qualitative record is necessary.
            if node_id in rsd_producers:
                continue
            evidence_kind = "repeatability_statement"
        else:
            continue

        evidence.append(
            _make_evidence(
                graph=graph,
                paper_id=paper_id,
                primary_node_id=node_id,
                evidence_kind=evidence_kind,
                source_node_ids=(node_id,),
            )
        )

    # A standalone MeasurementGroup can preserve explicit group-level
    # reproducibility when no direct RSD already references that group.
    for node_id in sorted(map(str, graph.nodes)):
        attrs = graph.nodes[node_id]
        if str(attrs.get("type", "")) != "MeasurementGroup":
            continue
        if node_id in rsd_groups:
            continue
        text = _node_text(graph, node_id)
        if not _REPRODUCIBILITY_MARKER_RE.search(text):
            continue
        evidence.append(
            _make_evidence(
                graph=graph,
                paper_id=paper_id,
                primary_node_id=node_id,
                evidence_kind="repeatability_statement",
                source_node_ids=(node_id,),
            )
        )

    return _consolidate_exact_results(graph, evidence)


SERS_AU_AG_REPRODUCIBILITY_ADAPTER = ReproducibilityDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    semantics_id=SERS_REPRODUCIBILITY_SEMANTICS_ID,
    scope_labels=SERS_REPRODUCIBILITY_SCOPES,
    extract_evidence_fn=extract_sers_reproducibility_evidence,
)
