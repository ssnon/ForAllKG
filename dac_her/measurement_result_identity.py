from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import networkx as nx

from dac_her.comparison_domain import (
    ComparisonContext,
    ComparisonDimensionValue,
)
from dac_her.method_context import (
    MethodContext,
    MethodDimensionValue,
)
from dac_her.metric_definition_domain import MetricDefinitionContext
from dac_her.semantic_repairs import node_composition_signature


MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID = (
    "measurement_result_identity_v1_alpha4b4a1"
)

MeasurementResultIdentityStatus = Literal[
    "single_mention",
    "consolidated_exact",
]

_STRUCTURAL_ROLE_TOKENS = frozenset({
    "alloy",
    "core",
    "shell",
    "dimer",
    "nanoplate",
    "nanorod",
    "nanowire",
    "nanostar",
    "nanocube",
    "film",
    "array",
    "island",
    "hollow",
    "porous",
})


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\bnps?\b", " nanoparticle ", text)
    text = re.sub(r"\bnanoparticles?\b", " nanoparticle ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _unit_key(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("μ", "u").replace("µ", "u")
    text = re.sub(r"\s+", "", text)
    return text


def _origin_local_id(node_id: str, attrs: Mapping[str, Any]) -> str:
    return str(attrs.get("source_local_id") or node_id).strip()


def _numeric_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    except (TypeError, ValueError):
        return False


def _value_signature(attrs: Mapping[str, Any]) -> tuple[str, str, str]:
    numeric = attrs.get("value_numeric")
    text = attrs.get("value_text")
    has_numeric = not _blank(numeric)
    has_text = not _blank(text)
    if has_numeric == has_text:
        return ("invalid", str(numeric or ""), str(text or ""))
    if has_numeric:
        return ("numeric", str(float(numeric)), _unit_key(attrs.get("unit")))
    return ("text", _norm_text(text), _unit_key(attrs.get("unit")))


def _condition_payload(condition: Mapping[str, Any]) -> tuple[str, str, str, str]:
    numeric = condition.get("value_numeric")
    text = condition.get("value_text")
    if not _blank(numeric):
        value_kind = "numeric"
        try:
            value = str(float(numeric))
        except (TypeError, ValueError):
            value = str(numeric)
    else:
        value_kind = "text"
        value = _norm_text(text)
    return (
        value_kind,
        value,
        _unit_key(condition.get("unit")),
        _norm_text(condition.get("reference")),
    )


def _conditions(attrs: Mapping[str, Any]) -> dict[str, set[tuple[str, str, str, str]]]:
    raw = attrs.get("conditions_json", "[]")
    if isinstance(raw, list):
        rows = raw
    else:
        try:
            rows = json.loads(str(raw or "[]"))
        except (TypeError, json.JSONDecodeError):
            rows = []
    result: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _norm_text(row.get("name"))
        if not name:
            continue
        result[name].add(_condition_payload(row))
    return dict(result)


def _conditions_compatible(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    left_conditions = _conditions(left)
    right_conditions = _conditions(right)
    conflicts = []
    for name in sorted(set(left_conditions) & set(right_conditions)):
        if left_conditions[name].isdisjoint(right_conditions[name]):
            conflicts.append(name)
    return (not conflicts, tuple(conflicts))


def _explicit_scalar_compatible(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    key: str,
) -> bool:
    left_value = _norm_text(left.get(key))
    right_value = _norm_text(right.get(key))
    if left_value and right_value and left_value != right_value:
        return False
    return True


def _subject_role_tokens(label: str) -> frozenset[str]:
    tokens = set(re.findall(r"[a-z0-9]+", _norm_text(label)))
    return frozenset(tokens & _STRUCTURAL_ROLE_TOKENS)


def _subject_compatibility(
    graph: nx.Graph,
    left_subject: str,
    right_subject: str,
) -> tuple[bool, str]:
    if left_subject == right_subject and left_subject:
        return True, "same_subject_id"
    if not left_subject or not right_subject:
        return False, "subject_missing"
    if left_subject not in graph or right_subject not in graph:
        return False, "subject_node_missing"

    left = graph.nodes[left_subject]
    right = graph.nodes[right_subject]
    left_type = str(left.get("type", "")).strip()
    right_type = str(right.get("type", "")).strip()
    if not left_type or left_type != right_type:
        return False, "subject_type_mismatch"

    left_label = _norm_text(left.get("label"))
    right_label = _norm_text(right.get("label"))
    if left_label and left_label == right_label:
        return True, "same_subject_label"

    left_signature = node_composition_signature(graph, left_subject)
    right_signature = node_composition_signature(graph, right_subject)
    if not left_signature or left_signature != right_signature:
        return False, "subject_identity_unresolved"

    left_roles = _subject_role_tokens(left_label)
    right_roles = _subject_role_tokens(right_label)
    if left_roles and right_roles and left_roles != right_roles:
        return False, "subject_structural_role_mismatch"

    # Same local Measurement lineage + same composition + no explicit
    # structural-role contradiction is sufficient subject evidence. One
    # mention may simply have a less specific subject label.
    return True, "same_composition_no_structural_conflict"


@dataclass(frozen=True)
class MeasurementResultIdentity:
    identity_id: str
    semantics_id: str
    paper_id: str
    representative_measurement_id: str
    source_mention_ids: tuple[str, ...]
    origin_local_id: str
    status: MeasurementResultIdentityStatus
    consolidation_reasons: tuple[str, ...] = ()
    subject_compatibility: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.identity_id.strip():
            raise ValueError("identity_id must not be empty.")
        if self.semantics_id != MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID:
            raise ValueError("Unexpected measurement-result identity semantics.")
        if not self.paper_id.strip():
            raise ValueError("paper_id must not be empty.")
        if not self.source_mention_ids:
            raise ValueError("source_mention_ids must not be empty.")
        if len(self.source_mention_ids) != len(set(self.source_mention_ids)):
            raise ValueError("source_mention_ids must be unique.")
        if self.representative_measurement_id not in self.source_mention_ids:
            raise ValueError("representative must be one of source_mention_ids.")
        if self.status == "single_mention" and len(self.source_mention_ids) != 1:
            raise ValueError("single_mention identity must have exactly one mention.")
        if self.status == "consolidated_exact" and len(self.source_mention_ids) < 2:
            raise ValueError("consolidated_exact requires at least two mentions.")

    @property
    def source_mention_count(self) -> int:
        return len(self.source_mention_ids)

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["source_mention_ids"] = list(self.source_mention_ids)
        row["consolidation_reasons"] = list(self.consolidation_reasons)
        row["subject_compatibility"] = list(self.subject_compatibility)
        row["source_mention_count"] = self.source_mention_count
        return row


@dataclass(frozen=True)
class MeasurementIdentityCandidate:
    paper_id: str
    origin_local_id: str
    source_mention_ids: tuple[str, ...]
    exact_consolidation_allowed: bool
    blockers: tuple[str, ...]
    subject_compatibility: tuple[str, ...]

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["source_mention_ids"] = list(self.source_mention_ids)
        row["blockers"] = list(self.blockers)
        row["subject_compatibility"] = list(self.subject_compatibility)
        return row


@dataclass(frozen=True)
class MeasurementResultIdentityAudit:
    semantics_id: str
    paper_count: int
    source_mention_count: int
    scientific_result_count: int
    single_mention_result_count: int
    consolidated_exact_result_count: int
    consolidated_source_mention_count: int
    unresolved_same_lineage_group_count: int
    issues: tuple[str, ...]
    structural_gate: bool

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = list(self.issues)
        return row


def _stable_identity_id(
    *,
    paper_id: str,
    representative_measurement_id: str,
    source_mention_ids: Iterable[str],
) -> str:
    payload = "|".join(
        (
            MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
            paper_id,
            representative_measurement_id,
            *sorted(source_mention_ids),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"measurement-result:{digest}"


def _representative_score(
    graph: nx.Graph,
    measurement_id: str,
) -> tuple[int, int, int, int, int, str]:
    attrs = graph.nodes[measurement_id]
    conditions = sum(len(values) for values in _conditions(attrs).values())
    group_links = sum(
        1
        for _, _, edge in graph.out_edges(measurement_id, data=True)
        if str(edge.get("relation", "")) == "IN_MEASUREMENT_GROUP"
    )
    qualifier = int(bool(_norm_text(attrs.get("qualifier"))))
    non_collision = int(not bool(str(attrs.get("id_collision_reason", "")).strip()))
    source_length = len(str(attrs.get("source_expression", "")))
    return (
        non_collision,
        conditions,
        group_links,
        qualifier,
        source_length,
        measurement_id,
    )


def _pair_exact_compatibility(
    graph: nx.Graph,
    left_id: str,
    right_id: str,
) -> tuple[bool, tuple[str, ...], str]:
    left = graph.nodes[left_id]
    right = graph.nodes[right_id]
    blockers: list[str] = []

    left_metric = str(left.get("metric_id", "")).strip()
    right_metric = str(right.get("metric_id", "")).strip()
    if not left_metric or left_metric != right_metric:
        blockers.append("metric_id_mismatch")

    left_value = _value_signature(left)
    right_value = _value_signature(right)
    if left_value[0] == "invalid" or right_value[0] == "invalid":
        blockers.append("invalid_measurement_value_payload")
    elif left_value[0] != right_value[0]:
        blockers.append("value_representation_mismatch")
    elif left_value[0] == "numeric":
        if not _numeric_equal(left.get("value_numeric"), right.get("value_numeric")):
            blockers.append("numeric_value_mismatch")
        left_unit = _unit_key(left.get("unit"))
        right_unit = _unit_key(right.get("unit"))
        if not left_unit or not right_unit:
            blockers.append("numeric_unit_missing")
        elif left_unit != right_unit:
            blockers.append("numeric_unit_mismatch")
    elif left_value[1:] != right_value[1:]:
        blockers.append("text_value_mismatch")

    if not _explicit_scalar_compatible(left, right, "basis"):
        blockers.append("basis_mismatch")
    if not _explicit_scalar_compatible(left, right, "qualifier"):
        blockers.append("qualifier_mismatch")

    conditions_ok, condition_conflicts = _conditions_compatible(left, right)
    if not conditions_ok:
        blockers.extend(
            f"condition_mismatch:{name}"
            for name in condition_conflicts
        )

    left_subject = str(left.get("subject_id", "")).strip()
    right_subject = str(right.get("subject_id", "")).strip()
    subject_ok, subject_reason = _subject_compatibility(
        graph,
        left_subject,
        right_subject,
    )
    if not subject_ok:
        blockers.append(subject_reason)

    return (not blockers, tuple(sorted(set(blockers))), subject_reason)


def build_measurement_result_identities(
    graph: nx.Graph,
    paper_id: str,
) -> tuple[
    list[MeasurementResultIdentity],
    list[MeasurementIdentityCandidate],
]:
    measurements = {
        str(node_id): dict(attrs)
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("type", "")) == "Measurement"
    }
    by_origin: dict[str, list[str]] = defaultdict(list)
    for measurement_id, attrs in measurements.items():
        by_origin[_origin_local_id(measurement_id, attrs)].append(
            measurement_id
        )

    identities: list[MeasurementResultIdentity] = []
    candidates: list[MeasurementIdentityCandidate] = []

    for origin_local_id, member_ids in sorted(by_origin.items()):
        members = tuple(sorted(member_ids))
        if len(members) == 1:
            representative = members[0]
            identities.append(
                MeasurementResultIdentity(
                    identity_id=_stable_identity_id(
                        paper_id=paper_id,
                        representative_measurement_id=representative,
                        source_mention_ids=members,
                    ),
                    semantics_id=MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
                    paper_id=paper_id,
                    representative_measurement_id=representative,
                    source_mention_ids=members,
                    origin_local_id=origin_local_id,
                    status="single_mention",
                )
            )
            continue

        pair_blockers: set[str] = set()
        subject_reasons: set[str] = set()
        for left_index, left_id in enumerate(members):
            for right_id in members[left_index + 1 :]:
                allowed, blockers, subject_reason = _pair_exact_compatibility(
                    graph,
                    left_id,
                    right_id,
                )
                subject_reasons.add(subject_reason)
                if not allowed:
                    pair_blockers.update(blockers)

        exact = not pair_blockers
        candidates.append(
            MeasurementIdentityCandidate(
                paper_id=paper_id,
                origin_local_id=origin_local_id,
                source_mention_ids=members,
                exact_consolidation_allowed=exact,
                blockers=tuple(sorted(pair_blockers)),
                subject_compatibility=tuple(sorted(subject_reasons)),
            )
        )

        if not exact:
            for member_id in members:
                identities.append(
                    MeasurementResultIdentity(
                        identity_id=_stable_identity_id(
                            paper_id=paper_id,
                            representative_measurement_id=member_id,
                            source_mention_ids=(member_id,),
                        ),
                        semantics_id=MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
                        paper_id=paper_id,
                        representative_measurement_id=member_id,
                        source_mention_ids=(member_id,),
                        origin_local_id=origin_local_id,
                        status="single_mention",
                        consolidation_reasons=(
                            "same_local_lineage_but_not_exact",
                        ),
                        subject_compatibility=tuple(
                            sorted(subject_reasons)
                        ),
                    )
                )
            continue

        representative = max(
            members,
            key=lambda item: _representative_score(graph, item),
        )
        identities.append(
            MeasurementResultIdentity(
                identity_id=_stable_identity_id(
                    paper_id=paper_id,
                    representative_measurement_id=representative,
                    source_mention_ids=members,
                ),
                semantics_id=MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
                paper_id=paper_id,
                representative_measurement_id=representative,
                source_mention_ids=members,
                origin_local_id=origin_local_id,
                status="consolidated_exact",
                consolidation_reasons=(
                    "same_origin_local_measurement_id",
                    "same_metric_id",
                    "same_result_value",
                    "compatible_unit",
                    "no_explicit_condition_conflict",
                    "subject_identity_compatible",
                ),
                subject_compatibility=tuple(sorted(subject_reasons)),
            )
        )

    return (
        sorted(
            identities,
            key=lambda item: (
                item.paper_id,
                item.representative_measurement_id,
                item.identity_id,
            ),
        ),
        sorted(
            candidates,
            key=lambda item: (
                item.paper_id,
                item.origin_local_id,
                item.source_mention_ids,
            ),
        ),
    )


def audit_measurement_result_identities(
    *,
    identities: Iterable[MeasurementResultIdentity],
    candidates: Iterable[MeasurementIdentityCandidate],
    source_graphs: Mapping[str, nx.Graph],
) -> MeasurementResultIdentityAudit:
    rows = list(identities)
    candidate_rows = list(candidates)
    issues: list[str] = []

    expected_mentions = {
        (paper_id, str(node_id))
        for paper_id, graph in source_graphs.items()
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("type", "")) == "Measurement"
    }
    observed_mentions: list[tuple[str, str]] = [
        (row.paper_id, mention_id)
        for row in rows
        for mention_id in row.source_mention_ids
    ]
    observed_set = set(observed_mentions)

    if len(observed_mentions) != len(observed_set):
        issues.append("source_mention_assigned_to_multiple_results")
    missing = expected_mentions - observed_set
    invented = observed_set - expected_mentions
    if missing:
        issues.append(
            "missing_source_mentions:"
            + ",".join(
                f"{paper}:{measurement}"
                for paper, measurement in sorted(missing)
            )
        )
    if invented:
        issues.append(
            "invented_source_mentions:"
            + ",".join(
                f"{paper}:{measurement}"
                for paper, measurement in sorted(invented)
            )
        )

    identity_ids = [row.identity_id for row in rows]
    if len(identity_ids) != len(set(identity_ids)):
        issues.append("duplicate_identity_id")

    for row in rows:
        graph = source_graphs.get(row.paper_id)
        if graph is None:
            issues.append(f"missing_source_graph:{row.paper_id}")
            continue
        if row.representative_measurement_id not in graph:
            issues.append(
                "representative_missing:"
                + row.paper_id
                + ":"
                + row.representative_measurement_id
            )

    unresolved_groups = sum(
        not item.exact_consolidation_allowed
        for item in candidate_rows
    )
    consolidated_rows = [
        row for row in rows if row.status == "consolidated_exact"
    ]
    return MeasurementResultIdentityAudit(
        semantics_id=MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
        paper_count=len(source_graphs),
        source_mention_count=len(expected_mentions),
        scientific_result_count=len(rows),
        single_mention_result_count=sum(
            row.status == "single_mention" for row in rows
        ),
        consolidated_exact_result_count=len(consolidated_rows),
        consolidated_source_mention_count=sum(
            row.source_mention_count for row in consolidated_rows
        ),
        unresolved_same_lineage_group_count=unresolved_groups,
        issues=tuple(sorted(set(issues))),
        structural_gate=not issues,
    )


def graph_has_measurement_payload_collisions(graph: nx.Graph) -> bool:
    return any(
        str(attrs.get("type", "")) == "Measurement"
        and str(attrs.get("id_collision_reason", "")).strip()
        == "measurement_payload_conflict"
        for _, attrs in graph.nodes(data=True)
    )


def _identity_by_member(
    identities: Iterable[MeasurementResultIdentity],
) -> dict[str, MeasurementResultIdentity]:
    result: dict[str, MeasurementResultIdentity] = {}
    for identity in identities:
        for member_id in identity.source_mention_ids:
            if member_id in result:
                raise ValueError(
                    f"Measurement mention assigned twice: {member_id!r}"
                )
            result[member_id] = identity
    return result


def _group_contexts_by_identity(
    rows: Iterable[Any],
    identities: Iterable[MeasurementResultIdentity],
) -> list[tuple[MeasurementResultIdentity, list[Any]]]:
    identity_rows = list(identities)
    by_member = _identity_by_member(identity_rows)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        identity = by_member.get(str(row.measurement_id))
        if identity is None:
            raise ValueError(
                "Measurement-result identity missing for context "
                f"{row.measurement_id!r}."
            )
        grouped[identity.identity_id].append(row)
    by_id = {row.identity_id: row for row in identity_rows}
    return [
        (by_id[identity_id], grouped_rows)
        for identity_id, grouped_rows in sorted(grouped.items())
    ]


def _merge_method_dimension(
    rows: list[MethodDimensionValue],
) -> MethodDimensionValue:
    name = rows[0].name
    if any(row.name != name for row in rows):
        raise ValueError("Cannot merge different method dimensions.")
    source_values = tuple(sorted({
        value
        for row in rows
        for value in row.source_values
    }))
    source_node_ids = tuple(sorted({
        value
        for row in rows
        for value in row.source_node_ids
    }))
    provenance_scopes = tuple(sorted({
        value
        for row in rows
        for value in row.provenance_scopes
    }))
    explicit = [row for row in rows if row.status != "unknown"]
    if not explicit:
        return MethodDimensionValue(
            name=name,
            status="unknown",
            source_values=source_values,
            source_node_ids=source_node_ids,
        )
    if any(row.status == "ambiguous" for row in explicit):
        return MethodDimensionValue(
            name=name,
            status="ambiguous",
            source_values=source_values,
            source_node_ids=source_node_ids,
            provenance_scopes=provenance_scopes,
        )
    normalized = {
        row.normalized_value for row in explicit if row.normalized_value
    }
    if len(normalized) == 1:
        return MethodDimensionValue(
            name=name,
            status="known",
            normalized_value=next(iter(normalized)),
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


def apply_identity_to_method_contexts(
    contexts: Iterable[MethodContext],
    identities: Iterable[MeasurementResultIdentity],
) -> list[MethodContext]:
    output: list[MethodContext] = []
    for identity, rows in _group_contexts_by_identity(contexts, identities):
        representative = next(
            (
                row
                for row in rows
                if row.measurement_id
                == identity.representative_measurement_id
            ),
            None,
        )
        if representative is None:
            raise ValueError(
                "Representative MethodContext missing for identity "
                f"{identity.identity_id}."
            )
        dimensions = tuple(
            _merge_method_dimension(
                [
                    row.dimension_map[name]
                    for row in rows
                ]
            )
            for name in (
                dimension.name for dimension in representative.dimensions
            )
        )
        output.append(
            replace(
                representative,
                producer_ids=tuple(sorted({
                    value
                    for row in rows
                    for value in row.producer_ids
                })),
                subject_ids=tuple(sorted({
                    value
                    for row in rows
                    for value in row.subject_ids
                })),
                dimensions=dimensions,
                source_node_ids=tuple(sorted({
                    *identity.source_mention_ids,
                    *(
                        value
                        for row in rows
                        for value in row.source_node_ids
                    ),
                })),
            )
        )
    return sorted(
        output,
        key=lambda item: (item.paper_id, item.measurement_id),
    )


def _merge_comparison_dimension(
    rows: list[ComparisonDimensionValue],
) -> ComparisonDimensionValue:
    name = rows[0].name
    if any(row.name != name for row in rows):
        raise ValueError("Cannot merge different comparison dimensions.")
    source_values = tuple(sorted({
        value
        for row in rows
        for value in row.source_values
    }))
    source_node_ids = tuple(sorted({
        value
        for row in rows
        for value in row.source_node_ids
    }))
    explicit = [row for row in rows if row.status != "unknown"]
    if not explicit:
        return ComparisonDimensionValue(
            name=name,
            status="unknown",
            source_values=source_values,
            source_node_ids=source_node_ids,
        )
    if any(row.status == "ambiguous" for row in explicit):
        return ComparisonDimensionValue(
            name=name,
            status="ambiguous",
            source_values=source_values,
            source_node_ids=source_node_ids,
        )
    normalized = {
        row.normalized_value for row in explicit if row.normalized_value
    }
    if len(normalized) == 1:
        return ComparisonDimensionValue(
            name=name,
            status="known",
            normalized_value=next(iter(normalized)),
            source_values=source_values,
            source_node_ids=source_node_ids,
        )
    return ComparisonDimensionValue(
        name=name,
        status="ambiguous",
        source_values=source_values,
        source_node_ids=source_node_ids,
    )


def apply_identity_to_comparison_contexts(
    contexts: Iterable[ComparisonContext],
    identities: Iterable[MeasurementResultIdentity],
) -> list[ComparisonContext]:
    output: list[ComparisonContext] = []
    for identity, rows in _group_contexts_by_identity(contexts, identities):
        representative = next(
            (
                row
                for row in rows
                if row.measurement_id
                == identity.representative_measurement_id
            ),
            None,
        )
        if representative is None:
            raise ValueError(
                "Representative ComparisonContext missing for identity "
                f"{identity.identity_id}."
            )
        if len({row.observable_key for row in rows}) != 1:
            raise ValueError(
                "Identity group crosses observable keys: "
                f"{identity.identity_id}."
            )
        dimensions = tuple(
            _merge_comparison_dimension(
                [row.dimension_map[name] for row in rows]
            )
            for name in (
                dimension.name for dimension in representative.dimensions
            )
        )
        output.append(
            replace(
                representative,
                subject_ids=tuple(sorted({
                    value
                    for row in rows
                    for value in row.subject_ids
                })),
                dimensions=dimensions,
                source_node_ids=tuple(sorted({
                    *identity.source_mention_ids,
                    *(
                        value
                        for row in rows
                        for value in row.source_node_ids
                    ),
                })),
            )
        )
    return sorted(
        output,
        key=lambda item: (item.paper_id, item.measurement_id),
    )


def apply_identity_to_metric_definition_contexts(
    contexts: Iterable[MetricDefinitionContext],
    identities: Iterable[MeasurementResultIdentity],
) -> list[MetricDefinitionContext]:
    context_rows = list(contexts)
    identity_rows = list(identities)
    by_member = _identity_by_member(identity_rows)
    grouped: dict[str, list[MetricDefinitionContext]] = defaultdict(list)
    for row in context_rows:
        identity = by_member.get(row.measurement_id)
        if identity is None:
            raise ValueError(
                "Measurement-result identity missing for metric definition "
                f"{row.measurement_id!r}."
            )
        grouped[identity.identity_id].append(row)

    by_id = {row.identity_id: row for row in identity_rows}
    output: list[MetricDefinitionContext] = []
    for identity_id, rows in sorted(grouped.items()):
        identity = by_id[identity_id]
        representative = next(
            (
                row
                for row in rows
                if row.measurement_id
                == identity.representative_measurement_id
            ),
            None,
        )
        if representative is None:
            raise ValueError(
                "Representative MetricDefinitionContext missing for identity "
                f"{identity.identity_id}."
            )
        if len({row.observable_key for row in rows}) != 1:
            raise ValueError(
                "Metric-definition identity group crosses observables: "
                f"{identity.identity_id}."
            )
        output.append(
            replace(
                representative,
                source_measurement_ids=tuple(sorted({
                    *identity.source_mention_ids,
                    *(
                        value
                        for row in rows
                        for value in row.source_measurement_ids
                    ),
                })),
                source_measurement_group_ids=tuple(sorted({
                    value
                    for row in rows
                    for value in row.source_measurement_group_ids
                })),
                source_experiment_ids=tuple(sorted({
                    value
                    for row in rows
                    for value in row.source_experiment_ids
                })),
                source_calculation_ids=tuple(sorted({
                    value
                    for row in rows
                    for value in row.source_calculation_ids
                })),
                source_node_ids=tuple(sorted({
                    *identity.source_mention_ids,
                    *(
                        value
                        for row in rows
                        for value in row.source_node_ids
                    ),
                })),
            )
        )
    return sorted(
        output,
        key=lambda item: (item.paper_id, item.measurement_id),
    )


def load_measurement_result_identity_sidecar(
    *,
    corpus_root: Path,
    identity_id: str,
    profile_id: str,
    corpus_id: str,
    corpus_mode: str,
) -> tuple[
    dict[str, list[MeasurementResultIdentity]],
    dict[str, Any],
    dict[str, Any],
]:
    root = corpus_root / "measurement_result_identity" / identity_id
    summary_path = root / "summary.json"
    identities_path = root / "identities.jsonl"
    audit_path = root / "audit.json"
    for path in (summary_path, identities_path, audit_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Measurement-result identity sidecar file not found: {path}"
            )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(audit, dict):
        raise ValueError("Measurement-result identity metadata must be objects.")

    expected = {
        "measurement_result_identity_id": identity_id,
        "measurement_result_identity_semantics_id": (
            MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID
        ),
        "domain_profile_id": profile_id,
        "corpus_id": corpus_id,
        "corpus_mode": corpus_mode,
    }
    for key, expected_value in expected.items():
        observed = str(summary.get(key, ""))
        if observed != str(expected_value):
            raise ValueError(
                "Measurement-result identity sidecar binding mismatch for "
                f"{key}: {observed!r} != {expected_value!r}."
            )
    if not bool(audit.get("structural_gate", False)):
        raise ValueError(
            "Measurement-result identity sidecar structural gate is false."
        )

    by_paper: dict[str, list[MeasurementResultIdentity]] = defaultdict(list)
    with identities_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    "Identity JSONL row must be an object at line "
                    f"{line_number}."
                )
            item = MeasurementResultIdentity(
                identity_id=str(row["identity_id"]),
                semantics_id=str(row["semantics_id"]),
                paper_id=str(row["paper_id"]),
                representative_measurement_id=str(
                    row["representative_measurement_id"]
                ),
                source_mention_ids=tuple(
                    str(value)
                    for value in row.get("source_mention_ids", [])
                ),
                origin_local_id=str(row["origin_local_id"]),
                status=str(row["status"]),
                consolidation_reasons=tuple(
                    str(value)
                    for value in row.get("consolidation_reasons", [])
                ),
                subject_compatibility=tuple(
                    str(value)
                    for value in row.get("subject_compatibility", [])
                ),
            )
            by_paper[item.paper_id].append(item)

    return dict(by_paper), summary, audit


def identity_source_hashes(
    summary: Mapping[str, Any],
) -> dict[str, str]:
    return {
        str(row.get("paper_id", "")): str(
            row.get("canonical_graph_sha256", "")
        )
        for row in summary.get("source_graphs", [])
        if isinstance(row, dict)
    }

IDENTITY_AWARE_DOMAIN_RECONSTRUCTION_ID = (
    "identity_aware_domain_reconstruction_v1_alpha4b4a11"
)


def _identity_edge_key(
    graph: nx.Graph,
    source: str,
    target: str,
    base_key: object,
    source_mention_id: str,
) -> str:
    payload = "|".join(
        (
            str(base_key),
            source_mention_id,
            source,
            target,
            IDENTITY_AWARE_DOMAIN_RECONSTRUCTION_ID,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    candidate = f"{base_key}:identity:{digest}"
    suffix = 1
    if not graph.is_multigraph():
        return candidate
    while graph.has_edge(source, target, key=candidate):
        candidate = f"{base_key}:identity:{digest}:{suffix}"
        suffix += 1
    return candidate


def build_identity_interpretation_graph(
    graph: nx.Graph,
    identities: Iterable[MeasurementResultIdentity],
) -> nx.Graph:
    """Build a transient one-node-per-scientific-result interpretation graph.

    The canonical graph is never mutated. For a consolidated_exact identity,
    incident provenance edges from every source mention are re-homed onto the
    representative Measurement and the non-representative mention nodes are
    removed only from this transient graph. Measurement value attributes are
    never field-wise merged, so alpha4b.4.1's numeric/text XOR invariant stays
    intact.

    Domain adapters can then re-extract MethodContext/ComparisonContext from
    the unioned grounded provenance instead of generically merging already-
    normalized dimension values.
    """
    overlay = graph.copy()

    for identity in sorted(
        identities,
        key=lambda item: (
            item.paper_id,
            item.representative_measurement_id,
            item.identity_id,
        ),
    ):
        if identity.status != "consolidated_exact":
            continue

        representative = identity.representative_measurement_id
        members = set(identity.source_mention_ids)
        if representative not in overlay:
            raise ValueError(
                "Identity representative is missing from source graph: "
                f"{identity.identity_id}:{representative}"
            )
        for member in identity.source_mention_ids:
            if member not in overlay:
                raise ValueError(
                    "Identity source mention is missing from source graph: "
                    f"{identity.identity_id}:{member}"
                )
            if str(overlay.nodes[member].get("type", "")) != "Measurement":
                raise ValueError(
                    "Identity source mention is not a Measurement: "
                    f"{identity.identity_id}:{member}"
                )

        for member in identity.source_mention_ids:
            if member == representative:
                continue

            if overlay.is_multigraph():
                incoming = list(
                    overlay.in_edges(member, keys=True, data=True)
                ) if overlay.is_directed() else []
                outgoing = list(
                    overlay.out_edges(member, keys=True, data=True)
                ) if overlay.is_directed() else []

                if not overlay.is_directed():
                    incident = list(
                        overlay.edges(member, keys=True, data=True)
                    )
                    incoming = []
                    outgoing = incident

                for source, target, edge_key, edge_data in incoming + outgoing:
                    new_source = (
                        representative if str(source) in members else str(source)
                    )
                    new_target = (
                        representative if str(target) in members else str(target)
                    )
                    if new_source == new_target:
                        continue
                    payload = dict(edge_data)
                    payload["measurement_result_source_mention_id"] = member
                    payload["measurement_result_identity_id"] = identity.identity_id
                    new_key = _identity_edge_key(
                        overlay,
                        new_source,
                        new_target,
                        edge_key,
                        member,
                    )
                    overlay.add_edge(
                        new_source,
                        new_target,
                        key=new_key,
                        **payload,
                    )
            else:
                if overlay.is_directed():
                    incident = list(overlay.in_edges(member, data=True)) + list(
                        overlay.out_edges(member, data=True)
                    )
                else:
                    incident = list(overlay.edges(member, data=True))
                for source, target, edge_data in incident:
                    new_source = (
                        representative if str(source) in members else str(source)
                    )
                    new_target = (
                        representative if str(target) in members else str(target)
                    )
                    if new_source == new_target:
                        continue
                    payload = dict(edge_data)
                    payload["measurement_result_source_mention_id"] = member
                    payload["measurement_result_identity_id"] = identity.identity_id
                    overlay.add_edge(new_source, new_target, **payload)

            overlay.remove_node(member)

        representative_attrs = overlay.nodes[representative]
        representative_attrs["measurement_result_identity_id"] = (
            identity.identity_id
        )
        representative_attrs["measurement_result_identity_status"] = (
            identity.status
        )
        representative_attrs["measurement_result_source_mention_count"] = (
            identity.source_mention_count
        )
        representative_attrs["measurement_result_source_mentions_json"] = (
            json.dumps(
                list(identity.source_mention_ids),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        representative_attrs["identity_aware_domain_reconstruction_id"] = (
            IDENTITY_AWARE_DOMAIN_RECONSTRUCTION_ID
        )

    expected_results = len(list(identities))
    observed_results = sum(
        str(attrs.get("type", "")) == "Measurement"
        for _, attrs in overlay.nodes(data=True)
    )
    if observed_results != expected_results:
        raise ValueError(
            "Identity interpretation graph Measurement count mismatch: "
            f"expected {expected_results}, observed {observed_results}."
        )
    return overlay


def _identity_by_representative(
    identities: Iterable[MeasurementResultIdentity],
) -> dict[str, MeasurementResultIdentity]:
    rows = list(identities)
    result = {
        item.representative_measurement_id: item
        for item in rows
    }
    if len(result) != len(rows):
        raise ValueError(
            "MeasurementResultIdentity representatives must be unique."
        )
    return result


def attach_identity_provenance_to_method_contexts(
    contexts: Iterable[MethodContext],
    identities: Iterable[MeasurementResultIdentity],
) -> list[MethodContext]:
    by_representative = _identity_by_representative(identities)
    output: list[MethodContext] = []
    for context in contexts:
        identity = by_representative.get(context.measurement_id)
        if identity is None:
            raise ValueError(
                "Domain-reconstructed MethodContext has no result identity: "
                f"{context.measurement_id!r}."
            )
        output.append(
            replace(
                context,
                source_node_ids=tuple(sorted({
                    *context.source_node_ids,
                    *identity.source_mention_ids,
                })),
            )
        )
    return sorted(output, key=lambda item: (item.paper_id, item.measurement_id))


def attach_identity_provenance_to_comparison_contexts(
    contexts: Iterable[ComparisonContext],
    identities: Iterable[MeasurementResultIdentity],
) -> list[ComparisonContext]:
    by_representative = _identity_by_representative(identities)
    output: list[ComparisonContext] = []
    for context in contexts:
        identity = by_representative.get(context.measurement_id)
        if identity is None:
            raise ValueError(
                "Domain-reconstructed ComparisonContext has no result identity: "
                f"{context.measurement_id!r}."
            )
        output.append(
            replace(
                context,
                source_node_ids=tuple(sorted({
                    *context.source_node_ids,
                    *identity.source_mention_ids,
                })),
            )
        )
    return sorted(output, key=lambda item: (item.paper_id, item.measurement_id))

