from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from dac_her.trend_domain import TrendDomainAdapter, TrendEvidence, TrendEvidenceSource


_ALLOWED_SOURCE_TYPES = frozenset({
    "Measurement", "MeasurementGroup", "Experiment", "Calculation",
    "ObservationClaim", "MechanismClaim",
})


def stable_trend_id(
    *,
    paper_id: str,
    independent_variable_key: str,
    dependent_observable_key: str,
    evidence_basis: str,
    source_node_ids: Iterable[str],
) -> str:
    source_ids = tuple(sorted(set(map(str, source_node_ids))))
    if not source_ids:
        raise ValueError("stable trend IDs require grounded source nodes.")
    payload = "|".join(
        (paper_id, independent_variable_key, dependent_observable_key,
         evidence_basis, *source_ids)
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"trend:{digest}"


@dataclass(frozen=True)
class TrendEvidenceAudit:
    domain_profile_id: str
    trend_semantics_id: str
    evidence_count: int
    quantitative_evidence_count: int
    claim_evidence_count: int
    source_asserted_causal_count: int
    evidence_basis_counts: dict[str, int]
    direction_counts: dict[str, int]
    shape_counts: dict[str, int]
    source_node_reference_count: int
    source_measurement_result_reference_count: int
    source_method_context_reference_count: int
    source_comparison_context_reference_count: int
    issues: tuple[str, ...]
    structural_gate: bool

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = list(self.issues)
        return row


def _row_ids(rows: Iterable[Mapping[str, Any]], key: str) -> set[str]:
    return {
        str(row.get(key, "")).strip()
        for row in rows
        if str(row.get(key, "")).strip()
    }


def _validate_sidecar_references(
    *,
    item: TrendEvidence,
    source: TrendEvidenceSource,
    issues: list[str],
) -> None:
    if item.source_measurement_result_ids:
        known = _row_ids(source.measurement_result_rows, "identity_id")
        if not known:
            issues.append(f"missing_measurement_result_sidecar:{item.trend_id}")
        else:
            for value in item.source_measurement_result_ids:
                if value not in known:
                    issues.append(
                        f"missing_measurement_result_reference:{item.trend_id}:{value}"
                    )

    if item.source_method_context_ids:
        known = _row_ids(source.method_context_rows, "method_context_id")
        if not known:
            issues.append(f"missing_method_context_sidecar:{item.trend_id}")
        else:
            for value in item.source_method_context_ids:
                if value not in known:
                    issues.append(
                        f"missing_method_context_reference:{item.trend_id}:{value}"
                    )

    if item.source_comparison_context_ids:
        known = _row_ids(source.comparison_context_rows, "context_id")
        if not known:
            issues.append(f"missing_comparison_context_sidecar:{item.trend_id}")
        else:
            for value in item.source_comparison_context_ids:
                if value not in known:
                    issues.append(
                        f"missing_comparison_context_reference:{item.trend_id}:{value}"
                    )


def audit_trend_evidence(
    *,
    evidence: list[TrendEvidence],
    sources: dict[str, TrendEvidenceSource],
    adapter: TrendDomainAdapter,
) -> TrendEvidenceAudit:
    issues: list[str] = []
    seen: set[str] = set()

    for item in evidence:
        if item.trend_id in seen:
            issues.append(f"duplicate_trend_id:{item.trend_id}")
        seen.add(item.trend_id)

        source = sources.get(item.paper_id)
        if source is None:
            issues.append(f"missing_source_bundle:{item.paper_id}")
            continue
        graph = source.graph

        for node_id in item.source_node_ids:
            if node_id not in graph:
                issues.append(f"missing_source_node:{item.trend_id}:{node_id}")
                continue
            node_type = str(graph.nodes[node_id].get("type", ""))
            if node_type not in _ALLOWED_SOURCE_TYPES:
                issues.append(
                    f"unsupported_source_type:{item.trend_id}:{node_id}:{node_type}"
                )

        for subject_id in item.subject_ids:
            if subject_id not in graph:
                issues.append(f"missing_subject_node:{item.trend_id}:{subject_id}")

        for claim_id in item.source_claim_ids:
            if claim_id not in graph:
                continue
            claim_type = str(graph.nodes[claim_id].get("type", ""))
            if claim_type not in {"ObservationClaim", "MechanismClaim"}:
                issues.append(
                    f"invalid_claim_source_type:{item.trend_id}:{claim_id}:{claim_type}"
                )

        for measurement_id in item.source_measurement_ids:
            if measurement_id not in graph:
                continue
            measurement_type = str(graph.nodes[measurement_id].get("type", ""))
            if measurement_type != "Measurement":
                issues.append(
                    "invalid_measurement_source_type:"
                    f"{item.trend_id}:{measurement_id}:{measurement_type}"
                )

        _validate_sidecar_references(item=item, source=source, issues=issues)

        # Numeric evidence belongs to exactly one paper/source bundle. The
        # contract never derives a trend by comparing absolute values from
        # different paper bundles.
        if item.is_quantitative and item.paper_id != source.paper_id:
            issues.append(f"cross_paper_numeric_trend:{item.trend_id}")

    basis_counts = Counter(item.evidence_basis for item in evidence)
    direction_counts = Counter(item.direction for item in evidence)
    shape_counts = Counter(item.shape for item in evidence)
    quantitative = sum(item.is_quantitative for item in evidence)
    causal = sum(item.causal_status == "source_asserted" for item in evidence)
    return TrendEvidenceAudit(
        domain_profile_id=adapter.domain_profile_id,
        trend_semantics_id=adapter.semantics_id,
        evidence_count=len(evidence),
        quantitative_evidence_count=quantitative,
        claim_evidence_count=len(evidence) - quantitative,
        source_asserted_causal_count=causal,
        evidence_basis_counts=dict(sorted(basis_counts.items())),
        direction_counts=dict(sorted(direction_counts.items())),
        shape_counts=dict(sorted(shape_counts.items())),
        source_node_reference_count=sum(len(item.source_node_ids) for item in evidence),
        source_measurement_result_reference_count=sum(
            len(item.source_measurement_result_ids) for item in evidence
        ),
        source_method_context_reference_count=sum(
            len(item.source_method_context_ids) for item in evidence
        ),
        source_comparison_context_reference_count=sum(
            len(item.source_comparison_context_ids) for item in evidence
        ),
        issues=tuple(sorted(set(issues))),
        structural_gate=not issues,
    )
