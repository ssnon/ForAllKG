from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.comparison_context import (
    apply_protocol_numeric_gate,
    audit_comparison_outputs,
    build_pairwise_assessments,
    build_protocol_assessments,
)
from dac_her.quality_aware_comparison import (
    QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID,
    apply_metric_definition_numeric_gate,
    build_metric_definition_assessments,
)
from dac_her.domains.comparison_registry import get_comparison_adapter
from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.metric_definition_registry import (
    get_metric_definition_adapter,
)
from dac_her.domains.registry import get_domain_profile
from dac_her.metric_definition_context import (
    audit_metric_definition_contexts,
)
from dac_her.metric_definition_domain import MetricDefinitionContext


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_definition_context_from_row(
    row: dict[str, Any],
) -> MetricDefinitionContext:
    values = dict(row)
    for key in (
        "source_measurement_ids",
        "source_measurement_group_ids",
        "source_experiment_ids",
        "source_calculation_ids",
        "source_node_ids",
    ):
        values[key] = tuple(
            str(value)
            for value in values.get(key, [])
        )
    return MetricDefinitionContext(**values)


def _load_metric_definition_sidecar(
    *,
    corpus_root: Path,
    metric_definition_id: str,
    profile_id: str,
    corpus_id: str,
    corpus_mode: str,
    metric_adapter,
    source_graphs: dict[str, nx.Graph],
    source_rows: list[dict[str, str]],
) -> tuple[list[MetricDefinitionContext], dict[str, Any], dict[str, Any]]:
    root = (
        corpus_root
        / "metric_definition"
        / metric_definition_id
    )
    summary_path = root / "summary.json"
    contexts_path = root / "contexts.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Metric-definition summary not found: {summary_path}"
        )
    if not contexts_path.exists():
        raise FileNotFoundError(
            f"Metric-definition contexts not found: {contexts_path}"
        )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    if not isinstance(summary, dict):
        raise ValueError(
            "Metric-definition summary must be a JSON object."
        )

    expected_fields = {
        "metric_definition_id": metric_definition_id,
        "domain_profile_id": profile_id,
        "corpus_id": corpus_id,
        "corpus_mode": corpus_mode,
        "metric_definition_semantics_id": (
            metric_adapter.semantics_id
        ),
    }
    for key, expected in expected_fields.items():
        observed = str(summary.get(key, ""))
        if observed != str(expected):
            raise ValueError(
                "Metric-definition sidecar binding mismatch for "
                f"{key}: {observed!r} != {expected!r}."
            )

    expected_hashes = {
        row["paper_id"]: row["canonical_graph_sha256"]
        for row in source_rows
    }
    observed_hashes = {
        str(row.get("paper_id", "")): str(
            row.get("canonical_graph_sha256", "")
        )
        for row in summary.get("source_graphs", [])
        if isinstance(row, dict)
    }
    if observed_hashes != expected_hashes:
        raise ValueError(
            "Metric-definition sidecar canonical graph hashes do not "
            "match the comparison build inputs."
        )

    contexts: list[MetricDefinitionContext] = []
    with contexts_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    "Metric-definition JSONL row must be an object: "
                    f"line {line_number}."
                )
            contexts.append(
                _metric_definition_context_from_row(row)
            )

    audit = audit_metric_definition_contexts(
        contexts=contexts,
        source_graphs=source_graphs,
        adapter=metric_adapter,
    )
    if not audit.structural_gate:
        raise ValueError(
            "Metric-definition sidecar failed structural re-audit: "
            f"{list(audit.issues)!r}."
        )

    return contexts, summary, audit.to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fail-closed cross-paper ComparisonContext sidecars "
            "from strict canonical scientific graphs."
        )
    )
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument(
        "--metric-definition-id",
        default=None,
        help=(
            "Frozen MetricDefinitionContext sidecar ID used by the "
            "quality-aware numeric gate."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    adapter = get_comparison_adapter(profile)
    extraction_adapter = get_extraction_adapter(profile.profile_id)
    metric_definition_adapter = None
    if getattr(profile, "metric_definition_adapter_id", None):
        metric_definition_adapter = get_metric_definition_adapter(profile)
        if not args.metric_definition_id:
            raise ValueError(
                "--metric-definition-id is required for domain profile "
                f"{profile.profile_id!r}."
            )
    elif args.metric_definition_id:
        raise ValueError(
            "--metric-definition-id was supplied, but this domain has "
            "no metric-definition adapter."
        )

    data_root = Path(
        args.data_root or extraction_adapter.default_data_root
    )
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    corpus_root = (
        data_root
        / "corpus"
        / args.corpus_id
        / args.mode
    )
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Corpus manifest not found: {manifest_path}"
        )
    corpus_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(corpus_manifest, dict):
        raise ValueError("Corpus manifest must be a JSON object.")

    corpus_domain = str(
        corpus_manifest.get("domain_profile_id", "")
    )
    if corpus_domain != profile.profile_id:
        raise ValueError(
            "Comparison corpus/domain mismatch: "
            f"{corpus_domain!r} != {profile.profile_id!r}."
        )

    if profile.corpus is not None:
        corpus_semantics_id = str(
            corpus_manifest.get("corpus_semantics_id", "")
        )
        if corpus_semantics_id != profile.corpus.semantics_id:
            raise ValueError(
                "Comparison corpus semantics mismatch: "
                f"{corpus_semantics_id!r} != "
                f"{profile.corpus.semantics_id!r}."
            )

    if int(
        corpus_manifest.get(
            "destructive_cross_paper_merges",
            -1,
        )
    ) != 0:
        raise ValueError(
            "Comparison sidecar requires a non-destructive corpus."
        )

    paper_ids = [
        str(value)
        for value in corpus_manifest.get("paper_ids", [])
        if str(value).strip()
    ]
    if not paper_ids:
        raise ValueError("Corpus manifest contains no paper IDs.")
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Corpus manifest contains duplicate paper IDs.")

    source_graphs: dict[str, nx.Graph] = {}
    source_rows: list[dict[str, str]] = []
    contexts = []
    method_contexts = []

    for paper_id in paper_ids:
        graph_path = (
            data_root
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Canonical graph not found: {graph_path}"
            )
        graph = nx.read_graphml(
            graph_path,
            force_multigraph=True,
        )
        graph_domain = str(
            graph.graph.get("domain_profile_id", "")
        )
        if graph_domain and graph_domain != profile.profile_id:
            raise ValueError(
                "Canonical graph/domain mismatch for "
                f"{paper_id}: {graph_domain!r} != "
                f"{profile.profile_id!r}."
            )

        source_graphs[paper_id] = graph
        paper_method_contexts = adapter.extract_method_contexts(
            graph,
            paper_id,
        )
        method_contexts.extend(paper_method_contexts)
        paper_contexts = adapter.extract_contexts(
            graph,
            paper_id,
        )
        contexts.extend(paper_contexts)
        source_rows.append({
            "paper_id": paper_id,
            "canonical_graphml": str(graph_path),
            "canonical_graph_sha256": _sha256_file(graph_path),
            "context_count": str(len(paper_contexts)),
        })

    assessments = build_pairwise_assessments(
        contexts,
        adapter=adapter,
    )
    protocol_assessments = build_protocol_assessments(
        contexts,
        method_contexts,
        adapter=adapter,
    )
    assessments = apply_protocol_numeric_gate(
        assessments,
        protocol_assessments,
        adapter=adapter,
    )

    metric_definition_contexts = []
    metric_definition_assessments = []
    metric_definition_summary: dict[str, Any] = {}
    metric_definition_audit: dict[str, Any] = {}
    if metric_definition_adapter is not None:
        (
            metric_definition_contexts,
            metric_definition_summary,
            metric_definition_audit,
        ) = _load_metric_definition_sidecar(
            corpus_root=corpus_root,
            metric_definition_id=str(args.metric_definition_id),
            profile_id=profile.profile_id,
            corpus_id=args.corpus_id,
            corpus_mode=args.mode,
            metric_adapter=metric_definition_adapter,
            source_graphs=source_graphs,
            source_rows=source_rows,
        )
        metric_definition_assessments = (
            build_metric_definition_assessments(
                comparison_assessments=assessments,
                comparison_contexts=contexts,
                metric_definition_contexts=(
                    metric_definition_contexts
                ),
                adapter=metric_definition_adapter,
            )
        )
        assessments = apply_metric_definition_numeric_gate(
            assessments,
            metric_definition_assessments,
        )

    audit = audit_comparison_outputs(
        contexts=contexts,
        assessments=assessments,
        source_graphs=source_graphs,
        adapter=adapter,
        method_contexts=method_contexts,
        protocol_assessments=protocol_assessments,
        metric_definition_assessments=(
            metric_definition_assessments
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            corpus_root
            / "comparison"
            / args.comparison_id
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts_path = _write_jsonl(
        output_dir / "contexts.jsonl",
        (item.to_row() for item in contexts),
    )
    assessments_path = _write_jsonl(
        output_dir / "assessments.jsonl",
        (item.to_row() for item in assessments),
    )

    method_contexts_path = _write_jsonl(
        output_dir / "method_contexts.jsonl",
        (item.to_row() for item in method_contexts),
    )
    protocol_assessments_path = _write_jsonl(
        output_dir / "protocol_assessments.jsonl",
        (item.to_row() for item in protocol_assessments),
    )
    metric_definition_assessments_path = _write_jsonl(
        output_dir / "metric_definition_assessments.jsonl",
        (
            item.to_row()
            for item in metric_definition_assessments
        ),
    )

    compatibility_counts = Counter(
        item.compatibility for item in assessments
    )
    summary = {
        "comparison_id": args.comparison_id,
        "domain_profile_id": profile.profile_id,
        "comparison_adapter_id": adapter.adapter_id,
        "comparison_semantics_id": adapter.semantics_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "corpus_manifest": str(manifest_path),
        "corpus_semantics_id": str(
            corpus_manifest.get("corpus_semantics_id", "")
        ),
        "paper_ids": paper_ids,
        "paper_count": len(paper_ids),
        "dimensions": list(adapter.dimensions),
        "required_for_numeric_ranking": sorted(
            adapter.required_for_numeric_ranking
        ),
        "context_count": len(contexts),
        "method_context_count": len(method_contexts),
        "method_semantics_id": (
            adapter.method_semantics.semantics_id
            if adapter.method_semantics is not None
            else ""
        ),
        "numeric_context_count": sum(
            item.value_numeric is not None for item in contexts
        ),
        "assessment_count": len(assessments),
        "protocol_assessment_count": len(protocol_assessments),
        "protocol_comparability_counts": dict(sorted(Counter(
            item.comparability for item in protocol_assessments
        ).items())),
        "quality_gate_semantics_id": (
            QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID
            if metric_definition_adapter is not None
            else ""
        ),
        "metric_definition_id": (
            str(args.metric_definition_id or "")
        ),
        "metric_definition_semantics_id": (
            metric_definition_adapter.semantics_id
            if metric_definition_adapter is not None
            else ""
        ),
        "metric_definition_context_count": len(
            metric_definition_contexts
        ),
        "metric_definition_assessment_count": len(
            metric_definition_assessments
        ),
        "metric_definition_compatibility_counts": audit[
            "metric_definition_compatibility_counts"
        ],
        "metric_definition_applicable_assessment_count": audit[
            "metric_definition_applicable_assessment_count"
        ],
        "metric_definition_gate_pass_count": audit[
            "metric_definition_gate_pass_count"
        ],
        "metric_definition_gate_blocked_applicable_count": audit[
            "metric_definition_gate_blocked_applicable_count"
        ],
        "metric_definition_ranking_relevant_assessment_count": audit[
            "metric_definition_ranking_relevant_assessment_count"
        ],
        "metric_definition_ranking_relevant_gate_pass_count": audit[
            "metric_definition_ranking_relevant_gate_pass_count"
        ],
        "metric_definition_ranking_relevant_gate_blocked_count": audit[
            "metric_definition_ranking_relevant_gate_blocked_count"
        ],
        "compatibility_counts": dict(
            sorted(compatibility_counts.items())
        ),
        "observable_policy_counts": dict(sorted(Counter(
            item.observable_policy_id for item in assessments
        ).items())),
        "observable_family_counts": dict(sorted(Counter(
            item.observable_family for item in assessments
        ).items())),
        "unregistered_observable_assessment_count": sum(
            item.observable_policy_id == "unregistered"
            for item in assessments
        ),
        "numeric_ranking_allowed_count": sum(
            item.numeric_ranking_allowed for item in assessments
        ),
        "missing_context_is_not_quarantine": True,
        "global_entity_concentration_consumed": False,
        "measurement_local_concentration_context_count": sum(
            method.dimension_map["analyte_concentration"].status
            == "known"
            for method in method_contexts
        ),
        "method_dimension_status_counts": audit[
            "method_dimension_status_counts"
        ],
        "method_provenance_scope_counts": audit[
            "method_provenance_scope_counts"
        ],
        "protocol_matched_dimension_counts": audit[
            "protocol_matched_dimension_counts"
        ],
        "protocol_mismatched_dimension_counts": audit[
            "protocol_mismatched_dimension_counts"
        ],
        "protocol_pairs_with_any_match": audit[
            "protocol_pairs_with_any_match"
        ],
        "source_graphs": source_rows,
        "contexts": str(contexts_path),
        "assessments": str(assessments_path),
        "method_contexts": str(method_contexts_path),
        "protocol_assessments": str(protocol_assessments_path),
        "metric_definition_assessments": str(
            metric_definition_assessments_path
        ),
        "metric_definition_source_summary": (
            str(
                corpus_root
                / "metric_definition"
                / str(args.metric_definition_id)
                / "summary.json"
            )
            if metric_definition_adapter is not None
            else ""
        ),
        "metric_definition_source_audit": (
            metric_definition_audit
        ),
        "audit": str(output_dir / "audit.json"),
        "passes_structural_gate": audit["passes_structural_gate"],
    }

    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "audit.json").write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Comparison contexts built")
    print("Comparison ID:", args.comparison_id)
    print("Domain profile:", profile.profile_id)
    print("Comparison semantics:", adapter.semantics_id)
    print("Corpus:", args.corpus_id, f"({args.mode})")
    print("Papers:", len(paper_ids))
    print("Contexts:", len(contexts))
    print("Numeric contexts:", summary["numeric_context_count"])
    print("Assessments:", len(assessments))
    print("Method contexts:", len(method_contexts))
    print("Protocol assessments:", len(protocol_assessments))
    print(
        "Protocol comparability:",
        json.dumps(
            summary["protocol_comparability_counts"],
            sort_keys=True,
        ),
    )
    print(
        "Measurement-local concentration contexts:",
        summary["measurement_local_concentration_context_count"],
    )
    method_known_counts = {
        name: counts.get("known", 0)
        for name, counts in summary[
            "method_dimension_status_counts"
        ].items()
    }
    print(
        "Method known coverage:",
        json.dumps(method_known_counts, sort_keys=True),
    )
    print(
        "Protocol pairs with any matched method dimension:",
        summary["protocol_pairs_with_any_match"],
    )
    print(
        "Protocol matched dimensions:",
        json.dumps(
            summary["protocol_matched_dimension_counts"],
            sort_keys=True,
        ),
    )
    print(
        "Compatibility:",
        json.dumps(
            summary["compatibility_counts"],
            sort_keys=True,
        ),
    )
    if metric_definition_adapter is not None:
        print(
            "Metric-definition sidecar:",
            summary["metric_definition_id"],
        )
        print(
            "Metric-definition contexts:",
            summary["metric_definition_context_count"],
        )
        print(
            "Metric-definition compatibility:",
            json.dumps(
                summary[
                    "metric_definition_compatibility_counts"
                ],
                sort_keys=True,
            ),
        )
        print(
            "Metric-definition applicable assessments:",
            summary[
                "metric_definition_applicable_assessment_count"
            ],
        )
        print(
            "Metric-definition gate pass / blocked:",
            summary["metric_definition_gate_pass_count"],
            "/",
            summary[
                "metric_definition_gate_blocked_applicable_count"
            ],
        )
        print(
            "Metric-definition ranking-relevant assessments:",
            summary[
                "metric_definition_ranking_relevant_assessment_count"
            ],
        )
        print(
            "Metric-definition ranking-relevant gate pass / blocked:",
            summary[
                "metric_definition_ranking_relevant_gate_pass_count"
            ],
            "/",
            summary[
                "metric_definition_ranking_relevant_gate_blocked_count"
            ],
        )
    print(
        "Observable families:",
        json.dumps(
            summary["observable_family_counts"],
            sort_keys=True,
        ),
    )
    print(
        "Unregistered observable assessments:",
        summary["unregistered_observable_assessment_count"],
    )
    print(
        "Numeric ranking allowed:",
        summary["numeric_ranking_allowed_count"],
    )
    print("Structural gate:", audit["passes_structural_gate"])
    print("Saved:", output_dir)

    if not audit["passes_structural_gate"]:
        raise RuntimeError(
            "Comparison structural audit failed. See: "
            f"{output_dir / 'audit.json'}"
        )


if __name__ == "__main__":
    main()
