from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from dac_her.cross_context_trend import (
    CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    CrossContextTrendSource,
)
from dac_her.domains.comparison_registry import (
    get_comparison_adapter,
)
from dac_her.domains.cross_context_trend_registry import (
    get_cross_context_trend_adapter,
)
from dac_her.domains.registry import get_domain_profile
from dac_her.domains.sers_au_ag_cross_context_trend import (
    audit_sers_au_ag_trend_context_projection,
)
from dac_her.domains.trend_precision_registry import (
    get_trend_precision_adapter,
)
from dac_her.domains.trend_registry import get_trend_adapter
from dac_her.trend_precision import PaperLocalTrendResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row must be an object at "
                    f"{path}:{line_number}."
                )
            rows.append(row)
    return rows


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        str(row.get("paper_id", "")): str(
            row.get("canonical_graph_sha256", "")
        )
        for row in rows
        if str(row.get("paper_id", "")).strip()
    }


def _paper_local_result_from_row(
    row: Mapping[str, Any],
) -> PaperLocalTrendResult:
    values = dict(row)
    for key in (
        "member_trend_ids",
        "evidence_kinds",
        "trend_subject_ids",
        "reference_subject_ids",
        "source_claim_ids",
        "source_measurement_ids",
        "source_measurement_result_ids",
        "source_calculation_ids",
        "source_node_ids",
    ):
        values[key] = tuple(
            str(value)
            for value in values.get(key, [])
        )
    return PaperLocalTrendResult(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project provenance-safe domain context onto frozen "
            "PaperLocalTrendResult objects. alpha4c.3b builds "
            "TrendContextProfile only; it does not create pairwise "
            "contrasts or final cross-context assessments."
        )
    )
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--data-root", default="data_sers")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--trend-id", required=True)
    parser.add_argument("--precision-id", required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    trend_adapter = get_trend_adapter(profile)
    precision_adapter = get_trend_precision_adapter(profile)
    comparison_adapter = get_comparison_adapter(profile)
    context_adapter = get_cross_context_trend_adapter(profile)

    if profile.profile_id != "sers_au_ag":
        raise ValueError(
            "alpha4c.3b currently implements only the SERS "
            "context-projection adapter."
        )
    if (
        precision_adapter.trend_semantics_id
        != trend_adapter.semantics_id
    ):
        raise ValueError(
            "Trend/precision semantics mismatch."
        )

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    corpus_root = (
        data_root
        / "corpus"
        / args.corpus_id
        / args.mode
    )
    trend_root = corpus_root / "trend" / args.trend_id
    precision_root = (
        trend_root
        / "precision"
        / args.precision_id
    )

    trend_summary_path = trend_root / "summary.json"
    trend_audit_path = trend_root / "audit.json"
    precision_summary_path = precision_root / "summary.json"
    precision_audit_path = precision_root / "audit.json"
    local_results_path = precision_root / "local_results.jsonl"

    for path in (
        trend_summary_path,
        trend_audit_path,
        precision_summary_path,
        precision_audit_path,
        local_results_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    trend_summary = json.loads(
        trend_summary_path.read_text(encoding="utf-8")
    )
    trend_audit = json.loads(
        trend_audit_path.read_text(encoding="utf-8")
    )
    precision_summary = json.loads(
        precision_summary_path.read_text(encoding="utf-8")
    )
    precision_audit = json.loads(
        precision_audit_path.read_text(encoding="utf-8")
    )

    if not bool(trend_summary.get("structural_gate", False)):
        raise ValueError(
            "Parent TrendEvidence structural gate is false."
        )
    if not bool(trend_audit.get("structural_gate", False)):
        raise ValueError(
            "Parent TrendEvidence audit structural gate is false."
        )
    if not bool(
        precision_summary.get("structural_gate", False)
    ):
        raise ValueError(
            "Parent trend precision structural gate is false."
        )
    if not bool(
        precision_audit.get("structural_gate", False)
    ):
        raise ValueError(
            "Parent trend precision audit structural gate is false."
        )

    expected_precision = {
        "domain_profile_id": profile.profile_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "trend_id": args.trend_id,
        "trend_semantics_id": trend_adapter.semantics_id,
        "precision_semantics_id":
            precision_adapter.precision_semantics_id,
    }
    for key, expected in expected_precision.items():
        observed = str(
            precision_summary.get(key, "")
        )
        if observed != str(expected):
            raise ValueError(
                "Precision sidecar binding mismatch for "
                f"{key}: {observed!r} != {expected!r}."
            )

    expected_trend = {
        "domain_profile_id": profile.profile_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "trend_id": args.trend_id,
        "trend_semantics_id": trend_adapter.semantics_id,
    }
    for key, expected in expected_trend.items():
        observed = str(trend_summary.get(key, ""))
        if observed != str(expected):
            raise ValueError(
                "Trend sidecar binding mismatch for "
                f"{key}: {observed!r} != {expected!r}."
            )

    comparison_id = str(
        trend_summary.get("comparison_id", "")
    ).strip()
    if not comparison_id:
        raise ValueError(
            "Parent trend summary does not bind a comparison sidecar."
        )

    comparison_root = (
        corpus_root
        / "comparison"
        / comparison_id
    )
    comparison_summary_path = (
        comparison_root / "summary.json"
    )
    comparison_audit_path = (
        comparison_root / "audit.json"
    )
    contexts_path = (
        comparison_root / "contexts.jsonl"
    )
    methods_path = (
        comparison_root / "method_contexts.jsonl"
    )
    for path in (
        comparison_summary_path,
        comparison_audit_path,
        contexts_path,
        methods_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    comparison_summary = json.loads(
        comparison_summary_path.read_text(
            encoding="utf-8"
        )
    )
    comparison_audit = json.loads(
        comparison_audit_path.read_text(
            encoding="utf-8"
        )
    )

    expected_comparison = {
        "comparison_id": comparison_id,
        "domain_profile_id": profile.profile_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "comparison_semantics_id":
            comparison_adapter.semantics_id,
        "measurement_result_identity_id": str(
            trend_summary.get(
                "measurement_result_identity_id",
                "",
            )
        ),
    }
    for key, expected in expected_comparison.items():
        observed = str(
            comparison_summary.get(key, "")
        )
        if observed != str(expected):
            raise ValueError(
                "Comparison sidecar binding mismatch for "
                f"{key}: {observed!r} != {expected!r}."
            )

    if comparison_adapter.method_semantics is None:
        raise ValueError(
            "SERS comparison adapter must expose MethodContext "
            "semantics for alpha4c.3b."
        )
    observed_method_semantics = str(
        comparison_summary.get(
            "method_semantics_id",
            "",
        )
    )
    expected_method_semantics = (
        comparison_adapter.method_semantics.semantics_id
    )
    if observed_method_semantics != expected_method_semantics:
        raise ValueError(
            "MethodContext semantics mismatch: "
            f"{observed_method_semantics!r} != "
            f"{expected_method_semantics!r}."
        )

    if not bool(
        comparison_summary.get(
            "passes_structural_gate",
            False,
        )
    ):
        raise ValueError(
            "Comparison sidecar structural gate is false."
        )
    if not bool(
        comparison_audit.get(
            "passes_structural_gate",
            False,
        )
    ):
        raise ValueError(
            "Comparison audit structural gate is false."
        )

    trend_hashes = _source_hashes_from_rows(
        trend_summary.get("source_graphs", [])
    )
    comparison_hashes = _source_hashes_from_rows(
        comparison_summary.get("source_graphs", [])
    )
    precision_hashes = {
        str(key): str(value)
        for key, value in dict(
            precision_summary.get(
                "source_graph_sha256",
                {},
            )
        ).items()
    }
    if (
        not trend_hashes
        or trend_hashes != comparison_hashes
        or trend_hashes != precision_hashes
    ):
        raise ValueError(
            "Trend/precision/comparison canonical graph hash "
            "bindings do not match."
        )

    # Verify the current canonical graph files against all bound sidecars.
    for source_row in trend_summary.get(
        "source_graphs",
        [],
    ):
        if not isinstance(source_row, dict):
            continue
        paper_id = str(
            source_row.get("paper_id", "")
        )
        graph_path = Path(
            str(
                source_row.get(
                    "canonical_graphml",
                    "",
                )
            )
        )
        if not graph_path.is_absolute():
            graph_path = PROJECT_ROOT / graph_path
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        observed_hash = _sha256_file(graph_path)
        expected_hash = trend_hashes[paper_id]
        if observed_hash != expected_hash:
            raise ValueError(
                f"Canonical graph hash drift for "
                f"{paper_id}: {observed_hash} != "
                f"{expected_hash}."
            )

    local_results = [
        _paper_local_result_from_row(row)
        for row in _read_jsonl(local_results_path)
    ]
    comparison_rows = _read_jsonl(contexts_path)
    method_rows = _read_jsonl(methods_path)

    source = CrossContextTrendSource(
        local_results=tuple(local_results),
        comparison_context_rows=tuple(
            comparison_rows
        ),
        method_context_rows=tuple(method_rows),
    )
    profiles = context_adapter.project_contexts(
        source
    )
    audit = audit_sers_au_ag_trend_context_projection(
        source=source,
        profiles=profiles,
    )

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else (
            precision_root
            / "cross_context"
            / args.context_id
        )
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles_path = _write_jsonl(
        output_root / "context_profiles.jsonl",
        (profile.to_row() for profile in profiles),
    )
    audit_output_path = output_root / "audit.json"
    audit_output_path.write_text(
        json.dumps(
            audit.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "context_id": args.context_id,
        "domain_profile_id": profile.profile_id,
        "contract_semantics_id":
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        "context_semantics_id":
            context_adapter.context_semantics_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "trend_id": args.trend_id,
        "trend_semantics_id":
            trend_adapter.semantics_id,
        "precision_id": args.precision_id,
        "precision_semantics_id":
            precision_adapter.precision_semantics_id,
        "comparison_id": comparison_id,
        "comparison_semantics_id":
            comparison_adapter.semantics_id,
        "method_semantics_id":
            expected_method_semantics,
        "source_graph_sha256": trend_hashes,
        "local_result_count":
            audit.local_result_count,
        "profile_count": audit.profile_count,
        "direct_measurement_profile_count":
            audit.direct_measurement_profile_count,
        "no_direct_measurement_profile_count":
            audit.no_direct_measurement_profile_count,
        "profiles_with_known_context":
            audit.profiles_with_known_context,
        "profiles_with_ambiguous_context":
            audit.profiles_with_ambiguous_context,
        "varied_control_profile_count":
            audit.varied_control_profile_count,
        "dimension_status_counts":
            audit.dimension_status_counts,
        "paper_global_leakage_count":
            audit.paper_global_leakage_count,
        "unresolved_direct_measurement_count":
            audit.unresolved_direct_measurement_count,
        "pairwise_contrasts_built": False,
        "cross_context_assessments_built": False,
        "numeric_ranking_reused_as_trend_policy": False,
        "paper_global_context_fallback_used": False,
        "structural_gate": audit.structural_gate,
        "sources": {
            "trend_summary": str(
                trend_summary_path
            ),
            "precision_summary": str(
                precision_summary_path
            ),
            "comparison_summary": str(
                comparison_summary_path
            ),
            "comparison_contexts": str(
                contexts_path
            ),
            "method_contexts": str(
                methods_path
            ),
        },
        "outputs": {
            "context_profiles": str(
                profiles_path
            ),
            "audit": str(
                audit_output_path
            ),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Cross-context trend profiles built")
    print("Context ID:", args.context_id)
    print(
        "Contract semantics:",
        CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    )
    print(
        "Context semantics:",
        context_adapter.context_semantics_id,
    )
    print(
        "Local results / profiles:",
        audit.local_result_count,
        "/",
        audit.profile_count,
    )
    print(
        "Direct / no-direct Measurement profiles:",
        audit.direct_measurement_profile_count,
        "/",
        audit.no_direct_measurement_profile_count,
    )
    print(
        "Profiles with known / ambiguous context:",
        audit.profiles_with_known_context,
        "/",
        audit.profiles_with_ambiguous_context,
    )
    print(
        "Varied-control profiles:",
        audit.varied_control_profile_count,
    )
    print(
        "Paper-global leakage:",
        audit.paper_global_leakage_count,
    )
    print(
        "Unresolved direct Measurement provenance:",
        audit.unresolved_direct_measurement_count,
    )
    print("Structural gate:", audit.structural_gate)
    print("Saved:", output_root)
    return 0 if audit.structural_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
