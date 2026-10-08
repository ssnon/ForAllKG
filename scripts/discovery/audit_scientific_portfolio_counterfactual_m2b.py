"""M2-B: frozen-evaluation early scientific portfolio selection sensitivity audit.

Read-only. Reuses existing evaluation records; does not generate candidates,
call an LLM, query literature, rewrite evidence, or change production decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Callable

from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioEvaluationReport,
    ScientificPortfolioSelectionReport,
    build_scientific_portfolio_selection,
)


def _sha_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_kinds(candidate: Any) -> set[str]:
    kinds = set(str(x) for x in (candidate.parent_source_kinds or []))
    if candidate.source_kind:
        kinds.add(str(candidate.source_kind))
    return kinds


def scenarios() -> dict[str, Callable[[Any], bool]]:
    return {
        "BASELINE": lambda c: True,
        "FRONTIER_ONLY": lambda c: c.origin == "FRONTIER",
        "EVOLUTION_ONLY": lambda c: c.origin == "EVOLUTION",
        "NO_CROSS_SOURCE_EVOLUTION": lambda c: not (c.origin == "EVOLUTION" and c.cross_source_composition),
        "NO_BACKBONE_MUTATION": lambda c: c.operator_id != "BACKBONE_MUTATION",
        "NO_CANDIDATE_INTERPRETATION": lambda c: c.operator_id != "CANDIDATE_INTERPRETATION",
        "NO_EXPLICIT_EXTERNAL_LINEAGE": lambda c: not c.external_literature_lineage,
        "NO_HIGHER_ORDER_ANCESTRY": lambda c: "HIGHER_ORDER" not in _source_kinds(c),
        "NO_DIRECT_HIGHER_ORDER_ANCESTRY": lambda c: "DIRECT_HIGHER_ORDER" not in _source_kinds(c),
    }


def _filter_candidate_ids(pool: Any, predicate: Callable[[Any], bool]) -> set[str]:
    return {c.candidate_id for c in pool.candidates if predicate(c)}


def replay_frozen_selection(
    *,
    pool: ScientificPortfolioCandidatePool,
    evaluation: ScientificPortfolioEvaluationReport,
    eligible_ids: set[str],
    max_retained: int,
    max_per_profile: int,
    label: str,
) -> ScientificPortfolioSelectionReport:
    """Call the actual production selector with synthetic read-only views.

    The synthetic pool/evaluation identity prevents the diagnostic result from
    masquerading as a selection over the original unfiltered population.
    """
    eligible_candidates = [c for c in pool.candidates if c.candidate_id in eligible_ids]
    eligible_evaluations = [e for e in evaluation.evaluations if e.candidate_id in eligible_ids]
    if {x.candidate_id for x in eligible_candidates} != {x.candidate_id for x in eligible_evaluations}:
        raise ValueError(f"{label}: frozen candidate/evaluation coverage mismatch")
    fingerprint = _sha_bytes(json.dumps(
        {"pool": pool.pool_id, "evaluation": evaluation.report_id, "label": label, "ids": sorted(eligible_ids)},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8"))
    diagnostic_id = f"m2b_filtered_pool:{fingerprint[:20]}"
    synthetic_pool = SimpleNamespace(pool_id=diagnostic_id, pool_sha256=fingerprint, candidates=eligible_candidates)
    synthetic_eval = SimpleNamespace(
        source_pool_id=diagnostic_id,
        source_pool_sha256=fingerprint,
        report_id=f"m2b_filtered_evaluation:{fingerprint[:20]}",
        report_sha256=fingerprint,
        evaluations=eligible_evaluations,
    )
    return build_scientific_portfolio_selection(
        pool=synthetic_pool,  # Diagnostic protocol view; does not alter production models.
        evaluation=synthetic_eval,
        max_retained_candidates=max_retained,
        max_retained_per_profile=max_per_profile,
    )


def _validate_m1_pin(manifest: dict[str, Any], key: str, source_path: Path) -> None:
    rows = [x for x in manifest.get("artifacts", []) if x.get("key") == key]
    if len(rows) != 1 or not rows[0].get("present"):
        raise ValueError(f"M1 manifest lacks pinned {key} input")
    expected = rows[0].get("source_sha256")
    observed = _sha_path(source_path)
    if observed != expected:
        raise ValueError(f"{key}: frozen SHA mismatch: expected {expected}, got {observed}")


def run_audit(*, case_dir: Path, output_dir: Path, freeze_manifest: Path | None = None) -> dict[str, Any]:
    case_dir = case_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_dir = case_dir / "scientific_portfolio_shadow"
    pool_path = source_dir / "candidate_pool.json"
    eval_path = source_dir / "evaluation.json"
    selection_path = source_dir / "selection.json"
    if not all(p.is_file() for p in (pool_path, eval_path, selection_path)):
        raise FileNotFoundError("M2-B requires candidate_pool.json, evaluation.json and selection.json under the original case scientific_portfolio_shadow")
    if output_dir.exists():
        raise FileExistsError(f"M2-B output already exists, refusing overwrite: {output_dir}")
    if output_dir.is_relative_to(case_dir):
        raise ValueError("Refusing to create audit outputs inside the original case directory")
    if freeze_manifest is not None:
        manifest = _load_json(freeze_manifest)
        _validate_m1_pin(manifest, "stage7_candidate_pool", pool_path)
        _validate_m1_pin(manifest, "stage7_selection", selection_path)
        expected_case = Path(manifest["source_case"]).expanduser().resolve()
        if expected_case != case_dir:
            raise ValueError("M1 manifest is pinned to a different case root")

    pool = ScientificPortfolioCandidatePool.model_validate_json(pool_path.read_text(encoding="utf-8"))
    evaluation = ScientificPortfolioEvaluationReport.model_validate_json(eval_path.read_text(encoding="utf-8"))
    selection = ScientificPortfolioSelectionReport.model_validate_json(selection_path.read_text(encoding="utf-8"))
    if evaluation.source_pool_id != pool.pool_id or evaluation.source_pool_sha256 != pool.pool_sha256:
        raise ValueError("Original evaluation/pool lineage mismatch")
    if selection.source_pool_id != pool.pool_id or selection.source_pool_sha256 != pool.pool_sha256:
        raise ValueError("Original selection/pool lineage mismatch")
    if selection.source_evaluation_report_id != evaluation.report_id:
        raise ValueError("Original selection/evaluation lineage mismatch")
    if {x.candidate_id for x in pool.candidates} != {x.candidate_id for x in evaluation.evaluations}:
        raise ValueError("Original pool and evaluation candidate coverage mismatch")

    baseline_ids = [x.candidate_id for x in selection.entries]
    candidates = {x.candidate_id: x for x in pool.candidates}
    rows = []
    base_entries: list[Any] | None = None
    for label, pred in scenarios().items():
        eligible_ids = _filter_candidate_ids(pool, pred)
        if not eligible_ids:
            # The production selector cannot choose from an empty candidate set.
            selected_entries = []
        else:
            replay = replay_frozen_selection(
                pool=pool, evaluation=evaluation,
                eligible_ids=eligible_ids,
                max_retained=selection.max_retained_candidates,
                max_per_profile=selection.max_retained_per_profile,
                label=label,
            )
            selected_entries = replay.entries
        ids = [e.candidate_id for e in selected_entries]
        if label == "BASELINE":
            expected = [(e.candidate_id, e.assigned_profile, e.pareto_layer) for e in selection.entries]
            actual = [(e.candidate_id, e.assigned_profile, e.pareto_layer) for e in selected_entries]
            if actual != expected:
                raise RuntimeError("BASELINE_FROZEN_SELECTION_PARITY_FAILED; abort diagnostic counterfactual")
            base_entries = selected_entries
        rows.append({
            "scenario": label,
            "eligible_candidate_count": len(eligible_ids),
            "selected_candidate_count": len(ids),
            "selected_candidate_ids": ids,
            "selected_source_object_ids": [candidates[cid].source_object_id for cid in ids],
            "selected_origins": dict(sorted(Counter(candidates[cid].origin for cid in ids).items())),
            "selected_profiles": dict(sorted(Counter(e.assigned_profile for e in selected_entries).items())),
            "baseline_retained_ids": [cid for cid in baseline_ids if cid in ids],
            "lost_baseline_ids": [cid for cid in baseline_ids if cid not in ids],
            "newly_selected_ids": [cid for cid in ids if cid not in baseline_ids],
            "same_exact_selection_as_baseline": ids == baseline_ids,
        })
    assert base_entries is not None
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(eval_path, output_dir / "frozen_input_evaluation.json")
    _write_json(output_dir / "selection_sensitivity.json", {
        "schema_version": "m2b-frozen-evaluation-selection-sensitivity-v1",
        "audit_only": True,
        "no_llm_or_network_calls": True,
        "no_source_data_mutated": True,
        "same_model_generated_evaluations_reused": True,
        "does_not_re_evaluate_missing_or_new_candidates": True,
        "does_not_estimate_true_abolition_effect": True,
        "candidate_pool_sha256": _sha_path(pool_path),
        "evaluation_sha256": _sha_path(eval_path),
        "selection_sha256": _sha_path(selection_path),
        "baseline_parity": "PASS",
        "baseline_retained_count": len(baseline_ids),
        "scenarios": rows,
    })
    with (output_dir / "selection_sensitivity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "eligible_candidate_count", "selected_candidate_count", "baseline_retained_count", "lost_baseline_count", "newly_selected_count", "same_exact_selection_as_baseline"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "scenario": row["scenario"],
                "eligible_candidate_count": row["eligible_candidate_count"],
                "selected_candidate_count": row["selected_candidate_count"],
                "baseline_retained_count": len(row["baseline_retained_ids"]),
                "lost_baseline_count": len(row["lost_baseline_ids"]),
                "newly_selected_count": len(row["newly_selected_ids"]),
                "same_exact_selection_as_baseline": row["same_exact_selection_as_baseline"],
            })
    lines = [
        "# M2-B — Frozen-Evaluation Early Portfolio Selection Sensitivity",
        "",
        "Status: **BASELINE_PARITY_PASSED** · No LLM, no network, no mutation.",
        "",
        "These are filtered-choice counterfactuals with *fixed original evaluations*,",
        "not estimates of scientific quality or results from regenerating the pipeline.",
        "Candidates omitted earlier from the 525 Frontier population are not scored here.",
        "",
        "| Scenario | Eligible | Selected | Baseline kept | Lost | New |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['eligible_candidate_count']} | {row['selected_candidate_count']} | {len(row['baseline_retained_ids'])} | {len(row['lost_baseline_ids'])} | {len(row['newly_selected_ids'])} |")
    lines += ["", "No deletion is authorized by these results. Retain original seed and all strict evidence/identity boundaries.", ""]
    (output_dir / "M2B_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "BASELINE_PARITY_PASSED", "output_dir": str(output_dir), "scenarios": len(rows)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--freeze-manifest", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    result = run_audit(case_dir=args.case_dir, output_dir=args.output_dir, freeze_manifest=args.freeze_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
