from __future__ import annotations

import json
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from campaigns.sers_alpha4_epoch.holdout.alpha4c4d2_holdout_support import (
    Alpha4c4d2Error,
    ROOT,
    DATA_ROOT,
    atomic_json,
    read_json,
    read_jsonl,
    verify_locked_input_record,
)


PROTOCOL_PATH = (
    ROOT / "configs/heldout/sers_alpha4c4d2_trend_holdout_v2_run.json"
)


def run_stage(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    name: str,
    command: list[str],
    outputs: list[Path],
) -> None:
    existing = manifest.setdefault("stages", {}).get(name)
    if isinstance(existing, dict) and existing.get("status") == "complete":
        for path in outputs:
            rel = str(path.relative_to(ROOT))
            expected = existing.get("outputs", {}).get(rel)
            if not path.exists() or not expected:
                raise Alpha4c4d2Error(
                    f"Completed stage output missing: {name}/{rel}"
                )
            from campaigns.sers_alpha4_epoch.holdout.alpha4c4d2_holdout_support import sha256
            if sha256(path) != expected:
                raise Alpha4c4d2Error(
                    f"Completed stage output drift: {name}/{rel}"
                )
        print("[SKIP VERIFIED]", name)
        return

    manifest["stages"][name] = {
        "status": "running",
        "command": command,
    }
    atomic_json(manifest_path, manifest)

    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        manifest["stages"][name] = {
            "status": "failed",
            "command": command,
            "exit_code": result.returncode,
        }
        atomic_json(manifest_path, manifest)
        raise Alpha4c4d2Error(
            f"Stage {name} failed with exit code {result.returncode}"
        )

    from campaigns.sers_alpha4_epoch.holdout.alpha4c4d2_holdout_support import sha256
    locked = {}
    for path in outputs:
        if not path.exists():
            raise Alpha4c4d2Error(
                f"Stage {name} missing output: {path}"
            )
        locked[str(path.relative_to(ROOT))] = sha256(path)

    manifest["stages"][name] = {
        "status": "complete",
        "command": command,
        "outputs": locked,
    }
    atomic_json(manifest_path, manifest)


def verify_protocol_and_lock() -> tuple[dict, dict]:
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("phase") != "alpha4c.4d.2":
        raise Alpha4c4d2Error("Unexpected protocol phase.")
    if protocol.get("source_split_sha256") != '6eebae74732070408e920154ba898841c879a8e97cb866a8971cb9feef526966':
        raise Alpha4c4d2Error("v2 split SHA drift.")
    if protocol.get("holdout_papers") != ['Kiwook_SERS_21', 'Kiwook_SERS_38', 'Kiwook_SERS_12', 'Kiwook_SERS_28', 'Kiwook_SERS_17', 'Kiwook_SERS_22', 'Kiwook_SERS_23', 'Kiwook_SERS_11']:
        raise Alpha4c4d2Error("v2 paper set drift.")
    if (
        protocol["frozen_semantics"]["metric_definition"]
        != 'sers_au_ag_metric_definition_v3_alpha4c4c1'
    ):
        raise Alpha4c4d2Error("MetricDefinition semantics drift.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "campaigns.sers_alpha4_epoch.holdout.cli.verify_sers_alpha4c4d1_holdout_v2_protocol",
        ],
        cwd=ROOT,
        check=True,
    )

    lock_path = ROOT / protocol["canonical_input_lock"]
    lock = read_json(lock_path)
    if lock.get("state") != "frozen_v2_canonical_input_lock":
        raise Alpha4c4d2Error("Canonical input lock state invalid.")
    if lock.get("holdout_papers") != protocol["holdout_papers"]:
        raise Alpha4c4d2Error("Input lock paper set drift.")

    for paper_id in protocol["holdout_papers"]:
        verify_locked_input_record(
            paper_id,
            lock["papers"][paper_id],
        )

    # The input-preparation policy already permits PARTIAL_CRITICAL with an
    # explicit override. Preserve that exact policy into corpus construction;
    # never silently broaden it to REJECTED/unknown inputs.
    allowed_statuses = {
        "complete",
        "partial_acceptable",
        "partial_critical",
    }
    for paper_id in protocol["holdout_papers"]:
        status = lock["papers"][paper_id]["strict_source"][
            "extraction_quality"
        ]["graph_materialization_status"]
        if status not in allowed_statuses:
            raise Alpha4c4d2Error(
                f"{paper_id}: locked extraction quality is not eligible "
                f"for the frozen v2 corpus: {status!r}"
            )
    return protocol, lock


def main() -> int:
    protocol, lock = verify_protocol_and_lock()

    eval_root = ROOT / protocol["evaluation_root"]
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_root / "manifest.json"
    report_path = eval_root / "holdout_report.json"

    if report_path.exists():
        report = read_json(report_path)
        if report.get("state") == "passed":
            print("alpha4c.4d.2 holdout already complete: PASS")
            print("Report:", report_path)
            return 0

    manifest = (
        read_json(manifest_path)
        if manifest_path.exists()
        else {
            "phase": "alpha4c.4d.2",
            "state": "running",
            "source_split_sha256": protocol["source_split_sha256"],
            "holdout_papers": protocol["holdout_papers"],
            "canonical_input_lock":
                protocol["canonical_input_lock"],
            "llm_calls_performed_by_runner": False,
            "bridge_used": False,
            "projection_mode": "evidence",
            "stages": {},
        }
    )
    atomic_json(manifest_path, manifest)

    try:
        ids = protocol["artifact_ids"]
        mode = "evidence"
        papers = protocol["holdout_papers"]

        # Re-verify lock before any scientific stage.
        for paper_id in papers:
            verify_locked_input_record(
                paper_id,
                lock["papers"][paper_id],
            )

        for paper_id in papers:
            canonical = (
                DATA_ROOT / "extracted" / paper_id /
                f"{paper_id}.graphml"
            )
            root = (
                DATA_ROOT / "extracted" / paper_id /
                "graphagents" / mode
            )
            run_stage(
                manifest=manifest,
                manifest_path=manifest_path,
                name=f"projection:{paper_id}",
                command=[
                    sys.executable,
                    "-m",
                    "scripts.build_graphagents_projection",
                    "--paper-id", paper_id,
                    "--domain-profile", "sers_au_ag",
                    "--data-root", "data_sers",
                    "--mode", mode,
                    "--canonical-graphml", str(canonical),
                ],
                outputs=[
                    root / "graph.graphml",
                    root / "summary.json",
                    root / "node_text.jsonl",
                    root / "edge_evidence.jsonl",
                ],
            )

        corpus_root = (
            DATA_ROOT / "corpus" / ids["corpus"] / mode
        )
        run_stage(
            manifest=manifest,
            manifest_path=manifest_path,
            name="corpus",
            command=(
                [
                    sys.executable,
                    "-m",
                    "scripts.build_corpus_graph",
                    "--corpus-id", ids["corpus"],
                    "--domain-profile", "sers_au_ag",
                    "--data-root", "data_sers",
                    "--paper-ids", *papers,
                    "--mode", mode,
                ]
                + (
                    ["--allow-critical-partial"]
                    if any(
                        lock["papers"][paper_id]["strict_source"][
                            "extraction_quality"
                        ]["graph_materialization_status"]
                        == "partial_critical"
                        for paper_id in papers
                    )
                    else []
                )
            ),
            outputs=[
                corpus_root / "manifest.json",
                corpus_root / "audit.json",
                corpus_root / "graph.graphml",
            ],
        )

        identity_root = (
            corpus_root / "measurement_result_identity" /
            ids["measurement_result_identity"]
        )
        run_stage(
            manifest=manifest,
            manifest_path=manifest_path,
            name="measurement_result_identity",
            command=[
                sys.executable,
                "-m",
                "scripts.build_measurement_result_identities",
                "--domain-profile", "sers_au_ag",
                "--data-root", "data_sers",
                "--corpus-id", ids["corpus"],
                "--mode", mode,
                "--measurement-result-identity-id",
                ids["measurement_result_identity"],
            ],
            outputs=[
                identity_root / "summary.json",
                identity_root / "audit.json",
                identity_root / "identities.jsonl",
                identity_root / "same_lineage_candidates.jsonl",
            ],
        )

        metric_root = (
            corpus_root / "metric_definition" /
            ids["metric_definition"]
        )
        run_stage(
            manifest=manifest,
            manifest_path=manifest_path,
            name="metric_definition",
            command=[
                sys.executable,
                "-m",
                "scripts.build_metric_definition_contexts",
                "--domain-profile", "sers_au_ag",
                "--data-root", "data_sers",
                "--corpus-id", ids["corpus"],
                "--mode", mode,
                "--metric-definition-id", ids["metric_definition"],
                "--measurement-result-identity-id",
                ids["measurement_result_identity"],
            ],
            outputs=[
                metric_root / "summary.json",
                metric_root / "audit.json",
                metric_root / "contexts.jsonl",
            ],
        )

        comparison_root = (
            corpus_root / "comparison" / ids["comparison"]
        )
        run_stage(
            manifest=manifest,
            manifest_path=manifest_path,
            name="comparison",
            command=[
                sys.executable,
                "-m",
                "scripts.build_comparison_contexts",
                "--domain-profile", "sers_au_ag",
                "--data-root", "data_sers",
                "--corpus-id", ids["corpus"],
                "--mode", mode,
                "--comparison-id", ids["comparison"],
                "--metric-definition-id", ids["metric_definition"],
                "--measurement-result-identity-id",
                ids["measurement_result_identity"],
            ],
            outputs=[
                comparison_root / "summary.json",
                comparison_root / "audit.json",
                comparison_root / "contexts.jsonl",
                comparison_root / "method_contexts.jsonl",
                comparison_root / "assessments.jsonl",
                comparison_root / "protocol_assessments.jsonl",
                comparison_root / "metric_definition_assessments.jsonl",
            ],
        )

        trend_root = corpus_root / "trend" / ids["trend"]
        run_stage(
            manifest=manifest,
            manifest_path=manifest_path,
            name="trend",
            command=[
                sys.executable,
                "-m",
                "scripts.build_trend_evidence",
                "--domain-profile", "sers_au_ag",
                "--data-root", "data_sers",
                "--corpus-id", ids["corpus"],
                "--mode", mode,
                "--trend-id", ids["trend"],
                "--comparison-id", ids["comparison"],
                "--measurement-result-identity-id",
                ids["measurement_result_identity"],
            ],
            outputs=[
                trend_root / "summary.json",
                trend_root / "audit.json",
                trend_root / "evidence.jsonl",
            ],
        )

        precision_root = (
            trend_root / "precision" / ids["precision"]
        )
        run_stage(
            manifest=manifest,
            manifest_path=manifest_path,
            name="precision",
            command=[
                sys.executable,
                "-m",
                "scripts.build_trend_precision",
                "--domain-profile", "sers_au_ag",
                "--data-root", "data_sers",
                "--corpus-id", ids["corpus"],
                "--mode", mode,
                "--trend-id", ids["trend"],
                "--precision-id", ids["precision"],
            ],
            outputs=[
                precision_root / "summary.json",
                precision_root / "audit.json",
                precision_root / "annotations.jsonl",
                precision_root / "local_results.jsonl",
            ],
        )

        trend_summary = read_json(trend_root / "summary.json")
        precision_summary = read_json(
            precision_root / "summary.json"
        )
        local_results = read_jsonl(
            precision_root / "local_results.jsonl"
        )

        if trend_summary.get("structural_gate") is not True:
            raise Alpha4c4d2Error("Trend structural gate failed.")
        if precision_summary.get("structural_gate") is not True:
            raise Alpha4c4d2Error("Precision structural gate failed.")

        local_count = int(
            precision_summary.get(
                "local_result_count",
                len(local_results),
            )
        )
        if local_count != len(local_results):
            raise Alpha4c4d2Error(
                "Precision local_result_count/JSONL mismatch."
            )

        cross_context_status = None
        assessment_status = None
        status_counts = {}
        pair_count = 0
        relation_count = 0

        if local_count == 0:
            if int(trend_summary.get("evidence_count", 0)) != 0:
                raise Alpha4c4d2Error(
                    "Zero local results despite nonzero TrendEvidence."
                )
            cross_context_status = (
                "not_applicable_zero_local_results"
            )
            assessment_status = (
                "not_applicable_zero_local_results"
            )
            manifest["stages"]["cross_context"] = {
                "status": "not_applicable_zero_local_results",
                "reason": "precision.local_result_count == 0",
            }
            manifest["stages"]["assessment"] = {
                "status": "not_applicable_zero_local_results",
                "reason": "precision.local_result_count == 0",
            }
            atomic_json(manifest_path, manifest)
            print(
                "Zero local TrendResults: CrossContext/Assessment "
                "not applicable (valid count-free terminal outcome)."
            )
        else:
            context_root = (
                precision_root / "cross_context" / ids["context"]
            )
            run_stage(
                manifest=manifest,
                manifest_path=manifest_path,
                name="cross_context",
                command=[
                    sys.executable,
                    "-m",
                    "scripts.build_cross_context_profiles",
                    "--domain-profile", "sers_au_ag",
                    "--data-root", "data_sers",
                    "--corpus-id", ids["corpus"],
                    "--mode", mode,
                    "--trend-id", ids["trend"],
                    "--precision-id", ids["precision"],
                    "--context-id", ids["context"],
                ],
                outputs=[
                    context_root / "summary.json",
                    context_root / "audit.json",
                    context_root / "context_profiles.jsonl",
                ],
            )

            assessment_root = (
                context_root / "assessment" / ids["assessment"]
            )
            run_stage(
                manifest=manifest,
                manifest_path=manifest_path,
                name="assessment",
                command=[
                    sys.executable,
                    "-m",
                    "scripts.build_cross_context_assessments",
                    "--domain-profile", "sers_au_ag",
                    "--data-root", "data_sers",
                    "--corpus-id", ids["corpus"],
                    "--mode", mode,
                    "--trend-id", ids["trend"],
                    "--precision-id", ids["precision"],
                    "--context-id", ids["context"],
                    "--assessment-id", ids["assessment"],
                ],
                outputs=[
                    assessment_root / "summary.json",
                    assessment_root / "audit.json",
                    assessment_root / "pairwise_contrasts.jsonl",
                    assessment_root / "assessments.jsonl",
                ],
            )

            context_summary = read_json(
                context_root / "summary.json"
            )
            context_audit = read_json(
                context_root / "audit.json"
            )
            assessment_summary = read_json(
                assessment_root / "summary.json"
            )
            assessment_audit = read_json(
                assessment_root / "audit.json"
            )
            assessments = read_jsonl(
                assessment_root / "assessments.jsonl"
            )
            contrasts = read_jsonl(
                assessment_root / "pairwise_contrasts.jsonl"
            )

            if (
                context_summary.get("structural_gate") is not True
                or context_audit.get("structural_gate") is not True
            ):
                raise Alpha4c4d2Error(
                    "CrossContext structural gate failed."
                )
            if context_summary.get(
                "paper_global_context_fallback_used"
            ) is True:
                raise Alpha4c4d2Error(
                    "Paper-global context fallback used."
                )
            if int(
                context_summary.get(
                    "paper_global_leakage_count", 0
                )
            ) != 0:
                raise Alpha4c4d2Error(
                    "Paper-global context leakage detected."
                )

            if (
                assessment_summary.get("structural_gate") is not True
                or assessment_audit.get("structural_gate") is not True
            ):
                raise Alpha4c4d2Error(
                    "CrossContextAssessment structural gate failed."
                )
            if assessment_summary.get("majority_vote_used") is True:
                raise Alpha4c4d2Error("Majority vote used.")
            if assessment_summary.get(
                "numeric_ranking_reused_as_trend_policy"
            ) is True:
                raise Alpha4c4d2Error(
                    "Numeric ranking reused as Trend policy."
                )
            if assessment_summary.get(
                "causal_status_promoted"
            ) is True:
                raise Alpha4c4d2Error(
                    "Causal status promoted."
                )
            if assessment_summary.get(
                "context_reprojected"
            ) is True:
                raise Alpha4c4d2Error(
                    "Frozen context reprojected."
                )

            if (
                assessment_summary.get("relation_count")
                != assessment_summary.get("assessment_count")
                or assessment_summary.get("assessment_count")
                != len(assessments)
            ):
                raise Alpha4c4d2Error(
                    "One-assessment-per-relation invariant failed."
                )
            if (
                assessment_summary.get(
                    "expected_cross_paper_pair_count"
                )
                != assessment_summary.get("pairwise_contrast_count")
                or assessment_summary.get("pairwise_contrast_count")
                != len(contrasts)
            ):
                raise Alpha4c4d2Error(
                    "Complete pair generation invariant failed."
                )
            if any(
                row.get("left_paper_id")
                == row.get("right_paper_id")
                for row in contrasts
            ):
                raise Alpha4c4d2Error(
                    "Same-paper pair generated."
                )

            cross_context_status = "completed"
            assessment_status = "completed"
            status_counts = assessment_summary.get(
                "status_counts", {}
            )
            pair_count = len(contrasts)
            relation_count = len(assessments)

        # Re-verify canonical lock after scientific execution.
        for paper_id in papers:
            verify_locked_input_record(
                paper_id,
                lock["papers"][paper_id],
            )

        corpus_summary = read_json(corpus_root / "manifest.json")
        corpus_audit = read_json(corpus_root / "audit.json")
        identity_summary = read_json(
            identity_root / "summary.json"
        )
        metric_summary = read_json(
            metric_root / "summary.json"
        )
        comparison_summary = read_json(
            comparison_root / "summary.json"
        )

        violations = []
        if corpus_summary.get("paper_ids") != papers:
            violations.append("corpus_paper_set")
        if corpus_summary.get("passes_structural_gate") is not True:
            violations.append("corpus_structural_gate")
        if corpus_audit.get("passes_structural_gate") is not True:
            violations.append("corpus_audit_structural_gate")
        if int(
            corpus_summary.get("destructive_cross_paper_merges", 0)
        ) != 0:
            violations.append("destructive_cross_paper_merge")
        if identity_summary.get("structural_gate") is not True:
            violations.append("identity_structural_gate")
        if metric_summary.get("structural_gate") is not True:
            violations.append("metric_structural_gate")
        if (
            metric_summary.get("metric_definition_semantics_id")
            != 'sers_au_ag_metric_definition_v3_alpha4c4c1'
        ):
            violations.append("metric_semantics")
        if comparison_summary.get(
            "passes_structural_gate"
        ) is not True:
            violations.append("comparison_structural_gate")
        if comparison_summary.get(
            "global_entity_concentration_consumed"
        ) is True:
            violations.append("global_concentration_leakage")
        if comparison_summary.get(
            "missing_context_is_not_quarantine"
        ) is not True:
            violations.append("missing_context_policy")
        if trend_summary.get(
            "cross_paper_numeric_series_built"
        ) is True:
            violations.append("cross_paper_numeric_series")
        if trend_summary.get(
            "numeric_ranking_reused_as_trend_policy"
        ) is True:
            violations.append("ranking_policy_reuse")

        report = {
            "phase": "alpha4c.4d.2",
            "state": "passed" if not violations else "failed",
            "source_split_sha256":
                protocol["source_split_sha256"],
            "holdout_papers": papers,
            "count_thresholds_used": False,
            "zero_yield_valid": True,
            "llm_calls_performed_by_runner": False,
            "bridge_used": False,
            "projection_mode": "evidence",
            "distribution_observations": {
                "trend_evidence_count":
                    int(trend_summary.get("evidence_count", 0)),
                "local_result_count": local_count,
                "cross_context_status": cross_context_status,
                "assessment_status": assessment_status,
                "cross_paper_pair_count": pair_count,
                "relation_count": relation_count,
                "assessment_status_counts": status_counts,
                "metric_definition_status_counts":
                    metric_summary.get("definition_status_counts"),
                "comparison_context_count":
                    comparison_summary.get("context_count"),
                "numeric_ranking_allowed_count":
                    comparison_summary.get(
                        "numeric_ranking_allowed_count"
                    ),
            },
            "violations": violations,
            "interpretation": (
                "Pass/fail is invariant-only. Zero TrendEvidence, zero "
                "local TrendResults, zero cross-paper overlap, and all-"
                "insufficient assessments are valid outcomes."
            ),
        }
        atomic_json(report_path, report)

        manifest["state"] = report["state"]
        manifest["holdout_report"] = str(
            report_path.relative_to(ROOT)
        )
        atomic_json(manifest_path, manifest)

        print()
        print(
            "alpha4c.4d.2 frozen v2 Trend holdout verdict:",
            report["state"].upper(),
        )
        print(
            "Distribution:",
            json.dumps(
                report["distribution_observations"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        print("Violations:", violations)
        print("Report:", report_path)

        return 0 if not violations else 2

    except Exception as exc:
        manifest["state"] = "failed"
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        atomic_json(manifest_path, manifest)
        print(
            f"alpha4c.4d.2 ERROR: {exc}",
            file=sys.stderr,
        )
        print(
            "This v2 holdout has entered scientific execution. "
            "If the failure requires a scientific code change after "
            "inspection, retire all 8 papers to seen regression and "
            "use only the 14-paper v3 reserve.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
