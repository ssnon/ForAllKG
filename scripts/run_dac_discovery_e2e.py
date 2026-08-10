from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _returned_path_count(payload: dict[str, Any]) -> int:
    """Read traversal cardinality without depending on one artifact schema revision."""
    for key in ("returned_path_count", "selected_path_count", "path_count"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    for key in ("paths", "selected_paths", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("returned_path_count", "selected_path_count", "path_count"):
            value = summary.get(key)
            if isinstance(value, int):
                return value
    return 0


def _hypothesis_count(path: Path) -> int:
    payload = _load_json(path)
    rows = payload.get("hypotheses", [])
    return len(rows) if isinstance(rows, list) else 0


def _portfolio_id(path: Path) -> str:
    payload = _load_json(path)
    value = payload.get("portfolio_id")
    if not value:
        raise RuntimeError(f"portfolio_id missing from {path}")
    return str(value)


def _external_source_portfolio_id(path: Path) -> str:
    payload = _load_json(path)
    value = payload.get("source_portfolio_id")
    if not value:
        raise RuntimeError(f"source_portfolio_id missing from {path}")
    return str(value)


_ALPHA6_DEGRADED_DECISIONS = {
    "compile_rejected",
    "validation_rejected",
    "grounding_drift_rejected",
}


def _alpha6_empty_is_degraded(report_payload: dict[str, Any]) -> bool:
    attempts = report_payload.get("attempts", [])
    if not isinstance(attempts, list) or not attempts:
        return False
    decisions = [
        str(row.get("decision"))
        for row in attempts
        if isinstance(row, dict)
    ]
    return bool(decisions) and all(
        decision in _ALPHA6_DEGRADED_DECISIONS
        for decision in decisions
    )


class PipelineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.manifest_path = self.run_dir / "e2e_runner.manifest.json"
        self.manifest: dict[str, Any] = {
            "schema_version": "dac-discovery-e2e-runner-v1",
            "started_at_utc": _now(),
            "status": "initializing",
            "corpus_id": args.corpus_id,
            "source": args.source,
            "stop": args.stop,
            "target": args.target,
            "question": args.question,
            "objective": args.objective,
            "title": args.title,
            "grounding_policy": args.grounding_policy,
            "grounding_algorithm_used": None,
            "stages": [],
            "failure": None,
        }

    def _save_manifest(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.manifest_path, self.manifest)

    def prepare(self) -> None:
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            if not self.args.overwrite_run:
                raise RuntimeError(
                    f"Run directory is not empty: {self.run_dir}. "
                    "Use a fresh run name or pass --overwrite-run. This guard prevents "
                    "stale artifacts from being mixed across executions."
                )
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest["status"] = "running"
        self._save_manifest()

    def run_stage(
        self,
        name: str,
        module: str,
        argv: list[str],
        *,
        expected: list[Path] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "name": name,
            "module": module,
            "started_at_utc": _now(),
            "status": "running",
        }
        self.manifest["stages"].append(record)
        self._save_manifest()
        command = [sys.executable, "-m", module, *argv]
        print()
        print("=" * 72)
        print(name)
        print("=" * 72)
        print("$", " ".join(command))
        subprocess.run(command, check=True)
        missing = [str(x) for x in (expected or []) if not x.exists()]
        if missing:
            raise RuntimeError(
                f"Stage {name!r} completed without expected artifacts: {missing}"
            )
        record["status"] = "complete"
        record["finished_at_utc"] = _now()
        self._save_manifest()

    def fail(self, exc: BaseException) -> None:
        self.manifest["status"] = "failed"
        self.manifest["finished_at_utc"] = _now()
        self.manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if self.manifest.get("stages"):
            row = self.manifest["stages"][-1]
            if row.get("status") == "running":
                row["status"] = "failed"
                row["finished_at_utc"] = _now()
        self._save_manifest()

    def complete(self) -> None:
        self.manifest["status"] = "complete"
        self.manifest["finished_at_utc"] = _now()
        self._save_manifest()


def _base_model_args(args: argparse.Namespace, *, critic: bool = False) -> list[str]:
    model = args.critic_model if critic else args.model
    if not model:
        name = "--critic-model" if critic else "--model"
        raise RuntimeError(f"{name} is required or must be available from the environment")
    result = ["--model", model]
    if args.base_url:
        result += ["--base-url", args.base_url]
    result += ["--api-key-env", args.api_key_env]
    return result


def _check_alpha6_available() -> None:
    try:
        __import__("scripts.run_novelty_refinement")
    except Exception as exc:
        raise RuntimeError(
            "scripts.run_novelty_refinement is unavailable. Apply the alpha6 targeted "
            "novelty-refinement bundle before using this full E2E runner."
        ) from exc


def run_pipeline(args: argparse.Namespace) -> int:
    runner = PipelineRunner(args)
    runner.prepare()
    _check_alpha6_available()
    run = runner.run_dir

    # ------------------------------------------------------------------
    # 1. Grounding retrieval: semantic-stop when useful, deterministic
    #    fallback to ordinary top_n when the hard waypoint has zero paths.
    # ------------------------------------------------------------------
    semantic_path = run / "traversal.semantic_stop.json"
    final_traversal = run / "traversal.json"
    use_semantic = bool(args.stop) and args.grounding_policy != "top_n"

    if use_semantic:
        runner.run_stage(
            "[1a/13] Grounding traversal: semantic_stop attempt",
            "scripts.run_graph_traversal",
            [
                "--corpus-id", args.corpus_id,
                "--mode", "mechanism",
                "--algorithm", "semantic_stop",
                "--source", args.source,
                "--stop", args.stop,
                "--target", args.target,
                "--node-map-k", str(args.node_map_k),
                "--waypoint-k", str(args.waypoint_k),
                "--endpoint-pair-k", str(args.endpoint_pair_k),
                "--semantic-stop-max-depth", str(args.max_depth),
                "--top-k", str(args.top_k),
                "--include-candidate-paths",
                "--output", str(semantic_path),
            ],
            expected=[semantic_path],
        )
        semantic_count = _returned_path_count(_load_json(semantic_path))
        runner.manifest["semantic_stop_returned_path_count"] = semantic_count
        runner._save_manifest()
        if semantic_count > 0:
            shutil.copy2(semantic_path, final_traversal)
            runner.manifest["grounding_algorithm_used"] = "semantic_stop"
            runner._save_manifest()
        elif args.grounding_policy == "semantic_stop_only":
            raise RuntimeError(
                "semantic_stop returned zero grounded paths and fallback is disabled"
            )
        else:
            print(
                "WARNING: semantic_stop returned zero paths; falling back to top_n "
                "grounding. The stop concept remains in the natural-language question "
                "but is no longer a hard graph waypoint."
            )

    if not final_traversal.exists():
        runner.run_stage(
            "[1b/13] Grounding traversal: top_n fallback",
            "scripts.run_graph_traversal",
            [
                "--corpus-id", args.corpus_id,
                "--mode", "mechanism",
                "--algorithm", "top_n",
                "--source", args.source,
                "--target", args.target,
                "--node-map-k", str(args.node_map_k),
                "--endpoint-pair-k", str(args.endpoint_pair_k),
                "--max-depth", str(args.max_depth),
                "--top-k", str(args.top_k),
                "--include-candidate-paths",
                "--output", str(final_traversal),
            ],
            expected=[final_traversal],
        )
        topn_count = _returned_path_count(_load_json(final_traversal))
        runner.manifest["top_n_returned_path_count"] = topn_count
        runner.manifest["grounding_algorithm_used"] = "top_n"
        runner._save_manifest()
        if topn_count <= 0:
            raise RuntimeError(
                "Both semantic-stop grounding and top_n grounding produced zero paths. "
                "Do not continue into hypothesis generation with zero evidence."
            )

    # ------------------------------------------------------------------
    # 2-4. Grounded evidence context
    # ------------------------------------------------------------------
    packet = run / "explorer.packet.json"
    runner.run_stage(
        "[2/13] Build GraphExplorerPacket",
        "scripts.build_explorer_packet",
        [
            "--traversal-result", str(final_traversal),
            "--question", args.question,
            "--objective", args.objective,
            "--output", str(packet),
        ],
        expected=[packet],
    )

    explorer_prefix = run / "explorer"
    explorer_report = run / "explorer.report.json"
    runner.run_stage(
        "[3/13] Graph Explorer",
        "scripts.run_graph_explorer",
        [
            "--packet", str(packet),
            *_base_model_args(args),
            "--output-prefix", str(explorer_prefix),
            "--save-prompt",
        ],
        expected=[explorer_report],
    )

    context = run / "hypothesis.context.json"
    runner.run_stage(
        "[4/13] Build grounded HypothesisContext",
        "scripts.build_hypothesis_context",
        [
            "--packet", str(packet),
            "--report", str(explorer_report),
            "--output", str(context),
        ],
        expected=[context],
    )
    context_payload = _load_json(context)
    evidence_rows = context_payload.get("evidence_statements", [])
    evidence_count = len(evidence_rows) if isinstance(evidence_rows, list) else 0
    eligible_count = sum(
        1
        for row in evidence_rows
        if isinstance(row, dict) and bool(row.get("eligible_as_premise"))
    ) if isinstance(evidence_rows, list) else 0
    runner.manifest["grounded_evidence_statement_count"] = evidence_count
    runner.manifest["eligible_positive_premise_count"] = eligible_count
    runner._save_manifest()
    if evidence_count == 0:
        raise RuntimeError(
            "Grounding traversal produced paths, but HypothesisContext contains zero "
            "evidence statements. Stop before discovery-axis synthesis."
        )

    # ------------------------------------------------------------------
    # 5-7. Discovery lane and dual context
    # ------------------------------------------------------------------
    candidate_traversal = run / "candidate_unit.traversal.a3.json"
    runner.run_stage(
        "[5/13] Candidate-unit discovery",
        "scripts.run_candidate_unit_traversal",
        [
            "--corpus-id", args.corpus_id,
            "--source", args.source,
            "--target", args.target,
            "--node-map-k", str(args.node_map_k),
            "--max-depth", str(args.max_depth),
            "--top-k", str(max(args.top_k, 12)),
            "--include-candidate-paths",
            "--output", str(candidate_traversal),
        ],
        expected=[candidate_traversal],
    )

    bundle = run / "discovery.bundle.a3.json"
    runner.run_stage(
        "[6/13] DiscoveryBundle",
        "scripts.build_discovery_bundle",
        [
            "--traversal", str(final_traversal),
            "--traversal", str(candidate_traversal),
            "--top-k", str(args.discovery_top_k),
            "--output", str(bundle),
        ],
        expected=[bundle],
    )
    bundle_payload = _load_json(bundle)
    inspirations = bundle_payload.get("inspirations", [])
    if not isinstance(inspirations, list) or not inspirations:
        raise RuntimeError(
            "DiscoveryBundle contains zero inspirations. Canonical fallback is disabled."
        )

    dual_context = run / "hypothesis.dual_context.a3.json"
    runner.run_stage(
        "[7/13] Dual hypothesis context",
        "scripts.build_dual_hypothesis_context",
        [
            "--context", str(context),
            "--discovery-bundle", str(bundle),
            "--output", str(dual_context),
        ],
        expected=[dual_context],
    )

    # ------------------------------------------------------------------
    # 8-9. Alpha4 generation and semantic gate
    # ------------------------------------------------------------------
    axis_prefix = run / "hypothesis_axis_a4"
    axis_portfolio = run / "hypothesis_axis_a4.portfolio.json"
    axis_plan = run / "hypothesis_axis_a4.axis_plan.json"
    lineage = run / "hypothesis_axis_a4.lineage.json"
    runner.run_stage(
        "[8/13] Discovery-axis hypothesis synthesis",
        "scripts.run_discovery_axis_hypothesis_maker",
        [
            "--dual-context", str(dual_context),
            *_base_model_args(args),
            "--max-axes", str(args.max_axes),
            "--output-prefix", str(axis_prefix),
            "--save-prompts",
        ],
        expected=[axis_portfolio, axis_plan, lineage],
    )
    initial_hypotheses = _hypothesis_count(axis_portfolio)
    runner.manifest["initial_hypothesis_count"] = initial_hypotheses
    runner._save_manifest()
    if initial_hypotheses == 0:
        print(
            "No hypotheses survived alpha4. This is a valid fail-closed result; "
            "external novelty/refinement will not run."
        )
        runner.manifest["status"] = "complete_no_hypotheses_after_alpha4"
        runner.manifest["finished_at_utc"] = _now()
        runner._save_manifest()
        return 0

    semantic_a4_prefix = run / "semantic_axis_a4"
    semantic_a4_review = run / "semantic_axis_a4.review.json"
    runner.run_stage(
        "[9/13] Semantic critic: alpha4 portfolio",
        "scripts.run_hypothesis_semantic_critic",
        [
            "--context", str(context),
            "--portfolio", str(axis_portfolio),
            *_base_model_args(args, critic=True),
            "--output-prefix", str(semantic_a4_prefix),
            "--save-prompt",
        ],
        expected=[semantic_a4_review],
    )

    # ------------------------------------------------------------------
    # 10. External novelty. Fresh run directory + subprocess check=True
    #     prevent old report reuse after an assessor crash.
    # ------------------------------------------------------------------
    external_prefix = run / "external_novelty_a52"
    external_report = run / "external_novelty_a52.report.json"
    external_plan = run / "external_novelty_a52.claims_queries.json"
    external_prior = run / "external_novelty_a52.prior_art.json"
    runner.run_stage(
        "[10/13] External novelty alpha5.2",
        "scripts.run_external_novelty",
        [
            "--portfolio", str(axis_portfolio),
            "--lineage", str(lineage),
            *_base_model_args(args, critic=True),
            "--providers", args.providers,
            "--results-per-query", str(args.results_per_query),
            "--output-prefix", str(external_prefix),
            "--save-prompts",
        ],
        expected=[external_report, external_plan, external_prior],
    )
    current_portfolio_id = _portfolio_id(axis_portfolio)
    external_source_id = _external_source_portfolio_id(external_report)
    if external_source_id != current_portfolio_id:
        raise RuntimeError(
            "External novelty provenance mismatch immediately after stage 10: "
            f"portfolio={current_portfolio_id}, report_source={external_source_id}"
        )

    # ------------------------------------------------------------------
    # 11. Alpha6 targeted novelty refinement
    # ------------------------------------------------------------------
    refinement_prefix = run / "novelty_refinement_a6"
    refined_portfolio = run / "novelty_refinement_a6.portfolio.json"
    refined_report = run / "novelty_refinement_a6.report.json"
    runner.run_stage(
        "[11/13] Targeted novelty refinement alpha6",
        "scripts.run_novelty_refinement",
        [
            "--dual-context", str(dual_context),
            "--axis-plan", str(axis_plan),
            "--portfolio", str(axis_portfolio),
            "--lineage", str(lineage),
            "--external-report", str(external_report),
            "--external-query-plan", str(external_plan),
            "--external-prior-art", str(external_prior),
            "--model", args.model,
            "--critic-model", args.critic_model,
            *( ["--base-url", args.base_url] if args.base_url else [] ),
            "--api-key-env", args.api_key_env,
            "--providers", args.providers,
            "--results-per-query", str(args.results_per_query),
            "--output-prefix", str(refinement_prefix),
        ],
        expected=[refined_portfolio, refined_report],
    )
    final_hypotheses = _hypothesis_count(refined_portfolio)
    runner.manifest["final_hypothesis_count"] = final_hypotheses
    runner._save_manifest()
    if final_hypotheses == 0:
        alpha6_report_payload = _load_json(refined_report)
        if _alpha6_empty_is_degraded(alpha6_report_payload):
            raise RuntimeError(
                "Alpha6 produced an empty portfolio exclusively through deterministic "
                "compile/validation/provenance failures. This is a degraded pipeline "
                "state, not a scientific fail-closed result. Inspect attempt reason_codes."
            )
        print(
            "No hypotheses survived alpha6 after scientific/policy gates. "
            "Final semantic/feasibility stages are skipped."
        )
        runner.complete()
        return 0

    # ------------------------------------------------------------------
    # 12-13. Final semantic gate, feasibility, viewer
    # ------------------------------------------------------------------
    semantic_final_prefix = run / "semantic_final"
    semantic_final_review = run / "semantic_final.review.json"
    runner.run_stage(
        "[12/13] Final semantic critic",
        "scripts.run_hypothesis_semantic_critic",
        [
            "--context", str(context),
            "--portfolio", str(refined_portfolio),
            *_base_model_args(args, critic=True),
            "--output-prefix", str(semantic_final_prefix),
            "--save-prompt",
        ],
        expected=[semantic_final_review],
    )

    feasibility_dir = run / "feasibility_final"
    feasibility_manifest = feasibility_dir / "manifest.json"
    runner.run_stage(
        "[13a/13] Feasibility",
        "scripts.run_feasibility_e2e",
        [
            "--context", str(context),
            "--portfolio", str(refined_portfolio),
            "--semantic-review", str(semantic_final_review),
            "--output-dir", str(feasibility_dir),
        ],
        expected=[feasibility_manifest],
    )

    viewer = run / "demo" / "index.html"
    runner.run_stage(
        "[13b/13] Demo viewer",
        "scripts.build_demo_viewer",
        [
            "--run-dir", str(run),
            "--feasibility-dir", str(feasibility_dir),
            "--title", args.title,
        ],
        expected=[viewer],
    )
    runner.complete()
    print()
    print("Pipeline complete")
    print("Grounding algorithm:", runner.manifest["grounding_algorithm_used"])
    print("Initial hypotheses:", initial_hypotheses)
    print("Final hypotheses:", final_hypotheses)
    print("Viewer:", viewer)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-fast GraphAgentsDAC discovery E2E runner with semantic-stop -> "
            "top_n grounding fallback and stale-artifact/provenance guards."
        )
    )
    parser.add_argument("--corpus-id", default="dac_her_expanded_v1")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stop", default=None)
    parser.add_argument("--target", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--objective", default="explain_connection")
    parser.add_argument(
        "--title",
        default="GraphAgentsDAC Hypothesis Lineage & Validation Viewer",
    )
    parser.add_argument(
        "--grounding-policy",
        choices=("semantic_stop_fallback_top_n", "semantic_stop_only", "top_n"),
        default="semantic_stop_fallback_top_n",
    )
    parser.add_argument("--node-map-k", type=int, default=20)
    parser.add_argument("--waypoint-k", type=int, default=12)
    parser.add_argument("--endpoint-pair-k", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--discovery-top-k", type=int, default=8)
    parser.add_argument("--max-axes", type=int, default=5)
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_AGENT_MODEL"),
    )
    parser.add_argument(
        "--critic-model",
        default=(
            os.getenv("OPENROUTER_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--providers", default="semantic_scholar,crossref")
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help=(
            "Delete the specified run directory before starting. Without this flag, "
            "the runner refuses non-empty directories to prevent stale-artifact mixing."
        ),
    )
    return parser.parse_args()


def _mark_failed_manifest(run_dir: Path, exc: BaseException) -> None:
    path = run_dir / "e2e_runner.manifest.json"
    if not path.exists():
        return
    try:
        payload = _load_json(path)
        payload["status"] = "failed"
        payload["finished_at_utc"] = _now()
        payload["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        stages = payload.get("stages")
        if isinstance(stages, list) and stages:
            last = stages[-1]
            if isinstance(last, dict) and last.get("status") == "running":
                last["status"] = "failed"
                last["finished_at_utc"] = _now()
        _write_json(path, payload)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    try:
        return run_pipeline(args)
    except subprocess.CalledProcessError as exc:
        _mark_failed_manifest(Path(args.run_dir), exc)
        print(
            f"\nPIPELINE FAILED: stage command exited with status {exc.returncode}.",
            file=sys.stderr,
        )
        print(
            "Downstream stages were not executed. Re-run with a fresh --run-dir after fixing the cause.",
            file=sys.stderr,
        )
        return int(exc.returncode or 1)
    except Exception as exc:
        _mark_failed_manifest(Path(args.run_dir), exc)
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
