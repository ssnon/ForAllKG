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

from pipeline_core.domain_profile import ScientificDomainProfile
from domains.feasibility_registry import resolve_feasibility_adapter
from domains.registry import get_domain_profile
from pipeline_core.feasibility_domain import FeasibilityDomainAdapter
from dac_her.literature_provider_plan import (
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)


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


def _resolve_feasibility_capability(
    profile: ScientificDomainProfile,
) -> FeasibilityDomainAdapter | None:
    # Missing capability is a valid multidomain state. A profile that
    # explicitly names an adapter still resolves strictly, so unknown or
    # cross-domain adapters remain errors rather than being silently skipped.
    adapter_id = (profile.feasibility_adapter_id or "").strip()
    if not adapter_id:
        return None
    return resolve_feasibility_adapter(profile)


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
            "domain_profile_id": args.domain_profile,
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

    def skip_stage(self, name: str, *, reason: str) -> None:
        record: dict[str, Any] = {
            "name": name,
            "module": None,
            "started_at_utc": _now(),
            "finished_at_utc": _now(),
            "status": "skipped",
            "reason": reason,
        }
        self.manifest["stages"].append(record)
        print()
        print("=" * 72)
        print(name)
        print("=" * 72)
        print("SKIPPED:", reason)
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


def _data_root_args(
    args: argparse.Namespace,
) -> list[str]:
    value = str(
        args.data_root or ""
    ).strip()

    return (
        ["--data-root", value]
        if value
        else []
    )


def _mechanism_index_args(
    args: argparse.Namespace,
) -> list[str]:
    value = str(
        args.data_root or ""
    ).strip()

    if not value:
        return []

    index_dir = (
        Path(value)
        / "corpus"
        / args.corpus_id
        / "mechanism"
        / "navigation"
        / "node_index"
    )

    return [
        "--index-dir",
        str(index_dir),
    ]


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

    provider_arg = str(args.providers or "").strip()
    if provider_arg.lower() == "auto":
        provider_requested = None
    else:
        provider_requested = [
            x.strip()
            for x in provider_arg.split(",")
            if x.strip()
        ]
    literature_provider_plan = resolve_literature_provider_plan(
        requested=provider_requested
    )
    require_standard_or_full_auto_plan(
        literature_provider_plan
    )
    literature_provider_plan_path = (
        runner.run_dir / "literature_provider_plan.json"
    )
    _write_json(
        literature_provider_plan_path,
        literature_provider_plan.model_dump(mode="json"),
    )
    runner.manifest["literature_provider_plan"] = {
        "plan_id": literature_provider_plan.plan_id,
        "plan_sha256": literature_provider_plan.plan_sha256,
        "requested_mode": literature_provider_plan.requested_mode,
        "mode": literature_provider_plan.mode,
        "active_providers": list(
            literature_provider_plan.active_providers
        ),
        "artifact": str(literature_provider_plan_path),
        "provider_set_frozen_for_run": True,
        "runtime_failure_changes_provider_set": False,
        "secret_values_persisted": False,
    }
    runner._save_manifest()

    _check_alpha6_available()
    domain_profile = get_domain_profile(args.domain_profile)
    feasibility_adapter = _resolve_feasibility_capability(domain_profile)
    runner.manifest["domain_profile_id"] = domain_profile.profile_id
    runner.manifest["capabilities"] = {
        "feasibility": feasibility_adapter is not None,
    }
    runner.manifest["feasibility_status"] = (
        "available"
        if feasibility_adapter is not None
        else "not_supported_for_domain"
    )
    runner.manifest["feasibility_adapter_id"] = (
        feasibility_adapter.adapter_id
        if feasibility_adapter is not None
        else None
    )
    runner._save_manifest()
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
                "--domain-profile", domain_profile.profile_id,
                *_data_root_args(args),
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
                "--domain-profile", domain_profile.profile_id,
                *_data_root_args(args),
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
            "--domain-profile", domain_profile.profile_id,
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
    evidence_compression = (
        run / "hypothesis_context.evidence_compression.json"
    )
    evidence_family_diagnostics = (
        run / "hypothesis_context.evidence_family_diagnostics.json"
    )
    path_lineage_diagnostics = (
        run / "hypothesis_context.path_lineage_diagnostics.json"
    )
    path_lineage_propagation = (
        run / "hypothesis_context.path_lineage_propagation.json"
    )
    runner.run_stage(
        "[4/13] Build grounded HypothesisContext",
        "scripts.build_hypothesis_context",
        [
            "--packet", str(packet),
            "--report", str(explorer_report),
            "--output", str(context),
            "--compression-output", str(evidence_compression),
            "--family-diagnostics-output", str(evidence_family_diagnostics),
            "--path-lineage-output", str(path_lineage_diagnostics),
            *(
                ["--disable-path-lineage-propagation"]
                if args.disable_path_lineage_propagation
                else [
                    "--path-lineage-propagation-output",
                    str(path_lineage_propagation),
                ]
            ),
        ],
        expected=[
            context,
            evidence_compression,
            evidence_family_diagnostics,
            path_lineage_diagnostics,
            *(
                []
                if args.disable_path_lineage_propagation
                else [path_lineage_propagation]
            ),
        ],
    )
    context_payload = _load_json(context)
    compression_payload = _load_json(evidence_compression)
    family_diagnostics_payload = _load_json(
        evidence_family_diagnostics
    )
    path_lineage_payload = _load_json(
        path_lineage_diagnostics
    )
    propagation_payload = (
        {}
        if args.disable_path_lineage_propagation
        else _load_json(path_lineage_propagation)
    )
    evidence_rows = context_payload.get("evidence_statements", [])
    evidence_count = len(evidence_rows) if isinstance(evidence_rows, list) else 0
    eligible_count = sum(
        1
        for row in evidence_rows
        if isinstance(row, dict) and bool(row.get("eligible_as_premise"))
    ) if isinstance(evidence_rows, list) else 0
    runner.manifest["grounded_evidence_statement_count"] = evidence_count
    runner.manifest["eligible_positive_premise_count"] = eligible_count
    runner.manifest["evidence_compression"] = {
        "report_id": compression_payload.get("report_id"),
        "selected_path_paper_count": compression_payload.get(
            "selected_path_paper_count"
        ),
        "explorer_statement_paper_count": compression_payload.get(
            "explorer_statement_paper_count"
        ),
        "eligible_premise_paper_count": compression_payload.get(
            "eligible_premise_paper_count"
        ),
        "eligible_statement_count": compression_payload.get(
            "eligible_statement_count"
        ),
        "eligible_multi_paper_statement_count": compression_payload.get(
            "eligible_multi_paper_statement_count"
        ),
        "mean_papers_per_eligible_statement": compression_payload.get(
            "mean_papers_per_eligible_statement"
        ),
        "eligible_papers_only_in_multi_paper_statements_count": compression_payload.get(
            "eligible_papers_only_in_multi_paper_statements_count"
        ),
        "eligible_multi_paper_heterogeneous_profile_count": compression_payload.get(
            "eligible_multi_paper_heterogeneous_profile_count"
        ),
        "statements_with_declared_support_mismatch_count": compression_payload.get(
            "statements_with_declared_support_mismatch_count"
        ),
        "diagnostic_only": True,
        "scientific_selection_changed": False,
    }
    runner.manifest["evidence_family_diagnostics"] = {
        "report_id": family_diagnostics_payload.get("report_id"),
        "decomposition_candidate_count": family_diagnostics_payload.get(
            "decomposition_candidate_count"
        ),
        "decomposition_candidate_statement_ids": family_diagnostics_payload.get(
            "decomposition_candidate_statement_ids"
        ),
        "eligible_homogeneous_multi_paper_statement_count": (
            family_diagnostics_payload.get(
                "eligible_homogeneous_multi_paper_statement_count"
            )
        ),
        "eligible_heterogeneous_multi_paper_statement_count": (
            family_diagnostics_payload.get(
                "eligible_heterogeneous_multi_paper_statement_count"
            )
        ),
        "eligible_statements_without_explicit_path_lineage_count": (
            family_diagnostics_payload.get(
                "eligible_statements_without_explicit_path_lineage_count"
            )
        ),
        "eligible_statements_without_explicit_path_lineage_fraction": (
            family_diagnostics_payload.get(
                "eligible_statements_without_explicit_path_lineage_fraction"
            )
        ),
        "diagnostic_only": True,
        "scientific_selection_changed": False,
        "automatic_statement_decomposition_allowed": False,
    }
    runner.manifest["path_lineage_diagnostics"] = {
        "report_id": path_lineage_payload.get("report_id"),
        "selected_path_count": path_lineage_payload.get(
            "selected_path_count"
        ),
        "selected_mechanistic_path_count": path_lineage_payload.get(
            "selected_mechanistic_path_count"
        ),
        "eligible_statement_count": path_lineage_payload.get(
            "eligible_statement_count"
        ),
        "eligible_with_explicit_path_lineage_count": path_lineage_payload.get(
            "eligible_with_explicit_path_lineage_count"
        ),
        "eligible_with_deterministic_attribution_count": path_lineage_payload.get(
            "eligible_with_deterministic_attribution_count"
        ),
        "eligible_with_deterministic_mechanistic_attribution_count": path_lineage_payload.get(
            "eligible_with_deterministic_mechanistic_attribution_count"
        ),
        "recoverable_missing_explicit_path_lineage_count": path_lineage_payload.get(
            "recoverable_missing_explicit_path_lineage_count"
        ),
        "unrecoverable_missing_explicit_path_lineage_count": path_lineage_payload.get(
            "unrecoverable_missing_explicit_path_lineage_count"
        ),
        "diagnostic_only": True,
        "scientific_selection_changed": False,
        "automatic_path_propagation_allowed": False,
    }
    runner.manifest["path_lineage_propagation"] = {
        "enabled": not args.disable_path_lineage_propagation,
        "report_id": propagation_payload.get("report_id"),
        "propagated_statement_count": propagation_payload.get("propagated_statement_count"),
        "eligible_statement_count": propagation_payload.get("eligible_statement_count"),
        "total_propagated_path_id_count": propagation_payload.get("total_propagated_path_id_count"),
        "pre_explicit_path_lineage_statement_count": propagation_payload.get("pre_explicit_path_lineage_statement_count"),
        "post_explicit_path_lineage_statement_count": propagation_payload.get("post_explicit_path_lineage_statement_count"),
        "scientific_support_changed_statement_count": propagation_payload.get("scientific_support_changed_statement_count"),
        "premise_eligibility_changed_statement_count": propagation_payload.get("premise_eligibility_changed_statement_count"),
        "mode": "minimal_deterministic_cover",
    }
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
            "--domain-profile", domain_profile.profile_id,
            *_data_root_args(args),
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
            "--domain-profile", domain_profile.profile_id,
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
    axis_evidence_diversity = (
        run / "hypothesis_axis_a4.evidence_diversity.json"
    )
    runner.run_stage(
        "[8/13] Discovery-axis hypothesis synthesis",
        "scripts.run_discovery_axis_hypothesis_maker",
        [
            "--dual-context", str(dual_context),
            *_base_model_args(args),
            *_mechanism_index_args(args),
            "--max-axes", str(args.max_axes),
            "--parse-retries", str(args.hypothesis_parse_retries),
            "--output-prefix", str(axis_prefix),
            "--save-prompts",
        ],
        expected=[
            axis_portfolio,
            axis_plan,
            lineage,
            axis_evidence_diversity,
        ],
    )
    initial_hypotheses = _hypothesis_count(axis_portfolio)
    diversity_payload = _load_json(axis_evidence_diversity)
    runner.manifest["initial_hypothesis_count"] = initial_hypotheses
    runner.manifest["hypothesis_evidence_diversity"] = {
        "report_id": diversity_payload.get("report_id"),
        "eligible_statement_count": diversity_payload.get(
            "eligible_statement_count"
        ),
        "used_statement_count": diversity_payload.get(
            "used_statement_count"
        ),
        "eligible_statement_coverage": diversity_payload.get(
            "eligible_statement_coverage"
        ),
        "shared_core_statement_count": diversity_payload.get(
            "shared_core_statement_count"
        ),
        "distinct_premise_set_count": diversity_payload.get(
            "distinct_premise_set_count"
        ),
        "exact_premise_set_duplicate_group_count": diversity_payload.get(
            "exact_premise_set_duplicate_group_count"
        ),
        "mean_pairwise_statement_jaccard": diversity_payload.get(
            "mean_pairwise_statement_jaccard"
        ),
        "max_pairwise_statement_jaccard": diversity_payload.get(
            "max_pairwise_statement_jaccard"
        ),
        "diagnostic_only": True,
        "scientific_selection_changed": False,
    }
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
            "--domain-profile", domain_profile.profile_id,
            "--lineage", str(lineage),
            *_base_model_args(args, critic=True),
            "--provider-plan", str(literature_provider_plan_path),
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
            "--domain-profile", domain_profile.profile_id,
            "--axis-plan", str(axis_plan),
            "--portfolio", str(axis_portfolio),
            "--lineage", str(lineage),
            "--external-report", str(external_report),
            "--external-query-plan", str(external_plan),
            "--external-prior-art", str(external_prior),
            *_mechanism_index_args(args),
            "--model", args.model,
            "--critic-model", args.critic_model,
            *( ["--base-url", args.base_url] if args.base_url else [] ),
            "--api-key-env", args.api_key_env,
            "--provider-plan", str(literature_provider_plan_path),
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
    viewer = run / "demo" / "index.html"

    if feasibility_adapter is not None:
        feasibility_manifest = feasibility_dir / "manifest.json"
        runner.run_stage(
            "[13a/13] Feasibility",
            "scripts.run_feasibility_e2e",
            [
                "--context", str(context),
                "--domain-profile", domain_profile.profile_id,
                "--portfolio", str(refined_portfolio),
                "--semantic-review", str(semantic_final_review),
                "--output-dir", str(feasibility_dir),
            ],
            expected=[feasibility_manifest],
        )
        runner.manifest["feasibility_status"] = "complete"
        runner._save_manifest()
    else:
        runner.skip_stage(
            "[13a/13] Feasibility",
            reason=(
                f"Scientific domain profile {domain_profile.profile_id!r} does not "
                "declare a feasibility adapter. Core discovery/refinement output is "
                "still complete; another domain's feasibility rules will not be used."
            ),
        )

    viewer_args = [
        "--run-dir", str(run),
        "--title", args.title,
    ]
    if feasibility_adapter is not None:
        viewer_args += ["--feasibility-dir", str(feasibility_dir)]

    runner.run_stage(
        "[13b/13] Demo viewer",
        "scripts.build_demo_viewer",
        viewer_args,
        expected=[viewer],
    )
    runner.manifest["viewer_status"] = (
        "complete_with_feasibility"
        if feasibility_adapter is not None
        else "complete_core_without_feasibility"
    )
    runner._save_manifest()

    runner.complete()
    print()
    print("Pipeline complete")
    print("Grounding algorithm:", runner.manifest["grounding_algorithm_used"])
    print("Initial hypotheses:", initial_hypotheses)
    print("Final hypotheses:", final_hypotheses)
    print("Feasibility:", runner.manifest["feasibility_status"])
    print(
        "Viewer:",
        viewer if viewer.exists() else runner.manifest.get("viewer_status", "not_generated"),
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-fast GraphAgentsDAC discovery E2E runner with semantic-stop -> "
            "top_n grounding fallback and stale-artifact/provenance guards."
        )
    )
    parser.add_argument("--corpus-id", default="dac_her_expanded_v1")
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Override the scientific-domain data root for "
            "grounding and candidate-unit traversal stages. "
            "When omitted, child stages retain the domain "
            "adapter default."
        ),
    )
    parser.add_argument(
        "--domain-profile",
        default="dac_her",
        help="Scientific domain profile propagated through discovery, novelty, refinement, and feasibility.",
    )
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
        "--hypothesis-parse-retries",
        type=int,
        default=3,
        help=(
            "Instructor structured-output retries for discovery-axis hypothesis "
            "generation. Retries are used only when the model output fails the "
            "strict HypothesisPortfolioDraft schema."
        ),
    )
    parser.add_argument(
        "--disable-path-lineage-propagation",
        action="store_true",
        help="Disable PL1-B minimal deterministic path-lineage repair.",
    )
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
    parser.add_argument(
        "--providers",
        default="auto",
        help=(
            "Literature provider set. Default 'auto' freezes OpenAlex+Crossref "
            "for the whole E2E run, adding Semantic Scholar only when "
            "SEMANTIC_SCHOLAR_API_KEY is configured."
        ),
    )
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
