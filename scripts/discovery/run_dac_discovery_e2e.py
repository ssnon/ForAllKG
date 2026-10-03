from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_core.domain.domain_profile import ScientificDomainProfile
from domains.context_review_registry import (
    available_context_review_profiles,
    get_context_review_adapter,
)
from domains.candidate_unit_applicability_registry import (
    available_candidate_unit_applicability_profiles,
    get_candidate_unit_applicability_adapter,
)
from domains.feasibility_registry import resolve_feasibility_adapter
from domains.registry import get_domain_profile
from pipeline_core.domain.feasibility_domain import FeasibilityDomainAdapter
from pipeline_core.discovery.prior_art_provider_plan import (
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


def _sha256_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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



def _resolve_context_review_capability(
    profile: ScientificDomainProfile,
) -> Any | None:
    """Resolve optional scientific-context capability without mutating
    ScientificDomainProfile identity.

    Absence is a valid multidomain state; registered capability remains
    domain-specific and fail-closed through the context registry.
    """

    if (
        profile.profile_id
        not in available_context_review_profiles()
    ):
        return None

    return get_context_review_adapter(
        profile.profile_id
    )


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


_POST_GENERATION_N10_AUTHORITY_MODES = (
    "hard_filter",
    "certification_only",
)


def _post_generation_n10_output_contract(
    run: Path,
    authority_mode: str,
) -> dict[str, Path | None]:
    if authority_mode == "hard_filter":
        legacy_portfolio = run / "novelty_refinement_a6.n10.portfolio.json"
        legacy_report = run / "novelty_refinement_a6.n10.enforcement.json"
        return {
            "legacy_output_portfolio": legacy_portfolio,
            "legacy_output_report": legacy_report,
            "scientific_candidate_portfolio": None,
            "certification_report": None,
            "certified_novelty_portfolio": None,
            "downstream_portfolio": legacy_portfolio,
        }

    if authority_mode == "certification_only":
        candidate = (
            run / "novelty_refinement_a6.n10.candidate.portfolio.json"
        )
        certification = (
            run / "novelty_refinement_a6.n10.certification.json"
        )
        certified = (
            run / "novelty_refinement_a6.n10.certified.portfolio.json"
        )
        return {
            "legacy_output_portfolio": None,
            "legacy_output_report": None,
            "scientific_candidate_portfolio": candidate,
            "certification_report": certification,
            "certified_novelty_portfolio": certified,
            "downstream_portfolio": candidate,
        }

    raise ValueError(
        "unsupported post-generation N10 authority mode: " + authority_mode
    )


def _post_generation_n10_consumer_contract(
    authority_mode: str,
) -> dict[str, str]:
    if authority_mode == "hard_filter":
        return {
            "stage12_semantic": "legacy_n10_filtered_portfolio",
            "stage13_feasibility": "legacy_n10_filtered_portfolio",
            "demo_viewer": "legacy_alpha6_portfolio",
            "novelty_certified_export": "legacy_n10_filtered_portfolio",
        }

    if authority_mode == "certification_only":
        return {
            "stage12_semantic": "scientific_candidate_portfolio",
            "stage13_feasibility": "scientific_candidate_portfolio",
            "demo_viewer":
                "scientific_candidate_portfolio+certification_report",
            "novelty_certified_export": "certified_novelty_portfolio",
        }

    raise ValueError(
        "unsupported post-generation N10 authority mode: " + authority_mode
    )


def _open_world_discovery_output_contract(
    run: Path,
) -> dict[str, Path]:
    prefix = run / "hypothesis_axis_open_world"
    return {
        "prefix": prefix,
        "provider_plan":
            Path(str(prefix) + ".provider_plan.json"),
        "retrieval":
            Path(str(prefix) + ".retrieval.json"),
        "selected_works":
            Path(str(prefix) + ".selected_works.json"),
        "axis_synthesis":
            Path(str(prefix) + ".axis_synthesis.json"),
        "axis_validation":
            Path(str(prefix) + ".axis_validation.json"),
        "external_axis_plan":
            Path(str(prefix) + ".external_axis_plan.json"),
        "external_axis_bundle":
            Path(str(prefix) + ".external_axis_bundle.json"),
        "external_axis_rejections":
            Path(str(prefix) + ".external_axis_rejections.json"),
        "manifest":
            Path(str(prefix) + ".manifest.json"),
        "prompt":
            Path(str(prefix) + ".axis_synthesis_prompt.json"),
    }


def _validate_open_world_discovery_parent_manifest(
    payload: dict[str, Any],
    *,
    expected_provider_plan_sha256: str,
) -> None:
    if (
        payload.get("provider_plan_sha256")
        != expected_provider_plan_sha256
    ):
        raise RuntimeError(
            "Open-world discovery provider plan drifted from "
            "the frozen parent E2E provider plan."
        )

    expected_false = (
        "positive_premise_authority_changed",
        "novelty_authority_created",
        "hypothesis_generation_performed",
        "canonical_fallback_authorized",
    )
    for key in expected_false:
        if payload.get(key) is not False:
            raise RuntimeError(
                "Open-world discovery authority contract violated: "
                f"{key} must be false."
            )

    if (
        payload.get("external_literature_authority")
        != "INSPIRATION_ONLY"
    ):
        raise RuntimeError(
            "Open-world discovery literature authority must remain "
            "INSPIRATION_ONLY."
        )

    try:
        external_axis_count = int(
            payload.get("external_axis_count", 0)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Open-world discovery external_axis_count is invalid."
        ) from exc

    if external_axis_count <= 0:
        raise RuntimeError(
            "Open-world discovery produced zero validated external axes. "
            "Do not silently fall back to the persistent-KG axis plan."
        )


def _stage8_axis_plan_input(
    *,
    control_plan: Path,
    open_world_plan: Path | None,
    open_world_enabled: bool,
) -> Path:
    if not open_world_enabled:
        return control_plan
    if open_world_plan is None:
        raise RuntimeError(
            "Open-world discovery is enabled but no external axis plan "
            "was supplied to stage 8."
        )
    return open_world_plan


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
            "candidate_unit_policy": {
                "min_candidate_unit_score": float(
                    getattr(
                        args,
                        "min_candidate_unit_score",
                        0.30,
                    )
                ),
                "shared_between_discovery_bundle_and_alpha4": True,
            },
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


def _finish_after_initial_semantic_if_requested(
    *,
    runner: PipelineRunner,
    args: argparse.Namespace,
    portfolio_path: Path,
    context_path: Path,
    semantic_run_path: Path,
    semantic_review_path: Path,
    provider_plan_path: Path,
) -> bool:
    if not bool(getattr(args, "stop_after_initial_semantic", False)):
        return False

    required = {
        "portfolio": portfolio_path,
        "hypothesis_context": context_path,
        "semantic_run": semantic_run_path,
        "provider_plan": provider_plan_path,
    }
    missing = [
        name
        for name, path in required.items()
        if not path.expanduser().resolve().is_file()
    ]
    if missing:
        raise RuntimeError(
            "Prospective initial-semantic cut point is missing required "
            "artifacts: " + repr(missing)
        )

    artifacts: dict[str, object] = {}
    for name, path in required.items():
        resolved = path.expanduser().resolve()
        artifacts[name] = {
            "path": str(resolved),
            "file_sha256": _sha256_file(resolved),
        }

    review_file = semantic_review_path.expanduser().resolve()
    artifacts["semantic_review"] = (
        {
            "path": str(review_file),
            "file_sha256": _sha256_file(review_file),
        }
        if review_file.is_file()
        else None
    )

    runner.manifest["prospective_initial_semantic_cutpoint"] = {
        "enabled": True,
        "cut_after_stage": 9,
        "artifacts": artifacts,
        "semantic_review_optional_after_hard_gate_failure": True,
        "external_novelty_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "refinement_performed": False,
        "final_semantic_performed": False,
        "feasibility_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    runner.manifest["status"] = "complete_after_initial_semantic"
    runner.manifest["finished_at_utc"] = _now()
    runner._save_manifest()
    return True


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
        __import__("scripts.discovery.run_novelty_refinement")
    except Exception as exc:
        raise RuntimeError(
            "scripts.discovery.run_novelty_refinement is unavailable. Apply the alpha6 targeted "
            "novelty-refinement bundle before using this full E2E runner."
        ) from exc


_RCF_CORRECTED_CANDIDATE_MAX_DEPTH = 13


def _candidate_unit_correction_contract(
    args: argparse.Namespace,
    profile: ScientificDomainProfile,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve the atomic RCF candidate-routing correction chain.

    Grounding traversal depth remains controlled by ``args.max_depth``.
    Candidate depth 13 is activated only when a domain-owned applicability
    adapter explicitly supports the requested semantic stop.
    """

    legacy = (
        ["--max-depth", str(args.max_depth)],
        {
            "active": False,
            "candidate_max_depth": int(args.max_depth),
            "grounding_max_depth": int(args.max_depth),
            "semantic_stop": args.stop,
            "reason": "unsupported_or_absent_stop",
            "corrected_route_contract": False,
            "owner_gate": False,
            "stop_conditioned_relevance": False,
        },
    )

    stop = str(args.stop or "").strip()

    if not stop:
        return legacy

    if (
        profile.profile_id
        not in available_candidate_unit_applicability_profiles()
    ):
        return legacy

    adapter = get_candidate_unit_applicability_adapter(
        profile.profile_id
    )

    if not adapter.supports_stop(stop):
        return legacy

    return (
        [
            "--semantic-stop",
            stop,
            "--corrected-route-contract",
            "--max-depth",
            str(_RCF_CORRECTED_CANDIDATE_MAX_DEPTH),
        ],
        {
            "active": True,
            "candidate_max_depth":
                _RCF_CORRECTED_CANDIDATE_MAX_DEPTH,
            "grounding_max_depth": int(args.max_depth),
            "semantic_stop": stop,
            "applicability_adapter_id":
                adapter.adapter_id,
            "relevance_stop_context":
                adapter.relevance_context(stop),
            "corrected_route_contract": True,
            "owner_gate": True,
            "stop_conditioned_relevance": True,
            "score_first": True,
            "semantic_state_retention": True,
            "corrected_provenance": True,
            "narrow_carrier_reuse": True,
            "candidate_anchor_reuse_allowed": False,
            "threshold_changed": False,
            "score_weights_changed": False,
            "grounding_depth_changed": False,
        },
    )



def _run_question_task_preservation_shadow_chain(
    *,
    runner: PipelineRunner,
    args: argparse.Namespace,
    run: Path,
    semantic_conflict_shadow: Path,
    final_traversal: Path,
    candidate_traversal: Path,
) -> tuple[Path, Path]:
    """Run the complete question-task preservation chain in shadow mode.

    This helper produces diagnostics only. It does not mutate DiscoveryBundle
    inspirations, discovery axes, hypotheses, or any production selection.
    """

    responsiveness_shadow = (
        run
        / "question_task_preservation."
          "responsiveness.shadow.json"
    )

    pair_proposals_shadow = (
        run
        / "question_task_preservation."
          "pair_proposals.shadow.json"
    )

    responsiveness_telemetry = (
        run
        / "question_task_preservation."
          "responsiveness.telemetry.jsonl"
    )

    responsiveness_debug_dir = (
        run
        / "question_task_preservation."
          "responsiveness_debug"
    )

    responsiveness_model = (
        args.critic_model
        or args.model
    )

    if not responsiveness_model:
        raise RuntimeError(
            "--question-task-preservation-shadow "
            "requires --critic-model or --model."
        )

    task_preservation_group = (
        run.name
        or "runtime"
    )

    runner.run_stage(
        "[6S-a/13] Question-task conflict responsiveness shadow",
        "scripts.discovery."
        "run_question_task_conflict_responsiveness",
        [
            "--raw-conflicts",
            str(
                semantic_conflict_shadow
            ),
            "--traversal",
            str(
                final_traversal
            ),
            "--traversal",
            str(
                candidate_traversal
            ),
            "--group",
            task_preservation_group,
            "--question",
            args.question,
            "--model",
            str(
                responsiveness_model
            ),
            "--reasoning-effort",
            "medium",
            "--temperature",
            "0",
            "--telemetry-path",
            str(
                responsiveness_telemetry
            ),
            "--debug-dir",
            str(
                responsiveness_debug_dir
            ),
            "--output",
            str(
                responsiveness_shadow
            ),
        ],
        expected=[
            responsiveness_shadow
        ],
    )

    runner.run_stage(
        "[6S-b/13] Question-task pair arbitration shadow",
        "scripts.discovery."
        "build_question_task_preservation_shadow",
        [
            "--raw-conflicts",
            str(
                semantic_conflict_shadow
            ),
            "--responsiveness-audit",
            str(
                responsiveness_shadow
            ),
            "--group",
            task_preservation_group,
            "--output",
            str(
                pair_proposals_shadow
            ),
        ],
        expected=[
            pair_proposals_shadow
        ],
    )

    return (
        responsiveness_shadow,
        pair_proposals_shadow,
    )



def _run_scientific_novelty_action_shadow_chain(
    *,
    runner: PipelineRunner,
    args: argparse.Namespace,
    run: Path,
    external_report: Path,
    external_plan: Path,
    external_prior: Path,
) -> Path:
    """Materialize the complete N1 novelty-action signal chain.

    This chain is observational only. It creates deterministic scientific
    distinctiveness, two semantic-distinctiveness passes per hypothesis,
    and N1 action decisions. It does not modify Alpha6 inputs or production
    selection.
    """

    scientific_report = (
        run
        / "scientific_distinctiveness_a10.shadow.json"
    )

    report_suffix = ".report.json"
    if not external_report.name.endswith(report_suffix):
        raise RuntimeError(
            "Scientific novelty shadow expected an external report ending "
            f"with {report_suffix!r}: {external_report}"
        )

    external_stem = external_report.name[:-len(report_suffix)]
    external_diagnostic_plan = (
        external_report.parent
        / f"{external_stem}.diagnostic_queries.json"
    )
    external_diagnostic_prior = (
        external_report.parent
        / f"{external_stem}.diagnostic_prior_art.json"
    )
    external_diagnostic_review = (
        external_report.parent
        / f"{external_stem}.diagnostic_review.json"
    )

    diagnostic_paths = (
        external_diagnostic_plan,
        external_diagnostic_prior,
        external_diagnostic_review,
    )
    diagnostic_presence = tuple(
        path.is_file()
        for path in diagnostic_paths
    )

    if any(diagnostic_presence) and not all(diagnostic_presence):
        raise RuntimeError(
            "External novelty diagnostic provenance is partial: "
            f"query_plan={external_diagnostic_plan.is_file()} "
            f"prior_art={external_diagnostic_prior.is_file()} "
            f"review={external_diagnostic_review.is_file()}"
        )

    scientific_diagnostic_args = (
        [
            "--external-diagnostic-query-plan",
            str(external_diagnostic_plan),
            "--external-diagnostic-prior-art",
            str(external_diagnostic_prior),
            "--external-diagnostic-review",
            str(external_diagnostic_review),
        ]
        if all(diagnostic_presence)
        else []
    )
    semantic_diagnostic_args = (
        [
            "--external-diagnostic-prior-art",
            str(external_diagnostic_prior),
            "--external-diagnostic-review",
            str(external_diagnostic_review),
        ]
        if all(diagnostic_presence)
        else []
    )

    runner.run_stage(
        "[10S-a/13] Scientific distinctiveness shadow",
        "scripts.discovery."
        "run_scientific_distinctiveness_diagnostic",
        [
            "--external-report",
            str(external_report),
            "--external-query-plan",
            str(external_plan),
            "--external-prior-art",
            str(external_prior),
            *scientific_diagnostic_args,
            "--output",
            str(scientific_report),
        ],
        expected=[
            scientific_report
        ],
    )

    external_payload = _load_json(
        external_report
    )

    cards = external_payload.get(
        "cards"
    )

    if not isinstance(cards, list):
        raise RuntimeError(
            "External novelty report cards must be a list "
            "for scientific novelty action shadow."
        )

    hypothesis_ids: list[str] = []
    seen_ids: set[str] = set()

    for card in cards:
        if not isinstance(card, dict):
            raise RuntimeError(
                "External novelty report card must be an object."
            )

        hypothesis_id = str(
            card.get(
                "hypothesis_id"
            )
            or ""
        ).strip()

        if not hypothesis_id:
            raise RuntimeError(
                "External novelty report card is missing hypothesis_id."
            )

        if hypothesis_id in seen_ids:
            raise RuntimeError(
                "Duplicate external novelty hypothesis_id: "
                f"{hypothesis_id}"
            )

        seen_ids.add(
            hypothesis_id
        )
        hypothesis_ids.append(
            hypothesis_id
        )

    if not hypothesis_ids:
        raise RuntimeError(
            "Scientific novelty action shadow received zero hypotheses."
        )

    semantic_model = (
        args.critic_model
        or args.model
    )

    if not semantic_model:
        raise RuntimeError(
            "--scientific-novelty-action-shadow requires "
            "--critic-model or --model."
        )

    semantic_paths: list[Path] = []

    for index, hypothesis_id in enumerate(
        hypothesis_ids,
        start=1,
    ):
        for pass_index in (1, 2):
            semantic_path = (
                run
                / (
                    "semantic_distinctiveness_a10."
                    f"h{index:02d}.pass_{pass_index}.shadow.json"
                )
            )

            semantic_paths.append(
                semantic_path
            )

            runner.run_stage(
                (
                    "[10S-b/13] Semantic distinctiveness shadow "
                    f"h{index:02d} pass {pass_index}"
                ),
                "scripts.discovery."
                "run_semantic_distinctiveness_review",
                [
                    "--scientific-report",
                    str(scientific_report),
                    "--external-report",
                    str(external_report),
                    "--external-prior-art",
                    str(external_prior),
                    *semantic_diagnostic_args,
                    "--hypothesis-id",
                    hypothesis_id,
                    "--output",
                    str(semantic_path),
                    "--model",
                    str(semantic_model),
                    "--review-pass-index",
                    str(pass_index),
                    "--temperature",
                    "0",
                    "--reasoning-effort",
                    "medium",
                ],
                expected=[
                    semantic_path
                ],
            )

    action_batch = (
        run
        / "scientific_novelty_actions_a10.shadow.json"
    )

    action_args = [
        "--external-report",
        str(external_report),
    ]

    for semantic_path in semantic_paths:
        action_args.extend(
            [
                "--semantic-review",
                str(semantic_path),
            ]
        )

    action_args.extend(
        [
            "--output",
            str(action_batch),
        ]
    )

    runner.run_stage(
        "[10S-c/13] Scientific novelty action shadow",
        "scripts.discovery."
        "build_scientific_novelty_action_shadow",
        action_args,
        expected=[
            action_batch
        ],
    )

    return action_batch


def _run_realization_candidate_chain(
    *,
    runner: PipelineRunner,
    args: argparse.Namespace,
    slot_index: int,
    slot_run: Path,
    dual_context: Path,
    frozen_axis_plan: Path,
    domain_profile_id: str,
    literature_provider_plan_path: Path,
    context_review_enabled: bool,
    external_axis_bundle: Path | None = None,
) -> dict[str, object]:
    """Run one independent realization over an already frozen axis plan.

    This helper deliberately does not perform production winner
    selection.  It materializes one independent candidate trajectory:

        frozen axis plan
            -> Alpha4 synthesis
            -> generic semantic critic
            -> external novelty
            -> scientific distinctiveness
            -> two semantic-distinctiveness passes

    A zero-hypothesis Alpha4 result is a valid realization-level
    failure and is returned as data rather than terminating the parent
    best-of-k search.

    The axis plan is never rebuilt here.
    """

    if slot_index < 0:
        raise ValueError(
            "realization slot_index must be non-negative"
        )

    if not frozen_axis_plan.is_file():
        raise FileNotFoundError(
            "Frozen realization-search axis plan missing: "
            f"{frozen_axis_plan}"
        )

    slot_run.mkdir(
        parents=True,
        exist_ok=True,
    )

    axis_prefix = (
        slot_run
        / "hypothesis_axis_a4"
    )

    axis_portfolio = Path(
        str(axis_prefix)
        + ".portfolio.json"
    )

    axis_plan_copy = Path(
        str(axis_prefix)
        + ".axis_plan.json"
    )

    lineage = Path(
        str(axis_prefix)
        + ".lineage.json"
    )

    axis_inference = Path(
        str(axis_prefix)
        + ".inference.json"
    )

    axis_context = Path(
        str(axis_prefix)
        + ".context.json"
    )

    axis_evidence_diversity = Path(
        str(axis_prefix)
        + ".evidence_diversity.json"
    )

    stage8_expected = [
        axis_portfolio,
        axis_plan_copy,
        lineage,
        axis_inference,
        axis_evidence_diversity,
    ]

    if context_review_enabled:
        stage8_expected.append(
            axis_context
        )

    runner.run_stage(
        (
            "[R/Alpha4] Realization "
            f"{slot_index}: discovery-axis synthesis"
        ),
        (
            "scripts.discovery."
            "run_discovery_axis_hypothesis_maker"
        ),
        [
            "--dual-context",
            str(dual_context),
            "--task-question",
            str(args.question),
            "--task-source",
            str(args.source),
            "--task-target",
            str(args.target),
            *_base_model_args(args),
            *_mechanism_index_args(args),
            "--max-axes",
            str(args.max_axes),
            "--min-candidate-unit-score",
            str(
                args.min_candidate_unit_score
            ),
            "--parse-retries",
            str(
                args.hypothesis_parse_retries
            ),
            "--inference-critic-model",
            str(args.critic_model),
            *(
                [
                    "--context-critic-model",
                    str(args.critic_model),
                ]
                if context_review_enabled
                else []
            ),
            "--axis-plan-input",
            str(frozen_axis_plan),
            *(
                [
                    "--external-axis-bundle",
                    str(external_axis_bundle),
                ]
                if external_axis_bundle is not None
                else []
            ),
            "--output-prefix",
            str(axis_prefix),
            "--save-prompts",
        ],
        expected=stage8_expected,
    )

    hypothesis_count = (
        _hypothesis_count(
            axis_portfolio
        )
    )

    result: dict[str, object] = {
        "slot_index":
            slot_index,

        "status":
            (
                "ALPHA4_EMPTY"
                if hypothesis_count == 0
                else "ALPHA4_GENERATED"
            ),

        "hypothesis_count":
            hypothesis_count,

        "frozen_axis_plan":
            str(frozen_axis_plan),

        "materialized_axis_plan":
            str(axis_plan_copy),

        "portfolio":
            str(axis_portfolio),

        "lineage":
            str(lineage),

        "semantic_review":
            None,

        "external_report":
            None,

        "external_plan":
            None,

        "external_prior_art":
            None,

        "scientific_action_batch":
            None,

        "reached_two_pass_semantic":
            False,
    }

    # A realization may fail closed independently.  Best-of-k
    # orchestration must still allow the other realization slots to
    # execute.
    if hypothesis_count == 0:
        return result

    semantic_prefix = (
        slot_run
        / "semantic_axis_a4"
    )

    semantic_review = Path(
        str(semantic_prefix)
        + ".review.json"
    )

    runner.run_stage(
        (
            "[R/Semantic] Realization "
            f"{slot_index}: Alpha4 semantic critic"
        ),
        (
            "scripts.discovery."
            "run_hypothesis_semantic_critic"
        ),
        [
            "--context",
            str(
                slot_run.parent.parent
                / "hypothesis.context.json"
            ),
            "--portfolio",
            str(axis_portfolio),
            *_base_model_args(
                args,
                critic=True,
            ),
            "--output-prefix",
            str(semantic_prefix),
            "--save-prompt",
        ],
        expected=[
            semantic_review
        ],
    )

    external_prefix = (
        slot_run
        / "external_novelty_a52"
    )

    external_report = Path(
        str(external_prefix)
        + ".report.json"
    )

    external_plan = Path(
        str(external_prefix)
        + ".claims_queries.json"
    )

    external_prior = Path(
        str(external_prefix)
        + ".prior_art.json"
    )

    runner.run_stage(
        (
            "[R/External] Realization "
            f"{slot_index}: external novelty"
        ),
        (
            "scripts.discovery."
            "run_external_novelty"
        ),
        [
            "--portfolio",
            str(axis_portfolio),
            "--domain-profile",
            domain_profile_id,
            "--lineage",
            str(lineage),
            *_base_model_args(
                args,
                critic=True,
            ),
            "--provider-plan",
            str(
                literature_provider_plan_path
            ),
            "--results-per-query",
            str(args.results_per_query),
            "--output-prefix",
            str(external_prefix),
            "--save-prompts",
        ],
        expected=[
            external_report,
            external_plan,
            external_prior,
        ],
    )

    current_portfolio_id = (
        _portfolio_id(
            axis_portfolio
        )
    )

    external_source_id = (
        _external_source_portfolio_id(
            external_report
        )
    )

    if (
        external_source_id
        != current_portfolio_id
    ):
        raise RuntimeError(
            "Realization-search external novelty provenance "
            "mismatch: "
            f"slot={slot_index}, "
            f"portfolio={current_portfolio_id}, "
            f"report_source={external_source_id}"
        )

    action_batch = (
        _run_scientific_novelty_action_shadow_chain(
            runner=runner,
            args=args,
            run=slot_run,
            external_report=external_report,
            external_plan=external_plan,
            external_prior=external_prior,
        )
    )

    result.update(
        {
            "status":
                "TWO_PASS_SEMANTIC_EVALUATED",

            "semantic_review":
                str(semantic_review),

            "external_report":
                str(external_report),

            "external_plan":
                str(external_plan),

            "external_prior_art":
                str(external_prior),

            "scientific_action_batch":
                str(action_batch),

            "reached_two_pass_semantic":
                True,
        }
    )

    return result



def _write_realization_json(
    path: Path,
    value,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json"
        )

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


def _realization_semantic_observations(
    *,
    slot_index: int,
    slot_run: Path,
):
    from pipeline_core.discovery.semantic_distinctiveness_contracts import (
        SemanticDistinctivenessReview,
    )
    from pipeline_core.discovery.realization_search_shadow import (
        RealizationSemanticObservation,
    )

    paths = sorted(
        slot_run.glob(
            "semantic_distinctiveness_a10."
            "h*.pass_*.shadow.json"
        )
    )

    grouped = {}

    for path in paths:
        review = (
            SemanticDistinctivenessReview
            .model_validate_json(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )

        grouped.setdefault(
            review.hypothesis_id,
            {},
        )[
            review.review_pass_index
        ] = review

    result = {}

    for (
        hypothesis_id,
        passes,
    ) in grouped.items():
        if (
            set(
                passes
            )
            != {
                1,
                2,
            }
        ):
            raise RuntimeError(
                "Realization hypothesis requires exactly "
                "semantic pass 1 and pass 2: "
                f"slot={slot_index}, "
                f"hypothesis={hypothesis_id}, "
                f"passes={sorted(passes)}"
            )

        first = passes[1]
        second = passes[2]

        result[
            hypothesis_id
        ] = (
            RealizationSemanticObservation(
                slot_index=(
                    slot_index
                ),
                hypothesis_id=(
                    hypothesis_id
                ),
                pass_tiers=(
                    first.overall_tier,
                    second.overall_tier,
                ),
                pass_aggregation_versions=(
                    first
                    .overall_tier_aggregation_version,

                    second
                    .overall_tier_aggregation_version,
                ),
                pass_served_models=(
                    first.served_model,
                    second.served_model,
                ),
                pass_diagnostic_only=(
                    first.diagnostic_only,
                    second.diagnostic_only,
                ),
                pass_action_policy_applied=(
                    first.action_policy_applied,
                    second.action_policy_applied,
                ),
                pass_scientific_selection_changed=(
                    first.scientific_selection_changed,
                    second.scientific_selection_changed,
                ),
            )
        )

    return result


def _materialize_realization_review_artifact(
    *,
    artifact_kind: str,
    output_path: Path,
    winner_portfolio_id: str,
    plan,
    selections_by_axis: dict,
    slot_paths: dict,
    global_selection_enforced: bool = False,
    global_winner_axis_id: str | None = None,
) -> None:
    """Rebind already-computed inference/context reviews to winners."""

    payloads = {
        slot:
            _load_json(
                artifact_path
            )
        for (
            slot,
            artifact_path,
        ) in slot_paths.items()
    }

    if not payloads:
        raise RuntimeError(
            f"{artifact_kind} materialization received "
            "zero slot artifacts."
        )

    template = dict(
        payloads[
            min(
                payloads
            )
        ]
    )

    records = []
    review_history = []

    for axis in plan.axes:
        selection = (
            selections_by_axis[
                axis.axis_id
            ]
        )

        if (
            selection.status
            != "WINNER_SELECTED"
        ):
            continue

        if (
            global_selection_enforced
            and axis.axis_id
            != global_winner_axis_id
        ):
            continue

        slot = (
            selection
            .winner_slot_index
        )

        hypothesis_id = (
            selection
            .winner_hypothesis_id
        )

        if (
            slot is None
            or not hypothesis_id
        ):
            raise RuntimeError(
                f"{artifact_kind} winner lacks "
                "slot/hypothesis metadata."
            )

        payload = payloads[
            slot
        ]

        matching_records = [
            row
            for row
            in payload.get(
                "records",
                [],
            )
            if (
                str(
                    row.get(
                        "final_hypothesis_id"
                    )
                )
                == hypothesis_id
            )
        ]

        if (
            len(
                matching_records
            )
            != 1
        ):
            raise RuntimeError(
                f"Selected {artifact_kind} winner "
                "requires exactly one final record: "
                f"axis={axis.axis_id}, "
                f"slot={slot}, "
                f"hypothesis={hypothesis_id}, "
                f"records={len(matching_records)}"
            )

        records.append(
            matching_records[0]
        )

        review_history.extend(
            row
            for row
            in payload.get(
                "review_history",
                [],
            )
            if (
                str(
                    row.get(
                        "axis_id"
                    )
                )
                == axis.axis_id
            )
        )

    if global_selection_enforced:
        if global_winner_axis_id is None:
            if records:
                raise RuntimeError(
                    f"{artifact_kind} global selection has no "
                    "winner but materialized review records."
                )
        elif len(records) != 1:
            raise RuntimeError(
                f"{artifact_kind} global selection requires "
                "exactly one materialized review record: "
                f"records={len(records)}"
            )

    template[
        "portfolio_id"
    ] = winner_portfolio_id

    template[
        "records"
    ] = records

    template[
        "review_history"
    ] = review_history

    template[
        "final_record_count"
    ] = len(
        records
    )

    template[
        "review_history_count"
    ] = len(
        review_history
    )

    _write_realization_json(
        output_path,
        template,
    )


def _run_realization_search_production_stage8(
    *,
    runner: PipelineRunner,
    args: argparse.Namespace,
    run: Path,
    dual_context: Path,
    axis_prefix: Path,
    axis_portfolio: Path,
    axis_plan: Path,
    lineage: Path,
    axis_inference: Path,
    axis_context: Path,
    axis_evidence_diversity: Path,
    literature_provider_plan_path: Path,
    domain_profile_id: str,
    context_review_enabled: bool,
    frozen_axis_plan_input: Path | None = None,
    external_axis_bundle: Path | None = None,
) -> None:
    """Production-authoritative width-3 search over one frozen axis plan."""

    from pipeline_core.discovery.discovery_axis_contracts import (
        DiscoveryAxisPlan,
        DiscoveryAxisSynthesisReport,
    )
    from pipeline_core.discovery.dual_hypothesis_context import (
        DualHypothesisContext,
    )
    from pipeline_core.discovery.hypothesis_contracts import (
        HypothesisPortfolio,
    )
    from pipeline_core.discovery.hypothesis_evidence_diversity import (
        HypothesisEvidenceDiversityAssessor,
    )
    from pipeline_core.discovery.realization_search_shadow import (
        RealizationSearchPolicy,
    )
    from pipeline_core.discovery.realization_search_cohort import (
        build_axis_realization_cohort,
    )
    from pipeline_core.discovery.realization_search_production import (
        select_axis_realization_production_winner,
    )
    from pipeline_core.discovery.realization_search_materialize import (
        materialize_realization_winners,
    )
    from pipeline_core.discovery.realization_search_task_aware import (
        select_axis_task_aware_production_winner,
    )
    from pipeline_core.discovery.realization_search_global import (
        select_global_axis_production_winner,
    )
    from pipeline_core.discovery.question_axis_responsiveness_llm import (
        OpenRouterQuestionAxisResponsivenessBackend,
    )
    from pipeline_core.discovery.question_hypothesis_responsiveness import (
        evaluate_hypothesis_task_preservation,
    )

    policy = (
        RealizationSearchPolicy(
            search_width=args.realization_search_width,
            retained_hypotheses_per_axis=1,
        )
    )

    dual = (
        DualHypothesisContext
        .model_validate_json(
            dual_context.read_text(
                encoding="utf-8"
            )
        )
    )

    task_responsiveness_backend = (
        OpenRouterQuestionAxisResponsivenessBackend(
            model=args.critic_model,
            temperature=0.0,
            reasoning_effort="medium",
            telemetry_context={
                "stage":
                    (
                        "realization_search_"
                        "task_preservation"
                    ),
            },
        )
    )

    # --------------------------------------------------------------
    # A. Freeze one discovery-axis plan.
    # --------------------------------------------------------------

    if frozen_axis_plan_input is None:
        runner.run_stage(
            (
                "[8R-plan/13] Freeze discovery-axis plan "
                "for realization search"
            ),
            (
                "scripts.discovery."
                "run_discovery_axis_hypothesis_maker"
            ),
            [
                "--dual-context",
                str(
                    dual_context
                ),
                "--max-axes",
                str(
                    args.max_axes
                ),
                "--min-candidate-unit-score",
                str(
                    args.min_candidate_unit_score
                ),
                "--output-prefix",
                str(
                    axis_prefix
                ),
                "--dry-run-plan",
            ],
            expected=[
                axis_plan
            ],
        )
    else:
        if not frozen_axis_plan_input.is_file():
            raise FileNotFoundError(
                "Precomputed frozen discovery-axis plan missing: "
                f"{frozen_axis_plan_input}"
            )

        axis_plan.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            frozen_axis_plan_input.resolve()
            !=
            axis_plan.resolve()
        ):
            axis_plan.write_bytes(
                frozen_axis_plan_input.read_bytes()
            )

    plan = (
        DiscoveryAxisPlan
        .model_validate_json(
            axis_plan.read_text(
                encoding="utf-8"
            )
        )
    )

    if not plan.axes:
        raise RuntimeError(
            "Production realization search received "
            "zero frozen discovery axes."
        )

    # --------------------------------------------------------------
    # B. Run R0 / R1 / R2 independently on that exact plan.
    # --------------------------------------------------------------

    realization_root = (
        run
        / "realization_search"
    )

    slot_portfolios = {}
    slot_lineages = {}
    slot_payloads = []

    slot_inference_paths = {}
    slot_context_paths = {}

    task_assessments_by_slot_hypothesis = {}

    for slot_index in range(
        policy.search_width
    ):
        slot_run = (
            realization_root
            / f"slot_{slot_index}"
        )

        result = (
            _run_realization_candidate_chain(
                runner=runner,
                args=args,
                slot_index=(
                    slot_index
                ),
                slot_run=(
                    slot_run
                ),
                dual_context=(
                    dual_context
                ),
                frozen_axis_plan=(
                    axis_plan
                ),
                domain_profile_id=(
                    domain_profile_id
                ),
                literature_provider_plan_path=(
                    literature_provider_plan_path
                ),
                context_review_enabled=(
                    context_review_enabled
                ),
                external_axis_bundle=(
                    external_axis_bundle
                ),
            )
        )

        portfolio_path = Path(
            str(
                result[
                    "portfolio"
                ]
            )
        )

        lineage_path = Path(
            str(
                result[
                    "lineage"
                ]
            )
        )

        portfolio = (
            HypothesisPortfolio
            .model_validate_json(
                portfolio_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        lineage_report = (
            DiscoveryAxisSynthesisReport
            .model_validate_json(
                lineage_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        slot_portfolios[
            slot_index
        ] = portfolio

        slot_lineages[
            slot_index
        ] = lineage_report

        task_artifact_rows = []

        for card in (
            portfolio.hypotheses
        ):
            debug_prefix = str(
                slot_run
                / (
                    "question_task_preservation."
                    + card.hypothesis_id.split(":")[-1]
                )
            )

            (
                task_assessment,
                _task_stability,
            ) = (
                evaluate_hypothesis_task_preservation(
                    question=(
                        dual
                        .grounded_context
                        .question
                    ),
                    hypothesis=card,
                    backend=(
                        task_responsiveness_backend
                    ),
                    debug_path_prefix=(
                        debug_prefix
                    ),
                )
            )

            task_key = (
                slot_index,
                card.hypothesis_id,
            )

            if (
                task_key
                in task_assessments_by_slot_hypothesis
            ):
                raise RuntimeError(
                    "Duplicate realization task-assessment key"
                )

            task_assessments_by_slot_hypothesis[
                task_key
            ] = task_assessment

            task_artifact_rows.append(
                {
                    "slot_index":
                        slot_index,

                    "hypothesis_id":
                        card.hypothesis_id,

                    "task_class":
                        task_assessment.task_class,

                    "decision_stable":
                        task_assessment.decision_stable,

                    "source_decision_stable":
                        (
                            task_assessment
                            .source_decision_stable
                        ),

                    "quality_eligible":
                        task_assessment.quality_eligible,

                    "winner_ranking_eligible":
                        (
                            task_assessment.quality_eligible
                            and
                            task_assessment.decision_stable
                            and
                            task_assessment.task_class
                            in {
                                "DIRECT",
                                "SUBORDINATE",
                            }
                        ),
                }
            )

        _write_realization_json(
            (
                slot_run
                / (
                    "question_task_preservation."
                    "realization.json"
                )
            ),
            {
                "schema_version":
                    (
                        "realization-task-"
                        "preservation-audit-v1"
                    ),

                "slot_index":
                    slot_index,

                "question":
                    (
                        dual
                        .grounded_context
                        .question
                    ),

                "records":
                    task_artifact_rows,

                "production_winner_eligibility":
                    True,

                "semantic_evaluation_removed":
                    False,
            },
        )

        hypothesis_by_axis = {
            row.axis_id:
                row.hypothesis_id
            for row
            in lineage_report.lineages
        }

        if (
            len(
                hypothesis_by_axis
            )
            != len(
                lineage_report.lineages
            )
        ):
            raise RuntimeError(
                "A realization produced multiple accepted "
                "hypotheses for the same frozen axis."
            )

        semantic_by_hypothesis = {}

        if bool(
            result[
                "reached_two_pass_semantic"
            ]
        ):
            semantic_by_hypothesis = (
                _realization_semantic_observations(
                    slot_index=(
                        slot_index
                    ),
                    slot_run=(
                        slot_run
                    ),
                )
            )

            expected_ids = {
                card.hypothesis_id
                for card
                in portfolio.hypotheses
            }

            if (
                set(
                    semantic_by_hypothesis
                )
                != expected_ids
            ):
                raise RuntimeError(
                    "Realization semantic observation IDs "
                    "do not match its Alpha4 portfolio: "
                    f"slot={slot_index}"
                )

        slot_payloads.append(
            {
                "slot_index":
                    slot_index,

                "alpha4_empty":
                    (
                        len(
                            portfolio.hypotheses
                        )
                        == 0
                    ),

                "hypothesis_by_axis":
                    hypothesis_by_axis,

                "semantic_by_hypothesis":
                    semantic_by_hypothesis,
            }
        )

        inference_path = (
            slot_run
            / "hypothesis_axis_a4.inference.json"
        )

        if not (
            inference_path.is_file()
        ):
            raise RuntimeError(
                "Realization missing inference artifact: "
                f"slot={slot_index}"
            )

        slot_inference_paths[
            slot_index
        ] = inference_path

        if context_review_enabled:
            context_path = (
                slot_run
                / "hypothesis_axis_a4.context.json"
            )

            if not (
                context_path.is_file()
            ):
                raise RuntimeError(
                    "Context-capable realization missing "
                    "context artifact: "
                    f"slot={slot_index}"
                )

            slot_context_paths[
                slot_index
            ] = context_path

    # --------------------------------------------------------------
    # C. Axis-wise cohort and production selection.
    # --------------------------------------------------------------

    cohort_report = (
        build_axis_realization_cohort(
            axis_ids=[
                axis.axis_id
                for axis
                in plan.axes
            ],
            search_width=(
                policy.search_width
            ),
            slot_payloads=(
                slot_payloads
            ),
        )
    )

    task_selection_by_axis = {
        axis_cohort.axis_id:
            (
                select_axis_task_aware_production_winner(
                    axis_cohort,
                    task_assessments_by_slot_hypothesis=(
                        task_assessments_by_slot_hypothesis
                    ),
                    policy=policy,
                )
            )
        for axis_cohort
        in cohort_report.axes
    }

    selections_by_axis = {
        axis_id:
            report.selection
        for (
            axis_id,
            report,
        ) in task_selection_by_axis.items()
    }

    global_selection_enforced = bool(
        getattr(
            args,
            "cross_axis_global_selection_enforce",
            False,
        )
    )

    global_selection = None

    if global_selection_enforced:
        global_selection = (
            select_global_axis_production_winner(
                axis_order=[
                    axis.axis_id
                    for axis
                    in plan.axes
                ],
                task_aware_selections_by_axis=(
                    task_selection_by_axis
                ),
            )
        )

    # --------------------------------------------------------------
    # D. Materialize production-authoritative winners.
    # --------------------------------------------------------------

    materialized = (
        materialize_realization_winners(
            plan=plan,
            slot_portfolios=(
                slot_portfolios
            ),
            slot_lineage_reports=(
                slot_lineages
            ),
            cohort_report=(
                cohort_report
            ),
            selections_by_axis=(
                selections_by_axis
            ),
            global_selection_enforced=(
                global_selection_enforced
            ),
            global_winner_axis_id=(
                None
                if global_selection is None
                else global_selection.winner_axis_id
            ),
        )
    )

    _write_realization_json(
        axis_portfolio,
        materialized.portfolio,
    )

    _write_realization_json(
        lineage,
        materialized.lineage_report,
    )

    # --------------------------------------------------------------
    # E. Preserve canonical Stage-8 reviewer contracts.
    # --------------------------------------------------------------

    _materialize_realization_review_artifact(
        artifact_kind="inference",
        output_path=(
            axis_inference
        ),
        winner_portfolio_id=(
            materialized
            .portfolio
            .portfolio_id
        ),
        plan=plan,
        selections_by_axis=(
            selections_by_axis
        ),
        slot_paths=(
            slot_inference_paths
        ),
        global_selection_enforced=(
            global_selection_enforced
        ),
        global_winner_axis_id=(
            None
            if global_selection is None
            else global_selection.winner_axis_id
        ),
    )

    if context_review_enabled:
        _materialize_realization_review_artifact(
            artifact_kind="context",
            output_path=(
                axis_context
            ),
            winner_portfolio_id=(
                materialized
                .portfolio
                .portfolio_id
            ),
            plan=plan,
            selections_by_axis=(
                selections_by_axis
            ),
            slot_paths=(
                slot_context_paths
            ),
            global_selection_enforced=(
                global_selection_enforced
            ),
            global_winner_axis_id=(
                None
                if global_selection is None
                else global_selection.winner_axis_id
            ),
        )

    # --------------------------------------------------------------
    # F. Recompute evidence diversity on winner portfolio.
    # --------------------------------------------------------------

    dual = (
        DualHypothesisContext
        .model_validate_json(
            dual_context.read_text(
                encoding="utf-8"
            )
        )
    )

    winner_diversity = (
        HypothesisEvidenceDiversityAssessor()
        .assess(
            dual.grounded_context,
            materialized.portfolio,
        )
    )

    _write_realization_json(
        axis_evidence_diversity,
        winner_diversity,
    )

    # --------------------------------------------------------------
    # G. Durable selection artifacts.
    # --------------------------------------------------------------

    cohort_path = (
        run
        / "realization_search.cohort.production.json"
    )

    selection_path = (
        run
        / "realization_search.selection.production.json"
    )

    materialization_path = (
        run
        / "realization_search.materialization.production.json"
    )

    _write_realization_json(
        cohort_path,
        cohort_report,
    )

    _write_realization_json(
        selection_path,
        {
            "schema_version":
                (
                    "realization-search-"
                    "production-selections-v1"
                ),

            "search_width":
                policy.search_width,

            "global_selection_enforced":
                global_selection_enforced,

            "global_selection":
                (
                    None
                    if global_selection is None
                    else global_selection.model_dump(
                        mode="json"
                    )
                ),

            "selections":
                [
                    {
                        "axis_id":
                            axis.axis_id,

                        "selection":
                            selections_by_axis[
                                axis.axis_id
                            ].model_dump(
                                mode="json"
                            ),

                        "task_aware_selection":
                            task_selection_by_axis[
                                axis.axis_id
                            ].model_dump(
                                mode="json"
                            ),
                    }
                    for axis
                    in plan.axes
                ],

            "production_selection_applied":
                True,

            "production_selection_changed":
                True,
        },
    )

    _write_realization_json(
        materialization_path,
        materialized.report,
    )

    runner.manifest[
        "realization_search"
    ] = {
        "status":
            "production_enforced",

        "search_width":
            policy.search_width,

        "retained_hypotheses_per_axis":
            policy.retained_hypotheses_per_axis,

        "cross_axis_global_selection_enforced":
            global_selection_enforced,

        "global_winner_axis_id":
            (
                None
                if global_selection is None
                else global_selection.winner_axis_id
            ),

        "global_winner_hypothesis_id":
            (
                None
                if global_selection is None
                else global_selection.winner_hypothesis_id
            ),

        "global_winner_tier":
            (
                None
                if global_selection is None
                else global_selection.winner_tier
            ),

        "frozen_axis_plan":
            str(
                axis_plan
            ),

        "cohort_artifact":
            str(
                cohort_path
            ),

        "selection_artifact":
            str(
                selection_path
            ),

        "materialization_artifact":
            str(
                materialization_path
            ),

        "materialized_winner_count":
            (
                materialized
                .report
                .materialized_winner_count
            ),

        "production_selection_applied":
            True,

        "task_preservation_before_winner_ranking":
            True,

        "task_eligible_classes":
            [
                "DIRECT",
                "SUBORDINATE",
            ],

        "task_ineligible_classes":
            [
                "TASK_REPLACING",
                "UNRESOLVED",
            ],

        "semantic_evaluation_preserved_for_task_ineligible":
            True,

        "production_selection_changed":
            True,
    }

    runner._save_manifest()


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
    feasibility_adapter = _resolve_feasibility_capability(
        domain_profile
    )
    context_review_mode = str(
        getattr(
            args,
            "context_review_mode",
            "auto",
        )
    ).strip().lower()

    native_context_review_adapter = (
        _resolve_context_review_capability(
            domain_profile
        )
    )

    context_review_adapter = (
        None
        if context_review_mode == "off"
        else native_context_review_adapter
    )

    runner.manifest["domain_profile_id"] = (
        domain_profile.profile_id
    )

    runner.manifest["capabilities"] = {
        "feasibility":
            feasibility_adapter is not None,
        "context_review":
            context_review_adapter is not None,
        "context_review_native_available":
            native_context_review_adapter is not None,
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

    runner.manifest["context_review_status"] = (
        "disabled_by_run_policy"
        if context_review_mode == "off"
        else (
            "available"
            if context_review_adapter is not None
            else "not_supported_for_domain"
        )
    )

    runner.manifest["context_review_adapter_id"] = (
        context_review_adapter.adapter_id
        if context_review_adapter is not None
        else None
    )

    runner.manifest["context_review_mode"] = (
        context_review_mode
    )

    runner.manifest["context_review_native_adapter_id"] = (
        native_context_review_adapter.adapter_id
        if native_context_review_adapter is not None
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
            "scripts.discovery.run_graph_traversal",
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
            "scripts.discovery.run_graph_traversal",
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
        "scripts.discovery.build_explorer_packet",
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
        "scripts.discovery.run_graph_explorer",
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
        "scripts.discovery.build_hypothesis_context",
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
    (
        candidate_correction_args,
        candidate_correction_manifest,
    ) = _candidate_unit_correction_contract(
        args,
        domain_profile,
    )

    runner.manifest[
        "candidate_unit_correction_chain"
    ] = candidate_correction_manifest
    runner._save_manifest()

    candidate_traversal = run / "candidate_unit.traversal.a3.json"
    runner.run_stage(
        "[5/13] Candidate-unit discovery",
        "scripts.discovery.run_candidate_unit_traversal",
        [
            "--corpus-id", args.corpus_id,
            "--domain-profile", domain_profile.profile_id,
            *_data_root_args(args),
            "--source", args.source,
            "--target", args.target,
            "--node-map-k", str(args.node_map_k),
            *candidate_correction_args,
            "--top-k", str(max(args.top_k, 12)),
            "--include-candidate-paths",
            "--output", str(candidate_traversal),
        ],
        expected=[candidate_traversal],
    )

    bundle = run / "discovery.bundle.a3.json"

    semantic_conflict_shadow = (
        run
        / "question_task_preservation."
          "semantic_conflicts.shadow.json"
    )

    bundle_stage_args = [
        "--traversal", str(final_traversal),
        "--traversal", str(candidate_traversal),
        "--domain-profile", domain_profile.profile_id,
        "--top-k", str(args.discovery_top_k),
        "--min-reserved-candidate-unit-score",
        str(args.min_candidate_unit_score),
        "--output", str(bundle),
    ]

    bundle_expected = [
        bundle
    ]

    if args.question_task_preservation_shadow:
        bundle_stage_args += [
            "--semantic-conflict-shadow-output",
            str(
                semantic_conflict_shadow
            ),
        ]

        bundle_expected.append(
            semantic_conflict_shadow
        )

    runner.run_stage(
        "[6/13] DiscoveryBundle",
        "scripts.discovery.build_discovery_bundle",
        bundle_stage_args,
        expected=bundle_expected,
    )

    if args.question_task_preservation_shadow:
        _run_question_task_preservation_shadow_chain(
            runner=runner,
            args=args,
            run=run,
            semantic_conflict_shadow=semantic_conflict_shadow,
            final_traversal=final_traversal,
            candidate_traversal=candidate_traversal,
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
        "scripts.discovery.build_dual_hypothesis_context",
        [
            "--context", str(context),
            "--discovery-bundle", str(bundle),
            "--output", str(dual_context),
        ],
        expected=[dual_context],
    )

    task_conditioned_dual_context = (
        run
        / "hypothesis.task_conditioned.dual_context.a3.json"
    )

    task_conditioned_axis_plan = (
        run
        / "hypothesis_axis_a4.task_conditioned.axis_plan.json"
    )

    task_conditioned_axis_report = (
        run
        / "hypothesis_axis_a4.task_conditioned.report.json"
    )

    task_bridge_fidelity_shadow = (
        run
        / "hypothesis_axis_a4.task_bridge_fidelity.shadow.json"
    )

    runner.run_stage(
        "[7.5/13] Task-conditioned discovery-axis plan",
        "scripts.discovery.build_task_conditioned_axis_plan",
        [
            "--question", str(args.question),
            "--requested-source", str(args.source),
            "--requested-target", str(args.target),
            "--final-traversal", str(final_traversal),
            "--candidate-traversal", str(candidate_traversal),
            "--discovery-bundle", str(bundle),
            "--dual-context", str(dual_context),
            *(
                [
                    "--accepted-patterns",
                    str(args.accepted_patterns),
                ]
                if args.accepted_patterns
                else []
            ),
            "--domain-profile", domain_profile.profile_id,
            "--discovery-top-k", str(args.discovery_top_k),
            "--min-candidate-unit-score",
            str(args.min_candidate_unit_score),
            "--max-axes", str(args.max_axes),
            "--output-dual-context",
            str(task_conditioned_dual_context),
            "--output-axis-plan",
            str(task_conditioned_axis_plan),
            "--output-report",
            str(task_conditioned_axis_report),
            "--output-legacy-bridge-shadow",
            str(task_bridge_fidelity_shadow),
        ],
        expected=[
            task_conditioned_dual_context,
            task_conditioned_axis_plan,
            task_conditioned_axis_report,
            task_bridge_fidelity_shadow,
        ],
    )

    dual_context = task_conditioned_dual_context

    direct_relationpattern_task_shadow = (
        run
        / "hypothesis_axis_a4.direct_relationpattern_task.shadow.json"
    )

    if args.direct_relationpattern_task_shadow:
        responsiveness_model = (
            args.critic_model
            or args.model
        )
        if not responsiveness_model:
            raise RuntimeError(
                "--direct-relationpattern-task-shadow requires "
                "--critic-model or --model."
            )
        if int(
            args.direct_relationpattern_top_k
        ) < 1:
            raise RuntimeError(
                "--direct-relationpattern-top-k must be >= 1"
            )

        direct_relationpattern_debug = (
            run
            / "direct_relationpattern_task_shadow_debug"
        )

        runner.run_stage(
            (
                "[7.52/13] Direct accepted RelationPattern "
                "task-responsiveness shadow"
            ),
            (
                "scripts.discovery."
                "run_direct_relationpattern_task_shadow"
            ),
            [
                "--final-traversal",
                str(final_traversal),
                "--question",
                str(args.question),
                # These are graph-retrieval anchors only.
                # Full-question responsiveness remains the semantic
                # authority for task preservation.
                "--retrieval-source",
                str(args.source),
                "--retrieval-target",
                str(args.target),
                "--model",
                str(responsiveness_model),
                "--retrieval-top-k",
                str(
                    args.direct_relationpattern_top_k
                ),
                "--debug-dir",
                str(
                    direct_relationpattern_debug
                ),
                "--output",
                str(
                    direct_relationpattern_task_shadow
                ),
            ],
            expected=[
                direct_relationpattern_task_shadow
            ],
        )

        direct_rp_payload = _load_json(
            direct_relationpattern_task_shadow
        )

        runner.manifest[
            "direct_relationpattern_task_shadow"
        ] = {
            "enabled": True,
            "report": str(
                direct_relationpattern_task_shadow
            ),
            "query_contract": (
                "GRAPH_RETRIEVAL_SOURCE_PLUS_TARGET"
            ),
            "retrieval_source": str(
                args.source
            ),
            "retrieval_target": str(
                args.target
            ),
            "full_question_is_task_semantic_authority":
                True,
            "retrieval_top_k": int(
                args.direct_relationpattern_top_k
            ),
            "accepted_relationpattern_count":
                direct_rp_payload.get(
                    "accepted_relationpattern_count",
                    0,
                ),
            "stable_direct_count":
                direct_rp_payload.get(
                    "stable_direct_count",
                    0,
                ),
            "stable_subordinate_count":
                direct_rp_payload.get(
                    "stable_subordinate_count",
                    0,
                ),
            "stable_task_replacing_count":
                direct_rp_payload.get(
                    "stable_task_replacing_count",
                    0,
                ),
            "unresolved_count":
                direct_rp_payload.get(
                    "unresolved_count",
                    0,
                ),
            "responsive_candidate_ids":
                direct_rp_payload.get(
                    "responsive_candidate_ids",
                    [],
                ),
            "shadow_only": True,
            "production_selection_changed": False,
            "task_conditioned_axis_plan_changed": False,
            "dual_context_changed_by_shadow": False,
            "stage8_input_changed_by_shadow": False,
            "novelty_authority_created": False,
            "positive_premise_authority_created": False,
            "canonical_graph_mutated": False,
        }
        runner._save_manifest()
    else:
        runner.manifest[
            "direct_relationpattern_task_shadow"
        ] = {
            "enabled": False,
            "production_selection_changed": False,
            "task_conditioned_axis_plan_changed": False,
            "dual_context_changed_by_shadow": False,
            "stage8_input_changed_by_shadow": False,
        }
        runner._save_manifest()

    if args.direct_higher_order_shadow:
        if not args.direct_relationpattern_task_shadow:
            raise RuntimeError(
                "--direct-higher-order-shadow requires "
                "--direct-relationpattern-task-shadow."
            )
        if not args.model:
            raise RuntimeError(
                "--direct-higher-order-shadow requires --model."
            )
        if int(
            args.direct_higher_order_max_contexts
        ) < 1:
            raise RuntimeError(
                "--direct-higher-order-max-contexts must be >= 1"
            )

        direct_higher_order_dir = (
            run
            / "direct_higher_order_shadow"
        )
        direct_higher_order_report = (
            direct_higher_order_dir
            / "generation_report.json"
        )

        runner.run_stage(
            (
                "[7.53/13] Direct higher-order "
                "continuation shadow"
            ),
            (
                "scripts.discovery."
                "run_direct_higher_order_shadow_lane"
            ),
            [
                "--context",
                str(context),
                "--direct-relationpattern-report",
                str(
                    direct_relationpattern_task_shadow
                ),
                "--canonical-root",
                str(run),
                "--domain-profile",
                domain_profile.profile_id,
                "--output-dir",
                str(
                    direct_higher_order_dir
                ),
                "--max-contexts",
                str(
                    args.direct_higher_order_max_contexts
                ),
                "--model",
                str(args.model),
                *(
                    [
                        "--base-url",
                        str(args.base_url),
                    ]
                    if args.base_url
                    else []
                ),
                "--api-key-env",
                str(args.api_key_env),
                "--parse-retries",
                str(
                    args.hypothesis_parse_retries
                ),
                "--max-repairs",
                "1",
            ],
            expected=[
                direct_higher_order_report
            ],
        )

        direct_ho_payload = _load_json(
            direct_higher_order_report
        )

        runner.manifest[
            "direct_higher_order_shadow"
        ] = {
            "enabled": True,
            "report":
                str(
                    direct_higher_order_report
                ),
            "status":
                direct_ho_payload.get(
                    "status"
                ),
            "direct_backbone_count":
                direct_ho_payload.get(
                    "direct_backbone_count",
                    0,
                ),
            "candidate_modifier_component_count":
                direct_ho_payload.get(
                    "candidate_modifier_component_count",
                    0,
                ),
            "eligible_modifier_count":
                direct_ho_payload.get(
                    "eligible_modifier_count",
                    0,
                ),
            "topology_count":
                direct_ho_payload.get(
                    "topology_count",
                    0,
                ),
            "selected_context_count":
                direct_ho_payload.get(
                    "selected_context_count",
                    0,
                ),
            "proposed_count":
                direct_ho_payload.get(
                    "proposed_count",
                    0,
                ),
            "shadow_only":
                True,
            "production_selection_changed":
                False,
            "task_conditioned_axis_plan_changed":
                False,
            "dual_context_changed_by_shadow":
                False,
            "stage8_input_changed_by_shadow":
                False,
            "external_novelty_review_performed":
                False,
            "conceptual_knownness_performed":
                False,
            "novelty_authority_created":
                False,
            "positive_premise_authority_created":
                False,
            "canonical_graph_mutated":
                False,
        }
        runner._save_manifest()
    else:
        runner.manifest[
            "direct_higher_order_shadow"
        ] = {
            "enabled": False,
            "production_selection_changed":
                False,
            "task_conditioned_axis_plan_changed":
                False,
            "dual_context_changed_by_shadow":
                False,
            "stage8_input_changed_by_shadow":
                False,
        }
        runner._save_manifest()

    if args.direct_higher_order_downstream_shadow:
        if not args.direct_higher_order_shadow:
            raise RuntimeError(
                "--direct-higher-order-downstream-shadow requires "
                "--direct-higher-order-shadow."
            )

        direct_higher_order_dir = (
            run
            / "direct_higher_order_shadow"
        )
        direct_higher_order_report = (
            direct_higher_order_dir
            / "generation_report.json"
        )
        direct_higher_order_contexts = (
            direct_higher_order_dir
            / "synthesis_contexts.json"
        )
        direct_higher_order_downstream_dir = (
            direct_higher_order_dir
            / "downstream"
        )
        direct_higher_order_downstream_report = (
            direct_higher_order_downstream_dir
            / "downstream_report.json"
        )

        runner.run_stage(
            (
                "[7.54/13] Direct higher-order "
                "semantic/novelty downstream shadow"
            ),
            (
                "scripts.discovery."
                "run_direct_higher_order_downstream_shadow"
            ),
            [
                "--generation-report",
                str(
                    direct_higher_order_report
                ),
                "--source-context",
                str(context),
                "--synthesis-context-report",
                str(
                    direct_higher_order_contexts
                ),
                "--output-dir",
                str(
                    direct_higher_order_downstream_dir
                ),
                "--domain-profile",
                domain_profile.profile_id,
                "--semantic-model",
                str(
                    args.critic_model
                    or args.model
                ),
                "--novelty-model",
                str(
                    args.critic_model
                    or args.model
                ),
                *(
                    [
                        "--base-url",
                        str(args.base_url),
                    ]
                    if args.base_url
                    else []
                ),
                "--api-key-env",
                str(args.api_key_env),
                "--semantic-parse-retries",
                "1",
                "--provider-plan",
                str(
                    literature_provider_plan_path
                ),
                "--results-per-query",
                str(
                    args.results_per_query
                ),
            ],
            expected=[
                direct_higher_order_downstream_report
            ],
        )

        direct_ho_downstream_payload = _load_json(
            direct_higher_order_downstream_report
        )

        runner.manifest[
            "direct_higher_order_downstream_shadow"
        ] = {
            "enabled":
                True,
            "report":
                str(
                    direct_higher_order_downstream_report
                ),
            "bundle":
                direct_ho_downstream_payload.get(
                    "bundle"
                ),
            "bundle_id":
                direct_ho_downstream_payload.get(
                    "bundle_id"
                ),
            "semantic_attempted_count":
                direct_ho_downstream_payload.get(
                    "semantic_attempted_count",
                    0,
                ),
            "semantic_accepted_count":
                direct_ho_downstream_payload.get(
                    "semantic_accepted_count",
                    0,
                ),
            "external_novelty_completed_count":
                direct_ho_downstream_payload.get(
                    "external_novelty_completed_count",
                    0,
                ),
            "structural_view_count":
                direct_ho_downstream_payload.get(
                    "structural_view_count",
                    0,
                ),
            "conceptual_knownness_integrated":
                False,
            "shadow_only":
                True,
            "candidate_survival_authority":
                False,
            "semantic_rejection_authority":
                False,
            "novelty_authority_created":
                False,
            "positive_premise_authority_created":
                False,
            "production_selection_changed":
                False,
            "stage8_input_changed_by_shadow":
                False,
        }
        runner._save_manifest()
    else:
        runner.manifest[
            "direct_higher_order_downstream_shadow"
        ] = {
            "enabled":
                False,
            "production_selection_changed":
                False,
            "stage8_input_changed_by_shadow":
                False,
        }
        runner._save_manifest()

    if args.higher_order_shadow:
        if int(args.higher_order_max_contexts) < 1:
            raise RuntimeError("--higher-order-max-contexts must be >= 1")

        higher_order_dir = run / "higher_order_shadow"

        higher_order_accepted_patterns = args.accepted_patterns
        higher_order_accepted_patterns_source = "explicit_cli"
        higher_order_accepted_patterns_audit = None

        if higher_order_accepted_patterns is None:
            higher_order_accepted_patterns = (
                higher_order_dir
                / "accepted_relationpatterns.from_canonical_graph.csv"
            )
            higher_order_accepted_patterns_audit = (
                higher_order_dir
                / "accepted_relationpatterns.export_audit.json"
            )

            runner.run_stage(
                (
                    "[7.545/13] Canonical accepted RelationPattern "
                    "export shadow"
                ),
                (
                    "scripts.discovery."
                    "export_accepted_relationpatterns"
                ),
                [
                    "--traversal",
                    str(final_traversal),
                    "--output",
                    str(higher_order_accepted_patterns),
                    "--audit-output",
                    str(higher_order_accepted_patterns_audit),
                ],
                expected=[
                    higher_order_accepted_patterns,
                    higher_order_accepted_patterns_audit,
                ],
            )
            higher_order_accepted_patterns_source = (
                "canonical_graph_export"
            )

        higher_order_report = higher_order_dir / "generation_report.json"
        runner.run_stage(
            "[7.55/13] Higher-order composition/generation shadow",
            "scripts.discovery.run_higher_order_shadow_lane",
            [
                "--context", str(context),
                "--accepted-patterns", str(higher_order_accepted_patterns),
                "--task-axis-report", str(task_conditioned_axis_report),
                "--output-dir", str(higher_order_dir),
                "--max-contexts", str(args.higher_order_max_contexts),
                "--model", str(args.model),
                *(["--base-url", str(args.base_url)] if args.base_url else []),
                "--api-key-env", str(args.api_key_env),
                "--parse-retries", str(args.hypothesis_parse_retries),
            ],
            expected=[higher_order_report],
        )
        ho_payload = _load_json(higher_order_report)
        runner.manifest["higher_order_shadow"] = {
            "enabled": True,
            "report": str(higher_order_report),
            "accepted_patterns_source": higher_order_accepted_patterns_source,
            "accepted_patterns_artifact": str(higher_order_accepted_patterns),
            "accepted_patterns_export_audit": (
                None
                if higher_order_accepted_patterns_audit is None
                else str(higher_order_accepted_patterns_audit)
            ),
            "stage7_5_accepted_patterns_input_changed_by_auto_export": False,
            "strict_backbone_count": ho_payload.get("strict_backbone_count"),
            "eligible_modifier_count": ho_payload.get("eligible_modifier_count"),
            "higher_order_topology_count": ho_payload.get("higher_order_topology_count"),
            "selected_context_count": ho_payload.get("selected_context_count", 0),
            "proposed_count": ho_payload.get("proposed_count", 0),
            "shadow_only": True,
            "production_selection_changed": False,
            "legacy_portfolio_mutated": False,
        }
        runner._save_manifest()
    else:
        runner.manifest["higher_order_shadow"] = {
            "enabled": False,
            "production_selection_changed": False,
        }
        runner._save_manifest()

    open_world_outputs: dict[str, Path] | None = None
    if args.open_world_discovery:
        open_world_outputs = (
            _open_world_discovery_output_contract(
                run
            )
        )

        runner.run_stage(
            "[7.6/13] Bounded open-world discovery-axis synthesis",
            "scripts.discovery.run_open_world_discovery_axes",
            [
                "--dual-context",
                str(dual_context),
                "--control-axis-plan",
                str(task_conditioned_axis_plan),
                *_mechanism_index_args(args),
                "--providers",
                str(args.providers),
                *_base_model_args(args),
                "--output-prefix",
                str(open_world_outputs["prefix"]),
                "--save-prompt",
            ],
            expected=[
                open_world_outputs["provider_plan"],
                open_world_outputs["retrieval"],
                open_world_outputs["selected_works"],
                open_world_outputs["axis_synthesis"],
                open_world_outputs["axis_validation"],
                open_world_outputs["external_axis_plan"],
                open_world_outputs["external_axis_bundle"],
                open_world_outputs["external_axis_rejections"],
                open_world_outputs["manifest"],
                open_world_outputs["prompt"],
            ],
        )

        open_world_manifest = _load_json(
            open_world_outputs["manifest"]
        )
        _validate_open_world_discovery_parent_manifest(
            open_world_manifest,
            expected_provider_plan_sha256=(
                literature_provider_plan.plan_sha256
            ),
        )

        runner.manifest["open_world_discovery"] = {
            "enabled": True,
            "control_axis_plan":
                str(task_conditioned_axis_plan),
            "external_axis_plan":
                str(open_world_outputs["external_axis_plan"]),
            "external_axis_bundle":
                str(open_world_outputs["external_axis_bundle"]),
            "stage8_axis_plan_source":
                "validated_external_open_world_plan",
            "provider_plan_sha256":
                open_world_manifest.get(
                    "provider_plan_sha256"
                ),
            "external_axis_count":
                open_world_manifest.get(
                    "external_axis_count"
                ),
            "external_literature_authority":
                "INSPIRATION_ONLY",
            "positive_premise_authority_changed":
                False,
            "novelty_authority_created":
                False,
            "canonical_fallback_authorized":
                False,
        }
        runner._save_manifest()


    frontier_population_shadow = (
        run
        / "frontier_idea_population.shadow.json"
    )

    if args.frontier_idea_population_shadow:
        frontier_stage_args = [
            "--source-context",
            str(context),
            "--question",
            str(args.question),
            "--task-source",
            str(args.source),
            "--task-target",
            str(args.target),
            "--kg-axis-plan",
            str(task_conditioned_axis_plan),
            "--output",
            str(frontier_population_shadow),
        ]

        if open_world_outputs is not None:
            frontier_stage_args.extend(
                [
                    "--open-world-axis-plan",
                    str(
                        open_world_outputs[
                            "external_axis_plan"
                        ]
                    ),
                    "--open-world-axis-bundle",
                    str(
                        open_world_outputs[
                            "external_axis_bundle"
                        ]
                    ),
                ]
            )

        higher_order_dir = (
            run
            / "higher_order_shadow"
        )
        higher_order_topologies = (
            higher_order_dir
            / "topologies.json"
        )
        if higher_order_topologies.is_file():
            frontier_stage_args.extend(
                [
                    "--higher-order-topologies",
                    str(
                        higher_order_topologies
                    ),
                ]
            )

        higher_order_tensions = (
            higher_order_dir
            / "higher_order.tension_candidates.json"
        )
        if higher_order_tensions.is_file():
            frontier_stage_args.extend(
                [
                    "--tension-candidates",
                    str(
                        higher_order_tensions
                    ),
                ]
            )

        higher_order_explanations = (
            higher_order_dir
            / "higher_order.competing_explanations.json"
        )
        if higher_order_explanations.is_file():
            frontier_stage_args.extend(
                [
                    "--competing-explanations",
                    str(
                        higher_order_explanations
                    ),
                ]
            )

        direct_higher_order_topologies = (
            run
            / "direct_higher_order_shadow"
            / "topologies.json"
        )
        if direct_higher_order_topologies.is_file():
            frontier_stage_args.extend(
                [
                    "--direct-higher-order-topologies",
                    str(
                        direct_higher_order_topologies
                    ),
                ]
            )

        runner.run_stage(
            (
                "[7.7/13] Frontier idea population shadow"
            ),
            (
                "scripts.discovery."
                "run_frontier_idea_population_shadow"
            ),
            frontier_stage_args,
            expected=[
                frontier_population_shadow
            ],
        )

        frontier_payload = _load_json(
            frontier_population_shadow
        )

        runner.manifest[
            "frontier_idea_population_shadow"
        ] = {
            "enabled":
                True,
            "artifact":
                str(
                    frontier_population_shadow
                ),
            "population_id":
                frontier_payload.get(
                    "population_id"
                ),
            "population_sha256":
                frontier_payload.get(
                    "population_sha256"
                ),
            "total_idea_count":
                frontier_payload.get(
                    "total_idea_count",
                    0,
                ),
            "idea_count_by_source_kind":
                frontier_payload.get(
                    "idea_count_by_source_kind",
                    {},
                ),
            "idea_count_by_idea_form":
                frontier_payload.get(
                    "idea_count_by_idea_form",
                    {},
                ),
            "cross_source_exact_duplicate_group_count":
                frontier_payload.get(
                    "cross_source_exact_duplicate_group_count",
                    0,
                ),
            "cross_source_overlap_pair_count":
                frontier_payload.get(
                    "cross_source_overlap_pair_count",
                    0,
                ),
            "cross_source_structural_overlap_pair_count":
                frontier_payload.get(
                    "cross_source_structural_overlap_pair_count",
                    0,
                ),
            "shadow_only":
                True,
            "new_llm_calls":
                False,
            "new_retrieval_calls":
                False,
            "positive_premise_authority_created":
                False,
            "novelty_authority_created":
                False,
            "generation_authority_created":
                False,
            "production_selection_authority":
                False,
            "stage8_input_changed":
                False,
            "canonical_graph_mutated":
                False,
        }
        runner._save_manifest()

        frontier_exploration_audit = (
            run
            / "frontier_exploration.audit.json"
        )
        frontier_audit_args = [
            "--population",
            str(frontier_population_shadow),
            "--output",
            str(frontier_exploration_audit),
        ]

        higher_order_contexts = (
            higher_order_dir
            / "contexts.json"
        )
        higher_order_generation_report = (
            higher_order_dir
            / "generation_report.json"
        )

        if higher_order_contexts.is_file():
            frontier_audit_args.extend(
                [
                    "--higher-order-contexts",
                    str(higher_order_contexts),
                ]
            )
        if higher_order_generation_report.is_file():
            frontier_audit_args.extend(
                [
                    "--higher-order-generation-report",
                    str(higher_order_generation_report),
                ]
            )

        runner.run_stage(
            (
                "[7.71/13] Exploration Frontier factorization shadow"
            ),
            (
                "scripts.discovery."
                "run_frontier_exploration_audit"
            ),
            frontier_audit_args,
            expected=[
                frontier_exploration_audit
            ],
        )

        frontier_audit_payload = _load_json(
            frontier_exploration_audit
        )
        primitive_payload = (
            frontier_audit_payload.get(
                "primitive_layer",
                {},
            )
        )
        topology_payload = (
            frontier_audit_payload.get(
                "topology_layer",
                {},
            )
        )
        interpretive_payload = (
            frontier_audit_payload.get(
                "interpretive_layer",
                {},
            )
        )
        capacity_payload = (
            frontier_audit_payload.get(
                "capacity",
                {},
            )
        )

        runner.manifest[
            "frontier_idea_population_shadow"
        ]["exploration_frontier_audit"] = {
            "artifact": str(frontier_exploration_audit),
            "audit_id": frontier_audit_payload.get("audit_id"),
            "audit_sha256": frontier_audit_payload.get("audit_sha256"),
            "unique_primitive_family_count": primitive_payload.get(
                "unique_primitive_family_count",
                0,
            ),
            "open_world_exclusive_primitive_family_count": primitive_payload.get(
                "open_world_exclusive_primitive_family_count",
                0,
            ),
            "open_world_axis_primitive_reused_in_topology_count": primitive_payload.get(
                "open_world_axis_primitive_reused_in_topology_count",
                0,
            ),
            "unique_backbone_family_count": topology_payload.get(
                "unique_backbone_family_count",
                0,
            ),
            "unique_modifier_family_count": topology_payload.get(
                "unique_modifier_family_count",
                0,
            ),
            "largest_backbone_family_share": topology_payload.get(
                "largest_backbone_family_share",
                0.0,
            ),
            "tension_seed_count": interpretive_payload.get(
                "tension_seed_count",
                0,
            ),
            "competing_explanation_pair_count": interpretive_payload.get(
                "competing_explanation_pair_count",
                0,
            ),
            "max_verified_structural_derivation_depth": interpretive_payload.get(
                "max_verified_structural_derivation_depth",
                0,
            ),
            "higher_order_topology_cap_reached": capacity_payload.get(
                "higher_order_topology_cap_reached"
            ),
            "diagnostic_only": True,
            "scientific_quality_ranking_performed": False,
            "production_selection_authority": False,
            "stage8_input_changed": False,
        }
        runner._save_manifest()
    else:
        runner.manifest[
            "frontier_idea_population_shadow"
        ] = {
            "enabled":
                False,
            "stage8_input_changed":
                False,
            "production_selection_authority":
                False,
        }
        runner._save_manifest()



    idea_evolution_plan = run / "idea_evolution.plan.json"
    idea_evolution_report = run / "idea_evolution.shadow.json"
    idea_evolution_audit = run / "idea_evolution.audit.json"

    if args.idea_evolution_shadow:
        if not args.frontier_idea_population_shadow:
            raise RuntimeError(
                "--idea-evolution-shadow requires --frontier-idea-population-shadow "
                "so evolution lineage is anchored to the audited Frontier."
            )

        idea_evolution_args = [
            "--population",
            str(frontier_population_shadow),
            "--frontier-audit",
            str(frontier_exploration_audit),
            "--max-cross-source-outputs",
            str(args.idea_evolution_max_cross_source_outputs),
            "--max-backbone-mutation-outputs",
            str(args.idea_evolution_max_backbone_mutation_outputs),
            "--max-candidate-interpretation-outputs",
            str(args.idea_evolution_max_candidate_interpretation_outputs),
            "--max-candidate-parent-pool",
            str(args.idea_evolution_max_candidate_parent_pool),
            "--output-plan",
            str(idea_evolution_plan),
            "--output-report",
            str(idea_evolution_report),
            "--output-audit",
            str(idea_evolution_audit),
            "--save-prompts-dir",
            str(run / "idea_evolution_prompts"),
            *_base_model_args(args),
        ]

        if args.idea_evolution_scientific_reframe_shadow is not None:
            idea_evolution_args.extend(
                [
                    "--scientific-reframe-shadow",
                    str(args.idea_evolution_scientific_reframe_shadow),
                ]
            )
        if args.idea_evolution_proxy_challenge_shadow is not None:
            idea_evolution_args.extend(
                [
                    "--proxy-challenge-shadow",
                    str(args.idea_evolution_proxy_challenge_shadow),
                ]
            )

        runner.run_stage(
            "[7.72/13] Idea Evolution shadow",
            "scripts.discovery.run_idea_evolution_shadow",
            idea_evolution_args,
            expected=[
                idea_evolution_plan,
                idea_evolution_report,
                idea_evolution_audit,
            ],
        )

        idea_evolution_payload = _load_json(idea_evolution_report)
        idea_evolution_audit_payload = _load_json(idea_evolution_audit)
        runner.manifest["idea_evolution_shadow"] = {
            "enabled": True,
            "plan": str(idea_evolution_plan),
            "report": str(idea_evolution_report),
            "audit": str(idea_evolution_audit),
            "report_id": idea_evolution_payload.get("report_id"),
            "report_sha256": idea_evolution_payload.get("report_sha256"),
            "idea_count": idea_evolution_payload.get("idea_count", 0),
            "idea_count_by_operator": idea_evolution_payload.get(
                "idea_count_by_operator", {}
            ),
            "native_llm_calls_attempted": idea_evolution_payload.get(
                "native_llm_calls_attempted", 0
            ),
            "native_llm_calls_succeeded": idea_evolution_payload.get(
                "native_llm_calls_succeeded", 0
            ),
            "cross_source_bridge_count": idea_evolution_audit_payload.get(
                "cross_source_bridge_count", 0
            ),
            "new_mutated_backbone_family_count": idea_evolution_audit_payload.get(
                "new_mutated_backbone_family_count", 0
            ),
            "candidate_interpretation_count": idea_evolution_audit_payload.get(
                "candidate_interpretation_count", 0
            ),
            "reframe_idea_count": idea_evolution_audit_payload.get(
                "reframe_idea_count", 0
            ),
            "conceptual_transition_observed": idea_evolution_audit_payload.get(
                "conceptual_transition_observed", False
            ),
            "shadow_only": True,
            "new_retrieval_calls": False,
            "positive_premise_authority_created": False,
            "novelty_authority_created": False,
            "generation_authority_created": False,
            "production_selection_authority": False,
            "stage8_input_changed": False,
            "canonical_graph_mutated": False,
        }
        runner._save_manifest()
    else:
        runner.manifest["idea_evolution_shadow"] = {
            "enabled": False,
            "production_selection_authority": False,
            "stage8_input_changed": False,
        }
        runner._save_manifest()



    scientific_portfolio_dir = run / "scientific_portfolio_shadow"
    scientific_portfolio_pool = scientific_portfolio_dir / "candidate_pool.json"
    scientific_portfolio_evaluation = scientific_portfolio_dir / "evaluation.json"
    scientific_portfolio_selection = scientific_portfolio_dir / "selection.json"
    scientific_portfolio_materialization = (
        scientific_portfolio_dir / "materialization.report.json"
    )
    scientific_portfolio_materialized = (
        scientific_portfolio_dir / "materialized.shadow.portfolio.json"
    )
    scientific_portfolio_audit = scientific_portfolio_dir / "audit.json"

    if args.scientific_portfolio_selection_shadow:
        if not args.idea_evolution_shadow:
            raise RuntimeError(
                "--scientific-portfolio-selection-shadow requires "
                "--idea-evolution-shadow so portfolio lineage is anchored to "
                "the audited Frontier + Idea Evolution population."
            )
        if int(args.scientific_portfolio_max_evaluation_candidates) < 1:
            raise RuntimeError(
                "--scientific-portfolio-max-evaluation-candidates must be >= 1"
            )
        if int(args.scientific_portfolio_max_retained) < 1:
            raise RuntimeError(
                "--scientific-portfolio-max-retained must be >= 1"
            )
        if int(args.scientific_portfolio_max_retained_per_profile) < 1:
            raise RuntimeError(
                "--scientific-portfolio-max-retained-per-profile must be >= 1"
            )

        runner.run_stage(
            "[7.73/13] Scientific Portfolio Selection shadow",
            "scripts.discovery.run_scientific_portfolio_selection_shadow",
            [
                "--context",
                str(context),
                "--population",
                str(frontier_population_shadow),
                "--frontier-audit",
                str(frontier_exploration_audit),
                "--evolution-report",
                str(idea_evolution_report),
                "--task-source",
                str(args.source),
                "--task-target",
                str(args.target),
                "--max-evaluation-candidates",
                str(args.scientific_portfolio_max_evaluation_candidates),
                "--max-retained",
                str(args.scientific_portfolio_max_retained),
                "--max-retained-per-profile",
                str(args.scientific_portfolio_max_retained_per_profile),
                "--output-dir",
                str(scientific_portfolio_dir),
                "--save-prompts",
                *_base_model_args(args),
            ],
            expected=[
                scientific_portfolio_pool,
                scientific_portfolio_evaluation,
                scientific_portfolio_selection,
                scientific_portfolio_materialization,
                scientific_portfolio_materialized,
                scientific_portfolio_audit,
            ],
        )

        scientific_portfolio_audit_payload = _load_json(
            scientific_portfolio_audit
        )
        scientific_portfolio_materialization_payload = _load_json(
            scientific_portfolio_materialization
        )
        runner.manifest["scientific_portfolio_selection_shadow"] = {
            "enabled": True,
            "candidate_pool": str(scientific_portfolio_pool),
            "evaluation": str(scientific_portfolio_evaluation),
            "selection": str(scientific_portfolio_selection),
            "materialization_report": str(scientific_portfolio_materialization),
            "materialized_portfolio": str(scientific_portfolio_materialized),
            "audit": str(scientific_portfolio_audit),
            "projected_candidate_count": scientific_portfolio_audit_payload.get(
                "projected_candidate_count", 0
            ),
            "retained_candidate_count": scientific_portfolio_audit_payload.get(
                "retained_candidate_count", 0
            ),
            "retained_unique_family_count": scientific_portfolio_audit_payload.get(
                "retained_unique_family_count", 0
            ),
            "materialized_hypothesis_count": scientific_portfolio_audit_payload.get(
                "materialized_hypothesis_count", 0
            ),
            "materialization_status_counts": (
                scientific_portfolio_materialization_payload.get("status_counts", {})
            ),
            "selection_policy": "PROFILE_PARETO_DIVERSITY_RETENTION_V2_NORMALIZED_BLIND",
            "single_scalar_score_used": False,
            "overall_winner_selected": False,
            "idea_inspiration_as_positive_premise": False,
            "positive_premise_authority_created": False,
            "novelty_authority_created": False,
            "production_selection_authority": False,
            "stage8_input_changed": False,
            "canonical_graph_mutated": False,
            "shadow_only": True,
        }
        runner._save_manifest()
    else:
        runner.manifest["scientific_portfolio_selection_shadow"] = {
            "enabled": False,
            "production_selection_authority": False,
            "stage8_input_changed": False,
        }
        runner._save_manifest()

    scientific_portfolio_verification_dir = (
        scientific_portfolio_dir / "downstream_verification"
    )
    scientific_portfolio_verification_summary = (
        scientific_portfolio_verification_dir / "verification.summary.json"
    )
    scientific_portfolio_production_candidate = (
        scientific_portfolio_verification_dir
        / "production.candidate.portfolio.json"
    )
    scientific_portfolio_n10_certified = (
        scientific_portfolio_verification_dir
        / "n10.certified.portfolio.json"
    )

    if args.scientific_portfolio_verification_shadow:
        if not args.scientific_portfolio_selection_shadow:
            raise RuntimeError(
                "--scientific-portfolio-verification-shadow requires "
                "--scientific-portfolio-selection-shadow."
            )

        scientific_portfolio_verification_args = [
            "--context",
            str(context),
            "--materialization-report",
            str(scientific_portfolio_materialization),
            "--portfolio",
            str(scientific_portfolio_materialized),
            "--domain-profile",
            str(domain_profile.profile_id),
            "--output-dir",
            str(scientific_portfolio_verification_dir),
            "--model",
            str(args.critic_model or args.model),
            "--api-key-env",
            str(args.api_key_env),
            "--providers",
            str(args.providers),
            "--provider-plan",
            str(literature_provider_plan_path),
            "--results-per-query",
            str(args.results_per_query),
        ]
        if args.scientific_portfolio_production_enforce:
            scientific_portfolio_verification_args.append(
                "--production-enforce"
            )
        if args.base_url:
            scientific_portfolio_verification_args.extend(
                ["--base-url", str(args.base_url)]
            )

        runner.run_stage(
            "[7.74/13] Scientific Portfolio downstream verification shadow",
            "scripts.discovery.run_scientific_portfolio_verification_shadow",
            scientific_portfolio_verification_args,
            expected=[scientific_portfolio_verification_summary],
        )

        scientific_portfolio_verification_payload = _load_json(
            scientific_portfolio_verification_summary
        )
        runner.manifest["scientific_portfolio_verification_shadow"] = {
            "enabled": True,
            "summary": str(scientific_portfolio_verification_summary),
            "status": scientific_portfolio_verification_payload.get("status"),
            "hypothesis_count": scientific_portfolio_verification_payload.get(
                "hypothesis_count", 0
            ),
            "semantic": scientific_portfolio_verification_payload.get("semantic", {}),
            "external_novelty": scientific_portfolio_verification_payload.get(
                "external_novelty", {}
            ),
            "n9": scientific_portfolio_verification_payload.get("n9", {}),
            "feasibility": scientific_portfolio_verification_payload.get(
                "feasibility", {}
            ),
            "n10_run": bool(
                scientific_portfolio_verification_payload.get("n10_run", False)
            ),
            "novelty_certification_authority": bool(
                scientific_portfolio_verification_payload.get(
                    "novelty_certification_authority", False
                )
            ),
            "production_selection_authority": bool(
                scientific_portfolio_verification_payload.get(
                    "production_selection_authority", False
                )
            ),
            "production_binding": scientific_portfolio_verification_payload.get(
                "production_binding", {}
            ),
            "stage8_input_changed": False,
            "canonical_graph_mutated": False,
            "shadow_only": True,
        }
        runner._save_manifest()
    else:
        runner.manifest["scientific_portfolio_verification_shadow"] = {
            "enabled": False,
            "n10_run": False,
            "production_selection_authority": False,
            "stage8_input_changed": False,
        }
        runner._save_manifest()

    scientific_portfolio_closed_loop_dir = (
        scientific_portfolio_dir / "closed_loop_shadow"
    )
    scientific_portfolio_closed_loop_summary = (
        scientific_portfolio_closed_loop_dir / "closed_loop.summary.json"
    )
    scientific_portfolio_closed_loop_effective = (
        scientific_portfolio_closed_loop_dir / "effective_gen1.portfolio.json"
    )

    if args.scientific_portfolio_closed_loop_shadow:
        if not args.scientific_portfolio_verification_shadow:
            raise RuntimeError(
                "--scientific-portfolio-closed-loop-shadow requires "
                "--scientific-portfolio-verification-shadow"
            )

        verification_external_prefix = (
            scientific_portfolio_verification_dir / "external_novelty"
        )
        verification_external_plan = Path(
            str(verification_external_prefix) + ".claims_queries.json"
        )
        verification_external_report = Path(
            str(verification_external_prefix) + ".report.json"
        )
        verification_external_prior = Path(
            str(verification_external_prefix) + ".prior_art.json"
        )
        verification_external_binding = Path(
            str(verification_external_prefix) + ".atomic_source_binding.json"
        )
        verification_provider_plan = Path(
            str(verification_external_prefix) + ".provider_plan.json"
        )

        for required_path in (
            verification_external_plan,
            verification_external_report,
            verification_external_prior,
            verification_external_binding,
            verification_provider_plan,
        ):
            if not required_path.is_file():
                raise RuntimeError(
                    "Scientific Portfolio closed loop requires completed "
                    "Stage-7.74 external artifacts; missing "
                    + str(required_path)
                )

        closed_loop_args = [
            "--context",
            str(context),
            "--materialization-report",
            str(scientific_portfolio_materialization),
            "--portfolio",
            str(scientific_portfolio_materialized),
            "--external-query-plan",
            str(verification_external_plan),
            "--external-report",
            str(verification_external_report),
            "--external-prior-art",
            str(verification_external_prior),
            "--external-source-binding",
            str(verification_external_binding),
            "--provider-plan",
            str(verification_provider_plan),
            "--domain-profile",
            str(domain_profile.profile_id),
            "--model",
            str(args.model),
            "--critic-model",
            str(args.critic_model or args.model),
            "--api-key-env",
            str(args.api_key_env),
            "--results-per-query",
            str(args.results_per_query),
            "--output-dir",
            str(scientific_portfolio_closed_loop_dir),
        ]
        if args.base_url:
            closed_loop_args.extend(["--base-url", str(args.base_url)])

        runner.run_stage(
            "[7.75/13] Scientific Portfolio residual-aware closed loop shadow",
            "scripts.discovery.run_scientific_portfolio_closed_loop_shadow",
            closed_loop_args,
            expected=[
                scientific_portfolio_closed_loop_summary,
                scientific_portfolio_closed_loop_effective,
            ],
        )

        closed_loop_payload = _load_json(
            scientific_portfolio_closed_loop_summary
        )
        runner.manifest["scientific_portfolio_closed_loop_shadow"] = {
            "enabled": True,
            "summary": str(scientific_portfolio_closed_loop_summary),
            "status": closed_loop_payload.get("status"),
            "gen0_state_counts": closed_loop_payload.get(
                "gen0_state_counts", {}
            ),
            "feedback_route_counts": closed_loop_payload.get(
                "feedback_route_counts", {}
            ),
            "generation_decision_counts": closed_loop_payload.get(
                "generation_decision_counts", {}
            ),
            "gen1_residual_disposition_counts": closed_loop_payload.get(
                "gen1_residual_disposition_counts", {}
            ),
            "post_verification_decision_counts": closed_loop_payload.get(
                "post_verification_decision_counts", {}
            ),
            "effective_gen1_portfolio": str(
                scientific_portfolio_closed_loop_effective
            ),
            "effective_gen1_hypothesis_count": closed_loop_payload.get(
                "effective_gen1_hypothesis_count", 0
            ),
            "effective_gen1_hypothesis_ids": closed_loop_payload.get(
                "effective_gen1_hypothesis_ids", []
            ),
            "single_feedback_generation_only": True,
            "n10_research_selection_authority": False,
            "stage8_input_changed": False,
            "production_selection_changed": False,
            "canonical_graph_mutated": False,
        }
        runner._save_manifest()
    else:
        runner.manifest["scientific_portfolio_closed_loop_shadow"] = {
            "enabled": False,
            "stage8_input_changed": False,
            "production_selection_changed": False,
        }
        runner._save_manifest()

    scientific_portfolio_adaptive_dir = (
        scientific_portfolio_dir / "adaptive_controller_shadow"
    )
    scientific_portfolio_adaptive_summary = (
        scientific_portfolio_adaptive_dir
        / "adaptive_controller.summary.json"
    )
    scientific_portfolio_adaptive_effective = (
        scientific_portfolio_adaptive_dir
        / "adaptive_effective.portfolio.json"
    )
    scientific_portfolio_adaptive_handoff = (
        scientific_portfolio_adaptive_dir
        / "graph_retraversal.handoff.json"
    )

    if args.adaptive_discovery_controller_shadow:
        if not scientific_portfolio_closed_loop_summary.is_file():
            raise RuntimeError(
                "Adaptive Discovery Controller requires completed Stage 7.75."
            )

        adaptive_args = [
            "--context",
            str(context),
            "--seed-closed-loop-dir",
            str(scientific_portfolio_closed_loop_dir),
            "--provider-plan",
            str(verification_provider_plan),
            "--domain-profile",
            str(domain_profile.profile_id),
            "--model",
            str(args.model),
            "--critic-model",
            str(args.critic_model or args.model),
            "--controller-model",
            str(args.critic_model or args.model),
            "--api-key-env",
            str(args.api_key_env),
            "--results-per-query",
            str(args.results_per_query),
            "--max-controller-rounds",
            str(args.adaptive_controller_max_rounds),
            "--max-local-attempts-per-lineage",
            str(args.adaptive_controller_max_local_attempts),
            "--output-dir",
            str(scientific_portfolio_adaptive_dir),
        ]
        if args.adaptive_controller_disable_llm:
            adaptive_args.append("--disable-controller-llm")
        if args.base_url:
            adaptive_args.extend(["--base-url", str(args.base_url)])

        runner.run_stage(
            "[7.76/13] Adaptive Discovery Controller shadow",
            "scripts.discovery.run_adaptive_discovery_controller_shadow",
            adaptive_args,
            expected=[
                scientific_portfolio_adaptive_summary,
                scientific_portfolio_adaptive_effective,
                scientific_portfolio_adaptive_handoff,
            ],
        )

        adaptive_payload = _load_json(
            scientific_portfolio_adaptive_summary
        )
        runner.manifest["adaptive_discovery_controller_shadow"] = {
            "enabled": True,
            "summary": str(scientific_portfolio_adaptive_summary),
            "status": adaptive_payload.get("status"),
            "adaptive_rounds_completed": adaptive_payload.get(
                "adaptive_rounds_completed", 0
            ),
            "action_counts": adaptive_payload.get(
                "action_counts", {}
            ),
            "final_effective_count": adaptive_payload.get(
                "final_effective_count", 0
            ),
            "final_effective_hypothesis_ids": adaptive_payload.get(
                "final_effective_hypothesis_ids", []
            ),
            "graph_retraversal_request_count": adaptive_payload.get(
                "graph_retraversal_request_count", 0
            ),
            "graph_retraversal_executes_in_v1": False,
            "n10_research_selection_authority": False,
            "stage8_input_changed": False,
            "production_selection_changed": False,
            "canonical_graph_mutated": False,
        }
        runner._save_manifest()
    else:
        runner.manifest["adaptive_discovery_controller_shadow"] = {
            "enabled": False,
            "stage8_input_changed": False,
            "production_selection_changed": False,
        }
        runner._save_manifest()

    scientific_portfolio_adaptive_v2_dir = (
        scientific_portfolio_dir
        / "adaptive_graph_retraversal_shadow"
    )
    scientific_portfolio_adaptive_v2_summary = (
        scientific_portfolio_adaptive_v2_dir
        / "adaptive_v2.summary.json"
    )
    scientific_portfolio_adaptive_v2_effective_population = (
        scientific_portfolio_adaptive_v2_dir
        / "adaptive_v2.effective_population.json"
    )
    scientific_portfolio_adaptive_v2_lineage = (
        scientific_portfolio_adaptive_v2_dir
        / "context_retraversal.lineage.json"
    )
    scientific_portfolio_adaptive_v2_handoff = (
        scientific_portfolio_adaptive_v2_dir
        / "graph_retraversal.handoff.json"
    )

    if args.adaptive_graph_retraversal_shadow:
        for required_path in (
            scientific_portfolio_closed_loop_summary,
            scientific_portfolio_adaptive_summary,
            scientific_portfolio_adaptive_effective,
            scientific_portfolio_adaptive_handoff,
            final_traversal,
            context,
            verification_provider_plan,
        ):
            if not required_path.is_file():
                raise RuntimeError(
                    "Adaptive Graph Retraversal Stage 7.77 requires the "
                    "completed Stage-7.75/7.76 lineage; missing "
                    + str(required_path)
                )

        adaptive_v2_args = [
            "--context",
            str(context),
            "--grounding-traversal",
            str(final_traversal),
            "--seed-closed-loop-dir",
            str(scientific_portfolio_closed_loop_dir),
            "--initial-local-controller-dir",
            str(scientific_portfolio_adaptive_dir),
            "--provider-plan",
            str(verification_provider_plan),
            "--domain-profile",
            str(domain_profile.profile_id),
            "--model",
            str(args.model),
            "--critic-model",
            str(args.critic_model or args.model),
            "--controller-model",
            str(args.critic_model or args.model),
            "--api-key-env",
            str(args.api_key_env),
            "--results-per-query",
            str(args.results_per_query),
            "--max-controller-rounds",
            str(args.adaptive_controller_max_rounds),
            "--max-local-attempts-per-lineage",
            str(args.adaptive_controller_max_local_attempts),
            "--max-graph-retraversals",
            str(args.adaptive_graph_max_retraversals),
            "--retraversal-candidate-top-k",
            str(args.adaptive_graph_candidate_top_k),
            "--retraversal-selected-top-k",
            str(args.adaptive_graph_selected_top_k),
            "--retraversal-paper-expansion-reserve",
            str(args.adaptive_graph_paper_expansion_reserve),
            "--retraversal-node-map-k",
            str(args.adaptive_graph_node_map_k),
            "--retraversal-endpoint-pair-k",
            str(args.adaptive_graph_endpoint_pair_k),
            "--retraversal-max-depth-increment",
            str(args.adaptive_graph_max_depth_increment),
            "--retraversal-max-depth-cap",
            str(args.adaptive_graph_max_depth_cap),
            "--retraversal-min-new-edge-fraction",
            str(args.adaptive_graph_min_new_edge_fraction),
            "--retraversal-min-new-paper-fraction",
            str(args.adaptive_graph_min_new_paper_fraction),
            "--retraversal-max-prior-edge-jaccard",
            str(args.adaptive_graph_max_prior_edge_jaccard),
            "--retraversal-max-selected-edge-jaccard",
            str(args.adaptive_graph_max_selected_edge_jaccard),
            "--output-dir",
            str(scientific_portfolio_adaptive_v2_dir),
        ]
        if args.adaptive_controller_disable_llm:
            adaptive_v2_args.append(
                "--disable-controller-llm"
            )
        if not args.adaptive_graph_disable_top_n_fallback:
            adaptive_v2_args.append(
                "--retraversal-allow-top-n-fallback"
            )
        if args.base_url:
            adaptive_v2_args.extend(
                ["--base-url", str(args.base_url)]
            )

        runner.run_stage(
            "[7.77/13] Adaptive grounded context expansion shadow",
            "scripts.discovery.run_adaptive_discovery_controller_v2_shadow",
            adaptive_v2_args,
            expected=[
                scientific_portfolio_adaptive_v2_summary,
                scientific_portfolio_adaptive_v2_effective_population,
                scientific_portfolio_adaptive_v2_lineage,
                scientific_portfolio_adaptive_v2_handoff,
            ],
        )

        adaptive_v2_payload = _load_json(
            scientific_portfolio_adaptive_v2_summary
        )

        if (
            adaptive_v2_payload.get(
                "initial_local_controller_reused"
            )
            is not True
        ):
            raise RuntimeError(
                "Stage 7.77 must reuse the completed Stage-7.76 local "
                "controller as context epoch 0."
            )

        if (
            adaptive_v2_payload.get(
                "stage8_input_changed"
            )
            is not False
            or adaptive_v2_payload.get(
                "production_selection_changed"
            )
            is not False
            or adaptive_v2_payload.get(
                "canonical_graph_mutated"
            )
            is not False
            or adaptive_v2_payload.get(
                "external_prior_art_as_positive_premise"
            )
            is not False
        ):
            raise RuntimeError(
                "Stage 7.77 authority/provenance contract violated."
            )

        runner.manifest[
            "adaptive_graph_retraversal_shadow"
        ] = {
            "enabled": True,
            "implementation_version":
                adaptive_v2_payload.get(
                    "implementation_version",
                    "adaptive-discovery-controller-v2b",
                ),
            "summary":
                str(
                    scientific_portfolio_adaptive_v2_summary
                ),
            "status":
                adaptive_v2_payload.get("status"),
            "epoch_count":
                adaptive_v2_payload.get(
                    "epoch_count", 0
                ),
            "graph_retraversal_execution_count":
                adaptive_v2_payload.get(
                    "graph_retraversal_execution_count",
                    0,
                ),
            "graph_retraversal_handoff_count":
                adaptive_v2_payload.get(
                    "graph_retraversal_handoff_count",
                    0,
                ),
            "context_lineage_event_count":
                adaptive_v2_payload.get(
                    "context_lineage_event_count",
                    0,
                ),
            "final_effective_count":
                adaptive_v2_payload.get(
                    "final_effective_count", 0
                ),
            "final_effective_hypothesis_ids":
                adaptive_v2_payload.get(
                    "final_effective_hypothesis_ids",
                    [],
                ),
            "effective_population":
                str(
                    scientific_portfolio_adaptive_v2_effective_population
                ),
            "context_lineage":
                str(
                    scientific_portfolio_adaptive_v2_lineage
                ),
            "remaining_handoff":
                str(
                    scientific_portfolio_adaptive_v2_handoff
                ),
            "stage_7_76_reused_as_epoch_0":
                True,
            "external_prior_art_as_positive_premise":
                False,
            "controller_has_novelty_authority":
                False,
            "controller_has_generation_authority":
                False,
            "n10_research_selection_authority":
                False,
            "stage8_input_changed":
                False,
            "production_selection_changed":
                False,
            "canonical_graph_mutated":
                False,
        }
        runner._save_manifest()
    else:
        runner.manifest[
            "adaptive_graph_retraversal_shadow"
        ] = {
            "enabled": False,
            "stage8_input_changed": False,
            "production_selection_changed": False,
            "canonical_graph_mutated": False,
        }
        runner._save_manifest()

    stage8_axis_plan_input = _stage8_axis_plan_input(
        control_plan=task_conditioned_axis_plan,
        open_world_plan=(
            None
            if open_world_outputs is None
            else open_world_outputs[
                "external_axis_plan"
            ]
        ),
        open_world_enabled=args.open_world_discovery,
    )

    # ------------------------------------------------------------------
    # 8-9. Alpha4 generation and semantic gate
    # ------------------------------------------------------------------
    axis_prefix = run / "hypothesis_axis_a4"
    axis_portfolio = run / "hypothesis_axis_a4.portfolio.json"
    axis_plan = run / "hypothesis_axis_a4.axis_plan.json"
    lineage = run / "hypothesis_axis_a4.lineage.json"
    axis_inference = run / "hypothesis_axis_a4.inference.json"
    axis_context = run / "hypothesis_axis_a4.context.json"
    axis_evidence_diversity = (
        run / "hypothesis_axis_a4.evidence_diversity.json"
    )

    stage8_expected = [
        axis_portfolio,
        axis_plan,
        lineage,
        axis_inference,
        axis_evidence_diversity,
    ]

    if context_review_adapter is not None:
        stage8_expected.append(
            axis_context
        )

    if args.realization_search_enforce:
        _run_realization_search_production_stage8(
            runner=runner,
            args=args,
            run=run,
            dual_context=dual_context,
            axis_prefix=axis_prefix,
            axis_portfolio=axis_portfolio,
            axis_plan=axis_plan,
            lineage=lineage,
            axis_inference=axis_inference,
            axis_context=axis_context,
            axis_evidence_diversity=axis_evidence_diversity,
            literature_provider_plan_path=literature_provider_plan_path,
            domain_profile_id=domain_profile.profile_id,
            context_review_enabled=(
                context_review_adapter
                is not None
            ),
            frozen_axis_plan_input=(
                stage8_axis_plan_input
            ),
            external_axis_bundle=(
                None
                if open_world_outputs is None
                else open_world_outputs[
                    "external_axis_bundle"
                ]
            ),
        )
    else:
        runner.run_stage(
            "[8/13] Discovery-axis hypothesis synthesis",
            "scripts.discovery.run_discovery_axis_hypothesis_maker",
            [
                "--dual-context", str(dual_context),
                *_base_model_args(args),
                *_mechanism_index_args(args),
                "--max-axes", str(args.max_axes),
                "--min-candidate-unit-score",
                str(args.min_candidate_unit_score),
                "--parse-retries", str(args.hypothesis_parse_retries),
                "--inference-critic-model", str(args.critic_model),
                *(
                    [
                        "--context-critic-model",
                        str(args.critic_model),
                    ]
                    if context_review_adapter is not None
                    else []
                ),
                "--axis-plan-input",
                str(stage8_axis_plan_input),
                *(
                    [
                        "--external-axis-bundle",
                        str(
                            open_world_outputs[
                                "external_axis_bundle"
                            ]
                        ),
                    ]
                    if open_world_outputs is not None
                    else []
                ),
                "--output-prefix", str(axis_prefix),
                "--save-prompts",
            ],
            expected=stage8_expected,
        )

    initial_hypotheses = _hypothesis_count(
        axis_portfolio
    )

    if context_review_adapter is not None:
        context_payload = _load_json(
            axis_context
        )

        if (
            context_payload.get("schema_version")
            != "discovery-axis-context-artifact-v3"
        ):
            raise RuntimeError(
                "Unexpected discovery-axis context artifact schema."
            )

        if (
            context_payload.get("domain_profile_id")
            != domain_profile.profile_id
        ):
            raise RuntimeError(
                "Context artifact domain_profile_id does not "
                "match the active E2E domain profile."
            )

        if (
            context_payload.get("adapter_id")
            != context_review_adapter.adapter_id
        ):
            raise RuntimeError(
                "Context artifact adapter_id does not match "
                "the resolved E2E context capability."
            )

        records = context_payload.get(
            "records"
        )

        review_history = context_payload.get(
            "review_history"
        )

        if not isinstance(
            records,
            list,
        ):
            raise RuntimeError(
                "Context artifact records must be a list."
            )

        if not isinstance(
            review_history,
            list,
        ):
            raise RuntimeError(
                "Context artifact review_history must be a list."
            )

        if (
            len(records)
            != initial_hypotheses
        ):
            raise RuntimeError(
                "Final context-review record count does not "
                "match the accepted Alpha4 hypothesis count: "
                f"context={len(records)}, "
                f"hypotheses={initial_hypotheses}"
            )

        portfolio_payload = _load_json(
            axis_portfolio
        )

        portfolio_rows = (
            portfolio_payload.get(
                "hypotheses"
            )
        )

        if not isinstance(
            portfolio_rows,
            list,
        ):
            raise RuntimeError(
                "Alpha4 portfolio hypotheses must be a list."
            )

        final_portfolio_ids = {
            str(
                row.get(
                    "hypothesis_id"
                )
            )
            for row in portfolio_rows
            if isinstance(
                row,
                dict,
            )
        }

        if (
            len(final_portfolio_ids)
            != len(portfolio_rows)
        ):
            raise RuntimeError(
                "Alpha4 portfolio contains missing or "
                "duplicate hypothesis IDs."
            )

        context_final_ids = set()

        for record in records:
            if not isinstance(
                record,
                dict,
            ):
                raise RuntimeError(
                    "Context final record must be an object."
                )

            required_record_fields = (
                "final_hypothesis_id",
                "axis_id",
                "source_review_hypothesis_id",
                "context_review_id",
                "status",
                "review",
            )

            for field in required_record_fields:
                if field not in record:
                    raise RuntimeError(
                        "Context final record missing field: "
                        f"{field}"
                    )

            final_hypothesis_id = str(
                record[
                    "final_hypothesis_id"
                ]
            )

            if (
                final_hypothesis_id
                in context_final_ids
            ):
                raise RuntimeError(
                    "Duplicate final hypothesis ID in "
                    "context artifact: "
                    f"{final_hypothesis_id}"
                )

            context_final_ids.add(
                final_hypothesis_id
            )

            review = record[
                "review"
            ]

            if not isinstance(
                review,
                dict,
            ):
                raise RuntimeError(
                    "Context final record review must be an object."
                )

            if (
                str(
                    review.get(
                        "hypothesis_id"
                    )
                )
                != str(
                    record[
                        "source_review_hypothesis_id"
                    ]
                )
            ):
                raise RuntimeError(
                    "Context final record source-review "
                    "hypothesis binding mismatch."
                )

            if (
                str(
                    review.get(
                        "review_id"
                    )
                )
                != str(
                    record[
                        "context_review_id"
                    ]
                )
            ):
                raise RuntimeError(
                    "Context final record review ID mismatch."
                )

            if (
                str(
                    review.get(
                        "status"
                    )
                )
                != str(
                    record[
                        "status"
                    ]
                )
            ):
                raise RuntimeError(
                    "Context final record status mismatch."
                )

        if (
            context_final_ids
            != final_portfolio_ids
        ):
            raise RuntimeError(
                "Context final hypothesis IDs do not match "
                "the accepted Alpha4 portfolio."
            )

        if (
            context_payload.get(
                "portfolio_id"
            )
            != portfolio_payload.get(
                "portfolio_id"
            )
        ):
            raise RuntimeError(
                "Context artifact portfolio_id mismatch."
            )

        if (
            context_payload.get(
                "final_record_count"
            )
            != len(records)
        ):
            raise RuntimeError(
                "Context artifact final_record_count mismatch."
            )

        if (
            context_payload.get(
                "review_history_count"
            )
            != len(review_history)
        ):
            raise RuntimeError(
                "Context artifact review_history_count mismatch."
            )

        for provenance_key in (
            "grounded_source_graph",
            "grounded_source_graph_sha256",
            "axis_source_graph",
            "axis_source_graph_sha256",
        ):
            if not str(
                context_payload.get(
                    provenance_key
                )
                or ""
            ).strip():
                raise RuntimeError(
                    "Context artifact dual-lane provenance "
                    f"is missing: {provenance_key}"
                )

        if (
            context_payload.get(
                "context_source_policy"
            )
            != "sers-dual-lane-claim-local-v1"
        ):
            raise RuntimeError(
                "Unexpected SERS context source policy."
            )

        if (
            context_payload.get(
                "action_policy_applied"
            )
            is not False
        ):
            raise RuntimeError(
                "S1 context artifact unexpectedly claims "
                "an action policy was applied."
            )

        runner.manifest[
            "context_review"
        ] = {
            "status":
                "assessed",
            "artifact":
                str(axis_context),
            "schema_version":
                context_payload.get(
                    "schema_version"
                ),
            "adapter_id":
                context_payload.get(
                    "adapter_id"
                ),
            "model":
                context_payload.get(
                    "model"
                ),
            "context_source_policy":
                context_payload.get(
                    "context_source_policy"
                ),

            "grounded_source_graph":
                context_payload.get(
                    "grounded_source_graph"
                ),
            "grounded_source_graph_sha256":
                context_payload.get(
                    "grounded_source_graph_sha256"
                ),
            "grounded_source_graph_node_count":
                context_payload.get(
                    "grounded_source_graph_node_count"
                ),
            "grounded_source_graph_edge_count":
                context_payload.get(
                    "grounded_source_graph_edge_count"
                ),

            "axis_source_graph":
                context_payload.get(
                    "axis_source_graph"
                ),
            "axis_source_graph_sha256":
                context_payload.get(
                    "axis_source_graph_sha256"
                ),
            "axis_source_graph_node_count":
                context_payload.get(
                    "axis_source_graph_node_count"
                ),
            "axis_source_graph_edge_count":
                context_payload.get(
                    "axis_source_graph_edge_count"
                ),
            "final_record_count":
                len(records),
            "review_history_count":
                len(review_history),
            "action_policy_applied":
                False,
            "g1_action_policy_deferred":
                True,
        }

    else:
        runner.manifest[
            "context_review"
        ] = {
            "status":
                (
                    "disabled_by_run_policy"
                    if context_review_mode == "off"
                    else "not_supported_for_domain"
                ),
            "artifact":
                None,
            "action_policy_applied":
                False,
        }

    diversity_payload = _load_json(
        axis_evidence_diversity
    )
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
    semantic_a4_run = run / "semantic_axis_a4.run.json"
    semantic_a4_review = run / "semantic_axis_a4.review.json"
    runner.run_stage(
        "[9/13] Semantic critic: alpha4 portfolio",
        "scripts.discovery.run_hypothesis_semantic_critic",
        [
            "--context", str(context),
            "--portfolio", str(axis_portfolio),
            *_base_model_args(args, critic=True),
            "--output-prefix", str(semantic_a4_prefix),
            "--save-prompt",
        ],
        expected=(
            [semantic_a4_run]
            if args.stop_after_initial_semantic
            else [semantic_a4_review]
        ),
    )

    if _finish_after_initial_semantic_if_requested(
        runner=runner,
        args=args,
        portfolio_path=axis_portfolio,
        context_path=context,
        semantic_run_path=semantic_a4_run,
        semantic_review_path=semantic_a4_review,
        provider_plan_path=literature_provider_plan_path,
    ):
        print(
            "Prospective initial-semantic cut point reached. "
            "External novelty, N9, N10, refinement, final semantic, "
            "and feasibility were not executed."
        )
        return 0

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
        "scripts.discovery.run_external_novelty",
        [
            "--portfolio", str(axis_portfolio),
            "--domain-profile", domain_profile.profile_id,
            "--lineage", str(lineage),
            "--inference-audit", str(axis_inference),
            *_base_model_args(args, critic=True),
            "--provider-plan", str(literature_provider_plan_path),
            "--results-per-query", str(args.results_per_query),
            "--pre-review-metadata-resolution",
            *_prior_art_memory_cli_args(args),
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

    nonobviousness_shadow = None

    if (
        args.nonobviousness_shadow
        or args.nonobviousness_full_shadow
        or args.nonobviousness_original_fallback_enforce
    ):
        nonobviousness_shadow = (
            run
            / "nonobviousness_n9.shadow.json"
        )

        runner.run_stage(
            "[10N9-a/13] Non-obviousness shadow intake",
            "scripts.discovery.build_nonobviousness_shadow",
            [
                "--query-plan",
                str(external_plan),
                "--external-report",
                str(external_report),
                "--portfolio",
                str(axis_portfolio),
                "--output",
                str(nonobviousness_shadow),
            ],
            expected=[
                nonobviousness_shadow,
            ],
        )

    nonobviousness_full_shadow = None

    if (
        args.nonobviousness_full_shadow
        or args.nonobviousness_original_fallback_enforce
    ):
        if nonobviousness_shadow is None:
            raise RuntimeError(
                "N9 full shadow requires the intake shadow artifact."
            )

        nonobviousness_full_shadow = (
            run
            / "nonobviousness_n9.full_shadow.json"
        )

        nonobviousness_ready_count = sum(
            len(
                row.get(
                    "ready_for_closure_claim_ids",
                    [],
                )
            )
            for row
            in _load_json(
                nonobviousness_shadow
            ).get(
                "hypotheses",
                [],
            )
        )

        runner.run_stage(
            "[10N9-b/13] Non-obviousness full closure shadow",
            "scripts.discovery.run_nonobviousness_full_shadow",
            [
                "--query-plan",
                str(external_plan),
                "--external-report",
                str(external_report),
                "--external-prior-art",
                str(external_prior),
                "--portfolio",
                str(axis_portfolio),
                "--hypothesis-context",
                str(context),
                "--intake-shadow",
                str(nonobviousness_shadow),
                "--provider-plan",
                str(literature_provider_plan_path),
                "--domain-profile",
                domain_profile.profile_id,
                *_base_model_args(
                    args,
                    critic=True,
                ),
                "--results-per-query",
                str(args.results_per_query),
                *(
                    [
                        "--max-ready-claims",
                        str(
                            max(
                                1,
                                nonobviousness_ready_count,
                            )
                        ),
                    ]
                    if args.nonobviousness_original_fallback_enforce
                    else []
                ),
                "--output",
                str(nonobviousness_full_shadow),
            ],
            expected=[
                nonobviousness_full_shadow,
            ],
        )

    scientific_novelty_action_batch = None
    scientific_novelty_gate = None

    if (
        args.nonobviousness_original_fallback_enforce
        and args.scientific_novelty_action_enforce
    ):
        raise RuntimeError(
            "N10 non-obviousness original-fallback authority "
            "and legacy scientific-novelty action authority "
            "are mutually exclusive. Use "
            "--scientific-novelty-action-shadow for comparison."
        )

    if (
        args.scientific_novelty_action_shadow
        or args.scientific_novelty_action_enforce
    ):
        scientific_novelty_action_batch = (
            _run_scientific_novelty_action_shadow_chain(
                runner=runner,
                args=args,
                run=run,
                external_report=external_report,
                external_plan=external_plan,
                external_prior=external_prior,
            )
        )

    if args.scientific_novelty_action_enforce:
        if scientific_novelty_action_batch is None:
            raise RuntimeError(
                "Scientific novelty enforcement requires "
                "the action batch to exist."
            )

        scientific_novelty_gate = (
            run
            / "scientific_novelty_fallback_gate_a10.production.json"
        )

        runner.run_stage(
            "[10P/13] Scientific novelty production fallback gate",
            "scripts.discovery."
            "build_scientific_novelty_production_gate",
            [
                "--action-batch",
                str(
                    scientific_novelty_action_batch
                ),
                "--output",
                str(
                    scientific_novelty_gate
                ),
            ],
            expected=[
                scientific_novelty_gate
            ],
        )

    elif args.nonobviousness_original_fallback_enforce:
        if (
            nonobviousness_shadow is None
            or nonobviousness_full_shadow is None
        ):
            raise RuntimeError(
                "N10 original-fallback enforcement requires "
                "both intake and full non-obviousness artifacts."
            )

        # --------------------------------------------------------------
        # Frozen V1 production artifact.
        #
        # Keep building this artifact for rollback, comparison, and
        # observational continuity. It is no longer the gate consumed
        # by Alpha6 in role-aware V2 production mode.
        # --------------------------------------------------------------
        nonobviousness_v1_production_gate = (
            run
            / "nonobviousness_n10."
              "fallback_gate.production.json"
        )

        runner.run_stage(
            "[10N10-P/13] N10 non-obviousness "
            "original-fallback production gate v1",
            "scripts.discovery."
            "build_nonobviousness_production_gate",
            [
                "--intake-shadow",
                str(nonobviousness_shadow),
                "--full-shadow",
                str(nonobviousness_full_shadow),
                "--output",
                str(
                    nonobviousness_v1_production_gate
                ),
            ],
            expected=[
                nonobviousness_v1_production_gate,
            ],
        )

        # --------------------------------------------------------------
        # Observational V1/V2 comparison remains non-authoritative.
        # --------------------------------------------------------------
        nonobviousness_dual_run_comparison = (
            run
            / "nonobviousness_n10."
              "dual_run_comparison.shadow.json"
        )

        runner.run_stage(
            "[10N10-C/13] N10 v1-v2 "
            "non-obviousness comparison shadow",
            "scripts.discovery."
            "build_nonobviousness_dual_run_comparison",
            [
                "--query-plan",
                str(external_plan),
                "--intake-shadow",
                str(nonobviousness_shadow),
                "--full-shadow",
                str(nonobviousness_full_shadow),
                "--runtime-authority-policy",
                "v2_production",
                "--output",
                str(
                    nonobviousness_dual_run_comparison
                ),
            ],
            expected=[
                nonobviousness_dual_run_comparison,
            ],
        )

        # --------------------------------------------------------------
        # Role-aware V2 candidate.
        #
        # This stage has zero production authority and must preserve
        # the frozen role-aware aggregation result exactly.
        # --------------------------------------------------------------
        nonobviousness_v2_candidate_gate = (
            run
            / "nonobviousness_n10."
              "fallback_gate_v2.candidate.json"
        )

        runner.run_stage(
            "[10N10-V2C/13] N10 role-aware "
            "original-fallback candidate gate",
            "scripts.discovery."
            "build_nonobviousness_production_gate_v2_candidate",
            [
                "--query-plan",
                str(external_plan),
                "--intake-shadow",
                str(nonobviousness_shadow),
                "--full-shadow",
                str(nonobviousness_full_shadow),
                "--output",
                str(
                    nonobviousness_v2_candidate_gate
                ),
            ],
            expected=[
                nonobviousness_v2_candidate_gate,
            ],
        )

        # --------------------------------------------------------------
        # Authoritative role-aware V2 production gate.
        #
        # This is the only N10 artifact passed to Alpha6 as original
        # fallback authority. V1 remains available above for audit and
        # rollback but is not consumed by Alpha6.
        # --------------------------------------------------------------
        scientific_novelty_gate = (
            run
            / "nonobviousness_n10."
              "fallback_gate_v2.production.json"
        )

        runner.run_stage(
            "[10N10-V2P/13] N10 role-aware "
            "original-fallback production gate v2",
            "scripts.discovery."
            "build_nonobviousness_production_gate_v2",
            [
                "--candidate-gate",
                str(
                    nonobviousness_v2_candidate_gate
                ),
                "--output",
                str(
                    scientific_novelty_gate
                ),
            ],
            expected=[
                scientific_novelty_gate,
            ],
        )

    if (
        args.nonobviousness_post_generation_enforce
        and not args.nonobviousness_original_fallback_enforce
    ):
        raise RuntimeError(
            "N10 post-generation enforcement requires "
            "--nonobviousness-original-fallback-enforce so "
            "original and generated candidates share the same "
            "non-obviousness authority."
        )

    if (
        args.nonobviousness_post_generation_enforce
        and args.post_generation_scientific_novelty_enforce
    ):
        raise RuntimeError(
            "N10 and legacy post-generation scientific novelty "
            "authorities are mutually exclusive."
        )

    # ------------------------------------------------------------------
    # 11. Alpha6 targeted novelty refinement
    # ------------------------------------------------------------------
    refinement_prefix = run / "novelty_refinement_a6"
    refined_portfolio = run / "novelty_refinement_a6.portfolio.json"
    refined_report = run / "novelty_refinement_a6.report.json"
    novelty_depth_edge_coverage = (
        run
        / "novelty_refinement_a6.novelty_depth_edge_coverage.json"
    )
    hypothesis_edge_graph_v2 = (
        run
        / "novelty_refinement_a6.hypothesis_edge_graph_v2.json"
    )
    knownness_depth_fusion = (
        run
        / "novelty_refinement_a6.knownness_depth_fusion.json"
    )
    actionable_refinement_plan = (
        run
        / "novelty_refinement_a6.actionable_refinement_plan.json"
    )
    controlled_routing_plan = (
        run
        / "novelty_refinement_a6.controlled_routing.plan.json"
    )
    controlled_routing_execution = (
        run
        / "novelty_refinement_a6.controlled_routing.execution.json"
    )
    negative_space_planner_plan = (
        run
        / "novelty_refinement_a6.negative_space_planner.plan.json"
    )
    negative_space_planner_execution = (
        run
        / "novelty_refinement_a6.negative_space_planner.execution.json"
    )
    runner.run_stage(
        "[11/13] Targeted novelty refinement alpha6",
        "scripts.discovery.run_novelty_refinement",
        [
            "--dual-context", str(dual_context),
            "--domain-profile", domain_profile.profile_id,
            "--axis-plan", str(axis_plan),
            "--portfolio", str(axis_portfolio),
            "--lineage", str(lineage),
            "--external-report", str(external_report),
            "--external-query-plan", str(external_plan),
            "--external-prior-art", str(external_prior),
            *(
                [
                    "--scientific-novelty-gate",
                    str(scientific_novelty_gate),
                ]
                if scientific_novelty_gate is not None
                else []
            ),
            *(
                [
                    "--question-task-preservation-enforce",
                ]
                if args.question_task_preservation_enforce
                else []
            ),
            *(
                [
                    "--post-generation-scientific-novelty-enforce",
                ]
                if args.post_generation_scientific_novelty_enforce
                else []
            ),
            *_mechanism_index_args(args),
            "--model", args.model,
            "--critic-model", args.critic_model,
            *( ["--base-url", args.base_url] if args.base_url else [] ),
            "--api-key-env", args.api_key_env,
            "--provider-plan", str(literature_provider_plan_path),
            "--results-per-query", str(args.results_per_query),
            *(
                [
                    "--planner-controlled-discovery-routing-experimental"
                ]
                if (
                    args
                    .planner_controlled_discovery_routing_experimental
                )
                else []
            ),
            "--output-prefix", str(refinement_prefix),
        ],
        expected=[
            refined_portfolio,
            refined_report,
            novelty_depth_edge_coverage,
            hypothesis_edge_graph_v2,
            knownness_depth_fusion,
            actionable_refinement_plan,
            controlled_routing_plan,
            controlled_routing_execution,
            negative_space_planner_plan,
            negative_space_planner_execution,
        ],
    )

    novelty_depth_payload = _load_json(
        novelty_depth_edge_coverage
    )
    runner.manifest[
        "novelty_depth_causal_edge_coverage"
    ] = {
        "status": "complete",
        "artifact": str(novelty_depth_edge_coverage),
        "hypothesis_count":
            novelty_depth_payload.get("hypothesis_count", 0),
        "novelty_depth_counts":
            novelty_depth_payload.get("novelty_depth_counts", {}),
        "planner_advisory_counts":
            novelty_depth_payload.get("planner_advisory_counts", {}),
        "weak_bridge_hypothesis_count":
            novelty_depth_payload.get(
                "weak_bridge_hypothesis_count",
                0,
            ),
        "higher_order_gap_hypothesis_count":
            novelty_depth_payload.get(
                "higher_order_gap_hypothesis_count",
                0,
            ),
        "shallow_local_extension_count":
            novelty_depth_payload.get(
                "shallow_local_extension_count",
                0,
            ),
        "foundational_knownness_checked": False,
        "conceptual_l1_l2_l3_integrated": False,
        "novelty_authority_created": False,
        "production_selection_changed": False,
    }
    runner._save_manifest()

    edge_graph_v2_payload = _load_json(
        hypothesis_edge_graph_v2
    )
    runner.manifest[
        "hypothesis_causal_edge_graph_v2"
    ] = {
        "status": "complete",
        "artifact": str(hypothesis_edge_graph_v2),
        "hypothesis_count":
            edge_graph_v2_payload.get(
                "hypothesis_count",
                0,
            ),
        "novelty_depth_counts":
            edge_graph_v2_payload.get(
                "novelty_depth_counts",
                {},
            ),
        "planner_advisory_counts":
            edge_graph_v2_payload.get(
                "planner_advisory_counts",
                {},
            ),
        "cross_claim_backbone_hypothesis_count":
            edge_graph_v2_payload.get(
                "cross_claim_backbone_hypothesis_count",
                0,
            ),
        "higher_order_gap_hypothesis_count":
            edge_graph_v2_payload.get(
                "higher_order_gap_hypothesis_count",
                0,
            ),
        "shallow_local_extension_count":
            edge_graph_v2_payload.get(
                "shallow_local_extension_count",
                0,
            ),
        "weak_bridge_hypothesis_count":
            edge_graph_v2_payload.get(
                "weak_bridge_hypothesis_count",
                0,
            ),
        "conceptual_knownness_signal_count":
            edge_graph_v2_payload.get(
                "conceptual_knownness_signal_count",
                0,
            ),
        "conceptual_knownness_changed_depth_count":
            0,
        "foundational_knownness_checked":
            False,
        "planner_input":
            True,
        "novelty_authority_created":
            False,
        "production_selection_changed":
            False,
    }
    runner._save_manifest()

    fusion_payload = _load_json(
        knownness_depth_fusion
    )
    action_payload = _load_json(
        actionable_refinement_plan
    )
    runner.manifest[
        "knownness_depth_actionable_refinement"
    ] = {
        "status": "complete",
        "fusion_artifact":
            str(knownness_depth_fusion),
        "action_plan_artifact":
            str(actionable_refinement_plan),
        "fusion_state_counts":
            fusion_payload.get(
                "fusion_state_counts",
                {},
            ),
        "conceptual_signal_counts":
            fusion_payload.get(
                "conceptual_signal_counts",
                {},
            ),
        "action_counts":
            action_payload.get(
                "action_counts",
                {},
            ),
        "operator_candidate_counts":
            action_payload.get(
                "operator_candidate_counts",
                {},
            ),
        "preferred_operator_counts":
            action_payload.get(
                "preferred_operator_counts",
                {},
            ),
        "alpha6_runtime_behavior_changed":
            False,
        "novelty_authority_created":
            False,
        "production_selection_changed":
            False,
    }
    runner._save_manifest()

    controlled_routing_payload = _load_json(
        controlled_routing_plan
    )
    controlled_routing_execution_payload = _load_json(
        controlled_routing_execution
    )
    runner.manifest[
        "planner_controlled_discovery_routing"
    ] = {
        "enabled":
            bool(
                args
                .planner_controlled_discovery_routing_experimental
            ),
        "authority_scope":
            "ALPHA6_EXPERIMENTAL_RUN_ONLY",
        "plan_artifact":
            str(controlled_routing_plan),
        "execution_artifact":
            str(controlled_routing_execution),
        "route_counts":
            controlled_routing_payload.get(
                "route_counts",
                {},
            ),
        "execution_match_count":
            controlled_routing_execution_payload.get(
                "match_count",
                0,
            ),
        "execution_diverged_count":
            controlled_routing_execution_payload.get(
                "diverged_count",
                0,
            ),
        "experimental_runtime_selection_changed_possible":
            bool(
                args
                .planner_controlled_discovery_routing_experimental
            ),
        "production_default_changed":
            False,
        "production_selection_authority":
            False,
    }
    runner._save_manifest()

    planner_payload = _load_json(
        negative_space_planner_plan
    )
    planner_execution_payload = _load_json(
        negative_space_planner_execution
    )
    runner.manifest[
        "negative_space_discovery_planner"
    ] = {
        "status": "complete",
        "plan_artifact":
            str(negative_space_planner_plan),
        "execution_artifact":
            str(negative_space_planner_execution),
        "target_count":
            planner_payload.get(
                "target_count",
                0,
            ),
        "retrieve_first_count":
            planner_payload.get(
                "retrieve_first_count",
                0,
            ),
        "evidence_reaxis_available_count":
            planner_payload.get(
                "evidence_reaxis_available_count",
                0,
            ),
        "operator_sharpen_available_count":
            planner_payload.get(
                "operator_sharpen_available_count",
                0,
            ),
        "novelty_depth_profile_consumed":
            planner_payload.get(
                "novelty_depth_profile_consumed",
                False,
            ),
        "novelty_depth_advisory_counts":
            planner_payload.get(
                "novelty_depth_advisory_counts",
                {},
            ),
        "actionable_refinement_plan_consumed":
            planner_payload.get(
                "actionable_refinement_plan_consumed",
                False,
            ),
        "actionable_refinement_action_counts":
            planner_payload.get(
                "actionable_refinement_action_counts",
                {},
            ),
        "actionable_operator_target_count":
            planner_payload.get(
                "actionable_operator_target_count",
                0,
            ),
        "execution_match_count":
            planner_execution_payload.get(
                "match_count",
                0,
            ),
        "execution_diverged_count":
            planner_execution_payload.get(
                "diverged_count",
                0,
            ),
        "planner_changed_runtime_behavior":
            False,
        "production_selection_changed":
            False,
    }
    runner._save_manifest()

    novelty_guided_negative_space = (
        run / "novelty_guided_discovery.negative_space.json"
    )
    novelty_guided_closure = (
        run / "novelty_guided_discovery.closure.json"
    )

    if args.novelty_guided_discovery_closure_shadow:
        novelty_guided_gap_plan = Path(
            str(refinement_prefix) + ".gap_plan.json"
        )

        runner.run_stage(
            "[11ng/13] Novelty-guided discovery closure shadow",
            "scripts.discovery.run_novelty_guided_discovery_closure",
            [
                "--context",
                str(context),
                "--portfolio",
                str(axis_portfolio),
                "--gap-plan",
                str(novelty_guided_gap_plan),
                "--evidence-diversity",
                str(axis_evidence_diversity),
                "--refinement-report",
                str(refined_report),
                "--plan-output",
                str(novelty_guided_negative_space),
                "--output",
                str(novelty_guided_closure),
            ],
            expected=[
                novelty_guided_negative_space,
                novelty_guided_closure,
            ],
        )

        ng_payload = _load_json(
            novelty_guided_closure
        )
        ng_plan_payload = _load_json(
            novelty_guided_negative_space
        )

        runner.manifest[
            "novelty_guided_discovery_closure"
        ] = {
            "enabled": True,
            "status": "complete",
            "negative_space_plan":
                str(novelty_guided_negative_space),
            "closure_report":
                str(novelty_guided_closure),
            "target_count":
                ng_plan_payload.get("target_count", 0),
            "regeneration_eligible_count":
                ng_plan_payload.get(
                    "regeneration_eligible_count", 0
                ),
            "operator_target_count":
                ng_plan_payload.get(
                    "operator_target_count", 0
                ),
            "operator_opportunity_count":
                ng_plan_payload.get(
                    "operator_opportunity_count", 0
                ),
            "evidence_alternative_capacity_target_count":
                ng_plan_payload.get(
                    "evidence_alternative_capacity_target_count",
                    0,
                ),
            "regeneration_generated_count":
                ng_payload.get(
                    "regeneration_generated_count", 0
                ),
            "accepted_regeneration_count":
                ng_payload.get(
                    "accepted_regeneration_count", 0
                ),
            "fresh_external_verification_count":
                ng_payload.get(
                    "fresh_external_verification_count", 0
                ),
            "closed_loop_observed_count":
                ng_payload.get(
                    "closed_loop_observed_count", 0
                ),
            "external_prior_art_as_positive_premise":
                False,
            "generation_authority_created":
                False,
            "novelty_authority_created":
                False,
            "candidate_survival_authority_created":
                False,
            "production_selection_changed":
                False,
        }
        runner._save_manifest()
    else:
        runner.manifest[
            "novelty_guided_discovery_closure"
        ] = {
            "enabled": False,
            "production_selection_changed": False,
        }
        runner._save_manifest()

    if args.nonobviousness_post_generation_enforce:
        post_n10_portfolio = (
            run
            / "novelty_refinement_a6."
              "n10.portfolio.json"
        )

        post_n10_report = (
            run
            / "novelty_refinement_a6."
              "n10.enforcement.json"
        )

        post_n10_details = (
            run
            / "novelty_refinement_a6."
              "n10.details"
        )

        post_n10_authority_mode = args.post_generation_n10_authority_mode
        post_n10_contract = _post_generation_n10_output_contract(
            run,
            post_n10_authority_mode,
        )
        post_n10_consumers = _post_generation_n10_consumer_contract(
            post_n10_authority_mode,
        )

        if post_n10_authority_mode == "hard_filter":
            post_n10_portfolio = post_n10_contract["legacy_output_portfolio"]
            post_n10_report = post_n10_contract["legacy_output_report"]
            if post_n10_portfolio is None or post_n10_report is None:
                raise RuntimeError("hard_filter output contract incomplete")

            runner.run_stage(
                "[11N10/13] Fresh Alpha6 candidate non-obviousness enforcement",
                "scripts.discovery.enforce_alpha6_nonobviousness",
                [
                    "--authority-mode",
                    "hard_filter",
                    "--portfolio",
                    str(refined_portfolio),
                    "--hypothesis-context",
                    str(context),
                    "--refinement-report",
                    str(refined_report),
                    "--external-dir",
                    str(Path(str(refinement_prefix) + ".external")),
                    "--provider-plan",
                    str(literature_provider_plan_path),
                    "--domain-profile",
                    domain_profile.profile_id,
                    "--model",
                    (args.critic_model or args.model),
                    *(
                        ["--base-url", args.base_url]
                        if args.base_url
                        else []
                    ),
                    "--api-key-env",
                    args.api_key_env,
                    *(
                        ["--device", getattr(args, "device")]
                        if getattr(args, "device", None)
                        else []
                    ),
                    "--results-per-query",
                    str(args.results_per_query),
                    "--work-dir",
                    str(post_n10_details),
                    "--output-portfolio",
                    str(post_n10_portfolio),
                    "--output-report",
                    str(post_n10_report),
                ],
                expected=[
                    post_n10_portfolio,
                    post_n10_report,
                ],
            )

            refined_portfolio = post_n10_portfolio

            runner.manifest["post_generation_n10_authority"] = {
                "mode": "hard_filter",
                "candidate_survival_authority": True,
                "novelty_certification_authority": True,
                "downstream_portfolio": str(refined_portfolio),
                "legacy_enforcement_report": str(post_n10_report),
                "consumer_bindings": post_n10_consumers,
            }

        else:
            post_n10_candidate_portfolio = post_n10_contract[
                "scientific_candidate_portfolio"
            ]
            post_n10_certification_report = post_n10_contract[
                "certification_report"
            ]
            post_n10_certified_portfolio = post_n10_contract[
                "certified_novelty_portfolio"
            ]
            if any(
                value is None
                for value in (
                    post_n10_candidate_portfolio,
                    post_n10_certification_report,
                    post_n10_certified_portfolio,
                )
            ):
                raise RuntimeError(
                    "certification_only output contract incomplete"
                )

            runner.run_stage(
                "[11N10/13] Fresh Alpha6 candidate novelty certification",
                "scripts.discovery.enforce_alpha6_nonobviousness",
                [
                    "--authority-mode",
                    "certification_only",
                    "--portfolio",
                    str(refined_portfolio),
                    "--hypothesis-context",
                    str(context),
                    "--refinement-report",
                    str(refined_report),
                    "--external-dir",
                    str(Path(str(refinement_prefix) + ".external")),
                    "--provider-plan",
                    str(literature_provider_plan_path),
                    "--domain-profile",
                    domain_profile.profile_id,
                    "--model",
                    (args.critic_model or args.model),
                    *(
                        ["--base-url", args.base_url]
                        if args.base_url
                        else []
                    ),
                    "--api-key-env",
                    args.api_key_env,
                    *(
                        ["--device", getattr(args, "device")]
                        if getattr(args, "device", None)
                        else []
                    ),
                    "--results-per-query",
                    str(args.results_per_query),
                    "--work-dir",
                    str(post_n10_details),
                    "--output-candidate-portfolio",
                    str(post_n10_candidate_portfolio),
                    "--output-certification-report",
                    str(post_n10_certification_report),
                    "--output-certified-portfolio",
                    str(post_n10_certified_portfolio),
                ],
                expected=[
                    post_n10_candidate_portfolio,
                    post_n10_certification_report,
                    post_n10_certified_portfolio,
                ],
            )

            refined_portfolio = post_n10_candidate_portfolio
            post_n10_portfolio = post_n10_candidate_portfolio
            post_n10_report = post_n10_certification_report

            certification_payload = _load_json(
                post_n10_certification_report
            )
            runner.manifest["post_generation_n10_authority"] = {
                "mode": "certification_only",
                "candidate_survival_authority": False,
                "novelty_certification_authority": True,
                "scientific_candidate_portfolio":
                    str(post_n10_candidate_portfolio),
                "certification_report":
                    str(post_n10_certification_report),
                "certified_novelty_portfolio":
                    str(post_n10_certified_portfolio),
                "downstream_portfolio": str(refined_portfolio),
                "scientific_candidate_count":
                    _hypothesis_count(post_n10_candidate_portfolio),
                "novelty_certified_count":
                    _hypothesis_count(post_n10_certified_portfolio),
                "novelty_unresolved_count":
                    certification_payload.get("unresolved_count", 0),
                "novelty_rejected_count":
                    certification_payload.get("rejected_count", 0),
                "certification_report_id":
                    certification_payload.get("report_id"),
                "consumer_bindings": post_n10_consumers,
            }

        runner._save_manifest()

        if args.nonobviousness_bounded_continuation_enforce:
            bounded_index_dir = (
                Path(
                    str(
                        args.data_root
                    )
                )
                / "corpus"
                / args.corpus_id
                / "mechanism"
                / "navigation"
                / "node_index"
            )

            bounded_continuation_dir = (
                run
                / "novelty_refinement_a6."
                  "n10.continuation"
            )

            bounded_continuation_report = (
                run
                / "novelty_refinement_a6."
                  "n10.continuation.json"
            )

            runner.run_stage(
                "[11N10-R/13] Bounded diagnostic "
                "post-generation continuation",
                "scripts.discovery."
                "run_n10_bounded_post_generation_continuation",
                [
                    "--first-post-n10-report",
                    str(
                        post_n10_report
                    ),
                    "--source-lineage",
                    str(
                        lineage
                    ),
                    "--axis-plan",
                    str(
                        axis_plan
                    ),
                    "--dual-context",
                    str(
                        dual_context
                    ),
                    "--provider-plan",
                    str(
                        literature_provider_plan_path
                    ),
                    "--index-dir",
                    str(
                        bounded_index_dir
                    ),
                    "--domain-profile",
                    domain_profile.profile_id,
                    "--model",
                    args.model,
                    "--critic-model",
                    args.critic_model,
                    *(
                        [
                            "--base-url",
                            args.base_url,
                        ]
                        if args.base_url
                        else []
                    ),
                    "--api-key-env",
                    args.api_key_env,
                    "--results-per-query",
                    str(
                        args.results_per_query
                    ),
                    *(
                        [
                            "--question-task-preservation-enforce",
                        ]
                        if args.question_task_preservation_enforce
                        else []
                    ),
                    "--work-dir",
                    str(
                        bounded_continuation_dir
                    ),
                    "--output-report",
                    str(
                        bounded_continuation_report
                    ),
                ],
                expected=[
                    bounded_continuation_report,
                ],
            )

            bounded_final_portfolio = (
                run
                / "novelty_refinement_a6."
                  "n10.bounded.portfolio.json"
            )

            bounded_merge_audit = (
                run
                / "novelty_refinement_a6."
                  "n10.bounded.merge_audit.json"
            )

            runner.run_stage(
                "[11N10-M/13] Merge authoritative "
                "first-pass and bounded-H2 survivors",
                "scripts.discovery."
                "merge_n10_bounded_continuation_portfolio",
                [
                    "--first-post-n10-portfolio",
                    str(
                        post_n10_portfolio
                    ),
                    "--first-post-n10-report",
                    str(
                        post_n10_report
                    ),
                    "--continuation-report",
                    str(
                        bounded_continuation_report
                    ),
                    "--output-portfolio",
                    str(
                        bounded_final_portfolio
                    ),
                    "--output-audit",
                    str(
                        bounded_merge_audit
                    ),
                ],
                expected=[
                    bounded_final_portfolio,
                    bounded_merge_audit,
                ],
            )

            refined_portfolio = (
                bounded_final_portfolio
            )

            bounded_report_payload = (
                _load_json(
                    bounded_continuation_report
                )
            )

            bounded_merge_payload = (
                _load_json(
                    bounded_merge_audit
                )
            )

            runner.manifest[
                "n10_bounded_continuation"
            ] = {
                "enabled":
                    True,

                "continuation_work_item_count":
                    bounded_report_payload.get(
                        "continuation_work_item_count"
                    ),

                "bounded_h2_survivor_count":
                    bounded_merge_payload.get(
                        "bounded_h2_survivor_count"
                    ),

                "continuation_report":
                    str(
                        bounded_continuation_report
                    ),

                "merge_audit":
                    str(
                        bounded_merge_audit
                    ),

                "final_portfolio":
                    str(
                        bounded_final_portfolio
                    ),

                "max_continuation_depth":
                    1,

                "scientific_policy_changed":
                    False,
            }

            runner._save_manifest()

        else:
            # Disabled staged mode retains the historical
            # first-post-N10 downstream binding established above.
            runner.manifest[
                "n10_bounded_continuation"
            ] = {
                "enabled":
                    False,

                "scientific_policy_changed":
                    False,
            }

            runner._save_manifest()

    if args.scientific_portfolio_production_enforce:
        verification_payload = _load_json(
            scientific_portfolio_verification_summary
        )
        production_binding = verification_payload.get(
            "production_binding",
            {},
        )
        if (
            verification_payload.get("n10_run") is not True
            or verification_payload.get("production_selection_authority")
            is not True
            or production_binding.get("status") != "COMPLETE"
            or production_binding.get("mode") != "certification_only"
            or production_binding.get("scientific_candidate_authority") is not True
            or production_binding.get("n10_candidate_survival_authority") is not False
            or not scientific_portfolio_production_candidate.is_file()
            or not scientific_portfolio_n10_certified.is_file()
        ):
            raise RuntimeError(
                "Scientific Portfolio production enforcement requested, "
                "but certification-only N10 integration is incomplete."
            )

        control_refined_portfolio = refined_portfolio
        refined_portfolio = scientific_portfolio_production_candidate

        runner.manifest["scientific_portfolio_production_binding"] = {
            "enabled": True,
            "mode": "SCIENTIFIC_PORTFOLIO_CANDIDATE_N10_CERTIFICATION",
            "authority_scope": "SCIENTIFIC_PORTFOLIO_CANDIDATE_SELECTION",
            "n10_required": True,
            "n10_run": True,
            "scientific_candidate_authority": True,
            "novelty_certification_authority": True,
            "n10_candidate_survival_authority": False,
            "conditional_candidates_retained": True,
            "production_selection_authority": True,
            "final_portfolio": str(refined_portfolio),
            "certified_novelty_portfolio": str(scientific_portfolio_n10_certified),
            "control_refined_portfolio": str(control_refined_portfolio),
            "stage8_input_changed": False,
            "legacy_alpha4_alpha6_retained_as_control": True,
        }
        runner._save_manifest()
    else:
        runner.manifest["scientific_portfolio_production_binding"] = {
            "enabled": False,
            "production_selection_authority": False,
        }
        runner._save_manifest()

    final_hypotheses = _hypothesis_count(refined_portfolio)
    runner.manifest["final_hypothesis_count"] = final_hypotheses
    runner._save_manifest()
    if final_hypotheses == 0:
        if args.scientific_portfolio_production_enforce:
            print(
                "No hypotheses survived Scientific Portfolio role-aware N10 "
                "production binding. Final semantic/feasibility stages are skipped."
            )
            runner.complete()
            return 0

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
        "scripts.discovery.run_hypothesis_semantic_critic",
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
            "scripts.discovery.run_feasibility_e2e",
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

    research_value_shadow = (
        run
        / "research_value.shadow.json"
    )

    if args.research_value_shadow:
        if feasibility_adapter is not None:
            runner.run_stage(
                "[13rv/13] Research value shadow",
                "scripts.discovery.run_research_value_shadow",
                [
                    "--portfolio",
                    str(refined_portfolio),
                    "--feasibility-dir",
                    str(feasibility_dir),
                    "--output",
                    str(research_value_shadow),
                ],
                expected=[
                    research_value_shadow
                ],
            )

            rv_payload = _load_json(
                research_value_shadow
            )

            runner.manifest[
                "research_value_shadow"
            ] = {
                "enabled":
                    True,
                "status":
                    "complete",
                "artifact":
                    str(
                        research_value_shadow
                    ),
                "hypothesis_count":
                    rv_payload.get(
                        "hypothesis_count",
                        0,
                    ),
                "assessed_count":
                    rv_payload.get(
                        "assessed_count",
                        0,
                    ),
                "novelty_signal_consumed":
                    False,
                "research_value_selection_authority":
                    False,
                "production_selection_changed":
                    False,
            }
            runner._save_manifest()
        else:
            runner.skip_stage(
                "[13rv/13] Research value shadow",
                reason=(
                    "Research-value shadow requires the domain feasibility "
                    "capability because experimental resolvability/resource "
                    "burden must not be guessed."
                ),
            )
            runner.manifest[
                "research_value_shadow"
            ] = {
                "enabled":
                    True,
                "status":
                    "not_supported_without_feasibility",
                "novelty_signal_consumed":
                    False,
                "research_value_selection_authority":
                    False,
                "production_selection_changed":
                    False,
            }
            runner._save_manifest()
    else:
        runner.manifest[
            "research_value_shadow"
        ] = {
            "enabled":
                False,
            "production_selection_changed":
                False,
        }
        runner._save_manifest()

    viewer_args = [
        "--run-dir", str(run),
        "--title", args.title,
    ]
    if feasibility_adapter is not None:
        viewer_args += ["--feasibility-dir", str(feasibility_dir)]

    runner.run_stage(
        "[13b/13] Demo viewer",
        "scripts.utilities.build_demo_viewer",
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



def _prior_art_memory_cli_args(
    args: argparse.Namespace,
) -> list[str]:
    values = (
        getattr(args, "prior_art_memory_query_plan", None),
        getattr(args, "prior_art_memory_report", None),
        getattr(args, "prior_art_memory_packet", None),
    )

    if not any(values):
        return []

    if not all(values):
        raise ValueError(
            "Prior-art memory requires all three E2E inputs: "
            "--prior-art-memory-query-plan, "
            "--prior-art-memory-report, and "
            "--prior-art-memory-packet."
        )

    return [
        "--prior-art-memory-query-plan",
        str(values[0]),
        "--prior-art-memory-report",
        str(values[1]),
        "--prior-art-memory-packet",
        str(values[2]),
    ]

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
        "--accepted-patterns",
        default=None,
        type=Path,
        help=(
            "Optional accepted bridge-pattern table "
            "(bridge_patterns.csv) for S22a "
            "authority-safe relation-component composition."
        ),
    )
    parser.add_argument(
        "--domain-profile",
        default="dac_her",
        help="Scientific domain profile propagated through discovery, novelty, refinement, and feasibility.",
    )
    parser.add_argument(
        "--context-review-mode",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Context-review capability policy. 'auto' uses a domain-owned "
            "context reviewer when registered; 'off' disables context review "
            "for common-denominator cross-domain evaluation without changing "
            "other scientific gates."
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stop", default=None)
    parser.add_argument("--target", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--question-task-preservation-shadow",
        action="store_true",
        help=(
            "Enable the shadow-only question-task-preservation chain: "
            "collect DiscoveryBundle semantic conflicts, review conflict "
            "candidates for question responsiveness, and emit pair-level "
            "arbitration proposals. Production selection remains unchanged."
        ),
    )
    parser.add_argument(
        "--nonobviousness-shadow",
        action="store_true",
        help=(
            "Materialize the N9 non-obviousness shadow intake "
            "after external novelty: atomic residue plus "
            "branch-specification gating only. No targeted closure, "
            "adjudication, refinement, or production selection is changed."
        ),
    )

    parser.add_argument(
        "--nonobviousness-full-shadow",
        action="store_true",
        help=(
            'Materialize the full N9 shadow chain after the intake artifact: targeted closure retrieval, slot review, evidence closure, structural/readiness gates, and only deterministic final dispositions. READY candidates receive independent adjudication. Production selection remains unchanged.'
        ),
    )

    parser.add_argument(
        "--nonobviousness-enforce",
        action="store_true",
        help=(
            "Enable full N10 scientific non-obviousness production "
            "enforcement. Original hypotheses must pass N10 before "
            "Alpha6 fallback, every accepted Alpha6 refinement or "
            "re-axis candidate must pass fresh external N10, and exact "
            "CONDITIONAL + REFINE_NOVELTY_BEARING_SPECIFICATION "
            "candidates receive at most one diagnostic-aware repair "
            "attempt followed by fresh H2 N10. Requires explicit "
            "--data-root for bounded continuation."
        ),
    )

    parser.add_argument(
        "--nonobviousness-original-fallback-enforce",
        action="store_true",
        help=(
            "Grant N10 atomic non-obviousness adjudication "
            "production authority over Alpha6 ORIGINAL fallback. "
            "Automatically materializes intake/full N10 closure and "
            "evaluates all READY atomic claims. This flag does not yet "
            "claim post-generation candidate enforcement; use only for "
            "the D2b staged integration path."
        ),
    )

    parser.add_argument(
        "--nonobviousness-post-generation-enforce",
        action="store_true",
        help=(
            "Require each Alpha6 accepted refinement or fresh "
            "re-axis candidate to pass a fresh N10 external "
            "closure and non-obviousness adjudication before "
            "stage 12/13 selection. Requires "
            "--nonobviousness-original-fallback-enforce."
        ),
    )


    parser.add_argument(
        "--post-generation-n10-authority-mode",
        choices=_POST_GENERATION_N10_AUTHORITY_MODES,
        default="hard_filter",
        help=(
            "Authority contract for post-generation N10. "
            "hard_filter preserves legacy candidate deletion. "
            "certification_only preserves Alpha6 scientific candidates "
            "and emits separate certification/certified artifacts."
        ),
    )

    parser.add_argument(
        "--nonobviousness-bounded-continuation-enforce",
        action="store_true",
        help=(
            "Staged N10 bounded-continuation integration. "
            "After first post-generation N10, only exact "
            "CONDITIONAL + REFINE_NOVELTY_BEARING_SPECIFICATION "
            "candidates receive one diagnostic-aware second Alpha6 "
            "attempt followed by fresh H2 N10 and deterministic "
            "authority-preserving final merge. Requires "
            "--nonobviousness-post-generation-enforce and an explicit "
            "--data-root. The public --nonobviousness-enforce switch "
            "also enables this path; this explicit flag remains available "
            "for staged/debug control."
        ),
    )

    parser.add_argument(
        "--scientific-novelty-action-shadow",
        action="store_true",
        help=(
            "Materialize scientific distinctiveness, two-pass semantic "
            "distinctiveness, and deterministic scientific-novelty action "
            "decisions after external novelty. Shadow only; Alpha6 and "
            "production selection remain unchanged."
        ),
    )

    parser.add_argument(
        "--scientific-novelty-action-enforce",
        action="store_true",
        help=(
            "Grant scientific-novelty action decisions production "
            "authority over Alpha6 original fallback. Automatically "
            "materializes the frozen scientific/semantic signal chain."
        ),
    )

    parser.add_argument(
        "--question-task-preservation-enforce",
        action="store_true",
        help=(
            "Require stable Question-to-fresh-reaxis task "
            "preservation before Alpha6 may accept a fresh re-axis."
        ),
    )

    parser.add_argument(
        "--post-generation-scientific-novelty-enforce",
        action="store_true",
        help=(
            "Require Alpha6-generated candidates to pass the frozen "
            "post-generation scientific/semantic novelty action policy "
            "before final acceptance."
        ),
    )

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
    parser.add_argument(
        "--min-candidate-unit-score",
        type=float,
        default=0.30,
        help=(
            "Shared candidate-unit quality floor used by both "
            "DiscoveryBundle reserved candidate selection and the "
            "Alpha4 discovery-axis planner."
        ),
    )
    parser.add_argument("--max-axes", type=int, default=5)
    parser.add_argument(
        "--direct-relationpattern-task-shadow",
        action="store_true",
        help=(
            "Run a shadow-only direct accepted-RelationPattern retrieval "
            "lane after Stage 7.5. The query is the graph retrieval "
            "source+target pair; the full scientific question is evaluated "
            "by the existing two-pass task-responsiveness critic. The "
            "artifact has no production-selection, novelty, or positive-"
            "premise authority."
        ),
    )
    parser.add_argument(
        "--direct-relationpattern-top-k",
        type=int,
        default=20,
        help=(
            "Frozen embedding-index top-k for the direct accepted "
            "RelationPattern shadow lane. Default: 20."
        ),
    )
    parser.add_argument(
        "--direct-higher-order-shadow",
        action="store_true",
        help=(
            "Continue the Stage-7.52 direct accepted-RelationPattern shadow "
            "through direct-task backbone materialization, candidate-modifier "
            "screening, direct higher-order topology composition, and canonical "
            "shadow hypothesis generation. Requires "
            "--direct-relationpattern-task-shadow. Production Stage-8 inputs "
            "and selection remain unchanged."
        ),
    )
    parser.add_argument(
        "--direct-higher-order-max-contexts",
        type=int,
        default=4,
        help=(
            "Maximum deterministic direct higher-order synthesis contexts "
            "evaluated by the shadow continuation lane. Default: 4."
        ),
    )
    parser.add_argument(
        "--planner-controlled-discovery-routing-experimental",
        action="store_true",
        help=(
            "Opt-in experimental Alpha6 routing controlled by the "
            "Actionable Refinement Plan. This may change the portfolio "
            "inside this run but does not change the production default."
        ),
    )
    parser.add_argument(
        "--novelty-guided-discovery-closure-shadow",
        action="store_true",
        help=(
            "Materialize a diagnostic NegativeSpace -> evidence/operator "
            "allocation -> Alpha6 regeneration -> fresh-verification closure "
            "after Stage 11. This does not change candidate survival or "
            "production selection."
        ),
    )
    parser.add_argument(
        "--research-value-shadow",
        action="store_true",
        help=(
            "Run diagnostic-only post-feasibility research-value assessment. "
            "Uses validation and experimental artifacts but intentionally "
            "does not consume novelty signals or change production selection."
        ),
    )
    parser.add_argument(
        "--direct-higher-order-downstream-shadow",
        action="store_true",
        help=(
            "Continue direct-HO shadow generation through non-blocking generic "
            "semantic review, existing external novelty, and stable structural "
            "BASE/MODIFIER/FULL views. Requires --direct-higher-order-shadow. "
            "No candidate-survival or production-selection authority is created."
        ),
    )
    parser.add_argument(
        "--higher-order-shadow",
        action="store_true",
        help=(
            "Run the authority-neutral S24 higher-order composition and "
            "canonical hypothesis-generation lane in parallel. An explicit "
            "--accepted-patterns table is reused when supplied; otherwise the "
            "lane receives a validator-approved run-local compatibility export "
            "from the canonical graph. Automatic export does not alter Stage-7.5 "
            "inputs or legacy production selection."
        ),
    )
    parser.add_argument(
        "--higher-order-max-contexts",
        type=int,
        default=12,
        help="Maximum deterministic modifier-coverage contexts in the shadow lane.",
    )
    parser.add_argument(
        "--open-world-discovery",
        action="store_true",
        help=(
            "Opt in to the bounded external-literature discovery-axis lane "
            "after the task-conditioned persistent-KG control plan. "
            "The external lane is inspiration-only, uses the frozen E2E "
            "provider configuration, and must produce at least one validated "
            "external axis; zero external axes fail closed rather than "
            "silently reverting to the control plan."
        ),
    )

    parser.add_argument(
        "--frontier-idea-population-shadow",
        action="store_true",
        help=(
            "Build Frontier Idea Population v1 after the Stage-7 "
            "exploration lanes. This is a shadow-only union of existing "
            "KG/open-world/higher-order/direct-HO/tension artifacts. "
            "It performs no new LLM or retrieval calls and does not alter "
            "the canonical Stage-8 axis-plan input or production selection."
        ),
    )


    parser.add_argument(
        "--idea-evolution-shadow",
        action="store_true",
        help=(
            "Run the shadow-only Idea Evolution milestone after Exploration "
            "Frontier factorization. Native operators perform cross-source "
            "bridging, backbone mutation, and candidate interpretation. The "
            "canonical Stage-8 input remains unchanged."
        ),
    )
    parser.add_argument(
        "--idea-evolution-max-cross-source-outputs",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--idea-evolution-max-backbone-mutation-outputs",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--idea-evolution-max-candidate-interpretation-outputs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--idea-evolution-max-candidate-parent-pool",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--idea-evolution-scientific-reframe-shadow",
        type=Path,
        default=None,
        help=(
            "Optional existing scientific_reframing_shadow.json to import into "
            "the common evolution lineage. This does not rerun reframing."
        ),
    )
    parser.add_argument(
        "--idea-evolution-proxy-challenge-shadow",
        type=Path,
        default=None,
        help=(
            "Optional existing PROXY_CHALLENGE shadow report to import into "
            "the common evolution lineage. This does not rerun proxy analysis."
        ),
    )


    parser.add_argument(
        "--scientific-portfolio-selection-shadow",
        action="store_true",
        help=(
            "Run shadow Scientific Portfolio Selection after Idea Evolution: "
            "structural projection, ordinal profile evaluation, Pareto/diversity "
            "retention, and grounded hypothesis materialization. Stage 8 remains unchanged."
        ),
    )
    parser.add_argument(
        "--scientific-portfolio-verification-shadow",
        action="store_true",
        help=(
            "Run existing semantic, external-prior-art, N9, and feasibility machinery "
            "over the grounded shadow portfolio produced by Scientific Portfolio Selection. "
            "N10 and production selection are not run."
        ),
    )
    parser.add_argument(
        "--scientific-portfolio-closed-loop-shadow",
        action="store_true",
        help=(
            "Run one bounded residual-aware novelty feedback generation after "
            "Scientific Portfolio verification: lower-order prior-art saturation, "
            "source-bound topology, bounded full-text escalation, residual-state "
            "compilation, one re-axis/refinement generation, fresh external "
            "verification, and claim-role-aware consolidation. Shadow only; "
            "Stage 8 and production selection remain unchanged."
        ),
    )

    parser.add_argument(
        "--adaptive-discovery-controller-shadow",
        action="store_true",
        help=(
            "Continue unresolved/rejected Stage-7.75 Scientific Portfolio "
            "candidates with bounded failure-aware escalation: retrieve more, "
            "same-premise sharpen, evidence re-axis, axis mutation, or an "
            "explicit graph-retraversal handoff. Shadow only."
        ),
    )
    parser.add_argument(
        "--adaptive-controller-max-rounds",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--adaptive-controller-max-local-attempts",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--adaptive-controller-disable-llm",
        action="store_true",
        help=(
            "Use deterministic bounded escalation only; skip the optional "
            "controller LLM advisory."
        ),
    )

    parser.add_argument(
        "--adaptive-graph-retraversal-shadow",
        action="store_true",
        help=(
            "Run Stage 7.77 grounded context expansion after Adaptive-v1. "
            "Stage 7.76 is reused verbatim as context epoch 0; the stage may "
            "select a different grounded KG neighborhood, rebuild a validated "
            "HypothesisContext, generate from structurally new premises, and "
            "restart bounded local search. Shadow only; Stage 8 and production "
            "selection remain unchanged."
        ),
    )
    parser.add_argument(
        "--adaptive-graph-max-retraversals",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--adaptive-graph-candidate-top-k",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--adaptive-graph-selected-top-k",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--adaptive-graph-paper-expansion-reserve",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--adaptive-graph-node-map-k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--adaptive-graph-endpoint-pair-k",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--adaptive-graph-max-depth-increment",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--adaptive-graph-max-depth-cap",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--adaptive-graph-min-new-edge-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--adaptive-graph-min-new-paper-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--adaptive-graph-max-prior-edge-jaccard",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--adaptive-graph-max-selected-edge-jaccard",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--adaptive-graph-disable-top-n-fallback",
        action="store_true",
        help=(
            "Disable the Stage-7.77 semantic-stop to top_n fallback. "
            "Default E2E policy allows the fallback, matching the upstream "
            "grounding failover philosophy."
        ),
    )

    parser.add_argument(
        "--scientific-portfolio-production-enforce",
        action="store_true",
        help=(
            "Opt-in development production integration. Promote the grounded "
            "Scientific Portfolio as the final scientific-candidate portfolio; "
            "run role-aware N10-v2 as novelty certification without deleting candidates. "
            "Legacy Alpha4/Alpha6 still runs as a control/rollback lane."
        ),
    )
    parser.add_argument(
        "--scientific-portfolio-max-evaluation-candidates",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--scientific-portfolio-max-retained",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--scientific-portfolio-max-retained-per-profile",
        type=int,
        default=2,
    )

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
    parser.add_argument(
        "--stop-after-initial-semantic",
        action="store_true",
        help=(
            "Prospective-evaluation cut point. Run through Alpha4 hypothesis "
            "generation and the initial semantic critic, then stop before "
            "external novelty, N9, N10, refinement, final semantic, or "
            "feasibility. In this mode semantic_axis_a4.run.json is the "
            "required stage-9 authority artifact and semantic_axis_a4.review.json "
            "is optional when the semantic hard gate fails. Default: disabled."
        ),
    )
    parser.add_argument(
        "--realization-search-width",
        type=int,
        choices=range(1, 5),
        default=3,
        help=(
            "Number of independent hypothesis realizations generated "
            "per discovery axis when production realization search is "
            "enabled. Default: 3. N7 budget-matched multi-axis search "
            "uses 1."
        ),
    )
    parser.add_argument(
        "--cross-axis-global-selection-enforce",
        action="store_true",
        help=(
            "After task-aware per-axis realization selection, "
            "collapse axis-local winners to one global canonical "
            "winner using stable semantic tier and frozen axis-plan "
            "order. Default: disabled."
        ),
    )
    parser.add_argument(
        "--realization-search-enforce",
        action="store_true",
        help=(
            "Production-authoritative realization search. "
            "Freeze one discovery-axis plan, generate the configured number "
            "of independent realizations per axis, evaluate each through "
            "external prior-art and two-pass semantic distinctiveness, "
            "retain the best stable determinate realization per axis, then "
            "continue downstream from the materialized winner portfolio."
        ),
    )
    parser.add_argument(
        "--prior-art-memory-query-plan",
        default=None,
        help=(
            "Optional historical LiteratureQueryPlan for "
            "cross-claim prior-art continuity."
        ),
    )
    parser.add_argument(
        "--prior-art-memory-report",
        default=None,
        help=(
            "Historical ExternalNoveltyReport corresponding "
            "to the memory query plan."
        ),
    )
    parser.add_argument(
        "--prior-art-memory-packet",
        default=None,
        help=(
            "Historical PriorArtPacket containing memory works."
        ),
    )

    args = parser.parse_args()

    if args.adaptive_graph_retraversal_shadow:
        # Adaptive Stage 7.77 consumes the exact completed Stage-7.76 local
        # controller as context epoch 0. Never rerun that epoch inside v2.
        args.adaptive_discovery_controller_shadow = True

    if args.adaptive_discovery_controller_shadow:
        # Adaptive Stage 7.76 consumes the frozen Stage-7.75 closed-loop
        # baseline and therefore enables its complete upstream shadow stack.
        args.scientific_portfolio_closed_loop_shadow = True

    if args.scientific_portfolio_closed_loop_shadow:
        # Closed-loop evaluation consumes the same grounded Scientific Portfolio
        # stack but remains authority-neutral and does not alter Stage 8.
        args.frontier_idea_population_shadow = True
        args.idea_evolution_shadow = True
        args.scientific_portfolio_selection_shadow = True
        args.scientific_portfolio_verification_shadow = True

    if args.adaptive_graph_max_retraversals < 1:
        parser.error(
            "--adaptive-graph-max-retraversals must be >= 1"
        )
    if args.adaptive_graph_candidate_top_k < 1:
        parser.error(
            "--adaptive-graph-candidate-top-k must be >= 1"
        )
    if (
        args.adaptive_graph_selected_top_k < 1
        or args.adaptive_graph_selected_top_k
        > args.adaptive_graph_candidate_top_k
    ):
        parser.error(
            "--adaptive-graph-selected-top-k must be between 1 and "
            "--adaptive-graph-candidate-top-k"
        )
    if (
        args.adaptive_graph_paper_expansion_reserve < 0
        or args.adaptive_graph_paper_expansion_reserve
        > args.adaptive_graph_selected_top_k
    ):
        parser.error(
            "--adaptive-graph-paper-expansion-reserve must be between 0 "
            "and --adaptive-graph-selected-top-k"
        )
    for flag, value in (
        (
            "--adaptive-graph-min-new-edge-fraction",
            args.adaptive_graph_min_new_edge_fraction,
        ),
        (
            "--adaptive-graph-min-new-paper-fraction",
            args.adaptive_graph_min_new_paper_fraction,
        ),
        (
            "--adaptive-graph-max-prior-edge-jaccard",
            args.adaptive_graph_max_prior_edge_jaccard,
        ),
        (
            "--adaptive-graph-max-selected-edge-jaccard",
            args.adaptive_graph_max_selected_edge_jaccard,
        ),
    ):
        if not 0.0 <= float(value) <= 1.0:
            parser.error(
                flag + " must be between 0 and 1"
            )

    if args.scientific_portfolio_production_enforce:
        # Production integration consumes the existing exploration stack.
        # Optional idea sources (open-world / HO / direct-HO) remain explicit.
        args.frontier_idea_population_shadow = True
        args.idea_evolution_shadow = True
        args.scientific_portfolio_selection_shadow = True
        args.scientific_portfolio_verification_shadow = True

    # Final N10 production mode.
    #
    # Keep the two staged flags as internal/debug controls, but the
    # public --nonobviousness-enforce switch activates BOTH sides of
    # the production authority:
    #
    #   1. original fallback enforcement;
    #   2. fresh post-generation candidate enforcement;
    #   3. one bounded diagnostic specification-repair continuation,
    #      followed by fresh H2 N10 and deterministic final merge.
    #
    if args.nonobviousness_enforce:
        args.nonobviousness_original_fallback_enforce = True
        args.nonobviousness_post_generation_enforce = True
        if args.post_generation_n10_authority_mode == "hard_filter":
            args.nonobviousness_bounded_continuation_enforce = True

    if (
        args.nonobviousness_bounded_continuation_enforce
        and args.post_generation_n10_authority_mode != "hard_filter"
    ):
        parser.error(
            "--nonobviousness-bounded-continuation-enforce currently "
            "requires --post-generation-n10-authority-mode hard_filter. "
            "The continuation merger still has legacy survivor semantics."
        )

    if (
        args.nonobviousness_bounded_continuation_enforce
        and not args.nonobviousness_post_generation_enforce
    ):
        parser.error(
            "--nonobviousness-bounded-continuation-enforce requires "
            "--nonobviousness-post-generation-enforce "
            "(or --nonobviousness-enforce)."
        )

    if (
        args.nonobviousness_bounded_continuation_enforce
        and not str(
            args.data_root
            or ""
        ).strip()
    ):
        parser.error(
            "--nonobviousness-bounded-continuation-enforce requires "
            "an explicit --data-root so the second Alpha6 mechanism "
            "index cannot fall back to repository-local historical data."
        )

    if (
        args.cross_axis_global_selection_enforce
        and not args.realization_search_enforce
    ):
        parser.error(
            "--cross-axis-global-selection-enforce requires "
            "--realization-search-enforce"
        )

    if not 0.0 <= args.min_candidate_unit_score <= 1.0:
        parser.error(
            "--min-candidate-unit-score must be between 0 and 1"
        )
    return args


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
