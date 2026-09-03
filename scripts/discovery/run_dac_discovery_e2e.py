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

    runner.run_stage(
        "[7.5/13] Task-conditioned discovery-axis plan",
        "scripts.discovery.build_task_conditioned_axis_plan",
        [
            "--question", str(args.question),
            "--final-traversal", str(final_traversal),
            "--candidate-traversal", str(candidate_traversal),
            "--discovery-bundle", str(bundle),
            "--dual-context", str(dual_context),
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
        ],
        expected=[
            task_conditioned_dual_context,
            task_conditioned_axis_plan,
            task_conditioned_axis_report,
        ],
    )

    dual_context = task_conditioned_dual_context

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
                task_conditioned_axis_plan
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
                str(task_conditioned_axis_plan),
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
        "scripts.discovery.run_external_novelty",
        [
            "--portfolio", str(axis_portfolio),
            "--domain-profile", domain_profile.profile_id,
            "--lineage", str(lineage),
            "--inference-audit", str(axis_inference),
            *_base_model_args(args, critic=True),
            "--provider-plan", str(literature_provider_plan_path),
            "--results-per-query", str(args.results_per_query),
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

        scientific_novelty_gate = (
            run
            / "nonobviousness_n10."
              "fallback_gate.production.json"
        )

        runner.run_stage(
            "[10N10-P/13] N10 non-obviousness "
            "original-fallback production gate",
            "scripts.discovery."
            "build_nonobviousness_production_gate",
            [
                "--intake-shadow",
                str(nonobviousness_shadow),
                "--full-shadow",
                str(nonobviousness_full_shadow),
                "--output",
                str(scientific_novelty_gate),
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
            "--output-prefix", str(refinement_prefix),
        ],
        expected=[refined_portfolio, refined_report],
    )

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

        runner.run_stage(
            "[11N10/13] Fresh Alpha6 candidate "
            "non-obviousness enforcement",
            "scripts.discovery."
            "enforce_alpha6_nonobviousness",
            [
                "--portfolio",
                str(refined_portfolio),
                "--hypothesis-context",
                str(context),
                "--refinement-report",
                str(refined_report),
                "--external-dir",
                str(
                    Path(
                        str(
                            refinement_prefix
                        )
                        + ".external"
                    )
                ),
                "--provider-plan",
                str(
                    literature_provider_plan_path
                ),
                "--domain-profile",
                domain_profile.profile_id,
                "--model",
                (
                    args.critic_model
                    or args.model
                ),
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
                *(
                    [
                        "--device",
                        getattr(args, "device"),
                    ]
                    if getattr(args, "device", None)
                    else []
                ),
                "--results-per-query",
                str(
                    args.results_per_query
                ),
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

        # Stage 12/13 now consume the N10-filtered
        # selection artifact, while the original Alpha6
        # portfolio/report remain intact for lineage audit.
        refined_portfolio = (
            post_n10_portfolio
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
            "Alpha6 fallback, and every accepted Alpha6 refinement "
            "or re-axis candidate must pass a fresh external N10 "
            "closure and adjudication before downstream selection."
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

    # Final N10 production mode.
    #
    # Keep the two staged flags as internal/debug controls, but the
    # public --nonobviousness-enforce switch activates BOTH sides of
    # the production authority:
    #
    #   1. original fallback enforcement;
    #   2. fresh post-generation candidate enforcement.
    #
    if args.nonobviousness_enforce:
        args.nonobviousness_original_fallback_enforce = True
        args.nonobviousness_post_generation_enforce = True

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
