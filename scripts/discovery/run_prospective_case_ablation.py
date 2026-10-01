from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.prospective_validation import (
    ARM_ORDER,
    ProspectiveArmSelection,
    build_arm_metrics,
    build_case_audit,
    cap_legacy_portfolio,
    verification_summary_needs_resume,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
)


def _load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
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


def _run(
    label: str,
    cmd: list[str],
    log_dir: Path,
) -> int:
    print()
    print("=" * 96)
    print(label)
    print("=" * 96)
    print("$", " ".join(cmd))
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )
    safe = "_".join(
        label.lower().split()
    )
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (log_dir / f"{safe}.stdout.txt").write_text(
        result.stdout or "",
        encoding="utf-8",
    )
    (log_dir / f"{safe}.stderr.txt").write_text(
        result.stderr or "",
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(
            result.stderr.rstrip(),
            file=sys.stderr,
        )
    return result.returncode


def _portfolio_selected_arm_selection(
    *,
    pool: ScientificPortfolioCandidatePool,
    payload: dict[str, Any],
    max_hypotheses: int,
) -> ProspectiveArmSelection:
    selected_ids = [
        str(value)
        for value in payload.get(
            "retained_candidate_ids",
            [],
        )
    ]
    candidate_by_id = {
        row.candidate_id: row
        for row in pool.candidates
    }
    selected = [
        candidate_by_id[candidate_id]
        for candidate_id in selected_ids
        if candidate_id in candidate_by_id
    ]
    family_by_id = {
        str(row.get("candidate_id")):
            str(
                row.get(
                    "conceptual_family_signature"
                )
                or ""
            )
        for row in payload.get("entries", [])
        if isinstance(row, dict)
    }
    families = [
        family_by_id.get(
            row.candidate_id,
            row.conceptual_family_signature,
        )
        for row in selected
    ]
    selection_id = str(
        payload.get("selection_id")
        or "unknown-selection"
    )
    selection_sha = str(
        payload.get("selection_sha256")
        or "unknown-selection-sha"
    )

    return ProspectiveArmSelection(
        selection_id=(
            "prospective_alias:"
            + selection_id
        ),
        selection_sha256=selection_sha,
        arm="PORTFOLIO_SELECTED",
        source_pool_id=pool.pool_id,
        source_pool_sha256=pool.pool_sha256,
        source_context_id=pool.source_context_id,
        source_context_sha256=pool.source_context_sha256,
        max_hypotheses=max_hypotheses,
        eligible_candidate_count=(
            pool.projected_candidate_count
        ),
        selected_candidate_count=len(selected),
        selected_unique_family_count=(
            len(set(families))
        ),
        selected_count_by_origin=dict(
            payload.get(
                "retained_count_by_origin",
                {},
            )
        ),
        selected_count_by_form={
            key: sum(
                row.idea_form == key
                for row in selected
            )
            for key in sorted(
                {
                    row.idea_form
                    for row in selected
                }
            )
        },
        retained_candidate_ids=[
            row.candidate_id
            for row in selected
        ],
        selected_family_signatures=families,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run one four-arm creativity-stack ablation case: "
            "LEGACY vs FRONTIER_BALANCED vs EVOLUTION_BALANCED "
            "vs PORTFOLIO_SELECTED."
        )
    )
    p.add_argument(
        "--case-id",
        required=True,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--max-hypotheses",
        type=int,
        default=8,
    )
    p.add_argument(
        "--model",
        required=True,
    )
    p.add_argument(
        "--base-url",
        default=None,
    )
    p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    p.add_argument(
        "--providers",
        default="auto",
    )
    p.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )
    p.add_argument(
        "--force",
        action="store_true",
    )
    p.add_argument(
        "--rebuild-selected-arm",
        action="store_true",
        help=(
            "Re-run Scientific Portfolio Selection from the frozen "
            "Frontier/Evolution inputs into the ablation workspace. "
            "Use this for held-out runs frozen before arm-D execution."
        ),
    )
    return p


def main() -> int:
    args = parser().parse_args()
    if args.max_hypotheses < 1:
        raise ValueError(
            "--max-hypotheses must be >= 1"
        )

    run = args.run_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    context_path = run / "hypothesis.context.json"
    legacy_path = (
        run / "hypothesis_axis_a4.portfolio.json"
    )
    augmented = run / "frontier_augmented"
    population_path = (
        augmented
        / "frontier_idea_population.full.shadow.json"
    )
    frontier_audit_path = (
        augmented / "frontier_exploration.audit.json"
    )
    evolution_path = (
        augmented
        / "idea_evolution"
        / "idea_evolution.shadow.json"
    )
    scientific = (
        augmented / "scientific_portfolio"
    )
    provider_plan = (
        run / "literature_provider_plan.json"
    )

    frozen_required = [
        context_path,
        legacy_path,
        population_path,
        frontier_audit_path,
        evolution_path,
    ]
    missing = [
        str(path)
        for path in frozen_required
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "prospective case is missing required frozen "
            "legacy/Frontier/Evolution artifacts: "
            + repr(missing)
        )

    if args.rebuild_selected_arm:
        selected_source = (
            out
            / "PORTFOLIO_SELECTED"
            / "selection_source"
        )
        population_payload = _load(
            population_path
        )
        task_source = str(
            population_payload.get("task_source")
            or ""
        ).strip()
        task_target = str(
            population_payload.get("task_target")
            or ""
        ).strip()
        if not task_source or not task_target:
            raise ValueError(
                "Frontier population is missing task_source/task_target"
            )

        selected_candidate_pool = (
            selected_source / "candidate_pool.json"
        )
        selected_portfolio_source = (
            selected_source
            / "materialized.shadow.portfolio.json"
        )
        selected_selection_source = (
            selected_source / "selection.json"
        )
        if (
            args.force
            or not (
                selected_candidate_pool.is_file()
                and selected_portfolio_source.is_file()
                and selected_selection_source.is_file()
            )
        ):
            cmd = [
                sys.executable,
                "-m",
                "scripts.discovery."
                "run_scientific_portfolio_selection_shadow",
                "--context",
                str(context_path),
                "--population",
                str(population_path),
                "--frontier-audit",
                str(frontier_audit_path),
                "--evolution-report",
                str(evolution_path),
                "--task-source",
                task_source,
                "--task-target",
                task_target,
                "--max-evaluation-candidates",
                "48",
                "--max-retained",
                str(args.max_hypotheses),
                "--max-retained-per-profile",
                "2",
                "--output-dir",
                str(selected_source),
                "--save-prompts",
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
            ]
            if args.base_url:
                cmd += [
                    "--base-url",
                    args.base_url,
                ]
            rc = _run(
                f"{args.case_id} PORTFOLIO_SELECTED selection",
                cmd,
                out / "logs",
            )
            if rc != 0:
                raise RuntimeError(
                    "PORTFOLIO_SELECTED selection returned "
                    f"{rc}"
                )

        pool_path = selected_candidate_pool
        selected_path = selected_portfolio_source
        selected_selection_path = selected_selection_source
    else:
        pool_path = scientific / "candidate_pool.json"
        selected_path = (
            scientific
            / "materialized.shadow.portfolio.json"
        )
        selected_selection_path = (
            scientific / "selection.json"
        )
        current_required = [
            pool_path,
            selected_path,
            selected_selection_path,
        ]
        current_missing = [
            str(path)
            for path in current_required
            if not path.is_file()
        ]
        if current_missing:
            raise FileNotFoundError(
                "frozen Scientific Portfolio artifacts are missing; "
                "for a held-out run use --rebuild-selected-arm. Missing: "
                + repr(current_missing)
            )

    context = HypothesisContext.model_validate_json(
        context_path.read_text(
            encoding="utf-8"
        )
    )
    legacy = HypothesisPortfolio.model_validate_json(
        legacy_path.read_text(
            encoding="utf-8"
        )
    )
    pool = (
        ScientificPortfolioCandidatePool
        .model_validate_json(
            pool_path.read_text(
                encoding="utf-8"
            )
        )
    )
    selected_portfolio = (
        HypothesisPortfolio.model_validate_json(
            selected_path.read_text(
                encoding="utf-8"
            )
        )
    )
    selected_selection = (
        _portfolio_selected_arm_selection(
            pool=pool,
            payload=_load(
                selected_selection_path
            ),
            max_hypotheses=args.max_hypotheses,
        )
    )

    if (
        context.context_id
        != legacy.source_context_id
        or context.context_id
        != selected_portfolio.source_context_id
    ):
        raise ValueError(
            "case context / portfolio lineage mismatch"
        )

    arm_portfolios: dict[
        str,
        tuple[Path, ProspectiveArmSelection | None],
    ] = {}

    legacy_dir = out / "LEGACY"
    legacy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    legacy_out = (
        legacy_dir / "portfolio.json"
    )
    capped_legacy = cap_legacy_portfolio(
        legacy,
        max_hypotheses=args.max_hypotheses,
    )
    _write(
        legacy_out,
        capped_legacy,
    )
    arm_portfolios["LEGACY"] = (
        legacy_out,
        None,
    )

    for arm in (
        "FRONTIER_BALANCED",
        "EVOLUTION_BALANCED",
    ):
        arm_dir = out / arm
        portfolio_out = (
            arm_dir / "materialized.portfolio.json"
        )
        selection_out = (
            arm_dir / "selection.json"
        )
        if (
            args.force
            or not (
                portfolio_out.is_file()
                and selection_out.is_file()
            )
        ):
            cmd = [
                sys.executable,
                "-m",
                "scripts.discovery."
                "run_prospective_arm_materialization",
                "--context",
                str(context_path),
                "--candidate-pool",
                str(pool_path),
                "--arm",
                arm,
                "--max-hypotheses",
                str(args.max_hypotheses),
                "--output-dir",
                str(arm_dir),
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
                "--parse-retries",
                "2",
            ]
            if args.base_url:
                cmd += [
                    "--base-url",
                    args.base_url,
                ]
            rc = _run(
                f"{args.case_id} {arm} materialization",
                cmd,
                out / "logs",
            )
            if rc != 0:
                raise RuntimeError(
                    f"{arm} materialization returned {rc}"
                )

        selection = (
            ProspectiveArmSelection
            .model_validate_json(
                selection_out.read_text(
                    encoding="utf-8"
                )
            )
        )
        arm_portfolios[arm] = (
            portfolio_out,
            selection,
        )

    selected_dir = (
        out / "PORTFOLIO_SELECTED"
    )
    selected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    selected_out = (
        selected_dir / "portfolio.json"
    )
    # The selected arm is the already-frozen v2 result.
    # Copying it into the ablation workspace avoids mutating
    # or reinterpreting the source run.
    shutil.copyfile(
        selected_path,
        selected_out,
    )
    _write(
        selected_dir / "selection.alias.json",
        selected_selection,
    )
    arm_portfolios[
        "PORTFOLIO_SELECTED"
    ] = (
        selected_out,
        selected_selection,
    )

    metrics = []
    case_execution_status = "COMPLETE"
    paused_arm = None

    for arm in ARM_ORDER:
        portfolio_path, selection = arm_portfolios[arm]
        verification_dir = out / arm / "verification"
        summary_path = verification_dir / "verification.summary.json"

        needs_resume = True
        if not args.force and summary_path.is_file():
            try:
                existing_summary = _load(summary_path)
                needs_resume = verification_summary_needs_resume(existing_summary)
            except Exception:
                needs_resume = True

        rc = 0
        if args.force or needs_resume:
            cmd = [
                sys.executable,
                "-m",
                "scripts.discovery.run_standard_portfolio_verification_shadow",
                "--context",
                str(context_path),
                "--portfolio",
                str(portfolio_path),
                "--domain-profile",
                context.domain_profile_id,
                "--output-dir",
                str(verification_dir),
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
                "--providers",
                args.providers,
                "--results-per-query",
                str(args.results_per_query),
            ]
            if args.base_url:
                cmd += ["--base-url", args.base_url]
            if provider_plan.is_file():
                cmd += ["--provider-plan", str(provider_plan)]
            if args.force:
                cmd += ["--force"]

            rc = _run(
                f"{args.case_id} {arm} verification",
                cmd,
                out / "logs",
            )
        else:
            print(f"Reusing terminal verification summary for {arm}: {summary_path}")

        if not summary_path.is_file():
            raise RuntimeError(
                f"{arm} verification produced no summary; return_code={rc}"
            )

        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )
        verification = _load(summary_path)
        row = build_arm_metrics(
            case_id=args.case_id,
            arm=arm,
            context=context,
            portfolio=portfolio,
            portfolio_path=str(portfolio_path),
            verification_summary=verification,
            arm_selection=selection,
        )
        metrics.append(row)

        if row.operational_status == "PAUSED_PROVIDER_BUDGET_EXHAUSTED" or rc == 75:
            case_execution_status = "PAUSED_PROVIDER_BUDGET_EXHAUSTED"
            paused_arm = arm
            audit = build_case_audit(
                case_id=args.case_id,
                run_dir=str(run),
                context=context,
                arms=metrics,
                execution_status=case_execution_status,
                paused_arm=arm,
            )
            _write(out / "case.audit.json", audit)
            print()
            print("CASE_PAUSED_PROVIDER_BUDGET_EXHAUSTED")
            print("arm:", arm)
            print("resume stage:", row.provider_budget_pause_stage)
            print("audit:", out / "case.audit.json")
            return 75

        if row.operational_status == "FAILED_OPERATIONAL" or rc not in {0, 1}:
            case_execution_status = "PARTIAL"
        elif rc == 1:
            case_execution_status = "PARTIAL"

    audit = build_case_audit(
        case_id=args.case_id,
        run_dir=str(run),
        context=context,
        arms=metrics,
        execution_status=case_execution_status,
        paused_arm=paused_arm,
    )
    _write(out / "case.audit.json", audit)

    print()
    print("===== PROSPECTIVE CASE ABLATION =====")
    print("case:", args.case_id)
    for row in audit.arms:
        print(
            row.arm,
            "| hypotheses=",
            row.hypothesis_count,
            "| semantic=",
            row.semantic_status,
            "| external=",
            row.external_novelty_status,
            "| N9=",
            row.n9_status,
            "| feasibility=",
            row.feasibility_status,
            "| evidence coverage=",
            round(row.evidence_eligible_statement_coverage, 3),
            "| materialization yield=",
            (
                round(row.materialization_yield_fraction, 3)
                if row.materialization_yield_fraction is not None
                else "NA"
            ),
            "| conceptual family fraction=",
            (
                round(row.conceptual_selected_family_fraction, 3)
                if row.conceptual_selected_family_fraction is not None
                else "NA"
            ),
            "| premise distinct fraction=",
            round(row.premise_distinct_set_fraction, 3),
            "| operational=",
            row.operational_status,
        )
    print("CASE_EXECUTION_STATUS=", audit.execution_status)
    print(
        "ALL_FOUR_ARMS_PRESENT=",
        audit.all_four_arms_present,
    )
    print(
        "PRODUCTION_SELECTION_AUTHORITY=False"
    )
    print(
        "audit:",
        out / "case.audit.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
