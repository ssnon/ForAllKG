from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.prospective_relational_campaign import (
    ProspectiveCampaignStageRecord,
    build_campaign_launch,
    build_campaign_result,
    build_case_result,
)
from pipeline_core.discovery.prospective_relational_execution_plan import (
    ProspectiveRelationalExecutionPlan,
    ProspectiveRelationalHypothesisSelection,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    RelationalAtomicEndpointBindingReport,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    RelationalScientificVerifierRunManifest,
    write_json_exclusive,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_dirty(cwd: Path | None = None) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return bool(status.strip())


def _require_ancestor(older: str, newer: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "execution-plan scientific HEAD is not an ancestor "
            "of the launcher repository HEAD"
        )


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def _argv_value(argv: list[str], flag: str) -> str:
    try:
        index = argv.index(flag)
    except ValueError as exc:
        raise ValueError("frozen argv missing " + flag) from exc
    if index + 1 >= len(argv):
        raise ValueError("frozen argv missing value for " + flag)
    return argv[index + 1]


def _run_stage(
    *,
    stage_name: str,
    argv: list[str],
    cwd: Path,
    log_path: Path,
) -> ProspectiveCampaignStageRecord:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise ValueError("stage log already exists: " + str(log_path))

    command = list(argv)
    if command and command[0] == "python":
        command[0] = sys.executable

    print()
    print("==", stage_name, "==")
    print("cwd:", cwd)
    print("argv:", " ".join(command))

    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

    return ProspectiveCampaignStageRecord(
        stage_name=stage_name,
        argv=command,
        return_code=return_code,
        log_path=str(log_path),
    )


def _main_manifest_status(run_dir: Path) -> str | None:
    path = run_dir / "e2e_runner.manifest.json"
    if not path.is_file():
        return None
    value = _json_object(path).get("status")
    return str(value) if value is not None else None


def _stage_failed(
    row: ProspectiveCampaignStageRecord,
) -> bool:
    return row.return_code != 0


def _ensure_case_dirs_fresh(plan: ProspectiveRelationalExecutionPlan) -> None:
    for case in plan.cases:
        path = Path(case.run_dir)
        if path.exists():
            if not path.is_dir():
                raise ValueError("case run path is not a directory: " + str(path))
            if any(path.iterdir()):
                raise ValueError(
                    "prospective case run directory is not fresh: " + str(path)
                )


def _setup_pinned_worktree(
    *,
    repository_root: Path,
    worktree_path: Path,
    scientific_head: str,
) -> None:
    if worktree_path.exists():
        raise ValueError(
            "pinned scientific worktree path already exists: "
            + str(worktree_path)
        )
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(worktree_path),
            scientific_head,
        ],
        cwd=repository_root,
        check=True,
    )
    if _git_head(worktree_path) != scientific_head:
        raise ValueError("pinned worktree HEAD mismatch")
    if _tracked_dirty(worktree_path):
        raise ValueError("pinned scientific worktree is dirty")


def _result_and_write(
    *,
    result_path: Path,
    case,
    plan: ProspectiveRelationalExecutionPlan,
    stages: list[ProspectiveCampaignStageRecord],
    disposition: str,
    **kwargs: object,
):
    result = build_case_result(
        case_id=case.case_id,
        source_task_id=case.source_task_id,
        execution_plan_id=plan.plan_id,
        scientific_repository_head_sha=(
            plan.execution_plan_repository_head_sha
        ),
        disposition=disposition,
        stage_records=[row.model_dump(mode="json") for row in stages],
        **kwargs,
    )
    write_json_exclusive(result_path, result)
    print("Case disposition:", result.disposition)
    print("Case result:", result_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute frozen P06-P10 prospective relational-verifier cases "
            "sequentially under the exact repository HEAD recorded by the "
            "execution plan. Every case is accounted for; failures and "
            "abstentions never trigger replacement or later-case adaptation."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument(
        "--campaign-root",
        required=True,
        type=Path,
        help="Must equal the common parent of the frozen P06-P10 run dirs.",
    )
    parser.add_argument(
        "--repository-root",
        default=Path.cwd(),
        type=Path,
    )
    args = parser.parse_args()

    plan_path = args.execution_plan.expanduser().resolve()
    campaign_root = args.campaign_root.expanduser().resolve()
    repository_root = args.repository_root.expanduser().resolve()

    if not plan_path.is_file():
        raise ValueError("missing execution plan: " + str(plan_path))
    plan = ProspectiveRelationalExecutionPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )

    if [Path(row.run_dir).parent.resolve() for row in plan.cases] != [
        campaign_root
    ] * 5:
        raise ValueError(
            "campaign root does not match frozen case run-dir parents"
        )

    launcher_head = _git_head(repository_root)
    if _tracked_dirty(repository_root):
        raise ValueError("campaign launcher requires a clean tracked worktree")
    _require_ancestor(
        plan.execution_plan_repository_head_sha,
        launcher_head,
    )
    _ensure_case_dirs_fresh(plan)

    launch_path = campaign_root / "P06_P10.campaign_launch.json"
    final_path = campaign_root / "P06_P10.campaign_result.json"
    logs_dir = campaign_root / "P06_P10.logs"
    results_dir = campaign_root / "P06_P10.case_results"
    worktree_path = campaign_root / (
        "_frozen_code_" + plan.execution_plan_repository_head_sha[:12]
    )

    for reserved in (launch_path, final_path, logs_dir, results_dir, worktree_path):
        if reserved.exists():
            raise ValueError(
                "prospective campaign reserved path already exists: "
                + str(reserved)
            )

    _setup_pinned_worktree(
        repository_root=repository_root,
        worktree_path=worktree_path,
        scientific_head=plan.execution_plan_repository_head_sha,
    )

    launch = build_campaign_launch(
        execution_plan_id=plan.plan_id,
        execution_plan_sha256=plan.plan_sha256,
        execution_plan_file_sha256=_sha256_file(plan_path),
        launcher_repository_head_sha=launcher_head,
        scientific_repository_head_sha=(
            plan.execution_plan_repository_head_sha
        ),
        pinned_worktree_path=str(worktree_path),
    )
    write_json_exclusive(launch_path, launch)

    results = []
    for case in plan.cases:
        print()
        print("#" * 100)
        print("CASE", case.case_id)
        print("#" * 100)

        run_dir = Path(case.run_dir)
        case_log_dir = logs_dir / case.case_id
        result_path = results_dir / (case.case_id + ".json")
        stages: list[ProspectiveCampaignStageRecord] = []

        main = _run_stage(
            stage_name=case.case_id + ":main_e2e",
            argv=case.main_e2e_argv,
            cwd=worktree_path,
            log_path=case_log_dir / "01_main_e2e.log",
        )
        stages.append(main)
        main_status = _main_manifest_status(run_dir)

        if _stage_failed(main):
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="MAIN_E2E_STAGE_FAILED",
                    main_e2e_manifest_status=main_status,
                )
            )
            continue

        required_binding_inputs = [
            run_dir / "novelty_refinement_a6.n10.candidate.portfolio.json",
            run_dir / "novelty_refinement_a6.n10.certification.json",
        ]
        if not all(path.is_file() for path in required_binding_inputs):
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="UPSTREAM_NO_RELATIONAL_BINDING_INPUTS",
                    main_e2e_manifest_status=main_status,
                )
            )
            continue

        build_plan = _run_stage(
            stage_name=case.case_id + ":build_binding_plan",
            argv=[
                "python",
                "-m",
                "scripts.discovery.build_relational_atomic_binding_plan",
                "--run-dir",
                str(run_dir),
                "--output",
                case.full_binding_plan_path,
            ],
            cwd=worktree_path,
            log_path=case_log_dir / "02_build_binding_plan.log",
        )
        stages.append(build_plan)
        if _stage_failed(build_plan):
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="BINDING_PLAN_STAGE_FAILED",
                    main_e2e_manifest_status=main_status,
                )
            )
            continue

        selection = _run_stage(
            stage_name=case.case_id + ":structural_selection",
            argv=[
                "python",
                "-m",
                "scripts.discovery.select_prospective_relational_binding_plan",
                "--plan",
                case.full_binding_plan_path,
                "--selection-output",
                case.selection_report_path,
                "--selected-plan-output",
                case.selected_binding_plan_path,
            ],
            cwd=worktree_path,
            log_path=case_log_dir / "03_structural_selection.log",
        )
        stages.append(selection)
        if _stage_failed(selection):
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="STRUCTURAL_SELECTION_STAGE_FAILED",
                    main_e2e_manifest_status=main_status,
                )
            )
            continue

        selection_report = (
            ProspectiveRelationalHypothesisSelection.model_validate_json(
                Path(case.selection_report_path).read_text(encoding="utf-8")
            )
        )
        if selection_report.status == "NO_BINDING_READY_HYPOTHESIS":
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="NO_BINDING_READY_HYPOTHESIS",
                    main_e2e_manifest_status=main_status,
                )
            )
            continue

        endpoint = _run_stage(
            stage_name=case.case_id + ":endpoint_binding",
            argv=[
                "python",
                "-m",
                "scripts.discovery.run_relational_atomic_endpoint_binding",
                "--plan",
                case.selected_binding_plan_path,
                "--output",
                case.endpoint_binding_report_path,
                "--prompt-output",
                case.endpoint_binding_prompt_path,
                "--model",
                case.critic_model,
                "--base-url",
                plan.settings.base_url,
                "--api-key-env",
                plan.settings.api_key_env,
                "--temperature",
                str(plan.settings.endpoint_temperature),
                "--parse-retries",
                str(plan.settings.endpoint_parse_retries),
                "--timeout",
                str(plan.settings.endpoint_timeout_seconds),
                "--telemetry",
                case.endpoint_binding_telemetry_path,
            ],
            cwd=worktree_path,
            log_path=case_log_dir / "04_endpoint_binding.log",
        )
        stages.append(endpoint)
        if _stage_failed(endpoint):
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="ENDPOINT_BINDING_STAGE_FAILED",
                    main_e2e_manifest_status=main_status,
                    selected_final_hypothesis_id=(
                        selection_report.selected_final_hypothesis_id
                    ),
                    selected_candidate_hypothesis_id=(
                        selection_report.selected_candidate_hypothesis_id
                    ),
                    selected_original_hypothesis_id=(
                        selection_report.selected_original_hypothesis_id
                    ),
                )
            )
            continue

        endpoint_report = RelationalAtomicEndpointBindingReport.model_validate_json(
            Path(case.endpoint_binding_report_path).read_text(encoding="utf-8")
        )
        endpoint_fields = {
            "endpoint_selected_claim_count":
                endpoint_report.selected_claim_count,
            "endpoint_bound_claim_count":
                endpoint_report.bound_claim_count,
            "endpoint_abstained_claim_count":
                endpoint_report.abstained_claim_count,
            "endpoint_novelty_bearing_bound_claim_count":
                endpoint_report.novelty_bearing_bound_claim_count,
        }
        lineage_fields = {
            "selected_final_hypothesis_id":
                selection_report.selected_final_hypothesis_id,
            "selected_candidate_hypothesis_id":
                selection_report.selected_candidate_hypothesis_id,
            "selected_original_hypothesis_id":
                selection_report.selected_original_hypothesis_id,
        }

        if endpoint_report.novelty_bearing_bound_claim_count < 1:
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="ENDPOINT_BINDING_ABSTAINED",
                    main_e2e_manifest_status=main_status,
                    **lineage_fields,
                    **endpoint_fields,
                )
            )
            continue

        provider_plan = run_dir / "literature_provider_plan.json"
        external_report = Path(
            str(selection_report.selected_external_report)
        ).expanduser().resolve()
        if not provider_plan.is_file() or not external_report.is_file():
            # These are frozen lineage inputs required by the verifier.
            # Treat absence as a verifier-stage infrastructure failure.
            synthetic = ProspectiveCampaignStageRecord(
                stage_name=case.case_id + ":verifier_input_preflight",
                argv=[],
                return_code=1,
                log_path="",
            )
            stages.append(synthetic)
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="VERIFIER_STAGE_FAILED",
                    main_e2e_manifest_status=main_status,
                    **lineage_fields,
                    **endpoint_fields,
                )
            )
            continue

        verifier_argv = [
            "python",
            "-m",
            "scripts.discovery.run_relational_scientific_verifier_shadow_e2e",
            "--plan",
            case.selected_binding_plan_path,
            "--endpoint-report",
            case.endpoint_binding_report_path,
            "--provider-plan",
            str(provider_plan),
            "--source-external-report",
            str(external_report),
            "--final-hypothesis-id",
            str(selection_report.selected_final_hypothesis_id),
            "--domain-profile",
            _argv_value(case.main_e2e_argv, "--domain-profile"),
            "--output-dir",
            case.relational_verifier_output_dir,
            "--model",
            case.critic_model,
            "--base-url",
            plan.settings.base_url,
            "--api-key-env",
            plan.settings.api_key_env,
            "--support-results-per-query",
            str(plan.settings.support_results_per_query),
            "--second-pass-results-per-query",
            str(plan.settings.second_pass_results_per_query),
            "--max-review-works-per-claim",
            str(plan.settings.max_review_works_per_claim),
        ]
        verifier = _run_stage(
            stage_name=case.case_id + ":relational_verifier",
            argv=verifier_argv,
            cwd=worktree_path,
            log_path=case_log_dir / "05_relational_verifier.log",
        )
        stages.append(verifier)

        if _stage_failed(verifier):
            results.append(
                _result_and_write(
                    result_path=result_path,
                    case=case,
                    plan=plan,
                    stages=stages,
                    disposition="VERIFIER_STAGE_FAILED",
                    main_e2e_manifest_status=main_status,
                    **lineage_fields,
                    **endpoint_fields,
                )
            )
            continue

        verifier_manifest_path = (
            Path(case.relational_verifier_output_dir)
            / "relational_scientific_verifier.run_manifest.json"
        )
        verifier_manifest = (
            RelationalScientificVerifierRunManifest.model_validate_json(
                verifier_manifest_path.read_text(encoding="utf-8")
            )
        )
        results.append(
            _result_and_write(
                result_path=result_path,
                case=case,
                plan=plan,
                stages=stages,
                disposition="VERIFIER_COMPLETE",
                main_e2e_manifest_status=main_status,
                **lineage_fields,
                **endpoint_fields,
                verifier_manifest_id=verifier_manifest.manifest_id,
                certification_decision=verifier_manifest.certification_decision,
                bounded_closure_state=verifier_manifest.bounded_closure_state,
                bounded_external_distinctness_state=(
                    verifier_manifest.bounded_external_distinctness_state
                ),
                positive_nonobviousness_authority_state=(
                    verifier_manifest.positive_nonobviousness_authority_state
                ),
                fatal_blocker_state=verifier_manifest.fatal_blocker_state,
            )
        )

    campaign_result = build_campaign_result(
        launch=launch,
        execution_plan_id=plan.plan_id,
        scientific_repository_head_sha=(
            plan.execution_plan_repository_head_sha
        ),
        case_results=results,
    )
    write_json_exclusive(final_path, campaign_result)

    print()
    print("=" * 100)
    print("PROSPECTIVE RELATIONAL CAMPAIGN COMPLETE")
    print("=" * 100)
    print("Scientific HEAD:", campaign_result.scientific_repository_head_sha)
    print("Dispositions:", campaign_result.case_dispositions)
    print("Counts:", campaign_result.disposition_counts)
    print("Case replacement performed: false")
    print("Post-case adaptation performed: false")
    print("Final result:", final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
