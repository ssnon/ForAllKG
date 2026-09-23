from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.repaired_relational_verifier_campaign import (
    RepairedVerifierCaseResult,
    build_repaired_verifier_campaign_report,
)
from pipeline_core.discovery.prospective_relational_campaign import (
    ProspectiveRelationalCampaignLaunch,
)
from pipeline_core.discovery.prospective_relational_execution_plan import (
    ProspectiveRelationalExecutionPlan,
    ProspectiveRelationalHypothesisSelection,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    RelationalScientificVerifierRunManifest,
    write_json_exclusive,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(cwd: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_dirty(cwd: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _argv_value(argv: list[str], flag: str) -> str:
    try:
        index = argv.index(flag)
    except ValueError as exc:
        raise ValueError("frozen argv missing " + flag) from exc
    if index + 1 >= len(argv):
        raise ValueError("frozen argv missing value for " + flag)
    return argv[index + 1]


def _load_reentry(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected re-entry JSON object")
    if (
        value.get("schema_version")
        != "preverifier-specification-reentry-report-v1"
    ):
        raise ValueError("unexpected re-entry report schema")

    body = dict(value)
    observed_id = str(body.pop("report_id", ""))
    observed_sha = str(body.pop("report_sha256", ""))
    expected_sha = _sha256_json(body)
    if observed_sha != expected_sha:
        raise ValueError("source re-entry report SHA mismatch")
    if observed_id != (
        "preverifier_specification_reentry_report:"
        + expected_sha[:20]
    ):
        raise ValueError("source re-entry report ID mismatch")
    return value


def _run_logged(
    *,
    argv: list[str],
    cwd: Path,
    log_path: Path,
) -> int:
    if log_path.exists():
        raise ValueError("verifier log already exists: " + str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(cwd)
        if not current_pythonpath
        else str(cwd) + os.pathsep + current_pythonpath
    )
    command = list(argv)
    if command and command[0] == "python":
        command[0] = sys.executable

    print("argv:", " ".join(command))
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unchanged frozen relational scientific verifier only for "
            "P06-P10 cases that became verifier-ready after accepted R1 "
            "zero-scientific-delta specification repair."
        )
    )
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--reentry-report", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--campaign-launch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    campaign_root = args.campaign_root.expanduser().resolve()
    reentry_path = args.reentry_report.expanduser().resolve()
    execution_plan_path = args.execution_plan.expanduser().resolve()
    launch_path = args.campaign_launch.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists():
        raise ValueError(
            "repaired verifier campaign report is write-once; "
            "use a fresh output path"
        )

    reentry = _load_reentry(reentry_path)
    execution_plan = ProspectiveRelationalExecutionPlan.model_validate_json(
        execution_plan_path.read_text(encoding="utf-8")
    )
    launch = ProspectiveRelationalCampaignLaunch.model_validate_json(
        launch_path.read_text(encoding="utf-8")
    )

    if reentry["source_execution_plan_id"] != execution_plan.plan_id:
        raise ValueError("re-entry/execution-plan ID mismatch")
    if reentry["source_campaign_launch_id"] != launch.launch_id:
        raise ValueError("re-entry/campaign-launch ID mismatch")
    if launch.execution_plan_id != execution_plan.plan_id:
        raise ValueError("launch/execution-plan ID mismatch")
    if (
        reentry["frozen_scientific_head_sha"]
        != execution_plan.execution_plan_repository_head_sha
    ):
        raise ValueError("re-entry scientific HEAD mismatch")

    frozen_root = Path(launch.pinned_worktree_path).expanduser().resolve()
    frozen_head = execution_plan.execution_plan_repository_head_sha
    if not frozen_root.is_dir():
        raise ValueError("frozen scientific worktree is missing")
    if _git_head(frozen_root) != frozen_head:
        raise ValueError("frozen scientific worktree HEAD mismatch")
    if _tracked_dirty(frozen_root):
        raise ValueError("frozen scientific worktree is dirty")

    reentry_by_case = {
        str(row["case_id"]): row
        for row in reentry["cases"]
    }
    ready_ids = list(reentry["verifier_ready_case_ids"])
    derived_ready = [
        case_id
        for case_id in ["P06", "P07", "P08", "P09", "P10"]
        if reentry_by_case[case_id]["status"]
        == "VERIFIER_READY_AFTER_R1"
    ]
    if ready_ids != derived_ready:
        raise ValueError("re-entry verifier-ready population mismatch")

    execution_by_case = {
        row.case_id: row
        for row in execution_plan.cases
    }
    results: list[RepairedVerifierCaseResult] = []

    for case_id in ["P06", "P07", "P08", "P09", "P10"]:
        row = reentry_by_case[case_id]
        print()
        print("=" * 100)
        print(case_id, row["status"])
        print("=" * 100)

        if row["status"] != "VERIFIER_READY_AFTER_R1":
            results.append(
                RepairedVerifierCaseResult(
                    case_id=case_id,
                    status="NOT_VERIFIER_READY_AFTER_R1",
                    reentry_status=str(row["status"]),
                    final_hypothesis_id=row.get(
                        "selected_final_hypothesis_id"
                    ),
                    frozen_scientific_head_sha=frozen_head,
                )
            )
            print("Verifier not executed")
            continue

        r1_dir = (
            campaign_root
            / "specification_repair_r1"
            / case_id
        )
        selected_plan_path = (
            r1_dir / "relational_atomic_binding_plan.selected.r1.json"
        )
        endpoint_path = (
            r1_dir / "relational_atomic_endpoint_binding.r1.json"
        )
        selection_path = r1_dir / "structural_selection.r1.json"
        for required in (
            selected_plan_path,
            endpoint_path,
            selection_path,
        ):
            if not required.is_file():
                raise ValueError(
                    case_id
                    + ": missing frozen R1 verifier input "
                    + str(required)
                )

        selection = (
            ProspectiveRelationalHypothesisSelection.model_validate_json(
                selection_path.read_text(encoding="utf-8")
            )
        )
        if (
            selection.selected_final_hypothesis_id
            != row["selected_final_hypothesis_id"]
        ):
            raise ValueError(
                case_id + ": re-entry/selection final hypothesis mismatch"
            )
        if (
            selection.selected_binding_plan_id
            != row["selected_binding_plan_id"]
        ):
            raise ValueError(
                case_id + ": re-entry/selection plan ID mismatch"
            )
        if (
            selection.selected_binding_plan_sha256
            != row["selected_binding_plan_sha256"]
        ):
            raise ValueError(
                case_id + ": re-entry/selection plan SHA mismatch"
            )

        original_case_root = campaign_root / case_id
        provider_plan = original_case_root / "literature_provider_plan.json"
        external_report = Path(
            str(selection.selected_external_report)
        ).expanduser().resolve()
        if not provider_plan.is_file():
            raise ValueError(
                case_id + ": original provider plan missing"
            )
        if not external_report.is_file():
            raise ValueError(
                case_id + ": selected external novelty report missing"
            )

        execution_case = execution_by_case[case_id]
        verifier_output = (
            r1_dir / "relational_scientific_verifier_shadow.r1"
        )
        verifier_log = (
            r1_dir / "logs" / "03_relational_verifier.log"
        )
        if verifier_output.exists():
            raise ValueError(
                case_id
                + ": repaired verifier output already exists; "
                "do not overwrite immutable R1 verifier result"
            )

        argv = [
            "python",
            "-m",
            "scripts.discovery.run_relational_scientific_verifier_shadow_e2e",
            "--plan",
            str(selected_plan_path),
            "--endpoint-report",
            str(endpoint_path),
            "--provider-plan",
            str(provider_plan),
            "--source-external-report",
            str(external_report),
            "--final-hypothesis-id",
            str(selection.selected_final_hypothesis_id),
            "--domain-profile",
            _argv_value(execution_case.main_e2e_argv, "--domain-profile"),
            "--output-dir",
            str(verifier_output),
            "--model",
            execution_case.critic_model,
            "--base-url",
            execution_plan.settings.base_url,
            "--api-key-env",
            execution_plan.settings.api_key_env,
            "--support-results-per-query",
            str(execution_plan.settings.support_results_per_query),
            "--second-pass-results-per-query",
            str(execution_plan.settings.second_pass_results_per_query),
            "--max-review-works-per-claim",
            str(execution_plan.settings.max_review_works_per_claim),
        ]
        return_code = _run_logged(
            argv=argv,
            cwd=frozen_root,
            log_path=verifier_log,
        )
        if return_code != 0:
            results.append(
                RepairedVerifierCaseResult(
                    case_id=case_id,
                    status="VERIFIER_STAGE_FAILED_AFTER_R1",
                    reentry_status=str(row["status"]),
                    final_hypothesis_id=(
                        selection.selected_final_hypothesis_id
                    ),
                    frozen_scientific_head_sha=frozen_head,
                    verifier_output_dir=str(verifier_output),
                    verifier_log_path=str(verifier_log),
                )
            )
            print("Verifier stage failed")
            continue

        manifest_path = (
            verifier_output
            / "relational_scientific_verifier.run_manifest.json"
        )
        if not manifest_path.is_file():
            raise ValueError(
                case_id + ": verifier completed without run manifest"
            )
        manifest = RelationalScientificVerifierRunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.repository_head_sha != frozen_head:
            raise ValueError(
                case_id + ": verifier manifest scientific HEAD mismatch"
            )
        if manifest.repository_worktree_dirty:
            raise ValueError(
                case_id + ": verifier ran from dirty frozen worktree"
            )
        if (
            manifest.final_hypothesis_id
            != selection.selected_final_hypothesis_id
        ):
            raise ValueError(
                case_id + ": verifier manifest final hypothesis mismatch"
            )

        results.append(
            RepairedVerifierCaseResult(
                case_id=case_id,
                status="VERIFIER_COMPLETE_AFTER_R1",
                reentry_status=str(row["status"]),
                final_hypothesis_id=manifest.final_hypothesis_id,
                candidate_hypothesis_id=manifest.candidate_hypothesis_id,
                verifier_manifest_id=manifest.manifest_id,
                verifier_manifest_sha256=manifest.manifest_sha256,
                certification_report_id=manifest.certification_report_id,
                certification_decision=manifest.certification_decision,
                bounded_closure_state=manifest.bounded_closure_state,
                bounded_external_distinctness_state=(
                    manifest.bounded_external_distinctness_state
                ),
                positive_nonobviousness_authority_state=(
                    manifest.positive_nonobviousness_authority_state
                ),
                fatal_blocker_state=manifest.fatal_blocker_state,
                frozen_scientific_head_sha=frozen_head,
                verifier_output_dir=str(verifier_output),
                verifier_log_path=str(verifier_log),
            )
        )
        print(
            "Verifier complete:",
            manifest.certification_decision,
            "| C=", manifest.bounded_closure_state,
            "| D=", manifest.bounded_external_distinctness_state,
            "| N=", manifest.positive_nonobviousness_authority_state,
            "| X=", manifest.fatal_blocker_state,
        )

    report = build_repaired_verifier_campaign_report(
        source_reentry_report_id=str(reentry["report_id"]),
        source_reentry_report_sha256=str(reentry["report_sha256"]),
        source_reentry_report_file_sha256=_sha256_file(reentry_path),
        source_execution_plan_id=execution_plan.plan_id,
        source_campaign_launch_id=launch.launch_id,
        frozen_scientific_head_sha=frozen_head,
        cases=results,
    )
    write_json_exclusive(output_path, report)

    print()
    print("Repaired relational verifier campaign complete")
    print("Frozen scientific HEAD:", frozen_head)
    print("Statuses:", report.status_counts)
    print("Verifier attempted:", report.verifier_attempted_case_ids)
    print("Verifier complete:", report.verifier_complete_case_ids)
    print("Certification decisions:", report.certification_decision_counts)
    print("Second repair attempts: false")
    print("Production selection changed: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
