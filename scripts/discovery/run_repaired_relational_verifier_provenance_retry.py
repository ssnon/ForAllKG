from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.prospective_relational_campaign import (
    ProspectiveRelationalCampaignLaunch,
)
from pipeline_core.discovery.prospective_relational_execution_plan import (
    ProspectiveRelationalExecutionPlan,
    ProspectiveRelationalHypothesisSelection,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    RelationalAtomicEndpointBindingReport,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    RelationalScientificVerifierRunManifest,
    write_json_exclusive,
)
from pipeline_core.discovery.repaired_relational_verifier_campaign import (
    RepairedVerifierCaseResult,
    build_repaired_verifier_campaign_report,
)
from pipeline_core.discovery.repaired_verifier_provenance_rebind import (
    build_provenance_rebound_inputs,
)


_FAILURE_SIGNATURE = (
    "canonical N10 source claim changed after binding-plan freeze"
)


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
    index = argv.index(flag)
    return argv[index + 1]


def _run_logged(
    *,
    argv: list[str],
    cwd: Path,
    log_path: Path,
) -> int:
    if log_path.exists():
        raise ValueError("retry log already exists: " + str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(cwd)
        if not current
        else str(cwd) + os.pathsep + current
    )

    command = list(argv)
    if command and command[0] == "python":
        command[0] = sys.executable

    print("argv:", " ".join(command))
    with log_path.open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover repaired-verifier runs that failed solely because the "
            "R1 binding-plan claim fingerprint was not rebound to an actual "
            "repaired NoveltyClaim query-plan sidecar. Existing failed runs "
            "are preserved; endpoint anchors are deterministically rebound "
            "without another LLM call."
        )
    )
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--prior-verifier-report", required=True, type=Path)
    parser.add_argument("--reentry-report", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--campaign-launch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    campaign_root = args.campaign_root.expanduser().resolve()
    prior_path = args.prior_verifier_report.expanduser().resolve()
    reentry_path = args.reentry_report.expanduser().resolve()
    execution_path = args.execution_plan.expanduser().resolve()
    launch_path = args.campaign_launch.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists():
        raise ValueError("provenance-retry report is write-once")

    prior = _load_json(prior_path)
    reentry = _load_json(reentry_path)
    execution = ProspectiveRelationalExecutionPlan.model_validate_json(
        execution_path.read_text(encoding="utf-8")
    )
    launch = ProspectiveRelationalCampaignLaunch.model_validate_json(
        launch_path.read_text(encoding="utf-8")
    )

    frozen_root = Path(launch.pinned_worktree_path).expanduser().resolve()
    frozen_head = execution.execution_plan_repository_head_sha
    if _git_head(frozen_root) != frozen_head:
        raise ValueError("frozen scientific worktree HEAD mismatch")
    if _tracked_dirty(frozen_root):
        raise ValueError("frozen scientific worktree is dirty")

    prior_by_case = {
        str(row["case_id"]): row
        for row in prior["cases"]
    }
    reentry_by_case = {
        str(row["case_id"]): row
        for row in reentry["cases"]
    }
    execution_by_case = {
        row.case_id: row
        for row in execution.cases
    }

    results: list[RepairedVerifierCaseResult] = []

    for case_id in ["P06", "P07", "P08", "P09", "P10"]:
        prior_row = prior_by_case[case_id]
        reentry_row = reentry_by_case[case_id]

        print()
        print("=" * 100)
        print(case_id, prior_row["status"])
        print("=" * 100)

        if prior_row["status"] != "VERIFIER_STAGE_FAILED_AFTER_R1":
            results.append(
                RepairedVerifierCaseResult(
                    case_id=case_id,
                    status="NOT_VERIFIER_READY_AFTER_R1",
                    reentry_status=str(reentry_row["status"]),
                    final_hypothesis_id=reentry_row.get(
                        "selected_final_hypothesis_id"
                    ),
                    frozen_scientific_head_sha=frozen_head,
                )
            )
            print("No provenance retry")
            continue

        prior_log = Path(str(prior_row["verifier_log_path"]))
        if not prior_log.is_file():
            raise ValueError(case_id + ": prior verifier log missing")
        if _FAILURE_SIGNATURE not in prior_log.read_text(
            encoding="utf-8", errors="replace"
        ):
            raise ValueError(
                case_id
                + ": prior verifier failure was not the known source-claim "
                "fingerprint mismatch; automatic provenance retry forbidden"
            )

        r1_dir = campaign_root / "specification_repair_r1" / case_id
        source_plan_path = (
            r1_dir / "relational_atomic_binding_plan.selected.r1.json"
        )
        source_endpoint_path = (
            r1_dir / "relational_atomic_endpoint_binding.r1.json"
        )
        selection_path = r1_dir / "structural_selection.r1.json"

        source_plan = RelationalAtomicBindingPlan.model_validate_json(
            source_plan_path.read_text(encoding="utf-8")
        )
        source_endpoint = (
            RelationalAtomicEndpointBindingReport.model_validate_json(
                source_endpoint_path.read_text(encoding="utf-8")
            )
        )
        selection = (
            ProspectiveRelationalHypothesisSelection.model_validate_json(
                selection_path.read_text(encoding="utf-8")
            )
        )

        rebind_dir = r1_dir / "verifier_provenance_rebind"
        (
            rebound_plan,
            rebound_endpoint,
            _manifest,
            rebound_plan_path,
            rebound_endpoint_path,
        ) = build_provenance_rebound_inputs(
            case_id=case_id,
            source_binding_plan=source_plan,
            source_endpoint_report=source_endpoint,
            sidecar_directory=rebind_dir,
            write_json_exclusive=write_json_exclusive,
        )

        if selection.selected_final_hypothesis_id != (
            reentry_row["selected_final_hypothesis_id"]
        ):
            raise ValueError(case_id + ": selection/re-entry final mismatch")

        original_case_root = campaign_root / case_id
        provider_plan = original_case_root / "literature_provider_plan.json"
        external_report = Path(
            str(selection.selected_external_report)
        ).expanduser().resolve()

        retry_output = (
            r1_dir
            / "relational_scientific_verifier_shadow.r1_provenance_rebound"
        )
        retry_log = (
            r1_dir / "logs" / "04_relational_verifier_provenance_retry.log"
        )
        if retry_output.exists():
            raise ValueError(
                case_id + ": provenance-retry verifier output already exists"
            )

        execution_case = execution_by_case[case_id]
        argv = [
            "python",
            "-m",
            "scripts.discovery.run_relational_scientific_verifier_shadow_e2e",
            "--plan",
            str(rebound_plan_path),
            "--endpoint-report",
            str(rebound_endpoint_path),
            "--provider-plan",
            str(provider_plan),
            "--source-external-report",
            str(external_report),
            "--final-hypothesis-id",
            str(selection.selected_final_hypothesis_id),
            "--domain-profile",
            _argv_value(execution_case.main_e2e_argv, "--domain-profile"),
            "--output-dir",
            str(retry_output),
            "--model",
            execution_case.critic_model,
            "--base-url",
            execution.settings.base_url,
            "--api-key-env",
            execution.settings.api_key_env,
            "--support-results-per-query",
            str(execution.settings.support_results_per_query),
            "--second-pass-results-per-query",
            str(execution.settings.second_pass_results_per_query),
            "--max-review-works-per-claim",
            str(execution.settings.max_review_works_per_claim),
        ]

        rc = _run_logged(
            argv=argv,
            cwd=frozen_root,
            log_path=retry_log,
        )
        if rc != 0:
            results.append(
                RepairedVerifierCaseResult(
                    case_id=case_id,
                    status="VERIFIER_STAGE_FAILED_AFTER_R1",
                    reentry_status=str(reentry_row["status"]),
                    final_hypothesis_id=(
                        selection.selected_final_hypothesis_id
                    ),
                    frozen_scientific_head_sha=frozen_head,
                    verifier_output_dir=str(retry_output),
                    verifier_log_path=str(retry_log),
                )
            )
            print("Verifier provenance retry failed")
            continue

        manifest_path = (
            retry_output
            / "relational_scientific_verifier.run_manifest.json"
        )
        manifest = RelationalScientificVerifierRunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.repository_head_sha != frozen_head:
            raise ValueError(case_id + ": verifier scientific HEAD mismatch")
        if manifest.repository_worktree_dirty:
            raise ValueError(case_id + ": verifier worktree unexpectedly dirty")
        if manifest.final_hypothesis_id != (
            selection.selected_final_hypothesis_id
        ):
            raise ValueError(case_id + ": verifier final hypothesis mismatch")

        results.append(
            RepairedVerifierCaseResult(
                case_id=case_id,
                status="VERIFIER_COMPLETE_AFTER_R1",
                reentry_status=str(reentry_row["status"]),
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
                verifier_output_dir=str(retry_output),
                verifier_log_path=str(retry_log),
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
        source_execution_plan_id=execution.plan_id,
        source_campaign_launch_id=launch.launch_id,
        frozen_scientific_head_sha=frozen_head,
        cases=results,
    )
    write_json_exclusive(output_path, report)

    print()
    print("Repaired verifier provenance retry complete")
    print("Statuses:", report.status_counts)
    print("Verifier attempted:", report.verifier_attempted_case_ids)
    print("Verifier complete:", report.verifier_complete_case_ids)
    print("Certification decisions:", report.certification_decision_counts)
    print("Endpoint LLM recalled: false")
    print("Specification repair recalled: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
