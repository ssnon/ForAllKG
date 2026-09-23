from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.preverifier_specification_reentry import (
    RepairedBindingPlanMaterialization,
    materialize_repaired_binding_plan,
)
from pipeline_core.discovery.preverifier_specification_repair_executor import (
    SpecificationRepairExecutionReport,
)
from pipeline_core.discovery.preverifier_specification_repair_policy import (
    SpecificationRepairCampaignPlan,
)
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
    write_json_exclusive,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
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


ReentryStatus = Literal[
    "EXCLUDED_NO_AUTOMATIC_REPAIR",
    "NO_MATERIALIZED_R1_CLAIMS",
    "NO_BINDING_READY_HYPOTHESIS_AFTER_R1",
    "ENDPOINT_BINDING_STAGE_FAILED_AFTER_R1",
    "ENDPOINT_BINDING_ABSTAINED_AFTER_R1",
    "VERIFIER_READY_AFTER_R1",
]


class SpecificationReentryCaseResult(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    status: ReentryStatus
    source_original_disposition: str
    materialized_repair_count: int = Field(ge=0)

    repaired_plan_id: str | None = None
    repaired_plan_sha256: str | None = None
    selected_final_hypothesis_id: str | None = None
    selected_binding_plan_id: str | None = None
    selected_binding_plan_sha256: str | None = None

    endpoint_report_id: str | None = None
    endpoint_selected_claim_count: int | None = Field(default=None, ge=0)
    endpoint_bound_claim_count: int | None = Field(default=None, ge=0)
    endpoint_abstained_claim_count: int | None = Field(default=None, ge=0)
    endpoint_novelty_bearing_bound_claim_count: int | None = Field(
        default=None,
        ge=0,
    )

    frozen_scientific_head_sha: str
    second_repair_attempt_performed: Literal[False] = False
    original_prospective_result_preserved: Literal[True] = True
    verifier_executed: Literal[False] = False
    literature_retrieval_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False


class SpecificationReentryReport(StrictModel):
    schema_version: Literal[
        "preverifier-specification-reentry-report-v1"
    ] = "preverifier-specification-reentry-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_repair_plan_id: str
    source_repair_execution_report_id: str
    source_execution_plan_id: str
    source_campaign_launch_id: str
    frozen_scientific_head_sha: str

    cases: list[SpecificationReentryCaseResult]
    case_ids: list[str]
    case_count: Literal[5] = 5
    status_counts: dict[str, int]
    verifier_ready_case_ids: list[str]

    structural_selection_executed_with_frozen_scientific_code: Literal[
        True
    ] = True
    endpoint_binding_executed_with_frozen_scientific_code: Literal[
        True
    ] = True
    second_repair_attempt_performed: Literal[False] = False
    original_prospective_results_preserved: Literal[True] = True
    verifier_executed: Literal[False] = False
    literature_retrieval_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "SpecificationReentryReport":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        if self.case_ids != expected:
            raise ValueError("re-entry case IDs must be P06-P10")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("re-entry cases must be ordered P06-P10")
        counts = Counter(row.status for row in self.cases)
        if dict(sorted(counts.items())) != dict(
            sorted(self.status_counts.items())
        ):
            raise ValueError("re-entry status_counts mismatch")
        ready = [
            row.case_id
            for row in self.cases
            if row.status == "VERIFIER_READY_AFTER_R1"
        ]
        if ready != self.verifier_ready_case_ids:
            raise ValueError("verifier_ready_case_ids mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("specification re-entry report SHA mismatch")
        if observed_id != (
            "preverifier_specification_reentry_report:"
            + expected_sha[:20]
        ):
            raise ValueError("specification re-entry report ID mismatch")
        return self


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


def _run_logged(
    *,
    argv: list[str],
    cwd: Path,
    log_path: Path,
) -> int:
    if log_path.exists():
        raise ValueError("re-entry stage log already exists: " + str(log_path))
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


def _write_model_exclusive(path: Path, model: BaseModel) -> None:
    write_json_exclusive(path, model)


def _source_plan_path(
    *,
    campaign_root: Path,
    case_plan,
) -> Path:
    case_root = campaign_root / case_plan.case_id
    if case_plan.source_original_disposition == "ENDPOINT_BINDING_ABSTAINED":
        return case_root / "relational_atomic_binding_plan.selected.json"
    if case_plan.source_original_disposition == "NO_BINDING_READY_HYPOTHESIS":
        return case_root / "relational_atomic_binding_plan.full.json"
    raise ValueError(
        case_plan.case_id
        + ": automatic repair case has unsupported original disposition "
        + case_plan.source_original_disposition
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize accepted R1 repairs into new binding-plan lineage, "
            "then re-run frozen structural selection and strict literal "
            "endpoint binding using the original prospective scientific HEAD."
        )
    )
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--repair-plan", required=True, type=Path)
    parser.add_argument("--repair-execution", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--campaign-launch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    campaign_root = args.campaign_root.expanduser().resolve()
    repair_plan_path = args.repair_plan.expanduser().resolve()
    repair_execution_path = args.repair_execution.expanduser().resolve()
    execution_plan_path = args.execution_plan.expanduser().resolve()
    launch_path = args.campaign_launch.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists():
        raise ValueError(
            "specification re-entry report is write-once; "
            "use a fresh output path"
        )

    repair_plan = SpecificationRepairCampaignPlan.model_validate_json(
        repair_plan_path.read_text(encoding="utf-8")
    )
    repair_execution = SpecificationRepairExecutionReport.model_validate_json(
        repair_execution_path.read_text(encoding="utf-8")
    )
    execution_plan = ProspectiveRelationalExecutionPlan.model_validate_json(
        execution_plan_path.read_text(encoding="utf-8")
    )
    launch = ProspectiveRelationalCampaignLaunch.model_validate_json(
        launch_path.read_text(encoding="utf-8")
    )

    if repair_execution.source_repair_plan_id != repair_plan.plan_id:
        raise ValueError("repair execution/plan ID mismatch")
    if repair_execution.source_repair_plan_sha256 != repair_plan.plan_sha256:
        raise ValueError("repair execution/plan SHA mismatch")
    if repair_execution.source_repair_plan_file_sha256 != _sha256_file(
        repair_plan_path
    ):
        raise ValueError("repair-plan file SHA mismatch")
    if launch.execution_plan_id != execution_plan.plan_id:
        raise ValueError("campaign launch/execution-plan ID mismatch")
    if (
        launch.scientific_repository_head_sha
        != execution_plan.execution_plan_repository_head_sha
    ):
        raise ValueError("campaign launch scientific HEAD mismatch")

    frozen_root = Path(launch.pinned_worktree_path).expanduser().resolve()
    frozen_head = execution_plan.execution_plan_repository_head_sha
    if not frozen_root.is_dir():
        raise ValueError("frozen scientific worktree is missing")
    if _git_head(frozen_root) != frozen_head:
        raise ValueError("frozen scientific worktree HEAD mismatch")
    if _tracked_dirty(frozen_root):
        raise ValueError("frozen scientific worktree is dirty")

    execution_by_case = {
        row.case_id: row
        for row in repair_execution.cases
    }
    prospective_by_case = {
        row.case_id: row
        for row in execution_plan.cases
    }

    results: list[SpecificationReentryCaseResult] = []

    for case_plan in repair_plan.cases:
        case_id = case_plan.case_id
        print()
        print("=" * 100)
        print(case_id)
        print("=" * 100)

        case_execution = execution_by_case[case_id]
        if case_plan.status != "PLANNED_AUTOMATIC_REPAIR":
            results.append(
                SpecificationReentryCaseResult(
                    case_id=case_id,
                    status="EXCLUDED_NO_AUTOMATIC_REPAIR",
                    source_original_disposition=(
                        case_plan.source_original_disposition
                    ),
                    materialized_repair_count=0,
                    frozen_scientific_head_sha=frozen_head,
                )
            )
            print("Excluded: no automatic repair")
            continue

        if case_execution.materialized_claim_count < 1:
            results.append(
                SpecificationReentryCaseResult(
                    case_id=case_id,
                    status="NO_MATERIALIZED_R1_CLAIMS",
                    source_original_disposition=(
                        case_plan.source_original_disposition
                    ),
                    materialized_repair_count=0,
                    frozen_scientific_head_sha=frozen_head,
                )
            )
            print("No materialized R1 claims")
            continue

        source_plan_path = _source_plan_path(
            campaign_root=campaign_root,
            case_plan=case_plan,
        )
        source_plan = RelationalAtomicBindingPlan.model_validate_json(
            source_plan_path.read_text(encoding="utf-8")
        )
        repaired_plan, materialization = materialize_repaired_binding_plan(
            source_plan=source_plan,
            case_plan=case_plan,
            case_result=case_execution,
            repair_execution_report_id=repair_execution.report_id,
        )

        r1_dir = Path(case_plan.repair_output_dir).expanduser().resolve()
        r1_dir.mkdir(parents=True, exist_ok=True)
        repaired_plan_path = r1_dir / "relational_atomic_binding_plan.r1.json"
        materialization_path = r1_dir / "repair_materialization.json"
        selection_path = r1_dir / "structural_selection.r1.json"
        selected_plan_path = r1_dir / "relational_atomic_binding_plan.selected.r1.json"
        endpoint_path = r1_dir / "relational_atomic_endpoint_binding.r1.json"
        endpoint_prompt_path = r1_dir / "relational_atomic_endpoint_binding.r1.prompt.txt"
        endpoint_telemetry_path = r1_dir / "relational_atomic_endpoint_binding.r1.telemetry.jsonl"
        logs_dir = r1_dir / "logs"

        _write_model_exclusive(repaired_plan_path, repaired_plan)
        _write_model_exclusive(materialization_path, materialization)

        selector_rc = _run_logged(
            argv=[
                "python",
                "-m",
                "scripts.discovery.select_prospective_relational_binding_plan",
                "--plan",
                str(repaired_plan_path),
                "--selection-output",
                str(selection_path),
                "--selected-plan-output",
                str(selected_plan_path),
            ],
            cwd=frozen_root,
            log_path=logs_dir / "01_structural_selection.log",
        )
        if selector_rc != 0:
            raise RuntimeError(
                case_id + ": frozen structural selector failed"
            )

        selection = ProspectiveRelationalHypothesisSelection.model_validate_json(
            selection_path.read_text(encoding="utf-8")
        )
        if selection.status == "NO_BINDING_READY_HYPOTHESIS":
            results.append(
                SpecificationReentryCaseResult(
                    case_id=case_id,
                    status="NO_BINDING_READY_HYPOTHESIS_AFTER_R1",
                    source_original_disposition=(
                        case_plan.source_original_disposition
                    ),
                    materialized_repair_count=(
                        case_execution.materialized_claim_count
                    ),
                    repaired_plan_id=repaired_plan.plan_id,
                    repaired_plan_sha256=repaired_plan.plan_sha256,
                    frozen_scientific_head_sha=frozen_head,
                )
            )
            print("Re-entry: no binding-ready hypothesis")
            continue

        prospective_case = prospective_by_case[case_id]
        endpoint_rc = _run_logged(
            argv=[
                "python",
                "-m",
                "scripts.discovery.run_relational_atomic_endpoint_binding",
                "--plan",
                str(selected_plan_path),
                "--output",
                str(endpoint_path),
                "--prompt-output",
                str(endpoint_prompt_path),
                "--model",
                prospective_case.critic_model,
                "--base-url",
                execution_plan.settings.base_url,
                "--api-key-env",
                execution_plan.settings.api_key_env,
                "--temperature",
                str(execution_plan.settings.endpoint_temperature),
                "--parse-retries",
                str(execution_plan.settings.endpoint_parse_retries),
                "--timeout",
                str(execution_plan.settings.endpoint_timeout_seconds),
                "--telemetry",
                str(endpoint_telemetry_path),
            ],
            cwd=frozen_root,
            log_path=logs_dir / "02_endpoint_binding.log",
        )
        if endpoint_rc != 0:
            results.append(
                SpecificationReentryCaseResult(
                    case_id=case_id,
                    status="ENDPOINT_BINDING_STAGE_FAILED_AFTER_R1",
                    source_original_disposition=(
                        case_plan.source_original_disposition
                    ),
                    materialized_repair_count=(
                        case_execution.materialized_claim_count
                    ),
                    repaired_plan_id=repaired_plan.plan_id,
                    repaired_plan_sha256=repaired_plan.plan_sha256,
                    selected_final_hypothesis_id=(
                        selection.selected_final_hypothesis_id
                    ),
                    selected_binding_plan_id=selection.selected_binding_plan_id,
                    selected_binding_plan_sha256=(
                        selection.selected_binding_plan_sha256
                    ),
                    frozen_scientific_head_sha=frozen_head,
                )
            )
            print("Re-entry: endpoint binding stage failed")
            continue

        endpoint = RelationalAtomicEndpointBindingReport.model_validate_json(
            endpoint_path.read_text(encoding="utf-8")
        )
        status: ReentryStatus = (
            "VERIFIER_READY_AFTER_R1"
            if endpoint.novelty_bearing_bound_claim_count >= 1
            else "ENDPOINT_BINDING_ABSTAINED_AFTER_R1"
        )
        results.append(
            SpecificationReentryCaseResult(
                case_id=case_id,
                status=status,
                source_original_disposition=(
                    case_plan.source_original_disposition
                ),
                materialized_repair_count=(
                    case_execution.materialized_claim_count
                ),
                repaired_plan_id=repaired_plan.plan_id,
                repaired_plan_sha256=repaired_plan.plan_sha256,
                selected_final_hypothesis_id=(
                    selection.selected_final_hypothesis_id
                ),
                selected_binding_plan_id=selection.selected_binding_plan_id,
                selected_binding_plan_sha256=(
                    selection.selected_binding_plan_sha256
                ),
                endpoint_report_id=endpoint.report_id,
                endpoint_selected_claim_count=endpoint.selected_claim_count,
                endpoint_bound_claim_count=endpoint.bound_claim_count,
                endpoint_abstained_claim_count=endpoint.abstained_claim_count,
                endpoint_novelty_bearing_bound_claim_count=(
                    endpoint.novelty_bearing_bound_claim_count
                ),
                frozen_scientific_head_sha=frozen_head,
            )
        )
        print(
            "Re-entry:",
            status,
            "| bound=",
            endpoint.bound_claim_count,
            "| novelty-bearing bound=",
            endpoint.novelty_bearing_bound_claim_count,
        )

    counts = Counter(row.status for row in results)
    verifier_ready = [
        row.case_id
        for row in results
        if row.status == "VERIFIER_READY_AFTER_R1"
    ]
    body = {
        "schema_version": "preverifier-specification-reentry-report-v1",
        "source_repair_plan_id": repair_plan.plan_id,
        "source_repair_execution_report_id": repair_execution.report_id,
        "source_execution_plan_id": execution_plan.plan_id,
        "source_campaign_launch_id": launch.launch_id,
        "frozen_scientific_head_sha": frozen_head,
        "cases": [row.model_dump(mode="json") for row in results],
        "case_ids": ["P06", "P07", "P08", "P09", "P10"],
        "case_count": 5,
        "status_counts": dict(sorted(counts.items())),
        "verifier_ready_case_ids": verifier_ready,
        "structural_selection_executed_with_frozen_scientific_code": True,
        "endpoint_binding_executed_with_frozen_scientific_code": True,
        "second_repair_attempt_performed": False,
        "original_prospective_results_preserved": True,
        "verifier_executed": False,
        "literature_retrieval_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = SpecificationReentryReport(
        **body,
        report_id=(
            "preverifier_specification_reentry_report:"
            + digest[:20]
        ),
        report_sha256=digest,
    )
    write_json_exclusive(output_path, report)

    print()
    print("Specification R1 strict re-entry complete")
    print("Frozen scientific HEAD:", frozen_head)
    print("Statuses:", report.status_counts)
    print("Verifier-ready cases:", report.verifier_ready_case_ids)
    print("Second repair attempts: false")
    print("Verifier executed: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
