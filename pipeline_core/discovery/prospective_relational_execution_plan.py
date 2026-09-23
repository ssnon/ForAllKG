from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.prospective_source_task_freeze import (
    ProspectiveSourceTaskCampaignFreeze,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
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


class ProspectiveRelationalExecutionSettings(StrictModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    provider_request: Literal["auto"] = "auto"
    results_per_query: int = Field(default=12, ge=1)

    endpoint_temperature: Literal[0.0] = 0.0
    endpoint_parse_retries: int = Field(default=3, ge=1)
    endpoint_timeout_seconds: float = Field(default=180.0, gt=0)

    support_results_per_query: int = Field(default=12, ge=1)
    second_pass_results_per_query: int = Field(default=16, ge=1)
    max_review_works_per_claim: int = Field(default=20, ge=1)

    post_generation_n10_authority_mode: Literal[
        "certification_only"
    ] = "certification_only"
    original_fallback_n10_enabled: Literal[True] = True
    post_generation_n10_enabled: Literal[True] = True
    bounded_continuation_enabled: Literal[False] = False

    hypothesis_selection_rule: Literal[
        "binding_ready_structural_order_v1"
    ] = "binding_ready_structural_order_v1"

    verifier_result_may_change_later_case_settings: Literal[False] = False
    failed_or_abstained_case_may_be_replaced: Literal[False] = False


class ProspectiveRelationalCaseExecutionPlan(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    source_task_id: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_dir: str

    generation_model: str
    critic_model: str

    main_e2e_argv: list[str]
    full_binding_plan_path: str
    selection_report_path: str
    selected_binding_plan_path: str
    endpoint_binding_report_path: str
    endpoint_binding_prompt_path: str
    endpoint_binding_telemetry_path: str
    relational_verifier_output_dir: str

    main_e2e_runs_before_binding_plan: Literal[True] = True
    structural_hypothesis_selection_precedes_endpoint_binding: Literal[True] = True
    endpoint_binding_precedes_verifier: Literal[True] = True
    old_n10_outcome_fields_used_for_hypothesis_selection: Literal[False] = False
    external_novelty_outcomes_used_for_hypothesis_selection: Literal[False] = False
    verifier_outcomes_used_for_hypothesis_selection: Literal[False] = False


class ProspectiveRelationalExecutionPlan(StrictModel):
    schema_version: Literal[
        "prospective-relational-execution-plan-v1"
    ] = "prospective-relational-execution-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_task_freeze_id: str
    source_task_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_task_freeze_repository_head_sha: str

    execution_plan_repository_head_sha: str
    repository_tracked_worktree_dirty: Literal[False] = False

    settings: ProspectiveRelationalExecutionSettings
    cases: list[ProspectiveRelationalCaseExecutionPlan]
    case_ids: list[str]
    case_count: Literal[5] = 5

    all_case_settings_frozen_before_first_case_execution: Literal[True] = True
    case_order_fixed_p06_to_p10: Literal[True] = True
    post_case_adaptation_allowed: Literal[False] = False
    case_replacement_allowed: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "ProspectiveRelationalExecutionPlan":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        if self.case_ids != expected:
            raise ValueError("execution plan case IDs must be exactly P06-P10")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("execution plan cases must be ordered P06-P10")
        if len(self.cases) != 5:
            raise ValueError("execution plan requires exactly five cases")
        if len({row.source_task_id for row in self.cases}) != 5:
            raise ValueError("execution plan source task IDs must be unique")

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective relational execution plan SHA mismatch")
        if observed_id != (
            "prospective_relational_execution_plan:"
            + expected_sha[:20]
        ):
            raise ValueError("prospective relational execution plan ID mismatch")
        return self


def build_prospective_relational_execution_plan(
    *,
    source_freeze: ProspectiveSourceTaskCampaignFreeze,
    source_freeze_sha256: str,
    execution_plan_repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
    settings: ProspectiveRelationalExecutionSettings | None = None,
) -> ProspectiveRelationalExecutionPlan:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "prospective relational execution plan requires a clean tracked worktree"
        )
    if source_freeze.case_ids != ["P06", "P07", "P08", "P09", "P10"]:
        raise ValueError("source task freeze is not the P06-P10 campaign")

    resolved_settings = settings or ProspectiveRelationalExecutionSettings()
    cases: list[ProspectiveRelationalCaseExecutionPlan] = []

    for task in source_freeze.tasks:
        run = Path(task.run_dir).expanduser().resolve()
        argv = [
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
            "--run-dir",
            str(run),
            "--source",
            task.source,
        ]
        if task.stop is not None:
            argv.extend(["--stop", task.stop])
        argv.extend(
            [
                "--target",
                task.target,
                "--question",
                task.question,
                "--objective",
                task.objective,
                "--domain-profile",
                task.domain_profile_id,
                "--corpus-id",
                task.corpus_id,
                "--data-root",
                task.data_root,
                "--context-review-mode",
                "auto",
                "--model",
                task.generation_model,
                "--critic-model",
                task.critic_model,
                "--base-url",
                resolved_settings.base_url,
                "--api-key-env",
                resolved_settings.api_key_env,
                "--providers",
                resolved_settings.provider_request,
                "--results-per-query",
                str(resolved_settings.results_per_query),
                "--nonobviousness-original-fallback-enforce",
                "--nonobviousness-post-generation-enforce",
                "--post-generation-n10-authority-mode",
                "certification_only",
            ]
        )

        cases.append(
            ProspectiveRelationalCaseExecutionPlan(
                case_id=task.case_id,
                source_task_id=task.task_id,
                source_task_sha256=task.task_sha256,
                run_dir=str(run),
                generation_model=task.generation_model,
                critic_model=task.critic_model,
                main_e2e_argv=argv,
                full_binding_plan_path=str(
                    run / "relational_atomic_binding_plan.full.json"
                ),
                selection_report_path=str(
                    run / "prospective_relational_hypothesis_selection.json"
                ),
                selected_binding_plan_path=str(
                    run / "relational_atomic_binding_plan.selected.json"
                ),
                endpoint_binding_report_path=str(
                    run / "relational_atomic_endpoint_binding.selected.json"
                ),
                endpoint_binding_prompt_path=str(
                    run / "relational_atomic_endpoint_binding.selected.prompt.txt"
                ),
                endpoint_binding_telemetry_path=str(
                    run / "relational_atomic_endpoint_binding.selected.telemetry.jsonl"
                ),
                relational_verifier_output_dir=str(
                    run / "relational_scientific_verifier_shadow"
                ),
            )
        )

    body = {
        "schema_version": "prospective-relational-execution-plan-v1",
        "source_task_freeze_id": source_freeze.freeze_id,
        "source_task_freeze_sha256": source_freeze_sha256,
        "source_task_freeze_repository_head_sha":
            source_freeze.repository_head_sha,
        "execution_plan_repository_head_sha":
            execution_plan_repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "settings": resolved_settings.model_dump(mode="json"),
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P06", "P07", "P08", "P09", "P10"],
        "case_count": 5,
        "all_case_settings_frozen_before_first_case_execution": True,
        "case_order_fixed_p06_to_p10": True,
        "post_case_adaptation_allowed": False,
        "case_replacement_allowed": False,
        "scientific_quality_ranking_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRelationalExecutionPlan(
        **body,
        plan_id="prospective_relational_execution_plan:" + digest[:20],
        plan_sha256=digest,
    )


SelectionStatus = Literal[
    "SELECTED_BINDING_READY_HYPOTHESIS",
    "NO_BINDING_READY_HYPOTHESIS",
]


class ProspectiveRelationalHypothesisSelection(StrictModel):
    schema_version: Literal[
        "prospective-relational-hypothesis-selection-v1"
    ] = "prospective-relational-hypothesis-selection-v1"

    selection_id: str
    selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    status: SelectionStatus
    selected_original_hypothesis_id: str | None = None
    selected_candidate_hypothesis_id: str | None = None
    selected_final_hypothesis_id: str | None = None
    selected_binding_plan_id: str | None = None
    selected_binding_plan_sha256: str | None = None
    selected_external_report: str | None = None
    selected_source_query_plan: str | None = None

    eligible_final_hypothesis_ids: list[str]
    structural_order_keys: dict[str, list[object]]

    selection_rule: Literal[
        "binding_ready_structural_order_v1"
    ] = "binding_ready_structural_order_v1"
    selection_performed_before_endpoint_binding: Literal[True] = True
    endpoint_binding_outcome_used_for_selection: Literal[False] = False
    old_n10_status_used_for_selection: Literal[False] = False
    old_n10_selection_class_used_for_selection: Literal[False] = False
    external_novelty_outcome_used_for_selection: Literal[False] = False
    verifier_outcome_used_for_selection: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_selection(
        self,
    ) -> "ProspectiveRelationalHypothesisSelection":
        selected_values = (
            self.selected_original_hypothesis_id,
            self.selected_candidate_hypothesis_id,
            self.selected_final_hypothesis_id,
            self.selected_binding_plan_id,
            self.selected_binding_plan_sha256,
            self.selected_external_report,
            self.selected_source_query_plan,
        )
        if self.status == "SELECTED_BINDING_READY_HYPOTHESIS":
            if any(value is None for value in selected_values):
                raise ValueError("selected status requires complete selected lineage")
        elif any(value is not None for value in selected_values):
            raise ValueError("no-selection status cannot carry selected lineage")

        body = self.model_dump(mode="json")
        observed_id = body.pop("selection_id")
        observed_sha = body.pop("selection_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective relational selection SHA mismatch")
        if observed_id != (
            "prospective_relational_hypothesis_selection:"
            + expected_sha[:20]
        ):
            raise ValueError("prospective relational selection ID mismatch")
        return self


def _structural_selection_key(
    row: RelationalAtomicBindingHypothesisPlan,
) -> tuple[int, int, int, str]:
    # Deliberately excludes alpha6_decision, certification_status, and
    # n10_selection_class. Those are audit metadata, not selection features.
    return (
        -row.novelty_bearing_binding_ready_claim_count,
        -row.binding_ready_claim_count,
        row.claim_count,
        row.final_hypothesis_id,
    )


def _single_hypothesis_plan(
    *,
    source: RelationalAtomicBindingPlan,
    selected: RelationalAtomicBindingHypothesisPlan,
) -> RelationalAtomicBindingPlan:
    h_counts = Counter([selected.binding_status])
    c_counts = Counter(
        claim.binding_status
        for claim in selected.claims
    )
    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": source.run_dir,
        "source_alpha6_candidate_portfolio":
            source.source_alpha6_candidate_portfolio,
        "source_alpha6_candidate_portfolio_sha256":
            source.source_alpha6_candidate_portfolio_sha256,
        "source_certification_report":
            source.source_certification_report,
        "source_certification_report_sha256":
            source.source_certification_report_sha256,
        "hypotheses": [selected.model_dump(mode="json")],
        "hypothesis_count": 1,
        "ready_hypothesis_count": 1,
        "not_ready_hypothesis_count": 0,
        "claim_count": selected.claim_count,
        "binding_ready_claim_count":
            selected.binding_ready_claim_count,
        "novelty_bearing_binding_ready_claim_count":
            selected.novelty_bearing_binding_ready_claim_count,
        "hypothesis_status_counts": dict(sorted(h_counts.items())),
        "claim_status_counts": dict(sorted(c_counts.items())),
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def select_prospective_relational_hypothesis(
    *,
    binding_plan: RelationalAtomicBindingPlan,
) -> tuple[
    ProspectiveRelationalHypothesisSelection,
    RelationalAtomicBindingPlan | None,
]:
    ready = [
        row
        for row in binding_plan.hypotheses
        if row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
    ]
    ordered = sorted(ready, key=_structural_selection_key)
    structural_keys = {
        row.final_hypothesis_id: list(_structural_selection_key(row))
        for row in ordered
    }

    if not ordered:
        body = {
            "schema_version":
                "prospective-relational-hypothesis-selection-v1",
            "source_binding_plan_id": binding_plan.plan_id,
            "source_binding_plan_sha256": binding_plan.plan_sha256,
            "status": "NO_BINDING_READY_HYPOTHESIS",
            "selected_original_hypothesis_id": None,
            "selected_candidate_hypothesis_id": None,
            "selected_final_hypothesis_id": None,
            "selected_binding_plan_id": None,
            "selected_binding_plan_sha256": None,
            "selected_external_report": None,
            "selected_source_query_plan": None,
            "eligible_final_hypothesis_ids": [],
            "structural_order_keys": structural_keys,
            "selection_rule": "binding_ready_structural_order_v1",
            "selection_performed_before_endpoint_binding": True,
            "endpoint_binding_outcome_used_for_selection": False,
            "old_n10_status_used_for_selection": False,
            "old_n10_selection_class_used_for_selection": False,
            "external_novelty_outcome_used_for_selection": False,
            "verifier_outcome_used_for_selection": False,
            "scientific_quality_ranking_performed": False,
            "production_selection_authority": False,
        }
        digest = _sha256_json(body)
        return (
            ProspectiveRelationalHypothesisSelection(
                **body,
                selection_id=(
                    "prospective_relational_hypothesis_selection:"
                    + digest[:20]
                ),
                selection_sha256=digest,
            ),
            None,
        )

    selected = ordered[0]
    selected_plan = _single_hypothesis_plan(
        source=binding_plan,
        selected=selected,
    )

    certification_path = Path(
        binding_plan.source_certification_report
    ).expanduser().resolve()
    certification = json.loads(
        certification_path.read_text(encoding="utf-8")
    )
    artifacts = certification.get("candidate_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("source certification candidate_artifacts missing")
    matches = [
        row
        for row in artifacts
        if isinstance(row, dict)
        and str(row.get("candidate_id") or "")
        == selected.candidate_hypothesis_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "selected candidate does not resolve exactly one certification artifact"
        )
    artifact = matches[0]
    if (
        str(artifact.get("final_hypothesis_id") or "")
        != selected.final_hypothesis_id
    ):
        raise ValueError("selected certification artifact final lineage mismatch")
    if artifact.get("candidate_final_authority_equivalent") is not True:
        raise ValueError(
            "selected certification artifact lacks authority equivalence"
        )
    external_report = str(artifact.get("external_report") or "")
    source_query_plan = str(artifact.get("query_plan") or "")
    if not external_report or not source_query_plan:
        raise ValueError(
            "selected certification artifact lacks external/query-plan paths"
        )

    body = {
        "schema_version":
            "prospective-relational-hypothesis-selection-v1",
        "source_binding_plan_id": binding_plan.plan_id,
        "source_binding_plan_sha256": binding_plan.plan_sha256,
        "status": "SELECTED_BINDING_READY_HYPOTHESIS",
        "selected_original_hypothesis_id":
            selected.original_hypothesis_id,
        "selected_candidate_hypothesis_id":
            selected.candidate_hypothesis_id,
        "selected_final_hypothesis_id":
            selected.final_hypothesis_id,
        "selected_binding_plan_id": selected_plan.plan_id,
        "selected_binding_plan_sha256": selected_plan.plan_sha256,
        "selected_external_report": external_report,
        "selected_source_query_plan": source_query_plan,
        "eligible_final_hypothesis_ids": [
            row.final_hypothesis_id for row in ordered
        ],
        "structural_order_keys": structural_keys,
        "selection_rule": "binding_ready_structural_order_v1",
        "selection_performed_before_endpoint_binding": True,
        "endpoint_binding_outcome_used_for_selection": False,
        "old_n10_status_used_for_selection": False,
        "old_n10_selection_class_used_for_selection": False,
        "external_novelty_outcome_used_for_selection": False,
        "verifier_outcome_used_for_selection": False,
        "scientific_quality_ranking_performed": False,
        "production_selection_authority": False,
    }
    digest = _sha256_json(body)
    return (
        ProspectiveRelationalHypothesisSelection(
            **body,
            selection_id=(
                "prospective_relational_hypothesis_selection:"
                + digest[:20]
            ),
            selection_sha256=digest,
        ),
        selected_plan,
    )


__all__ = [
    "ProspectiveRelationalExecutionPlan",
    "ProspectiveRelationalExecutionSettings",
    "ProspectiveRelationalHypothesisSelection",
    "build_prospective_relational_execution_plan",
    "select_prospective_relational_hypothesis",
]
