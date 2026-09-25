from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
)
from pipeline_core.discovery.prospective_authority_campaign_freeze_v3 import (
    ProspectiveAuthorityCampaignFreezeV3,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
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
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProspectiveAuthorityExecutionSettingsV3(StrictModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    provider_request: Literal["auto"] = "auto"
    results_per_query: Literal[12] = 12
    context_review_mode: Literal["auto"] = "auto"

    parse_retries: Literal[1] = 1
    timeout_seconds: Literal[180.0] = 180.0
    max_claims: Literal[4] = 4
    max_queries_per_claim: Literal[2] = 2
    save_prompts: Literal[True] = True

    @model_validator(mode="after")
    def validate_settings(self) -> "ProspectiveAuthorityExecutionSettingsV3":
        if not self.base_url.strip():
            raise ValueError("v3 execution settings require base_url")
        if not self.api_key_env.strip():
            raise ValueError("v3 execution settings require api_key_env")
        return self


class ProspectiveAuthorityCaseExecutionPlanV3(StrictModel):
    case_id: Literal["P21", "P22", "P23", "P24", "P25"]
    source_task_id: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    initial_e2e_argv: list[str]

    initial_manifest_path: str
    initial_portfolio_path: str
    initial_hypothesis_context_path: str
    initial_semantic_run_path: str
    initial_semantic_review_path: str
    initial_provider_plan_path: str

    downstream_campaign_output_root: str
    downstream_campaign_argv_base: list[str]
    semantic_review_argv_policy: Literal[
        "APPEND_IF_PRESENT_ELSE_OMIT"
    ] = "APPEND_IF_PRESENT_ELSE_OMIT"

    initial_cutpoint_required: Literal[True] = True
    initial_external_novelty_allowed: Literal[False] = False
    initial_n9_allowed: Literal[False] = False
    initial_n10_allowed: Literal[False] = False
    initial_refinement_allowed: Literal[False] = False

    downstream_campaign_stage_order: list[str]
    downstream_second_regeneration_allowed: Literal[False] = False
    downstream_result_conditioned_route_changes_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(self) -> "ProspectiveAuthorityCaseExecutionPlanV3":
        expected_initial_prefix = [
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
        ]
        if self.initial_e2e_argv[:3] != expected_initial_prefix:
            raise ValueError("v3 initial argv uses unexpected runner")
        if "--stop-after-initial-semantic" not in self.initial_e2e_argv:
            raise ValueError("v3 initial argv lacks semantic cut point")
        forbidden_initial = {
            "--nonobviousness-original-fallback-enforce",
            "--nonobviousness-post-generation-enforce",
            "--nonobviousness-bounded-continuation-enforce",
            "--nonobviousness-enforce",
            "--overwrite-run",
        }
        observed_forbidden = sorted(
            flag for flag in forbidden_initial if flag in self.initial_e2e_argv
        )
        if observed_forbidden:
            raise ValueError(
                "v3 initial argv carries post-cutpoint/overwrite flags: "
                + repr(observed_forbidden)
            )

        expected_downstream_prefix = [
            "python",
            "-m",
            "scripts.discovery.run_pre_n10_prospective_campaign_v1",
        ]
        if self.downstream_campaign_argv_base[:3] != expected_downstream_prefix:
            raise ValueError("v3 downstream argv uses unexpected campaign runner")
        if "--semantic-review" in self.downstream_campaign_argv_base:
            raise ValueError(
                "v3 downstream base argv must defer semantic-review argument"
            )
        if "--allow-dirty-worktree" in self.downstream_campaign_argv_base:
            raise ValueError("v3 execution plan must not authorize dirty V_post")
        if "--regeneration-unit-freeze" not in self.downstream_campaign_argv_base:
            raise ValueError("v3 downstream argv lacks regeneration freeze")
        if "--save-prompts" not in self.downstream_campaign_argv_base:
            raise ValueError("v3 downstream argv must preserve prompts")
        if self.downstream_campaign_stage_order != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError("v3 downstream stage order mismatch")
        return self


class ProspectiveAuthorityExecutionPlanV3(StrictModel):
    schema_version: Literal[
        "prospective-authority-execution-plan-v3"
    ] = "prospective-authority-execution-plan-v3"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_campaign_freeze_id: str
    source_campaign_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_campaign_freeze_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_unit_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    execution_plan_repository_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_tracked_worktree_dirty: Literal[False] = False

    settings: ProspectiveAuthorityExecutionSettingsV3
    cases: list[ProspectiveAuthorityCaseExecutionPlanV3]
    case_ids: list[str]
    case_count: Literal[5] = 5

    source_tasks_and_models_frozen_before_execution: Literal[True] = True
    initial_argv_frozen_before_execution: Literal[True] = True
    downstream_argv_frozen_before_execution: Literal[True] = True
    semantic_review_argument_policy_frozen_before_execution: Literal[True] = True

    initial_hypothesis_outputs_observed_before_plan_freeze: Literal[False] = False
    semantic_outputs_observed_before_plan_freeze: Literal[False] = False
    primary_route_outputs_observed_before_plan_freeze: Literal[False] = False
    regeneration_outputs_observed_before_plan_freeze: Literal[False] = False
    external_novelty_outputs_observed_before_plan_freeze: Literal[False] = False
    n10_outputs_observed_before_plan_freeze: Literal[False] = False
    endpoint_binding_outputs_observed_before_plan_freeze: Literal[False] = False
    verifier_outputs_observed_before_plan_freeze: Literal[False] = False

    case_order_fixed_p21_to_p25: Literal[True] = True
    later_case_adaptation_allowed: Literal[False] = False
    failed_or_abstained_case_replacement_allowed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "ProspectiveAuthorityExecutionPlanV3":
        expected = ["P21", "P22", "P23", "P24", "P25"]
        if self.case_ids != expected:
            raise ValueError("v3 execution-plan case IDs must be P21-P25")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("v3 execution-plan cases must be ordered P21-P25")
        if len({row.source_task_id for row in self.cases}) != 5:
            raise ValueError("v3 execution-plan source task IDs must be unique")

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective authority execution-plan v3 SHA mismatch")
        if observed_id != (
            "prospective_authority_execution_plan_v3:" + expected_sha[:20]
        ):
            raise ValueError("prospective authority execution-plan v3 ID mismatch")
        return self


def _initial_e2e_argv(
    *,
    task,
    settings: ProspectiveAuthorityExecutionSettingsV3,
) -> list[str]:
    policy = task.model_policy
    argv = [
        "python",
        "-m",
        "scripts.discovery.run_dac_discovery_e2e",
        "--run-dir",
        str(Path(task.run_dir).expanduser().resolve()),
        "--source",
        task.source,
    ]
    if task.stop is not None:
        argv += ["--stop", task.stop]
    argv += [
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
        settings.context_review_mode,
        "--model",
        policy.initial_generation_model,
        "--critic-model",
        policy.initial_critic_model,
        "--base-url",
        settings.base_url,
        "--api-key-env",
        settings.api_key_env,
        "--providers",
        settings.provider_request,
        "--results-per-query",
        str(settings.results_per_query),
        "--stop-after-initial-semantic",
    ]
    return argv


def _downstream_campaign_argv_base(
    *,
    task,
    regeneration_unit_freeze_path: Path,
    settings: ProspectiveAuthorityExecutionSettingsV3,
) -> tuple[list[str], dict[str, str]]:
    run = Path(task.run_dir).expanduser().resolve()
    policy = task.model_policy
    paths = {
        "manifest": str(run / "e2e_runner.manifest.json"),
        "portfolio": str(run / "hypothesis_axis_a4.portfolio.json"),
        "context": str(run / "hypothesis.context.json"),
        "semantic_run": str(run / "semantic_axis_a4.run.json"),
        "semantic_review": str(run / "semantic_axis_a4.review.json"),
        "provider_plan": str(run / "literature_provider_plan.json"),
        "campaign_root": str(run / "prospective_authority_v3"),
    }
    argv = [
        "python",
        "-m",
        "scripts.discovery.run_pre_n10_prospective_campaign_v1",
        "--portfolio",
        paths["portfolio"],
        "--semantic-run",
        paths["semantic_run"],
        "--hypothesis-context",
        paths["context"],
        "--regeneration-unit-freeze",
        str(regeneration_unit_freeze_path.expanduser().resolve()),
        "--provider-plan",
        paths["provider_plan"],
        "--output-root",
        paths["campaign_root"],
        "--model",
        policy.initial_generation_model,
        "--decomposition-model",
        policy.decomposition_model,
        "--primary-model",
        policy.primary_model,
        "--specification-repair-model",
        policy.specification_repair_model,
        "--specification-audit-model",
        policy.specification_audit_model,
        "--source-alignment-model",
        policy.source_alignment_model,
        "--regeneration-model",
        policy.regeneration_model,
        "--semantic-critic-model",
        policy.semantic_critic_model,
        "--external-n10-model",
        policy.external_n10_model,
        "--vpost-model",
        policy.vpost_model,
        "--api-key-env",
        settings.api_key_env,
        "--base-url",
        settings.base_url,
        "--parse-retries",
        str(settings.parse_retries),
        "--timeout-seconds",
        str(settings.timeout_seconds),
        "--max-claims",
        str(settings.max_claims),
        "--max-queries-per-claim",
        str(settings.max_queries_per_claim),
        "--save-prompts",
    ]
    return argv, paths


def materialize_downstream_argv_v3(
    case: ProspectiveAuthorityCaseExecutionPlanV3,
) -> list[str]:
    """Materialize the only runtime-conditional argument in the frozen plan.

    The semantic review file exists only for an accepted semantic run.  The
    initial-semantic cutpoint already freezes this file-existence contract.
    No scientific result changes route/model/task selection here.
    """
    argv = list(case.downstream_campaign_argv_base)
    review = Path(case.initial_semantic_review_path)
    if review.is_file():
        argv += ["--semantic-review", str(review.expanduser().resolve())]
    return argv


def build_prospective_authority_execution_plan_v3(
    *,
    campaign_freeze: ProspectiveAuthorityCampaignFreezeV3,
    campaign_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_path: Path,
    regeneration_unit_freeze_file_sha256: str,
    execution_plan_repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
    settings: ProspectiveAuthorityExecutionSettingsV3 | None = None,
) -> ProspectiveAuthorityExecutionPlanV3:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "prospective authority execution-plan v3 requires a clean tracked worktree"
        )

    if (
        campaign_freeze.source_regeneration_unit_freeze_id
        != regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("v3 campaign/regeneration-unit freeze ID mismatch")
    if (
        campaign_freeze.source_regeneration_unit_freeze_sha256
        != regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("v3 campaign/regeneration-unit freeze SHA mismatch")
    if (
        campaign_freeze.source_regeneration_unit_freeze_file_sha256
        != regeneration_unit_freeze_file_sha256
    ):
        raise ValueError(
            "v3 campaign/regeneration-unit freeze file SHA mismatch"
        )

    resolved = settings or ProspectiveAuthorityExecutionSettingsV3()
    cases: list[ProspectiveAuthorityCaseExecutionPlanV3] = []
    for task in campaign_freeze.tasks:
        downstream_argv, paths = _downstream_campaign_argv_base(
            task=task,
            regeneration_unit_freeze_path=regeneration_unit_freeze_path,
            settings=resolved,
        )
        cases.append(
            ProspectiveAuthorityCaseExecutionPlanV3(
                case_id=task.case_id,
                source_task_id=task.task_id,
                source_task_sha256=task.task_sha256,
                run_dir=str(Path(task.run_dir).expanduser().resolve()),
                initial_e2e_argv=_initial_e2e_argv(
                    task=task,
                    settings=resolved,
                ),
                initial_manifest_path=paths["manifest"],
                initial_portfolio_path=paths["portfolio"],
                initial_hypothesis_context_path=paths["context"],
                initial_semantic_run_path=paths["semantic_run"],
                initial_semantic_review_path=paths["semantic_review"],
                initial_provider_plan_path=paths["provider_plan"],
                downstream_campaign_output_root=paths["campaign_root"],
                downstream_campaign_argv_base=downstream_argv,
                downstream_campaign_stage_order=list(CAMPAIGN_STAGE_ORDER),
            )
        )

    body = {
        "schema_version": "prospective-authority-execution-plan-v3",
        "source_campaign_freeze_id": campaign_freeze.freeze_id,
        "source_campaign_freeze_sha256": campaign_freeze.freeze_sha256,
        "source_campaign_freeze_file_sha256": campaign_freeze_file_sha256,
        "source_regeneration_unit_freeze_id": regeneration_unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256": (
            regeneration_unit_freeze.freeze_sha256
        ),
        "source_regeneration_unit_freeze_file_sha256": (
            regeneration_unit_freeze_file_sha256
        ),
        "execution_plan_repository_head_sha": execution_plan_repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "settings": resolved.model_dump(mode="json"),
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P21", "P22", "P23", "P24", "P25"],
        "case_count": 5,
        "source_tasks_and_models_frozen_before_execution": True,
        "initial_argv_frozen_before_execution": True,
        "downstream_argv_frozen_before_execution": True,
        "semantic_review_argument_policy_frozen_before_execution": True,
        "initial_hypothesis_outputs_observed_before_plan_freeze": False,
        "semantic_outputs_observed_before_plan_freeze": False,
        "primary_route_outputs_observed_before_plan_freeze": False,
        "regeneration_outputs_observed_before_plan_freeze": False,
        "external_novelty_outputs_observed_before_plan_freeze": False,
        "n10_outputs_observed_before_plan_freeze": False,
        "endpoint_binding_outputs_observed_before_plan_freeze": False,
        "verifier_outputs_observed_before_plan_freeze": False,
        "case_order_fixed_p21_to_p25": True,
        "later_case_adaptation_allowed": False,
        "failed_or_abstained_case_replacement_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "second_regeneration_allowed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveAuthorityExecutionPlanV3(
        **body,
        plan_id="prospective_authority_execution_plan_v3:" + digest[:20],
        plan_sha256=digest,
    )


__all__ = [
    "ProspectiveAuthorityCaseExecutionPlanV3",
    "ProspectiveAuthorityExecutionPlanV3",
    "ProspectiveAuthorityExecutionSettingsV3",
    "build_prospective_authority_execution_plan_v3",
    "materialize_downstream_argv_v3",
    "sha256_file",
]
