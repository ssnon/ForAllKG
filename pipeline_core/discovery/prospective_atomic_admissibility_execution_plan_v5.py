from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
)
from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    ProspectiveAtomicAdmissibilityFreezeV5,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
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
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProspectiveAtomicAdmissibilityCaseExecutionPlanV5(StrictModel):
    case_id: Literal["P29", "P30", "P31", "P32", "P33"]
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
    downstream_result_conditioned_route_changes_allowed: Literal[
        False
    ] = False

    comparison_collector_contract_required_before_execution: Literal[
        True
    ] = True
    execution_authority_granted_by_this_plan: Literal[False] = False

    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(
        self,
    ) -> "ProspectiveAtomicAdmissibilityCaseExecutionPlanV5":
        expected_initial_prefix = [
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
        ]
        if self.initial_e2e_argv[:3] != expected_initial_prefix:
            raise ValueError(
                "v5 atomic admissibility initial argv uses unexpected runner"
            )
        if "--stop-after-initial-semantic" not in self.initial_e2e_argv:
            raise ValueError(
                "v5 atomic admissibility initial argv lacks semantic cut point"
            )

        forbidden_initial = {
            "--nonobviousness-original-fallback-enforce",
            "--nonobviousness-post-generation-enforce",
            "--nonobviousness-bounded-continuation-enforce",
            "--nonobviousness-enforce",
            "--overwrite-run",
        }
        observed_forbidden = sorted(
            flag for flag in forbidden_initial
            if flag in self.initial_e2e_argv
        )
        if observed_forbidden:
            raise ValueError(
                "v5 atomic admissibility initial argv carries forbidden flags: "
                + repr(observed_forbidden)
            )

        expected_downstream_prefix = [
            "python",
            "-m",
            "scripts.discovery.run_pre_n10_prospective_campaign_v1",
        ]
        if self.downstream_campaign_argv_base[:3] != expected_downstream_prefix:
            raise ValueError(
                "v5 downstream argv uses unexpected campaign runner"
            )
        if "--semantic-review" in self.downstream_campaign_argv_base:
            raise ValueError(
                "v5 downstream base argv must defer semantic-review"
            )
        if "--allow-dirty-worktree" in self.downstream_campaign_argv_base:
            raise ValueError(
                "v5 execution plan must not authorize dirty V_post"
            )
        if "--regeneration-unit-freeze" not in self.downstream_campaign_argv_base:
            raise ValueError(
                "v5 downstream argv lacks regeneration freeze"
            )
        if "--save-prompts" not in self.downstream_campaign_argv_base:
            raise ValueError(
                "v5 downstream argv must preserve prompts"
            )
        if self.downstream_campaign_stage_order != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError(
                "v5 downstream campaign stage order mismatch"
            )
        return self


class ProspectiveAtomicAdmissibilityExecutionPlanV5(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-execution-plan-v5"
    ] = "prospective-atomic-admissibility-execution-plan-v5"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_campaign_freeze_id: str
    source_campaign_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_campaign_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_unit_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    execution_plan_repository_head_sha: str = Field(
        pattern=r"^[0-9a-f]{40}$"
    )
    repository_tracked_worktree_dirty: Literal[False] = False

    settings: ProspectiveAuthorityExecutionSettingsV3
    cases: list[ProspectiveAtomicAdmissibilityCaseExecutionPlanV5]
    case_ids: list[str]
    case_count: Literal[5] = 5

    prospective_goal: Literal[
        "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10"
    ] = "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10"

    source_tasks_and_models_frozen_before_execution: Literal[True] = True
    initial_argv_frozen_before_execution: Literal[True] = True
    downstream_argv_frozen_before_execution: Literal[True] = True
    semantic_review_argument_policy_frozen_before_execution: Literal[
        True
    ] = True

    initial_hypothesis_outputs_observed_before_plan_freeze: Literal[
        False
    ] = False
    semantic_outputs_observed_before_plan_freeze: Literal[False] = False
    legacy_vpre_outputs_observed_before_plan_freeze: Literal[False] = False
    neutral_pre_n10_outputs_observed_before_plan_freeze: Literal[
        False
    ] = False
    source_reference_outcomes_observed_before_plan_freeze: Literal[
        False
    ] = False
    semantic_fidelity_outcomes_observed_before_plan_freeze: Literal[
        False
    ] = False
    comparison_outputs_observed_before_plan_freeze: Literal[False] = False
    external_novelty_outputs_observed_before_plan_freeze: Literal[
        False
    ] = False
    n10_outputs_observed_before_plan_freeze: Literal[False] = False

    comparison_collector_contract_required_before_execution: Literal[
        True
    ] = True
    comparison_collector_contract_frozen_by_this_plan: Literal[False] = False
    execution_authority_granted_by_this_plan: Literal[False] = False

    case_order_fixed_p29_to_p33: Literal[True] = True
    later_case_adaptation_allowed: Literal[False] = False
    failed_or_terminal_case_replacement_allowed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False

    prospective_evidence_collection: Literal[True] = True
    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(
        self,
    ) -> "ProspectiveAtomicAdmissibilityExecutionPlanV5":
        expected = ["P29", "P30", "P31", "P32", "P33"]
        if self.case_ids != expected:
            raise ValueError(
                "v5 atomic admissibility execution case IDs must be P29-P33"
            )
        if [row.case_id for row in self.cases] != expected:
            raise ValueError(
                "v5 atomic admissibility cases must be ordered P29-P33"
            )
        if len({row.source_task_id for row in self.cases}) != 5:
            raise ValueError(
                "v5 atomic admissibility source task IDs must be unique"
            )
        if any(row.scientific_validation_authority for row in self.cases):
            raise ValueError(
                "v5 case cannot carry scientific validation authority"
            )
        if any(row.execution_authority_granted_by_this_plan for row in self.cases):
            raise ValueError(
                "v5 case plan cannot grant execution authority"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "prospective atomic admissibility execution-plan v5 SHA mismatch"
            )
        if observed_id != (
            "prospective_atomic_admissibility_execution_plan_v5:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "prospective atomic admissibility execution-plan v5 ID mismatch"
            )
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
        "campaign_root": str(
            run / "prospective_atomic_admissibility_v5"
        ),
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


def materialize_downstream_argv_v5(
    case: ProspectiveAtomicAdmissibilityCaseExecutionPlanV5,
) -> list[str]:
    argv = list(case.downstream_campaign_argv_base)
    review = Path(case.initial_semantic_review_path)
    if review.is_file():
        argv += ["--semantic-review", str(review.expanduser().resolve())]
    return argv


def build_prospective_atomic_admissibility_execution_plan_v5(
    *,
    campaign_freeze: ProspectiveAtomicAdmissibilityFreezeV5,
    campaign_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_path: Path,
    regeneration_unit_freeze_file_sha256: str,
    execution_plan_repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
    settings: ProspectiveAuthorityExecutionSettingsV3 | None = None,
) -> ProspectiveAtomicAdmissibilityExecutionPlanV5:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "prospective atomic admissibility execution-plan v5 "
            "requires a clean tracked worktree"
        )

    if (
        campaign_freeze.source_regeneration_unit_freeze_id
        != regeneration_unit_freeze.freeze_id
    ):
        raise ValueError(
            "v5 campaign/regeneration-unit freeze ID mismatch"
        )
    if (
        campaign_freeze.source_regeneration_unit_freeze_sha256
        != regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError(
            "v5 campaign/regeneration-unit freeze SHA mismatch"
        )
    if (
        campaign_freeze.source_regeneration_unit_freeze_file_sha256
        != regeneration_unit_freeze_file_sha256
    ):
        raise ValueError(
            "v5 campaign/regeneration-unit freeze file SHA mismatch"
        )

    if campaign_freeze.scientific_validation_authority:
        raise ValueError(
            "v5 campaign must not carry scientific validation authority"
        )
    if not campaign_freeze.engineering_diagnostic_only:
        raise ValueError("v5 campaign must be engineering diagnostic")

    resolved = settings or ProspectiveAuthorityExecutionSettingsV3()
    cases: list[ProspectiveAtomicAdmissibilityCaseExecutionPlanV5] = []

    for task in campaign_freeze.tasks:
        downstream_argv, paths = _downstream_campaign_argv_base(
            task=task,
            regeneration_unit_freeze_path=regeneration_unit_freeze_path,
            settings=resolved,
        )
        cases.append(
            ProspectiveAtomicAdmissibilityCaseExecutionPlanV5(
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
        "schema_version": (
            "prospective-atomic-admissibility-execution-plan-v5"
        ),
        "source_campaign_freeze_id": campaign_freeze.freeze_id,
        "source_campaign_freeze_sha256": campaign_freeze.freeze_sha256,
        "source_campaign_freeze_file_sha256": campaign_freeze_file_sha256,
        "source_regeneration_unit_freeze_id": (
            regeneration_unit_freeze.freeze_id
        ),
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
        "case_ids": ["P29", "P30", "P31", "P32", "P33"],
        "case_count": 5,
        "prospective_goal": "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10",
        "source_tasks_and_models_frozen_before_execution": True,
        "initial_argv_frozen_before_execution": True,
        "downstream_argv_frozen_before_execution": True,
        "semantic_review_argument_policy_frozen_before_execution": True,
        "initial_hypothesis_outputs_observed_before_plan_freeze": False,
        "semantic_outputs_observed_before_plan_freeze": False,
        "legacy_vpre_outputs_observed_before_plan_freeze": False,
        "neutral_pre_n10_outputs_observed_before_plan_freeze": False,
        "source_reference_outcomes_observed_before_plan_freeze": False,
        "semantic_fidelity_outcomes_observed_before_plan_freeze": False,
        "comparison_outputs_observed_before_plan_freeze": False,
        "external_novelty_outputs_observed_before_plan_freeze": False,
        "n10_outputs_observed_before_plan_freeze": False,
        "comparison_collector_contract_required_before_execution": True,
        "comparison_collector_contract_frozen_by_this_plan": False,
        "execution_authority_granted_by_this_plan": False,
        "case_order_fixed_p29_to_p33": True,
        "later_case_adaptation_allowed": False,
        "failed_or_terminal_case_replacement_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "second_regeneration_allowed": False,
        "prospective_evidence_collection": True,
        "engineering_diagnostic_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)

    return ProspectiveAtomicAdmissibilityExecutionPlanV5(
        **body,
        plan_id=(
            "prospective_atomic_admissibility_execution_plan_v5:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


__all__ = [
    "ProspectiveAtomicAdmissibilityCaseExecutionPlanV5",
    "ProspectiveAtomicAdmissibilityExecutionPlanV5",
    "build_prospective_atomic_admissibility_execution_plan_v5",
    "materialize_downstream_argv_v5",
    "sha256_file",
]
