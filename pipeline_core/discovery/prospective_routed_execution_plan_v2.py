from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    ProspectiveRegenerationDownstreamV2Freeze,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze_v2 import (
    ProspectiveRoutedCampaignFreezeV2,
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


RouteActionV2 = Literal[
    "PROCEED_TO_LITERAL_ENDPOINT_BINDING",
    "ZERO_DELTA_SPECIFICATION_REPAIR",
    "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE",
    "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE",
]


class RoutedExecutionSettingsV2(StrictModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    provider_request: Literal["auto"] = "auto"
    results_per_query: Literal[12] = 12

    endpoint_temperature: Literal[0.0] = 0.0
    endpoint_parse_retries: Literal[3] = 3
    endpoint_timeout_seconds: Literal[180.0] = 180.0

    route_temperature: Literal[0.0] = 0.0
    route_parse_retries: Literal[3] = 3
    route_timeout_seconds: Literal[180.0] = 180.0

    support_results_per_query: Literal[12] = 12
    second_pass_results_per_query: Literal[16] = 16
    max_review_works_per_claim: Literal[20] = 20

    post_generation_n10_authority_mode: Literal[
        "certification_only"
    ] = "certification_only"
    original_fallback_n10_enabled: Literal[True] = True
    post_generation_n10_enabled: Literal[True] = True
    bounded_continuation_enabled: Literal[False] = False

    pre_route_gate: Literal[
        "preverifier-contract-gate-v2"
    ] = "preverifier-contract-gate-v2"
    post_route_gate: Literal[
        "preverifier-contract-gate-v2"
    ] = "preverifier-contract-gate-v2"

    route_dispatch_scope: Literal[
        "claim_and_hypothesis_structural"
    ] = "claim_and_hypothesis_structural"

    source_alignment_generation_mode: Literal[
        "select_existing_hypothesis_card_surfaces_only"
    ] = "select_existing_hypothesis_card_surfaces_only"
    source_alignment_new_text_allowed: Literal[False] = False
    source_alignment_semantic_delta_audit_required: Literal[True] = True

    atomic_decomposition_generation_mode: Literal[
        "source_supported_atomic_components_only"
    ] = "source_supported_atomic_components_only"
    atomic_decomposition_semantic_audit_required: Literal[True] = True

    regeneration_operation: Literal[
        "PROSPECTIVE_REGENERATION_UNIT_V2"
    ] = "PROSPECTIVE_REGENERATION_UNIT_V2"
    regeneration_input_authority: Literal[
        "frozen_hypothesis_context_only"
    ] = "frozen_hypothesis_context_only"
    regeneration_full_e2e_argv_allowed: Literal[False] = False
    regeneration_previous_hypothesis_text_allowed: Literal[False] = False
    regeneration_novelty_or_verifier_outcome_allowed: Literal[False] = False

    regeneration_downstream_operation: Literal[
        "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"
    ] = "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"
    regeneration_downstream_full_e2e_rerun_allowed: Literal[False] = False

    specification_repair_reuses_policy_v1: Literal[True] = True
    specification_repair_semantic_delta_audit_required: Literal[True] = True

    post_route_selection_rule: Literal[
        "gate_v2_ready_structural_order_v1"
    ] = "gate_v2_ready_structural_order_v1"
    post_route_selection_uses_only_gate_ready_structure: Literal[True] = True

    endpoint_binding_requires_post_route_gate_ready: Literal[True] = True
    projection_preflight_required_before_verifier: Literal[True] = True
    verifier_requires_at_least_one_projected_claim: Literal[True] = True

    post_case_adaptation_allowed: Literal[False] = False
    failed_or_abstained_case_replacement_allowed: Literal[False] = False
    second_route_attempt_after_failed_post_gate_allowed: Literal[False] = False


class RoutedStageProtocolV2(StrictModel):
    route_action: RouteActionV2
    primary_operation: str
    primary_max_attempts: Literal[1] = 1
    primary_llm_calls_per_unit_max: int = Field(ge=0, le=2)
    primary_audit_calls_per_unit_max: int = Field(ge=0, le=1)

    fallback_operation: Literal[
        "PROSPECTIVE_REGENERATION_UNIT_V2"
    ] | None = None
    fallback_max_attempts: int = Field(default=0, ge=0, le=1)
    fallback_structured_generation_calls_per_hypothesis_max: int = Field(
        default=0,
        ge=0,
        le=1,
    )
    fallback_repair_calls_per_hypothesis_max: Literal[0] = 0
    fallback_full_e2e_argv_allowed: Literal[False] = False

    fallback_downstream_operation: Literal[
        "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"
    ] | None = None

    post_route_gate_v2_required: Literal[True] = True
    second_primary_attempt_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_protocol(self) -> "RoutedStageProtocolV2":
        is_regen_route = "THEN_REGENERATE_IF_UNAVAILABLE" in self.route_action

        if self.route_action == "PROCEED_TO_LITERAL_ENDPOINT_BINDING":
            if self.primary_llm_calls_per_unit_max != 0:
                raise ValueError("proceed route cannot perform route LLM calls")
            if self.primary_audit_calls_per_unit_max != 0:
                raise ValueError("proceed route cannot perform route audit calls")

        if is_regen_route:
            if self.fallback_operation != "PROSPECTIVE_REGENERATION_UNIT_V2":
                raise ValueError(
                    "regeneration route must use regeneration-unit-v2"
                )
            if self.fallback_max_attempts != 1:
                raise ValueError("regeneration fallback attempt budget must be one")
            if (
                self.fallback_structured_generation_calls_per_hypothesis_max
                != 1
            ):
                raise ValueError(
                    "regeneration fallback generation-call budget must be one"
                )
            if self.fallback_repair_calls_per_hypothesis_max != 0:
                raise ValueError("regeneration fallback repair calls must be zero")
            if self.fallback_full_e2e_argv_allowed:
                raise ValueError("regeneration fallback cannot use full E2E")
            if (
                self.fallback_downstream_operation
                != "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"
            ):
                raise ValueError(
                    "regeneration fallback must use downstream-v2 evaluation"
                )
        else:
            if self.fallback_operation is not None:
                raise ValueError("non-regeneration route cannot carry fallback")
            if self.fallback_max_attempts != 0:
                raise ValueError("non-regeneration route cannot carry fallback attempts")
            if self.fallback_structured_generation_calls_per_hypothesis_max != 0:
                raise ValueError(
                    "non-regeneration route cannot carry generation-call budget"
                )
            if self.fallback_downstream_operation is not None:
                raise ValueError(
                    "non-regeneration route cannot carry downstream fallback"
                )
        return self


class ProspectiveRoutedCaseExecutionPlanV2(StrictModel):
    case_id: Literal["P16", "P17", "P18", "P19", "P20"]
    source_task_id: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_dir: str

    generation_model: str
    critic_model: str
    route_model: str
    route_audit_model: str
    regeneration_model: str

    initial_main_e2e_argv: list[str]
    initial_main_e2e_is_regeneration: Literal[False] = False

    hypothesis_context_path: str
    full_binding_plan_path: str
    pre_route_gate_path: str
    routed_dispatch_path: str

    routed_lineage_dir: str
    regeneration_lineage_dir_template: str
    regeneration_unit_result_path_template: str
    regeneration_portfolio_path_template: str
    regeneration_downstream_dir_template: str
    regeneration_downstream_report_path_template: str

    regeneration_argv: None = None
    regeneration_full_e2e_argv_present: Literal[False] = False

    post_route_binding_plan_path: str
    post_route_gate_path: str

    selection_report_path: str
    selected_binding_plan_path: str

    endpoint_binding_report_path: str
    endpoint_binding_prompt_path: str
    endpoint_binding_telemetry_path: str

    projection_preflight_output_dir: str
    relational_verifier_output_dir: str

    initial_main_e2e_runs_before_binding_plan: Literal[True] = True
    binding_plan_precedes_pre_route_gate: Literal[True] = True
    pre_route_gate_precedes_route_dispatch: Literal[True] = True
    route_dispatch_precedes_route_primary_or_regeneration: Literal[True] = True
    regeneration_unit_precedes_regeneration_downstream: Literal[True] = True
    regeneration_downstream_precedes_post_route_gate: Literal[True] = True
    post_route_gate_precedes_structural_selection: Literal[True] = True
    structural_selection_precedes_endpoint_binding: Literal[True] = True
    endpoint_binding_precedes_projection_preflight: Literal[True] = True
    projection_preflight_precedes_verifier: Literal[True] = True

    novelty_outcome_used_for_route_dispatch: Literal[False] = False
    verifier_outcome_used_for_route_dispatch: Literal[False] = False
    earlier_case_outcome_used_for_settings: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(self) -> "ProspectiveRoutedCaseExecutionPlanV2":
        if self.regeneration_argv is not None:
            raise ValueError("v2 case plan must not carry regeneration argv")
        token = "{final_hypothesis_id_slug}"
        for value in (
            self.regeneration_lineage_dir_template,
            self.regeneration_unit_result_path_template,
            self.regeneration_portfolio_path_template,
            self.regeneration_downstream_dir_template,
            self.regeneration_downstream_report_path_template,
        ):
            if token not in value:
                raise ValueError(
                    "regeneration path templates must contain " + token
                )
        return self


class ProspectiveRoutedExecutionPlanV2(StrictModel):
    schema_version: Literal[
        "prospective-routed-execution-plan-v2"
    ] = "prospective-routed-execution-plan-v2"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_campaign_freeze_id: str
    source_campaign_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_campaign_freeze_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_campaign_freeze_repository_head_sha: str

    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_unit_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    source_regeneration_downstream_freeze_id: str
    source_regeneration_downstream_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_downstream_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    execution_plan_repository_head_sha: str
    repository_tracked_worktree_dirty: Literal[False] = False

    settings: RoutedExecutionSettingsV2
    route_protocols: list[RoutedStageProtocolV2]
    cases: list[ProspectiveRoutedCaseExecutionPlanV2]
    case_ids: list[str]
    case_count: Literal[5] = 5

    all_case_settings_frozen_before_first_case_execution: Literal[True] = True
    all_route_protocols_frozen_before_first_case_execution: Literal[True] = True
    regeneration_unit_contract_frozen_before_first_case_execution: Literal[
        True
    ] = True
    regeneration_downstream_contract_frozen_before_first_case_execution: Literal[
        True
    ] = True
    case_order_fixed_p16_to_p20: Literal[True] = True

    no_regeneration_full_e2e_argv_in_plan: Literal[True] = True
    initial_full_e2e_is_not_regeneration: Literal[True] = True

    gate_v2_results_observed_before_plan_freeze: Literal[False] = False
    repair_results_observed_before_plan_freeze: Literal[False] = False
    regeneration_results_observed_before_plan_freeze: Literal[False] = False
    endpoint_results_observed_before_plan_freeze: Literal[False] = False
    verifier_results_observed_before_plan_freeze: Literal[False] = False

    post_case_adaptation_allowed: Literal[False] = False
    case_replacement_allowed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "ProspectiveRoutedExecutionPlanV2":
        expected = ["P16", "P17", "P18", "P19", "P20"]
        if self.case_ids != expected:
            raise ValueError("routed execution-v2 case IDs must be P16-P20")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("routed execution-v2 cases must be ordered P16-P20")
        if len({row.source_task_id for row in self.cases}) != 5:
            raise ValueError("routed execution-v2 source task IDs must be unique")

        actions = [row.route_action for row in self.route_protocols]
        expected_actions = [
            "PROCEED_TO_LITERAL_ENDPOINT_BINDING",
            "ZERO_DELTA_SPECIFICATION_REPAIR",
            "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE",
            "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE",
        ]
        if actions != expected_actions:
            raise ValueError(
                "route protocols must cover frozen actions in canonical order"
            )

        if any(case.regeneration_argv is not None for case in self.cases):
            raise ValueError("v2 execution plan must contain no regeneration argv")

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective routed execution-v2 SHA mismatch")
        if observed_id != (
            "prospective_routed_execution_plan_v2:" + expected_sha[:20]
        ):
            raise ValueError("prospective routed execution-v2 ID mismatch")
        return self


def default_route_protocols_v2() -> list[RoutedStageProtocolV2]:
    return [
        RoutedStageProtocolV2(
            route_action="PROCEED_TO_LITERAL_ENDPOINT_BINDING",
            primary_operation="PASSTHROUGH_NO_MUTATION",
            primary_llm_calls_per_unit_max=0,
            primary_audit_calls_per_unit_max=0,
        ),
        RoutedStageProtocolV2(
            route_action="ZERO_DELTA_SPECIFICATION_REPAIR",
            primary_operation="BOUNDED_SPECIFICATION_REPAIR_V1",
            primary_llm_calls_per_unit_max=1,
            primary_audit_calls_per_unit_max=1,
        ),
        RoutedStageProtocolV2(
            route_action=(
                "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE"
            ),
            primary_operation="SELECT_EXISTING_HYPOTHESIS_CARD_SURFACES",
            primary_llm_calls_per_unit_max=1,
            primary_audit_calls_per_unit_max=1,
            fallback_operation="PROSPECTIVE_REGENERATION_UNIT_V2",
            fallback_max_attempts=1,
            fallback_structured_generation_calls_per_hypothesis_max=1,
            fallback_repair_calls_per_hypothesis_max=0,
            fallback_full_e2e_argv_allowed=False,
            fallback_downstream_operation=(
                "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"
            ),
        ),
        RoutedStageProtocolV2(
            route_action=(
                "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE"
            ),
            primary_operation="SOURCE_SUPPORTED_ATOMIC_DECOMPOSITION",
            primary_llm_calls_per_unit_max=1,
            primary_audit_calls_per_unit_max=1,
            fallback_operation="PROSPECTIVE_REGENERATION_UNIT_V2",
            fallback_max_attempts=1,
            fallback_structured_generation_calls_per_hypothesis_max=1,
            fallback_repair_calls_per_hypothesis_max=0,
            fallback_full_e2e_argv_allowed=False,
            fallback_downstream_operation=(
                "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"
            ),
        ),
    ]


def _initial_main_e2e_argv(
    *,
    task: object,
    settings: RoutedExecutionSettingsV2,
) -> list[str]:
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
            settings.base_url,
            "--api-key-env",
            settings.api_key_env,
            "--providers",
            settings.provider_request,
            "--results-per-query",
            str(settings.results_per_query),
            "--nonobviousness-original-fallback-enforce",
            "--nonobviousness-post-generation-enforce",
            "--post-generation-n10-authority-mode",
            settings.post_generation_n10_authority_mode,
        ]
    )
    return argv


def build_prospective_routed_execution_plan_v2(
    *,
    source_freeze: ProspectiveRoutedCampaignFreezeV2,
    source_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_file_sha256: str,
    regeneration_downstream_freeze: ProspectiveRegenerationDownstreamV2Freeze,
    regeneration_downstream_freeze_file_sha256: str,
    execution_plan_repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
    settings: RoutedExecutionSettingsV2 | None = None,
) -> ProspectiveRoutedExecutionPlanV2:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "prospective routed execution-v2 requires a clean tracked worktree"
        )

    expected = ["P16", "P17", "P18", "P19", "P20"]
    if source_freeze.case_ids != expected:
        raise ValueError("source routed v2 freeze is not P16-P20")

    if (
        source_freeze.source_regeneration_unit_freeze_id
        != regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("campaign/regeneration-unit freeze ID mismatch")
    if (
        source_freeze.source_regeneration_unit_freeze_sha256
        != regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("campaign/regeneration-unit freeze SHA mismatch")
    if (
        source_freeze.source_regeneration_downstream_freeze_id
        != regeneration_downstream_freeze.freeze_id
    ):
        raise ValueError("campaign/downstream freeze ID mismatch")
    if (
        source_freeze.source_regeneration_downstream_freeze_sha256
        != regeneration_downstream_freeze.freeze_sha256
    ):
        raise ValueError("campaign/downstream freeze SHA mismatch")

    if (
        regeneration_downstream_freeze.source_regeneration_unit_freeze_id
        != regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("downstream/regeneration-unit freeze ID mismatch")
    if (
        regeneration_downstream_freeze.source_regeneration_unit_freeze_sha256
        != regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("downstream/regeneration-unit freeze SHA mismatch")

    resolved = settings or RoutedExecutionSettingsV2()
    cases: list[ProspectiveRoutedCaseExecutionPlanV2] = []

    for task in source_freeze.tasks:
        run = Path(task.run_dir).expanduser().resolve()
        routed = run / "routed_preverifier_v2"
        lineage_template = (
            str(routed / "lineage")
            + "/{final_hypothesis_id_slug}/regeneration_v2"
        )

        cases.append(
            ProspectiveRoutedCaseExecutionPlanV2(
                case_id=task.case_id,
                source_task_id=task.task_id,
                source_task_sha256=task.task_sha256,
                run_dir=str(run),
                generation_model=task.generation_model,
                critic_model=task.critic_model,
                route_model=task.critic_model,
                route_audit_model=task.critic_model,
                regeneration_model=task.generation_model,
                initial_main_e2e_argv=_initial_main_e2e_argv(
                    task=task,
                    settings=resolved,
                ),
                hypothesis_context_path=str(
                    run / "hypothesis.context.json"
                ),
                full_binding_plan_path=str(
                    run / "relational_atomic_binding_plan.full.json"
                ),
                pre_route_gate_path=str(
                    routed / "pre_route_contract_gate_v2.json"
                ),
                routed_dispatch_path=str(
                    routed / "routed_dispatch.v2.json"
                ),
                routed_lineage_dir=str(routed / "lineage"),
                regeneration_lineage_dir_template=lineage_template,
                regeneration_unit_result_path_template=(
                    lineage_template + "/regeneration_unit_v2.result.json"
                ),
                regeneration_portfolio_path_template=(
                    lineage_template + "/regenerated.portfolio.json"
                ),
                regeneration_downstream_dir_template=(
                    lineage_template + "/downstream_v2"
                ),
                regeneration_downstream_report_path_template=(
                    lineage_template
                    + "/downstream_v2/downstream_v2.report.json"
                ),
                regeneration_argv=None,
                post_route_binding_plan_path=str(
                    routed / "relational_atomic_binding_plan.post_route.json"
                ),
                post_route_gate_path=str(
                    routed / "post_route_contract_gate_v2.json"
                ),
                selection_report_path=str(
                    routed / "structural_selection.post_route.json"
                ),
                selected_binding_plan_path=str(
                    routed
                    / "relational_atomic_binding_plan.selected.post_route.json"
                ),
                endpoint_binding_report_path=str(
                    routed
                    / "relational_atomic_endpoint_binding.post_route.json"
                ),
                endpoint_binding_prompt_path=str(
                    routed
                    / "relational_atomic_endpoint_binding.post_route.prompt.txt"
                ),
                endpoint_binding_telemetry_path=str(
                    routed
                    / "relational_atomic_endpoint_binding.post_route.telemetry.jsonl"
                ),
                projection_preflight_output_dir=str(
                    routed / "projection_preflight"
                ),
                relational_verifier_output_dir=str(
                    routed / "relational_scientific_verifier"
                ),
            )
        )

    body = {
        "schema_version": "prospective-routed-execution-plan-v2",
        "source_campaign_freeze_id": source_freeze.freeze_id,
        "source_campaign_freeze_sha256": source_freeze.freeze_sha256,
        "source_campaign_freeze_file_sha256": source_freeze_file_sha256,
        "source_campaign_freeze_repository_head_sha":
            source_freeze.repository_head_sha,
        "source_regeneration_unit_freeze_id":
            regeneration_unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256":
            regeneration_unit_freeze.freeze_sha256,
        "source_regeneration_unit_freeze_file_sha256":
            regeneration_unit_freeze_file_sha256,
        "source_regeneration_downstream_freeze_id":
            regeneration_downstream_freeze.freeze_id,
        "source_regeneration_downstream_freeze_sha256":
            regeneration_downstream_freeze.freeze_sha256,
        "source_regeneration_downstream_freeze_file_sha256":
            regeneration_downstream_freeze_file_sha256,
        "execution_plan_repository_head_sha":
            execution_plan_repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "settings": resolved.model_dump(mode="json"),
        "route_protocols": [
            row.model_dump(mode="json")
            for row in default_route_protocols_v2()
        ],
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": expected,
        "case_count": 5,
        "all_case_settings_frozen_before_first_case_execution": True,
        "all_route_protocols_frozen_before_first_case_execution": True,
        "regeneration_unit_contract_frozen_before_first_case_execution": True,
        "regeneration_downstream_contract_frozen_before_first_case_execution":
            True,
        "case_order_fixed_p16_to_p20": True,
        "no_regeneration_full_e2e_argv_in_plan": True,
        "initial_full_e2e_is_not_regeneration": True,
        "gate_v2_results_observed_before_plan_freeze": False,
        "repair_results_observed_before_plan_freeze": False,
        "regeneration_results_observed_before_plan_freeze": False,
        "endpoint_results_observed_before_plan_freeze": False,
        "verifier_results_observed_before_plan_freeze": False,
        "post_case_adaptation_allowed": False,
        "case_replacement_allowed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRoutedExecutionPlanV2(
        **body,
        plan_id="prospective_routed_execution_plan_v2:" + digest[:20],
        plan_sha256=digest,
    )


__all__ = [
    "ProspectiveRoutedCaseExecutionPlanV2",
    "ProspectiveRoutedExecutionPlanV2",
    "RouteActionV2",
    "RoutedExecutionSettingsV2",
    "RoutedStageProtocolV2",
    "build_prospective_routed_execution_plan_v2",
    "default_route_protocols_v2",
    "sha256_file",
]
