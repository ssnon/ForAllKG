from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_pre_n10_admissibility_shadow import (
    assess_atomic_pre_n10_shadow_row,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.legacy_atomic_admissibility_adapter import (
    adapt_legacy_atomic_specification_shadow_row,
)
from pipeline_core.discovery.legacy_atomic_specification_shadow import (
    compile_legacy_atomic_specification_shadow_row,
)
from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
    PreN10ProspectiveCampaignReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    PreN10RegenerationReentryReportV2,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
)
from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    ProspectiveAtomicAdmissibilityFreezeV5,
    default_p29_p33_tasks,
)
from pipeline_core.discovery.prospective_atomic_admissibility_v6 import (
    ProspectiveAtomicAdmissibilityFreezeV6,
    default_p34_p38_tasks,
)
from pipeline_core.discovery.prospective_atomic_admissibility_comparison_collector_v5 import (
    _card_index,
    _claim_index,
    _contract_claim_index,
    _load_json_object,
    _reconstruct_draft,
    _slug,
    classify_pair,
    classify_reason_transition,
    is_terminal_before_initial_vpre_status,
)
from pipeline_core.discovery.prospective_authority_campaign_freeze_v3 import (
    ProspectiveAuthorityModelPolicyV3,
    default_p21_p25_tasks,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    default_p26_p28_tasks,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze_v2 import (
    default_p16_p20_tasks,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CaseIdV7 = Literal["P39", "P40", "P41", "P42", "P43"]
DesignAxisV7 = Literal[
    "environmental_acidity",
    "thermal_transport",
    "chiral_response",
    "molecular_transport",
    "interfacial_charge_transfer",
]
ComparisonStageV7 = Literal["INITIAL_VPRE", "REGENERATION_REENTRY"]
ComparisonPairV7 = Literal[
    "LEGACY_READY__NEUTRAL_READY",
    "LEGACY_READY__NEUTRAL_NOT_READY",
    "LEGACY_NOT_READY__NEUTRAL_READY",
    "LEGACY_NOT_READY__NEUTRAL_NOT_READY",
]
CaseAccountingStatusV7 = Literal[
    "CLAIMS_COLLECTED",
    "TERMINAL_BEFORE_INITIAL_VPRE",
    "NO_CLAIMS_OBSERVED",
]


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


class ProspectiveAtomicAdmissibilityTaskV7(StrictModel):
    case_id: CaseIdV7
    design_axis: DesignAxisV7
    relation_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    stop: str | None = None
    target: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: Literal["explain_connection"] = "explain_connection"


class ProspectiveAtomicAdmissibilitySpecV7(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-spec-v7"
    ] = "prospective-atomic-admissibility-spec-v7"

    campaign_name: str = Field(min_length=1)
    campaign_root: str = Field(min_length=1)

    domain_profile_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    data_root: str = Field(min_length=1)
    semantic_roots: list[str] = Field(min_length=1)
    model_policy: ProspectiveAuthorityModelPolicyV3
    tasks: list[ProspectiveAtomicAdmissibilityTaskV7] = Field(
        min_length=5,
        max_length=5,
    )

    infrastructure_source_freeze_id: str = Field(min_length=1)
    infrastructure_source_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    prospective_goal: Literal[
        "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10"
    ] = "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10"

    tasks_defined_before_p39_p43_execution: Literal[True] = True
    expected_legacy_outcomes_predeclared: Literal[False] = False
    expected_neutral_outcomes_predeclared: Literal[False] = False
    expected_source_reference_outcomes_predeclared: Literal[False] = False
    expected_semantic_fidelity_outcomes_predeclared: Literal[False] = False

    prior_outcome_artifacts_consumed_by_builder: Literal[False] = False
    prior_task_definitions_used_only_for_nonoverlap_guard: Literal[True] = True
    infrastructure_and_model_policy_only_inherited_from_v6: Literal[True] = True

    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_spec(self) -> "ProspectiveAtomicAdmissibilitySpecV7":
        expected_cases = ["P39", "P40", "P41", "P42", "P43"]
        expected_axes = [
            "environmental_acidity",
            "thermal_transport",
            "chiral_response",
            "molecular_transport",
            "interfacial_charge_transfer",
        ]
        if [row.case_id for row in self.tasks] != expected_cases:
            raise ValueError("v7 source tasks must be exactly P39-P43")
        if [row.design_axis for row in self.tasks] != expected_axes:
            raise ValueError("v7 design-axis order mismatch")
        families = [row.relation_family for row in self.tasks]
        if len(families) != len(set(families)):
            raise ValueError("v7 relation families must be unique")
        surfaces = [
            (
                row.source.casefold().strip(),
                (row.stop or "").casefold().strip(),
                row.target.casefold().strip(),
                row.question.casefold().strip(),
            )
            for row in self.tasks
        ]
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("v7 source-task surfaces must be unique")
        if len(set(self.semantic_roots)) != len(self.semantic_roots):
            raise ValueError("semantic_roots must be unique")
        return self


class FrozenProspectiveAtomicAdmissibilityTaskV7(StrictModel):
    case_id: CaseIdV7
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    design_axis: DesignAxisV7
    relation_family: str
    source: str
    stop: str | None = None
    target: str
    question: str
    objective: Literal["explain_connection"]

    domain_profile_id: str
    corpus_id: str
    data_root: str
    semantic_roots: list[str]
    model_policy: ProspectiveAuthorityModelPolicyV3

    source_task_definition_only: Literal[True] = True
    hypothesis_content_predeclared: Literal[False] = False
    semantic_outcome_predeclared: Literal[False] = False
    legacy_vpre_outcome_predeclared: Literal[False] = False
    neutral_pre_n10_outcome_predeclared: Literal[False] = False

    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_task(
        self,
    ) -> "FrozenProspectiveAtomicAdmissibilityTaskV7":
        body = self.model_dump(mode="json")
        observed_id = body.pop("task_id")
        observed_sha = body.pop("task_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v7 task SHA mismatch")
        if observed_id != (
            "prospective_atomic_admissibility_v7_task:"
            + expected_sha[:20]
        ):
            raise ValueError("v7 task ID mismatch")
        return self


class ProspectiveAtomicAdmissibilityFreezeV7(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-freeze-v7"
    ] = "prospective-atomic-admissibility-freeze-v7"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    campaign_name: str
    campaign_root: str
    repository_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_worktree_dirty: Literal[False] = False

    infrastructure_source_freeze_id: str
    infrastructure_source_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    infrastructure_source_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_unit_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    tasks: list[FrozenProspectiveAtomicAdmissibilityTaskV7]
    case_ids: list[str]
    task_count: Literal[5] = 5
    design_axis_counts: dict[str, int]
    downstream_campaign_stage_order: list[str]

    source_tasks_frozen_before_p39_p43_execution: Literal[True] = True
    model_policy_frozen_before_p39_p43_execution: Literal[True] = True
    p39_p43_outputs_observed_before_freeze: Literal[False] = False

    prior_outcome_artifacts_consumed_by_freeze_builder: Literal[False] = False
    prior_task_definitions_used_only_for_nonoverlap_guard: Literal[True] = True

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
    def validate_freeze(
        self,
    ) -> "ProspectiveAtomicAdmissibilityFreezeV7":
        expected = ["P39", "P40", "P41", "P42", "P43"]
        if self.case_ids != expected:
            raise ValueError("v7 freeze case IDs must be P39-P43")
        if [row.case_id for row in self.tasks] != expected:
            raise ValueError("v7 frozen task order must be P39-P43")
        if len({row.task_id for row in self.tasks}) != 5:
            raise ValueError("v7 frozen task IDs must be unique")
        if self.downstream_campaign_stage_order != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError("v7 downstream stage order mismatch")
        expected_axis_counts = Counter(row.design_axis for row in self.tasks)
        if dict(sorted(expected_axis_counts.items())) != dict(
            sorted(self.design_axis_counts.items())
        ):
            raise ValueError("v7 design-axis counts mismatch")
        if any(count != 1 for count in self.design_axis_counts.values()):
            raise ValueError("each v7 design axis must appear once")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v7 freeze SHA mismatch")
        if observed_id != (
            "prospective_atomic_admissibility_freeze_v7:"
            + expected_sha[:20]
        ):
            raise ValueError("v7 freeze ID mismatch")
        return self


class ProspectiveAtomicAdmissibilityCaseExecutionPlanV7(StrictModel):
    case_id: CaseIdV7
    source_task_id: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    initial_e2e_argv: list[str]

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

    comparison_collector_contract_required_before_execution: Literal[
        True
    ] = True
    execution_authority_granted_by_this_plan: Literal[False] = False

    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(
        self,
    ) -> "ProspectiveAtomicAdmissibilityCaseExecutionPlanV7":
        if self.initial_e2e_argv[:3] != [
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
        ]:
            raise ValueError("v7 initial argv uses unexpected runner")
        if "--stop-after-initial-semantic" not in self.initial_e2e_argv:
            raise ValueError("v7 initial argv lacks semantic cut point")
        if "--overwrite-run" in self.initial_e2e_argv:
            raise ValueError("v7 initial argv must not overwrite run")
        if self.downstream_campaign_argv_base[:3] != [
            "python",
            "-m",
            "scripts.discovery.run_pre_n10_prospective_campaign_v1",
        ]:
            raise ValueError("v7 downstream argv uses unexpected runner")
        if "--semantic-review" in self.downstream_campaign_argv_base:
            raise ValueError("v7 downstream base argv must defer review")
        if "--allow-dirty-worktree" in self.downstream_campaign_argv_base:
            raise ValueError("v7 downstream argv must not allow dirty worktree")
        if "--save-prompts" not in self.downstream_campaign_argv_base:
            raise ValueError("v7 downstream argv must preserve prompts")
        return self


class ProspectiveAtomicAdmissibilityExecutionPlanV7(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-execution-plan-v7"
    ] = "prospective-atomic-admissibility-execution-plan-v7"

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
    repository_worktree_dirty: Literal[False] = False

    settings: ProspectiveAuthorityExecutionSettingsV3
    cases: list[ProspectiveAtomicAdmissibilityCaseExecutionPlanV7]
    case_ids: list[str]
    case_count: Literal[5] = 5

    source_tasks_and_models_frozen_before_execution: Literal[True] = True
    p39_p43_outputs_observed_before_plan_freeze: Literal[False] = False

    comparison_collector_contract_required_before_execution: Literal[
        True
    ] = True
    comparison_collector_contract_frozen_by_this_plan: Literal[False] = False
    execution_authority_granted_by_this_plan: Literal[False] = False

    case_order_fixed_p39_to_p43: Literal[True] = True
    later_case_adaptation_allowed: Literal[False] = False
    failed_or_terminal_case_replacement_allowed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False

    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(
        self,
    ) -> "ProspectiveAtomicAdmissibilityExecutionPlanV7":
        expected = ["P39", "P40", "P41", "P42", "P43"]
        if self.case_ids != expected:
            raise ValueError("v7 execution case IDs must be P39-P43")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("v7 execution cases must be ordered P39-P43")
        if len({row.source_task_id for row in self.cases}) != 5:
            raise ValueError("v7 execution task IDs must be unique")
        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v7 execution-plan SHA mismatch")
        if observed_id != (
            "prospective_atomic_admissibility_execution_plan_v7:"
            + expected_sha[:20]
        ):
            raise ValueError("v7 execution-plan ID mismatch")
        return self


class ProspectiveAtomicAdmissibilityComparisonCaseContractV7(StrictModel):
    case_id: CaseIdV7
    source_task_id: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    campaign_root: str
    campaign_report_path: str
    initial_portfolio_path: str
    initial_vpre_query_path: str
    initial_vpre_contract_path: str
    initial_vpre_audit_path: str
    regeneration_reentry_report_path: str
    regeneration_lineage_root: str

    terminal_case_policy: Literal[
        "RETAIN_CASE_WITH_ZERO_CLAIM_ROWS"
    ] = "RETAIN_CASE_WITH_ZERO_CLAIM_ROWS"

    @model_validator(mode="after")
    def validate_paths(
        self,
    ) -> "ProspectiveAtomicAdmissibilityComparisonCaseContractV7":
        campaign = Path(self.campaign_root).expanduser().resolve()
        if Path(self.campaign_report_path).expanduser().resolve() != (
            campaign / "campaign.report.json"
        ):
            raise ValueError("v7 campaign report path mismatch")
        initial = campaign / "01_initial_vpre"
        if Path(self.initial_vpre_query_path).expanduser().resolve() != (
            initial / "claims_queries.json"
        ):
            raise ValueError("v7 initial query path mismatch")
        if Path(self.initial_vpre_contract_path).expanduser().resolve() != (
            initial / "contract.report.json"
        ):
            raise ValueError("v7 initial contract path mismatch")
        if Path(self.initial_vpre_audit_path).expanduser().resolve() != (
            initial / "claim_decomposition.sanitization_audit.json"
        ):
            raise ValueError("v7 initial audit path mismatch")
        reentry = campaign / "04_regeneration_reentry"
        if Path(self.regeneration_reentry_report_path).expanduser().resolve() != (
            reentry / "reentry_v2.report.json"
        ):
            raise ValueError("v7 reentry report path mismatch")
        if Path(self.regeneration_lineage_root).expanduser().resolve() != (
            reentry / "lineage"
        ):
            raise ValueError("v7 lineage root mismatch")
        return self


class ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV7(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-comparison-collector-freeze-v7"
    ] = "prospective-atomic-admissibility-comparison-collector-freeze-v7"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_execution_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    collector_repository_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_worktree_dirty: Literal[False] = False

    cases: list[ProspectiveAtomicAdmissibilityComparisonCaseContractV7]
    case_ids: list[str]
    case_count: Literal[5] = 5
    comparison_output_path: str

    case_denominator_fixed_before_execution: Literal[True] = True
    terminal_before_decomposition_retained_in_case_denominator: Literal[
        True
    ] = True
    result_conditioned_artifact_selection_allowed: Literal[False] = False
    audit_record_subset_selection_allowed: Literal[False] = False

    collector_frozen_before_p39_p43_execution: Literal[True] = True
    p39_p43_outputs_observed_before_collector_freeze: Literal[False] = False
    campaign_execution_may_begin_after_this_freeze: Literal[True] = True

    llm_calls_allowed: Literal[False] = False
    artifact_mutation_allowed: Literal[False] = False
    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV7":
        expected = ["P39", "P40", "P41", "P42", "P43"]
        if self.case_ids != expected:
            raise ValueError("v7 collector case IDs must be P39-P43")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("v7 collector cases must be ordered P39-P43")
        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v7 collector freeze SHA mismatch")
        if observed_id != (
            "prospective_atomic_admissibility_comparison_collector_freeze_v7:"
            + expected_sha[:20]
        ):
            raise ValueError("v7 collector freeze ID mismatch")
        return self


class ProspectiveAtomicAdmissibilityClaimComparisonV7(StrictModel):
    case_id: CaseIdV7
    stage: ComparisonStageV7
    source_hypothesis_id: str | None = None

    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    kind: str
    novelty_selection_role: str | None = None

    pair: ComparisonPairV7
    reason_transition: str

    legacy_ready: bool
    legacy_contract_status: str
    legacy_binding_reason_codes: list[str]
    legacy_source_reason_codes: list[str]

    neutral_ready: bool
    neutral_blocking_dimensions: list[str]
    neutral_source_reference_status: str
    neutral_semantic_fidelity_status: str
    neutral_specification_status: str
    neutral_atomic_kind_status: str
    neutral_structural_compilation_status: str

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False


class ProspectiveAtomicAdmissibilityCaseAccountingV7(StrictModel):
    case_id: CaseIdV7
    case_in_denominator: Literal[True] = True
    campaign_final_status: str
    accounting_status: CaseAccountingStatusV7
    initial_claim_comparison_count: int = Field(ge=0)
    regeneration_claim_comparison_count: int = Field(ge=0)
    total_claim_comparison_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "ProspectiveAtomicAdmissibilityCaseAccountingV7":
        if self.total_claim_comparison_count != (
            self.initial_claim_comparison_count
            + self.regeneration_claim_comparison_count
        ):
            raise ValueError("v7 case comparison count mismatch")
        if self.accounting_status == "TERMINAL_BEFORE_INITIAL_VPRE":
            if self.total_claim_comparison_count != 0:
                raise ValueError(
                    "v7 terminal case cannot carry claim comparisons"
                )
        return self


class ProspectiveAtomicAdmissibilityComparisonReportV7(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-comparison-report-v7"
    ] = "prospective-atomic-admissibility-comparison-report-v7"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_collector_freeze_id: str
    source_collector_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_ids: list[str]
    case_count: Literal[5] = 5
    terminal_before_initial_vpre_case_count: int = Field(ge=0)
    claim_comparison_count: int = Field(ge=0)

    pair_counts: dict[str, int]
    reason_transition_counts: dict[str, int]
    neutral_blocker_dimension_counts: dict[str, int]

    cases: list[ProspectiveAtomicAdmissibilityCaseAccountingV7]
    claims: list[ProspectiveAtomicAdmissibilityClaimComparisonV7]

    case_denominator_preserved: Literal[True] = True
    claim_subset_selection_performed: Literal[False] = False
    llm_calls_performed: Literal[False] = False
    campaign_artifacts_mutated: Literal[False] = False

    diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ProspectiveAtomicAdmissibilityComparisonReportV7":
        expected = ["P39", "P40", "P41", "P42", "P43"]
        if self.case_ids != expected:
            raise ValueError("v7 report case IDs must be P39-P43")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("v7 report cases must be ordered P39-P43")
        if self.claim_comparison_count != len(self.claims):
            raise ValueError("v7 report claim count mismatch")
        if sum(row.total_claim_comparison_count for row in self.cases) != len(
            self.claims
        ):
            raise ValueError("v7 case/claim accounting mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v7 comparison report SHA mismatch")
        if observed_id != (
            "prospective_atomic_admissibility_comparison_report_v7:"
            + expected_sha[:20]
        ):
            raise ValueError("v7 comparison report ID mismatch")
        return self


def default_p39_p43_tasks() -> list[ProspectiveAtomicAdmissibilityTaskV7]:
    rows = [
        (
            "P39",
            "environmental_acidity",
            "solution_ph_protonation_sensitive_sers_band_ratio",
            "solution pH",
            "protonation-sensitive SERS band intensity ratio",
            (
                "How does solution pH relate to the intensity ratio of "
                "protonation-sensitive Raman bands in SERS spectra?"
            ),
        ),
        (
            "P40",
            "thermal_transport",
            "substrate_thermal_conductivity_hotspot_temperature_rise",
            "substrate thermal conductivity",
            "laser-induced local temperature rise at plasmonic hotspots",
            (
                "How does substrate thermal conductivity relate to "
                "laser-induced local temperature rise at plasmonic hotspots "
                "during SERS excitation?"
            ),
        ),
        (
            "P41",
            "chiral_response",
            "nanostructure_handedness_circular_sers_asymmetry",
            "plasmonic nanostructure handedness",
            "circular-polarization SERS intensity asymmetry",
            (
                "How does plasmonic nanostructure handedness relate to "
                "circular-polarization SERS intensity asymmetry?"
            ),
        ),
        (
            "P42",
            "molecular_transport",
            "analyte_diffusion_coefficient_sers_response_time",
            "analyte diffusion coefficient",
            "SERS signal response time",
            (
                "How does analyte diffusion coefficient relate to SERS signal "
                "response time in flow-through or microfluidic SERS substrates?"
            ),
        ),
        (
            "P43",
            "interfacial_charge_transfer",
            "metal_semiconductor_contact_area_charge_transfer_timescale",
            "metal-semiconductor interfacial contact area",
            "plasmon-induced charge-transfer timescale",
            (
                "How does metal-semiconductor interfacial contact area relate "
                "to plasmon-induced charge-transfer timescale in hybrid SERS "
                "nanostructures?"
            ),
        ),
    ]
    return [
        ProspectiveAtomicAdmissibilityTaskV7(
            case_id=case_id,
            design_axis=axis,
            relation_family=family,
            source=source,
            target=target,
            question=question,
        )
        for case_id, axis, family, source, target, question in rows
    ]


def _prior_task_families_and_surfaces() -> tuple[set[str], set[tuple[str, ...]]]:
    tasks = [
        *default_p16_p20_tasks(),
        *default_p21_p25_tasks(),
        *default_p26_p28_tasks(),
        *default_p29_p33_tasks(),
        *default_p34_p38_tasks(),
    ]
    families = {row.relation_family for row in tasks}
    surfaces = {
        (
            row.source.casefold().strip(),
            str(getattr(row, "stop", "") or "").casefold().strip(),
            row.target.casefold().strip(),
            row.question.casefold().strip(),
        )
        for row in tasks
    }
    return families, surfaces


def build_p39_p43_spec_from_v6_infrastructure(
    *,
    infrastructure_freeze: ProspectiveAtomicAdmissibilityFreezeV6,
    campaign_root: str,
) -> ProspectiveAtomicAdmissibilitySpecV7:
    if infrastructure_freeze.case_ids != ["P34", "P35", "P36", "P37", "P38"]:
        raise ValueError("v7 infrastructure source must be P34-P38 v6")
    if len(infrastructure_freeze.tasks) != 5:
        raise ValueError("v7 infrastructure source must contain five tasks")

    first = infrastructure_freeze.tasks[0]
    invariant_fields = (
        "domain_profile_id",
        "corpus_id",
        "data_root",
        "semantic_roots",
        "model_policy",
    )
    for task in infrastructure_freeze.tasks[1:]:
        for field in invariant_fields:
            if getattr(task, field) != getattr(first, field):
                raise ValueError("v7 infrastructure is non-uniform for " + field)

    tasks = default_p39_p43_tasks()
    prior_families, prior_surfaces = _prior_task_families_and_surfaces()
    new_families = {row.relation_family for row in tasks}
    new_surfaces = {
        (
            row.source.casefold().strip(),
            (row.stop or "").casefold().strip(),
            row.target.casefold().strip(),
            row.question.casefold().strip(),
        )
        for row in tasks
    }
    overlap = sorted(prior_families & new_families)
    if overlap:
        raise ValueError("v7 relation families overlap prior cohorts: " + repr(overlap))
    if prior_surfaces & new_surfaces:
        raise ValueError("v7 task surfaces overlap prior cohorts")

    return ProspectiveAtomicAdmissibilitySpecV7(
        campaign_name="P39_P43_atomic_admissibility_v7",
        campaign_root=str(Path(campaign_root).expanduser().resolve()),
        domain_profile_id=first.domain_profile_id,
        corpus_id=first.corpus_id,
        data_root=first.data_root,
        semantic_roots=list(first.semantic_roots),
        model_policy=first.model_policy,
        tasks=tasks,
        infrastructure_source_freeze_id=infrastructure_freeze.freeze_id,
        infrastructure_source_freeze_sha256=infrastructure_freeze.freeze_sha256,
    )


def build_prospective_atomic_admissibility_freeze_v7(
    *,
    spec: ProspectiveAtomicAdmissibilitySpecV7,
    source_spec_sha256: str,
    infrastructure_freeze: ProspectiveAtomicAdmissibilityFreezeV6,
    infrastructure_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_file_sha256: str,
    repository_head_sha: str,
    repository_worktree_dirty: bool,
) -> ProspectiveAtomicAdmissibilityFreezeV7:
    if repository_worktree_dirty:
        raise ValueError("v7 freeze requires clean worktree including untracked files")
    if infrastructure_freeze.freeze_id != spec.infrastructure_source_freeze_id:
        raise ValueError("v7 spec/infrastructure ID mismatch")
    if infrastructure_freeze.freeze_sha256 != spec.infrastructure_source_freeze_sha256:
        raise ValueError("v7 spec/infrastructure SHA mismatch")
    if (
        infrastructure_freeze.source_regeneration_unit_freeze_id
        != regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("v7 regeneration-unit ID mismatch")
    if (
        infrastructure_freeze.source_regeneration_unit_freeze_sha256
        != regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("v7 regeneration-unit SHA mismatch")

    root = Path(spec.campaign_root).expanduser().resolve()
    tasks = []
    for row in spec.tasks:
        body = {
            "case_id": row.case_id,
            "run_dir": str(root / row.case_id),
            "design_axis": row.design_axis,
            "relation_family": row.relation_family,
            "source": row.source,
            "stop": row.stop,
            "target": row.target,
            "question": row.question,
            "objective": row.objective,
            "domain_profile_id": spec.domain_profile_id,
            "corpus_id": spec.corpus_id,
            "data_root": spec.data_root,
            "semantic_roots": list(spec.semantic_roots),
            "model_policy": spec.model_policy.model_dump(mode="json"),
            "source_task_definition_only": True,
            "hypothesis_content_predeclared": False,
            "semantic_outcome_predeclared": False,
            "legacy_vpre_outcome_predeclared": False,
            "neutral_pre_n10_outcome_predeclared": False,
            "engineering_diagnostic_only": True,
            "scientific_validation_authority": False,
        }
        digest = _sha256_json(body)
        tasks.append(
            FrozenProspectiveAtomicAdmissibilityTaskV7(
                **body,
                task_id=(
                    "prospective_atomic_admissibility_v7_task:"
                    + digest[:20]
                ),
                task_sha256=digest,
            )
        )

    body = {
        "schema_version": "prospective-atomic-admissibility-freeze-v7",
        "source_spec_sha256": source_spec_sha256,
        "campaign_name": spec.campaign_name,
        "campaign_root": str(root),
        "repository_head_sha": repository_head_sha,
        "repository_worktree_dirty": False,
        "infrastructure_source_freeze_id": infrastructure_freeze.freeze_id,
        "infrastructure_source_freeze_sha256": infrastructure_freeze.freeze_sha256,
        "infrastructure_source_freeze_file_sha256": infrastructure_freeze_file_sha256,
        "source_regeneration_unit_freeze_id": regeneration_unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256": regeneration_unit_freeze.freeze_sha256,
        "source_regeneration_unit_freeze_file_sha256": regeneration_unit_freeze_file_sha256,
        "tasks": [row.model_dump(mode="json") for row in tasks],
        "case_ids": ["P39", "P40", "P41", "P42", "P43"],
        "task_count": 5,
        "design_axis_counts": dict(
            sorted(Counter(row.design_axis for row in tasks).items())
        ),
        "downstream_campaign_stage_order": list(CAMPAIGN_STAGE_ORDER),
        "source_tasks_frozen_before_p39_p43_execution": True,
        "model_policy_frozen_before_p39_p43_execution": True,
        "p39_p43_outputs_observed_before_freeze": False,
        "prior_outcome_artifacts_consumed_by_freeze_builder": False,
        "prior_task_definitions_used_only_for_nonoverlap_guard": True,
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
    return ProspectiveAtomicAdmissibilityFreezeV7(
        **body,
        freeze_id=(
            "prospective_atomic_admissibility_freeze_v7:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


def _initial_argv(task, settings: ProspectiveAuthorityExecutionSettingsV3) -> list[str]:
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


def _downstream_argv(
    task,
    regeneration_unit_freeze_path: Path,
    settings: ProspectiveAuthorityExecutionSettingsV3,
) -> tuple[list[str], dict[str, str]]:
    run = Path(task.run_dir).expanduser().resolve()
    policy = task.model_policy
    paths = {
        "portfolio": str(run / "hypothesis_axis_a4.portfolio.json"),
        "context": str(run / "hypothesis.context.json"),
        "semantic_run": str(run / "semantic_axis_a4.run.json"),
        "semantic_review": str(run / "semantic_axis_a4.review.json"),
        "provider_plan": str(run / "literature_provider_plan.json"),
        "campaign_root": str(run / "prospective_atomic_admissibility_v7"),
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


def materialize_downstream_argv_v7(
    case: ProspectiveAtomicAdmissibilityCaseExecutionPlanV7,
) -> list[str]:
    argv = list(case.downstream_campaign_argv_base)
    review = Path(case.initial_semantic_review_path)
    if review.is_file():
        argv += ["--semantic-review", str(review.expanduser().resolve())]
    return argv


def build_prospective_atomic_admissibility_execution_plan_v7(
    *,
    campaign_freeze: ProspectiveAtomicAdmissibilityFreezeV7,
    campaign_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_path: Path,
    regeneration_unit_freeze_file_sha256: str,
    execution_plan_repository_head_sha: str,
    repository_worktree_dirty: bool,
    settings: ProspectiveAuthorityExecutionSettingsV3 | None = None,
) -> ProspectiveAtomicAdmissibilityExecutionPlanV7:
    if repository_worktree_dirty:
        raise ValueError("v7 execution plan requires clean worktree including untracked files")
    if campaign_freeze.source_regeneration_unit_freeze_id != regeneration_unit_freeze.freeze_id:
        raise ValueError("v7 execution regeneration-unit ID mismatch")
    if campaign_freeze.source_regeneration_unit_freeze_sha256 != regeneration_unit_freeze.freeze_sha256:
        raise ValueError("v7 execution regeneration-unit SHA mismatch")
    if campaign_freeze.source_regeneration_unit_freeze_file_sha256 != regeneration_unit_freeze_file_sha256:
        raise ValueError("v7 execution regeneration-unit file SHA mismatch")

    resolved = settings or ProspectiveAuthorityExecutionSettingsV3()
    cases = []
    for task in campaign_freeze.tasks:
        downstream, paths = _downstream_argv(
            task,
            regeneration_unit_freeze_path,
            resolved,
        )
        cases.append(
            ProspectiveAtomicAdmissibilityCaseExecutionPlanV7(
                case_id=task.case_id,
                source_task_id=task.task_id,
                source_task_sha256=task.task_sha256,
                run_dir=str(Path(task.run_dir).expanduser().resolve()),
                initial_e2e_argv=_initial_argv(task, resolved),
                initial_portfolio_path=paths["portfolio"],
                initial_hypothesis_context_path=paths["context"],
                initial_semantic_run_path=paths["semantic_run"],
                initial_semantic_review_path=paths["semantic_review"],
                initial_provider_plan_path=paths["provider_plan"],
                downstream_campaign_output_root=paths["campaign_root"],
                downstream_campaign_argv_base=downstream,
            )
        )

    body = {
        "schema_version": "prospective-atomic-admissibility-execution-plan-v7",
        "source_campaign_freeze_id": campaign_freeze.freeze_id,
        "source_campaign_freeze_sha256": campaign_freeze.freeze_sha256,
        "source_campaign_freeze_file_sha256": campaign_freeze_file_sha256,
        "source_regeneration_unit_freeze_id": regeneration_unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256": regeneration_unit_freeze.freeze_sha256,
        "source_regeneration_unit_freeze_file_sha256": regeneration_unit_freeze_file_sha256,
        "execution_plan_repository_head_sha": execution_plan_repository_head_sha,
        "repository_worktree_dirty": False,
        "settings": resolved.model_dump(mode="json"),
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P39", "P40", "P41", "P42", "P43"],
        "case_count": 5,
        "source_tasks_and_models_frozen_before_execution": True,
        "p39_p43_outputs_observed_before_plan_freeze": False,
        "comparison_collector_contract_required_before_execution": True,
        "comparison_collector_contract_frozen_by_this_plan": False,
        "execution_authority_granted_by_this_plan": False,
        "case_order_fixed_p39_to_p43": True,
        "later_case_adaptation_allowed": False,
        "failed_or_terminal_case_replacement_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "second_regeneration_allowed": False,
        "engineering_diagnostic_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
    }
    digest = _sha256_json(body)
    return ProspectiveAtomicAdmissibilityExecutionPlanV7(
        **body,
        plan_id=(
            "prospective_atomic_admissibility_execution_plan_v7:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


def build_prospective_atomic_admissibility_comparison_collector_freeze_v7(
    *,
    execution_plan: ProspectiveAtomicAdmissibilityExecutionPlanV7,
    execution_plan_file_sha256: str,
    collector_repository_head_sha: str,
    repository_worktree_dirty: bool,
    comparison_output_path: Path,
) -> ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV7:
    if repository_worktree_dirty:
        raise ValueError("v7 collector freeze requires clean worktree including untracked files")
    cases = []
    for row in execution_plan.cases:
        campaign = Path(row.downstream_campaign_output_root).expanduser().resolve()
        initial = campaign / "01_initial_vpre"
        reentry = campaign / "04_regeneration_reentry"
        cases.append(
            ProspectiveAtomicAdmissibilityComparisonCaseContractV7(
                case_id=row.case_id,
                source_task_id=row.source_task_id,
                source_task_sha256=row.source_task_sha256,
                run_dir=str(Path(row.run_dir).expanduser().resolve()),
                campaign_root=str(campaign),
                campaign_report_path=str(campaign / "campaign.report.json"),
                initial_portfolio_path=str(Path(row.initial_portfolio_path).resolve()),
                initial_vpre_query_path=str(initial / "claims_queries.json"),
                initial_vpre_contract_path=str(initial / "contract.report.json"),
                initial_vpre_audit_path=str(
                    initial / "claim_decomposition.sanitization_audit.json"
                ),
                regeneration_reentry_report_path=str(
                    reentry / "reentry_v2.report.json"
                ),
                regeneration_lineage_root=str(reentry / "lineage"),
            )
        )

    body = {
        "schema_version": (
            "prospective-atomic-admissibility-comparison-collector-freeze-v7"
        ),
        "source_execution_plan_id": execution_plan.plan_id,
        "source_execution_plan_sha256": execution_plan.plan_sha256,
        "source_execution_plan_file_sha256": execution_plan_file_sha256,
        "collector_repository_head_sha": collector_repository_head_sha,
        "repository_worktree_dirty": False,
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P39", "P40", "P41", "P42", "P43"],
        "case_count": 5,
        "comparison_output_path": str(comparison_output_path.expanduser().resolve()),
        "case_denominator_fixed_before_execution": True,
        "terminal_before_decomposition_retained_in_case_denominator": True,
        "result_conditioned_artifact_selection_allowed": False,
        "audit_record_subset_selection_allowed": False,
        "collector_frozen_before_p39_p43_execution": True,
        "p39_p43_outputs_observed_before_collector_freeze": False,
        "campaign_execution_may_begin_after_this_freeze": True,
        "llm_calls_allowed": False,
        "artifact_mutation_allowed": False,
        "engineering_diagnostic_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
    }
    digest = _sha256_json(body)
    return ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV7(
        **body,
        freeze_id=(
            "prospective_atomic_admissibility_comparison_collector_freeze_v7:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


def _collect_claim_set_v7(
    *,
    case_id: CaseIdV7,
    stage: ComparisonStageV7,
    source_hypothesis_id: str | None,
    portfolio_path: Path,
    query_path: Path,
    contract_path: Path,
    audit_path: Path,
    expected_audit_schema: str,
) -> list[ProspectiveAtomicAdmissibilityClaimComparisonV7]:
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    query = LiteratureQueryPlan.model_validate_json(
        query_path.read_text(encoding="utf-8")
    )
    contract = PreN10ScientificContractReportV1.model_validate_json(
        contract_path.read_text(encoding="utf-8")
    )
    audit = _load_json_object(audit_path)

    if str(audit.get("schema_version") or "") != expected_audit_schema:
        raise ValueError("v7 unexpected audit schema")
    if audit.get("diagnostic_only") is not True:
        raise ValueError("v7 audit lost diagnostic-only boundary")

    claims = _claim_index(query)
    cards = _card_index(portfolio)
    frozen_claims = _contract_claim_index(contract)
    records = [row for row in (audit.get("records") or []) if isinstance(row, dict)]
    record_ids = [str(row.get("claim_id") or "").strip() for row in records]
    if set(record_ids) != set(claims) or set(record_ids) != set(frozen_claims):
        raise ValueError("v7 audit/claim population mismatch")

    claim_id_to_local_id = {
        str(row.get("claim_id")): str(row.get("claim_local_id") or "").strip()
        for row in records
    }
    rows = []
    for record in records:
        claim_id = str(record["claim_id"])
        hypothesis_id, canonical = claims[claim_id]
        card = cards[hypothesis_id]
        draft = _reconstruct_draft(
            canonical=canonical,
            record=record,
            claim_id_to_local_id=claim_id_to_local_id,
        )
        legacy_atomic = compile_legacy_atomic_specification_shadow_row(
            hypothesis=card,
            draft_claim=draft,
            canonical_claim=canonical,
        )
        frozen_fidelity = record.get("semantic_fidelity_shadow")
        frozen_fidelity = frozen_fidelity if isinstance(frozen_fidelity, dict) else {}
        if list(frozen_fidelity.get("reason_codes") or []) != list(
            legacy_atomic.semantic_fidelity_reason_codes
        ):
            raise ValueError("v7 semantic-fidelity replay drift")

        neutral = assess_atomic_pre_n10_shadow_row(
            adapt_legacy_atomic_specification_shadow_row(legacy_atomic)
        )
        legacy = frozen_claims[claim_id]
        legacy_ready = legacy.contract_status == "READY_FOR_N10_CONTRACT"
        neutral_ready = neutral.readiness_status == "READY_FOR_N10_SHADOW"
        blockers = list(neutral.blocking_dimensions)
        source_reasons = list(legacy.source_contract_reason_codes)

        rows.append(
            ProspectiveAtomicAdmissibilityClaimComparisonV7(
                case_id=case_id,
                stage=stage,
                source_hypothesis_id=source_hypothesis_id,
                hypothesis_id=hypothesis_id,
                claim_id=claim_id,
                claim_rank=canonical.claim_rank,
                kind=canonical.kind,
                novelty_selection_role=canonical.novelty_selection_role,
                pair=classify_pair(
                    legacy_ready=legacy_ready,
                    neutral_ready=neutral_ready,
                ),
                reason_transition=classify_reason_transition(
                    legacy_source_reasons=source_reasons,
                    neutral_source_reference_status=neutral.source_reference_status,
                    neutral_semantic_fidelity_status=neutral.semantic_fidelity_status,
                    neutral_blocking_dimensions=blockers,
                    neutral_ready=neutral_ready,
                ),
                legacy_ready=legacy_ready,
                legacy_contract_status=legacy.contract_status,
                legacy_binding_reason_codes=list(
                    legacy.binding_contract_reason_codes
                ),
                legacy_source_reason_codes=source_reasons,
                neutral_ready=neutral_ready,
                neutral_blocking_dimensions=blockers,
                neutral_source_reference_status=neutral.source_reference_status,
                neutral_semantic_fidelity_status=neutral.semantic_fidelity_status,
                neutral_specification_status=neutral.specification_status,
                neutral_atomic_kind_status=neutral.atomic_kind_status,
                neutral_structural_compilation_status=(
                    neutral.structural_compilation_status
                ),
            )
        )
    return rows


def collect_prospective_atomic_admissibility_comparison_v7(
    freeze: ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV7,
) -> ProspectiveAtomicAdmissibilityComparisonReportV7:
    all_claims = []
    case_rows = []

    for case in freeze.cases:
        campaign_path = Path(case.campaign_report_path)
        campaign = PreN10ProspectiveCampaignReportV1.model_validate_json(
            campaign_path.read_text(encoding="utf-8")
        )
        stage_map = {row.stage_name: row for row in campaign.stages}
        initial_stage = stage_map["initial_vpre"]
        reentry_stage = stage_map["regeneration_semantic_reentry"]

        initial_count = 0
        regen_count = 0

        if is_terminal_before_initial_vpre_status(campaign.final_status):
            if initial_stage.status != "SKIPPED_TERMINAL":
                raise ValueError("v7 terminal campaign has active initial V_pre")
            forbidden = [
                Path(case.initial_vpre_query_path),
                Path(case.initial_vpre_contract_path),
                Path(case.initial_vpre_audit_path),
            ]
            if any(path.exists() for path in forbidden):
                raise ValueError("v7 terminal campaign unexpectedly has V_pre artifacts")
            if Path(case.regeneration_reentry_report_path).exists():
                raise ValueError("v7 terminal campaign unexpectedly has reentry report")
            status: CaseAccountingStatusV7 = "TERMINAL_BEFORE_INITIAL_VPRE"
        else:
            if initial_stage.status not in {"EXECUTED", "REUSED_VALIDATED"}:
                raise ValueError("v7 nonterminal campaign lacks initial V_pre")
            initial_rows = _collect_claim_set_v7(
                case_id=case.case_id,
                stage="INITIAL_VPRE",
                source_hypothesis_id=None,
                portfolio_path=Path(case.initial_portfolio_path),
                query_path=Path(case.initial_vpre_query_path),
                contract_path=Path(case.initial_vpre_contract_path),
                audit_path=Path(case.initial_vpre_audit_path),
                expected_audit_schema=(
                    "pre-n10-specification-sanitization-audit-v1"
                ),
            )
            all_claims.extend(initial_rows)
            initial_count = len(initial_rows)

            expected_audits = set()
            reentry_path = Path(case.regeneration_reentry_report_path)
            if reentry_stage.status in {"EXECUTED", "REUSED_VALIDATED"}:
                reentry = PreN10RegenerationReentryReportV2.model_validate_json(
                    reentry_path.read_text(encoding="utf-8")
                )
                for lineage in reentry.lineages:
                    if lineage.claim_decomposition_request_count == 0:
                        continue
                    query_path = Path(lineage.query_plan_path).resolve()
                    expected_dir = (
                        Path(case.regeneration_lineage_root)
                        / _slug(lineage.source_hypothesis_id)
                    ).resolve()
                    if query_path.parent != expected_dir:
                        raise ValueError("v7 reentry lineage directory mismatch")
                    audit_path = (
                        expected_dir / "claim_decomposition.sanitization_audit.json"
                    )
                    expected_audits.add(audit_path.resolve())
                    rows = _collect_claim_set_v7(
                        case_id=case.case_id,
                        stage="REGENERATION_REENTRY",
                        source_hypothesis_id=lineage.source_hypothesis_id,
                        portfolio_path=Path(
                            lineage.regenerated_portfolio_path
                        ).resolve(),
                        query_path=query_path,
                        contract_path=Path(lineage.contract_report_path).resolve(),
                        audit_path=audit_path,
                        expected_audit_schema=(
                            "pre-n10-regeneration-decomposition-"
                            "sanitization-audit-v1"
                        ),
                    )
                    all_claims.extend(rows)
                    regen_count += len(rows)

                lineage_root = Path(case.regeneration_lineage_root)
                observed = (
                    {
                        path.resolve()
                        for path in lineage_root.glob(
                            "*/claim_decomposition.sanitization_audit.json"
                        )
                        if path.is_file()
                    }
                    if lineage_root.is_dir()
                    else set()
                )
                if observed != expected_audits:
                    raise ValueError("v7 reentry audit population drift")
            elif reentry_path.exists():
                raise ValueError("v7 skipped reentry unexpectedly has report")

            status = (
                "CLAIMS_COLLECTED"
                if initial_count + regen_count > 0
                else "NO_CLAIMS_OBSERVED"
            )

        case_rows.append(
            ProspectiveAtomicAdmissibilityCaseAccountingV7(
                case_id=case.case_id,
                campaign_final_status=campaign.final_status,
                accounting_status=status,
                initial_claim_comparison_count=initial_count,
                regeneration_claim_comparison_count=regen_count,
                total_claim_comparison_count=initial_count + regen_count,
            )
        )

    pair_counts = dict(sorted(Counter(row.pair for row in all_claims).items()))
    transition_counts = dict(
        sorted(Counter(row.reason_transition for row in all_claims).items())
    )
    blocker_counts = dict(
        sorted(
            Counter(
                blocker
                for row in all_claims
                for blocker in row.neutral_blocking_dimensions
            ).items()
        )
    )
    body = {
        "schema_version": "prospective-atomic-admissibility-comparison-report-v7",
        "source_collector_freeze_id": freeze.freeze_id,
        "source_collector_freeze_sha256": freeze.freeze_sha256,
        "case_ids": ["P39", "P40", "P41", "P42", "P43"],
        "case_count": 5,
        "terminal_before_initial_vpre_case_count": sum(
            row.accounting_status == "TERMINAL_BEFORE_INITIAL_VPRE"
            for row in case_rows
        ),
        "claim_comparison_count": len(all_claims),
        "pair_counts": pair_counts,
        "reason_transition_counts": transition_counts,
        "neutral_blocker_dimension_counts": blocker_counts,
        "cases": [row.model_dump(mode="json") for row in case_rows],
        "claims": [row.model_dump(mode="json") for row in all_claims],
        "case_denominator_preserved": True,
        "claim_subset_selection_performed": False,
        "llm_calls_performed": False,
        "campaign_artifacts_mutated": False,
        "diagnostic_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
    }
    digest = _sha256_json(body)
    return ProspectiveAtomicAdmissibilityComparisonReportV7(
        **body,
        report_id=(
            "prospective_atomic_admissibility_comparison_report_v7:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV7",
    "ProspectiveAtomicAdmissibilityExecutionPlanV7",
    "ProspectiveAtomicAdmissibilityFreezeV7",
    "ProspectiveAtomicAdmissibilitySpecV7",
    "build_p39_p43_spec_from_v6_infrastructure",
    "build_prospective_atomic_admissibility_comparison_collector_freeze_v7",
    "build_prospective_atomic_admissibility_execution_plan_v7",
    "build_prospective_atomic_admissibility_freeze_v7",
    "collect_prospective_atomic_admissibility_comparison_v7",
    "default_p39_p43_tasks",
    "materialize_downstream_argv_v7",
    "sha256_file",
]
