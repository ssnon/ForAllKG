from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    AtomicScientificSourceBindingBundle,
)
from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
    PreN10CanonicalSourceReferenceReportV1,
)
from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    PreN10DownstreamHandoffReportV1,
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    PreN10PrimaryRouterReportV1,
)
from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
    PreN10ProspectiveCampaignReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    PreN10RegenerationReentryReportV2,
)
from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    PreN10RelationalBindingBridgeReportV1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    PreN10ScientificContractReportV2,
)
from pipeline_core.discovery.pre_n10_vpost_shadow_v1 import (
    PreN10VPostShadowPlanV1,
    PreN10VPostShadowReportV1,
)
from pipeline_core.discovery.prospective_atomic_admissibility_v7 import (
    ProspectiveAtomicAdmissibilityFreezeV7,
    default_p39_p43_tasks,
)
from pipeline_core.discovery.prospective_atomic_admissibility_v6 import (
    default_p34_p38_tasks,
)
from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    default_p29_p33_tasks,
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


CaseIdV8 = Literal["P44", "P45", "P46", "P47", "P48"]
DesignAxisV8 = Literal[
    "illumination_geometry",
    "vertical_gap_mode",
    "surface_redox_state",
    "thermal_desorption",
    "microfluidic_transport",
]
CaseAccountingStatusV8 = Literal[
    "TERMINAL_BEFORE_INITIAL_VPRE",
    "PRE_N10_TERMINAL",
    "CANONICAL_CHAIN_TO_EXTERNAL",
    "CANONICAL_CHAIN_TO_VPOST",
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


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProspectiveCanonicalValidationTaskV8(StrictModel):
    case_id: CaseIdV8
    design_axis: DesignAxisV8
    relation_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    stop: str | None = None
    target: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: Literal["explain_connection"] = "explain_connection"


class ProspectiveCanonicalValidationSpecV8(StrictModel):
    schema_version: Literal[
        "prospective-canonical-validation-spec-v8"
    ] = "prospective-canonical-validation-spec-v8"

    campaign_name: str
    campaign_root: str
    domain_profile_id: str
    corpus_id: str
    data_root: str
    semantic_roots: list[str] = Field(min_length=1)
    model_policy: ProspectiveAuthorityModelPolicyV3
    tasks: list[ProspectiveCanonicalValidationTaskV8] = Field(
        min_length=5,
        max_length=5,
    )

    infrastructure_source_freeze_id: str
    infrastructure_source_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    prospective_goal: Literal[
        "CANONICAL_STABLE_ID_END_TO_END_INTEGRITY"
    ] = "CANONICAL_STABLE_ID_END_TO_END_INTEGRITY"

    tasks_defined_before_p44_p48_execution: Literal[True] = True
    prior_outcome_artifacts_consumed_by_builder: Literal[False] = False
    prior_task_definitions_used_only_for_nonoverlap_guard: Literal[True] = True
    infrastructure_and_model_policy_only_inherited_from_v7: Literal[
        True
    ] = True

    engineering_validation_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_spec(self) -> "ProspectiveCanonicalValidationSpecV8":
        expected_cases = ["P44", "P45", "P46", "P47", "P48"]
        expected_axes = [
            "illumination_geometry",
            "vertical_gap_mode",
            "surface_redox_state",
            "thermal_desorption",
            "microfluidic_transport",
        ]
        if [row.case_id for row in self.tasks] != expected_cases:
            raise ValueError("v8 source tasks must be exactly P44-P48")
        if [row.design_axis for row in self.tasks] != expected_axes:
            raise ValueError("v8 design-axis order mismatch")
        families = [row.relation_family for row in self.tasks]
        if len(families) != len(set(families)):
            raise ValueError("v8 relation families must be unique")
        if len(set(self.semantic_roots)) != len(self.semantic_roots):
            raise ValueError("v8 semantic_roots must be unique")
        return self


class FrozenProspectiveCanonicalValidationTaskV8(StrictModel):
    case_id: CaseIdV8
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    design_axis: DesignAxisV8
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
    scientific_outcome_predeclared: Literal[False] = False
    canonical_integrity_outcome_predeclared: Literal[False] = False
    engineering_validation_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_task(
        self,
    ) -> "FrozenProspectiveCanonicalValidationTaskV8":
        body = self.model_dump(mode="json")
        observed_id = body.pop("task_id")
        observed_sha = body.pop("task_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v8 task SHA mismatch")
        if observed_id != (
            "prospective_canonical_validation_v8_task:"
            + expected_sha[:20]
        ):
            raise ValueError("v8 task ID mismatch")
        return self


class ProspectiveCanonicalValidationFreezeV8(StrictModel):
    schema_version: Literal[
        "prospective-canonical-validation-freeze-v8"
    ] = "prospective-canonical-validation-freeze-v8"

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

    tasks: list[FrozenProspectiveCanonicalValidationTaskV8]
    case_ids: list[str]
    task_count: Literal[5] = 5
    design_axis_counts: dict[str, int]
    downstream_campaign_stage_order: list[str]

    source_tasks_frozen_before_p44_p48_execution: Literal[True] = True
    model_policy_frozen_before_p44_p48_execution: Literal[True] = True
    p44_p48_outputs_observed_before_freeze: Literal[False] = False
    prior_outcome_artifacts_consumed_by_freeze_builder: Literal[False] = False

    later_case_adaptation_allowed: Literal[False] = False
    failed_or_terminal_case_replacement_allowed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False

    prospective_evidence_collection: Literal[True] = True
    engineering_validation_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "ProspectiveCanonicalValidationFreezeV8":
        expected = ["P44", "P45", "P46", "P47", "P48"]
        if self.case_ids != expected:
            raise ValueError("v8 freeze case IDs must be P44-P48")
        if [row.case_id for row in self.tasks] != expected:
            raise ValueError("v8 frozen task order must be P44-P48")
        if len({row.task_id for row in self.tasks}) != 5:
            raise ValueError("v8 frozen task IDs must be unique")
        if self.downstream_campaign_stage_order != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError("v8 downstream stage order mismatch")
        expected_counts = Counter(row.design_axis for row in self.tasks)
        if dict(sorted(expected_counts.items())) != dict(
            sorted(self.design_axis_counts.items())
        ):
            raise ValueError("v8 design-axis counts mismatch")
        if any(value != 1 for value in self.design_axis_counts.values()):
            raise ValueError("each v8 design axis must appear once")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v8 freeze SHA mismatch")
        if observed_id != (
            "prospective_canonical_validation_freeze_v8:"
            + expected_sha[:20]
        ):
            raise ValueError("v8 freeze ID mismatch")
        return self


class ProspectiveCanonicalValidationCaseExecutionPlanV8(StrictModel):
    case_id: CaseIdV8
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

    collector_contract_required_before_execution: Literal[True] = True
    execution_authority_granted_by_this_plan: Literal[False] = False
    engineering_validation_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(
        self,
    ) -> "ProspectiveCanonicalValidationCaseExecutionPlanV8":
        if self.initial_e2e_argv[:3] != [
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
        ]:
            raise ValueError("v8 initial argv uses unexpected runner")
        if "--stop-after-initial-semantic" not in self.initial_e2e_argv:
            raise ValueError("v8 initial argv lacks semantic cut point")
        if "--overwrite-run" in self.initial_e2e_argv:
            raise ValueError("v8 initial argv must not overwrite run")
        if self.downstream_campaign_argv_base[:3] != [
            "python",
            "-m",
            "scripts.discovery.run_pre_n10_prospective_campaign_v1",
        ]:
            raise ValueError("v8 downstream argv uses unexpected runner")
        if "--semantic-review" in self.downstream_campaign_argv_base:
            raise ValueError("v8 downstream base argv must defer review")
        if "--allow-dirty-worktree" in self.downstream_campaign_argv_base:
            raise ValueError("v8 downstream argv must require clean worktree")
        if "--save-prompts" not in self.downstream_campaign_argv_base:
            raise ValueError("v8 downstream argv must preserve prompts")
        return self


class ProspectiveCanonicalValidationExecutionPlanV8(StrictModel):
    schema_version: Literal[
        "prospective-canonical-validation-execution-plan-v8"
    ] = "prospective-canonical-validation-execution-plan-v8"

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

    cases: list[ProspectiveCanonicalValidationCaseExecutionPlanV8]
    case_ids: list[str]
    case_count: Literal[5] = 5

    source_tasks_and_models_frozen_before_execution: Literal[True] = True
    p44_p48_outputs_observed_before_plan_freeze: Literal[False] = False
    collector_contract_required_before_execution: Literal[True] = True
    collector_contract_frozen_by_this_plan: Literal[False] = False
    execution_authority_granted_by_this_plan: Literal[False] = False
    case_order_fixed_p44_to_p48: Literal[True] = True
    later_case_adaptation_allowed: Literal[False] = False
    failed_or_terminal_case_replacement_allowed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False
    engineering_validation_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(
        self,
    ) -> "ProspectiveCanonicalValidationExecutionPlanV8":
        expected = ["P44", "P45", "P46", "P47", "P48"]
        if self.case_ids != expected:
            raise ValueError("v8 execution case IDs must be P44-P48")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("v8 execution cases must be ordered P44-P48")
        if len({row.source_task_id for row in self.cases}) != 5:
            raise ValueError("v8 execution task IDs must be unique")
        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v8 execution-plan SHA mismatch")
        if observed_id != (
            "prospective_canonical_validation_execution_plan_v8:"
            + expected_sha[:20]
        ):
            raise ValueError("v8 execution-plan ID mismatch")
        return self


class ProspectiveCanonicalValidationCaseContractV8(StrictModel):
    case_id: CaseIdV8
    source_task_id: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_dir: str
    campaign_root: str
    campaign_report_path: str

    initial_source_binding_bundle_path: str
    initial_canonical_source_reference_path: str
    initial_contract_v2_path: str

    primary_report_path: str
    post_primary_source_binding_bundle_path: str
    post_primary_canonical_source_reference_path: str
    post_primary_contract_v2_path: str

    regeneration_reentry_report_path: str
    downstream_handoff_path: str
    relational_binding_bridge_path: str
    vpost_plan_path: str
    vpost_report_path: str

    terminal_case_policy: Literal[
        "RETAIN_CASE_WITH_ZERO_CANONICAL_CHAIN_ROWS"
    ] = "RETAIN_CASE_WITH_ZERO_CANONICAL_CHAIN_ROWS"


class ProspectiveCanonicalValidationCollectorFreezeV8(StrictModel):
    schema_version: Literal[
        "prospective-canonical-validation-collector-freeze-v8"
    ] = "prospective-canonical-validation-collector-freeze-v8"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_execution_plan_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    collector_repository_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_worktree_dirty: Literal[False] = False

    cases: list[ProspectiveCanonicalValidationCaseContractV8]
    case_ids: list[str]
    case_count: Literal[5] = 5
    validation_output_path: str

    case_denominator_fixed_before_execution: Literal[True] = True
    terminal_cases_retained_in_denominator: Literal[True] = True
    result_conditioned_artifact_selection_allowed: Literal[False] = False
    collector_frozen_before_p44_p48_execution: Literal[True] = True
    p44_p48_outputs_observed_before_collector_freeze: Literal[False] = False
    campaign_execution_may_begin_after_this_freeze: Literal[True] = True

    llm_calls_allowed: Literal[False] = False
    artifact_mutation_allowed: Literal[False] = False
    engineering_validation_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "ProspectiveCanonicalValidationCollectorFreezeV8":
        expected = ["P44", "P45", "P46", "P47", "P48"]
        if self.case_ids != expected:
            raise ValueError("v8 collector case IDs must be P44-P48")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("v8 collector cases must be ordered P44-P48")
        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v8 collector freeze SHA mismatch")
        if observed_id != (
            "prospective_canonical_validation_collector_freeze_v8:"
            + expected_sha[:20]
        ):
            raise ValueError("v8 collector freeze ID mismatch")
        return self


class ProspectiveCanonicalValidationCaseResultV8(StrictModel):
    case_id: CaseIdV8
    campaign_final_status: str
    accounting_status: CaseAccountingStatusV8

    initial_v2_verified: bool
    primary_v2_verified: bool
    regenerated_v2_lineage_count: int = Field(ge=0)

    handoff_lineage_count: int = Field(ge=0)
    handoff_stable_id_lineage_count: int = Field(ge=0)
    handoff_legacy_authority_lineage_count: int = Field(ge=0)

    bridge_reached: bool
    bridge_stable_source_ids_used: bool | None = None
    bridge_exact_text_reconstruction_required: bool | None = None

    vpost_reached: bool
    vpost_source_binding_mode: str | None = None
    vpost_legacy_flag_count: int = Field(ge=0)
    vpost_completed_count: int = Field(ge=0)
    certification_decision_counts: dict[str, int] = Field(default_factory=dict)

    fresh_legacy_authority_violation: bool

    @model_validator(mode="after")
    def validate_result(
        self,
    ) -> "ProspectiveCanonicalValidationCaseResultV8":
        if self.fresh_legacy_authority_violation:
            return self
        if self.bridge_reached:
            if self.bridge_stable_source_ids_used is not True:
                raise ValueError(
                    "canonical v8 bridge must use stable source IDs"
                )
            if self.bridge_exact_text_reconstruction_required is not False:
                raise ValueError(
                    "canonical v8 bridge cannot require exact-text reconstruction"
                )
        if self.vpost_reached:
            if self.vpost_source_binding_mode != "CANONICAL_STABLE_ID":
                raise ValueError(
                    "canonical v8 V_post must use CANONICAL_STABLE_ID"
                )
            if self.vpost_legacy_flag_count != 0:
                raise ValueError(
                    "canonical v8 V_post cannot carry legacy exact-text flag"
                )
        return self


class ProspectiveCanonicalValidationReportV8(StrictModel):
    schema_version: Literal[
        "prospective-canonical-validation-report-v8"
    ] = "prospective-canonical-validation-report-v8"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_collector_freeze_id: str
    source_collector_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    cases: list[ProspectiveCanonicalValidationCaseResultV8]
    case_ids: list[str]
    case_count: Literal[5] = 5

    terminal_before_initial_vpre_case_count: int = Field(ge=0)
    pre_n10_terminal_case_count: int = Field(ge=0)
    bridge_reached_case_count: int = Field(ge=0)
    vpost_reached_case_count: int = Field(ge=0)
    fresh_legacy_authority_violation_count: int = Field(ge=0)
    vpost_source_binding_mode_counts: dict[str, int]
    certification_decision_counts: dict[str, int]

    llm_calls_performed_by_collector: Literal[0] = 0
    artifact_mutation_performed_by_collector: Literal[False] = False
    engineering_validation_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ProspectiveCanonicalValidationReportV8":
        expected = ["P44", "P45", "P46", "P47", "P48"]
        if self.case_ids != expected:
            raise ValueError("v8 report case IDs must be P44-P48")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("v8 report cases must be ordered P44-P48")
        if self.terminal_before_initial_vpre_case_count != sum(
            row.accounting_status == "TERMINAL_BEFORE_INITIAL_VPRE"
            for row in self.cases
        ):
            raise ValueError("v8 terminal-before-Vpre count mismatch")
        if self.pre_n10_terminal_case_count != sum(
            row.accounting_status == "PRE_N10_TERMINAL"
            for row in self.cases
        ):
            raise ValueError("v8 pre-N10 terminal count mismatch")
        if self.bridge_reached_case_count != sum(
            row.bridge_reached for row in self.cases
        ):
            raise ValueError("v8 bridge reached count mismatch")
        if self.vpost_reached_case_count != sum(
            row.vpost_reached for row in self.cases
        ):
            raise ValueError("v8 V_post reached count mismatch")
        if self.fresh_legacy_authority_violation_count != sum(
            row.fresh_legacy_authority_violation for row in self.cases
        ):
            raise ValueError("v8 legacy-authority violation count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v8 report SHA mismatch")
        if observed_id != (
            "prospective_canonical_validation_report_v8:"
            + expected_sha[:20]
        ):
            raise ValueError("v8 report ID mismatch")
        return self


def default_p44_p48_tasks() -> list[ProspectiveCanonicalValidationTaskV8]:
    rows = [
        (
            "P44",
            "illumination_geometry",
            "incidence_angle_surface_lattice_resonance_coupling",
            "illumination incidence angle",
            "surface-lattice resonance coupling strength",
            (
                "How does illumination incidence angle relate to "
                "surface-lattice resonance coupling strength in periodic "
                "SERS substrates?"
            ),
        ),
        (
            "P45",
            "vertical_gap_mode",
            "nanoparticle_mirror_separation_gap_plasmon_resonance",
            "nanoparticle-to-metal-film separation distance",
            "gap-plasmon resonance wavelength",
            (
                "How does nanoparticle-to-metal-film separation distance "
                "relate to gap-plasmon resonance wavelength in "
                "nanoparticle-on-mirror SERS architectures?"
            ),
        ),
        (
            "P46",
            "surface_redox_state",
            "surface_oxidation_state_charge_transfer_resonance",
            "metal surface oxidation state",
            "charge-transfer resonance energy",
            (
                "How does metal surface oxidation state relate to "
                "charge-transfer resonance energy in SERS-active interfaces?"
            ),
        ),
        (
            "P47",
            "thermal_desorption",
            "substrate_temperature_analyte_desorption_rate",
            "substrate temperature",
            "analyte desorption rate from SERS-active sites",
            (
                "How does substrate temperature relate to analyte desorption "
                "rate from SERS-active sites during measurement?"
            ),
        ),
        (
            "P48",
            "microfluidic_transport",
            "flow_velocity_hotspot_occupancy_event_rate",
            "microfluidic flow velocity",
            "single-molecule SERS hotspot occupancy event rate",
            (
                "How does microfluidic flow velocity relate to the event rate "
                "of single-molecule occupancy at SERS hotspots?"
            ),
        ),
    ]
    return [
        ProspectiveCanonicalValidationTaskV8(
            case_id=case_id,
            design_axis=axis,
            relation_family=family,
            source=source,
            target=target,
            question=question,
        )
        for case_id, axis, family, source, target, question in rows
    ]


def _prior_task_families_and_surfaces() -> tuple[
    set[str],
    set[tuple[str, str, str, str]],
]:
    tasks = [
        *default_p16_p20_tasks(),
        *default_p21_p25_tasks(),
        *default_p26_p28_tasks(),
        *default_p29_p33_tasks(),
        *default_p34_p38_tasks(),
        *default_p39_p43_tasks(),
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


def build_p44_p48_spec_from_v7_infrastructure(
    *,
    infrastructure_freeze: ProspectiveAtomicAdmissibilityFreezeV7,
    campaign_root: str,
) -> ProspectiveCanonicalValidationSpecV8:
    if infrastructure_freeze.case_ids != ["P39", "P40", "P41", "P42", "P43"]:
        raise ValueError("v8 infrastructure source must be P39-P43 v7")
    if len(infrastructure_freeze.tasks) != 5:
        raise ValueError("v8 infrastructure source must contain five tasks")

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
                raise ValueError("v8 infrastructure is non-uniform for " + field)

    tasks = default_p44_p48_tasks()
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
        raise ValueError(
            "v8 relation families overlap prior cohorts: " + repr(overlap)
        )
    if prior_surfaces & new_surfaces:
        raise ValueError("v8 task surfaces overlap prior cohorts")

    return ProspectiveCanonicalValidationSpecV8(
        campaign_name="P44_P48_canonical_validation_v8",
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


def build_prospective_canonical_validation_freeze_v8(
    *,
    spec: ProspectiveCanonicalValidationSpecV8,
    source_spec_sha256: str,
    infrastructure_freeze: ProspectiveAtomicAdmissibilityFreezeV7,
    infrastructure_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_file_sha256: str,
    repository_head_sha: str,
    repository_worktree_dirty: bool,
) -> ProspectiveCanonicalValidationFreezeV8:
    if repository_worktree_dirty:
        raise ValueError(
            "v8 freeze requires clean worktree including untracked files"
        )
    if infrastructure_freeze.freeze_id != spec.infrastructure_source_freeze_id:
        raise ValueError("v8 spec/infrastructure ID mismatch")
    if infrastructure_freeze.freeze_sha256 != (
        spec.infrastructure_source_freeze_sha256
    ):
        raise ValueError("v8 spec/infrastructure SHA mismatch")
    if infrastructure_freeze.source_regeneration_unit_freeze_id != (
        regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("v8 regeneration-unit ID mismatch")
    if infrastructure_freeze.source_regeneration_unit_freeze_sha256 != (
        regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("v8 regeneration-unit SHA mismatch")

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
            "scientific_outcome_predeclared": False,
            "canonical_integrity_outcome_predeclared": False,
            "engineering_validation_only": True,
            "scientific_validation_authority": False,
        }
        digest = _sha256_json(body)
        tasks.append(
            FrozenProspectiveCanonicalValidationTaskV8(
                **body,
                task_id=(
                    "prospective_canonical_validation_v8_task:"
                    + digest[:20]
                ),
                task_sha256=digest,
            )
        )

    body = {
        "schema_version": "prospective-canonical-validation-freeze-v8",
        "source_spec_sha256": source_spec_sha256,
        "campaign_name": spec.campaign_name,
        "campaign_root": str(root),
        "repository_head_sha": repository_head_sha,
        "repository_worktree_dirty": False,
        "infrastructure_source_freeze_id": infrastructure_freeze.freeze_id,
        "infrastructure_source_freeze_sha256": infrastructure_freeze.freeze_sha256,
        "infrastructure_source_freeze_file_sha256": (
            infrastructure_freeze_file_sha256
        ),
        "source_regeneration_unit_freeze_id": (
            regeneration_unit_freeze.freeze_id
        ),
        "source_regeneration_unit_freeze_sha256": (
            regeneration_unit_freeze.freeze_sha256
        ),
        "source_regeneration_unit_freeze_file_sha256": (
            regeneration_unit_freeze_file_sha256
        ),
        "tasks": [row.model_dump(mode="json") for row in tasks],
        "case_ids": ["P44", "P45", "P46", "P47", "P48"],
        "task_count": 5,
        "design_axis_counts": dict(
            sorted(Counter(row.design_axis for row in tasks).items())
        ),
        "downstream_campaign_stage_order": list(CAMPAIGN_STAGE_ORDER),
        "source_tasks_frozen_before_p44_p48_execution": True,
        "model_policy_frozen_before_p44_p48_execution": True,
        "p44_p48_outputs_observed_before_freeze": False,
        "prior_outcome_artifacts_consumed_by_freeze_builder": False,
        "later_case_adaptation_allowed": False,
        "failed_or_terminal_case_replacement_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "second_regeneration_allowed": False,
        "prospective_evidence_collection": True,
        "engineering_validation_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveCanonicalValidationFreezeV8(
        **body,
        freeze_id=(
            "prospective_canonical_validation_freeze_v8:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


def _initial_argv(
    task: FrozenProspectiveCanonicalValidationTaskV8,
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


def _downstream_argv(
    task: FrozenProspectiveCanonicalValidationTaskV8,
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
        "campaign_root": str(run / "prospective_canonical_validation_v8"),
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


def materialize_downstream_argv_v8(
    case: ProspectiveCanonicalValidationCaseExecutionPlanV8,
) -> list[str]:
    argv = list(case.downstream_campaign_argv_base)
    review = Path(case.initial_semantic_review_path)
    if review.is_file():
        argv += ["--semantic-review", str(review.expanduser().resolve())]
    return argv


def build_prospective_canonical_validation_execution_plan_v8(
    *,
    campaign_freeze: ProspectiveCanonicalValidationFreezeV8,
    campaign_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_path: Path,
    regeneration_unit_freeze_file_sha256: str,
    execution_plan_repository_head_sha: str,
    repository_worktree_dirty: bool,
    settings: ProspectiveAuthorityExecutionSettingsV3 | None = None,
) -> ProspectiveCanonicalValidationExecutionPlanV8:
    if repository_worktree_dirty:
        raise ValueError(
            "v8 execution plan requires clean worktree including untracked files"
        )
    if campaign_freeze.source_regeneration_unit_freeze_id != (
        regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("v8 execution regeneration-unit ID mismatch")
    if campaign_freeze.source_regeneration_unit_freeze_sha256 != (
        regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("v8 execution regeneration-unit SHA mismatch")
    if campaign_freeze.source_regeneration_unit_freeze_file_sha256 != (
        regeneration_unit_freeze_file_sha256
    ):
        raise ValueError("v8 execution regeneration-unit file SHA mismatch")

    resolved = settings or ProspectiveAuthorityExecutionSettingsV3()
    cases = []
    for task in campaign_freeze.tasks:
        downstream, paths = _downstream_argv(
            task,
            regeneration_unit_freeze_path,
            resolved,
        )
        cases.append(
            ProspectiveCanonicalValidationCaseExecutionPlanV8(
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
        "schema_version": (
            "prospective-canonical-validation-execution-plan-v8"
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
        "repository_worktree_dirty": False,
        "settings": resolved.model_dump(mode="json"),
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P44", "P45", "P46", "P47", "P48"],
        "case_count": 5,
        "source_tasks_and_models_frozen_before_execution": True,
        "p44_p48_outputs_observed_before_plan_freeze": False,
        "collector_contract_required_before_execution": True,
        "collector_contract_frozen_by_this_plan": False,
        "execution_authority_granted_by_this_plan": False,
        "case_order_fixed_p44_to_p48": True,
        "later_case_adaptation_allowed": False,
        "failed_or_terminal_case_replacement_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "second_regeneration_allowed": False,
        "engineering_validation_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
    }
    digest = _sha256_json(body)
    return ProspectiveCanonicalValidationExecutionPlanV8(
        **body,
        plan_id=(
            "prospective_canonical_validation_execution_plan_v8:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


def build_prospective_canonical_validation_collector_freeze_v8(
    *,
    execution_plan: ProspectiveCanonicalValidationExecutionPlanV8,
    execution_plan_file_sha256: str,
    collector_repository_head_sha: str,
    repository_worktree_dirty: bool,
    validation_output_path: Path,
) -> ProspectiveCanonicalValidationCollectorFreezeV8:
    if repository_worktree_dirty:
        raise ValueError(
            "v8 collector freeze requires clean worktree including untracked files"
        )
    cases = []
    for row in execution_plan.cases:
        campaign = Path(row.downstream_campaign_output_root).expanduser().resolve()
        initial = campaign / "01_initial_vpre"
        primary = campaign / "02_primary_router"
        reentry = campaign / "04_regeneration_reentry"
        handoff = campaign / "05_downstream_handoff"
        bridge = campaign / "07_relational_binding_bridge"
        vpost = campaign / "08_vpost_shadow"
        cases.append(
            ProspectiveCanonicalValidationCaseContractV8(
                case_id=row.case_id,
                source_task_id=row.source_task_id,
                source_task_sha256=row.source_task_sha256,
                run_dir=str(Path(row.run_dir).expanduser().resolve()),
                campaign_root=str(campaign),
                campaign_report_path=str(campaign / "campaign.report.json"),
                initial_source_binding_bundle_path=str(
                    initial / "atomic_source_binding.bundle.json"
                ),
                initial_canonical_source_reference_path=str(
                    initial / "canonical_source_reference.report.json"
                ),
                initial_contract_v2_path=str(
                    initial / "contract_v2.report.json"
                ),
                primary_report_path=str(
                    primary / "primary_router.report.json"
                ),
                post_primary_source_binding_bundle_path=str(
                    primary
                    / "atomic_source_binding.after_primary.bundle.json"
                ),
                post_primary_canonical_source_reference_path=str(
                    primary
                    / "canonical_source_reference.after_primary.report.json"
                ),
                post_primary_contract_v2_path=str(
                    primary / "contract_v2.after_primary_router.json"
                ),
                regeneration_reentry_report_path=str(
                    reentry / "reentry_v2.report.json"
                ),
                downstream_handoff_path=str(
                    handoff / "downstream_handoff.report.json"
                ),
                relational_binding_bridge_path=str(
                    bridge / "relational_binding_bridge.report.json"
                ),
                vpost_plan_path=str(
                    vpost / "vpost_execution.plan.json"
                ),
                vpost_report_path=str(
                    vpost / "vpost_shadow.report.json"
                ),
            )
        )

    body = {
        "schema_version": (
            "prospective-canonical-validation-collector-freeze-v8"
        ),
        "source_execution_plan_id": execution_plan.plan_id,
        "source_execution_plan_sha256": execution_plan.plan_sha256,
        "source_execution_plan_file_sha256": execution_plan_file_sha256,
        "collector_repository_head_sha": collector_repository_head_sha,
        "repository_worktree_dirty": False,
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P44", "P45", "P46", "P47", "P48"],
        "case_count": 5,
        "validation_output_path": str(
            validation_output_path.expanduser().resolve()
        ),
        "case_denominator_fixed_before_execution": True,
        "terminal_cases_retained_in_denominator": True,
        "result_conditioned_artifact_selection_allowed": False,
        "collector_frozen_before_p44_p48_execution": True,
        "p44_p48_outputs_observed_before_collector_freeze": False,
        "campaign_execution_may_begin_after_this_freeze": True,
        "llm_calls_allowed": False,
        "artifact_mutation_allowed": False,
        "engineering_validation_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
    }
    digest = _sha256_json(body)
    return ProspectiveCanonicalValidationCollectorFreezeV8(
        **body,
        freeze_id=(
            "prospective_canonical_validation_collector_freeze_v8:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("missing v8 " + label + ": " + str(resolved))
    return resolved


def _verify_v2_chain(
    *,
    bundle_path: str | Path,
    canonical_path: str | Path,
    contract_path: str | Path,
) -> tuple[
    AtomicScientificSourceBindingBundle,
    PreN10CanonicalSourceReferenceReportV1,
    PreN10ScientificContractReportV2,
]:
    bundle_file = _require_file(bundle_path, "source-binding bundle")
    canonical_file = _require_file(
        canonical_path,
        "canonical source-reference",
    )
    contract_file = _require_file(contract_path, "V2 contract")

    bundle = AtomicScientificSourceBindingBundle.model_validate_json(
        bundle_file.read_text(encoding="utf-8")
    )
    canonical = PreN10CanonicalSourceReferenceReportV1.model_validate_json(
        canonical_file.read_text(encoding="utf-8")
    )
    contract = PreN10ScientificContractReportV2.model_validate_json(
        contract_file.read_text(encoding="utf-8")
    )

    if canonical.source_binding_bundle_id != bundle.bundle_id:
        raise ValueError("v8 canonical report/bundle ID mismatch")
    if canonical.source_binding_bundle_sha256 != bundle.bundle_sha256:
        raise ValueError("v8 canonical report/bundle SHA mismatch")
    if contract.source_binding_bundle_id != bundle.bundle_id:
        raise ValueError("v8 V2 contract/bundle ID mismatch")
    if contract.source_binding_bundle_sha256 != bundle.bundle_sha256:
        raise ValueError("v8 V2 contract/bundle SHA mismatch")
    if contract.canonical_source_reference_report_id != canonical.report_id:
        raise ValueError("v8 V2 contract/canonical report ID mismatch")
    if contract.canonical_source_reference_report_sha256 != canonical.report_sha256:
        raise ValueError("v8 V2 contract/canonical report SHA mismatch")
    if not contract.stable_source_ids_used_for_readiness:
        raise ValueError("v8 V2 contract lost stable-ID readiness authority")
    if contract.exact_text_reconstruction_used_for_readiness:
        raise ValueError("v8 V2 contract uses exact-text readiness authority")
    return bundle, canonical, contract


def collect_prospective_canonical_validation_v8(
    freeze: ProspectiveCanonicalValidationCollectorFreezeV8,
) -> ProspectiveCanonicalValidationReportV8:
    results: list[ProspectiveCanonicalValidationCaseResultV8] = []

    for case in freeze.cases:
        campaign = PreN10ProspectiveCampaignReportV1.model_validate_json(
            _require_file(
                case.campaign_report_path,
                "campaign report",
            ).read_text(encoding="utf-8")
        )
        stages = {row.stage_name: row for row in campaign.stages}
        initial_active = stages["initial_vpre"].status in {
            "EXECUTED",
            "REUSED_VALIDATED",
        }
        primary_active = stages["primary_router"].status in {
            "EXECUTED",
            "REUSED_VALIDATED",
        }
        bridge_active = stages["relational_binding_bridge"].status in {
            "EXECUTED",
            "REUSED_VALIDATED",
        }
        vpost_active = stages["vpost_shadow"].status in {
            "EXECUTED",
            "REUSED_VALIDATED",
        }

        if not initial_active:
            results.append(
                ProspectiveCanonicalValidationCaseResultV8(
                    case_id=case.case_id,
                    campaign_final_status=campaign.final_status,
                    accounting_status="TERMINAL_BEFORE_INITIAL_VPRE",
                    initial_v2_verified=False,
                    primary_v2_verified=False,
                    regenerated_v2_lineage_count=0,
                    handoff_lineage_count=0,
                    handoff_stable_id_lineage_count=0,
                    handoff_legacy_authority_lineage_count=0,
                    bridge_reached=False,
                    vpost_reached=False,
                    fresh_legacy_authority_violation=False,
                )
            )
            continue

        _verify_v2_chain(
            bundle_path=case.initial_source_binding_bundle_path,
            canonical_path=case.initial_canonical_source_reference_path,
            contract_path=case.initial_contract_v2_path,
        )
        initial_v2_verified = True

        primary_v2_verified = False
        if primary_active:
            _, _, post_v2 = _verify_v2_chain(
                bundle_path=case.post_primary_source_binding_bundle_path,
                canonical_path=case.post_primary_canonical_source_reference_path,
                contract_path=case.post_primary_contract_v2_path,
            )
            primary = PreN10PrimaryRouterReportV1.model_validate_json(
                _require_file(
                    case.primary_report_path,
                    "primary router report",
                ).read_text(encoding="utf-8")
            )
            if primary.post_contract_report_id != post_v2.report_id:
                raise ValueError("v8 primary report/post V2 ID mismatch")
            if primary.post_contract_report_sha256 != post_v2.report_sha256:
                raise ValueError("v8 primary report/post V2 SHA mismatch")
            primary_v2_verified = True

        regenerated_v2_lineage_count = 0
        reentry_path = Path(case.regeneration_reentry_report_path)
        if stages["regeneration_semantic_reentry"].status in {
            "EXECUTED",
            "REUSED_VALIDATED",
        }:
            reentry = PreN10RegenerationReentryReportV2.model_validate_json(
                _require_file(
                    reentry_path,
                    "regeneration re-entry report",
                ).read_text(encoding="utf-8")
            )
            for lineage in reentry.lineages:
                if lineage.claim_decomposition_request_count == 0:
                    continue
                _verify_v2_chain(
                    bundle_path=str(lineage.source_binding_bundle_path),
                    canonical_path=str(
                        lineage.canonical_source_reference_path
                    ),
                    contract_path=str(lineage.contract_v2_report_path),
                )
                regenerated_v2_lineage_count += 1
        elif reentry_path.exists():
            raise ValueError(
                "v8 skipped regeneration re-entry unexpectedly has report"
            )

        handoff_lineage_count = 0
        handoff_stable_count = 0
        handoff_legacy_count = 0
        handoff_path = Path(case.downstream_handoff_path)
        if stages["downstream_handoff"].status in {
            "EXECUTED",
            "REUSED_VALIDATED",
        }:
            handoff = PreN10DownstreamHandoffReportV1.model_validate_json(
                _require_file(
                    handoff_path,
                    "downstream handoff",
                ).read_text(encoding="utf-8")
            )
            handoff_lineage_count = len(handoff.lineages)
            handoff_stable_count = sum(
                row.stable_source_ids_used_for_pre_n10_authority
                and not row.exact_text_reconstruction_used_for_pre_n10_authority
                for row in handoff.lineages
            )
            handoff_legacy_count = sum(
                not row.stable_source_ids_used_for_pre_n10_authority
                or row.exact_text_reconstruction_used_for_pre_n10_authority
                for row in handoff.lineages
            )

        if campaign.final_status == "PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE":
            results.append(
                ProspectiveCanonicalValidationCaseResultV8(
                    case_id=case.case_id,
                    campaign_final_status=campaign.final_status,
                    accounting_status="PRE_N10_TERMINAL",
                    initial_v2_verified=initial_v2_verified,
                    primary_v2_verified=primary_v2_verified,
                    regenerated_v2_lineage_count=(
                        regenerated_v2_lineage_count
                    ),
                    handoff_lineage_count=handoff_lineage_count,
                    handoff_stable_id_lineage_count=handoff_stable_count,
                    handoff_legacy_authority_lineage_count=(
                        handoff_legacy_count
                    ),
                    bridge_reached=False,
                    vpost_reached=False,
                    fresh_legacy_authority_violation=(
                        handoff_legacy_count > 0
                    ),
                )
            )
            continue

        bridge = PreN10RelationalBindingBridgeReportV1.model_validate_json(
            _require_file(
                case.relational_binding_bridge_path,
                "relational binding bridge",
            ).read_text(encoding="utf-8")
        ) if bridge_active else None

        vpost_plan = PreN10VPostShadowPlanV1.model_validate_json(
            _require_file(
                case.vpost_plan_path,
                "V_post execution plan",
            ).read_text(encoding="utf-8")
        ) if vpost_active else None

        vpost_report = PreN10VPostShadowReportV1.model_validate_json(
            _require_file(
                case.vpost_report_path,
                "V_post report",
            ).read_text(encoding="utf-8")
        ) if vpost_active else None

        legacy_flag_count = 0
        if vpost_plan is not None:
            legacy_flag_count = sum(
                "--allow-legacy-exact-text-source-binding" in stage.argv
                for lineage in vpost_plan.lineages
                for stage in lineage.stages
                if stage.stage == "relational_scientific_verifier"
            )

        violation = bool(
            handoff_legacy_count > 0
            or (
                bridge is not None
                and (
                    not bridge.stable_source_ids_used_for_relational_input
                    or bridge.exact_text_reconstruction_required_for_relational_input
                )
            )
            or (
                vpost_plan is not None
                and (
                    vpost_plan.source_binding_mode
                    != "CANONICAL_STABLE_ID"
                    or legacy_flag_count > 0
                )
            )
        )

        results.append(
            ProspectiveCanonicalValidationCaseResultV8(
                case_id=case.case_id,
                campaign_final_status=campaign.final_status,
                accounting_status=(
                    "CANONICAL_CHAIN_TO_VPOST"
                    if vpost_active
                    else "CANONICAL_CHAIN_TO_EXTERNAL"
                ),
                initial_v2_verified=initial_v2_verified,
                primary_v2_verified=primary_v2_verified,
                regenerated_v2_lineage_count=regenerated_v2_lineage_count,
                handoff_lineage_count=handoff_lineage_count,
                handoff_stable_id_lineage_count=handoff_stable_count,
                handoff_legacy_authority_lineage_count=handoff_legacy_count,
                bridge_reached=bridge is not None,
                bridge_stable_source_ids_used=(
                    bridge.stable_source_ids_used_for_relational_input
                    if bridge is not None
                    else None
                ),
                bridge_exact_text_reconstruction_required=(
                    bridge.exact_text_reconstruction_required_for_relational_input
                    if bridge is not None
                    else None
                ),
                vpost_reached=vpost_plan is not None,
                vpost_source_binding_mode=(
                    vpost_plan.source_binding_mode
                    if vpost_plan is not None
                    else None
                ),
                vpost_legacy_flag_count=legacy_flag_count,
                vpost_completed_count=(
                    vpost_report.completed_count
                    if vpost_report is not None
                    else 0
                ),
                certification_decision_counts=(
                    dict(vpost_report.certification_decision_counts)
                    if vpost_report is not None
                    else {}
                ),
                fresh_legacy_authority_violation=violation,
            )
        )

    mode_counts = dict(
        sorted(
            Counter(
                row.vpost_source_binding_mode
                for row in results
                if row.vpost_source_binding_mode is not None
            ).items()
        )
    )
    certification_counts = dict(
        sorted(
            Counter(
                decision
                for row in results
                for decision, count in row.certification_decision_counts.items()
                for _ in range(count)
            ).items()
        )
    )

    body = {
        "schema_version": "prospective-canonical-validation-report-v8",
        "source_collector_freeze_id": freeze.freeze_id,
        "source_collector_freeze_sha256": freeze.freeze_sha256,
        "cases": [row.model_dump(mode="json") for row in results],
        "case_ids": ["P44", "P45", "P46", "P47", "P48"],
        "case_count": 5,
        "terminal_before_initial_vpre_case_count": sum(
            row.accounting_status == "TERMINAL_BEFORE_INITIAL_VPRE"
            for row in results
        ),
        "pre_n10_terminal_case_count": sum(
            row.accounting_status == "PRE_N10_TERMINAL"
            for row in results
        ),
        "bridge_reached_case_count": sum(
            row.bridge_reached for row in results
        ),
        "vpost_reached_case_count": sum(
            row.vpost_reached for row in results
        ),
        "fresh_legacy_authority_violation_count": sum(
            row.fresh_legacy_authority_violation for row in results
        ),
        "vpost_source_binding_mode_counts": mode_counts,
        "certification_decision_counts": certification_counts,
        "llm_calls_performed_by_collector": 0,
        "artifact_mutation_performed_by_collector": False,
        "engineering_validation_only": True,
        "scientific_validation_authority": False,
        "production_selection_authority": False,
    }
    digest = _sha256_json(body)
    return ProspectiveCanonicalValidationReportV8(
        **body,
        report_id=(
            "prospective_canonical_validation_report_v8:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "ProspectiveCanonicalValidationCollectorFreezeV8",
    "ProspectiveCanonicalValidationExecutionPlanV8",
    "ProspectiveCanonicalValidationFreezeV8",
    "ProspectiveCanonicalValidationReportV8",
    "ProspectiveCanonicalValidationSpecV8",
    "build_p44_p48_spec_from_v7_infrastructure",
    "build_prospective_canonical_validation_collector_freeze_v8",
    "build_prospective_canonical_validation_execution_plan_v8",
    "build_prospective_canonical_validation_freeze_v8",
    "collect_prospective_canonical_validation_v8",
    "default_p44_p48_tasks",
    "materialize_downstream_argv_v8",
    "sha256_file",
]
