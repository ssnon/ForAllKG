from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
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
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CaseIdV3 = Literal["P21", "P22", "P23", "P24", "P25"]
DesignAxisV3 = Literal[
    "ensemble_geometry",
    "gap_orientation",
    "surface_access",
    "mesoscale_morphology",
    "hybrid_interface",
]


class ProspectiveAuthoritySourceTaskDefinitionV3(StrictModel):
    case_id: CaseIdV3
    design_axis: DesignAxisV3
    relation_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    stop: str | None = None
    target: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: Literal["explain_connection"] = "explain_connection"


class ProspectiveAuthorityModelPolicyV3(StrictModel):
    initial_generation_model: str = Field(min_length=1)
    initial_critic_model: str = Field(min_length=1)
    decomposition_model: str = Field(min_length=1)
    primary_model: str = Field(min_length=1)
    specification_repair_model: str = Field(min_length=1)
    specification_audit_model: str = Field(min_length=1)
    source_alignment_model: str = Field(min_length=1)
    regeneration_model: str = Field(min_length=1)
    semantic_critic_model: str = Field(min_length=1)
    external_n10_model: str = Field(min_length=1)
    vpost_model: str = Field(min_length=1)

    @classmethod
    def from_generation_and_critic(
        cls,
        *,
        generation_model: str,
        critic_model: str,
    ) -> "ProspectiveAuthorityModelPolicyV3":
        generation = str(generation_model).strip()
        critic = str(critic_model).strip()
        if not generation or not critic:
            raise ValueError("v3 model policy requires generation and critic models")
        return cls(
            initial_generation_model=generation,
            initial_critic_model=critic,
            decomposition_model=generation,
            primary_model=critic,
            specification_repair_model=generation,
            specification_audit_model=critic,
            source_alignment_model=critic,
            regeneration_model=generation,
            semantic_critic_model=critic,
            external_n10_model=critic,
            vpost_model=critic,
        )


class ProspectiveAuthorityCampaignSpecV3(StrictModel):
    schema_version: Literal[
        "prospective-authority-campaign-spec-v3"
    ] = "prospective-authority-campaign-spec-v3"

    campaign_name: str = Field(min_length=1)
    campaign_root: str = Field(min_length=1)

    domain_profile_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    data_root: str = Field(min_length=1)
    semantic_roots: list[str] = Field(min_length=1)

    model_policy: ProspectiveAuthorityModelPolicyV3
    tasks: list[ProspectiveAuthoritySourceTaskDefinitionV3] = Field(
        min_length=5,
        max_length=5,
    )

    infrastructure_source_freeze_id: str = Field(min_length=1)
    infrastructure_source_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    source_task_design_axes_predeclared: Literal[True] = True
    tasks_defined_without_prior_scientific_outcomes: Literal[True] = True
    prior_v2_task_definitions_used_only_for_nonoverlap_guard: Literal[
        True
    ] = True
    infrastructure_only_inherited_from_prior_campaign: Literal[True] = True

    initial_generation_cutpoint: Literal[
        "AFTER_INITIAL_SEMANTIC_BEFORE_EXTERNAL_NOVELTY"
    ] = "AFTER_INITIAL_SEMANTIC_BEFORE_EXTERNAL_NOVELTY"
    downstream_campaign_schema: Literal[
        "pre-n10-prospective-campaign-v1"
    ] = "pre-n10-prospective-campaign-v1"

    @model_validator(mode="after")
    def validate_spec(self) -> "ProspectiveAuthorityCampaignSpecV3":
        expected_cases = ["P21", "P22", "P23", "P24", "P25"]
        if [row.case_id for row in self.tasks] != expected_cases:
            raise ValueError("v3 source tasks must be exactly P21-P25")

        expected_axes = [
            "ensemble_geometry",
            "gap_orientation",
            "surface_access",
            "mesoscale_morphology",
            "hybrid_interface",
        ]
        if [row.design_axis for row in self.tasks] != expected_axes:
            raise ValueError(
                "v3 source-task design axes must use the predeclared canonical order"
            )

        families = [row.relation_family for row in self.tasks]
        if len(set(families)) != len(families):
            raise ValueError("P21-P25 relation families must be unique")

        task_keys = [
            (
                row.source.casefold().strip(),
                (row.stop or "").casefold().strip(),
                row.target.casefold().strip(),
                row.question.casefold().strip(),
            )
            for row in self.tasks
        ]
        if len(set(task_keys)) != len(task_keys):
            raise ValueError("P21-P25 source tasks must be distinct")

        if len(set(self.semantic_roots)) != len(self.semantic_roots):
            raise ValueError("semantic_roots must be unique")
        return self


class FrozenProspectiveAuthorityTaskV3(StrictModel):
    case_id: CaseIdV3
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    design_axis: DesignAxisV3
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
    pre_n10_route_predeclared: Literal[False] = False
    n10_outcome_predeclared: Literal[False] = False
    verifier_outcome_predeclared: Literal[False] = False

    @model_validator(mode="after")
    def validate_task(self) -> "FrozenProspectiveAuthorityTaskV3":
        body = self.model_dump(mode="json")
        observed_id = body.pop("task_id")
        observed_sha = body.pop("task_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective authority v3 task SHA mismatch")
        if observed_id != (
            "prospective_authority_v3_task:" + expected_sha[:20]
        ):
            raise ValueError("prospective authority v3 task ID mismatch")
        return self


class ProspectiveAuthorityCampaignFreezeV3(StrictModel):
    schema_version: Literal[
        "prospective-authority-campaign-freeze-v3"
    ] = "prospective-authority-campaign-freeze-v3"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    campaign_name: str
    campaign_root: str
    repository_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_tracked_worktree_dirty: Literal[False] = False

    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_unit_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    infrastructure_source_freeze_id: str
    infrastructure_source_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    infrastructure_source_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    tasks: list[FrozenProspectiveAuthorityTaskV3]
    case_ids: list[str]
    task_count: Literal[5] = 5
    design_axis_counts: dict[str, int]

    initial_e2e_cutpoint_required: Literal[True] = True
    initial_e2e_cutpoint_flag: Literal[
        "--stop-after-initial-semantic"
    ] = "--stop-after-initial-semantic"
    initial_external_novelty_allowed: Literal[False] = False
    initial_n9_allowed: Literal[False] = False
    initial_n10_allowed: Literal[False] = False
    initial_refinement_allowed: Literal[False] = False

    downstream_campaign_schema: Literal[
        "pre-n10-prospective-campaign-v1"
    ] = "pre-n10-prospective-campaign-v1"
    downstream_campaign_stage_order: list[str]

    provider_request: Literal["auto"] = "auto"
    results_per_query: Literal[12] = 12

    source_tasks_frozen_before_initial_generation: Literal[True] = True
    model_policy_frozen_before_initial_generation: Literal[True] = True
    initial_cutpoint_frozen_before_initial_generation: Literal[True] = True
    downstream_authority_path_frozen_before_initial_generation: Literal[
        True
    ] = True

    initial_hypothesis_outputs_observed_before_freeze: Literal[False] = False
    semantic_outputs_observed_before_freeze: Literal[False] = False
    primary_route_outputs_observed_before_freeze: Literal[False] = False
    regeneration_outputs_observed_before_freeze: Literal[False] = False
    external_novelty_outputs_observed_before_freeze: Literal[False] = False
    n10_outputs_observed_before_freeze: Literal[False] = False
    endpoint_binding_outputs_observed_before_freeze: Literal[False] = False
    verifier_outputs_observed_before_freeze: Literal[False] = False

    prior_scientific_outputs_used_to_define_tasks: Literal[False] = False
    prior_v2_task_definitions_used_only_for_nonoverlap_guard: Literal[
        True
    ] = True
    prior_campaign_used_for_infrastructure_only: Literal[True] = True

    legacy_case_reuse_allowed: Literal[False] = False
    failed_or_abstained_case_replacement_allowed: Literal[False] = False
    post_freeze_task_edit_allowed: Literal[False] = False
    post_freeze_model_policy_edit_allowed: Literal[False] = False
    post_freeze_cutpoint_edit_allowed: Literal[False] = False
    post_freeze_downstream_authority_edit_allowed: Literal[False] = False
    later_case_adaptation_allowed: Literal[False] = False

    second_regeneration_allowed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(self) -> "ProspectiveAuthorityCampaignFreezeV3":
        expected_cases = ["P21", "P22", "P23", "P24", "P25"]
        if self.case_ids != expected_cases:
            raise ValueError("v3 campaign freeze case IDs must be P21-P25")
        if [row.case_id for row in self.tasks] != expected_cases:
            raise ValueError("v3 campaign freeze task order must be P21-P25")
        if len({row.task_id for row in self.tasks}) != 5:
            raise ValueError("v3 frozen task IDs must be unique")
        if self.downstream_campaign_stage_order != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError("v3 downstream campaign stage order mismatch")

        expected_axis_counts = Counter(row.design_axis for row in self.tasks)
        if dict(sorted(expected_axis_counts.items())) != dict(
            sorted(self.design_axis_counts.items())
        ):
            raise ValueError("v3 design-axis counts mismatch")
        if any(count != 1 for count in self.design_axis_counts.values()):
            raise ValueError("each v3 design axis must appear exactly once")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective authority v3 freeze SHA mismatch")
        if observed_id != (
            "prospective_authority_campaign_freeze_v3:"
            + expected_sha[:20]
        ):
            raise ValueError("prospective authority v3 freeze ID mismatch")
        return self


def default_p21_p25_tasks() -> list[ProspectiveAuthoritySourceTaskDefinitionV3]:
    rows = [
        (
            "P21",
            "ensemble_geometry",
            "particle_size_polydispersity_ensemble_resonance_bandwidth",
            "nanoparticle size polydispersity",
            "ensemble plasmon resonance bandwidth",
            (
                "How does nanoparticle size polydispersity relate to ensemble "
                "plasmon resonance bandwidth in SERS nanostructures?"
            ),
        ),
        (
            "P22",
            "gap_orientation",
            "nanogap_orientation_distribution_polarization_anisotropy",
            "nanogap orientation distribution",
            "polarization anisotropy of SERS enhancement",
            (
                "How does nanogap orientation distribution relate to "
                "polarization anisotropy of SERS enhancement?"
            ),
        ),
        (
            "P23",
            "surface_access",
            "ligand_packing_density_hotspot_analyte_access",
            "surface ligand packing density",
            "analyte access to electromagnetic hotspots",
            (
                "How does surface ligand packing density relate to analyte "
                "access to electromagnetic hotspots in SERS substrates?"
            ),
        ),
        (
            "P24",
            "mesoscale_morphology",
            "roughness_correlation_length_intensity_autocorrelation",
            "substrate roughness correlation length",
            "spatial SERS intensity autocorrelation length",
            (
                "How does substrate roughness correlation length relate to "
                "spatial SERS intensity autocorrelation length?"
            ),
        ),
        (
            "P25",
            "hybrid_interface",
            "oxide_overlayer_thickness_charge_transfer_contribution",
            "metal-oxide overlayer thickness",
            "charge-transfer contribution to SERS enhancement",
            (
                "How does metal-oxide overlayer thickness relate to the "
                "charge-transfer contribution to SERS enhancement?"
            ),
        ),
    ]
    return [
        ProspectiveAuthoritySourceTaskDefinitionV3(
            case_id=case_id,
            design_axis=axis,
            relation_family=family,
            source=source,
            target=target,
            question=question,
        )
        for case_id, axis, family, source, target, question in rows
    ]


def build_p21_p25_spec_from_v2_infrastructure(
    *,
    infrastructure_freeze: ProspectiveRoutedCampaignFreezeV2,
    campaign_root: str,
) -> ProspectiveAuthorityCampaignSpecV3:
    expected = ["P16", "P17", "P18", "P19", "P20"]
    if infrastructure_freeze.case_ids != expected:
        raise ValueError("v3 infrastructure source freeze must be P16-P20")
    if len(infrastructure_freeze.tasks) != 5:
        raise ValueError("v3 infrastructure source freeze must contain five tasks")

    first = infrastructure_freeze.tasks[0]
    invariant_fields = (
        "domain_profile_id",
        "corpus_id",
        "data_root",
        "semantic_roots",
        "generation_model",
        "critic_model",
    )
    for task in infrastructure_freeze.tasks[1:]:
        for field in invariant_fields:
            if getattr(task, field) != getattr(first, field):
                raise ValueError(
                    "v3 infrastructure source freeze has non-uniform " + field
                )

    tasks = default_p21_p25_tasks()
    old_families = {row.relation_family for row in infrastructure_freeze.tasks}
    new_families = {row.relation_family for row in tasks}
    overlap = sorted(old_families & new_families)
    if overlap:
        raise ValueError(
            "v3 relation families overlap P16-P20 infrastructure tasks: "
            + repr(overlap)
        )

    return ProspectiveAuthorityCampaignSpecV3(
        campaign_name="P21_P25_authority_v3",
        campaign_root=str(Path(campaign_root).expanduser().resolve()),
        domain_profile_id=first.domain_profile_id,
        corpus_id=first.corpus_id,
        data_root=first.data_root,
        semantic_roots=list(first.semantic_roots),
        model_policy=ProspectiveAuthorityModelPolicyV3.from_generation_and_critic(
            generation_model=first.generation_model,
            critic_model=first.critic_model,
        ),
        tasks=tasks,
        infrastructure_source_freeze_id=infrastructure_freeze.freeze_id,
        infrastructure_source_freeze_sha256=infrastructure_freeze.freeze_sha256,
    )


def build_prospective_authority_campaign_freeze_v3(
    *,
    spec: ProspectiveAuthorityCampaignSpecV3,
    source_spec_sha256: str,
    infrastructure_freeze: ProspectiveRoutedCampaignFreezeV2,
    infrastructure_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_file_sha256: str,
    repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
) -> ProspectiveAuthorityCampaignFreezeV3:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "prospective authority v3 freeze requires a clean tracked worktree"
        )
    if infrastructure_freeze.freeze_id != spec.infrastructure_source_freeze_id:
        raise ValueError("v3 spec/infrastructure freeze ID mismatch")
    if (
        infrastructure_freeze.freeze_sha256
        != spec.infrastructure_source_freeze_sha256
    ):
        raise ValueError("v3 spec/infrastructure freeze SHA mismatch")

    if (
        infrastructure_freeze.source_regeneration_unit_freeze_id
        != regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("v3 infrastructure/regeneration-unit freeze ID mismatch")
    if (
        infrastructure_freeze.source_regeneration_unit_freeze_sha256
        != regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("v3 infrastructure/regeneration-unit freeze SHA mismatch")

    old_families = {row.relation_family for row in infrastructure_freeze.tasks}
    new_families = {row.relation_family for row in spec.tasks}
    overlap = sorted(old_families & new_families)
    if overlap:
        raise ValueError(
            "v3 frozen relation families overlap P16-P20: " + repr(overlap)
        )

    root = Path(spec.campaign_root).expanduser().resolve()
    data_root = Path(spec.data_root).expanduser().resolve()
    semantic_roots = [
        str(Path(value).expanduser().resolve())
        for value in spec.semantic_roots
    ]

    tasks: list[FrozenProspectiveAuthorityTaskV3] = []
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
            "data_root": str(data_root),
            "semantic_roots": semantic_roots,
            "model_policy": spec.model_policy.model_dump(mode="json"),
            "source_task_definition_only": True,
            "hypothesis_content_predeclared": False,
            "semantic_outcome_predeclared": False,
            "pre_n10_route_predeclared": False,
            "n10_outcome_predeclared": False,
            "verifier_outcome_predeclared": False,
        }
        digest = _sha256_json(body)
        tasks.append(
            FrozenProspectiveAuthorityTaskV3(
                **body,
                task_id="prospective_authority_v3_task:" + digest[:20],
                task_sha256=digest,
            )
        )

    axis_counts = Counter(row.design_axis for row in tasks)
    body = {
        "schema_version": "prospective-authority-campaign-freeze-v3",
        "source_spec_sha256": str(source_spec_sha256),
        "campaign_name": spec.campaign_name,
        "campaign_root": str(root),
        "repository_head_sha": str(repository_head_sha),
        "repository_tracked_worktree_dirty": False,
        "source_regeneration_unit_freeze_id": regeneration_unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256": (
            regeneration_unit_freeze.freeze_sha256
        ),
        "source_regeneration_unit_freeze_file_sha256": str(
            regeneration_unit_freeze_file_sha256
        ),
        "infrastructure_source_freeze_id": infrastructure_freeze.freeze_id,
        "infrastructure_source_freeze_sha256": infrastructure_freeze.freeze_sha256,
        "infrastructure_source_freeze_file_sha256": str(
            infrastructure_freeze_file_sha256
        ),
        "tasks": [row.model_dump(mode="json") for row in tasks],
        "case_ids": ["P21", "P22", "P23", "P24", "P25"],
        "task_count": 5,
        "design_axis_counts": dict(sorted(axis_counts.items())),
        "initial_e2e_cutpoint_required": True,
        "initial_e2e_cutpoint_flag": "--stop-after-initial-semantic",
        "initial_external_novelty_allowed": False,
        "initial_n9_allowed": False,
        "initial_n10_allowed": False,
        "initial_refinement_allowed": False,
        "downstream_campaign_schema": "pre-n10-prospective-campaign-v1",
        "downstream_campaign_stage_order": list(CAMPAIGN_STAGE_ORDER),
        "provider_request": "auto",
        "results_per_query": 12,
        "source_tasks_frozen_before_initial_generation": True,
        "model_policy_frozen_before_initial_generation": True,
        "initial_cutpoint_frozen_before_initial_generation": True,
        "downstream_authority_path_frozen_before_initial_generation": True,
        "initial_hypothesis_outputs_observed_before_freeze": False,
        "semantic_outputs_observed_before_freeze": False,
        "primary_route_outputs_observed_before_freeze": False,
        "regeneration_outputs_observed_before_freeze": False,
        "external_novelty_outputs_observed_before_freeze": False,
        "n10_outputs_observed_before_freeze": False,
        "endpoint_binding_outputs_observed_before_freeze": False,
        "verifier_outputs_observed_before_freeze": False,
        "prior_scientific_outputs_used_to_define_tasks": False,
        "prior_v2_task_definitions_used_only_for_nonoverlap_guard": True,
        "prior_campaign_used_for_infrastructure_only": True,
        "legacy_case_reuse_allowed": False,
        "failed_or_abstained_case_replacement_allowed": False,
        "post_freeze_task_edit_allowed": False,
        "post_freeze_model_policy_edit_allowed": False,
        "post_freeze_cutpoint_edit_allowed": False,
        "post_freeze_downstream_authority_edit_allowed": False,
        "later_case_adaptation_allowed": False,
        "second_regeneration_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveAuthorityCampaignFreezeV3(
        **body,
        freeze_id="prospective_authority_campaign_freeze_v3:" + digest[:20],
        freeze_sha256=digest,
    )


__all__ = [
    "CaseIdV3",
    "DesignAxisV3",
    "FrozenProspectiveAuthorityTaskV3",
    "ProspectiveAuthorityCampaignFreezeV3",
    "ProspectiveAuthorityCampaignSpecV3",
    "ProspectiveAuthorityModelPolicyV3",
    "ProspectiveAuthoritySourceTaskDefinitionV3",
    "build_p21_p25_spec_from_v2_infrastructure",
    "build_prospective_authority_campaign_freeze_v3",
    "default_p21_p25_tasks",
    "sha256_file",
]
