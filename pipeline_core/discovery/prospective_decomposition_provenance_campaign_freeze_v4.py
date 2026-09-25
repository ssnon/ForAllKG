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
from pipeline_core.discovery.prospective_authority_campaign_freeze_v3 import (
    ProspectiveAuthorityCampaignFreezeV3,
    ProspectiveAuthorityModelPolicyV3,
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


CaseIdV4 = Literal["P26", "P27", "P28"]
DesignAxisV4 = Literal[
    "gap_coupling",
    "resonance_detuning",
    "aggregation_topology",
]


class ProspectiveDecompositionProvenanceTaskV4(StrictModel):
    case_id: CaseIdV4
    design_axis: DesignAxisV4
    relation_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    stop: str | None = None
    target: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: Literal["explain_connection"] = "explain_connection"


class ProspectiveDecompositionProvenanceSpecV4(StrictModel):
    schema_version: Literal[
        "prospective-decomposition-provenance-spec-v4"
    ] = "prospective-decomposition-provenance-spec-v4"

    campaign_name: str = Field(min_length=1)
    campaign_root: str = Field(min_length=1)

    domain_profile_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    data_root: str = Field(min_length=1)
    semantic_roots: list[str] = Field(min_length=1)
    model_policy: ProspectiveAuthorityModelPolicyV3

    tasks: list[ProspectiveDecompositionProvenanceTaskV4] = Field(
        min_length=3,
        max_length=3,
    )

    infrastructure_source_freeze_id: str = Field(min_length=1)
    infrastructure_source_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    diagnostic_goal: Literal[
        "REGENERATION_DECOMPOSITION_PROVENANCE_OBSERVABILITY"
    ] = "REGENERATION_DECOMPOSITION_PROVENANCE_OBSERVABILITY"
    diagnostic_goal_motivated_by_prior_v3_outcomes: Literal[True] = True
    prior_v3_outcome_artifacts_consumed_by_builder: Literal[False] = False
    prior_v3_task_definitions_used_only_for_nonoverlap_guard: Literal[
        True
    ] = True
    infrastructure_and_model_policy_only_inherited_from_v3: Literal[
        True
    ] = True

    scientific_validation_authority: Literal[False] = False
    source_tasks_predeclared_before_v4_execution: Literal[True] = True
    expected_scientific_outcomes_predeclared: Literal[False] = False

    initial_generation_cutpoint: Literal[
        "AFTER_INITIAL_SEMANTIC_BEFORE_EXTERNAL_NOVELTY"
    ] = "AFTER_INITIAL_SEMANTIC_BEFORE_EXTERNAL_NOVELTY"
    downstream_campaign_schema: Literal[
        "pre-n10-prospective-campaign-v1"
    ] = "pre-n10-prospective-campaign-v1"

    @model_validator(mode="after")
    def validate_spec(
        self,
    ) -> "ProspectiveDecompositionProvenanceSpecV4":
        expected_cases = ["P26", "P27", "P28"]
        expected_axes = [
            "gap_coupling",
            "resonance_detuning",
            "aggregation_topology",
        ]
        if [row.case_id for row in self.tasks] != expected_cases:
            raise ValueError("v4 diagnostic source tasks must be exactly P26-P28")
        if [row.design_axis for row in self.tasks] != expected_axes:
            raise ValueError("v4 diagnostic design-axis order mismatch")
        families = [row.relation_family for row in self.tasks]
        if len(set(families)) != len(families):
            raise ValueError("P26-P28 relation families must be unique")
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
            raise ValueError("P26-P28 source tasks must be distinct")
        if len(set(self.semantic_roots)) != len(self.semantic_roots):
            raise ValueError("semantic_roots must be unique")
        return self


class FrozenProspectiveDecompositionProvenanceTaskV4(StrictModel):
    case_id: CaseIdV4
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    design_axis: DesignAxisV4
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

    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    hypothesis_content_predeclared: Literal[False] = False
    semantic_outcome_predeclared: Literal[False] = False
    pre_n10_route_predeclared: Literal[False] = False
    source_id_coverage_predeclared: Literal[False] = False
    exact_source_rebind_recovery_predeclared: Literal[False] = False

    @model_validator(mode="after")
    def validate_task(
        self,
    ) -> "FrozenProspectiveDecompositionProvenanceTaskV4":
        body = self.model_dump(mode="json")
        observed_id = body.pop("task_id")
        observed_sha = body.pop("task_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective decomposition provenance v4 task SHA mismatch")
        if observed_id != (
            "prospective_decomposition_provenance_v4_task:"
            + expected_sha[:20]
        ):
            raise ValueError("prospective decomposition provenance v4 task ID mismatch")
        return self


class ProspectiveDecompositionProvenanceFreezeV4(StrictModel):
    schema_version: Literal[
        "prospective-decomposition-provenance-freeze-v4"
    ] = "prospective-decomposition-provenance-freeze-v4"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    campaign_name: str
    campaign_root: str
    repository_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_tracked_worktree_dirty: Literal[False] = False

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

    tasks: list[FrozenProspectiveDecompositionProvenanceTaskV4]
    case_ids: list[str]
    task_count: Literal[3] = 3
    design_axis_counts: dict[str, int]

    diagnostic_goal: Literal[
        "REGENERATION_DECOMPOSITION_PROVENANCE_OBSERVABILITY"
    ] = "REGENERATION_DECOMPOSITION_PROVENANCE_OBSERVABILITY"
    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False

    required_regeneration_decomposition_sidecar_filename: Literal[
        "claim_decomposition.sanitization_audit.json"
    ] = "claim_decomposition.sanitization_audit.json"
    required_regeneration_decomposition_sidecar_schema: Literal[
        "pre-n10-regeneration-decomposition-sanitization-audit-v1"
    ] = "pre-n10-regeneration-decomposition-sanitization-audit-v1"
    sidecar_required_if_regeneration_decomposition_executes: Literal[
        True
    ] = True
    sidecar_missing_is_diagnostic_failure: Literal[True] = True
    sidecar_has_scientific_authority: Literal[False] = False

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

    diagnostic_goal_motivated_by_prior_v3_outcomes: Literal[True] = True
    prior_v3_outcome_artifacts_consumed_by_freeze_builder: Literal[
        False
    ] = False
    prior_v3_task_definitions_used_only_for_nonoverlap_guard: Literal[
        True
    ] = True
    infrastructure_and_model_policy_only_inherited_from_v3: Literal[
        True
    ] = True

    source_tasks_frozen_before_v4_execution: Literal[True] = True
    model_policy_frozen_before_v4_execution: Literal[True] = True
    initial_cutpoint_frozen_before_v4_execution: Literal[True] = True
    downstream_authority_path_frozen_before_v4_execution: Literal[True] = True

    v4_initial_outputs_observed_before_freeze: Literal[False] = False
    v4_semantic_outputs_observed_before_freeze: Literal[False] = False
    v4_primary_route_outputs_observed_before_freeze: Literal[False] = False
    v4_regeneration_outputs_observed_before_freeze: Literal[False] = False
    v4_source_id_coverage_observed_before_freeze: Literal[False] = False

    failed_or_terminal_case_replacement_allowed: Literal[False] = False
    post_freeze_task_edit_allowed: Literal[False] = False
    post_freeze_model_policy_edit_allowed: Literal[False] = False
    post_freeze_cutpoint_edit_allowed: Literal[False] = False
    post_freeze_downstream_path_edit_allowed: Literal[False] = False
    later_case_adaptation_allowed: Literal[False] = False

    second_regeneration_allowed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "ProspectiveDecompositionProvenanceFreezeV4":
        expected_cases = ["P26", "P27", "P28"]
        if self.case_ids != expected_cases:
            raise ValueError("v4 diagnostic freeze case IDs must be P26-P28")
        if [row.case_id for row in self.tasks] != expected_cases:
            raise ValueError("v4 diagnostic frozen task order must be P26-P28")
        if len({row.task_id for row in self.tasks}) != 3:
            raise ValueError("v4 diagnostic frozen task IDs must be unique")
        if self.downstream_campaign_stage_order != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError("v4 diagnostic downstream stage order mismatch")
        expected_axis_counts = Counter(row.design_axis for row in self.tasks)
        if dict(sorted(expected_axis_counts.items())) != dict(
            sorted(self.design_axis_counts.items())
        ):
            raise ValueError("v4 diagnostic design-axis counts mismatch")
        if any(count != 1 for count in self.design_axis_counts.values()):
            raise ValueError("each v4 diagnostic design axis must appear once")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective decomposition provenance v4 freeze SHA mismatch")
        if observed_id != (
            "prospective_decomposition_provenance_freeze_v4:"
            + expected_sha[:20]
        ):
            raise ValueError("prospective decomposition provenance v4 freeze ID mismatch")
        return self


def default_p26_p28_tasks() -> list[ProspectiveDecompositionProvenanceTaskV4]:
    rows = [
        (
            "P26",
            "gap_coupling",
            "interparticle_gap_distance_sers_enhancement",
            "interparticle gap distance",
            "electromagnetic SERS enhancement factor",
            (
                "How does interparticle gap distance relate to electromagnetic "
                "SERS enhancement factor in coupled plasmonic nanostructures?"
            ),
        ),
        (
            "P27",
            "resonance_detuning",
            "excitation_lspr_detuning_sers_intensity",
            "excitation wavelength detuning from localized surface plasmon resonance",
            "SERS signal intensity",
            (
                "How does excitation wavelength detuning from localized surface "
                "plasmon resonance relate to SERS signal intensity?"
            ),
        ),
        (
            "P28",
            "aggregation_topology",
            "aggregate_connectivity_hotspot_density",
            "nanoparticle aggregate connectivity",
            "electromagnetic hotspot density",
            (
                "How does nanoparticle aggregate connectivity relate to "
                "electromagnetic hotspot density in SERS nanostructures?"
            ),
        ),
    ]
    return [
        ProspectiveDecompositionProvenanceTaskV4(
            case_id=case_id,
            design_axis=axis,
            relation_family=family,
            source=source,
            target=target,
            question=question,
        )
        for case_id, axis, family, source, target, question in rows
    ]


def build_p26_p28_spec_from_v3_infrastructure(
    *,
    infrastructure_freeze: ProspectiveAuthorityCampaignFreezeV3,
    campaign_root: str,
) -> ProspectiveDecompositionProvenanceSpecV4:
    if infrastructure_freeze.case_ids != [
        "P21",
        "P22",
        "P23",
        "P24",
        "P25",
    ]:
        raise ValueError("v4 diagnostic infrastructure source must be P21-P25 v3")
    if len(infrastructure_freeze.tasks) != 5:
        raise ValueError("v4 diagnostic infrastructure source must contain five tasks")

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
                raise ValueError(
                    "v4 diagnostic infrastructure is non-uniform for " + field
                )

    tasks = default_p26_p28_tasks()
    old_families = {
        row.relation_family for row in infrastructure_freeze.tasks
    }
    new_families = {row.relation_family for row in tasks}
    overlap = sorted(old_families & new_families)
    if overlap:
        raise ValueError(
            "v4 diagnostic relation families overlap P21-P25: "
            + repr(overlap)
        )

    return ProspectiveDecompositionProvenanceSpecV4(
        campaign_name="P26_P28_decomposition_provenance_v4",
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


def build_prospective_decomposition_provenance_freeze_v4(
    *,
    spec: ProspectiveDecompositionProvenanceSpecV4,
    source_spec_sha256: str,
    infrastructure_freeze: ProspectiveAuthorityCampaignFreezeV3,
    infrastructure_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_file_sha256: str,
    repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
) -> ProspectiveDecompositionProvenanceFreezeV4:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "prospective decomposition provenance v4 freeze requires "
            "a clean tracked worktree"
        )
    if infrastructure_freeze.freeze_id != spec.infrastructure_source_freeze_id:
        raise ValueError("v4 diagnostic spec/infrastructure freeze ID mismatch")
    if (
        infrastructure_freeze.freeze_sha256
        != spec.infrastructure_source_freeze_sha256
    ):
        raise ValueError("v4 diagnostic spec/infrastructure freeze SHA mismatch")

    if (
        infrastructure_freeze.source_regeneration_unit_freeze_id
        != regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("v4 diagnostic regeneration-unit freeze ID mismatch")
    if (
        infrastructure_freeze.source_regeneration_unit_freeze_sha256
        != regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("v4 diagnostic regeneration-unit freeze SHA mismatch")

    tasks: list[FrozenProspectiveDecompositionProvenanceTaskV4] = []
    for row in spec.tasks:
        body = {
            "case_id": row.case_id,
            "run_dir": str(
                (
                    Path(spec.campaign_root).expanduser().resolve()
                    / row.case_id
                )
            ),
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
            "engineering_diagnostic_only": True,
            "scientific_validation_authority": False,
            "hypothesis_content_predeclared": False,
            "semantic_outcome_predeclared": False,
            "pre_n10_route_predeclared": False,
            "source_id_coverage_predeclared": False,
            "exact_source_rebind_recovery_predeclared": False,
        }
        digest = _sha256_json(body)
        tasks.append(
            FrozenProspectiveDecompositionProvenanceTaskV4(
                **body,
                task_id=(
                    "prospective_decomposition_provenance_v4_task:"
                    + digest[:20]
                ),
                task_sha256=digest,
            )
        )

    body = {
        "schema_version": "prospective-decomposition-provenance-freeze-v4",
        "source_spec_sha256": source_spec_sha256,
        "campaign_name": spec.campaign_name,
        "campaign_root": spec.campaign_root,
        "repository_head_sha": repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "infrastructure_source_freeze_id": infrastructure_freeze.freeze_id,
        "infrastructure_source_freeze_sha256": infrastructure_freeze.freeze_sha256,
        "infrastructure_source_freeze_file_sha256": (
            infrastructure_freeze_file_sha256
        ),
        "source_regeneration_unit_freeze_id": regeneration_unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256": (
            regeneration_unit_freeze.freeze_sha256
        ),
        "source_regeneration_unit_freeze_file_sha256": (
            regeneration_unit_freeze_file_sha256
        ),
        "tasks": [row.model_dump(mode="json") for row in tasks],
        "case_ids": ["P26", "P27", "P28"],
        "task_count": 3,
        "design_axis_counts": dict(
            sorted(Counter(row.design_axis for row in tasks).items())
        ),
        "diagnostic_goal": (
            "REGENERATION_DECOMPOSITION_PROVENANCE_OBSERVABILITY"
        ),
        "engineering_diagnostic_only": True,
        "scientific_validation_authority": False,
        "required_regeneration_decomposition_sidecar_filename": (
            "claim_decomposition.sanitization_audit.json"
        ),
        "required_regeneration_decomposition_sidecar_schema": (
            "pre-n10-regeneration-decomposition-sanitization-audit-v1"
        ),
        "sidecar_required_if_regeneration_decomposition_executes": True,
        "sidecar_missing_is_diagnostic_failure": True,
        "sidecar_has_scientific_authority": False,
        "initial_e2e_cutpoint_required": True,
        "initial_e2e_cutpoint_flag": "--stop-after-initial-semantic",
        "initial_external_novelty_allowed": False,
        "initial_n9_allowed": False,
        "initial_n10_allowed": False,
        "initial_refinement_allowed": False,
        "downstream_campaign_schema": "pre-n10-prospective-campaign-v1",
        "downstream_campaign_stage_order": list(CAMPAIGN_STAGE_ORDER),
        "provider_request": infrastructure_freeze.provider_request,
        "results_per_query": infrastructure_freeze.results_per_query,
        "diagnostic_goal_motivated_by_prior_v3_outcomes": True,
        "prior_v3_outcome_artifacts_consumed_by_freeze_builder": False,
        "prior_v3_task_definitions_used_only_for_nonoverlap_guard": True,
        "infrastructure_and_model_policy_only_inherited_from_v3": True,
        "source_tasks_frozen_before_v4_execution": True,
        "model_policy_frozen_before_v4_execution": True,
        "initial_cutpoint_frozen_before_v4_execution": True,
        "downstream_authority_path_frozen_before_v4_execution": True,
        "v4_initial_outputs_observed_before_freeze": False,
        "v4_semantic_outputs_observed_before_freeze": False,
        "v4_primary_route_outputs_observed_before_freeze": False,
        "v4_regeneration_outputs_observed_before_freeze": False,
        "v4_source_id_coverage_observed_before_freeze": False,
        "failed_or_terminal_case_replacement_allowed": False,
        "post_freeze_task_edit_allowed": False,
        "post_freeze_model_policy_edit_allowed": False,
        "post_freeze_cutpoint_edit_allowed": False,
        "post_freeze_downstream_path_edit_allowed": False,
        "later_case_adaptation_allowed": False,
        "second_regeneration_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveDecompositionProvenanceFreezeV4(
        **body,
        freeze_id=(
            "prospective_decomposition_provenance_freeze_v4:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


__all__ = [
    "CaseIdV4",
    "DesignAxisV4",
    "FrozenProspectiveDecompositionProvenanceTaskV4",
    "ProspectiveDecompositionProvenanceFreezeV4",
    "ProspectiveDecompositionProvenanceSpecV4",
    "ProspectiveDecompositionProvenanceTaskV4",
    "build_p26_p28_spec_from_v3_infrastructure",
    "build_prospective_decomposition_provenance_freeze_v4",
    "default_p26_p28_tasks",
    "sha256_file",
]
