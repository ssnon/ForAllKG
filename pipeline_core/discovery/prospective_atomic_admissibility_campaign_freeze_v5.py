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
    ProspectiveAuthorityModelPolicyV3,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    ProspectiveDecompositionProvenanceFreezeV4,
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


CaseIdV5 = Literal["P29", "P30", "P31", "P32", "P33"]
DesignAxisV5 = Literal[
    "particle_shape",
    "alloy_composition",
    "surface_coverage",
    "excitation_power",
    "crystal_surface",
]


class ProspectiveAtomicAdmissibilityTaskV5(StrictModel):
    case_id: CaseIdV5
    design_axis: DesignAxisV5
    relation_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    stop: str | None = None
    target: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: Literal["explain_connection"] = "explain_connection"


class ProspectiveAtomicAdmissibilitySpecV5(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-spec-v5"
    ] = "prospective-atomic-admissibility-spec-v5"

    campaign_name: str = Field(min_length=1)
    campaign_root: str = Field(min_length=1)

    domain_profile_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    data_root: str = Field(min_length=1)
    semantic_roots: list[str] = Field(min_length=1)
    model_policy: ProspectiveAuthorityModelPolicyV3

    tasks: list[ProspectiveAtomicAdmissibilityTaskV5] = Field(
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

    tasks_defined_before_p29_p33_execution: Literal[True] = True
    expected_legacy_outcomes_predeclared: Literal[False] = False
    expected_neutral_outcomes_predeclared: Literal[False] = False
    expected_source_reference_outcomes_predeclared: Literal[False] = False
    expected_semantic_fidelity_outcomes_predeclared: Literal[False] = False

    prior_v4_outcome_artifacts_consumed_by_builder: Literal[False] = False
    prior_v4_task_definitions_used_only_for_nonoverlap_guard: Literal[
        True
    ] = True
    infrastructure_and_model_policy_only_inherited_from_v4: Literal[
        True
    ] = True

    initial_generation_cutpoint: Literal[
        "AFTER_INITIAL_SEMANTIC_BEFORE_EXTERNAL_NOVELTY"
    ] = "AFTER_INITIAL_SEMANTIC_BEFORE_EXTERNAL_NOVELTY"
    downstream_campaign_schema: Literal[
        "pre-n10-prospective-campaign-v1"
    ] = "pre-n10-prospective-campaign-v1"

    prospective_evidence_collection: Literal[True] = True
    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_spec(
        self,
    ) -> "ProspectiveAtomicAdmissibilitySpecV5":
        expected_cases = ["P29", "P30", "P31", "P32", "P33"]
        expected_axes = [
            "particle_shape",
            "alloy_composition",
            "surface_coverage",
            "excitation_power",
            "crystal_surface",
        ]
        if [row.case_id for row in self.tasks] != expected_cases:
            raise ValueError(
                "v5 atomic admissibility source tasks must be exactly P29-P33"
            )
        if [row.design_axis for row in self.tasks] != expected_axes:
            raise ValueError(
                "v5 atomic admissibility design-axis order mismatch"
            )
        families = [row.relation_family for row in self.tasks]
        if len(set(families)) != len(families):
            raise ValueError("P29-P33 relation families must be unique")
        surfaces = [
            (
                row.source.casefold().strip(),
                (row.stop or "").casefold().strip(),
                row.target.casefold().strip(),
                row.question.casefold().strip(),
            )
            for row in self.tasks
        ]
        if len(set(surfaces)) != len(surfaces):
            raise ValueError("P29-P33 source tasks must be distinct")
        if len(set(self.semantic_roots)) != len(self.semantic_roots):
            raise ValueError("semantic_roots must be unique")
        return self


class FrozenProspectiveAtomicAdmissibilityTaskV5(StrictModel):
    case_id: CaseIdV5
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    design_axis: DesignAxisV5
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
    source_reference_outcome_predeclared: Literal[False] = False
    semantic_fidelity_outcome_predeclared: Literal[False] = False

    engineering_diagnostic_only: Literal[True] = True
    scientific_validation_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_task(
        self,
    ) -> "FrozenProspectiveAtomicAdmissibilityTaskV5":
        body = self.model_dump(mode="json")
        observed_id = body.pop("task_id")
        observed_sha = body.pop("task_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("v5 atomic admissibility task SHA mismatch")
        if observed_id != (
            "prospective_atomic_admissibility_v5_task:"
            + expected_sha[:20]
        ):
            raise ValueError("v5 atomic admissibility task ID mismatch")
        return self


class ProspectiveAtomicAdmissibilityFreezeV5(StrictModel):
    schema_version: Literal[
        "prospective-atomic-admissibility-freeze-v5"
    ] = "prospective-atomic-admissibility-freeze-v5"

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

    tasks: list[FrozenProspectiveAtomicAdmissibilityTaskV5]
    case_ids: list[str]
    task_count: Literal[5] = 5
    design_axis_counts: dict[str, int]

    prospective_goal: Literal[
        "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10"
    ] = "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10"

    source_tasks_frozen_before_p29_p33_execution: Literal[True] = True
    model_policy_frozen_before_p29_p33_execution: Literal[True] = True
    initial_cutpoint_frozen_before_p29_p33_execution: Literal[True] = True
    downstream_campaign_stage_order: list[str]

    p29_p33_initial_outputs_observed_before_freeze: Literal[False] = False
    p29_p33_semantic_outputs_observed_before_freeze: Literal[False] = False
    p29_p33_legacy_vpre_outputs_observed_before_freeze: Literal[False] = False
    p29_p33_neutral_pre_n10_outputs_observed_before_freeze: Literal[
        False
    ] = False
    p29_p33_source_reference_outcomes_observed_before_freeze: Literal[
        False
    ] = False
    p29_p33_semantic_fidelity_outcomes_observed_before_freeze: Literal[
        False
    ] = False
    p29_p33_external_novelty_outputs_observed_before_freeze: Literal[
        False
    ] = False
    p29_p33_n10_outputs_observed_before_freeze: Literal[False] = False

    prior_v4_outcome_artifacts_consumed_by_freeze_builder: Literal[
        False
    ] = False
    prior_v4_task_definitions_used_only_for_nonoverlap_guard: Literal[
        True
    ] = True
    infrastructure_and_model_policy_only_inherited_from_v4: Literal[
        True
    ] = True

    post_freeze_task_edit_allowed: Literal[False] = False
    post_freeze_model_policy_edit_allowed: Literal[False] = False
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
    ) -> "ProspectiveAtomicAdmissibilityFreezeV5":
        expected = ["P29", "P30", "P31", "P32", "P33"]
        if self.case_ids != expected:
            raise ValueError(
                "v5 atomic admissibility freeze case IDs must be P29-P33"
            )
        if [row.case_id for row in self.tasks] != expected:
            raise ValueError(
                "v5 atomic admissibility frozen task order must be P29-P33"
            )
        if len({row.task_id for row in self.tasks}) != 5:
            raise ValueError(
                "v5 atomic admissibility frozen task IDs must be unique"
            )
        if self.downstream_campaign_stage_order != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError(
                "v5 atomic admissibility downstream stage order mismatch"
            )

        expected_axis_counts = Counter(row.design_axis for row in self.tasks)
        if dict(sorted(expected_axis_counts.items())) != dict(
            sorted(self.design_axis_counts.items())
        ):
            raise ValueError(
                "v5 atomic admissibility design-axis counts mismatch"
            )
        if any(count != 1 for count in self.design_axis_counts.values()):
            raise ValueError(
                "each v5 atomic admissibility design axis must appear once"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "prospective atomic admissibility v5 freeze SHA mismatch"
            )
        if observed_id != (
            "prospective_atomic_admissibility_freeze_v5:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "prospective atomic admissibility v5 freeze ID mismatch"
            )
        return self


def default_p29_p33_tasks() -> list[ProspectiveAtomicAdmissibilityTaskV5]:
    rows = [
        (
            "P29",
            "particle_shape",
            "aspect_ratio_longitudinal_resonance",
            "nanoparticle aspect ratio",
            "longitudinal localized surface plasmon resonance wavelength",
            (
                "How does nanoparticle aspect ratio relate to the longitudinal "
                "localized surface plasmon resonance wavelength in anisotropic "
                "SERS nanostructures?"
            ),
        ),
        (
            "P30",
            "alloy_composition",
            "au_ag_composition_resonance_position",
            "Au/Ag composition ratio",
            "localized surface plasmon resonance wavelength",
            (
                "How does Au/Ag composition ratio relate to localized surface "
                "plasmon resonance wavelength in alloyed plasmonic SERS "
                "nanostructures?"
            ),
        ),
        (
            "P31",
            "surface_coverage",
            "analyte_surface_coverage_sers_saturation",
            "analyte surface coverage",
            "SERS signal saturation behavior",
            (
                "How does analyte surface coverage relate to SERS signal "
                "saturation behavior at plasmonic hotspots?"
            ),
        ),
        (
            "P32",
            "excitation_power",
            "excitation_power_temporal_signal_stability",
            "excitation power density",
            "temporal SERS signal stability",
            (
                "How does excitation power density relate to temporal SERS "
                "signal stability during repeated or sustained illumination?"
            ),
        ),
        (
            "P33",
            "crystal_surface",
            "facet_exposure_adsorption_configuration",
            "exposed metal crystallographic facet",
            "analyte adsorption configuration",
            (
                "How does exposed metal crystallographic facet relate to "
                "analyte adsorption configuration on plasmonic SERS substrates?"
            ),
        ),
    ]
    return [
        ProspectiveAtomicAdmissibilityTaskV5(
            case_id=case_id,
            design_axis=axis,
            relation_family=family,
            source=source,
            target=target,
            question=question,
        )
        for case_id, axis, family, source, target, question in rows
    ]


def build_p29_p33_spec_from_v4_infrastructure(
    *,
    infrastructure_freeze: ProspectiveDecompositionProvenanceFreezeV4,
    campaign_root: str,
) -> ProspectiveAtomicAdmissibilitySpecV5:
    if infrastructure_freeze.case_ids != ["P26", "P27", "P28"]:
        raise ValueError(
            "v5 atomic admissibility infrastructure source must be P26-P28 v4"
        )
    if len(infrastructure_freeze.tasks) != 3:
        raise ValueError(
            "v5 atomic admissibility infrastructure must contain three tasks"
        )

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
                    "v5 infrastructure is non-uniform for " + field
                )

    tasks = default_p29_p33_tasks()
    old_families = {
        row.relation_family for row in infrastructure_freeze.tasks
    }
    new_families = {row.relation_family for row in tasks}
    overlap = sorted(old_families & new_families)
    if overlap:
        raise ValueError(
            "v5 relation families overlap P26-P28: " + repr(overlap)
        )

    old_surfaces = {
        (
            row.source.casefold().strip(),
            (row.stop or "").casefold().strip(),
            row.target.casefold().strip(),
            row.question.casefold().strip(),
        )
        for row in infrastructure_freeze.tasks
    }
    new_surfaces = {
        (
            row.source.casefold().strip(),
            (row.stop or "").casefold().strip(),
            row.target.casefold().strip(),
            row.question.casefold().strip(),
        )
        for row in tasks
    }
    if old_surfaces & new_surfaces:
        raise ValueError("v5 source-task surfaces overlap P26-P28")

    return ProspectiveAtomicAdmissibilitySpecV5(
        campaign_name="P29_P33_atomic_admissibility_v5",
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


def build_prospective_atomic_admissibility_freeze_v5(
    *,
    spec: ProspectiveAtomicAdmissibilitySpecV5,
    source_spec_sha256: str,
    infrastructure_freeze: ProspectiveDecompositionProvenanceFreezeV4,
    infrastructure_freeze_file_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_file_sha256: str,
    repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
) -> ProspectiveAtomicAdmissibilityFreezeV5:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "prospective atomic admissibility v5 freeze requires "
            "a clean tracked worktree"
        )
    if infrastructure_freeze.freeze_id != spec.infrastructure_source_freeze_id:
        raise ValueError("v5 spec/infrastructure freeze ID mismatch")
    if (
        infrastructure_freeze.freeze_sha256
        != spec.infrastructure_source_freeze_sha256
    ):
        raise ValueError("v5 spec/infrastructure freeze SHA mismatch")

    if (
        infrastructure_freeze.source_regeneration_unit_freeze_id
        != regeneration_unit_freeze.freeze_id
    ):
        raise ValueError("v5 regeneration-unit freeze ID mismatch")
    if (
        infrastructure_freeze.source_regeneration_unit_freeze_sha256
        != regeneration_unit_freeze.freeze_sha256
    ):
        raise ValueError("v5 regeneration-unit freeze SHA mismatch")

    tasks: list[FrozenProspectiveAtomicAdmissibilityTaskV5] = []
    root = Path(spec.campaign_root).expanduser().resolve()

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
            "source_reference_outcome_predeclared": False,
            "semantic_fidelity_outcome_predeclared": False,
            "engineering_diagnostic_only": True,
            "scientific_validation_authority": False,
        }
        digest = _sha256_json(body)
        tasks.append(
            FrozenProspectiveAtomicAdmissibilityTaskV5(
                **body,
                task_id=(
                    "prospective_atomic_admissibility_v5_task:"
                    + digest[:20]
                ),
                task_sha256=digest,
            )
        )

    body = {
        "schema_version": "prospective-atomic-admissibility-freeze-v5",
        "source_spec_sha256": source_spec_sha256,
        "campaign_name": spec.campaign_name,
        "campaign_root": str(root),
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
        "case_ids": ["P29", "P30", "P31", "P32", "P33"],
        "task_count": 5,
        "design_axis_counts": dict(
            sorted(Counter(row.design_axis for row in tasks).items())
        ),
        "prospective_goal": "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10",
        "source_tasks_frozen_before_p29_p33_execution": True,
        "model_policy_frozen_before_p29_p33_execution": True,
        "initial_cutpoint_frozen_before_p29_p33_execution": True,
        "downstream_campaign_stage_order": list(CAMPAIGN_STAGE_ORDER),
        "p29_p33_initial_outputs_observed_before_freeze": False,
        "p29_p33_semantic_outputs_observed_before_freeze": False,
        "p29_p33_legacy_vpre_outputs_observed_before_freeze": False,
        "p29_p33_neutral_pre_n10_outputs_observed_before_freeze": False,
        "p29_p33_source_reference_outcomes_observed_before_freeze": False,
        "p29_p33_semantic_fidelity_outcomes_observed_before_freeze": False,
        "p29_p33_external_novelty_outputs_observed_before_freeze": False,
        "p29_p33_n10_outputs_observed_before_freeze": False,
        "prior_v4_outcome_artifacts_consumed_by_freeze_builder": False,
        "prior_v4_task_definitions_used_only_for_nonoverlap_guard": True,
        "infrastructure_and_model_policy_only_inherited_from_v4": True,
        "post_freeze_task_edit_allowed": False,
        "post_freeze_model_policy_edit_allowed": False,
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

    return ProspectiveAtomicAdmissibilityFreezeV5(
        **body,
        freeze_id=(
            "prospective_atomic_admissibility_freeze_v5:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


__all__ = [
    "CaseIdV5",
    "DesignAxisV5",
    "FrozenProspectiveAtomicAdmissibilityTaskV5",
    "ProspectiveAtomicAdmissibilityFreezeV5",
    "ProspectiveAtomicAdmissibilitySpecV5",
    "ProspectiveAtomicAdmissibilityTaskV5",
    "build_p29_p33_spec_from_v4_infrastructure",
    "build_prospective_atomic_admissibility_freeze_v5",
    "default_p29_p33_tasks",
    "sha256_file",
]
