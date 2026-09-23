from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    ProspectiveRegenerationDownstreamV2Freeze,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRepairRegenerationRouterPolicy,
    ProspectiveRoutedCampaignFreeze,
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


CaseIdV2 = Literal["P16", "P17", "P18", "P19", "P20"]


class ProspectiveRoutedSourceTaskDefinitionV2(StrictModel):
    case_id: CaseIdV2
    relation_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    stop: str | None = None
    target: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: str = Field(default="explain_connection", min_length=1)


class ProspectiveRoutedCampaignSpecV2(StrictModel):
    schema_version: Literal[
        "prospective-routed-campaign-spec-v2"
    ] = "prospective-routed-campaign-spec-v2"

    campaign_name: str = Field(min_length=1)
    campaign_root: str = Field(min_length=1)

    domain_profile_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    data_root: str = Field(min_length=1)
    semantic_roots: list[str] = Field(min_length=1)

    generation_model: str = Field(min_length=1)
    critic_model: str = Field(min_length=1)

    tasks: list[ProspectiveRoutedSourceTaskDefinitionV2] = Field(
        min_length=5,
        max_length=5,
    )

    infrastructure_source_freeze_id: str
    infrastructure_source_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    tasks_defined_without_prior_scientific_outcomes: Literal[True] = True
    infrastructure_only_inherited_from_prior_campaign: Literal[True] = True

    @model_validator(mode="after")
    def validate_campaign(self) -> "ProspectiveRoutedCampaignSpecV2":
        expected = ["P16", "P17", "P18", "P19", "P20"]
        observed = [row.case_id for row in self.tasks]
        if observed != expected:
            raise ValueError(
                "routed prospective v2 source tasks must be exactly P16-P20"
            )

        families = [row.relation_family for row in self.tasks]
        if len(set(families)) != len(families):
            raise ValueError("P16-P20 relation families must be unique")

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
            raise ValueError("P16-P20 source tasks must be distinct")

        if len(set(self.semantic_roots)) != len(self.semantic_roots):
            raise ValueError("semantic_roots must be unique")
        return self


class FrozenProspectiveRoutedTaskV2(StrictModel):
    case_id: CaseIdV2
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    run_dir: str
    relation_family: str
    source: str
    stop: str | None = None
    target: str
    question: str
    objective: str

    domain_profile_id: str
    corpus_id: str
    data_root: str
    semantic_roots: list[str]
    generation_model: str
    critic_model: str

    source_task_definition_only: Literal[True] = True
    hypothesis_content_predeclared: Literal[False] = False
    router_outcome_predeclared: Literal[False] = False
    verifier_outcome_predeclared: Literal[False] = False

    @model_validator(mode="after")
    def validate_hash(self) -> "FrozenProspectiveRoutedTaskV2":
        body = self.model_dump(mode="json")
        observed_id = body.pop("task_id")
        observed_sha = body.pop("task_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("routed v2 task SHA mismatch")
        if observed_id != "prospective_routed_v2_task:" + expected_sha[:20]:
            raise ValueError("routed v2 task ID mismatch")
        return self


class ProspectiveRoutedCampaignFreezeV2(StrictModel):
    schema_version: Literal[
        "prospective-routed-campaign-freeze-v2"
    ] = "prospective-routed-campaign-freeze-v2"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    campaign_name: str
    campaign_root: str
    repository_head_sha: str
    repository_tracked_worktree_dirty: Literal[False] = False

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

    infrastructure_source_freeze_id: str
    infrastructure_source_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    router_policy: ProspectiveRepairRegenerationRouterPolicy
    tasks: list[FrozenProspectiveRoutedTaskV2]
    case_ids: list[str]
    task_count: Literal[5] = 5
    relation_family_counts: dict[str, int]

    source_tasks_frozen_before_generation: Literal[True] = True
    router_policy_frozen_before_generation: Literal[True] = True
    regeneration_unit_v2_frozen_before_source_tasks: Literal[True] = True
    regeneration_downstream_v2_frozen_before_source_tasks: Literal[True] = True

    gate_v2_outcomes_observed_before_freeze: Literal[False] = False
    repair_outputs_observed_before_freeze: Literal[False] = False
    regeneration_outputs_observed_before_freeze: Literal[False] = False
    endpoint_binding_observed_before_freeze: Literal[False] = False
    verifier_outcomes_observed_before_freeze: Literal[False] = False
    external_novelty_outcomes_observed_before_freeze: Literal[False] = False

    prior_scientific_outputs_used_to_define_tasks: Literal[False] = False
    prior_campaign_used_for_infrastructure_only: Literal[True] = True

    legacy_case_reuse_allowed: Literal[False] = False
    failed_or_abstained_case_replacement_allowed: Literal[False] = False
    post_freeze_task_edit_allowed: Literal[False] = False
    post_freeze_router_policy_edit_allowed: Literal[False] = False
    post_freeze_regeneration_contract_edit_allowed: Literal[False] = False
    post_freeze_downstream_contract_edit_allowed: Literal[False] = False
    later_case_adaptation_allowed: Literal[False] = False

    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(self) -> "ProspectiveRoutedCampaignFreezeV2":
        expected = ["P16", "P17", "P18", "P19", "P20"]
        if self.case_ids != expected:
            raise ValueError("routed v2 freeze case IDs must be P16-P20")
        if [row.case_id for row in self.tasks] != expected:
            raise ValueError("routed v2 task order must be P16-P20")

        counts = Counter(row.relation_family for row in self.tasks)
        if dict(sorted(counts.items())) != dict(
            sorted(self.relation_family_counts.items())
        ):
            raise ValueError("relation_family_counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("routed v2 campaign freeze SHA mismatch")
        if observed_id != (
            "prospective_routed_campaign_freeze_v2:" + expected_sha[:20]
        ):
            raise ValueError("routed v2 campaign freeze ID mismatch")
        return self


def default_p16_p20_tasks() -> list[ProspectiveRoutedSourceTaskDefinitionV2]:
    rows = [
        (
            "P16",
            "dielectric_environment_resonance_linewidth",
            "local dielectric environment",
            "plasmon resonance linewidth",
            "How does local dielectric environment relate to plasmon resonance linewidth in SERS nanostructures?",
        ),
        (
            "P17",
            "tip_curvature_hotspot_localization",
            "nanostructure tip curvature",
            "electromagnetic hotspot localization",
            "How does nanostructure tip curvature relate to electromagnetic hotspot localization in SERS substrates?",
        ),
        (
            "P18",
            "adsorption_orientation_polarization_response",
            "analyte adsorption orientation",
            "polarization-dependent SERS response",
            "How does analyte adsorption orientation relate to polarization-dependent SERS response?",
        ),
        (
            "P19",
            "spacing_disorder_spatial_uniformity",
            "interparticle spacing disorder",
            "spatial SERS uniformity",
            "How does interparticle spacing disorder relate to spatial SERS uniformity across a substrate?",
        ),
        (
            "P20",
            "surface_charge_adsorption_affinity",
            "surface charge density",
            "analyte adsorption affinity",
            "How does surface charge density relate to analyte adsorption affinity in plasmonic SERS systems?",
        ),
    ]
    return [
        ProspectiveRoutedSourceTaskDefinitionV2(
            case_id=case_id,
            relation_family=family,
            source=source,
            target=target,
            question=question,
        )
        for case_id, family, source, target, question in rows
    ]


def build_p16_p20_spec_from_infrastructure_freeze(
    *,
    infrastructure_freeze: ProspectiveRoutedCampaignFreeze,
    campaign_root: str,
) -> ProspectiveRoutedCampaignSpecV2:
    if infrastructure_freeze.case_ids != ["P11", "P12", "P13", "P14", "P15"]:
        raise ValueError("infrastructure source freeze must be P11-P15")
    if not infrastructure_freeze.tasks:
        raise ValueError("infrastructure source freeze has no tasks")

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
                    "infrastructure source freeze has non-uniform " + field
                )

    return ProspectiveRoutedCampaignSpecV2(
        campaign_name="P16_P20_routed_v2",
        campaign_root=str(Path(campaign_root).expanduser().resolve()),
        domain_profile_id=first.domain_profile_id,
        corpus_id=first.corpus_id,
        data_root=first.data_root,
        semantic_roots=list(first.semantic_roots),
        generation_model=first.generation_model,
        critic_model=first.critic_model,
        tasks=default_p16_p20_tasks(),
        infrastructure_source_freeze_id=infrastructure_freeze.freeze_id,
        infrastructure_source_freeze_sha256=infrastructure_freeze.freeze_sha256,
    )


def build_prospective_routed_campaign_freeze_v2(
    *,
    spec: ProspectiveRoutedCampaignSpecV2,
    source_spec_sha256: str,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_file_sha256: str,
    regeneration_downstream_freeze: ProspectiveRegenerationDownstreamV2Freeze,
    regeneration_downstream_freeze_file_sha256: str,
    repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
) -> ProspectiveRoutedCampaignFreezeV2:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "routed v2 campaign freeze requires a clean tracked worktree"
        )

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

    root = str(Path(spec.campaign_root).expanduser().resolve())
    data_root = str(Path(spec.data_root).expanduser().resolve())
    semantic_roots = [
        str(Path(value).expanduser().resolve())
        for value in spec.semantic_roots
    ]
    policy = ProspectiveRepairRegenerationRouterPolicy()

    tasks: list[FrozenProspectiveRoutedTaskV2] = []
    for row in spec.tasks:
        body = {
            "case_id": row.case_id,
            "run_dir": str(Path(root) / row.case_id),
            "relation_family": row.relation_family,
            "source": row.source,
            "stop": row.stop,
            "target": row.target,
            "question": row.question,
            "objective": row.objective,
            "domain_profile_id": spec.domain_profile_id,
            "corpus_id": spec.corpus_id,
            "data_root": data_root,
            "semantic_roots": semantic_roots,
            "generation_model": spec.generation_model,
            "critic_model": spec.critic_model,
            "source_task_definition_only": True,
            "hypothesis_content_predeclared": False,
            "router_outcome_predeclared": False,
            "verifier_outcome_predeclared": False,
        }
        digest = _sha256_json(body)
        tasks.append(
            FrozenProspectiveRoutedTaskV2(
                **body,
                task_id="prospective_routed_v2_task:" + digest[:20],
                task_sha256=digest,
            )
        )

    counts = Counter(row.relation_family for row in tasks)
    body = {
        "schema_version": "prospective-routed-campaign-freeze-v2",
        "source_spec_sha256": source_spec_sha256,
        "campaign_name": spec.campaign_name,
        "campaign_root": root,
        "repository_head_sha": repository_head_sha,
        "repository_tracked_worktree_dirty": False,
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
        "infrastructure_source_freeze_id":
            spec.infrastructure_source_freeze_id,
        "infrastructure_source_freeze_sha256":
            spec.infrastructure_source_freeze_sha256,
        "router_policy": policy.model_dump(mode="json"),
        "tasks": [row.model_dump(mode="json") for row in tasks],
        "case_ids": ["P16", "P17", "P18", "P19", "P20"],
        "task_count": 5,
        "relation_family_counts": dict(sorted(counts.items())),
        "source_tasks_frozen_before_generation": True,
        "router_policy_frozen_before_generation": True,
        "regeneration_unit_v2_frozen_before_source_tasks": True,
        "regeneration_downstream_v2_frozen_before_source_tasks": True,
        "gate_v2_outcomes_observed_before_freeze": False,
        "repair_outputs_observed_before_freeze": False,
        "regeneration_outputs_observed_before_freeze": False,
        "endpoint_binding_observed_before_freeze": False,
        "verifier_outcomes_observed_before_freeze": False,
        "external_novelty_outcomes_observed_before_freeze": False,
        "prior_scientific_outputs_used_to_define_tasks": False,
        "prior_campaign_used_for_infrastructure_only": True,
        "legacy_case_reuse_allowed": False,
        "failed_or_abstained_case_replacement_allowed": False,
        "post_freeze_task_edit_allowed": False,
        "post_freeze_router_policy_edit_allowed": False,
        "post_freeze_regeneration_contract_edit_allowed": False,
        "post_freeze_downstream_contract_edit_allowed": False,
        "later_case_adaptation_allowed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRoutedCampaignFreezeV2(
        **body,
        freeze_id="prospective_routed_campaign_freeze_v2:" + digest[:20],
        freeze_sha256=digest,
    )


__all__ = [
    "CaseIdV2",
    "FrozenProspectiveRoutedTaskV2",
    "ProspectiveRoutedCampaignFreezeV2",
    "ProspectiveRoutedCampaignSpecV2",
    "ProspectiveRoutedSourceTaskDefinitionV2",
    "build_p16_p20_spec_from_infrastructure_freeze",
    "build_prospective_routed_campaign_freeze_v2",
    "default_p16_p20_tasks",
    "sha256_file",
]
