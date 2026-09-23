from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


RouterAction = Literal[
    "PROCEED_TO_LITERAL_ENDPOINT_BINDING",
    "ZERO_DELTA_SPECIFICATION_REPAIR",
    "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE",
    "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE",
]


class ProspectiveRepairRegenerationRouterPolicy(StrictModel):
    schema_version: Literal[
        "preverifier-repair-regeneration-router-policy-v1"
    ] = "preverifier-repair-regeneration-router-policy-v1"

    gate_schema_version: Literal[
        "preverifier-contract-gate-v2"
    ] = "preverifier-contract-gate-v2"

    proceed_route: Literal[
        "PROCEED_TO_LITERAL_ENDPOINT_BINDING"
    ] = "PROCEED_TO_LITERAL_ENDPOINT_BINDING"

    specification_repair_route: Literal[
        "ZERO_DELTA_SPECIFICATION_REPAIR"
    ] = "ZERO_DELTA_SPECIFICATION_REPAIR"

    source_contract_route: Literal[
        "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE"
    ] = "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE"

    decompose_route: Literal[
        "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE"
    ] = "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE"

    max_specification_repair_attempts_per_claim: Literal[1] = 1
    max_source_alignment_attempts_per_claim: Literal[1] = 1
    max_atomic_decomposition_attempts_per_hypothesis: Literal[1] = 1
    max_regeneration_attempts_per_hypothesis: Literal[1] = 1

    source_alignment_may_only_reuse_existing_hypothesis_card_surfaces: Literal[
        True
    ] = True
    specification_repair_requires_zero_scientific_delta: Literal[True] = True
    source_alignment_requires_zero_scientific_delta: Literal[True] = True
    atomic_decomposition_requires_explicit_source_supported_components: Literal[
        True
    ] = True

    regeneration_may_change_scientific_commitment: Literal[True] = True
    regeneration_requires_new_lineage: Literal[True] = True
    repair_requires_new_lineage: Literal[True] = True

    second_repair_after_failed_reentry_allowed: Literal[False] = False
    repair_after_regeneration_allowed: Literal[False] = False
    regeneration_after_novelty_verdict_in_same_campaign_allowed: Literal[
        False
    ] = False

    novelty_outcome_may_influence_preverifier_route: Literal[False] = False
    verifier_outcome_may_influence_preverifier_route: Literal[False] = False
    external_novelty_outcome_may_influence_preverifier_route: Literal[
        False
    ] = False
    later_case_settings_may_adapt_to_earlier_case_outcomes: Literal[
        False
    ] = False
    failed_case_replacement_allowed: Literal[False] = False

    def route_for_hint(self, hint: str) -> RouterAction:
        mapping: dict[str, RouterAction] = {
            "PROCEED_TO_LITERAL_ENDPOINT_BINDING": self.proceed_route,
            "SPECIFICATION_REPAIR_REVIEW": self.specification_repair_route,
            "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW":
                self.source_contract_route,
            "DECOMPOSE_OR_REGENERATE_REVIEW": self.decompose_route,
        }
        if hint not in mapping:
            raise ValueError("unsupported gate-v2 router hint: " + hint)
        return mapping[hint]


class ProspectiveRoutedSourceTaskDefinition(StrictModel):
    case_id: Literal["P11", "P12", "P13", "P14", "P15"]
    relation_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    stop: str | None = None
    target: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: str = Field(default="explain_connection", min_length=1)


class ProspectiveRoutedCampaignSpec(StrictModel):
    schema_version: Literal[
        "prospective-routed-campaign-spec-v1"
    ] = "prospective-routed-campaign-spec-v1"

    campaign_name: str = Field(min_length=1)
    campaign_root: str = Field(min_length=1)

    domain_profile_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    data_root: str = Field(min_length=1)
    semantic_roots: list[str] = Field(min_length=1)

    generation_model: str = Field(min_length=1)
    critic_model: str = Field(min_length=1)

    tasks: list[ProspectiveRoutedSourceTaskDefinition] = Field(
        min_length=5,
        max_length=5,
    )

    @model_validator(mode="after")
    def validate_campaign(self) -> "ProspectiveRoutedCampaignSpec":
        expected = ["P11", "P12", "P13", "P14", "P15"]
        observed = [row.case_id for row in self.tasks]
        if observed != expected:
            raise ValueError(
                "routed prospective source tasks must be exactly P11-P15"
            )

        families = [row.relation_family for row in self.tasks]
        if len(set(families)) != len(families):
            raise ValueError("P11-P15 relation families must be unique")

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
            raise ValueError("P11-P15 source tasks must be distinct")

        if len(set(self.semantic_roots)) != len(self.semantic_roots):
            raise ValueError("semantic_roots must be unique")
        return self


class FrozenProspectiveRoutedTask(StrictModel):
    case_id: Literal["P11", "P12", "P13", "P14", "P15"]
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
    def validate_hash(self) -> "FrozenProspectiveRoutedTask":
        body = self.model_dump(mode="json")
        observed_id = body.pop("task_id")
        observed_sha = body.pop("task_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("routed task SHA mismatch")
        if observed_id != "prospective_routed_task:" + expected_sha[:20]:
            raise ValueError("routed task ID mismatch")
        return self


class ProspectiveRoutedCampaignFreeze(StrictModel):
    schema_version: Literal[
        "prospective-routed-campaign-freeze-v1"
    ] = "prospective-routed-campaign-freeze-v1"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    campaign_name: str
    campaign_root: str
    repository_head_sha: str
    repository_tracked_worktree_dirty: Literal[False] = False

    router_policy: ProspectiveRepairRegenerationRouterPolicy
    tasks: list[FrozenProspectiveRoutedTask]
    case_ids: list[str]
    task_count: Literal[5] = 5
    relation_family_counts: dict[str, int]

    source_tasks_frozen_before_generation: Literal[True] = True
    router_policy_frozen_before_generation: Literal[True] = True
    gate_v2_outcomes_observed_before_freeze: Literal[False] = False
    repair_outputs_observed_before_freeze: Literal[False] = False
    regeneration_outputs_observed_before_freeze: Literal[False] = False
    endpoint_binding_observed_before_freeze: Literal[False] = False
    verifier_outcomes_observed_before_freeze: Literal[False] = False
    external_novelty_outcomes_observed_before_freeze: Literal[False] = False

    legacy_case_reuse_allowed: Literal[False] = False
    failed_or_abstained_case_replacement_allowed: Literal[False] = False
    post_freeze_task_edit_allowed: Literal[False] = False
    post_freeze_router_policy_edit_allowed: Literal[False] = False
    later_case_adaptation_allowed: Literal[False] = False

    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(self) -> "ProspectiveRoutedCampaignFreeze":
        expected = ["P11", "P12", "P13", "P14", "P15"]
        if self.case_ids != expected:
            raise ValueError("routed freeze case IDs must be P11-P15")
        if [row.case_id for row in self.tasks] != expected:
            raise ValueError("routed task order must be P11-P15")

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
            raise ValueError("routed campaign freeze SHA mismatch")
        if observed_id != (
            "prospective_routed_campaign_freeze:" + expected_sha[:20]
        ):
            raise ValueError("routed campaign freeze ID mismatch")
        return self


def build_prospective_routed_campaign_freeze(
    *,
    spec: ProspectiveRoutedCampaignSpec,
    source_spec_sha256: str,
    repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
) -> ProspectiveRoutedCampaignFreeze:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "routed campaign freeze requires a clean tracked worktree"
        )

    root = str(Path(spec.campaign_root).expanduser().resolve())
    data_root = str(Path(spec.data_root).expanduser().resolve())
    semantic_roots = [
        str(Path(value).expanduser().resolve())
        for value in spec.semantic_roots
    ]
    policy = ProspectiveRepairRegenerationRouterPolicy()

    tasks: list[FrozenProspectiveRoutedTask] = []
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
            FrozenProspectiveRoutedTask(
                **body,
                task_id="prospective_routed_task:" + digest[:20],
                task_sha256=digest,
            )
        )

    counts = Counter(row.relation_family for row in tasks)
    body = {
        "schema_version": "prospective-routed-campaign-freeze-v1",
        "source_spec_sha256": source_spec_sha256,
        "campaign_name": spec.campaign_name,
        "campaign_root": root,
        "repository_head_sha": repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "router_policy": policy.model_dump(mode="json"),
        "tasks": [row.model_dump(mode="json") for row in tasks],
        "case_ids": ["P11", "P12", "P13", "P14", "P15"],
        "task_count": 5,
        "relation_family_counts": dict(sorted(counts.items())),
        "source_tasks_frozen_before_generation": True,
        "router_policy_frozen_before_generation": True,
        "gate_v2_outcomes_observed_before_freeze": False,
        "repair_outputs_observed_before_freeze": False,
        "regeneration_outputs_observed_before_freeze": False,
        "endpoint_binding_observed_before_freeze": False,
        "verifier_outcomes_observed_before_freeze": False,
        "external_novelty_outcomes_observed_before_freeze": False,
        "legacy_case_reuse_allowed": False,
        "failed_or_abstained_case_replacement_allowed": False,
        "post_freeze_task_edit_allowed": False,
        "post_freeze_router_policy_edit_allowed": False,
        "later_case_adaptation_allowed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRoutedCampaignFreeze(
        **body,
        freeze_id="prospective_routed_campaign_freeze:" + digest[:20],
        freeze_sha256=digest,
    )


__all__ = [
    "FrozenProspectiveRoutedTask",
    "ProspectiveRepairRegenerationRouterPolicy",
    "ProspectiveRoutedCampaignFreeze",
    "ProspectiveRoutedCampaignSpec",
    "ProspectiveRoutedSourceTaskDefinition",
    "RouterAction",
    "build_prospective_routed_campaign_freeze",
]
