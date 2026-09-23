from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class RegenerationDownstreamStageBudgetV2(StrictModel):
    budget_unit: Literal[
        "semantic_stage_invocation"
    ] = "semantic_stage_invocation"

    deterministic_hypothesis_benchmark_max: Literal[1] = 1
    semantic_critic_stage_max: Literal[1] = 1
    external_novelty_stage_max: Literal[1] = 1
    n9_shadow_intake_stage_max: Literal[1] = 1
    n9_full_closure_stage_max: Literal[1] = 1
    n10_certification_stage_max: Literal[1] = 1
    binding_plan_build_max: Literal[1] = 1
    preverifier_gate_v2_max: Literal[1] = 1

    targeted_novelty_continuation_max: Literal[0] = 0
    novelty_refinement_max: Literal[0] = 0
    same_lineage_hypothesis_regeneration_max: Literal[0] = 0
    semantic_repair_stage_max: Literal[0] = 0

    provider_parse_retries_max: Literal[3] = 3


class ProspectiveRegenerationDownstreamV2Policy(StrictModel):
    schema_version: Literal[
        "prospective-regeneration-downstream-v2-policy"
    ] = "prospective-regeneration-downstream-v2-policy"

    source_regeneration_unit_schema: Literal[
        "prospective-regeneration-unit-v2-freeze"
    ] = "prospective-regeneration-unit-v2-freeze"

    input_authority: Literal[
        "regenerated_portfolio_plus_frozen_hypothesis_context"
    ] = "regenerated_portfolio_plus_frozen_hypothesis_context"

    upstream_graph_exploration_rerun_allowed: Literal[False] = False
    upstream_context_construction_rerun_allowed: Literal[False] = False
    upstream_discovery_axis_generation_rerun_allowed: Literal[False] = False
    initial_hypothesis_generation_rerun_allowed: Literal[False] = False
    full_e2e_rerun_allowed: Literal[False] = False

    regenerated_portfolio_mutation_allowed: Literal[False] = False
    regenerated_portfolio_id_must_remain_stable: Literal[True] = True
    frozen_hypothesis_context_id_must_remain_stable: Literal[True] = True

    semantic_hard_gate_precedes_semantic_critic: Literal[True] = True
    semantic_critic_skipped_on_hard_gate_failure: Literal[True] = True

    external_novelty_uses_regenerated_portfolio: Literal[True] = True
    external_novelty_results_per_query: Literal[12] = 12
    max_review_works_per_claim: Literal[20] = 20

    n9_shadow_intake_required: Literal[True] = True
    n9_full_closure_required: Literal[True] = True
    n10_authority_mode: Literal[
        "certification_only"
    ] = "certification_only"

    bounded_continuation_enabled: Literal[False] = False
    post_generation_refinement_enabled: Literal[False] = False

    relational_binding_plan_required: Literal[True] = True
    preverifier_contract_gate_v2_required: Literal[True] = True

    novelty_or_verifier_outcome_may_mutate_regenerated_portfolio: Literal[
        False
    ] = False
    downstream_failure_retry_allowed: Literal[False] = False
    downstream_failure_triggers_second_regeneration: Literal[False] = False

    budget: RegenerationDownstreamStageBudgetV2 = Field(
        default_factory=RegenerationDownstreamStageBudgetV2
    )


class ProspectiveRegenerationDownstreamV2Freeze(StrictModel):
    schema_version: Literal[
        "prospective-regeneration-downstream-v2-freeze"
    ] = "prospective-regeneration-downstream-v2-freeze"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    repository_head_sha: str
    repository_tracked_worktree_dirty: Literal[False] = False

    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_unit_freeze_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_unit_repository_head_sha: str

    policy: ProspectiveRegenerationDownstreamV2Policy

    frozen_before_new_cohort_source_tasks: Literal[True] = True
    frozen_before_new_cohort_generation: Literal[True] = True

    budget_unit_is_not_provider_raw_call_count: Literal[True] = True
    provider_retry_ceiling_recorded_separately: Literal[True] = True

    s135_scientific_outputs_used_to_tune_downstream_content: Literal[
        False
    ] = False
    s135_protocol_failure_used_to_separate_regeneration_and_downstream: Literal[
        True
    ] = True

    post_freeze_budget_edit_allowed: Literal[False] = False
    post_freeze_stage_order_edit_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "ProspectiveRegenerationDownstreamV2Freeze":
        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("regeneration downstream-v2 SHA mismatch")
        if observed_id != (
            "prospective_regeneration_downstream_v2_freeze:"
            + expected_sha[:20]
        ):
            raise ValueError("regeneration downstream-v2 ID mismatch")
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_regeneration_downstream_v2_freeze(
    *,
    regeneration_unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    regeneration_unit_freeze_file_sha256: str,
    repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
) -> ProspectiveRegenerationDownstreamV2Freeze:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "regeneration downstream-v2 freeze requires a clean tracked "
            "worktree"
        )

    body = {
        "schema_version": "prospective-regeneration-downstream-v2-freeze",
        "repository_head_sha": repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "source_regeneration_unit_freeze_id":
            regeneration_unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256":
            regeneration_unit_freeze.freeze_sha256,
        "source_regeneration_unit_freeze_file_sha256":
            regeneration_unit_freeze_file_sha256,
        "source_regeneration_unit_repository_head_sha":
            regeneration_unit_freeze.repository_head_sha,
        "policy": ProspectiveRegenerationDownstreamV2Policy().model_dump(
            mode="json"
        ),
        "frozen_before_new_cohort_source_tasks": True,
        "frozen_before_new_cohort_generation": True,
        "budget_unit_is_not_provider_raw_call_count": True,
        "provider_retry_ceiling_recorded_separately": True,
        "s135_scientific_outputs_used_to_tune_downstream_content": False,
        "s135_protocol_failure_used_to_separate_regeneration_and_downstream":
            True,
        "post_freeze_budget_edit_allowed": False,
        "post_freeze_stage_order_edit_allowed": False,
        "production_selection_authority": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRegenerationDownstreamV2Freeze(
        **body,
        freeze_id=(
            "prospective_regeneration_downstream_v2_freeze:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


__all__ = [
    "ProspectiveRegenerationDownstreamV2Freeze",
    "ProspectiveRegenerationDownstreamV2Policy",
    "RegenerationDownstreamStageBudgetV2",
    "build_regeneration_downstream_v2_freeze",
    "sha256_file",
]
