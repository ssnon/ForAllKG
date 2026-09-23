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


class ProspectiveSourceTaskDefinition(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    relation_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    stop: str | None = None
    target: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: str = Field(default="explain_connection", min_length=1)

    @model_validator(mode="after")
    def validate_task(self) -> "ProspectiveSourceTaskDefinition":
        for label, value in (
            ("relation_family", self.relation_family),
            ("source", self.source),
            ("target", self.target),
            ("question", self.question),
            ("objective", self.objective),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be nonblank")
        if self.stop is not None and not self.stop.strip():
            raise ValueError("stop must be null or nonblank")
        return self


class ProspectiveSourceTaskCampaignSpec(StrictModel):
    schema_version: Literal[
        "prospective-source-task-campaign-spec-v1"
    ] = "prospective-source-task-campaign-spec-v1"

    campaign_name: str = Field(min_length=1)
    campaign_root: str = Field(min_length=1)

    domain_profile_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    data_root: str = Field(min_length=1)
    semantic_roots: list[str] = Field(min_length=1)

    generation_model: str = Field(min_length=1)
    critic_model: str = Field(min_length=1)

    tasks: list[ProspectiveSourceTaskDefinition] = Field(
        min_length=5,
        max_length=5,
    )

    @model_validator(mode="after")
    def validate_campaign(
        self,
    ) -> "ProspectiveSourceTaskCampaignSpec":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        observed = [row.case_id for row in self.tasks]
        if observed != expected:
            raise ValueError(
                "prospective source tasks must be ordered exactly P06-P10"
            )

        if len(set(self.semantic_roots)) != len(self.semantic_roots):
            raise ValueError("semantic_roots must be unique")
        if any(not value.strip() for value in self.semantic_roots):
            raise ValueError("semantic_roots cannot contain blank values")

        relation_families = [row.relation_family for row in self.tasks]
        if len(set(relation_families)) != len(relation_families):
            raise ValueError(
                "P06-P10 relation_family values must be unique"
            )

        task_surfaces = [
            (
                row.source.casefold().strip(),
                (row.stop or "").casefold().strip(),
                row.target.casefold().strip(),
                row.question.casefold().strip(),
            )
            for row in self.tasks
        ]
        if len(set(task_surfaces)) != len(task_surfaces):
            raise ValueError("P06-P10 source tasks must be distinct")
        return self


class FrozenProspectiveSourceTask(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
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
    candidate_content_predeclared: Literal[False] = False
    verifier_outcome_predeclared: Literal[False] = False

    @model_validator(mode="after")
    def validate_hash(self) -> "FrozenProspectiveSourceTask":
        body = self.model_dump(mode="json")
        observed_id = body.pop("task_id")
        observed_sha = body.pop("task_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective source task SHA mismatch")
        if observed_id != (
            "prospective_source_task:" + expected_sha[:20]
        ):
            raise ValueError("prospective source task ID mismatch")
        return self


class ProspectiveSourceTaskCampaignFreeze(StrictModel):
    schema_version: Literal[
        "prospective-source-task-campaign-freeze-v1"
    ] = "prospective-source-task-campaign-freeze-v1"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    campaign_name: str
    campaign_root: str
    repository_head_sha: str
    repository_tracked_worktree_dirty: Literal[False] = False

    tasks: list[FrozenProspectiveSourceTask]
    task_count: Literal[5] = 5
    case_ids: list[str]
    relation_family_counts: dict[str, int]

    source_tasks_frozen_before_generation: Literal[True] = True
    hypotheses_generated_before_freeze: Literal[False] = False
    scientific_reframing_generated_before_freeze: Literal[False] = False
    production_candidates_generated_before_freeze: Literal[False] = False
    atomic_synthesis_generated_before_freeze: Literal[False] = False
    endpoint_binding_observed_before_freeze: Literal[False] = False
    prior_art_observed_before_freeze: Literal[False] = False
    old_n10_observed_before_freeze: Literal[False] = False
    new_verifier_observed_before_freeze: Literal[False] = False
    external_novelty_observed_before_freeze: Literal[False] = False
    positive_nonobviousness_observed_before_freeze: Literal[False] = False

    legacy_case_reuse_allowed: Literal[False] = False
    post_freeze_task_replacement_allowed: Literal[False] = False
    post_freeze_question_edit_allowed: Literal[False] = False
    post_freeze_source_target_edit_allowed: Literal[False] = False
    failed_or_abstained_case_replacement_allowed: Literal[False] = False

    scientific_quality_ranking_performed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "ProspectiveSourceTaskCampaignFreeze":
        if len(self.tasks) != 5:
            raise ValueError("prospective source campaign requires five tasks")

        expected = ["P06", "P07", "P08", "P09", "P10"]
        observed = [row.case_id for row in self.tasks]
        if observed != expected or self.case_ids != expected:
            raise ValueError("frozen case IDs must be exactly P06-P10")

        if len({row.task_id for row in self.tasks}) != 5:
            raise ValueError("frozen prospective task IDs must be unique")

        expected_counts = Counter(
            row.relation_family for row in self.tasks
        )
        if dict(sorted(expected_counts.items())) != dict(
            sorted(self.relation_family_counts.items())
        ):
            raise ValueError("relation_family_counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective source campaign freeze SHA mismatch")
        if observed_id != (
            "prospective_source_task_campaign_freeze:"
            + expected_sha[:20]
        ):
            raise ValueError("prospective source campaign freeze ID mismatch")
        return self


def build_prospective_source_task_campaign_freeze(
    *,
    spec: ProspectiveSourceTaskCampaignSpec,
    source_spec_sha256: str,
    repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
) -> ProspectiveSourceTaskCampaignFreeze:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "prospective source-task freeze requires a clean tracked worktree"
        )

    campaign_root = str(
        Path(spec.campaign_root).expanduser().resolve()
    )
    data_root = str(Path(spec.data_root).expanduser().resolve())
    semantic_roots = [
        str(Path(value).expanduser().resolve())
        for value in spec.semantic_roots
    ]

    frozen_tasks: list[FrozenProspectiveSourceTask] = []
    for row in spec.tasks:
        task_body = {
            "case_id": row.case_id,
            "run_dir": str(Path(campaign_root) / row.case_id),
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
            "candidate_content_predeclared": False,
            "verifier_outcome_predeclared": False,
        }
        task_sha = _sha256_json(task_body)
        frozen_tasks.append(
            FrozenProspectiveSourceTask(
                **task_body,
                task_id="prospective_source_task:" + task_sha[:20],
                task_sha256=task_sha,
            )
        )

    body = {
        "schema_version":
            "prospective-source-task-campaign-freeze-v1",
        "source_spec_sha256": source_spec_sha256,
        "campaign_name": spec.campaign_name,
        "campaign_root": campaign_root,
        "repository_head_sha": repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "tasks": [
            row.model_dump(mode="json")
            for row in frozen_tasks
        ],
        "task_count": 5,
        "case_ids": ["P06", "P07", "P08", "P09", "P10"],
        "relation_family_counts": dict(
            sorted(
                Counter(
                    row.relation_family
                    for row in frozen_tasks
                ).items()
            )
        ),
        "source_tasks_frozen_before_generation": True,
        "hypotheses_generated_before_freeze": False,
        "scientific_reframing_generated_before_freeze": False,
        "production_candidates_generated_before_freeze": False,
        "atomic_synthesis_generated_before_freeze": False,
        "endpoint_binding_observed_before_freeze": False,
        "prior_art_observed_before_freeze": False,
        "old_n10_observed_before_freeze": False,
        "new_verifier_observed_before_freeze": False,
        "external_novelty_observed_before_freeze": False,
        "positive_nonobviousness_observed_before_freeze": False,
        "legacy_case_reuse_allowed": False,
        "post_freeze_task_replacement_allowed": False,
        "post_freeze_question_edit_allowed": False,
        "post_freeze_source_target_edit_allowed": False,
        "failed_or_abstained_case_replacement_allowed": False,
        "scientific_quality_ranking_performed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveSourceTaskCampaignFreeze(
        **body,
        freeze_id=(
            "prospective_source_task_campaign_freeze:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


__all__ = [
    "FrozenProspectiveSourceTask",
    "ProspectiveSourceTaskCampaignFreeze",
    "ProspectiveSourceTaskCampaignSpec",
    "ProspectiveSourceTaskDefinition",
    "build_prospective_source_task_campaign_freeze",
]
