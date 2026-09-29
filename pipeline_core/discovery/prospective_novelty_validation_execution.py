from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.prospective_novelty_validation_cohort import (
    ProspectiveNoveltyValidationCohortFreeze,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProspectiveNoveltyExecutionCase(StrictModel):
    prospective_case_id: str
    source_case_id: str
    domain_profile_id: str
    case_role: str

    run_dir: str
    command_argv: list[str]

    evaluation_focus: list[str] = Field(default_factory=list)
    research_value_capability_expected: bool

    result_observed_before_plan: Literal[False] = False


class ProspectiveNoveltyExecutionPlan(StrictModel):
    schema_version: Literal[
        "prospective-novelty-validation-execution-plan-v1"
    ] = "prospective-novelty-validation-execution-plan-v1"

    plan_id: str
    plan_sha256: str

    source_freeze_id: str
    source_freeze_sha256: str
    source_freeze_file_sha256: str

    run_root: str

    model_name: str
    critic_model_name: str
    base_url: str | None = None
    api_key_env: str
    results_per_query: int
    domain_data_roots: dict[str, str] = Field(default_factory=dict)

    cases: list[ProspectiveNoveltyExecutionCase]

    scientific_case_selection_changed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    overwrite_existing_case_runs_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_cases(self):
        ids = [row.prospective_case_id for row in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate prospective execution case")
        return self


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def build_prospective_novelty_execution_plan(
    *,
    freeze: ProspectiveNoveltyValidationCohortFreeze,
    source_freeze_file_sha256: str,
    run_root: str,
    model_name: str,
    critic_model_name: str,
    base_url: str | None,
    api_key_env: str,
    results_per_query: int,
    domain_data_roots: dict[str, str] | None = None,
) -> ProspectiveNoveltyExecutionPlan:
    if results_per_query < 1:
        raise ValueError("results_per_query must be >= 1")
    if not model_name.strip():
        raise ValueError("model_name must be non-empty")
    if not critic_model_name.strip():
        raise ValueError("critic_model_name must be non-empty")

    root = str(run_root).rstrip("/")
    data_roots = {
        str(domain): str(Path(path).expanduser().resolve())
        for domain, path in (domain_data_roots or {}).items()
    }

    cases: list[ProspectiveNoveltyExecutionCase] = []

    for row in freeze.frozen_cases:
        run_dir = (
            root
            + "/"
            + row.source_case_id.lower()
        )

        argv = [
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
            "--corpus-id",
            row.corpus_id,
            "--domain-profile",
            row.domain_profile_id,
            "--run-dir",
            run_dir,
        ]

        data_root = data_roots.get(row.domain_profile_id)
        if data_root is not None:
            expected_graph = (
                Path(data_root)
                / "corpus"
                / row.corpus_id
                / "mechanism"
                / "navigation"
                / "graph.graphml"
            )
            if not expected_graph.is_file():
                raise ValueError(
                    "domain data-root does not contain the frozen case graph: "
                    f"domain={row.domain_profile_id!r}, "
                    f"corpus={row.corpus_id!r}, "
                    f"expected={expected_graph}"
                )
            argv.extend([
                "--data-root",
                data_root,
            ])

        argv.extend([
            "--source",
            row.source,
        ])

        if row.stop is not None:
            argv.extend([
                "--stop",
                row.stop,
            ])

        argv.extend([
            "--target",
            row.target,
            "--question",
            row.question,
            "--objective",
            row.objective,
            "--grounding-policy",
            "semantic_stop_fallback_top_n",
            "--model",
            model_name,
            "--critic-model",
            critic_model_name,
        ])

        if base_url:
            argv.extend([
                "--base-url",
                base_url,
            ])

        argv.extend([
            "--api-key-env",
            api_key_env,
            "--node-map-k",
            "20",
            "--waypoint-k",
            "12",
            "--endpoint-pair-k",
            "12",
            "--max-depth",
            "12",
            "--top-k",
            "8",
            "--discovery-top-k",
            "8",
            "--max-axes",
            "5",
            "--results-per-query",
            str(results_per_query),

            # Shadow stack under validation.
            "--question-task-preservation-shadow",
            "--direct-relationpattern-task-shadow",
            "--direct-higher-order-shadow",
            "--direct-higher-order-max-contexts",
            "4",
            "--direct-higher-order-downstream-shadow",
            "--research-value-shadow",
            "--novelty-guided-discovery-closure-shadow",
        ])

        cases.append(
            ProspectiveNoveltyExecutionCase(
                prospective_case_id=row.prospective_case_id,
                source_case_id=row.source_case_id,
                domain_profile_id=row.domain_profile_id,
                case_role=row.case_role,
                run_dir=run_dir,
                command_argv=argv,
                evaluation_focus=list(row.evaluation_focus),
                research_value_capability_expected=(
                    row.research_value_capability_expected
                ),
            )
        )

    plan_id = _stable_id(
        "prospective_novelty_execution_plan",
        freeze.freeze_id,
        source_freeze_file_sha256,
        run_root,
        model_name,
        critic_model_name,
        base_url or "",
        api_key_env,
        results_per_query,
        *[
            f"{domain}={data_roots[domain]}"
            for domain in sorted(data_roots)
        ],
        *[row.prospective_case_id for row in cases],
    )

    base = {
        "schema_version":
            "prospective-novelty-validation-execution-plan-v1",
        "plan_id":
            plan_id,
        "source_freeze_id":
            freeze.freeze_id,
        "source_freeze_sha256":
            freeze.freeze_sha256,
        "source_freeze_file_sha256":
            source_freeze_file_sha256,
        "run_root":
            root,
        "model_name":
            model_name,
        "critic_model_name":
            critic_model_name,
        "base_url":
            base_url,
        "api_key_env":
            api_key_env,
        "results_per_query":
            results_per_query,
        "domain_data_roots":
            data_roots,
        "cases": [
            row.model_dump(mode="json")
            for row in cases
        ],
        "scientific_case_selection_changed":
            False,
        "result_conditioned_route_changes_allowed":
            False,
        "overwrite_existing_case_runs_allowed":
            False,
        "production_selection_authority":
            False,
    }

    return ProspectiveNoveltyExecutionPlan(
        **base,
        plan_sha256=_sha256_json(base),
    )


__all__ = [
    "ProspectiveNoveltyExecutionCase",
    "ProspectiveNoveltyExecutionPlan",
    "build_prospective_novelty_execution_plan",
]
