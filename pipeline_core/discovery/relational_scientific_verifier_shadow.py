from __future__ import annotations

import hashlib
import json
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RelationalVerifierArtifactFingerprint(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


def fingerprint_file(path: str | Path) -> RelationalVerifierArtifactFingerprint:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("missing artifact: " + str(resolved))
    return RelationalVerifierArtifactFingerprint(
        path=str(resolved),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


class RelationalScientificVerifierInputFreeze(StrictModel):
    schema_version: Literal[
        "relational-scientific-verifier-input-freeze-v1"
    ] = "relational-scientific-verifier-input-freeze-v1"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    repository_head_sha: str
    repository_worktree_dirty: bool

    final_hypothesis_id: str
    candidate_hypothesis_id: str
    binding_plan_id: str
    endpoint_binding_report_id: str
    source_external_novelty_report_id: str
    final_alpha6_portfolio_id: str
    domain_profile_id: str
    model_name: str

    support_results_per_query: int = Field(ge=1)
    second_pass_results_per_query: int = Field(ge=1)
    max_review_works_per_claim: int = Field(ge=1)
    max_exhaustive_rounds: int = Field(ge=1)
    max_second_pass_queries_per_claim: int = Field(ge=1)
    max_resolution_queries: int = Field(ge=0)
    max_alias_query_variants_per_projection: int = Field(ge=1)
    max_lower_order_factor_order: int = Field(ge=0)
    max_source_alias_variants_per_projection: int = Field(ge=1)

    input_artifacts: list[RelationalVerifierArtifactFingerprint]

    source_population_frozen_before_verifier: Literal[True] = True
    endpoint_binding_frozen_before_verifier: Literal[True] = True
    candidate_final_authority_equivalence_required: Literal[True] = True
    external_novelty_reassessment_allowed: Literal[False] = False
    verifier_result_observed_before_freeze: Literal[False] = False
    production_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "RelationalScientificVerifierInputFreeze":
        paths = [row.path for row in self.input_artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate input artifact path in freeze")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("relational verifier input freeze SHA mismatch")
        if observed_id != (
            "relational_scientific_verifier_input_freeze:"
            + expected_sha[:20]
        ):
            raise ValueError("relational verifier input freeze ID mismatch")
        return self


def build_relational_scientific_verifier_input_freeze(
    *,
    repository_head_sha: str,
    repository_worktree_dirty: bool,
    final_hypothesis_id: str,
    candidate_hypothesis_id: str,
    binding_plan_id: str,
    endpoint_binding_report_id: str,
    source_external_novelty_report_id: str,
    final_alpha6_portfolio_id: str,
    domain_profile_id: str,
    model_name: str,
    support_results_per_query: int,
    second_pass_results_per_query: int,
    max_review_works_per_claim: int,
    max_exhaustive_rounds: int,
    max_second_pass_queries_per_claim: int,
    max_resolution_queries: int,
    max_alias_query_variants_per_projection: int,
    max_lower_order_factor_order: int,
    max_source_alias_variants_per_projection: int,
    input_artifacts: list[RelationalVerifierArtifactFingerprint],
) -> RelationalScientificVerifierInputFreeze:
    body = {
        "schema_version":
            "relational-scientific-verifier-input-freeze-v1",
        "repository_head_sha": repository_head_sha,
        "repository_worktree_dirty": repository_worktree_dirty,
        "final_hypothesis_id": final_hypothesis_id,
        "candidate_hypothesis_id": candidate_hypothesis_id,
        "binding_plan_id": binding_plan_id,
        "endpoint_binding_report_id": endpoint_binding_report_id,
        "source_external_novelty_report_id":
            source_external_novelty_report_id,
        "final_alpha6_portfolio_id": final_alpha6_portfolio_id,
        "domain_profile_id": domain_profile_id,
        "model_name": model_name,
        "support_results_per_query": support_results_per_query,
        "second_pass_results_per_query":
            second_pass_results_per_query,
        "max_review_works_per_claim":
            max_review_works_per_claim,
        "max_exhaustive_rounds": max_exhaustive_rounds,
        "max_second_pass_queries_per_claim":
            max_second_pass_queries_per_claim,
        "max_resolution_queries": max_resolution_queries,
        "max_alias_query_variants_per_projection":
            max_alias_query_variants_per_projection,
        "max_lower_order_factor_order":
            max_lower_order_factor_order,
        "max_source_alias_variants_per_projection":
            max_source_alias_variants_per_projection,
        "input_artifacts": [
            row.model_dump(mode="json")
            for row in input_artifacts
        ],
        "source_population_frozen_before_verifier": True,
        "endpoint_binding_frozen_before_verifier": True,
        "candidate_final_authority_equivalence_required": True,
        "external_novelty_reassessment_allowed": False,
        "verifier_result_observed_before_freeze": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalScientificVerifierInputFreeze(
        **body,
        freeze_id=(
            "relational_scientific_verifier_input_freeze:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


class RelationalVerifierStageRecord(StrictModel):
    stage_index: int = Field(ge=1)
    stage_name: str
    argv: list[str]
    output_artifacts: list[RelationalVerifierArtifactFingerprint]


class RelationalScientificVerifierRunManifest(StrictModel):
    schema_version: Literal[
        "relational-scientific-verifier-run-manifest-v1"
    ] = "relational-scientific-verifier-run-manifest-v1"

    manifest_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    input_freeze_id: str
    input_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_head_sha: str
    repository_worktree_dirty: bool

    final_hypothesis_id: str
    candidate_hypothesis_id: str

    stage_records: list[RelationalVerifierStageRecord]
    stage_count: int = Field(ge=0)

    certification_report_id: str
    certification_decision: str
    bounded_closure_state: str
    bounded_external_distinctness_state: str
    positive_nonobviousness_authority_state: str
    fatal_blocker_state: str

    input_artifacts_unchanged_after_execution: Literal[True] = True
    external_novelty_reassessed: Literal[False] = False
    production_authority_created: Literal[False] = False
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    manifest_write_once: Literal[True] = True

    @model_validator(mode="after")
    def validate_manifest(
        self,
    ) -> "RelationalScientificVerifierRunManifest":
        if self.stage_count != len(self.stage_records):
            raise ValueError("stage_count mismatch")
        expected_indices = list(range(1, len(self.stage_records) + 1))
        if [row.stage_index for row in self.stage_records] != expected_indices:
            raise ValueError("stage indices must be contiguous and ordered")

        body = self.model_dump(mode="json")
        observed_id = body.pop("manifest_id")
        observed_sha = body.pop("manifest_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("relational verifier run manifest SHA mismatch")
        if observed_id != (
            "relational_scientific_verifier_run_manifest:"
            + expected_sha[:20]
        ):
            raise ValueError("relational verifier run manifest ID mismatch")
        return self


def build_relational_scientific_verifier_run_manifest(
    *,
    freeze: RelationalScientificVerifierInputFreeze,
    stage_records: list[RelationalVerifierStageRecord],
    certification_report_id: str,
    certification_decision: str,
    bounded_closure_state: str,
    bounded_external_distinctness_state: str,
    positive_nonobviousness_authority_state: str,
    fatal_blocker_state: str,
) -> RelationalScientificVerifierRunManifest:
    body = {
        "schema_version":
            "relational-scientific-verifier-run-manifest-v1",
        "input_freeze_id": freeze.freeze_id,
        "input_freeze_sha256": freeze.freeze_sha256,
        "repository_head_sha": freeze.repository_head_sha,
        "repository_worktree_dirty":
            freeze.repository_worktree_dirty,
        "final_hypothesis_id": freeze.final_hypothesis_id,
        "candidate_hypothesis_id":
            freeze.candidate_hypothesis_id,
        "stage_records": [
            row.model_dump(mode="json")
            for row in stage_records
        ],
        "stage_count": len(stage_records),
        "certification_report_id": certification_report_id,
        "certification_decision": certification_decision,
        "bounded_closure_state": bounded_closure_state,
        "bounded_external_distinctness_state":
            bounded_external_distinctness_state,
        "positive_nonobviousness_authority_state":
            positive_nonobviousness_authority_state,
        "fatal_blocker_state": fatal_blocker_state,
        "input_artifacts_unchanged_after_execution": True,
        "external_novelty_reassessed": False,
        "production_authority_created": False,
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "manifest_write_once": True,
    }
    digest = _sha256_json(body)
    return RelationalScientificVerifierRunManifest(
        **body,
        manifest_id=(
            "relational_scientific_verifier_run_manifest:"
            + digest[:20]
        ),
        manifest_sha256=digest,
    )


def write_json_exclusive(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = (
        payload.model_dump(mode="json")
        if hasattr(payload, "model_dump")
        else payload
    )
    with target.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def assert_fingerprints_unchanged(
    fingerprints: list[RelationalVerifierArtifactFingerprint],
) -> None:
    for frozen in fingerprints:
        current = fingerprint_file(frozen.path)
        if current != frozen:
            raise ValueError(
                "frozen verifier input changed during execution: "
                + frozen.path
            )


__all__ = [
    "RelationalScientificVerifierInputFreeze",
    "RelationalScientificVerifierRunManifest",
    "RelationalVerifierArtifactFingerprint",
    "RelationalVerifierStageRecord",
    "assert_fingerprints_unchanged",
    "build_relational_scientific_verifier_input_freeze",
    "build_relational_scientific_verifier_run_manifest",
    "fingerprint_file",
    "sha256_file",
    "write_json_exclusive",
]
