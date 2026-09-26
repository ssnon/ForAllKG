from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    AtomicScientificSpecificationBundle,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScientificVerifierShadowInputs(StrictModel):
    run_dir: str
    context: str
    candidate_portfolio: str
    atomic_report: str
    atomic_portfolio: str
    canonical_spec_bundle: str | None = None
    atomic_n10_manifest: str
    atomic_n10_external_report: str
    atomic_n10_provider_plan: str


class ScientificVerifierShadowLineage(StrictModel):
    schema_version: Literal[
        "scientific-verifier-shadow-lineage-v1"
    ] = "scientific-verifier-shadow-lineage-v1"

    lineage_id: str
    context_id: str
    domain_profile_id: str
    candidate_portfolio_id: str
    atomic_report_id: str
    atomic_portfolio_id: str
    canonical_spec_bundle_id: str | None = None
    canonical_spec_bundle_sha256: str | None = None
    canonical_spec_bundle_lineage_verified: bool = False
    canonical_spec_bundle_is_relation_ir_authority: bool = False
    atomic_hypothesis_ids: list[str]
    atomic_claim_ids: list[str]
    external_novelty_report_id: str
    external_prior_art_packet_id: str
    atomic_n10_authority_mode: str
    atomic_n10_manifest_status: str
    atomic_n10_completion_verified: bool
    legacy_atomic_n10_artifact_mode: bool
    legacy_atomic_artifact_autoresolution_used: bool
    manifest_source_portfolio_binding_verified: bool
    manifest_source_hypothesis_count_verified: bool
    source_hypothesis_count: int = Field(ge=1)

    completed_atomic_n10_required: bool
    legacy_mode_is_historical_smoke_only: Literal[True] = True
    prospective_validation_input_contract_satisfied: bool
    exact_candidate_to_atomic_binding: Literal[True] = True
    exact_atomic_hypothesis_set_binding: Literal[True] = True
    exact_external_report_binding: Literal[True] = True
    grounded_identity_annotation_independent_of_n10_closure: Literal[True] = True
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "ScientificVerifierShadowLineage":
        if len(self.atomic_hypothesis_ids) != len(set(self.atomic_hypothesis_ids)):
            raise ValueError("atomic hypothesis IDs must be unique")
        if len(self.atomic_claim_ids) != len(set(self.atomic_claim_ids)):
            raise ValueError("atomic claim IDs must be unique")
        if self.source_hypothesis_count != len(self.atomic_hypothesis_ids):
            raise ValueError("source_hypothesis_count mismatch")
        if (
            (self.canonical_spec_bundle_id is None)
            != (self.canonical_spec_bundle_sha256 is None)
        ):
            raise ValueError(
                "canonical specification bundle ID/SHA mismatch"
            )
        if (
            self.canonical_spec_bundle_is_relation_ir_authority
            and not self.canonical_spec_bundle_lineage_verified
        ):
            raise ValueError(
                "canonical relation-IR authority requires verified bundle lineage"
            )
        return self


class ScientificVerifierShadowOutputs(StrictModel):
    output_dir: str
    grounded_identity_annotation: str
    relation_ir: str
    relation_projection: str
    grounded_identity_factorization: str
    grounded_factor_projection: str
    supporting_prefix: str
    counterevidence_prefix: str
    second_pass_prefix: str
    relation_adjudication_prefix: str
    evidence_graph: str
    centrality: str
    aggregation: str
    positive_basis: str
    positive_adjudication_prefix: str
    certification_report: str
    manifest: str


class ScientificVerifierShadowCompanionPlan(StrictModel):
    schema_version: Literal[
        "scientific-verifier-shadow-companion-plan-v1"
    ] = "scientific-verifier-shadow-companion-plan-v1"

    inputs: ScientificVerifierShadowInputs
    lineage: ScientificVerifierShadowLineage
    outputs: ScientificVerifierShadowOutputs
    stage_names: list[str]
    stage_count: int = Field(ge=1)

    shadow_only: Literal[True] = True
    existing_atomic_n10_reused_read_only: Literal[True] = True
    external_novelty_report_reused_for_distinctness_only: Literal[True] = True
    relation_retrieval_is_independent_shadow_evidence: Literal[True] = True
    semantic_second_pass_precedes_final_adjudication: Literal[True] = True
    grounded_identity_annotation_independent_of_n10_closure: Literal[True] = True
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_stage_count(self) -> "ScientificVerifierShadowCompanionPlan":
        if self.stage_count != len(self.stage_names):
            raise ValueError("stage_count mismatch")
        if len(self.stage_names) != len(set(self.stage_names)):
            raise ValueError("duplicate stage name")
        return self


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"missing {label}: {resolved}")
    return resolved


def _require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"missing {label}: {resolved}")
    return resolved


def _require_schema(payload: dict[str, object], expected: str, label: str) -> None:
    observed = payload.get("schema_version")
    if observed != expected:
        raise ValueError(
            f"{label} schema mismatch: expected={expected!r}, observed={observed!r}"
        )


def _required_string(payload: dict[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} missing non-empty {key}")
    return value


def _hypothesis_ids(payload: dict[str, object], label: str) -> list[str]:
    rows = payload.get("hypotheses")
    if not isinstance(rows, list):
        raise ValueError(f"{label} hypotheses must be a list")
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} hypothesis must be an object")
        value = row.get("hypothesis_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} hypothesis missing hypothesis_id")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate hypothesis IDs")
    return ids


def _external_hypothesis_ids(payload: dict[str, object]) -> list[str]:
    rows = payload.get("cards")
    if not isinstance(rows, list):
        raise ValueError("external novelty cards must be a list")
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("external novelty card must be an object")
        value = row.get("hypothesis_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("external novelty card missing hypothesis_id")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError("external novelty report contains duplicate hypothesis IDs")
    return ids


def _atomic_claim_ids(payload: dict[str, object]) -> list[str]:
    rows = payload.get("hypotheses")
    if not isinstance(rows, list):
        raise ValueError("atomic report hypotheses must be a list")
    output: list[str] = []
    for hypothesis in rows:
        if not isinstance(hypothesis, dict):
            raise ValueError("atomic report hypothesis must be an object")
        specs = hypothesis.get("atomic_specifications")
        if not isinstance(specs, list):
            raise ValueError("atomic report atomic_specifications must be a list")
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError("atomic specification must be an object")
            claim_id = spec.get("claim_id")
            if not isinstance(claim_id, str) or not claim_id.strip():
                raise ValueError("atomic specification missing claim_id")
            output.append(claim_id)
    if len(output) != len(set(output)):
        raise ValueError("atomic report contains duplicate claim IDs")
    return output


def _stable_lineage_id(parts: list[str]) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"scientific_verifier_shadow_lineage:{digest}"


def _legacy_exact_atomic_portfolio(
    *,
    run: Path,
    external_path: Path,
) -> Path:
    external = _load_object(_require_file(external_path, "atomic N10 external novelty report"))
    _require_schema(external, "external-novelty-report-v1", "external novelty report")
    target_portfolio_id = _required_string(
        external, "source_portfolio_id", "external novelty report"
    )
    target_hypothesis_ids = set(_external_hypothesis_ids(external))

    matches: list[Path] = []
    for path in sorted(run.glob("scientific_atomic_cross_lane*.portfolio.json")):
        try:
            payload = _load_object(path)
            if payload.get("schema_version") != "hypothesis-portfolio-v1":
                continue
            if payload.get("portfolio_id") != target_portfolio_id:
                continue
            if set(_hypothesis_ids(payload, f"legacy atomic portfolio {path.name}")) != target_hypothesis_ids:
                continue
        except ValueError:
            continue
        matches.append(path.resolve())

    if len(matches) != 1:
        raise ValueError(
            "legacy atomic portfolio auto-resolution requires exactly one "
            "scientific_atomic_cross_lane*.portfolio.json matching the external "
            f"source_portfolio_id and hypothesis set; matches={[str(p) for p in matches]}"
        )
    return matches[0]


def _legacy_exact_atomic_report(
    *,
    run: Path,
    candidate_path: Path,
    atomic_portfolio_path: Path,
) -> Path:
    candidate = _load_object(_require_file(candidate_path, "candidate portfolio"))
    _require_schema(
        candidate,
        "production-facing-scientific-candidate-portfolio-v1",
        "candidate portfolio",
    )
    candidate_id = _required_string(candidate, "portfolio_id", "candidate portfolio")
    portfolio = _load_object(_require_file(atomic_portfolio_path, "atomic portfolio"))
    _require_schema(portfolio, "hypothesis-portfolio-v1", "atomic portfolio")
    target_hypothesis_ids = set(_hypothesis_ids(portfolio, "atomic portfolio"))

    matches: list[Path] = []
    for path in sorted(run.glob("scientific_atomic_cross_lane*.report.json")):
        try:
            payload = _load_object(path)
            if payload.get("schema_version") != "atomic-cross-lane-scientific-synthesis-report-v1":
                continue
            if payload.get("source_candidate_portfolio_id") != candidate_id:
                continue
            if set(_hypothesis_ids(payload, f"legacy atomic report {path.name}")) != target_hypothesis_ids:
                continue
        except ValueError:
            continue
        matches.append(path.resolve())

    # Atomic report and projected portfolio are emitted as a filename-paired
    # artifact set, but the current schemas do not directly cross-reference
    # each other's IDs.  Prefer the exact sibling stem selected by the external
    # report, then fall back to uniqueness across semantically matching reports.
    portfolio_name = atomic_portfolio_path.name
    if not portfolio_name.endswith(".portfolio.json"):
        raise ValueError("legacy atomic portfolio filename lacks .portfolio.json suffix")
    paired_name = portfolio_name[: -len(".portfolio.json")] + ".report.json"
    paired = (run / paired_name).resolve()
    if paired in matches:
        return paired

    if len(matches) != 1:
        raise ValueError(
            "legacy atomic report auto-resolution requires an exact filename-paired "
            "report or exactly one semantic match; "
            f"portfolio={atomic_portfolio_path.name!r}, matches={[str(p) for p in matches]}"
        )
    return matches[0]


def resolve_scientific_verifier_shadow_inputs(
    *,
    run_dir: str | Path,
    context: str | Path | None = None,
    candidate_portfolio: str | Path | None = None,
    atomic_report: str | Path | None = None,
    atomic_portfolio: str | Path | None = None,
    canonical_spec_bundle: str | Path | None = None,
    atomic_n10_manifest: str | Path | None = None,
    atomic_n10_external_report: str | Path | None = None,
    atomic_n10_provider_plan: str | Path | None = None,
    allow_legacy_atomic_n10_artifacts: bool = False,
) -> ScientificVerifierShadowInputs:
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"missing run directory: {run}")

    def chosen(value: str | Path | None, default: str) -> Path:
        return Path(value).expanduser().resolve() if value is not None else run / default

    context_path = chosen(context, "hypothesis.context.json")
    candidate_path = chosen(
        candidate_portfolio, "scientific_pre_n10_candidate_portfolio.json"
    )
    manifest_path = chosen(
        atomic_n10_manifest, "scientific_atomic_n10_e2e_manifest.json"
    )
    external_path = chosen(
        atomic_n10_external_report, "scientific_atomic_n10_external.report.json"
    )
    provider_path = chosen(
        atomic_n10_provider_plan, "scientific_atomic_n10_external.provider_plan.json"
    )

    if allow_legacy_atomic_n10_artifacts and atomic_portfolio is None:
        atomic_portfolio_path = _legacy_exact_atomic_portfolio(
            run=run,
            external_path=external_path,
        )
    else:
        atomic_portfolio_path = chosen(
            atomic_portfolio, "scientific_atomic_cross_lane.portfolio.json"
        )

    if allow_legacy_atomic_n10_artifacts and atomic_report is None:
        atomic_report_path = _legacy_exact_atomic_report(
            run=run,
            candidate_path=candidate_path,
            atomic_portfolio_path=atomic_portfolio_path,
        )
    else:
        atomic_report_path = chosen(
            atomic_report, "scientific_atomic_cross_lane.report.json"
        )

    return ScientificVerifierShadowInputs(
        run_dir=str(run),
        context=str(context_path),
        candidate_portfolio=str(candidate_path),
        atomic_report=str(atomic_report_path),
        atomic_portfolio=str(atomic_portfolio_path),
        canonical_spec_bundle=(
            str(Path(canonical_spec_bundle).expanduser().resolve())
            if canonical_spec_bundle is not None
            else None
        ),
        atomic_n10_manifest=str(manifest_path),
        atomic_n10_external_report=str(external_path),
        atomic_n10_provider_plan=str(provider_path),
    )


def validate_scientific_verifier_shadow_lineage(
    inputs: ScientificVerifierShadowInputs,
    *,
    allow_legacy_atomic_n10_artifacts: bool = False,
) -> ScientificVerifierShadowLineage:
    context_path = _require_file(Path(inputs.context), "hypothesis context")
    candidate_path = _require_file(Path(inputs.candidate_portfolio), "candidate portfolio")
    atomic_report_path = _require_file(Path(inputs.atomic_report), "atomic synthesis report")
    atomic_portfolio_path = _require_file(Path(inputs.atomic_portfolio), "atomic portfolio")
    canonical_bundle_path = (
        _require_file(
            Path(inputs.canonical_spec_bundle),
            "canonical atomic specification bundle",
        )
        if inputs.canonical_spec_bundle is not None
        else None
    )
    n10_manifest_path = _require_file(Path(inputs.atomic_n10_manifest), "atomic N10 manifest")
    external_path = _require_file(
        Path(inputs.atomic_n10_external_report), "atomic N10 external novelty report"
    )
    _require_file(Path(inputs.atomic_n10_provider_plan), "atomic N10 provider plan")

    context = _load_object(context_path)
    candidate = _load_object(candidate_path)
    atomic_report = _load_object(atomic_report_path)
    atomic_portfolio = _load_object(atomic_portfolio_path)
    canonical_bundle = (
        AtomicScientificSpecificationBundle.model_validate_json(
            canonical_bundle_path.read_text(encoding="utf-8")
        )
        if canonical_bundle_path is not None
        else None
    )
    n10_manifest = _load_object(n10_manifest_path)
    external = _load_object(external_path)

    _require_schema(context, "hypothesis-context-v1", "context")
    _require_schema(
        candidate,
        "production-facing-scientific-candidate-portfolio-v1",
        "candidate portfolio",
    )
    _require_schema(
        atomic_report,
        "atomic-cross-lane-scientific-synthesis-report-v1",
        "atomic synthesis report",
    )
    _require_schema(atomic_portfolio, "hypothesis-portfolio-v1", "atomic portfolio")
    _require_schema(n10_manifest, "scientific-atomic-n10-e2e-manifest-v1", "atomic N10 manifest")
    _require_schema(external, "external-novelty-report-v1", "external novelty report")

    manifest_status = str(n10_manifest.get("status") or "").strip()
    completion_verified = manifest_status == "complete"
    if not completion_verified and not allow_legacy_atomic_n10_artifacts:
        raise ValueError(
            "scientific verifier shadow companion requires completed atomic N10 E2E; "
            "historical calibration/debug runs may opt in with "
            "allow_legacy_atomic_n10_artifacts=True"
        )
    if not allow_legacy_atomic_n10_artifacts:
        if n10_manifest.get("precomputed_atomic_query_plan_reused") is not True:
            raise ValueError("atomic N10 manifest did not reuse the precomputed atomic query plan")
        if n10_manifest.get("external_novelty_llm_redecomposition_performed") is not False:
            raise ValueError("atomic N10 manifest unexpectedly performed external re-decomposition")

    context_id = _required_string(context, "context_id", "context")
    context_sha256 = _required_string(context, "context_sha256", "context")
    task_id = _required_string(context, "task_id", "context")
    domain_profile_id = _required_string(context, "domain_profile_id", "context")
    candidate_id = _required_string(candidate, "portfolio_id", "candidate portfolio")
    atomic_report_id = _required_string(atomic_report, "report_id", "atomic report")
    atomic_portfolio_id = _required_string(atomic_portfolio, "portfolio_id", "atomic portfolio")
    external_report_id = _required_string(external, "report_id", "external novelty report")
    external_packet_id = _required_string(
        external, "source_prior_art_packet_id", "external novelty report"
    )

    if candidate.get("source_context_id") != context_id:
        raise ValueError("candidate portfolio source_context_id mismatch")
    if candidate.get("source_context_sha256") != context_sha256:
        raise ValueError("candidate portfolio source_context_sha256 mismatch")
    if candidate.get("source_task_id") != task_id:
        raise ValueError("candidate portfolio source_task_id mismatch")
    if candidate.get("domain_profile_id") != domain_profile_id:
        raise ValueError("candidate portfolio domain_profile_id mismatch")
    if atomic_report.get("source_candidate_portfolio_id") != candidate_id:
        raise ValueError("atomic report source_candidate_portfolio_id mismatch")
    if atomic_report.get("source_context_id") != context_id:
        raise ValueError("atomic report source_context_id mismatch")
    if atomic_report.get("source_task_id") != task_id:
        raise ValueError("atomic report source_task_id mismatch")
    if atomic_portfolio.get("source_context_id") != context_id:
        raise ValueError("atomic portfolio source_context_id mismatch")
    if atomic_portfolio.get("source_context_sha256") != context_sha256:
        raise ValueError("atomic portfolio source_context_sha256 mismatch")
    if atomic_portfolio.get("domain_profile_id") != domain_profile_id:
        raise ValueError("atomic portfolio domain_profile_id mismatch")

    report_hypothesis_ids = _hypothesis_ids(atomic_report, "atomic report")
    portfolio_hypothesis_ids = _hypothesis_ids(atomic_portfolio, "atomic portfolio")
    if not portfolio_hypothesis_ids:
        raise ValueError("scientific verifier shadow companion requires atomic hypotheses")
    if set(report_hypothesis_ids) != set(portfolio_hypothesis_ids):
        raise ValueError("atomic report/portfolio hypothesis ID set mismatch")

    external_hypothesis_ids = _external_hypothesis_ids(external)
    if external.get("source_portfolio_id") != atomic_portfolio_id:
        raise ValueError("external novelty source_portfolio_id mismatch")
    if set(external_hypothesis_ids) != set(portfolio_hypothesis_ids):
        raise ValueError("external novelty/atomic portfolio hypothesis ID set mismatch")

    manifest_source_portfolio = n10_manifest.get("source_portfolio")
    manifest_source_portfolio_verified = bool(
        isinstance(manifest_source_portfolio, str)
        and manifest_source_portfolio.strip()
        and Path(manifest_source_portfolio).expanduser().resolve() == atomic_portfolio_path
    )
    manifest_source_hypothesis_count_verified = (
        n10_manifest.get("source_hypothesis_count") == len(portfolio_hypothesis_ids)
    )
    if not allow_legacy_atomic_n10_artifacts:
        if not isinstance(manifest_source_portfolio, str) or not manifest_source_portfolio.strip():
            raise ValueError("atomic N10 manifest missing source_portfolio")
        if not manifest_source_portfolio_verified:
            raise ValueError("atomic N10 manifest source_portfolio path mismatch")
        if not manifest_source_hypothesis_count_verified:
            raise ValueError("atomic N10 manifest source_hypothesis_count mismatch")

    claim_ids = _atomic_claim_ids(atomic_report)
    if not claim_ids:
        raise ValueError("atomic report contains no atomic claims")

    canonical_bundle_id = None
    canonical_bundle_sha = None
    canonical_bundle_verified = False
    if canonical_bundle is not None:
        if canonical_bundle.source_report_id != atomic_report_id:
            raise ValueError(
                "canonical bundle/atomic report ID mismatch"
            )
        if canonical_bundle.source_contract != atomic_report.get(
            "schema_version"
        ):
            raise ValueError(
                "canonical bundle/atomic report contract mismatch"
            )
        bundle_hypothesis_ids = sorted(
            row.hypothesis_id for row in canonical_bundle.hypotheses
        )
        bundle_claim_ids = sorted(
            specification.claim_id
            for row in canonical_bundle.hypotheses
            for specification in row.specifications
        )
        if bundle_hypothesis_ids != sorted(portfolio_hypothesis_ids):
            raise ValueError(
                "canonical bundle/atomic hypothesis ID set mismatch"
            )
        if bundle_claim_ids != sorted(claim_ids):
            raise ValueError(
                "canonical bundle/atomic claim ID set mismatch"
            )
        canonical_bundle_id = canonical_bundle.bundle_id
        canonical_bundle_sha = canonical_bundle.bundle_sha256
        canonical_bundle_verified = True

    authority_mode = str(n10_manifest.get("authority_mode") or "").strip() or "unknown"
    lineage_id = _stable_lineage_id(
        [
            context_id,
            candidate_id,
            atomic_report_id,
            atomic_portfolio_id,
            external_report_id,
            *sorted(portfolio_hypothesis_ids),
            *sorted(claim_ids),
        ]
    )
    return ScientificVerifierShadowLineage(
        lineage_id=lineage_id,
        context_id=context_id,
        domain_profile_id=domain_profile_id,
        candidate_portfolio_id=candidate_id,
        atomic_report_id=atomic_report_id,
        atomic_portfolio_id=atomic_portfolio_id,
        canonical_spec_bundle_id=canonical_bundle_id,
        canonical_spec_bundle_sha256=canonical_bundle_sha,
        canonical_spec_bundle_lineage_verified=(
            canonical_bundle_verified
        ),
        canonical_spec_bundle_is_relation_ir_authority=(
            canonical_bundle is not None
        ),
        atomic_hypothesis_ids=sorted(portfolio_hypothesis_ids),
        atomic_claim_ids=sorted(claim_ids),
        external_novelty_report_id=external_report_id,
        external_prior_art_packet_id=external_packet_id,
        atomic_n10_authority_mode=authority_mode,
        atomic_n10_manifest_status=manifest_status or "missing_status",
        atomic_n10_completion_verified=completion_verified,
        legacy_atomic_n10_artifact_mode=allow_legacy_atomic_n10_artifacts,
        legacy_atomic_artifact_autoresolution_used=allow_legacy_atomic_n10_artifacts,
        manifest_source_portfolio_binding_verified=(
            manifest_source_portfolio_verified
        ),
        manifest_source_hypothesis_count_verified=(
            manifest_source_hypothesis_count_verified
        ),
        source_hypothesis_count=len(portfolio_hypothesis_ids),
        completed_atomic_n10_required=not allow_legacy_atomic_n10_artifacts,
        prospective_validation_input_contract_satisfied=bool(
            completion_verified
            and manifest_source_portfolio_verified
            and manifest_source_hypothesis_count_verified
            and not allow_legacy_atomic_n10_artifacts
        ),
    )


def resolve_scientific_verifier_shadow_outputs(
    *,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> ScientificVerifierShadowOutputs:
    run = Path(run_dir).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run / "scientific_verifier_shadow"
    )
    return ScientificVerifierShadowOutputs(
        output_dir=str(output),
        grounded_identity_annotation=str(output / "grounded_identity.annotation.json"),
        relation_ir=str(output / "relation_ir.json"),
        relation_projection=str(output / "relation_projection.json"),
        grounded_identity_factorization=str(output / "grounded_identity_factorization.json"),
        grounded_factor_projection=str(output / "grounded_factor_projection.json"),
        supporting_prefix=str(output / "supporting_projection"),
        counterevidence_prefix=str(output / "counterevidence_projection"),
        second_pass_prefix=str(output / "relation_second_pass"),
        relation_adjudication_prefix=str(output / "relation_adjudication"),
        evidence_graph=str(output / "claim_evidence_graph.json"),
        centrality=str(output / "claim_centrality.json"),
        aggregation=str(output / "hypothesis_evidence_aggregation.json"),
        positive_basis=str(output / "positive_nonobviousness_basis.json"),
        positive_adjudication_prefix=str(output / "positive_nonobviousness_adjudication"),
        certification_report=str(output / "scientific_certification.report.json"),
        manifest=str(output / "scientific_verifier_shadow_e2e_manifest.json"),
    )


def build_scientific_verifier_shadow_companion_plan(
    *,
    inputs: ScientificVerifierShadowInputs,
    output_dir: str | Path | None = None,
    allow_legacy_atomic_n10_artifacts: bool = False,
) -> ScientificVerifierShadowCompanionPlan:
    lineage = validate_scientific_verifier_shadow_lineage(
        inputs,
        allow_legacy_atomic_n10_artifacts=allow_legacy_atomic_n10_artifacts,
    )
    outputs = resolve_scientific_verifier_shadow_outputs(
        run_dir=inputs.run_dir,
        output_dir=output_dir,
    )
    stage_names = [
        "grounded_identity_annotation",
        "scientific_relation_ir",
        "scientific_relation_projection",
        "grounded_identity_factorization",
        "grounded_factor_projection",
        "supporting_projection_retrieval",
        "counterevidence_projection_retrieval",
        "semantic_second_pass_resolution",
        "exhaustive_relation_adjudication",
        "claim_evidence_graph",
        "structural_claim_centrality",
        "hypothesis_evidence_aggregation",
        "positive_nonobviousness_basis",
        "positive_nonobviousness_adjudication",
        "scientific_certification_gate",
    ]
    return ScientificVerifierShadowCompanionPlan(
        inputs=inputs,
        lineage=lineage,
        outputs=outputs,
        stage_names=stage_names,
        stage_count=len(stage_names),
    )


__all__ = [
    "ScientificVerifierShadowCompanionPlan",
    "ScientificVerifierShadowInputs",
    "ScientificVerifierShadowLineage",
    "ScientificVerifierShadowOutputs",
    "build_scientific_verifier_shadow_companion_plan",
    "resolve_scientific_verifier_shadow_inputs",
    "resolve_scientific_verifier_shadow_outputs",
    "validate_scientific_verifier_shadow_lineage",
]
