from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_semantic_disposition import (
    HypothesisSemanticDispositionV1,
)
from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    DownstreamOriginV1,
    PreN10DownstreamHandoffLineageV1,
    PreN10DownstreamHandoffReportV1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
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


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    slug = str(value).replace(":", "_").replace("/", "_")
    if not slug or slug in {".", ".."}:
        raise ValueError("invalid downstream lineage ID")
    return slug


N10CertificationStatusV1 = Literal[
    "NOVELTY_CERTIFIED",
    "NOVELTY_UNRESOLVED",
    "NOVELTY_REJECTED",
]


class PreN10ExternalN10StagePlanV1(StrictModel):
    stage: Literal[
        "EXTERNAL_NOVELTY",
        "N9_INTAKE",
        "N9_FULL_CLOSURE",
        "N10_CANDIDATE_GATE",
        "N10_PRODUCTION_GATE",
    ]
    argv: list[str]
    expected_outputs: list[str]
    dynamic_max_ready_claims_from_intake: bool = False


class PreN10ExternalN10LineagePlanV1(StrictModel):
    lineage_id: str
    origin: DownstreamOriginV1
    source_hypothesis_id: str
    downstream_hypothesis_id: str

    portfolio_path: str
    portfolio_id: str
    portfolio_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_plan_path: str
    query_plan_id: str
    query_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_report_path: str
    contract_report_id: str
    contract_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_disposition_path: str
    semantic_disposition_id: str
    semantic_disposition_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    output_dir: str
    external_output_query_plan: str
    external_prior_art: str
    external_report: str
    n9_intake: str
    n9_full_closure: str
    n10_candidate_gate: str
    n10_production_gate: str

    expected_n10_authority_scope: Literal[
        "alpha6_original_fallback",
        "alpha6_post_generation_candidate",
    ]
    n10_production_module: Literal[
        "scripts.discovery.build_nonobviousness_production_gate_v2",
        "scripts.discovery.build_nonobviousness_post_generation_production_gate_v2",
    ]

    stages: list[PreN10ExternalN10StagePlanV1]

    query_plan_reuse_required: Literal[True] = True
    representation_regeneration_allowed: Literal[False] = False
    targeted_novelty_continuation_allowed: Literal[False] = False
    novelty_refinement_allowed: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_lineage_plan(self) -> "PreN10ExternalN10LineagePlanV1":
        names = [row.stage for row in self.stages]
        expected = [
            "EXTERNAL_NOVELTY",
            "N9_INTAKE",
            "N9_FULL_CLOSURE",
            "N10_CANDIDATE_GATE",
            "N10_PRODUCTION_GATE",
        ]
        if names != expected:
            raise ValueError("external/N9/N10 stage order mismatch")
        dynamic = [
            row.stage for row in self.stages
            if row.dynamic_max_ready_claims_from_intake
        ]
        if dynamic != ["N9_FULL_CLOSURE"]:
            raise ValueError("only N9 full closure may use dynamic ready count")
        external = self.stages[0]
        if "--reuse-query-plan" not in external.argv:
            raise ValueError("external novelty must reuse frozen V_pre query plan")
        if self.query_plan_path not in external.argv:
            raise ValueError("external novelty reuse path mismatch")
        production = self.stages[-1]
        if len(production.argv) < 2 or production.argv[1] != self.n10_production_module:
            raise ValueError("N10 production module/origin mismatch")
        return self


class PreN10ExternalN10ShadowPlanV1(StrictModel):
    schema_version: Literal[
        "pre-n10-external-n9-n10-shadow-plan-v1"
    ] = "pre-n10-external-n9-n10-shadow-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_handoff_report_id: str
    source_handoff_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_context_path: str
    source_context_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_context_id: str
    source_context_sha256: str
    provider_plan_path: str
    provider_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    model: str
    base_url: str | None = None
    api_key_env: str
    results_per_query: Literal[12] = 12
    max_ranked_works: Literal[8] = 8
    save_prompts: bool

    lineages: list[PreN10ExternalN10LineagePlanV1]
    lineage_count: int = Field(ge=0)

    exact_handoff_population_consumed: Literal[True] = True
    frozen_query_plan_reused: Literal[True] = True
    external_novelty_stage_max_per_lineage: Literal[1] = 1
    n9_intake_stage_max_per_lineage: Literal[1] = 1
    n9_full_closure_stage_max_per_lineage: Literal[1] = 1
    n10_candidate_stage_max_per_lineage: Literal[1] = 1
    n10_production_stage_max_per_lineage: Literal[1] = 1
    targeted_novelty_continuation_max_per_lineage: Literal[0] = 0
    novelty_refinement_max_per_lineage: Literal[0] = 0
    second_regeneration_max_per_lineage: Literal[0] = 0
    production_selection_authority_created_by_runner: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "PreN10ExternalN10ShadowPlanV1":
        if self.lineage_count != len(self.lineages):
            raise ValueError("shadow plan lineage_count mismatch")
        ids = [row.lineage_id for row in self.lineages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate downstream lineage ID")
        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("external/N9/N10 shadow plan SHA mismatch")
        if observed_id != "pre_n10_external_n9_n10_shadow_plan_v1:" + expected_sha[:20]:
            raise ValueError("external/N9/N10 shadow plan ID mismatch")
        return self


class PreN10ExternalN10LineageResultV1(StrictModel):
    lineage_id: str
    origin: DownstreamOriginV1
    source_hypothesis_id: str
    downstream_hypothesis_id: str
    portfolio_id: str
    query_plan_id: str

    external_report_path: str
    external_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n9_intake_path: str
    n9_intake_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n9_full_closure_path: str
    n9_full_closure_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n10_candidate_gate_path: str
    n10_candidate_gate_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n10_production_gate_path: str
    n10_production_gate_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    n10_authority_scope: Literal[
        "alpha6_original_fallback",
        "alpha6_post_generation_candidate",
    ]
    n10_selection_class: Literal["ELIGIBLE", "CONDITIONAL", "INELIGIBLE"]
    positive_nonobviousness_authority: bool
    fallback_allowed: bool
    certification_status: N10CertificationStatusV1

    external_novelty_stage_invocations: Literal[1] = 1
    n9_intake_stage_invocations: Literal[1] = 1
    n9_full_closure_stage_invocations: Literal[1] = 1
    n10_candidate_stage_invocations: Literal[1] = 1
    n10_production_stage_invocations: Literal[1] = 1

    frozen_query_plan_reused_exactly: Literal[True] = True
    source_representation_unchanged_after_execution: Literal[True] = True
    candidate_retained: Literal[True] = True
    candidate_survival_authority: Literal[False] = False
    novelty_certification_authority: Literal[True] = True
    conditional_is_positive: Literal[False] = False
    absence_is_novelty: Literal[False] = False
    ineligible_deletes_scientific_candidate: Literal[False] = False
    targeted_novelty_continuation_performed: Literal[False] = False
    novelty_refinement_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "PreN10ExternalN10LineageResultV1":
        expected = {
            "ELIGIBLE": "NOVELTY_CERTIFIED",
            "CONDITIONAL": "NOVELTY_UNRESOLVED",
            "INELIGIBLE": "NOVELTY_REJECTED",
        }[self.n10_selection_class]
        if self.certification_status != expected:
            raise ValueError("N10 selection/certification mapping mismatch")
        if self.n10_selection_class == "ELIGIBLE":
            if not self.positive_nonobviousness_authority or not self.fallback_allowed:
                raise ValueError("ELIGIBLE requires positive authority and frozen permission")
        else:
            if self.positive_nonobviousness_authority or self.fallback_allowed:
                raise ValueError("non-ELIGIBLE cannot carry positive authority or permission")
        expected_scope = (
            "alpha6_original_fallback"
            if self.origin == "INITIAL_PRIMARY_READY"
            else "alpha6_post_generation_candidate"
        )
        if self.n10_authority_scope != expected_scope:
            raise ValueError("N10 authority scope/origin mismatch")
        return self


class PreN10ExternalN10ShadowReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-external-n9-n10-shadow-report-v1"
    ] = "pre-n10-external-n9-n10-shadow-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_handoff_report_id: str
    source_handoff_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    lineages: list[PreN10ExternalN10LineageResultV1]
    lineage_count: int = Field(ge=0)
    certified_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)

    exact_handoff_population_consumed: Literal[True] = True
    pre_n10_ready_was_required: Literal[True] = True
    semantic_disposition_pass_was_required: Literal[True] = True
    frozen_query_plan_reused: Literal[True] = True
    candidate_survival_authority: Literal[False] = False
    novelty_certification_authority: Literal[True] = True
    authority_source: Literal[
        "n10_role_aware_nonobviousness_v2"
    ] = "n10_role_aware_nonobviousness_v2"
    authority_mode: Literal[
        "certification_only_shadow"
    ] = "certification_only_shadow"
    conditional_is_positive: Literal[False] = False
    absence_is_novelty: Literal[False] = False
    ineligible_deletes_scientific_candidate: Literal[False] = False

    targeted_novelty_continuation_performed: Literal[False] = False
    novelty_refinement_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10ExternalN10ShadowReportV1":
        if self.lineage_count != len(self.lineages):
            raise ValueError("shadow report lineage_count mismatch")
        counts = Counter(row.certification_status for row in self.lineages)
        if self.certified_count != counts["NOVELTY_CERTIFIED"]:
            raise ValueError("certified_count mismatch")
        if self.unresolved_count != counts["NOVELTY_UNRESOLVED"]:
            raise ValueError("unresolved_count mismatch")
        if self.rejected_count != counts["NOVELTY_REJECTED"]:
            raise ValueError("rejected_count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("external/N9/N10 shadow report SHA mismatch")
        if observed_id != "pre_n10_external_n9_n10_shadow_report_v1:" + expected_sha[:20]:
            raise ValueError("external/N9/N10 shadow report ID mismatch")
        return self


def _validate_handoff_lineage(
    row: PreN10DownstreamHandoffLineageV1,
    *,
    context: HypothesisContext,
) -> tuple[HypothesisPortfolio, LiteratureQueryPlan]:
    if not row.eligible_for_external_novelty:
        raise ValueError("handoff lineage is not external-novelty eligible")
    if row.semantic_disposition != "PASS" or row.pre_n10_status != "READY_FOR_N10":
        raise ValueError("handoff lineage lacks semantic/V_pre authority")

    paths = [
        (Path(row.portfolio_path), row.portfolio_file_sha256, "portfolio"),
        (Path(row.query_plan_path), row.query_plan_file_sha256, "query plan"),
        (Path(row.contract_report_path), row.contract_report_file_sha256, "contract"),
        (
            Path(row.semantic_disposition_path),
            row.semantic_disposition_file_sha256,
            "semantic disposition",
        ),
    ]
    for path, expected_sha, label in paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError("missing handoff " + label + ": " + str(resolved))
        if sha256_file(resolved) != expected_sha:
            raise ValueError("handoff " + label + " changed after freeze")

    portfolio = HypothesisPortfolio.model_validate_json(
        Path(row.portfolio_path).read_text(encoding="utf-8")
    )
    query_plan = LiteratureQueryPlan.model_validate_json(
        Path(row.query_plan_path).read_text(encoding="utf-8")
    )
    contract = PreN10ScientificContractReportV1.model_validate_json(
        Path(row.contract_report_path).read_text(encoding="utf-8")
    )
    disposition = HypothesisSemanticDispositionV1.model_validate_json(
        Path(row.semantic_disposition_path).read_text(encoding="utf-8")
    )

    if portfolio.portfolio_id != row.portfolio_id:
        raise ValueError("handoff portfolio ID mismatch")
    if len(portfolio.hypotheses) != 1:
        raise ValueError("downstream handoff lineage must contain one hypothesis")
    if portfolio.hypotheses[0].hypothesis_id != row.downstream_hypothesis_id:
        raise ValueError("handoff downstream hypothesis ID mismatch")
    if query_plan.plan_id != row.query_plan_id:
        raise ValueError("handoff query-plan ID mismatch")
    if query_plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("handoff query-plan portfolio provenance mismatch")
    if contract.report_id != row.contract_report_id:
        raise ValueError("handoff contract-report ID mismatch")
    if contract.disposition != "READY_FOR_N10":
        raise ValueError("handoff contract is not READY_FOR_N10")
    if disposition.disposition_id != row.semantic_disposition_id:
        raise ValueError("handoff semantic disposition ID mismatch")
    if not disposition.semantic_admissible_for_pre_n10:
        raise ValueError("handoff semantic disposition is not admissible")

    if portfolio.source_context_id != context.context_id:
        raise ValueError("handoff portfolio/context ID mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("handoff portfolio/context SHA mismatch")
    if portfolio.domain_profile_id != context.domain_profile_id:
        raise ValueError("handoff portfolio/context domain mismatch")
    return portfolio, query_plan


def _stage_paths(root: Path) -> dict[str, Path]:
    external_prefix = root / "external_novelty"
    return {
        "external_prefix": external_prefix,
        "external_plan": external_prefix.with_suffix(".claims_queries.json"),
        "external_prior": external_prefix.with_suffix(".prior_art.json"),
        "external_report": external_prefix.with_suffix(".report.json"),
        "n9_intake": root / "n9.intake.json",
        "n9_full": root / "n9.full.json",
        "n10_candidate": root / "n10.candidate.json",
        "n10_production": root / "n10.production.json",
    }


def _base_external_argv(
    *,
    portfolio_path: str,
    query_plan_path: str,
    domain_profile_id: str,
    model: str,
    base_url: str | None,
    api_key_env: str,
    provider_plan_path: str,
    output_prefix: str,
    save_prompts: bool,
) -> list[str]:
    argv = [
        "-m",
        "scripts.discovery.run_external_novelty",
        "--portfolio",
        portfolio_path,
        "--domain-profile",
        domain_profile_id,
        "--model",
        model,
        "--api-key-env",
        api_key_env,
        "--provider-plan",
        provider_plan_path,
        "--results-per-query",
        "12",
        "--max-ranked-works",
        "8",
        "--reuse-query-plan",
        query_plan_path,
        "--output-prefix",
        output_prefix,
    ]
    if base_url:
        argv += ["--base-url", base_url]
    if save_prompts:
        argv += ["--save-prompts"]
    return argv


def compile_pre_n10_external_n10_shadow_plan_v1(
    *,
    handoff: PreN10DownstreamHandoffReportV1,
    context_path: Path,
    provider_plan_path: Path,
    model: str,
    output_root: Path,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    save_prompts: bool = True,
) -> PreN10ExternalN10ShadowPlanV1:
    if not str(model).strip():
        raise ValueError("external/N9/N10 shadow requires a model")
    if not str(api_key_env).strip():
        raise ValueError("external/N9/N10 shadow requires an API-key env name")

    context_file = context_path.expanduser().resolve()
    provider_file = provider_plan_path.expanduser().resolve()
    if not context_file.is_file():
        raise ValueError("missing hypothesis context: " + str(context_file))
    if not provider_file.is_file():
        raise ValueError("missing frozen provider plan: " + str(provider_file))
    context = HypothesisContext.model_validate_json(
        context_file.read_text(encoding="utf-8")
    )

    root = output_root.expanduser().resolve()
    lineage_plans: list[PreN10ExternalN10LineagePlanV1] = []
    for row in handoff.lineages:
        portfolio, _query_plan = _validate_handoff_lineage(row, context=context)
        lineage_root = root / "lineage" / _slug(row.lineage_id)
        paths = _stage_paths(lineage_root)
        expected_scope = (
            "alpha6_original_fallback"
            if row.origin == "INITIAL_PRIMARY_READY"
            else "alpha6_post_generation_candidate"
        )
        production_module = (
            "scripts.discovery.build_nonobviousness_production_gate_v2"
            if row.origin == "INITIAL_PRIMARY_READY"
            else "scripts.discovery.build_nonobviousness_post_generation_production_gate_v2"
        )

        external_argv = _base_external_argv(
            portfolio_path=str(Path(row.portfolio_path).expanduser().resolve()),
            query_plan_path=str(Path(row.query_plan_path).expanduser().resolve()),
            domain_profile_id=portfolio.domain_profile_id,
            model=str(model),
            base_url=base_url,
            api_key_env=str(api_key_env),
            provider_plan_path=str(provider_file),
            output_prefix=str(paths["external_prefix"]),
            save_prompts=save_prompts,
        )
        intake_argv = [
            "-m",
            "scripts.discovery.build_nonobviousness_shadow",
            "--query-plan",
            str(paths["external_plan"]),
            "--external-report",
            str(paths["external_report"]),
            "--portfolio",
            str(Path(row.portfolio_path).expanduser().resolve()),
            "--output",
            str(paths["n9_intake"]),
        ]
        full_argv = [
            "-m",
            "scripts.discovery.run_nonobviousness_full_shadow",
            "--query-plan",
            str(paths["external_plan"]),
            "--external-report",
            str(paths["external_report"]),
            "--external-prior-art",
            str(paths["external_prior"]),
            "--portfolio",
            str(Path(row.portfolio_path).expanduser().resolve()),
            "--hypothesis-context",
            str(context_file),
            "--intake-shadow",
            str(paths["n9_intake"]),
            "--provider-plan",
            str(provider_file),
            "--domain-profile",
            portfolio.domain_profile_id,
            "--model",
            str(model),
            "--api-key-env",
            str(api_key_env),
            "--results-per-query",
            "12",
            "--max-ranked-works",
            "8",
            "--output",
            str(paths["n9_full"]),
        ]
        if base_url:
            full_argv += ["--base-url", base_url]
        candidate_argv = [
            "-m",
            "scripts.discovery.build_nonobviousness_production_gate_v2_candidate",
            "--query-plan",
            str(paths["external_plan"]),
            "--intake-shadow",
            str(paths["n9_intake"]),
            "--full-shadow",
            str(paths["n9_full"]),
            "--output",
            str(paths["n10_candidate"]),
        ]
        production_argv = [
            "-m",
            production_module,
            "--candidate-gate",
            str(paths["n10_candidate"]),
            "--output",
            str(paths["n10_production"]),
        ]

        stages = [
            PreN10ExternalN10StagePlanV1(
                stage="EXTERNAL_NOVELTY",
                argv=external_argv,
                expected_outputs=[
                    str(paths["external_plan"]),
                    str(paths["external_prior"]),
                    str(paths["external_report"]),
                ],
            ),
            PreN10ExternalN10StagePlanV1(
                stage="N9_INTAKE",
                argv=intake_argv,
                expected_outputs=[str(paths["n9_intake"])],
            ),
            PreN10ExternalN10StagePlanV1(
                stage="N9_FULL_CLOSURE",
                argv=full_argv,
                expected_outputs=[str(paths["n9_full"])],
                dynamic_max_ready_claims_from_intake=True,
            ),
            PreN10ExternalN10StagePlanV1(
                stage="N10_CANDIDATE_GATE",
                argv=candidate_argv,
                expected_outputs=[str(paths["n10_candidate"])],
            ),
            PreN10ExternalN10StagePlanV1(
                stage="N10_PRODUCTION_GATE",
                argv=production_argv,
                expected_outputs=[str(paths["n10_production"])],
            ),
        ]
        lineage_plans.append(
            PreN10ExternalN10LineagePlanV1(
                lineage_id=row.lineage_id,
                origin=row.origin,
                source_hypothesis_id=row.source_hypothesis_id,
                downstream_hypothesis_id=row.downstream_hypothesis_id,
                portfolio_path=str(Path(row.portfolio_path).expanduser().resolve()),
                portfolio_id=row.portfolio_id,
                portfolio_file_sha256=row.portfolio_file_sha256,
                query_plan_path=str(Path(row.query_plan_path).expanduser().resolve()),
                query_plan_id=row.query_plan_id,
                query_plan_file_sha256=row.query_plan_file_sha256,
                contract_report_path=str(Path(row.contract_report_path).expanduser().resolve()),
                contract_report_id=row.contract_report_id,
                contract_report_file_sha256=row.contract_report_file_sha256,
                semantic_disposition_path=str(
                    Path(row.semantic_disposition_path).expanduser().resolve()
                ),
                semantic_disposition_id=row.semantic_disposition_id,
                semantic_disposition_file_sha256=row.semantic_disposition_file_sha256,
                output_dir=str(lineage_root),
                external_output_query_plan=str(paths["external_plan"]),
                external_prior_art=str(paths["external_prior"]),
                external_report=str(paths["external_report"]),
                n9_intake=str(paths["n9_intake"]),
                n9_full_closure=str(paths["n9_full"]),
                n10_candidate_gate=str(paths["n10_candidate"]),
                n10_production_gate=str(paths["n10_production"]),
                expected_n10_authority_scope=expected_scope,
                n10_production_module=production_module,
                stages=stages,
            )
        )

    body = {
        "schema_version": "pre-n10-external-n9-n10-shadow-plan-v1",
        "source_handoff_report_id": handoff.report_id,
        "source_handoff_report_sha256": handoff.report_sha256,
        "source_context_path": str(context_file),
        "source_context_file_sha256": sha256_file(context_file),
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "provider_plan_path": str(provider_file),
        "provider_plan_file_sha256": sha256_file(provider_file),
        "model": str(model),
        "base_url": base_url,
        "api_key_env": str(api_key_env),
        "results_per_query": 12,
        "max_ranked_works": 8,
        "save_prompts": bool(save_prompts),
        "lineages": [row.model_dump(mode="json") for row in lineage_plans],
        "lineage_count": len(lineage_plans),
        "exact_handoff_population_consumed": True,
        "frozen_query_plan_reused": True,
        "external_novelty_stage_max_per_lineage": 1,
        "n9_intake_stage_max_per_lineage": 1,
        "n9_full_closure_stage_max_per_lineage": 1,
        "n10_candidate_stage_max_per_lineage": 1,
        "n10_production_stage_max_per_lineage": 1,
        "targeted_novelty_continuation_max_per_lineage": 0,
        "novelty_refinement_max_per_lineage": 0,
        "second_regeneration_max_per_lineage": 0,
        "production_selection_authority_created_by_runner": False,
    }
    digest = _sha256_json(body)
    return PreN10ExternalN10ShadowPlanV1(
        **body,
        plan_id="pre_n10_external_n9_n10_shadow_plan_v1:" + digest[:20],
        plan_sha256=digest,
    )


def classify_pre_n10_n10_production_gate_v1(
    *,
    origin: DownstreamOriginV1,
    downstream_hypothesis_id: str,
    portfolio_id: str,
    query_plan_id: str,
    gate: dict[str, object],
) -> tuple[N10CertificationStatusV1, dict[str, object]]:
    if gate.get("schema_version") != "scientific-novelty-fallback-gate-v2":
        raise ValueError("unexpected role-aware N10 production schema")
    if gate.get("production_authority") is not True:
        raise ValueError("N10 production gate lacks authority")
    expected_scope = (
        "alpha6_original_fallback"
        if origin == "INITIAL_PRIMARY_READY"
        else "alpha6_post_generation_candidate"
    )
    if gate.get("authority_scope") != expected_scope:
        raise ValueError("N10 production authority scope/origin mismatch")
    if gate.get("authority_source") != "n10_role_aware_nonobviousness_v2":
        raise ValueError("unexpected role-aware N10 authority source")
    if gate.get("positive_authority_requires") != (
        "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS"
    ):
        raise ValueError("unexpected role-aware positive-authority contract")
    if gate.get("conditional_is_positive") is not False:
        raise ValueError("CONDITIONAL must not be positive")
    if gate.get("absence_is_novelty") is not False:
        raise ValueError("search-bounded absence cannot become novelty")
    if gate.get("candidate_semantics_preserved") is not True:
        raise ValueError("N10 candidate semantics were not preserved")
    if gate.get("source_portfolio_id") != portfolio_id:
        raise ValueError("N10 production gate portfolio provenance mismatch")
    if gate.get("source_query_plan_id") != query_plan_id:
        raise ValueError("N10 production gate query-plan provenance mismatch")

    rows = gate.get("gates")
    if not isinstance(rows, list):
        raise ValueError("N10 production gate rows must be a list")
    matches = [
        row for row in rows
        if isinstance(row, dict)
        and str(row.get("hypothesis_id") or "") == downstream_hypothesis_id
    ]
    if len(matches) != 1:
        raise ValueError("N10 production gate must resolve exactly one downstream hypothesis")
    if len(rows) != 1:
        raise ValueError("single-lineage N10 production gate contains extra hypotheses")
    row = matches[0]
    selection = str(row.get("selection_class") or "")
    if selection not in {"ELIGIBLE", "CONDITIONAL", "INELIGIBLE"}:
        raise ValueError("invalid N10 selection class")
    positive = row.get("positive_nonobviousness_authority")
    fallback = row.get("fallback_allowed")
    if not isinstance(positive, bool) or not isinstance(fallback, bool):
        raise ValueError("N10 authority booleans are malformed")
    if selection == "ELIGIBLE":
        if positive is not True or fallback is not True:
            raise ValueError("ELIGIBLE N10 row lacks positive authority/permission")
        status: N10CertificationStatusV1 = "NOVELTY_CERTIFIED"
    elif selection == "CONDITIONAL":
        if positive is not False or fallback is not False:
            raise ValueError("CONDITIONAL cannot carry positive authority/permission")
        status = "NOVELTY_UNRESOLVED"
    else:
        if positive is not False or fallback is not False:
            raise ValueError("INELIGIBLE cannot carry positive authority/permission")
        status = "NOVELTY_REJECTED"
    return status, row


def build_pre_n10_external_n10_lineage_result_v1(
    *,
    plan: PreN10ExternalN10LineagePlanV1,
    production_gate: dict[str, object],
) -> PreN10ExternalN10LineageResultV1:
    source_plan = LiteratureQueryPlan.model_validate_json(
        Path(plan.query_plan_path).read_text(encoding="utf-8")
    )
    external_plan = LiteratureQueryPlan.model_validate_json(
        Path(plan.external_output_query_plan).read_text(encoding="utf-8")
    )
    if external_plan.model_dump(mode="json") != source_plan.model_dump(mode="json"):
        raise ValueError("external novelty changed the frozen V_pre query plan")
    if sha256_file(plan.portfolio_path) != plan.portfolio_file_sha256:
        raise ValueError("source portfolio changed during external/N9/N10 execution")
    if sha256_file(plan.query_plan_path) != plan.query_plan_file_sha256:
        raise ValueError("source query plan changed during external/N9/N10 execution")
    if sha256_file(plan.contract_report_path) != plan.contract_report_file_sha256:
        raise ValueError("source contract changed during external/N9/N10 execution")
    if sha256_file(plan.semantic_disposition_path) != plan.semantic_disposition_file_sha256:
        raise ValueError("semantic disposition changed during external/N9/N10 execution")

    required = [
        plan.external_report,
        plan.n9_intake,
        plan.n9_full_closure,
        plan.n10_candidate_gate,
        plan.n10_production_gate,
    ]
    for value in required:
        if not Path(value).is_file():
            raise ValueError("missing external/N9/N10 output: " + value)

    status, row = classify_pre_n10_n10_production_gate_v1(
        origin=plan.origin,
        downstream_hypothesis_id=plan.downstream_hypothesis_id,
        portfolio_id=plan.portfolio_id,
        query_plan_id=plan.query_plan_id,
        gate=production_gate,
    )
    return PreN10ExternalN10LineageResultV1(
        lineage_id=plan.lineage_id,
        origin=plan.origin,
        source_hypothesis_id=plan.source_hypothesis_id,
        downstream_hypothesis_id=plan.downstream_hypothesis_id,
        portfolio_id=plan.portfolio_id,
        query_plan_id=plan.query_plan_id,
        external_report_path=plan.external_report,
        external_report_file_sha256=sha256_file(plan.external_report),
        n9_intake_path=plan.n9_intake,
        n9_intake_file_sha256=sha256_file(plan.n9_intake),
        n9_full_closure_path=plan.n9_full_closure,
        n9_full_closure_file_sha256=sha256_file(plan.n9_full_closure),
        n10_candidate_gate_path=plan.n10_candidate_gate,
        n10_candidate_gate_file_sha256=sha256_file(plan.n10_candidate_gate),
        n10_production_gate_path=plan.n10_production_gate,
        n10_production_gate_file_sha256=sha256_file(plan.n10_production_gate),
        n10_authority_scope=plan.expected_n10_authority_scope,
        n10_selection_class=str(row["selection_class"]),
        positive_nonobviousness_authority=bool(
            row["positive_nonobviousness_authority"]
        ),
        fallback_allowed=bool(row["fallback_allowed"]),
        certification_status=status,
    )


def build_pre_n10_external_n10_shadow_report_v1(
    *,
    execution_plan: PreN10ExternalN10ShadowPlanV1,
    handoff: PreN10DownstreamHandoffReportV1,
    lineages: list[PreN10ExternalN10LineageResultV1],
) -> PreN10ExternalN10ShadowReportV1:
    if execution_plan.source_handoff_report_id != handoff.report_id:
        raise ValueError("execution plan/handoff report ID mismatch")
    if execution_plan.source_handoff_report_sha256 != handoff.report_sha256:
        raise ValueError("execution plan/handoff report SHA mismatch")
    expected_ids = [row.lineage_id for row in execution_plan.lineages]
    result_ids = [row.lineage_id for row in lineages]
    if result_ids != expected_ids:
        raise ValueError("external/N9/N10 result population/order differs from frozen plan")
    if set(expected_ids) != {row.lineage_id for row in handoff.lineages}:
        raise ValueError("external/N9/N10 plan population differs from handoff eligibility")

    counts = Counter(row.certification_status for row in lineages)
    body = {
        "schema_version": "pre-n10-external-n9-n10-shadow-report-v1",
        "source_execution_plan_id": execution_plan.plan_id,
        "source_execution_plan_sha256": execution_plan.plan_sha256,
        "source_handoff_report_id": handoff.report_id,
        "source_handoff_report_sha256": handoff.report_sha256,
        "lineages": [row.model_dump(mode="json") for row in lineages],
        "lineage_count": len(lineages),
        "certified_count": counts["NOVELTY_CERTIFIED"],
        "unresolved_count": counts["NOVELTY_UNRESOLVED"],
        "rejected_count": counts["NOVELTY_REJECTED"],
        "exact_handoff_population_consumed": True,
        "pre_n10_ready_was_required": True,
        "semantic_disposition_pass_was_required": True,
        "frozen_query_plan_reused": True,
        "candidate_survival_authority": False,
        "novelty_certification_authority": True,
        "authority_source": "n10_role_aware_nonobviousness_v2",
        "authority_mode": "certification_only_shadow",
        "conditional_is_positive": False,
        "absence_is_novelty": False,
        "ineligible_deletes_scientific_candidate": False,
        "targeted_novelty_continuation_performed": False,
        "novelty_refinement_performed": False,
        "second_regeneration_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreN10ExternalN10ShadowReportV1(
        **body,
        report_id="pre_n10_external_n9_n10_shadow_report_v1:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "N10CertificationStatusV1",
    "PreN10ExternalN10LineagePlanV1",
    "PreN10ExternalN10LineageResultV1",
    "PreN10ExternalN10ShadowPlanV1",
    "PreN10ExternalN10ShadowReportV1",
    "PreN10ExternalN10StagePlanV1",
    "build_pre_n10_external_n10_lineage_result_v1",
    "build_pre_n10_external_n10_shadow_report_v1",
    "classify_pre_n10_n10_production_gate_v1",
    "compile_pre_n10_external_n10_shadow_plan_v1",
    "sha256_file",
]
