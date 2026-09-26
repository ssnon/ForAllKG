from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    AtomicScientificSpecificationBundle,
)
from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    PreN10RelationalBindingBridgeReportV1,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    RelationalAtomicEndpointBindingReport,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    RelationalScientificVerifierInputFreeze,
    RelationalScientificVerifierRunManifest,
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


def pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_exact_or_validate(path: Path, payload: object) -> str:
    resolved = path.expanduser().resolve()
    expected = pretty_json_bytes(payload)
    if resolved.exists():
        observed = resolved.read_bytes()
        if observed != expected:
            raise ValueError(
                "existing write-once V_post artifact differs: " + str(resolved)
            )
        return hashlib.sha256(observed).hexdigest()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


def _slug(value: str) -> str:
    result = str(value).replace(":", "_").replace("/", "_")
    if not result or result in {".", ".."}:
        raise ValueError("invalid downstream lineage ID")
    return result


_VERIFIER_STAGE_NAMES: tuple[str, ...] = (
    "relational_atomic_projection_and_relation_ir",
    "relational_atomic_identity_and_factor_projection",
    "supporting_projection_retrieval",
    "counterevidence_projection_retrieval",
    "semantic_second_pass_resolution",
    "exhaustive_relation_adjudication",
    "claim_evidence_graph",
    "structural_claim_centrality",
    "hypothesis_evidence_aggregation",
    "positive_nonobviousness_basis",
    "positive_nonobviousness_adjudication",
    "external_novelty_lineage_projection",
    "scientific_certification_gate",
)


class PreN10VPostStagePlanV1(StrictModel):
    stage: Literal["literal_endpoint_binding", "relational_scientific_verifier"]
    argv: list[str]
    expected_outputs: list[str]


class PreN10VPostLineagePlanV1(StrictModel):
    lineage_id: str
    origin: Literal["INITIAL_PRIMARY_READY", "REGENERATED_REENTRY_READY"]
    source_hypothesis_id: str
    downstream_hypothesis_id: str

    n10_certification_status: Literal[
        "NOVELTY_CERTIFIED",
        "NOVELTY_UNRESOLVED",
        "NOVELTY_REJECTED",
    ]
    n10_selection_class: Literal["ELIGIBLE", "CONDITIONAL", "INELIGIBLE"]

    binding_status: str
    execution_required: bool

    binding_plan_path: str
    binding_plan_id: str
    binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_external_report_path: str
    source_external_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain_profile_id: str | None = None

    output_dir: str
    endpoint_report_path: str | None = None
    endpoint_prompt_path: str | None = None
    verifier_output_dir: str | None = None
    verifier_input_freeze_path: str | None = None
    verifier_manifest_path: str | None = None
    verifier_certification_path: str | None = None

    stages: list[PreN10VPostStagePlanV1] = Field(default_factory=list)

    n10_status_filters_vpost_reachability: Literal[False] = False
    representation_regeneration_allowed: Literal[False] = False
    external_novelty_reassessment_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "PreN10VPostLineagePlanV1":
        ready = self.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        if self.execution_required != ready:
            raise ValueError("V_post execution_required/binding_status mismatch")
        output_fields = (
            self.endpoint_report_path,
            self.verifier_output_dir,
            self.verifier_input_freeze_path,
            self.verifier_manifest_path,
            self.verifier_certification_path,
        )
        if ready:
            if self.domain_profile_id is None:
                raise ValueError("binding-ready V_post lineage lacks domain profile")
            if any(value is None for value in output_fields):
                raise ValueError("binding-ready V_post lineage lacks output paths")
            if [row.stage for row in self.stages] != [
                "literal_endpoint_binding",
                "relational_scientific_verifier",
            ]:
                raise ValueError("V_post stage order mismatch")
            endpoint = self.stages[0]
            verifier = self.stages[1]
            if len(endpoint.argv) < 2 or endpoint.argv[1] != (
                "scripts.discovery.run_relational_atomic_endpoint_binding"
            ):
                raise ValueError("unexpected endpoint binding module")
            if len(verifier.argv) < 2 or verifier.argv[1] != (
                "scripts.discovery.run_relational_scientific_verifier_shadow_e2e"
            ):
                raise ValueError("unexpected relational verifier module")
            if self.binding_plan_path not in endpoint.argv:
                raise ValueError("endpoint stage binding-plan path mismatch")
            if self.binding_plan_path not in verifier.argv:
                raise ValueError("verifier stage binding-plan path mismatch")
            if str(self.endpoint_report_path) not in verifier.argv:
                raise ValueError("verifier stage endpoint-report path mismatch")
            if self.source_external_report_path not in verifier.argv:
                raise ValueError("verifier source-external-report path mismatch")
        else:
            if self.stages:
                raise ValueError("non-binding-ready lineage cannot carry V_post stages")
            if any(value is not None for value in output_fields):
                raise ValueError(
                    "non-binding-ready lineage cannot reserve execution outputs"
                )
            if self.domain_profile_id is not None:
                raise ValueError(
                    "non-binding-ready lineage cannot carry verifier domain profile"
                )
        return self


class PreN10VPostShadowPlanV1(StrictModel):
    schema_version: Literal[
        "pre-n10-vpost-shadow-plan-v1"
    ] = "pre-n10-vpost-shadow-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_binding_bridge_report_id: str
    source_binding_bridge_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_plan_path: str
    provider_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_spec_bundle_path: str | None = None
    canonical_spec_bundle_file_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    model: str
    base_url: str | None = None
    api_key_env: str
    save_prompts: bool
    allow_dirty_worktree: bool

    lineages: list[PreN10VPostLineagePlanV1]
    lineage_count: int = Field(ge=0)
    execution_required_lineage_count: int = Field(ge=0)
    skipped_not_binding_ready_count: int = Field(ge=0)

    endpoint_binding_stage_max_per_ready_lineage: Literal[1] = 1
    relational_verifier_stage_max_per_ready_lineage: Literal[1] = 1
    verifier_expected_internal_stage_count: Literal[13] = 13
    n10_status_filters_vpost_reachability: Literal[False] = False
    external_novelty_reassessment_allowed: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "PreN10VPostShadowPlanV1":
        if self.lineage_count != len(self.lineages):
            raise ValueError("V_post plan lineage_count mismatch")
        required = sum(row.execution_required for row in self.lineages)
        if self.execution_required_lineage_count != required:
            raise ValueError("V_post execution-required count mismatch")
        if self.skipped_not_binding_ready_count != len(self.lineages) - required:
            raise ValueError("V_post skipped count mismatch")
        ids = [row.lineage_id for row in self.lineages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate V_post lineage ID")
        if (
            (self.canonical_spec_bundle_path is None)
            != (self.canonical_spec_bundle_file_sha256 is None)
        ):
            raise ValueError(
                "V_post canonical specification path/SHA mismatch"
            )
        for lineage in self.lineages:
            verifier_stages = [
                row
                for row in lineage.stages
                if row.stage == "relational_scientific_verifier"
            ]
            if not verifier_stages:
                continue
            argv = verifier_stages[0].argv
            has_bundle = "--canonical-spec-bundle" in argv
            if has_bundle != (
                self.canonical_spec_bundle_path is not None
            ):
                raise ValueError(
                    "V_post verifier canonical bundle argv mismatch"
                )
            if has_bundle:
                bundle_index = argv.index(
                    "--canonical-spec-bundle"
                )
                sha_index = argv.index(
                    "--canonical-spec-bundle-sha256"
                )
                if argv[bundle_index + 1] != (
                    self.canonical_spec_bundle_path
                ):
                    raise ValueError(
                        "V_post verifier canonical bundle path mismatch"
                    )
                if argv[sha_index + 1] != (
                    self.canonical_spec_bundle_file_sha256
                ):
                    raise ValueError(
                        "V_post verifier canonical bundle SHA mismatch"
                    )
        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("V_post shadow plan SHA mismatch")
        if observed_id != "pre_n10_vpost_shadow_plan_v1:" + expected_sha[:20]:
            raise ValueError("V_post shadow plan ID mismatch")
        return self


VPostLineageStatusV1 = Literal[
    "NOT_BINDING_READY",
    "VPOST_COMPLETED",
]


class PreN10VPostLineageResultV1(StrictModel):
    lineage_id: str
    origin: Literal["INITIAL_PRIMARY_READY", "REGENERATED_REENTRY_READY"]
    source_hypothesis_id: str
    downstream_hypothesis_id: str
    n10_certification_status: str
    n10_selection_class: str
    binding_status: str
    status: VPostLineageStatusV1

    endpoint_report_path: str | None = None
    endpoint_report_id: str | None = None
    endpoint_report_file_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    endpoint_llm_calls: int = Field(ge=0)

    verifier_input_freeze_path: str | None = None
    verifier_input_freeze_id: str | None = None
    verifier_manifest_path: str | None = None
    verifier_manifest_id: str | None = None
    verifier_certification_path: str | None = None
    verifier_certification_report_id: str | None = None
    certification_decision: str | None = None
    bounded_closure_state: str | None = None
    bounded_external_distinctness_state: str | None = None
    positive_nonobviousness_authority_state: str | None = None
    fatal_blocker_state: str | None = None

    endpoint_binding_stage_invocations: int = Field(ge=0, le=1)
    relational_verifier_stage_invocations: int = Field(ge=0, le=1)

    external_novelty_reassessed: Literal[False] = False
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "PreN10VPostLineageResultV1":
        completed = self.status == "VPOST_COMPLETED"
        output_fields = (
            self.endpoint_report_path,
            self.endpoint_report_id,
            self.endpoint_report_file_sha256,
            self.verifier_input_freeze_path,
            self.verifier_input_freeze_id,
            self.verifier_manifest_path,
            self.verifier_manifest_id,
            self.verifier_certification_path,
            self.verifier_certification_report_id,
            self.certification_decision,
            self.bounded_closure_state,
            self.bounded_external_distinctness_state,
            self.positive_nonobviousness_authority_state,
            self.fatal_blocker_state,
        )
        if completed:
            if any(value is None for value in output_fields):
                raise ValueError("completed V_post lineage lacks outputs")
            if self.endpoint_binding_stage_invocations != 1:
                raise ValueError("completed V_post lineage endpoint count mismatch")
            if self.relational_verifier_stage_invocations != 1:
                raise ValueError("completed V_post lineage verifier count mismatch")
        else:
            if any(value is not None for value in output_fields):
                raise ValueError("skipped V_post lineage cannot carry outputs")
            if self.endpoint_llm_calls != 0:
                raise ValueError("skipped V_post lineage cannot call endpoint LLM")
            if self.endpoint_binding_stage_invocations != 0:
                raise ValueError("skipped V_post lineage endpoint count mismatch")
            if self.relational_verifier_stage_invocations != 0:
                raise ValueError("skipped V_post lineage verifier count mismatch")
        return self


class PreN10VPostShadowReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-vpost-shadow-report-v1"
    ] = "pre-n10-vpost-shadow-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_bridge_report_id: str
    source_binding_bridge_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    lineages: list[PreN10VPostLineageResultV1]
    lineage_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    not_binding_ready_count: int = Field(ge=0)
    certification_decision_counts: dict[str, int]

    endpoint_binding_stage_invocations: int = Field(ge=0)
    relational_verifier_stage_invocations: int = Field(ge=0)

    n10_status_did_not_filter_vpost_reachability: Literal[True] = True
    literal_binding_required_before_verifier: Literal[True] = True
    verifier_internal_stage_contract_frozen: Literal[True] = True
    verifier_internal_stage_count: Literal[13] = 13
    external_novelty_reassessed: Literal[False] = False
    verifier_result_consumed_by_production: Literal[False] = False
    production_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10VPostShadowReportV1":
        if self.lineage_count != len(self.lineages):
            raise ValueError("V_post report lineage_count mismatch")
        completed = sum(row.status == "VPOST_COMPLETED" for row in self.lineages)
        if self.completed_count != completed:
            raise ValueError("V_post completed count mismatch")
        if self.not_binding_ready_count != len(self.lineages) - completed:
            raise ValueError("V_post not-binding-ready count mismatch")
        if self.endpoint_binding_stage_invocations != sum(
            row.endpoint_binding_stage_invocations for row in self.lineages
        ):
            raise ValueError("V_post endpoint invocation count mismatch")
        if self.relational_verifier_stage_invocations != sum(
            row.relational_verifier_stage_invocations for row in self.lineages
        ):
            raise ValueError("V_post verifier invocation count mismatch")
        expected = Counter(
            row.certification_decision
            for row in self.lineages
            if row.certification_decision is not None
        )
        if dict(sorted(expected.items())) != dict(
            sorted(self.certification_decision_counts.items())
        ):
            raise ValueError("V_post certification decision counts mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("V_post shadow report SHA mismatch")
        if observed_id != "pre_n10_vpost_shadow_report_v1:" + expected_sha[:20]:
            raise ValueError("V_post shadow report ID mismatch")
        return self


def _load_binding_plan(row) -> tuple[RelationalAtomicBindingPlan, HypothesisPortfolio]:
    plan_path = Path(row.binding_plan_path).expanduser().resolve()
    if not plan_path.is_file():
        raise ValueError("missing relational binding plan: " + str(plan_path))
    if sha256_file(plan_path) != row.binding_plan_file_sha256:
        raise ValueError("relational binding plan file changed after bridge")
    plan = RelationalAtomicBindingPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    if plan.plan_id != row.binding_plan_id:
        raise ValueError("binding bridge/plan ID mismatch")
    if plan.plan_sha256 != row.binding_plan_sha256:
        raise ValueError("binding bridge/plan SHA mismatch")
    if len(plan.hypotheses) != 1:
        raise ValueError("V_post lineage binding plan must contain one hypothesis")
    hypothesis = plan.hypotheses[0]
    if hypothesis.final_hypothesis_id != row.downstream_hypothesis_id:
        raise ValueError("binding bridge/final hypothesis mismatch")
    if hypothesis.candidate_hypothesis_id != row.downstream_hypothesis_id:
        raise ValueError("V_post requires exact candidate/final identity")
    if hypothesis.candidate_final_authority_equivalent is not True:
        raise ValueError("binding plan lacks candidate/final authority equivalence")

    portfolio_path = Path(
        plan.source_alpha6_candidate_portfolio
    ).expanduser().resolve()
    if not portfolio_path.is_file():
        raise ValueError("missing binding-plan source portfolio")
    if sha256_file(portfolio_path) != (
        plan.source_alpha6_candidate_portfolio_sha256
    ):
        raise ValueError("binding-plan source portfolio changed")
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    if len(portfolio.hypotheses) != 1:
        raise ValueError("V_post lineage source portfolio must contain one hypothesis")
    if portfolio.hypotheses[0].hypothesis_id != row.downstream_hypothesis_id:
        raise ValueError("V_post source portfolio hypothesis mismatch")
    return plan, portfolio


def compile_pre_n10_vpost_shadow_plan_v1(
    *,
    bridge: PreN10RelationalBindingBridgeReportV1,
    provider_plan_path: Path,
    model: str,
    output_root: Path,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    save_prompts: bool = False,
    allow_dirty_worktree: bool = False,
    canonical_spec_bundle_path: Path | None = None,
) -> PreN10VPostShadowPlanV1:
    if not str(model).strip():
        raise ValueError("V_post shadow requires a model")
    if not str(api_key_env).strip():
        raise ValueError("V_post shadow requires an API-key env name")
    provider = provider_plan_path.expanduser().resolve()
    if not provider.is_file():
        raise ValueError("missing frozen provider plan: " + str(provider))

    canonical_bundle = (
        canonical_spec_bundle_path.expanduser().resolve()
        if canonical_spec_bundle_path is not None
        else None
    )
    canonical_bundle_sha = None
    if canonical_bundle is not None:
        if not canonical_bundle.is_file():
            raise ValueError(
                "missing canonical specification bundle: "
                + str(canonical_bundle)
            )
        AtomicScientificSpecificationBundle.model_validate_json(
            canonical_bundle.read_text(encoding="utf-8")
        )
        canonical_bundle_sha = sha256_file(canonical_bundle)

    root = output_root.expanduser().resolve()
    lineages: list[PreN10VPostLineagePlanV1] = []

    for row in bridge.lineages:
        plan, portfolio = _load_binding_plan(row)
        external_path = Path(row.source_external_report_path).expanduser().resolve()
        if not external_path.is_file():
            raise ValueError(
                "missing frozen source external report: " + str(external_path)
            )
        if sha256_file(external_path) != row.source_external_report_file_sha256:
            raise ValueError("source external report changed after binding bridge")

        ready = row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        lineage_root = root / "lineage" / _slug(row.lineage_id)

        if not ready:
            lineages.append(
                PreN10VPostLineagePlanV1(
                    lineage_id=row.lineage_id,
                    origin=row.origin,
                    source_hypothesis_id=row.source_hypothesis_id,
                    downstream_hypothesis_id=row.downstream_hypothesis_id,
                    n10_certification_status=row.n10_certification_status,
                    n10_selection_class=row.n10_selection_class,
                    binding_status=row.binding_status,
                    execution_required=False,
                    binding_plan_path=str(
                        Path(row.binding_plan_path).expanduser().resolve()
                    ),
                    binding_plan_id=row.binding_plan_id,
                    binding_plan_sha256=row.binding_plan_sha256,
                    binding_plan_file_sha256=row.binding_plan_file_sha256,
                    source_external_report_path=str(external_path),
                    source_external_report_file_sha256=(
                        row.source_external_report_file_sha256
                    ),
                    domain_profile_id=None,
                    output_dir=str(lineage_root),
                )
            )
            continue

        endpoint_report = lineage_root / "literal_endpoint_binding.json"
        endpoint_prompt = (
            lineage_root / "literal_endpoint_binding.prompt.txt"
            if save_prompts
            else None
        )
        verifier_root = lineage_root / "relational_verifier"
        verifier_freeze = (
            verifier_root / "relational_scientific_verifier.input_freeze.json"
        )
        verifier_manifest = (
            verifier_root / "relational_scientific_verifier.run_manifest.json"
        )
        verifier_certification = (
            verifier_root / "scientific_certification.report.json"
        )

        endpoint_argv = [
            "-m",
            "scripts.discovery.run_relational_atomic_endpoint_binding",
            "--plan",
            str(Path(row.binding_plan_path).expanduser().resolve()),
            "--output",
            str(endpoint_report),
            "--model",
            model,
            "--api-key-env",
            api_key_env,
            "--instructor-mode",
            "JSON",
            "--temperature",
            "0.0",
            "--parse-retries",
            "3",
            "--timeout",
            "180.0",
        ]
        if base_url:
            endpoint_argv += ["--base-url", base_url]
        if endpoint_prompt is not None:
            endpoint_argv += ["--prompt-output", str(endpoint_prompt)]

        verifier_argv = [
            "-m",
            "scripts.discovery.run_relational_scientific_verifier_shadow_e2e",
            "--plan",
            str(Path(row.binding_plan_path).expanduser().resolve()),
            "--endpoint-report",
            str(endpoint_report),
            "--provider-plan",
            str(provider),
            "--source-external-report",
            str(external_path),
            "--final-hypothesis-id",
            row.downstream_hypothesis_id,
            "--domain-profile",
            portfolio.domain_profile_id,
            "--output-dir",
            str(verifier_root),
            "--model",
            model,
            "--api-key-env",
            api_key_env,
            "--support-results-per-query",
            "12",
            "--second-pass-results-per-query",
            "16",
            "--max-review-works-per-claim",
            "20",
            "--max-exhaustive-rounds",
            "3",
            "--max-second-pass-queries-per-claim",
            "8",
            "--max-resolution-queries",
            "24",
            "--max-alias-query-variants-per-projection",
            "4",
            "--max-lower-order-factor-order",
            "1",
            "--max-source-alias-variants-per-projection",
            "2",
        ]
        if canonical_bundle is not None:
            assert canonical_bundle_sha is not None
            verifier_argv += [
                "--canonical-spec-bundle",
                str(canonical_bundle),
                "--canonical-spec-bundle-sha256",
                canonical_bundle_sha,
            ]
        if base_url:
            verifier_argv += ["--base-url", base_url]
        if save_prompts:
            verifier_argv += ["--save-prompts"]
        if allow_dirty_worktree:
            verifier_argv += ["--allow-dirty-worktree"]

        lineages.append(
            PreN10VPostLineagePlanV1(
                lineage_id=row.lineage_id,
                origin=row.origin,
                source_hypothesis_id=row.source_hypothesis_id,
                downstream_hypothesis_id=row.downstream_hypothesis_id,
                n10_certification_status=row.n10_certification_status,
                n10_selection_class=row.n10_selection_class,
                binding_status=row.binding_status,
                execution_required=True,
                binding_plan_path=str(
                    Path(row.binding_plan_path).expanduser().resolve()
                ),
                binding_plan_id=row.binding_plan_id,
                binding_plan_sha256=row.binding_plan_sha256,
                binding_plan_file_sha256=row.binding_plan_file_sha256,
                source_external_report_path=str(external_path),
                source_external_report_file_sha256=(
                    row.source_external_report_file_sha256
                ),
                domain_profile_id=portfolio.domain_profile_id,
                output_dir=str(lineage_root),
                endpoint_report_path=str(endpoint_report),
                endpoint_prompt_path=(
                    str(endpoint_prompt) if endpoint_prompt is not None else None
                ),
                verifier_output_dir=str(verifier_root),
                verifier_input_freeze_path=str(verifier_freeze),
                verifier_manifest_path=str(verifier_manifest),
                verifier_certification_path=str(verifier_certification),
                stages=[
                    PreN10VPostStagePlanV1(
                        stage="literal_endpoint_binding",
                        argv=endpoint_argv,
                        expected_outputs=[str(endpoint_report)],
                    ),
                    PreN10VPostStagePlanV1(
                        stage="relational_scientific_verifier",
                        argv=verifier_argv,
                        expected_outputs=[
                            str(verifier_freeze),
                            str(verifier_manifest),
                            str(verifier_certification),
                        ],
                    ),
                ],
            )
        )

    body = {
        "schema_version": "pre-n10-vpost-shadow-plan-v1",
        "source_binding_bridge_report_id": bridge.report_id,
        "source_binding_bridge_report_sha256": bridge.report_sha256,
        "provider_plan_path": str(provider),
        "provider_plan_file_sha256": sha256_file(provider),
        "canonical_spec_bundle_path": (
            str(canonical_bundle)
            if canonical_bundle is not None
            else None
        ),
        "canonical_spec_bundle_file_sha256": canonical_bundle_sha,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "save_prompts": save_prompts,
        "allow_dirty_worktree": allow_dirty_worktree,
        "lineages": [row.model_dump(mode="json") for row in lineages],
        "lineage_count": len(lineages),
        "execution_required_lineage_count": sum(
            row.execution_required for row in lineages
        ),
        "skipped_not_binding_ready_count": sum(
            not row.execution_required for row in lineages
        ),
        "endpoint_binding_stage_max_per_ready_lineage": 1,
        "relational_verifier_stage_max_per_ready_lineage": 1,
        "verifier_expected_internal_stage_count": 13,
        "n10_status_filters_vpost_reachability": False,
        "external_novelty_reassessment_allowed": False,
        "second_regeneration_allowed": False,
        "production_selection_authority": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreN10VPostShadowPlanV1(
        **body,
        plan_id="pre_n10_vpost_shadow_plan_v1:" + digest[:20],
        plan_sha256=digest,
    )


def _load_json_object(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object: " + str(path))
    return payload


def build_pre_n10_vpost_lineage_result_v1(
    *,
    plan: PreN10VPostLineagePlanV1,
) -> PreN10VPostLineageResultV1:
    if sha256_file(plan.binding_plan_path) != plan.binding_plan_file_sha256:
        raise ValueError("binding plan changed during V_post execution")
    if (
        sha256_file(plan.source_external_report_path)
        != plan.source_external_report_file_sha256
    ):
        raise ValueError("source external report changed during V_post execution")

    if not plan.execution_required:
        return PreN10VPostLineageResultV1(
            lineage_id=plan.lineage_id,
            origin=plan.origin,
            source_hypothesis_id=plan.source_hypothesis_id,
            downstream_hypothesis_id=plan.downstream_hypothesis_id,
            n10_certification_status=plan.n10_certification_status,
            n10_selection_class=plan.n10_selection_class,
            binding_status=plan.binding_status,
            status="NOT_BINDING_READY",
            endpoint_llm_calls=0,
            endpoint_binding_stage_invocations=0,
            relational_verifier_stage_invocations=0,
        )

    required = (
        plan.endpoint_report_path,
        plan.verifier_input_freeze_path,
        plan.verifier_manifest_path,
        plan.verifier_certification_path,
    )
    if any(value is None for value in required):
        raise ValueError("binding-ready V_post plan lacks output path")
    for value in required:
        assert value is not None
        if not Path(value).is_file():
            raise ValueError("missing V_post execution output: " + value)

    binding_plan = RelationalAtomicBindingPlan.model_validate_json(
        Path(plan.binding_plan_path).read_text(encoding="utf-8")
    )
    endpoint = RelationalAtomicEndpointBindingReport.model_validate_json(
        Path(str(plan.endpoint_report_path)).read_text(encoding="utf-8")
    )
    if endpoint.source_binding_plan_id != binding_plan.plan_id:
        raise ValueError("endpoint report/binding plan ID mismatch")
    if endpoint.source_binding_plan_sha256 != binding_plan.plan_sha256:
        raise ValueError("endpoint report/binding plan SHA mismatch")
    if endpoint.selected_hypothesis_count != 1:
        raise ValueError("V_post endpoint report must select one hypothesis")
    if endpoint.selected_claim_count < 1:
        raise ValueError("V_post endpoint report must select at least one claim")
    if {
        row.final_hypothesis_id for row in endpoint.bindings
    } != {plan.downstream_hypothesis_id}:
        raise ValueError("endpoint report/final hypothesis population mismatch")

    freeze = RelationalScientificVerifierInputFreeze.model_validate_json(
        Path(str(plan.verifier_input_freeze_path)).read_text(encoding="utf-8")
    )
    if freeze.binding_plan_id != binding_plan.plan_id:
        raise ValueError("verifier freeze/binding plan mismatch")
    if freeze.endpoint_binding_report_id != endpoint.report_id:
        raise ValueError("verifier freeze/endpoint report mismatch")
    if freeze.final_hypothesis_id != plan.downstream_hypothesis_id:
        raise ValueError("verifier freeze/final hypothesis mismatch")
    if freeze.candidate_hypothesis_id != plan.downstream_hypothesis_id:
        raise ValueError("verifier freeze/candidate hypothesis mismatch")

    manifest = RelationalScientificVerifierRunManifest.model_validate_json(
        Path(str(plan.verifier_manifest_path)).read_text(encoding="utf-8")
    )
    if manifest.input_freeze_id != freeze.freeze_id:
        raise ValueError("verifier manifest/input-freeze ID mismatch")
    if manifest.input_freeze_sha256 != freeze.freeze_sha256:
        raise ValueError("verifier manifest/input-freeze SHA mismatch")
    if manifest.final_hypothesis_id != plan.downstream_hypothesis_id:
        raise ValueError("verifier manifest/final hypothesis mismatch")
    if manifest.candidate_hypothesis_id != plan.downstream_hypothesis_id:
        raise ValueError("verifier manifest/candidate hypothesis mismatch")
    if tuple(row.stage_name for row in manifest.stage_records) != (
        _VERIFIER_STAGE_NAMES
    ):
        raise ValueError("verifier manifest did not complete frozen 13-stage contract")
    if manifest.stage_count != len(_VERIFIER_STAGE_NAMES):
        raise ValueError("verifier manifest stage_count mismatch")

    certification = _load_json_object(str(plan.verifier_certification_path))
    if str(certification.get("report_id") or "") != manifest.certification_report_id:
        raise ValueError("verifier manifest/certification report ID mismatch")
    decisions = certification.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("V_post certification must contain one decision")
    decision = decisions[0]
    if not isinstance(decision, dict):
        raise ValueError("V_post certification decision must be an object")
    if decision.get("hypothesis_id") != plan.downstream_hypothesis_id:
        raise ValueError("V_post certification/final hypothesis mismatch")
    if str(decision.get("decision") or "") != manifest.certification_decision:
        raise ValueError("V_post manifest/certification decision mismatch")

    state_pairs = (
        ("bounded_closure_state", manifest.bounded_closure_state),
        (
            "bounded_external_distinctness_state",
            manifest.bounded_external_distinctness_state,
        ),
        (
            "positive_nonobviousness_authority_state",
            manifest.positive_nonobviousness_authority_state,
        ),
        ("fatal_blocker_state", manifest.fatal_blocker_state),
    )
    for key, expected in state_pairs:
        if str(decision.get(key) or "") != expected:
            raise ValueError("V_post manifest/certification state mismatch: " + key)

    return PreN10VPostLineageResultV1(
        lineage_id=plan.lineage_id,
        origin=plan.origin,
        source_hypothesis_id=plan.source_hypothesis_id,
        downstream_hypothesis_id=plan.downstream_hypothesis_id,
        n10_certification_status=plan.n10_certification_status,
        n10_selection_class=plan.n10_selection_class,
        binding_status=plan.binding_status,
        status="VPOST_COMPLETED",
        endpoint_report_path=str(plan.endpoint_report_path),
        endpoint_report_id=endpoint.report_id,
        endpoint_report_file_sha256=sha256_file(str(plan.endpoint_report_path)),
        endpoint_llm_calls=endpoint.llm_calls_performed,
        verifier_input_freeze_path=str(plan.verifier_input_freeze_path),
        verifier_input_freeze_id=freeze.freeze_id,
        verifier_manifest_path=str(plan.verifier_manifest_path),
        verifier_manifest_id=manifest.manifest_id,
        verifier_certification_path=str(plan.verifier_certification_path),
        verifier_certification_report_id=manifest.certification_report_id,
        certification_decision=manifest.certification_decision,
        bounded_closure_state=manifest.bounded_closure_state,
        bounded_external_distinctness_state=(
            manifest.bounded_external_distinctness_state
        ),
        positive_nonobviousness_authority_state=(
            manifest.positive_nonobviousness_authority_state
        ),
        fatal_blocker_state=manifest.fatal_blocker_state,
        endpoint_binding_stage_invocations=1,
        relational_verifier_stage_invocations=1,
    )


def build_pre_n10_vpost_shadow_report_v1(
    *,
    execution_plan: PreN10VPostShadowPlanV1,
    bridge: PreN10RelationalBindingBridgeReportV1,
    lineages: list[PreN10VPostLineageResultV1],
) -> PreN10VPostShadowReportV1:
    if execution_plan.source_binding_bridge_report_id != bridge.report_id:
        raise ValueError("V_post execution plan/binding bridge ID mismatch")
    if execution_plan.source_binding_bridge_report_sha256 != bridge.report_sha256:
        raise ValueError("V_post execution plan/binding bridge SHA mismatch")
    expected_ids = [row.lineage_id for row in execution_plan.lineages]
    if [row.lineage_id for row in lineages] != expected_ids:
        raise ValueError("V_post result population/order differs from frozen plan")
    if expected_ids != [row.lineage_id for row in bridge.lineages]:
        raise ValueError("V_post plan population differs from binding bridge")

    decisions = Counter(
        row.certification_decision
        for row in lineages
        if row.certification_decision is not None
    )
    body = {
        "schema_version": "pre-n10-vpost-shadow-report-v1",
        "source_execution_plan_id": execution_plan.plan_id,
        "source_execution_plan_sha256": execution_plan.plan_sha256,
        "source_binding_bridge_report_id": bridge.report_id,
        "source_binding_bridge_report_sha256": bridge.report_sha256,
        "lineages": [row.model_dump(mode="json") for row in lineages],
        "lineage_count": len(lineages),
        "completed_count": sum(
            row.status == "VPOST_COMPLETED" for row in lineages
        ),
        "not_binding_ready_count": sum(
            row.status == "NOT_BINDING_READY" for row in lineages
        ),
        "certification_decision_counts": dict(sorted(decisions.items())),
        "endpoint_binding_stage_invocations": sum(
            row.endpoint_binding_stage_invocations for row in lineages
        ),
        "relational_verifier_stage_invocations": sum(
            row.relational_verifier_stage_invocations for row in lineages
        ),
        "n10_status_did_not_filter_vpost_reachability": True,
        "literal_binding_required_before_verifier": True,
        "verifier_internal_stage_contract_frozen": True,
        "verifier_internal_stage_count": 13,
        "external_novelty_reassessed": False,
        "verifier_result_consumed_by_production": False,
        "production_authority_created": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreN10VPostShadowReportV1(
        **body,
        report_id="pre_n10_vpost_shadow_report_v1:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "PreN10VPostLineagePlanV1",
    "PreN10VPostLineageResultV1",
    "PreN10VPostShadowPlanV1",
    "PreN10VPostShadowReportV1",
    "PreN10VPostStagePlanV1",
    "VPostLineageStatusV1",
    "build_pre_n10_vpost_lineage_result_v1",
    "build_pre_n10_vpost_shadow_report_v1",
    "compile_pre_n10_vpost_shadow_plan_v1",
    "pretty_json_bytes",
    "sha256_file",
    "write_exact_or_validate",
]
