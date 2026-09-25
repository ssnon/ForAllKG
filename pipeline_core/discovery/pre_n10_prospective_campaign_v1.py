from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_semantic_disposition import (
    HypothesisSemanticDispositionV1,
)
from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    PreN10DownstreamHandoffReportV1,
)
from pipeline_core.discovery.pre_n10_external_n10_shadow_v1 import (
    PreN10ExternalN10ShadowReportV1,
)
from pipeline_core.discovery.pre_n10_initial_semantic_gate_v1 import (
    PreN10InitialSemanticGateReportV1,
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    PreN10PrimaryRouterReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    PreN10RegenerationReentryReportV2,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    PreN10RegenerationExecutionReportV1,
)
from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    PreN10RelationalBindingBridgeReportV1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
)
from pipeline_core.discovery.pre_n10_vpost_shadow_v1 import (
    PreN10VPostShadowReportV1,
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
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_exact_or_validate(path: Path, payload: object) -> str:
    resolved = path.expanduser().resolve()
    expected = _pretty_json_bytes(payload)
    if resolved.exists():
        observed = resolved.read_bytes()
        if observed != expected:
            raise ValueError(
                "existing write-once prospective campaign artifact differs: "
                + str(resolved)
            )
        return hashlib.sha256(observed).hexdigest()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


class CampaignArtifactFingerprintV1(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def fingerprint(path: str | Path) -> CampaignArtifactFingerprintV1:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("missing campaign artifact: " + str(resolved))
    return CampaignArtifactFingerprintV1(
        path=str(resolved),
        sha256=sha256_file(resolved),
    )


CAMPAIGN_STAGE_ORDER = (
    "initial_semantic_gate",
    "initial_vpre",
    "primary_router",
    "one_shot_regeneration",
    "regeneration_semantic_reentry",
    "downstream_handoff",
    "external_n9_n10",
    "relational_binding_bridge",
    "vpost_shadow",
)


class PreN10ProspectiveCampaignPlanV1(StrictModel):
    schema_version: Literal[
        "pre-n10-prospective-campaign-plan-v1"
    ] = "pre-n10-prospective-campaign-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    portfolio: CampaignArtifactFingerprintV1
    semantic_run: CampaignArtifactFingerprintV1
    semantic_review: CampaignArtifactFingerprintV1 | None = None
    hypothesis_context: CampaignArtifactFingerprintV1
    regeneration_unit_freeze: CampaignArtifactFingerprintV1
    provider_plan: CampaignArtifactFingerprintV1
    output_root: str

    decomposition_model: str
    primary_model: str
    specification_repair_model: str
    specification_audit_model: str
    source_alignment_model: str
    regeneration_model: str
    semantic_critic_model: str
    external_n10_model: str
    vpost_model: str

    api_key_env: str
    base_url: str | None = None
    parse_retries: int = Field(ge=0)
    timeout_seconds: float = Field(gt=0)
    max_claims: int = Field(ge=1)
    max_queries_per_claim: int = Field(ge=1)
    save_prompts: bool = False
    allow_dirty_worktree_for_vpost: bool = False

    stage_order: list[str]
    resume_scope: Literal[
        "BETWEEN_COMPLETED_WRITE_ONCE_STAGES_ONLY"
    ] = "BETWEEN_COMPLETED_WRITE_ONCE_STAGES_ONLY"
    partial_stage_resume_allowed: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False
    result_conditioned_route_changes_allowed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "PreN10ProspectiveCampaignPlanV1":
        if self.stage_order != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError("prospective campaign stage order mismatch")
        for name in (
            self.decomposition_model,
            self.primary_model,
            self.specification_repair_model,
            self.specification_audit_model,
            self.source_alignment_model,
            self.regeneration_model,
            self.semantic_critic_model,
            self.external_n10_model,
            self.vpost_model,
            self.api_key_env,
        ):
            if not str(name).strip():
                raise ValueError("campaign model/API-key-env names must be non-empty")
        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective campaign plan SHA mismatch")
        if observed_id != "pre_n10_prospective_campaign_plan_v1:" + expected_sha[:20]:
            raise ValueError("prospective campaign plan ID mismatch")
        return self


CampaignStageStatusV1 = Literal[
    "EXECUTED",
    "REUSED_VALIDATED",
    "SKIPPED_NOT_REQUIRED",
    "SKIPPED_TERMINAL",
]


class PreN10ProspectiveCampaignStageRecordV1(StrictModel):
    stage_index: int = Field(ge=1)
    stage_name: str
    status: CampaignStageStatusV1
    argv: list[str] = Field(default_factory=list)
    output_artifacts: list[CampaignArtifactFingerprintV1] = Field(
        default_factory=list
    )
    authority_artifact_id: str | None = None
    authority_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reason: str | None = None

    @model_validator(mode="after")
    def validate_record(self) -> "PreN10ProspectiveCampaignStageRecordV1":
        active = self.status in {"EXECUTED", "REUSED_VALIDATED"}
        if active:
            if not self.argv:
                raise ValueError("active campaign stage requires argv")
            if not self.output_artifacts:
                raise ValueError("active campaign stage requires output artifacts")
            if self.authority_artifact_id is None:
                raise ValueError("active campaign stage requires authority artifact ID")
            if self.authority_artifact_sha256 is None:
                raise ValueError("active campaign stage requires authority artifact SHA")
            if self.reason is not None:
                raise ValueError("active campaign stage cannot carry skip reason")
        else:
            if self.output_artifacts:
                raise ValueError("skipped campaign stage cannot carry output artifacts")
            if self.authority_artifact_id is not None:
                raise ValueError("skipped campaign stage cannot carry authority ID")
            if self.authority_artifact_sha256 is not None:
                raise ValueError("skipped campaign stage cannot carry authority SHA")
            if not (self.reason or "").strip():
                raise ValueError("skipped campaign stage requires reason")
        return self


CampaignFinalStatusV1 = Literal[
    "INITIAL_SEMANTIC_TERMINAL",
    "PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE",
    "VPOST_SHADOW_COMPLETE",
]


class PreN10ProspectiveCampaignReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-prospective-campaign-report-v1"
    ] = "pre-n10-prospective-campaign-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_plan_id: str
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    final_status: CampaignFinalStatusV1
    stages: list[PreN10ProspectiveCampaignStageRecordV1]
    stage_count: int = Field(ge=0)

    initial_semantic_status: str
    primary_regeneration_fallback_count: int = Field(ge=0)
    handoff_external_eligible_count: int = Field(ge=0)
    n10_certified_count: int = Field(ge=0)
    n10_unresolved_count: int = Field(ge=0)
    n10_rejected_count: int = Field(ge=0)
    binding_ready_lineage_count: int = Field(ge=0)
    vpost_completed_count: int = Field(ge=0)
    scientific_certification_decision_counts: dict[str, int]

    initial_semantic_llm_reinvoked: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    result_conditioned_route_changes_performed: Literal[False] = False
    external_novelty_reassessed_in_vpost: Literal[False] = False
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10ProspectiveCampaignReportV1":
        if self.stage_count != len(self.stages):
            raise ValueError("campaign stage_count mismatch")
        if self.stage_count != len(CAMPAIGN_STAGE_ORDER):
            raise ValueError("campaign report must account for every stage")
        if [row.stage_index for row in self.stages] != list(
            range(1, len(CAMPAIGN_STAGE_ORDER) + 1)
        ):
            raise ValueError("campaign stage indices must be contiguous")
        if [row.stage_name for row in self.stages] != list(CAMPAIGN_STAGE_ORDER):
            raise ValueError("campaign report stage order mismatch")

        statuses = {row.stage_name: row.status for row in self.stages}
        if self.final_status == "INITIAL_SEMANTIC_TERMINAL":
            if statuses["initial_semantic_gate"] not in {
                "EXECUTED",
                "REUSED_VALIDATED",
            }:
                raise ValueError("semantic-terminal campaign lacks semantic gate")
            if any(
                statuses[name] != "SKIPPED_TERMINAL"
                for name in CAMPAIGN_STAGE_ORDER[1:]
            ):
                raise ValueError("semantic-terminal campaign executed downstream")
        elif self.final_status == "PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE":
            for name in (
                "initial_semantic_gate",
                "initial_vpre",
                "primary_router",
                "downstream_handoff",
            ):
                if statuses[name] not in {"EXECUTED", "REUSED_VALIDATED"}:
                    raise ValueError("pre-N10 terminal campaign lacks required stage")
            if any(
                statuses[name] != "SKIPPED_TERMINAL"
                for name in (
                    "external_n9_n10",
                    "relational_binding_bridge",
                    "vpost_shadow",
                )
            ):
                raise ValueError("pre-N10 terminal campaign executed external/V_post")
        else:
            for name in (
                "initial_semantic_gate",
                "initial_vpre",
                "primary_router",
                "downstream_handoff",
                "external_n9_n10",
                "relational_binding_bridge",
                "vpost_shadow",
            ):
                if statuses[name] not in {"EXECUTED", "REUSED_VALIDATED"}:
                    raise ValueError("completed campaign lacks required stage")

        regen = statuses["one_shot_regeneration"]
        reentry = statuses["regeneration_semantic_reentry"]
        if (regen in {"EXECUTED", "REUSED_VALIDATED"}) != (
            reentry in {"EXECUTED", "REUSED_VALIDATED"}
        ):
            raise ValueError("regeneration/re-entry stage activity mismatch")
        if (regen == "SKIPPED_NOT_REQUIRED") != (
            reentry == "SKIPPED_NOT_REQUIRED"
        ):
            raise ValueError("regeneration/re-entry skip mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective campaign report SHA mismatch")
        if observed_id != (
            "pre_n10_prospective_campaign_report_v1:" + expected_sha[:20]
        ):
            raise ValueError("prospective campaign report ID mismatch")
        return self


def build_pre_n10_prospective_campaign_plan_v1(
    *,
    portfolio_path: Path,
    semantic_run_path: Path,
    semantic_review_path: Path | None,
    hypothesis_context_path: Path,
    regeneration_unit_freeze_path: Path,
    provider_plan_path: Path,
    output_root: Path,
    decomposition_model: str,
    primary_model: str,
    specification_repair_model: str,
    specification_audit_model: str,
    source_alignment_model: str,
    regeneration_model: str,
    semantic_critic_model: str,
    external_n10_model: str,
    vpost_model: str,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
    parse_retries: int = 1,
    timeout_seconds: float = 180.0,
    max_claims: int = 4,
    max_queries_per_claim: int = 2,
    save_prompts: bool = False,
    allow_dirty_worktree_for_vpost: bool = False,
) -> PreN10ProspectiveCampaignPlanV1:
    portfolio_file = portfolio_path.expanduser().resolve()
    context_file = hypothesis_context_path.expanduser().resolve()
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_file.read_text(encoding="utf-8")
    )
    context = HypothesisContext.model_validate_json(
        context_file.read_text(encoding="utf-8")
    )
    if portfolio.source_context_id != context.context_id:
        raise ValueError("campaign portfolio/context ID mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("campaign portfolio/context SHA mismatch")
    if portfolio.domain_profile_id != context.domain_profile_id:
        raise ValueError("campaign portfolio/context domain mismatch")

    freeze_file = regeneration_unit_freeze_path.expanduser().resolve()
    ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        freeze_file.read_text(encoding="utf-8")
    )

    body = {
        "schema_version": "pre-n10-prospective-campaign-plan-v1",
        "portfolio": fingerprint(portfolio_file).model_dump(mode="json"),
        "semantic_run": fingerprint(semantic_run_path).model_dump(mode="json"),
        "semantic_review": (
            fingerprint(semantic_review_path).model_dump(mode="json")
            if semantic_review_path is not None
            else None
        ),
        "hypothesis_context": fingerprint(context_file).model_dump(mode="json"),
        "regeneration_unit_freeze": fingerprint(freeze_file).model_dump(mode="json"),
        "provider_plan": fingerprint(provider_plan_path).model_dump(mode="json"),
        "output_root": str(output_root.expanduser().resolve()),
        "decomposition_model": str(decomposition_model),
        "primary_model": str(primary_model),
        "specification_repair_model": str(specification_repair_model),
        "specification_audit_model": str(specification_audit_model),
        "source_alignment_model": str(source_alignment_model),
        "regeneration_model": str(regeneration_model),
        "semantic_critic_model": str(semantic_critic_model),
        "external_n10_model": str(external_n10_model),
        "vpost_model": str(vpost_model),
        "api_key_env": str(api_key_env),
        "base_url": (
            str(base_url).strip() if base_url is not None and str(base_url).strip()
            else None
        ),
        "parse_retries": int(parse_retries),
        "timeout_seconds": float(timeout_seconds),
        "max_claims": int(max_claims),
        "max_queries_per_claim": int(max_queries_per_claim),
        "save_prompts": bool(save_prompts),
        "allow_dirty_worktree_for_vpost": bool(allow_dirty_worktree_for_vpost),
        "stage_order": list(CAMPAIGN_STAGE_ORDER),
        "resume_scope": "BETWEEN_COMPLETED_WRITE_ONCE_STAGES_ONLY",
        "partial_stage_resume_allowed": False,
        "second_regeneration_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "production_selection_authority": False,
    }
    digest = _sha256_json(body)
    return PreN10ProspectiveCampaignPlanV1(
        **body,
        plan_id="pre_n10_prospective_campaign_plan_v1:" + digest[:20],
        plan_sha256=digest,
    )


CommandRunner = Callable[[str, list[str]], None]


def _subprocess_runner(stage_name: str, argv: list[str]) -> None:
    del stage_name
    subprocess.run([sys.executable, *argv], check=True)


def _verify_fingerprint(row: CampaignArtifactFingerprintV1) -> None:
    path = Path(row.path)
    if not path.is_file():
        raise ValueError("completed campaign artifact is missing: " + row.path)
    if sha256_file(path) != row.sha256:
        raise ValueError("completed campaign artifact changed: " + row.path)


def _stage_has_any_files(root: Path) -> bool:
    return root.exists() and any(path.is_file() for path in root.rglob("*"))


def _run_or_reuse_stage(
    *,
    stage_index: int,
    stage_name: str,
    stage_root: Path,
    argv: list[str],
    expected_outputs: list[Path],
    validate: Callable[[], tuple[str, str]],
    runner: CommandRunner,
) -> PreN10ProspectiveCampaignStageRecordV1:
    existing = [path.is_file() for path in expected_outputs]
    if all(existing):
        authority_id, authority_sha = validate()
        return PreN10ProspectiveCampaignStageRecordV1(
            stage_index=stage_index,
            stage_name=stage_name,
            status="REUSED_VALIDATED",
            argv=argv,
            output_artifacts=[fingerprint(path) for path in expected_outputs],
            authority_artifact_id=authority_id,
            authority_artifact_sha256=authority_sha,
        )
    if any(existing) or _stage_has_any_files(stage_root):
        raise ValueError(
            "partial write-once campaign stage cannot be resumed in place: "
            + stage_name
        )

    runner(stage_name, argv)
    missing = [str(path) for path in expected_outputs if not path.is_file()]
    if missing:
        raise ValueError(
            stage_name + " completed without expected outputs: " + repr(missing)
        )
    authority_id, authority_sha = validate()
    return PreN10ProspectiveCampaignStageRecordV1(
        stage_index=stage_index,
        stage_name=stage_name,
        status="EXECUTED",
        argv=argv,
        output_artifacts=[fingerprint(path) for path in expected_outputs],
        authority_artifact_id=authority_id,
        authority_artifact_sha256=authority_sha,
    )


def _skip_stage(
    *,
    stage_index: int,
    stage_name: str,
    status: Literal["SKIPPED_NOT_REQUIRED", "SKIPPED_TERMINAL"],
    reason: str,
) -> PreN10ProspectiveCampaignStageRecordV1:
    return PreN10ProspectiveCampaignStageRecordV1(
        stage_index=stage_index,
        stage_name=stage_name,
        status=status,
        reason=reason,
    )


def _base_model_args(plan: PreN10ProspectiveCampaignPlanV1) -> list[str]:
    result = [
        "--api-key-env",
        plan.api_key_env,
        "--parse-retries",
        str(plan.parse_retries),
    ]
    if plan.base_url:
        result += ["--base-url", plan.base_url]
    return result


def _report(
    *,
    plan: PreN10ProspectiveCampaignPlanV1,
    final_status: CampaignFinalStatusV1,
    stages: list[PreN10ProspectiveCampaignStageRecordV1],
    semantic_gate: PreN10InitialSemanticGateReportV1,
    primary: PreN10PrimaryRouterReportV1 | None = None,
    handoff: PreN10DownstreamHandoffReportV1 | None = None,
    external: PreN10ExternalN10ShadowReportV1 | None = None,
    bridge: PreN10RelationalBindingBridgeReportV1 | None = None,
    vpost: PreN10VPostShadowReportV1 | None = None,
) -> PreN10ProspectiveCampaignReportV1:
    body = {
        "schema_version": "pre-n10-prospective-campaign-report-v1",
        "source_plan_id": plan.plan_id,
        "source_plan_sha256": plan.plan_sha256,
        "final_status": final_status,
        "stages": [row.model_dump(mode="json") for row in stages],
        "stage_count": len(stages),
        "initial_semantic_status": semantic_gate.status,
        "primary_regeneration_fallback_count": (
            primary.regeneration_fallback_required_count if primary else 0
        ),
        "handoff_external_eligible_count": (
            handoff.eligible_for_external_novelty_count if handoff else 0
        ),
        "n10_certified_count": external.certified_count if external else 0,
        "n10_unresolved_count": external.unresolved_count if external else 0,
        "n10_rejected_count": external.rejected_count if external else 0,
        "binding_ready_lineage_count": (
            bridge.binding_ready_lineage_count if bridge else 0
        ),
        "vpost_completed_count": vpost.completed_count if vpost else 0,
        "scientific_certification_decision_counts": (
            dict(vpost.certification_decision_counts) if vpost else {}
        ),
        "initial_semantic_llm_reinvoked": False,
        "second_regeneration_performed": False,
        "result_conditioned_route_changes_performed": False,
        "external_novelty_reassessed_in_vpost": False,
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreN10ProspectiveCampaignReportV1(
        **body,
        report_id="pre_n10_prospective_campaign_report_v1:" + digest[:20],
        report_sha256=digest,
    )


def _validate_existing_campaign_report(
    *,
    path: Path,
    plan: PreN10ProspectiveCampaignPlanV1,
) -> PreN10ProspectiveCampaignReportV1:
    report = PreN10ProspectiveCampaignReportV1.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if report.source_plan_id != plan.plan_id:
        raise ValueError("campaign report/source plan ID mismatch")
    if report.source_plan_sha256 != plan.plan_sha256:
        raise ValueError("campaign report/source plan SHA mismatch")
    for stage in report.stages:
        for artifact in stage.output_artifacts:
            _verify_fingerprint(artifact)
    return report


def execute_pre_n10_prospective_campaign_v1(
    *,
    plan: PreN10ProspectiveCampaignPlanV1,
    runner: CommandRunner = _subprocess_runner,
) -> PreN10ProspectiveCampaignReportV1:
    root = Path(plan.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "campaign.plan.json"
    report_path = root / "campaign.report.json"
    write_exact_or_validate(plan_path, plan)

    for source in (
        plan.portfolio,
        plan.semantic_run,
        plan.hypothesis_context,
        plan.regeneration_unit_freeze,
        plan.provider_plan,
    ):
        _verify_fingerprint(source)
    if plan.semantic_review is not None:
        _verify_fingerprint(plan.semantic_review)

    if report_path.is_file():
        return _validate_existing_campaign_report(path=report_path, plan=plan)

    portfolio_path = Path(plan.portfolio.path)
    semantic_run_path = Path(plan.semantic_run.path)
    semantic_review_path = (
        Path(plan.semantic_review.path) if plan.semantic_review is not None else None
    )
    context_path = Path(plan.hypothesis_context.path)
    freeze_path = Path(plan.regeneration_unit_freeze.path)
    provider_path = Path(plan.provider_plan.path)

    stages: list[PreN10ProspectiveCampaignStageRecordV1] = []

    semantic_root = root / "00_initial_semantic_gate"
    semantic_report_path = semantic_root / "initial_semantic_gate.report.json"
    semantic_argv = [
        "-m",
        "scripts.discovery.run_pre_n10_initial_semantic_gate_v1",
        "--portfolio",
        str(portfolio_path),
        "--semantic-run",
        str(semantic_run_path),
        "--output-dir",
        str(semantic_root),
    ]
    if semantic_review_path is not None:
        semantic_argv += ["--semantic-review", str(semantic_review_path)]

    def validate_semantic() -> tuple[str, str]:
        report = PreN10InitialSemanticGateReportV1.model_validate_json(
            semantic_report_path.read_text(encoding="utf-8")
        )
        if Path(report.source_portfolio_path).resolve() != portfolio_path.resolve():
            raise ValueError("campaign semantic gate portfolio path mismatch")
        if report.source_portfolio_file_sha256 != plan.portfolio.sha256:
            raise ValueError("campaign semantic gate portfolio SHA mismatch")
        if report.source_semantic_run_file_sha256 != plan.semantic_run.sha256:
            raise ValueError("campaign semantic gate run SHA mismatch")
        if plan.semantic_review is not None:
            if report.source_semantic_review_file_sha256 != plan.semantic_review.sha256:
                raise ValueError("campaign semantic gate review SHA mismatch")
        if report.semantic_disposition_path is not None:
            disposition_path = Path(report.semantic_disposition_path)
            if not disposition_path.is_file():
                raise ValueError("campaign semantic disposition artifact is missing")
            disposition = HypothesisSemanticDispositionV1.model_validate_json(
                disposition_path.read_text(encoding="utf-8")
            )
            if disposition.disposition_id != report.semantic_disposition_id:
                raise ValueError("campaign semantic disposition ID mismatch")
            if disposition.disposition_sha256 != report.semantic_disposition_sha256:
                raise ValueError("campaign semantic disposition SHA mismatch")
        return report.report_id, report.report_sha256

    stages.append(
        _run_or_reuse_stage(
            stage_index=1,
            stage_name="initial_semantic_gate",
            stage_root=semantic_root,
            argv=semantic_argv,
            expected_outputs=[semantic_report_path],
            validate=validate_semantic,
            runner=runner,
        )
    )
    semantic_gate = PreN10InitialSemanticGateReportV1.model_validate_json(
        semantic_report_path.read_text(encoding="utf-8")
    )

    if not semantic_gate.pre_n10_entry_authorized:
        for index, name in enumerate(CAMPAIGN_STAGE_ORDER[1:], start=2):
            stages.append(
                _skip_stage(
                    stage_index=index,
                    stage_name=name,
                    status="SKIPPED_TERMINAL",
                    reason="initial semantic authority did not authorize pre-N10 entry",
                )
            )
        report = _report(
            plan=plan,
            final_status="INITIAL_SEMANTIC_TERMINAL",
            stages=stages,
            semantic_gate=semantic_gate,
        )
        write_exact_or_validate(report_path, report)
        return report

    vpre_root = root / "01_initial_vpre"
    query_path = vpre_root / "claims_queries.json"
    contract_path = vpre_root / "contract.report.json"
    vpre_argv = [
        "-m",
        "scripts.discovery.run_pre_n10_scientific_contract_v1",
        "--portfolio",
        str(portfolio_path),
        "--model",
        plan.decomposition_model,
        "--api-key-env",
        plan.api_key_env,
        "--max-claims",
        str(plan.max_claims),
        "--max-queries-per-claim",
        str(plan.max_queries_per_claim),
        "--parse-retries",
        str(plan.parse_retries),
        "--query-plan-output",
        str(query_path),
        "--contract-output",
        str(contract_path),
    ]
    if plan.base_url:
        vpre_argv += ["--base-url", plan.base_url]
    if plan.save_prompts:
        vpre_argv += [
            "--prompt-output",
            str(vpre_root / "claim_decomposition.prompts.json"),
            "--specification-audit-output",
            str(vpre_root / "claim_decomposition.sanitization_audit.json"),
        ]

    def validate_vpre() -> tuple[str, str]:
        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )
        query = LiteratureQueryPlan.model_validate_json(
            query_path.read_text(encoding="utf-8")
        )
        contract = PreN10ScientificContractReportV1.model_validate_json(
            contract_path.read_text(encoding="utf-8")
        )
        if query.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("campaign V_pre query-plan/portfolio mismatch")
        if contract.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("campaign V_pre contract/portfolio mismatch")
        if contract.source_portfolio_sha256 != plan.portfolio.sha256:
            raise ValueError("campaign V_pre portfolio SHA mismatch")
        if contract.source_query_plan_id != query.plan_id:
            raise ValueError("campaign V_pre contract/query-plan ID mismatch")
        if contract.source_query_plan_sha256 != sha256_file(query_path):
            raise ValueError("campaign V_pre contract/query-plan SHA mismatch")
        return contract.report_id, contract.report_sha256

    stages.append(
        _run_or_reuse_stage(
            stage_index=2,
            stage_name="initial_vpre",
            stage_root=vpre_root,
            argv=vpre_argv,
            expected_outputs=[query_path, contract_path],
            validate=validate_vpre,
            runner=runner,
        )
    )
    initial_contract = PreN10ScientificContractReportV1.model_validate_json(
        contract_path.read_text(encoding="utf-8")
    )

    primary_root = root / "02_primary_router"
    post_query_path = primary_root / "post_primary.claims_queries.json"
    post_contract_path = primary_root / "contract.after_primary_router.json"
    primary_report_path = primary_root / "primary_router.report.json"
    primary_argv = [
        "-m",
        "scripts.discovery.run_pre_n10_primary_router_v1",
        "--portfolio",
        str(portfolio_path),
        "--query-plan",
        str(query_path),
        "--contract-report",
        str(contract_path),
        "--output-dir",
        str(primary_root),
        "--model",
        plan.primary_model,
        "--specification-repair-model",
        plan.specification_repair_model,
        "--specification-audit-model",
        plan.specification_audit_model,
        "--source-alignment-model",
        plan.source_alignment_model,
        "--api-key-env",
        plan.api_key_env,
        "--parse-retries",
        str(plan.parse_retries),
        "--timeout-seconds",
        str(plan.timeout_seconds),
    ]
    if plan.base_url:
        primary_argv += ["--base-url", plan.base_url]

    def validate_primary() -> tuple[str, str]:
        report = PreN10PrimaryRouterReportV1.model_validate_json(
            primary_report_path.read_text(encoding="utf-8")
        )
        post_query = LiteratureQueryPlan.model_validate_json(
            post_query_path.read_text(encoding="utf-8")
        )
        post_contract = PreN10ScientificContractReportV1.model_validate_json(
            post_contract_path.read_text(encoding="utf-8")
        )
        if report.source_contract_report_id != initial_contract.report_id:
            raise ValueError("campaign primary/source contract ID mismatch")
        if report.source_contract_report_sha256 != initial_contract.report_sha256:
            raise ValueError("campaign primary/source contract SHA mismatch")
        if report.post_primary_query_plan_id != post_query.plan_id:
            raise ValueError("campaign primary/post query-plan ID mismatch")
        if report.post_primary_query_plan_sha256 != post_query.plan_sha256:
            raise ValueError("campaign primary/post query-plan SHA mismatch")
        if report.post_contract_report_id != post_contract.report_id:
            raise ValueError("campaign primary/post contract ID mismatch")
        if report.post_contract_report_sha256 != post_contract.report_sha256:
            raise ValueError("campaign primary/post contract SHA mismatch")
        return report.report_id, report.report_sha256

    stages.append(
        _run_or_reuse_stage(
            stage_index=3,
            stage_name="primary_router",
            stage_root=primary_root,
            argv=primary_argv,
            expected_outputs=[
                post_query_path,
                post_contract_path,
                primary_report_path,
            ],
            validate=validate_primary,
            runner=runner,
        )
    )
    primary = PreN10PrimaryRouterReportV1.model_validate_json(
        primary_report_path.read_text(encoding="utf-8")
    )

    regeneration: PreN10RegenerationExecutionReportV1 | None = None
    reentry: PreN10RegenerationReentryReportV2 | None = None
    regeneration_report_path = root / "03_one_shot_regeneration" / "regeneration.execution.json"
    reentry_report_path = root / "04_regeneration_reentry" / "reentry_v2.report.json"

    if primary.regeneration_fallback_required_count > 0:
        regeneration_root = root / "03_one_shot_regeneration"
        regeneration_argv = [
            "-m",
            "scripts.discovery.run_pre_n10_regeneration_from_router_v1",
            "--context",
            str(context_path),
            "--primary-router-report",
            str(primary_report_path),
            "--regeneration-unit-freeze",
            str(freeze_path),
            "--output-dir",
            str(regeneration_root),
            "--model",
            plan.regeneration_model,
            "--api-key-env",
            plan.api_key_env,
            "--parse-retries",
            str(plan.parse_retries),
            "--timeout-seconds",
            str(plan.timeout_seconds),
        ]
        if plan.base_url:
            regeneration_argv += ["--base-url", plan.base_url]

        def validate_regeneration() -> tuple[str, str]:
            report = PreN10RegenerationExecutionReportV1.model_validate_json(
                regeneration_report_path.read_text(encoding="utf-8")
            )
            if report.source_primary_report_id != primary.report_id:
                raise ValueError("campaign regeneration/primary ID mismatch")
            if report.source_primary_report_sha256 != primary.report_sha256:
                raise ValueError("campaign regeneration/primary SHA mismatch")
            if report.regeneration_required_count != (
                primary.regeneration_fallback_required_count
            ):
                raise ValueError("campaign regeneration fallback population mismatch")
            return report.report_id, report.report_sha256

        stages.append(
            _run_or_reuse_stage(
                stage_index=4,
                stage_name="one_shot_regeneration",
                stage_root=regeneration_root,
                argv=regeneration_argv,
                expected_outputs=[regeneration_report_path],
                validate=validate_regeneration,
                runner=runner,
            )
        )
        regeneration = PreN10RegenerationExecutionReportV1.model_validate_json(
            regeneration_report_path.read_text(encoding="utf-8")
        )

        reentry_root = root / "04_regeneration_reentry"
        reentry_argv = [
            "-m",
            "scripts.discovery.run_pre_n10_regeneration_reentry_v2",
            "--context",
            str(context_path),
            "--regeneration-report",
            str(regeneration_report_path),
            "--output-dir",
            str(reentry_root),
            "--critic-model",
            plan.semantic_critic_model,
            "--decomposition-model",
            plan.decomposition_model,
            "--api-key-env",
            plan.api_key_env,
            "--parse-retries",
            str(plan.parse_retries),
            "--timeout-seconds",
            str(plan.timeout_seconds),
            "--max-claims",
            str(plan.max_claims),
            "--max-queries-per-claim",
            str(plan.max_queries_per_claim),
        ]
        if plan.base_url:
            reentry_argv += ["--base-url", plan.base_url]

        def validate_reentry() -> tuple[str, str]:
            report = PreN10RegenerationReentryReportV2.model_validate_json(
                reentry_report_path.read_text(encoding="utf-8")
            )
            if report.source_regeneration_report_id != regeneration.report_id:
                raise ValueError("campaign re-entry/regeneration ID mismatch")
            if report.source_regeneration_report_sha256 != regeneration.report_sha256:
                raise ValueError("campaign re-entry/regeneration SHA mismatch")
            return report.report_id, report.report_sha256

        stages.append(
            _run_or_reuse_stage(
                stage_index=5,
                stage_name="regeneration_semantic_reentry",
                stage_root=reentry_root,
                argv=reentry_argv,
                expected_outputs=[reentry_report_path],
                validate=validate_reentry,
                runner=runner,
            )
        )
        reentry = PreN10RegenerationReentryReportV2.model_validate_json(
            reentry_report_path.read_text(encoding="utf-8")
        )
    else:
        stages.append(
            _skip_stage(
                stage_index=4,
                stage_name="one_shot_regeneration",
                status="SKIPPED_NOT_REQUIRED",
                reason="primary router produced no regeneration fallback lineage",
            )
        )
        stages.append(
            _skip_stage(
                stage_index=5,
                stage_name="regeneration_semantic_reentry",
                status="SKIPPED_NOT_REQUIRED",
                reason="no regeneration lineage exists for semantic re-entry",
            )
        )

    handoff_root = root / "05_downstream_handoff"
    handoff_path = handoff_root / "downstream_handoff.report.json"
    handoff_argv = [
        "-m",
        "scripts.discovery.build_pre_n10_downstream_handoff_v1",
        "--initial-semantic-gate",
        str(semantic_report_path),
        "--initial-portfolio",
        str(portfolio_path),
        "--post-primary-query-plan",
        str(post_query_path),
        "--primary-router-report",
        str(primary_report_path),
        "--output-dir",
        str(handoff_root),
    ]
    if regeneration is not None and reentry is not None:
        handoff_argv += [
            "--regeneration-report",
            str(regeneration_report_path),
            "--regeneration-reentry-report",
            str(reentry_report_path),
        ]

    def validate_handoff() -> tuple[str, str]:
        report = PreN10DownstreamHandoffReportV1.model_validate_json(
            handoff_path.read_text(encoding="utf-8")
        )
        if report.source_initial_semantic_gate_report_id != semantic_gate.report_id:
            raise ValueError("campaign handoff/semantic gate ID mismatch")
        if report.source_initial_semantic_gate_report_sha256 != semantic_gate.report_sha256:
            raise ValueError("campaign handoff/semantic gate SHA mismatch")
        if report.source_primary_router_report_id != primary.report_id:
            raise ValueError("campaign handoff/primary ID mismatch")
        if report.source_primary_router_report_sha256 != primary.report_sha256:
            raise ValueError("campaign handoff/primary SHA mismatch")
        if regeneration is None:
            if report.source_regeneration_report_id is not None:
                raise ValueError("campaign handoff unexpectedly references regeneration")
        else:
            if report.source_regeneration_report_id != regeneration.report_id:
                raise ValueError("campaign handoff/regeneration ID mismatch")
            if report.source_regeneration_reentry_report_id != reentry.report_id:
                raise ValueError("campaign handoff/re-entry ID mismatch")
        return report.report_id, report.report_sha256

    stages.append(
        _run_or_reuse_stage(
            stage_index=6,
            stage_name="downstream_handoff",
            stage_root=handoff_root,
            argv=handoff_argv,
            expected_outputs=[handoff_path],
            validate=validate_handoff,
            runner=runner,
        )
    )
    handoff = PreN10DownstreamHandoffReportV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )

    if handoff.eligible_for_external_novelty_count == 0:
        for index, name in enumerate(CAMPAIGN_STAGE_ORDER[6:], start=7):
            stages.append(
                _skip_stage(
                    stage_index=index,
                    stage_name=name,
                    status="SKIPPED_TERMINAL",
                    reason="no PRE_N10_READY lineage is eligible for external novelty",
                )
            )
        report = _report(
            plan=plan,
            final_status="PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE",
            stages=stages,
            semantic_gate=semantic_gate,
            primary=primary,
            handoff=handoff,
        )
        write_exact_or_validate(report_path, report)
        return report

    external_root = root / "06_external_n9_n10"
    external_report_path = external_root / "external_n9_n10.report.json"
    external_argv = [
        "-m",
        "scripts.discovery.run_pre_n10_external_n10_shadow_v1",
        "--handoff",
        str(handoff_path),
        "--hypothesis-context",
        str(context_path),
        "--provider-plan",
        str(provider_path),
        "--model",
        plan.external_n10_model,
        "--api-key-env",
        plan.api_key_env,
        "--output-root",
        str(external_root),
    ]
    if plan.base_url:
        external_argv += ["--base-url", plan.base_url]
    if plan.save_prompts:
        external_argv += ["--save-prompts"]

    def validate_external() -> tuple[str, str]:
        report = PreN10ExternalN10ShadowReportV1.model_validate_json(
            external_report_path.read_text(encoding="utf-8")
        )
        if report.source_handoff_report_id != handoff.report_id:
            raise ValueError("campaign external/handoff ID mismatch")
        if report.source_handoff_report_sha256 != handoff.report_sha256:
            raise ValueError("campaign external/handoff SHA mismatch")
        return report.report_id, report.report_sha256

    stages.append(
        _run_or_reuse_stage(
            stage_index=7,
            stage_name="external_n9_n10",
            stage_root=external_root,
            argv=external_argv,
            expected_outputs=[external_report_path],
            validate=validate_external,
            runner=runner,
        )
    )
    external = PreN10ExternalN10ShadowReportV1.model_validate_json(
        external_report_path.read_text(encoding="utf-8")
    )

    bridge_root = root / "07_relational_binding_bridge"
    bridge_path = bridge_root / "relational_binding_bridge.report.json"
    bridge_argv = [
        "-m",
        "scripts.discovery.build_pre_n10_relational_binding_bridge_v1",
        "--handoff",
        str(handoff_path),
        "--external-n10-report",
        str(external_report_path),
        "--output-root",
        str(bridge_root),
    ]

    def validate_bridge() -> tuple[str, str]:
        report = PreN10RelationalBindingBridgeReportV1.model_validate_json(
            bridge_path.read_text(encoding="utf-8")
        )
        if report.source_handoff_report_id != handoff.report_id:
            raise ValueError("campaign binding bridge/handoff ID mismatch")
        if report.source_external_n10_report_id != external.report_id:
            raise ValueError("campaign binding bridge/external ID mismatch")
        if report.source_external_n10_report_sha256 != external.report_sha256:
            raise ValueError("campaign binding bridge/external SHA mismatch")
        return report.report_id, report.report_sha256

    stages.append(
        _run_or_reuse_stage(
            stage_index=8,
            stage_name="relational_binding_bridge",
            stage_root=bridge_root,
            argv=bridge_argv,
            expected_outputs=[bridge_path],
            validate=validate_bridge,
            runner=runner,
        )
    )
    bridge = PreN10RelationalBindingBridgeReportV1.model_validate_json(
        bridge_path.read_text(encoding="utf-8")
    )

    vpost_root = root / "08_vpost_shadow"
    vpost_path = vpost_root / "vpost_shadow.report.json"
    vpost_argv = [
        "-m",
        "scripts.discovery.run_pre_n10_vpost_shadow_v1",
        "--binding-bridge",
        str(bridge_path),
        "--provider-plan",
        str(provider_path),
        "--model",
        plan.vpost_model,
        "--api-key-env",
        plan.api_key_env,
        "--output-root",
        str(vpost_root),
    ]
    if plan.base_url:
        vpost_argv += ["--base-url", plan.base_url]
    if plan.save_prompts:
        vpost_argv += ["--save-prompts"]
    if plan.allow_dirty_worktree_for_vpost:
        vpost_argv += ["--allow-dirty-worktree"]

    def validate_vpost() -> tuple[str, str]:
        report = PreN10VPostShadowReportV1.model_validate_json(
            vpost_path.read_text(encoding="utf-8")
        )
        if report.source_binding_bridge_report_id != bridge.report_id:
            raise ValueError("campaign V_post/binding bridge ID mismatch")
        if report.source_binding_bridge_report_sha256 != bridge.report_sha256:
            raise ValueError("campaign V_post/binding bridge SHA mismatch")
        return report.report_id, report.report_sha256

    stages.append(
        _run_or_reuse_stage(
            stage_index=9,
            stage_name="vpost_shadow",
            stage_root=vpost_root,
            argv=vpost_argv,
            expected_outputs=[vpost_path],
            validate=validate_vpost,
            runner=runner,
        )
    )
    vpost = PreN10VPostShadowReportV1.model_validate_json(
        vpost_path.read_text(encoding="utf-8")
    )

    report = _report(
        plan=plan,
        final_status="VPOST_SHADOW_COMPLETE",
        stages=stages,
        semantic_gate=semantic_gate,
        primary=primary,
        handoff=handoff,
        external=external,
        bridge=bridge,
        vpost=vpost,
    )
    write_exact_or_validate(report_path, report)
    return report


__all__ = [
    "CAMPAIGN_STAGE_ORDER",
    "CampaignArtifactFingerprintV1",
    "PreN10ProspectiveCampaignPlanV1",
    "PreN10ProspectiveCampaignReportV1",
    "PreN10ProspectiveCampaignStageRecordV1",
    "build_pre_n10_prospective_campaign_plan_v1",
    "execute_pre_n10_prospective_campaign_v1",
    "fingerprint",
    "sha256_file",
    "write_exact_or_validate",
]
