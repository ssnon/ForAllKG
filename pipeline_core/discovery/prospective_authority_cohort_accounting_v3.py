from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    PreN10ProspectiveCampaignPlanV1,
    PreN10ProspectiveCampaignReportV1,
    sha256_file,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionPlanV3,
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


def _load_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def _argv_value(argv: list[str], flag: str) -> str:
    matches = [index for index, value in enumerate(argv) if value == flag]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one frozen argv flag "
            + repr(flag)
            + ", found "
            + str(len(matches))
        )
    index = matches[0]
    if index + 1 >= len(argv):
        raise ValueError("frozen argv flag has no value: " + flag)
    return argv[index + 1]


def _verify_fingerprint(*, path: str, digest: str, expected_path: Path) -> None:
    actual = Path(path).expanduser().resolve()
    expected = expected_path.expanduser().resolve()
    if actual != expected:
        raise ValueError(
            "artifact path drift: expected "
            + str(expected)
            + ", observed "
            + str(actual)
        )
    if not actual.is_file():
        raise ValueError("missing artifact: " + str(actual))
    if sha256_file(actual) != digest:
        raise ValueError("artifact SHA mismatch: " + str(actual))


def _verify_cutpoint_artifact(
    *,
    cutpoint: dict,
    key: str,
    expected_path: Path,
) -> None:
    artifacts = cutpoint.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("initial-semantic cutpoint lacks artifacts")
    row = artifacts.get(key)
    if not isinstance(row, dict):
        raise ValueError("missing initial-semantic artifact: " + key)
    _verify_fingerprint(
        path=str(row.get("path", "")),
        digest=str(row.get("file_sha256", "")),
        expected_path=expected_path,
    )


def _stage_payload(
    *,
    report: PreN10ProspectiveCampaignReportV1,
    stage_name: str,
    schema_version: str,
) -> dict | None:
    stages = [row for row in report.stages if row.stage_name == stage_name]
    if len(stages) != 1:
        raise ValueError("campaign report stage cardinality mismatch: " + stage_name)
    stage = stages[0]
    if stage.status not in {"EXECUTED", "REUSED_VALIDATED"}:
        return None

    matches: list[dict] = []
    for artifact in stage.output_artifacts:
        path = Path(artifact.path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("missing campaign stage artifact: " + str(path))
        if sha256_file(path) != artifact.sha256:
            raise ValueError("campaign stage artifact SHA mismatch: " + str(path))
        try:
            payload = _load_json_object(path)
        except (ValueError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == schema_version:
            matches.append(payload)

    if len(matches) != 1:
        raise ValueError(
            stage_name
            + " must expose exactly one "
            + schema_version
            + " artifact; found "
            + str(len(matches))
        )
    return matches[0]


CohortFinalStatusV3 = Literal[
    "PRE_SEMANTIC_ALPHA4_EMPTY_TERMINAL",
    "INITIAL_SEMANTIC_TERMINAL",
    "PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE",
    "VPOST_SHADOW_COMPLETE",
]


class ProspectiveAuthorityCaseOutcomeV3(StrictModel):
    case_id: Literal["P21", "P22", "P23", "P24", "P25"]
    source_task_id: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    initial_manifest_path: str
    initial_manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_manifest_status: str
    initial_portfolio_path: str
    initial_portfolio_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_hypothesis_count: int = Field(ge=0)

    semantic_reached: bool
    initial_semantic_run_file_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    initial_semantic_review_present: bool
    initial_semantic_review_file_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    semantic_entry_authorized: bool

    campaign_plan_path: str | None = None
    campaign_plan_file_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    campaign_plan_id: str | None = None
    campaign_report_path: str | None = None
    campaign_report_file_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    campaign_report_id: str | None = None

    vpre_executed: bool
    initial_vpre_hypothesis_count: int | None = Field(default=None, ge=0)
    initial_vpre_ready_hypothesis_count: int | None = Field(default=None, ge=0)
    initial_vpre_claim_count: int | None = Field(default=None, ge=0)
    initial_vpre_ready_claim_count: int | None = Field(default=None, ge=0)

    regeneration_fallback_count: int = Field(ge=0)
    regenerated_lineage_count: int | None = Field(default=None, ge=0)
    regenerated_semantic_admissible_count: int | None = Field(default=None, ge=0)
    regenerated_ready_for_n10_count: int | None = Field(default=None, ge=0)

    external_stage_reached: bool
    external_eligible_lineage_count: int = Field(ge=0)
    n10_certified_count: int = Field(ge=0)
    n10_unresolved_count: int = Field(ge=0)
    n10_rejected_count: int = Field(ge=0)
    binding_ready_lineage_count: int = Field(ge=0)
    vpost_completed_count: int = Field(ge=0)
    scientific_certification_decision_counts: dict[str, int]

    final_status: CohortFinalStatusV3

    second_regeneration_performed: Literal[False] = False
    result_conditioned_route_changes_performed: Literal[False] = False
    external_novelty_reassessed_in_vpost: Literal[False] = False
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(self) -> "ProspectiveAuthorityCaseOutcomeV3":
        if self.final_status == "PRE_SEMANTIC_ALPHA4_EMPTY_TERMINAL":
            if self.initial_hypothesis_count != 0:
                raise ValueError("pre-semantic empty terminal requires zero hypotheses")
            if self.semantic_reached or self.semantic_entry_authorized:
                raise ValueError("pre-semantic empty terminal cannot reach semantic authority")
            if self.vpre_executed:
                raise ValueError("pre-semantic empty terminal cannot execute V_pre")
            if self.campaign_report_id is not None:
                raise ValueError("pre-semantic empty terminal cannot have downstream campaign")
        else:
            if not self.semantic_reached:
                raise ValueError("downstream cohort outcome requires semantic runtime")
            if self.campaign_report_id is None or self.campaign_plan_id is None:
                raise ValueError("semantic-reached outcome requires campaign provenance")

        if self.vpre_executed:
            required = (
                self.initial_vpre_hypothesis_count,
                self.initial_vpre_ready_hypothesis_count,
                self.initial_vpre_claim_count,
                self.initial_vpre_ready_claim_count,
            )
            if any(value is None for value in required):
                raise ValueError("executed V_pre requires V_pre accounting")
        else:
            if any(
                value is not None
                for value in (
                    self.initial_vpre_hypothesis_count,
                    self.initial_vpre_ready_hypothesis_count,
                    self.initial_vpre_claim_count,
                    self.initial_vpre_ready_claim_count,
                )
            ):
                raise ValueError("non-executed V_pre cannot carry V_pre accounting")

        if self.regeneration_fallback_count > 0:
            if any(
                value is None
                for value in (
                    self.regenerated_lineage_count,
                    self.regenerated_semantic_admissible_count,
                    self.regenerated_ready_for_n10_count,
                )
            ):
                raise ValueError("regeneration fallback requires re-entry accounting")

        if not self.external_stage_reached:
            if any(
                value != 0
                for value in (
                    self.external_eligible_lineage_count,
                    self.n10_certified_count,
                    self.n10_unresolved_count,
                    self.n10_rejected_count,
                    self.binding_ready_lineage_count,
                    self.vpost_completed_count,
                )
            ):
                raise ValueError(
                    "non-reached external stage cannot carry downstream scientific counts"
                )
            if self.scientific_certification_decision_counts:
                raise ValueError(
                    "non-reached external stage cannot carry certification decisions"
                )
        return self


class ProspectiveAuthorityCohortAccountingV3(StrictModel):
    schema_version: Literal[
        "prospective-authority-cohort-accounting-v3"
    ] = "prospective-authority-cohort-accounting-v3"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_execution_plan_path: str
    source_execution_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_campaign_freeze_id: str
    source_regeneration_unit_freeze_id: str
    execution_plan_repository_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")

    cases: list[ProspectiveAuthorityCaseOutcomeV3]
    case_ids: list[str]
    case_count: Literal[5] = 5

    initial_hypothesis_total: int = Field(ge=0)
    alpha4_nonempty_case_count: int = Field(ge=0)
    pre_semantic_empty_case_count: int = Field(ge=0)
    semantic_reached_case_count: int = Field(ge=0)
    semantic_entry_authorized_case_count: int = Field(ge=0)
    vpre_executed_case_count: int = Field(ge=0)

    initial_vpre_hypothesis_evaluated_total: int = Field(ge=0)
    initial_vpre_ready_hypothesis_total: int = Field(ge=0)
    initial_vpre_claim_evaluated_total: int = Field(ge=0)
    initial_vpre_ready_claim_total: int = Field(ge=0)

    regeneration_fallback_lineage_count: int = Field(ge=0)
    regenerated_lineage_count: int = Field(ge=0)
    regenerated_semantic_admissible_count: int = Field(ge=0)
    regenerated_ready_for_n10_count: int = Field(ge=0)

    external_stage_reached_case_count: int = Field(ge=0)
    external_eligible_lineage_count: int = Field(ge=0)
    n10_certified_count: int = Field(ge=0)
    n10_unresolved_count: int = Field(ge=0)
    n10_rejected_count: int = Field(ge=0)
    binding_ready_lineage_count: int = Field(ge=0)
    vpost_completed_count: int = Field(ge=0)
    scientific_certification_decision_counts: dict[str, int]

    final_status_counts: dict[str, int]

    outcome_accounting_only: Literal[True] = True
    prospective_results_preserved: Literal[True] = True
    source_tasks_or_models_modified_after_observation: Literal[False] = False
    later_case_adaptation_performed: Literal[False] = False
    failed_or_abstained_case_replacement_performed: Literal[False] = False
    prospective_case_rerun_performed: Literal[False] = False
    result_conditioned_route_changes_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    n10_absence_interpreted_as_positive_novelty: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "ProspectiveAuthorityCohortAccountingV3":
        expected = ["P21", "P22", "P23", "P24", "P25"]
        if self.case_ids != expected:
            raise ValueError("cohort accounting case IDs must be P21-P25")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("cohort accounting rows must be ordered P21-P25")

        counts = Counter(row.final_status for row in self.cases)
        if dict(sorted(counts.items())) != dict(
            sorted(self.final_status_counts.items())
        ):
            raise ValueError("final_status_counts mismatch")

        scalar_expectations = {
            "initial_hypothesis_total": sum(
                row.initial_hypothesis_count for row in self.cases
            ),
            "alpha4_nonempty_case_count": sum(
                row.initial_hypothesis_count > 0 for row in self.cases
            ),
            "pre_semantic_empty_case_count": sum(
                row.final_status == "PRE_SEMANTIC_ALPHA4_EMPTY_TERMINAL"
                for row in self.cases
            ),
            "semantic_reached_case_count": sum(
                row.semantic_reached for row in self.cases
            ),
            "semantic_entry_authorized_case_count": sum(
                row.semantic_entry_authorized for row in self.cases
            ),
            "vpre_executed_case_count": sum(
                row.vpre_executed for row in self.cases
            ),
            "initial_vpre_hypothesis_evaluated_total": sum(
                row.initial_vpre_hypothesis_count or 0 for row in self.cases
            ),
            "initial_vpre_ready_hypothesis_total": sum(
                row.initial_vpre_ready_hypothesis_count or 0 for row in self.cases
            ),
            "initial_vpre_claim_evaluated_total": sum(
                row.initial_vpre_claim_count or 0 for row in self.cases
            ),
            "initial_vpre_ready_claim_total": sum(
                row.initial_vpre_ready_claim_count or 0 for row in self.cases
            ),
            "regeneration_fallback_lineage_count": sum(
                row.regeneration_fallback_count for row in self.cases
            ),
            "regenerated_lineage_count": sum(
                row.regenerated_lineage_count or 0 for row in self.cases
            ),
            "regenerated_semantic_admissible_count": sum(
                row.regenerated_semantic_admissible_count or 0
                for row in self.cases
            ),
            "regenerated_ready_for_n10_count": sum(
                row.regenerated_ready_for_n10_count or 0 for row in self.cases
            ),
            "external_stage_reached_case_count": sum(
                row.external_stage_reached for row in self.cases
            ),
            "external_eligible_lineage_count": sum(
                row.external_eligible_lineage_count for row in self.cases
            ),
            "n10_certified_count": sum(
                row.n10_certified_count for row in self.cases
            ),
            "n10_unresolved_count": sum(
                row.n10_unresolved_count for row in self.cases
            ),
            "n10_rejected_count": sum(
                row.n10_rejected_count for row in self.cases
            ),
            "binding_ready_lineage_count": sum(
                row.binding_ready_lineage_count for row in self.cases
            ),
            "vpost_completed_count": sum(
                row.vpost_completed_count for row in self.cases
            ),
        }
        for name, expected_value in scalar_expectations.items():
            if getattr(self, name) != expected_value:
                raise ValueError(name + " mismatch")

        certification = Counter()
        for row in self.cases:
            certification.update(row.scientific_certification_decision_counts)
        if dict(sorted(certification.items())) != dict(
            sorted(self.scientific_certification_decision_counts.items())
        ):
            raise ValueError("scientific certification counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective authority cohort accounting SHA mismatch")
        if observed_id != (
            "prospective_authority_cohort_accounting_v3:"
            + expected_sha[:20]
        ):
            raise ValueError("prospective authority cohort accounting ID mismatch")
        return self


def _validate_campaign_plan_against_frozen_case(
    *,
    case,
    campaign_plan: PreN10ProspectiveCampaignPlanV1,
) -> None:
    checks = (
        (campaign_plan.portfolio, Path(case.initial_portfolio_path)),
        (campaign_plan.semantic_run, Path(case.initial_semantic_run_path)),
        (
            campaign_plan.hypothesis_context,
            Path(case.initial_hypothesis_context_path),
        ),
        (campaign_plan.provider_plan, Path(case.initial_provider_plan_path)),
    )
    for fingerprint, expected in checks:
        _verify_fingerprint(
            path=fingerprint.path,
            digest=fingerprint.sha256,
            expected_path=expected,
        )

    review = Path(case.initial_semantic_review_path).expanduser().resolve()
    if review.is_file():
        if campaign_plan.semantic_review is None:
            raise ValueError(case.case_id + ": campaign plan omitted semantic review")
        _verify_fingerprint(
            path=campaign_plan.semantic_review.path,
            digest=campaign_plan.semantic_review.sha256,
            expected_path=review,
        )
    elif campaign_plan.semantic_review is not None:
        raise ValueError(case.case_id + ": campaign plan carries absent semantic review")

    expected_root = Path(case.downstream_campaign_output_root).expanduser().resolve()
    if Path(campaign_plan.output_root).expanduser().resolve() != expected_root:
        raise ValueError(case.case_id + ": campaign output root drift")

    frozen_argv = case.downstream_campaign_argv_base
    expected_regeneration_freeze = Path(
        _argv_value(frozen_argv, "--regeneration-unit-freeze")
    )
    _verify_fingerprint(
        path=campaign_plan.regeneration_unit_freeze.path,
        digest=campaign_plan.regeneration_unit_freeze.sha256,
        expected_path=expected_regeneration_freeze,
    )

    str_fields = {
        "--decomposition-model": "decomposition_model",
        "--primary-model": "primary_model",
        "--specification-repair-model": "specification_repair_model",
        "--specification-audit-model": "specification_audit_model",
        "--source-alignment-model": "source_alignment_model",
        "--regeneration-model": "regeneration_model",
        "--semantic-critic-model": "semantic_critic_model",
        "--external-n10-model": "external_n10_model",
        "--vpost-model": "vpost_model",
        "--api-key-env": "api_key_env",
        "--base-url": "base_url",
    }
    for flag, field in str_fields.items():
        if str(getattr(campaign_plan, field)) != _argv_value(frozen_argv, flag):
            raise ValueError(case.case_id + ": campaign plan drift for " + flag)

    int_fields = {
        "--parse-retries": "parse_retries",
        "--max-claims": "max_claims",
        "--max-queries-per-claim": "max_queries_per_claim",
    }
    for flag, field in int_fields.items():
        if int(getattr(campaign_plan, field)) != int(
            _argv_value(frozen_argv, flag)
        ):
            raise ValueError(case.case_id + ": campaign plan drift for " + flag)

    if float(campaign_plan.timeout_seconds) != float(
        _argv_value(frozen_argv, "--timeout-seconds")
    ):
        raise ValueError(case.case_id + ": campaign timeout drift")
    if ("--save-prompts" in frozen_argv) != campaign_plan.save_prompts:
        raise ValueError(case.case_id + ": campaign save-prompts drift")
    if campaign_plan.allow_dirty_worktree_for_vpost:
        raise ValueError(case.case_id + ": dirty-worktree V_post not frozen")


def build_prospective_authority_cohort_accounting_v3(
    *,
    execution_plan_path: Path,
) -> ProspectiveAuthorityCohortAccountingV3:
    source_plan_path = execution_plan_path.expanduser().resolve()
    if not source_plan_path.is_file():
        raise ValueError("missing prospective authority execution plan")
    execution_plan = ProspectiveAuthorityExecutionPlanV3.model_validate_json(
        source_plan_path.read_text(encoding="utf-8")
    )

    rows: list[ProspectiveAuthorityCaseOutcomeV3] = []

    for case in execution_plan.cases:
        manifest_path = Path(case.initial_manifest_path).expanduser().resolve()
        portfolio_path = Path(case.initial_portfolio_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise ValueError(case.case_id + ": missing initial manifest")
        if not portfolio_path.is_file():
            raise ValueError(case.case_id + ": missing initial portfolio")

        manifest = _load_json_object(manifest_path)
        portfolio = _load_json_object(portfolio_path)
        hypotheses = portfolio.get("hypotheses")
        if not isinstance(hypotheses, list):
            raise ValueError(case.case_id + ": initial portfolio hypotheses missing")
        initial_count = len(hypotheses)
        manifest_status = str(manifest.get("status", ""))

        common = {
            "case_id": case.case_id,
            "source_task_id": case.source_task_id,
            "source_task_sha256": case.source_task_sha256,
            "initial_manifest_path": str(manifest_path),
            "initial_manifest_file_sha256": sha256_file(manifest_path),
            "initial_manifest_status": manifest_status,
            "initial_portfolio_path": str(portfolio_path),
            "initial_portfolio_file_sha256": sha256_file(portfolio_path),
            "initial_hypothesis_count": initial_count,
        }

        downstream_root = Path(
            case.downstream_campaign_output_root
        ).expanduser().resolve()
        campaign_plan_path = downstream_root / "campaign.plan.json"
        campaign_report_path = downstream_root / "campaign.report.json"

        if manifest_status == "complete_no_hypotheses_after_alpha4":
            if initial_count != 0:
                raise ValueError(
                    case.case_id + ": alpha4-empty status with nonempty portfolio"
                )
            if Path(case.initial_semantic_run_path).is_file():
                raise ValueError(
                    case.case_id + ": alpha4-empty terminal has semantic run"
                )
            if Path(case.initial_semantic_review_path).is_file():
                raise ValueError(
                    case.case_id + ": alpha4-empty terminal has semantic review"
                )
            if campaign_plan_path.is_file() or campaign_report_path.is_file():
                raise ValueError(
                    case.case_id
                    + ": alpha4-empty terminal unexpectedly has downstream campaign"
                )
            rows.append(
                ProspectiveAuthorityCaseOutcomeV3(
                    **common,
                    semantic_reached=False,
                    initial_semantic_review_present=False,
                    semantic_entry_authorized=False,
                    vpre_executed=False,
                    regeneration_fallback_count=0,
                    external_stage_reached=False,
                    external_eligible_lineage_count=0,
                    n10_certified_count=0,
                    n10_unresolved_count=0,
                    n10_rejected_count=0,
                    binding_ready_lineage_count=0,
                    vpost_completed_count=0,
                    scientific_certification_decision_counts={},
                    final_status="PRE_SEMANTIC_ALPHA4_EMPTY_TERMINAL",
                )
            )
            continue

        if manifest_status != "complete_after_initial_semantic":
            raise ValueError(
                case.case_id
                + ": unexpected prospective initial manifest status: "
                + repr(manifest_status)
            )
        if initial_count == 0:
            raise ValueError(
                case.case_id + ": semantic cutpoint cannot have empty portfolio"
            )

        cutpoint = manifest.get("prospective_initial_semantic_cutpoint")
        if not isinstance(cutpoint, dict):
            raise ValueError(case.case_id + ": semantic cutpoint metadata missing")
        if cutpoint.get("enabled") is not True:
            raise ValueError(case.case_id + ": semantic cutpoint not enabled")
        if cutpoint.get("cut_after_stage") != 9:
            raise ValueError(case.case_id + ": semantic cutpoint stage drift")
        for flag in (
            "external_novelty_performed",
            "n9_performed",
            "n10_performed",
            "refinement_performed",
            "final_semantic_performed",
            "feasibility_performed",
            "production_selection_changed",
            "canonical_graph_mutated",
        ):
            if cutpoint.get(flag) is not False:
                raise ValueError(
                    case.case_id + ": forbidden initial cutpoint activity: " + flag
                )

        _verify_cutpoint_artifact(
            cutpoint=cutpoint,
            key="portfolio",
            expected_path=portfolio_path,
        )
        _verify_cutpoint_artifact(
            cutpoint=cutpoint,
            key="hypothesis_context",
            expected_path=Path(case.initial_hypothesis_context_path),
        )
        semantic_run_path = Path(
            case.initial_semantic_run_path
        ).expanduser().resolve()
        _verify_cutpoint_artifact(
            cutpoint=cutpoint,
            key="semantic_run",
            expected_path=semantic_run_path,
        )
        _verify_cutpoint_artifact(
            cutpoint=cutpoint,
            key="provider_plan",
            expected_path=Path(case.initial_provider_plan_path),
        )

        review_path = Path(
            case.initial_semantic_review_path
        ).expanduser().resolve()
        review_row = cutpoint.get("artifacts", {}).get("semantic_review")
        if review_path.is_file():
            if not isinstance(review_row, dict):
                raise ValueError(
                    case.case_id + ": semantic review exists but is not fingerprinted"
                )
            _verify_fingerprint(
                path=str(review_row.get("path", "")),
                digest=str(review_row.get("file_sha256", "")),
                expected_path=review_path,
            )
            review_sha: str | None = sha256_file(review_path)
        else:
            if review_row is not None:
                raise ValueError(
                    case.case_id + ": absent semantic review has manifest fingerprint"
                )
            review_sha = None

        if not campaign_plan_path.is_file() or not campaign_report_path.is_file():
            raise ValueError(
                case.case_id
                + ": semantic-reached prospective case lacks completed downstream campaign"
            )
        campaign_plan = PreN10ProspectiveCampaignPlanV1.model_validate_json(
            campaign_plan_path.read_text(encoding="utf-8")
        )
        campaign_report = PreN10ProspectiveCampaignReportV1.model_validate_json(
            campaign_report_path.read_text(encoding="utf-8")
        )
        if campaign_report.source_plan_id != campaign_plan.plan_id:
            raise ValueError(case.case_id + ": campaign report/plan ID mismatch")
        if campaign_report.source_plan_sha256 != campaign_plan.plan_sha256:
            raise ValueError(case.case_id + ": campaign report/plan SHA mismatch")
        _validate_campaign_plan_against_frozen_case(
            case=case,
            campaign_plan=campaign_plan,
        )

        stages = {row.stage_name: row.status for row in campaign_report.stages}
        vpre_executed = stages["initial_vpre"] in {
            "EXECUTED",
            "REUSED_VALIDATED",
        }
        vpre_payload = _stage_payload(
            report=campaign_report,
            stage_name="initial_vpre",
            schema_version="pre-n10-scientific-contract-report-v1",
        )
        if vpre_executed != (vpre_payload is not None):
            raise ValueError(case.case_id + ": V_pre stage/payload mismatch")

        if vpre_payload is None:
            initial_vpre_hypothesis_count = None
            initial_vpre_ready_hypothesis_count = None
            initial_vpre_claim_count = None
            initial_vpre_ready_claim_count = None
        else:
            initial_vpre_hypothesis_count = int(
                vpre_payload["hypothesis_count"]
            )
            initial_vpre_ready_hypothesis_count = int(
                vpre_payload["ready_hypothesis_count"]
            )
            initial_vpre_claim_count = int(vpre_payload["claim_count"])
            initial_vpre_ready_claim_count = int(
                vpre_payload["ready_claim_count"]
            )

        reentry_payload = _stage_payload(
            report=campaign_report,
            stage_name="regeneration_semantic_reentry",
            schema_version="pre-n10-regeneration-reentry-report-v2",
        )
        if campaign_report.primary_regeneration_fallback_count > 0:
            if reentry_payload is None:
                raise ValueError(
                    case.case_id + ": regeneration fallback lacks re-entry artifact"
                )
            regenerated_lineage_count: int | None = int(
                reentry_payload["lineage_count"]
            )
            regenerated_semantic_admissible_count: int | None = int(
                reentry_payload["semantic_admissible_count"]
            )
            regenerated_ready_for_n10_count: int | None = int(
                reentry_payload["ready_for_n10_count"]
            )
        else:
            regenerated_lineage_count = None
            regenerated_semantic_admissible_count = None
            regenerated_ready_for_n10_count = None

        external_reached = stages["external_n9_n10"] in {
            "EXECUTED",
            "REUSED_VALIDATED",
        }

        rows.append(
            ProspectiveAuthorityCaseOutcomeV3(
                **common,
                semantic_reached=True,
                initial_semantic_run_file_sha256=sha256_file(
                    semantic_run_path
                ),
                initial_semantic_review_present=review_path.is_file(),
                initial_semantic_review_file_sha256=review_sha,
                semantic_entry_authorized=(
                    campaign_report.initial_semantic_status
                    == "PRE_N10_ENTRY_AUTHORIZED"
                ),
                campaign_plan_path=str(campaign_plan_path),
                campaign_plan_file_sha256=sha256_file(campaign_plan_path),
                campaign_plan_id=campaign_plan.plan_id,
                campaign_report_path=str(campaign_report_path),
                campaign_report_file_sha256=sha256_file(
                    campaign_report_path
                ),
                campaign_report_id=campaign_report.report_id,
                vpre_executed=vpre_executed,
                initial_vpre_hypothesis_count=(
                    initial_vpre_hypothesis_count
                ),
                initial_vpre_ready_hypothesis_count=(
                    initial_vpre_ready_hypothesis_count
                ),
                initial_vpre_claim_count=initial_vpre_claim_count,
                initial_vpre_ready_claim_count=(
                    initial_vpre_ready_claim_count
                ),
                regeneration_fallback_count=(
                    campaign_report.primary_regeneration_fallback_count
                ),
                regenerated_lineage_count=regenerated_lineage_count,
                regenerated_semantic_admissible_count=(
                    regenerated_semantic_admissible_count
                ),
                regenerated_ready_for_n10_count=(
                    regenerated_ready_for_n10_count
                ),
                external_stage_reached=external_reached,
                external_eligible_lineage_count=(
                    campaign_report.handoff_external_eligible_count
                ),
                n10_certified_count=campaign_report.n10_certified_count,
                n10_unresolved_count=campaign_report.n10_unresolved_count,
                n10_rejected_count=campaign_report.n10_rejected_count,
                binding_ready_lineage_count=(
                    campaign_report.binding_ready_lineage_count
                ),
                vpost_completed_count=campaign_report.vpost_completed_count,
                scientific_certification_decision_counts=dict(
                    campaign_report.scientific_certification_decision_counts
                ),
                final_status=campaign_report.final_status,
                second_regeneration_performed=(
                    campaign_report.second_regeneration_performed
                ),
                result_conditioned_route_changes_performed=(
                    campaign_report.result_conditioned_route_changes_performed
                ),
                external_novelty_reassessed_in_vpost=(
                    campaign_report.external_novelty_reassessed_in_vpost
                ),
                verifier_result_consumed_by_production=(
                    campaign_report.verifier_result_consumed_by_production
                ),
                production_selection_changed=(
                    campaign_report.production_selection_changed
                ),
                canonical_graph_mutated=campaign_report.canonical_graph_mutated,
            )
        )

    final_counts = Counter(row.final_status for row in rows)
    certifications = Counter()
    for row in rows:
        certifications.update(row.scientific_certification_decision_counts)

    body = {
        "schema_version": "prospective-authority-cohort-accounting-v3",
        "source_execution_plan_path": str(source_plan_path),
        "source_execution_plan_file_sha256": sha256_file(source_plan_path),
        "source_execution_plan_id": execution_plan.plan_id,
        "source_execution_plan_sha256": execution_plan.plan_sha256,
        "source_campaign_freeze_id": execution_plan.source_campaign_freeze_id,
        "source_regeneration_unit_freeze_id": (
            execution_plan.source_regeneration_unit_freeze_id
        ),
        "execution_plan_repository_head_sha": (
            execution_plan.execution_plan_repository_head_sha
        ),
        "cases": [row.model_dump(mode="json") for row in rows],
        "case_ids": ["P21", "P22", "P23", "P24", "P25"],
        "case_count": 5,
        "initial_hypothesis_total": sum(
            row.initial_hypothesis_count for row in rows
        ),
        "alpha4_nonempty_case_count": sum(
            row.initial_hypothesis_count > 0 for row in rows
        ),
        "pre_semantic_empty_case_count": sum(
            row.final_status == "PRE_SEMANTIC_ALPHA4_EMPTY_TERMINAL"
            for row in rows
        ),
        "semantic_reached_case_count": sum(
            row.semantic_reached for row in rows
        ),
        "semantic_entry_authorized_case_count": sum(
            row.semantic_entry_authorized for row in rows
        ),
        "vpre_executed_case_count": sum(row.vpre_executed for row in rows),
        "initial_vpre_hypothesis_evaluated_total": sum(
            row.initial_vpre_hypothesis_count or 0 for row in rows
        ),
        "initial_vpre_ready_hypothesis_total": sum(
            row.initial_vpre_ready_hypothesis_count or 0 for row in rows
        ),
        "initial_vpre_claim_evaluated_total": sum(
            row.initial_vpre_claim_count or 0 for row in rows
        ),
        "initial_vpre_ready_claim_total": sum(
            row.initial_vpre_ready_claim_count or 0 for row in rows
        ),
        "regeneration_fallback_lineage_count": sum(
            row.regeneration_fallback_count for row in rows
        ),
        "regenerated_lineage_count": sum(
            row.regenerated_lineage_count or 0 for row in rows
        ),
        "regenerated_semantic_admissible_count": sum(
            row.regenerated_semantic_admissible_count or 0 for row in rows
        ),
        "regenerated_ready_for_n10_count": sum(
            row.regenerated_ready_for_n10_count or 0 for row in rows
        ),
        "external_stage_reached_case_count": sum(
            row.external_stage_reached for row in rows
        ),
        "external_eligible_lineage_count": sum(
            row.external_eligible_lineage_count for row in rows
        ),
        "n10_certified_count": sum(row.n10_certified_count for row in rows),
        "n10_unresolved_count": sum(row.n10_unresolved_count for row in rows),
        "n10_rejected_count": sum(row.n10_rejected_count for row in rows),
        "binding_ready_lineage_count": sum(
            row.binding_ready_lineage_count for row in rows
        ),
        "vpost_completed_count": sum(
            row.vpost_completed_count for row in rows
        ),
        "scientific_certification_decision_counts": dict(
            sorted(certifications.items())
        ),
        "final_status_counts": dict(sorted(final_counts.items())),
        "outcome_accounting_only": True,
        "prospective_results_preserved": True,
        "source_tasks_or_models_modified_after_observation": False,
        "later_case_adaptation_performed": False,
        "failed_or_abstained_case_replacement_performed": False,
        "prospective_case_rerun_performed": False,
        "result_conditioned_route_changes_performed": False,
        "second_regeneration_performed": False,
        "n10_absence_interpreted_as_positive_novelty": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveAuthorityCohortAccountingV3(
        **body,
        report_id=(
            "prospective_authority_cohort_accounting_v3:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "CohortFinalStatusV3",
    "ProspectiveAuthorityCaseOutcomeV3",
    "ProspectiveAuthorityCohortAccountingV3",
    "build_prospective_authority_cohort_accounting_v3",
]
