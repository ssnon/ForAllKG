from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
    PreN10ProspectiveCampaignPlanV1,
    PreN10ProspectiveCampaignReportV1,
    sha256_file,
)
from pipeline_core.discovery.prospective_authority_cohort_accounting_v3 import (
    ProspectiveAuthorityCohortAccountingV3,
    build_prospective_authority_cohort_accounting_v3,
)
from tests.discovery.test_prospective_authority_execution_plan_v3_s170 import (
    _plan,
)


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _campaign_plan(case) -> PreN10ProspectiveCampaignPlanV1:
    argv = case.downstream_campaign_argv_base
    review = Path(case.initial_semantic_review_path)
    body = {
        "schema_version": "pre-n10-prospective-campaign-plan-v1",
        "portfolio": _fingerprint(Path(case.initial_portfolio_path)),
        "semantic_run": _fingerprint(Path(case.initial_semantic_run_path)),
        "semantic_review": _fingerprint(review) if review.is_file() else None,
        "hypothesis_context": _fingerprint(
            Path(case.initial_hypothesis_context_path)
        ),
        "regeneration_unit_freeze": _fingerprint(
            Path(_argv_value(argv, "--regeneration-unit-freeze"))
        ),
        "provider_plan": _fingerprint(Path(case.initial_provider_plan_path)),
        "output_root": str(
            Path(case.downstream_campaign_output_root).resolve()
        ),
        "decomposition_model": _argv_value(argv, "--decomposition-model"),
        "primary_model": _argv_value(argv, "--primary-model"),
        "specification_repair_model": _argv_value(
            argv, "--specification-repair-model"
        ),
        "specification_audit_model": _argv_value(
            argv, "--specification-audit-model"
        ),
        "source_alignment_model": _argv_value(
            argv, "--source-alignment-model"
        ),
        "regeneration_model": _argv_value(argv, "--regeneration-model"),
        "semantic_critic_model": _argv_value(
            argv, "--semantic-critic-model"
        ),
        "external_n10_model": _argv_value(argv, "--external-n10-model"),
        "vpost_model": _argv_value(argv, "--vpost-model"),
        "api_key_env": _argv_value(argv, "--api-key-env"),
        "base_url": _argv_value(argv, "--base-url"),
        "parse_retries": int(_argv_value(argv, "--parse-retries")),
        "timeout_seconds": float(_argv_value(argv, "--timeout-seconds")),
        "max_claims": int(_argv_value(argv, "--max-claims")),
        "max_queries_per_claim": int(
            _argv_value(argv, "--max-queries-per-claim")
        ),
        "save_prompts": "--save-prompts" in argv,
        "allow_dirty_worktree_for_vpost": False,
        "stage_order": list(CAMPAIGN_STAGE_ORDER),
        "resume_scope": "BETWEEN_COMPLETED_WRITE_ONCE_STAGES_ONLY",
        "partial_stage_resume_allowed": False,
        "second_regeneration_allowed": False,
        "result_conditioned_route_changes_allowed": False,
        "production_selection_authority": False,
    }
    digest = _sha_json(body)
    return PreN10ProspectiveCampaignPlanV1(
        **body,
        plan_id="pre_n10_prospective_campaign_plan_v1:" + digest[:20],
        plan_sha256=digest,
    )


def _stage_artifact(
    root: Path,
    stage_name: str,
    *,
    schema: str | None = None,
    extra: dict | None = None,
) -> dict:
    path = root / ("fixture." + stage_name + ".json")
    payload = {"schema_version": schema or ("fixture-" + stage_name)}
    if extra:
        payload.update(extra)
    _write_json(path, payload)
    return _fingerprint(path)


def _campaign_report(
    *,
    case,
    campaign_plan: PreN10ProspectiveCampaignPlanV1,
    final_status: str,
    semantic_status: str,
    regeneration_fallback_count: int,
    hypothesis_count: int,
    claim_count: int,
) -> PreN10ProspectiveCampaignReportV1:
    root = Path(case.downstream_campaign_output_root)
    stages = []

    for index, name in enumerate(CAMPAIGN_STAGE_ORDER, start=1):
        if final_status == "INITIAL_SEMANTIC_TERMINAL":
            active = name == "initial_semantic_gate"
            status = "EXECUTED" if active else "SKIPPED_TERMINAL"
        else:
            if name in {
                "initial_semantic_gate",
                "initial_vpre",
                "primary_router",
                "downstream_handoff",
            }:
                active = True
                status = "EXECUTED"
            elif name in {
                "one_shot_regeneration",
                "regeneration_semantic_reentry",
            }:
                active = regeneration_fallback_count > 0
                status = "EXECUTED" if active else "SKIPPED_NOT_REQUIRED"
            else:
                active = False
                status = "SKIPPED_TERMINAL"

        if active:
            if name == "initial_vpre":
                artifact = _stage_artifact(
                    root,
                    name,
                    schema="pre-n10-scientific-contract-report-v1",
                    extra={
                        "hypothesis_count": hypothesis_count,
                        "ready_hypothesis_count": 0,
                        "claim_count": claim_count,
                        "ready_claim_count": 0,
                    },
                )
            elif name == "regeneration_semantic_reentry":
                artifact = _stage_artifact(
                    root,
                    name,
                    schema="pre-n10-regeneration-reentry-report-v2",
                    extra={
                        "lineage_count": regeneration_fallback_count,
                        "semantic_admissible_count": regeneration_fallback_count,
                        "ready_for_n10_count": 0,
                    },
                )
            else:
                artifact = _stage_artifact(root, name)
            stages.append(
                {
                    "stage_index": index,
                    "stage_name": name,
                    "status": status,
                    "argv": ["fixture", name],
                    "output_artifacts": [artifact],
                    "authority_artifact_id": "fixture:" + name,
                    "authority_artifact_sha256": "a" * 64,
                    "reason": None,
                }
            )
        else:
            stages.append(
                {
                    "stage_index": index,
                    "stage_name": name,
                    "status": status,
                    "argv": [],
                    "output_artifacts": [],
                    "authority_artifact_id": None,
                    "authority_artifact_sha256": None,
                    "reason": "fixture terminal",
                }
            )

    body = {
        "schema_version": "pre-n10-prospective-campaign-report-v1",
        "source_plan_id": campaign_plan.plan_id,
        "source_plan_sha256": campaign_plan.plan_sha256,
        "final_status": final_status,
        "stages": stages,
        "stage_count": len(stages),
        "initial_semantic_status": semantic_status,
        "primary_regeneration_fallback_count": regeneration_fallback_count,
        "handoff_external_eligible_count": 0,
        "n10_certified_count": 0,
        "n10_unresolved_count": 0,
        "n10_rejected_count": 0,
        "binding_ready_lineage_count": 0,
        "vpost_completed_count": 0,
        "scientific_certification_decision_counts": {},
        "initial_semantic_llm_reinvoked": False,
        "second_regeneration_performed": False,
        "result_conditioned_route_changes_performed": False,
        "external_novelty_reassessed_in_vpost": False,
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha_json(body)
    return PreN10ProspectiveCampaignReportV1(
        **body,
        report_id="pre_n10_prospective_campaign_report_v1:" + digest[:20],
        report_sha256=digest,
    )


def _materialize_fixture(tmp_path: Path) -> Path:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "execution.plan.json"
    _write_json(plan_path, plan.model_dump(mode="json"))

    outcomes = {
        "P21": ("PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE", "PRE_N10_ENTRY_AUTHORIZED", 3, 3, 9),
        "P22": ("INITIAL_SEMANTIC_TERMINAL", "SEMANTIC_INTERVENTION_REQUIRED", 0, 5, 0),
        "P23": ("PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE", "PRE_N10_ENTRY_AUTHORIZED", 3, 3, 8),
        "P24": ("PRE_SEMANTIC_ALPHA4_EMPTY_TERMINAL", None, 0, 0, 0),
        "P25": ("PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE", "PRE_N10_ENTRY_AUTHORIZED", 3, 3, 9),
    }

    for case in plan.cases:
        final_status, semantic_status, regen, hypothesis_count, claim_count = outcomes[
            case.case_id
        ]
        portfolio = Path(case.initial_portfolio_path)
        context = Path(case.initial_hypothesis_context_path)
        provider = Path(case.initial_provider_plan_path)
        manifest = Path(case.initial_manifest_path)
        _write_json(
            portfolio,
            {"hypotheses": [{"fixture": i} for i in range(hypothesis_count)]},
        )
        _write_json(context, {"fixture": "context"})
        _write_json(provider, {"fixture": "provider"})

        if case.case_id == "P24":
            _write_json(
                manifest,
                {"status": "complete_no_hypotheses_after_alpha4"},
            )
            continue

        semantic_run = Path(case.initial_semantic_run_path)
        semantic_review = Path(case.initial_semantic_review_path)
        _write_json(semantic_run, {"fixture": "semantic-run"})
        _write_json(semantic_review, {"fixture": "semantic-review"})

        artifacts = {
            "portfolio": {
                "path": str(portfolio.resolve()),
                "file_sha256": sha256_file(portfolio),
            },
            "hypothesis_context": {
                "path": str(context.resolve()),
                "file_sha256": sha256_file(context),
            },
            "semantic_run": {
                "path": str(semantic_run.resolve()),
                "file_sha256": sha256_file(semantic_run),
            },
            "semantic_review": {
                "path": str(semantic_review.resolve()),
                "file_sha256": sha256_file(semantic_review),
            },
            "provider_plan": {
                "path": str(provider.resolve()),
                "file_sha256": sha256_file(provider),
            },
        }
        _write_json(
            manifest,
            {
                "status": "complete_after_initial_semantic",
                "prospective_initial_semantic_cutpoint": {
                    "enabled": True,
                    "cut_after_stage": 9,
                    "artifacts": artifacts,
                    "external_novelty_performed": False,
                    "n9_performed": False,
                    "n10_performed": False,
                    "refinement_performed": False,
                    "final_semantic_performed": False,
                    "feasibility_performed": False,
                    "production_selection_changed": False,
                    "canonical_graph_mutated": False,
                },
            },
        )

        campaign_plan = _campaign_plan(case)
        root = Path(case.downstream_campaign_output_root)
        _write_json(
            root / "campaign.plan.json",
            campaign_plan.model_dump(mode="json"),
        )
        campaign_report = _campaign_report(
            case=case,
            campaign_plan=campaign_plan,
            final_status=final_status,
            semantic_status=semantic_status,
            regeneration_fallback_count=regen,
            hypothesis_count=hypothesis_count,
            claim_count=claim_count,
        )
        _write_json(
            root / "campaign.report.json",
            campaign_report.model_dump(mode="json"),
        )

    return plan_path


def test_p21_p25_accounting_separates_reachability_layers(
    tmp_path: Path,
) -> None:
    plan_path = _materialize_fixture(tmp_path)
    report = build_prospective_authority_cohort_accounting_v3(
        execution_plan_path=plan_path
    )

    assert report.case_ids == ["P21", "P22", "P23", "P24", "P25"]
    assert report.initial_hypothesis_total == 14
    assert report.alpha4_nonempty_case_count == 4
    assert report.pre_semantic_empty_case_count == 1
    assert report.semantic_reached_case_count == 4
    assert report.semantic_entry_authorized_case_count == 3
    assert report.vpre_executed_case_count == 3
    assert report.initial_vpre_hypothesis_evaluated_total == 9
    assert report.initial_vpre_ready_hypothesis_total == 0
    assert report.initial_vpre_claim_evaluated_total == 26
    assert report.initial_vpre_ready_claim_total == 0
    assert report.regeneration_fallback_lineage_count == 9
    assert report.regenerated_lineage_count == 9
    assert report.regenerated_semantic_admissible_count == 9
    assert report.regenerated_ready_for_n10_count == 0
    assert report.external_stage_reached_case_count == 0
    assert report.external_eligible_lineage_count == 0
    assert report.vpost_completed_count == 0
    assert report.n10_absence_interpreted_as_positive_novelty is False
    assert report.final_status_counts == {
        "INITIAL_SEMANTIC_TERMINAL": 1,
        "PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE": 3,
        "PRE_SEMANTIC_ALPHA4_EMPTY_TERMINAL": 1,
    }


def test_p24_is_distinct_pre_semantic_terminal(tmp_path: Path) -> None:
    plan_path = _materialize_fixture(tmp_path)
    report = build_prospective_authority_cohort_accounting_v3(
        execution_plan_path=plan_path
    )
    p24 = next(row for row in report.cases if row.case_id == "P24")
    assert p24.initial_hypothesis_count == 0
    assert p24.semantic_reached is False
    assert p24.vpre_executed is False
    assert p24.campaign_report_id is None
    assert p24.final_status == "PRE_SEMANTIC_ALPHA4_EMPTY_TERMINAL"


def test_accounting_rejects_semantic_artifact_on_p24(
    tmp_path: Path,
) -> None:
    plan_path = _materialize_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    p24 = next(row for row in plan["cases"] if row["case_id"] == "P24")
    _write_json(Path(p24["initial_semantic_run_path"]), {"unexpected": True})

    with pytest.raises(ValueError, match="alpha4-empty terminal has semantic run"):
        build_prospective_authority_cohort_accounting_v3(
            execution_plan_path=plan_path
        )


def test_accounting_rejects_missing_downstream_for_semantic_case(
    tmp_path: Path,
) -> None:
    plan_path = _materialize_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    p21 = next(row for row in plan["cases"] if row["case_id"] == "P21")
    Path(
        p21["downstream_campaign_output_root"], "campaign.report.json"
    ).unlink()

    with pytest.raises(ValueError, match="lacks completed downstream campaign"):
        build_prospective_authority_cohort_accounting_v3(
            execution_plan_path=plan_path
        )


def test_accounting_rejects_cutpoint_artifact_tamper(
    tmp_path: Path,
) -> None:
    plan_path = _materialize_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    p21 = next(row for row in plan["cases"] if row["case_id"] == "P21")
    Path(p21["initial_portfolio_path"]).write_text(
        '{"hypotheses":[{"tampered":true}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact SHA mismatch"):
        build_prospective_authority_cohort_accounting_v3(
            execution_plan_path=plan_path
        )


def test_accounting_report_hash_is_self_validating(
    tmp_path: Path,
) -> None:
    plan_path = _materialize_fixture(tmp_path)
    report = build_prospective_authority_cohort_accounting_v3(
        execution_plan_path=plan_path
    )
    payload = report.model_dump(mode="json")
    payload["external_eligible_lineage_count"] = 1

    with pytest.raises(ValueError):
        ProspectiveAuthorityCohortAccountingV3.model_validate(payload)
