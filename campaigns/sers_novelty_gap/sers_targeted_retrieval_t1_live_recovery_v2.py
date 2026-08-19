from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.literature_provider_plan import LiteratureProviderPlan
from dac_her.novelty_refinement_contracts import NoveltyGapPlan
from campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_validation_v2 import (
    audit_live_gap_outcome,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_spec_v1"
)
V1_RUN_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v1"
)
V1_FAILURE_FREEZE_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_v1_failure_freeze_v1"
)
V1_FAILURE_MANIFEST = (
    V1_FAILURE_FREEZE_ROOT / "failure_manifest.json"
)
V2_RUN_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_run_v2"
)

EXPECTED_V1_SOURCE_HEAD = (
    "729ff9995909da9c7095752d77e1aa29be6f8ee9"
)
EXPECTED_SPEC_ID = (
    "sers_targeted_retrieval_t1_live_spec:"
    "8ea007ccbea7cc1b9dea"
)
EXPECTED_V1_EXCEPTION = "TypeError"

V1_EVIDENCE_FILES = [
    "LIVE_ATTEMPT_CONSUMED.json",
    "FATAL_ERROR.json",
    "gap_01/augmented_plan.json",
    "gap_01/delta_plan.json",
    "gap_01/delta_packet.json",
    "gap_01/merged_packet.json",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_frozen_context(
    spec_root: Path = SPEC_ROOT,
) -> tuple[
    dict[str, Any],
    LiteratureQueryPlan,
    PriorArtPacket,
    NoveltyGapPlan,
    LiteratureProviderPlan,
]:
    spec = json.loads(
        (spec_root / "t1_spec.json").read_text(encoding="utf-8")
    )
    base_plan = LiteratureQueryPlan.model_validate_json(
        (spec_root / "base_query_plan.json").read_text(encoding="utf-8")
    )
    base_packet = PriorArtPacket.model_validate_json(
        (spec_root / "base_prior_art_packet.json")
        .read_text(encoding="utf-8")
    )
    gap_plan = NoveltyGapPlan.model_validate_json(
        (spec_root / "novelty_gap_plan.json").read_text(encoding="utf-8")
    )
    provider_plan = LiteratureProviderPlan.model_validate_json(
        (spec_root / "provider_plan.json").read_text(encoding="utf-8")
    )
    return spec, base_plan, base_packet, gap_plan, provider_plan


def load_v1_gap1_models(
    *,
    v1_run_root: Path = V1_RUN_ROOT,
) -> tuple[
    LiteratureQueryPlan,
    LiteratureQueryPlan,
    PriorArtPacket,
    PriorArtPacket,
]:
    gap_root = v1_run_root / "gap_01"
    augmented = LiteratureQueryPlan.model_validate_json(
        (gap_root / "augmented_plan.json").read_text(encoding="utf-8")
    )
    delta_plan = LiteratureQueryPlan.model_validate_json(
        (gap_root / "delta_plan.json").read_text(encoding="utf-8")
    )
    delta_packet = PriorArtPacket.model_validate_json(
        (gap_root / "delta_packet.json").read_text(encoding="utf-8")
    )
    merged_packet = PriorArtPacket.model_validate_json(
        (gap_root / "merged_packet.json").read_text(encoding="utf-8")
    )
    return augmented, delta_plan, delta_packet, merged_packet


def recover_v1_gap1_audit(
    *,
    spec_root: Path = SPEC_ROOT,
    v1_run_root: Path = V1_RUN_ROOT,
) -> dict[str, Any]:
    (
        _spec,
        base_plan,
        base_packet,
        gap_plan,
        provider_plan,
    ) = load_frozen_context(spec_root)
    if not gap_plan.gaps:
        raise RuntimeError("frozen gap plan is empty")
    gap = gap_plan.gaps[0]
    if not gap.targeted_queries:
        raise RuntimeError("v1 gap_01 is not a targeted gap")
    augmented, delta_plan, delta_packet, merged_packet = (
        load_v1_gap1_models(v1_run_root=v1_run_root)
    )
    return audit_live_gap_outcome(
        base_plan=base_plan,
        base_packet=base_packet,
        gap=gap,
        provider_plan=provider_plan,
        augmented_plan=augmented,
        delta_plan=delta_plan,
        delta_packet=delta_packet,
        merged_packet=merged_packet,
    )


def validate_v1_failure_evidence(
    *,
    root: Path = ROOT,
    spec_root: Path = SPEC_ROOT,
    v1_run_root: Path = V1_RUN_ROOT,
) -> dict[str, Any]:
    for rel in V1_EVIDENCE_FILES:
        path = v1_run_root / rel
        if not path.is_file():
            raise RuntimeError(f"missing v1 failure evidence: {rel}")

    marker = json.loads(
        (v1_run_root / "LIVE_ATTEMPT_CONSUMED.json")
        .read_text(encoding="utf-8")
    )
    fatal = json.loads(
        (v1_run_root / "FATAL_ERROR.json")
        .read_text(encoding="utf-8")
    )
    if marker.get("spec_id") != EXPECTED_SPEC_ID:
        raise RuntimeError("v1 marker spec ID mismatch")
    if marker.get("source_git_head") != EXPECTED_V1_SOURCE_HEAD:
        raise RuntimeError("v1 marker source git HEAD mismatch")
    if marker.get("guarded_preflight_passed") is not True:
        raise RuntimeError("v1 marker does not record guarded preflight PASS")
    if marker.get("one_shot") is not True:
        raise RuntimeError("v1 marker is not one-shot")
    if marker.get("rerun_authorized") is not False:
        raise RuntimeError("v1 marker unexpectedly authorizes rerun")
    if marker.get("fresh_reserve_c_consumed") is not False:
        raise RuntimeError("v1 marker records Reserve C consumption")
    if fatal.get("exception_type") != EXPECTED_V1_EXCEPTION:
        raise RuntimeError("unexpected v1 fatal exception type")
    if fatal.get("fresh_reserve_c_consumed") is not False:
        raise RuntimeError("v1 fatal evidence records Reserve C consumption")

    # The artifact boundary proves retrieval+merge completed before the
    # audit crash. There must be no completed v1 gap audit/report marker.
    forbidden = [
        v1_run_root / "gap_01/gap_audit.json",
        v1_run_root / "t1_live_report.json",
        v1_run_root / "MECHANICAL_PASS.json",
        v1_run_root / "MECHANICAL_INCOMPLETE_OR_FAIL.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError(
            "v1 evidence shape changed: completed audit/report artifact exists"
        )

    audit = recover_v1_gap1_audit(
        spec_root=spec_root,
        v1_run_root=v1_run_root,
    )
    if audit["structural_pass"] is not True:
        raise RuntimeError(
            "offline recovery audit of v1 gap_01 is structurally invalid"
        )

    return {
        "v1_source_git_head": marker["source_git_head"],
        "v1_spec_id": marker["spec_id"],
        "v1_exception_type": fatal["exception_type"],
        "gap_id": audit["gap_id"],
        "hypothesis_id": audit["hypothesis_id"],
        "delta_query_count": audit["delta_query_count"],
        "observed_execution_count": audit["observed_execution_count"],
        "successful_execution_count": audit[
            "successful_execution_count"
        ],
        "failed_execution_count": audit["failed_execution_count"],
        "delta_canonical_work_count": audit[
            "delta_canonical_work_count"
        ],
        "delta_abstract_work_count": audit[
            "delta_abstract_work_count"
        ],
        "every_query_operational": audit[
            "every_query_operational"
        ],
        "all_provider_executions_successful": audit[
            "all_provider_executions_successful"
        ],
        "structural_pass": audit["structural_pass"],
        "recovered_audit_sha256": audit["audit_sha256"],
        "network_replay_authorized": False,
        "fresh_reserve_c_consumed": False,
    }


def build_v2_report(
    *,
    gap_plan_id: str,
    provider_plan: LiteratureProviderPlan,
    gap_audits: list[dict[str, Any]],
    skipped_gaps: list[dict[str, Any]],
    total_targeted_query_count: int,
    recovered_v1_provider_execution_count: int,
    v2_new_provider_execution_count: int,
) -> dict[str, Any]:
    all_structural = all(
        row["structural_pass"] for row in gap_audits
    )
    every_query_operational = all(
        row["every_query_operational"] for row in gap_audits
    )
    any_provider_failure = any(
        row["failed_execution_count"] > 0 for row in gap_audits
    )
    if not all_structural:
        outcome = "SERS_T1_LIVE_TARGETED_RETRIEVAL_V2_STRUCTURAL_FAIL"
    elif not every_query_operational:
        outcome = (
            "SERS_T1_LIVE_TARGETED_RETRIEVAL_V2_"
            "INCOMPLETE_QUERY_EXECUTION_COVERAGE"
        )
    elif any_provider_failure:
        outcome = (
            "SERS_T1_LIVE_TARGETED_RETRIEVAL_V2_"
            "MECHANICAL_PASS_WITH_PROVIDER_FAILURES"
        )
    else:
        outcome = (
            "SERS_T1_LIVE_TARGETED_RETRIEVAL_V2_MECHANICAL_PASS"
        )

    body: dict[str, Any] = {
        "schema_version":
            "sers-targeted-retrieval-t1-live-report-v2",
        "gap_plan_id": gap_plan_id,
        "provider_plan_id": provider_plan.plan_id,
        "provider_mode": provider_plan.mode,
        "providers": list(provider_plan.active_providers),
        "targeted_gap_count": len(gap_audits),
        "skipped_gap_count": len(skipped_gaps),
        "total_targeted_query_count":
            total_targeted_query_count,
        "expected_provider_execution_count":
            total_targeted_query_count
            * len(provider_plan.active_providers),
        "observed_provider_execution_count": sum(
            row["observed_execution_count"]
            for row in gap_audits
        ),
        "successful_provider_execution_count": sum(
            row["successful_execution_count"]
            for row in gap_audits
        ),
        "failed_provider_execution_count": sum(
            row["failed_execution_count"]
            for row in gap_audits
        ),
        "delta_raw_work_count": sum(
            row["delta_raw_work_count"] for row in gap_audits
        ),
        "delta_canonical_work_count": sum(
            row["delta_canonical_work_count"]
            for row in gap_audits
        ),
        "delta_abstract_work_count": sum(
            row["delta_abstract_work_count"]
            for row in gap_audits
        ),
        "gap_audits": gap_audits,
        "skipped_gaps": skipped_gaps,
        "all_structural_checks_pass": all_structural,
        "every_targeted_query_operational":
            every_query_operational,
        "v1_failure_recovered_offline": True,
        "v1_gap1_network_replayed": False,
        "recovered_v1_provider_execution_count":
            recovered_v1_provider_execution_count,
        "v2_new_provider_execution_count":
            v2_new_provider_execution_count,
        "scientific_query_quality_approval": False,
        "scientific_novelty_reassessed": False,
        "ranker_called": False,
        "claim_reviewer_called": False,
        "llm_calls": 0,
        "hypothesis_rewrite_called": False,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
        "outcome": outcome,
    }
    body["report_sha256"] = _sha256_json(body)
    body["run_id"] = (
        "sers_targeted_retrieval_t1_live_v2:"
        + body["report_sha256"][:20]
    )
    return body
