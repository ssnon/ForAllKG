from __future__ import annotations

import hashlib
import json
from typing import Any

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.literature_provider_plan import LiteratureProviderPlan
from dac_her.literature_retrieval import canonicalize_prior_art_works
from dac_her.novelty_refinement_contracts import NoveltyGap
from dac_her.targeted_novelty_retrieval import build_augmented_query_plan


def _jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _query_claim_map(plan: LiteratureQueryPlan) -> dict[str, str | None]:
    return {row.query_id: row.claim_id for row in plan.queries}


def audit_live_gap_outcome(
    *,
    base_plan: LiteratureQueryPlan,
    base_packet: PriorArtPacket,
    gap: NoveltyGap,
    provider_plan: LiteratureProviderPlan,
    augmented_plan: LiteratureQueryPlan,
    delta_plan: LiteratureQueryPlan,
    delta_packet: PriorArtPacket,
    merged_packet: PriorArtPacket,
) -> dict[str, Any]:
    expected_augmented, expected_delta = build_augmented_query_plan(
        base_plan,
        gap,
    )
    delta_ids = [row.query_id for row in delta_plan.queries]
    delta_id_set = set(delta_ids)
    query_claim = _query_claim_map(delta_plan)
    planned_claim_ids = {
        row.claim_id for row in gap.targeted_queries
    }

    execution_query_ids = [row.query_id for row in delta_packet.executions]
    execution_pairs = [
        (row.query_id, row.provider)
        for row in delta_packet.executions
    ]
    expected_pairs = [
        (query.query_id, provider)
        for query in delta_plan.queries
        for provider in provider_plan.active_providers
    ]

    success_by_query = {
        query_id: 0 for query_id in delta_ids
    }
    results_by_query = {
        query_id: 0 for query_id in delta_ids
    }
    provider_failures: list[dict[str, Any]] = []
    for execution in delta_packet.executions:
        if execution.success:
            success_by_query[execution.query_id] = (
                success_by_query.get(execution.query_id, 0) + 1
            )
            results_by_query[execution.query_id] = (
                results_by_query.get(execution.query_id, 0)
                + execution.result_count
            )
        else:
            provider_failures.append(
                {
                    "query_id": execution.query_id,
                    "provider": execution.provider,
                    "error_type": (
                        str(execution.error or "").split(":", 1)[0]
                        or "UNKNOWN"
                    ),
                }
            )

    work_provenance_valid = True
    delta_work_claim_bindings: list[dict[str, Any]] = []
    for work in delta_packet.works:
        work_query_ids = set(work.retrieval_query_ids)
        expected_claims = {
            query_claim[qid]
            for qid in work_query_ids
            if qid in query_claim and query_claim[qid] is not None
        }
        observed_claims = set(work.retrieval_claim_ids)
        row_valid = (
            bool(work_query_ids)
            and work_query_ids <= delta_id_set
            and observed_claims == expected_claims
            and observed_claims <= planned_claim_ids
        )
        work_provenance_valid = work_provenance_valid and row_valid
        delta_work_claim_bindings.append(
            {
                "work_id": work.work_id,
                "query_ids": sorted(work_query_ids),
                "expected_claim_ids": sorted(expected_claims),
                "observed_claim_ids": sorted(observed_claims),
                "valid": row_valid,
            }
        )

    base_query_provenance = {
        query_id
        for work in base_packet.works
        for query_id in work.retrieval_query_ids
    }
    base_claim_provenance = {
        claim_id
        for work in base_packet.works
        for claim_id in work.retrieval_claim_ids
    }
    merged_query_provenance = {
        query_id
        for work in merged_packet.works
        for query_id in work.retrieval_query_ids
    }
    merged_claim_provenance = {
        claim_id
        for work in merged_packet.works
        for claim_id in work.retrieval_claim_ids
    }
    delta_query_provenance = {
        query_id
        for work in delta_packet.works
        for query_id in work.retrieval_query_ids
    }
    delta_claim_provenance = {
        claim_id
        for work in delta_packet.works
        for claim_id in work.retrieval_claim_ids
    }

    recanonicalized, _ = canonicalize_prior_art_works(
        list(merged_packet.works)
    )
    recanonicalized = sorted(
        recanonicalized,
        key=lambda row: (
            -(row.citation_count or 0),
            -(row.year or 0),
            row.title.lower(),
        ),
    )

    checks = {
        "augmented_plan_exact":
            _canonical_json(augmented_plan)
            == _canonical_json(expected_augmented),
        "delta_plan_exact":
            _canonical_json(delta_plan)
            == _canonical_json(expected_delta),
        "delta_queries_nonempty": bool(delta_ids),
        "delta_query_ids_unique":
            len(delta_ids) == len(delta_id_set),
        "delta_query_claims_exact":
            [
                (row.claim_id, row.query_text)
                for row in delta_plan.queries
            ]
            == [
                (row.claim_id, row.query_text)
                for row in expected_delta.queries
            ],
        "delta_packet_source_plan":
            delta_packet.source_query_plan_id == delta_plan.plan_id,
        "delta_packet_source_portfolio":
            delta_packet.source_portfolio_id
            == base_plan.source_portfolio_id,
        "execution_query_ids_known":
            set(execution_query_ids) <= delta_id_set,
        "execution_pairs_exact":
            execution_pairs == expected_pairs,
        "provider_set_exact":
            sorted(set(row.provider for row in delta_packet.executions))
            == sorted(provider_plan.active_providers),
        "delta_work_query_claim_provenance_exact":
            work_provenance_valid,
        "merged_packet_source_plan":
            merged_packet.source_query_plan_id == augmented_plan.plan_id,
        "merged_packet_source_portfolio":
            merged_packet.source_portfolio_id
            == base_plan.source_portfolio_id,
        "base_query_provenance_retained":
            base_query_provenance <= merged_query_provenance,
        "base_claim_provenance_retained":
            base_claim_provenance <= merged_claim_provenance,
        "delta_query_provenance_retained":
            delta_query_provenance <= merged_query_provenance,
        "delta_claim_provenance_retained":
            delta_claim_provenance <= merged_claim_provenance,
        "delta_count_accounting":
            delta_packet.canonical_work_count
            == len(delta_packet.works)
            and delta_packet.deduplicated_work_count
            == max(
                0,
                delta_packet.raw_work_count
                - len(delta_packet.works),
            ),
        "merged_count_accounting":
            merged_packet.canonical_work_count
            == len(merged_packet.works)
            and merged_packet.raw_work_count
            == (
                base_packet.raw_work_count
                + delta_packet.raw_work_count
            )
            and merged_packet.deduplicated_work_count
            == max(
                0,
                merged_packet.raw_work_count
                - len(merged_packet.works),
            ),
        "shared_canonicalization_idempotent":
            _canonical_json(recanonicalized)
            == _canonical_json(merged_packet.works),
        "epistemic_usage_unchanged":
            delta_packet.epistemic_usage
            == "prior_art_only_not_positive_premise"
            and merged_packet.epistemic_usage
            == "prior_art_only_not_positive_premise",
    }

    every_query_operational = all(
        success_by_query.get(query_id, 0) >= 1
        for query_id in delta_ids
    )
    all_provider_executions_successful = all(
        row.success for row in delta_packet.executions
    )
    structural_pass = all(checks.values())

    if not structural_pass:
        outcome = "T1_GAP_STRUCTURAL_FAIL"
    elif not every_query_operational:
        outcome = "T1_GAP_INCOMPLETE_QUERY_EXECUTION_COVERAGE"
    elif not all_provider_executions_successful:
        outcome = "T1_GAP_STRUCTURAL_PASS_WITH_PROVIDER_FAILURES"
    else:
        outcome = "T1_GAP_STRUCTURAL_PASS"

    body = {
        "schema_version":
            "sers-targeted-retrieval-t1-gap-audit-v2",
        "gap_id": gap.gap_id,
        "hypothesis_id": gap.hypothesis_id,
        "action": gap.action,
        "delta_query_count": len(delta_plan.queries),
        "providers": list(provider_plan.active_providers),
        "expected_execution_count": len(expected_pairs),
        "observed_execution_count": len(
            delta_packet.executions
        ),
        "successful_execution_count": sum(
            1 for row in delta_packet.executions if row.success
        ),
        "failed_execution_count": sum(
            1 for row in delta_packet.executions if not row.success
        ),
        "query_success_provider_counts": success_by_query,
        "query_success_result_counts": results_by_query,
        "provider_failures": provider_failures,
        "delta_raw_work_count": delta_packet.raw_work_count,
        "delta_canonical_work_count": len(delta_packet.works),
        "delta_abstract_work_count": sum(
            1 for row in delta_packet.works if row.abstract
        ),
        "merged_canonical_work_count": len(merged_packet.works),
        "merged_abstract_work_count": sum(
            1 for row in merged_packet.works if row.abstract
        ),
        "checks": checks,
        "delta_work_claim_bindings":
            delta_work_claim_bindings,
        "every_query_operational":
            every_query_operational,
        "all_provider_executions_successful":
            all_provider_executions_successful,
        "structural_pass": structural_pass,
        "outcome": outcome,
        "scientific_novelty_reassessed": False,
        "ranker_called": False,
        "claim_reviewer_called": False,
        "llm_calls": 0,
        "hypothesis_rewrite_called": False,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
    }
    body["audit_sha256"] = _sha256_json(body)
    return body


def aggregate_t1_report(
    *,
    gap_plan_id: str,
    provider_plan: LiteratureProviderPlan,
    gap_audits: list[dict[str, Any]],
    skipped_gaps: list[dict[str, Any]],
    total_targeted_query_count: int,
) -> dict[str, Any]:
    all_structural = all(
        row["structural_pass"] for row in gap_audits
    )
    every_query_operational = all(
        row["every_query_operational"] for row in gap_audits
    )
    any_provider_failure = any(
        row["failed_execution_count"] > 0
        for row in gap_audits
    )
    if not all_structural:
        outcome = "SERS_T1_LIVE_TARGETED_RETRIEVAL_STRUCTURAL_FAIL"
    elif not every_query_operational:
        outcome = (
            "SERS_T1_LIVE_TARGETED_RETRIEVAL_"
            "INCOMPLETE_QUERY_EXECUTION_COVERAGE"
        )
    elif any_provider_failure:
        outcome = (
            "SERS_T1_LIVE_TARGETED_RETRIEVAL_"
            "MECHANICAL_PASS_WITH_PROVIDER_FAILURES"
        )
    else:
        outcome = (
            "SERS_T1_LIVE_TARGETED_RETRIEVAL_MECHANICAL_PASS"
        )

    body = {
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
        "all_structural_checks_pass":
            all_structural,
        "every_targeted_query_operational":
            every_query_operational,
        "scientific_query_quality_approval":
            False,
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
        "sers_targeted_retrieval_t1_live:"
        + body["report_sha256"][:20]
    )
    return body
