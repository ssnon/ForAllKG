from dac_her.sers_targeted_retrieval_t1_live_validation import (
    aggregate_t1_report,
)


class _Plan:
    plan_id = "provider-plan"
    mode = "STANDARD_2_PROVIDER"
    active_providers = ["openalex", "crossref"]


def test_t1_aggregate_is_mechanical_only() -> None:
    gap_audit = {
        "structural_pass": True,
        "every_query_operational": True,
        "failed_execution_count": 0,
        "observed_execution_count": 4,
        "successful_execution_count": 4,
        "delta_raw_work_count": 10,
        "delta_canonical_work_count": 8,
        "delta_abstract_work_count": 6,
    }
    report = aggregate_t1_report(
        gap_plan_id="gap-plan",
        provider_plan=_Plan(),  # type: ignore[arg-type]
        gap_audits=[gap_audit],
        skipped_gaps=[],
        total_targeted_query_count=2,
    )
    assert report["outcome"].endswith("MECHANICAL_PASS")
    assert report["scientific_novelty_reassessed"] is False
    assert report["ranker_called"] is False
    assert report["claim_reviewer_called"] is False
    assert report["llm_calls"] == 0
    assert report["fresh_reserve_c_consumed"] is False
