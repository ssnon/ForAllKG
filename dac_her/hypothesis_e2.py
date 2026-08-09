from __future__ import annotations

import hashlib
from pathlib import Path

from dac_her.hypothesis_benchmark_contracts import HypothesisEvaluationReport
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_e2_contracts import (
    HypothesisE2HumanReviewWorksheet,
    HypothesisE2OutputRecord,
)
from dac_her.hypothesis_gold_contracts import SemanticGoldExpectation
from dac_her.hypothesis_real_gold_contracts import (
    HypothesisRealGoldCaseSpec,
    HypothesisRealGoldSpec,
)


REQUIRED_E2_SCENARIOS = {"candidate", "alignment", "partial", "abstention"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scenario_postcondition(
    record_case_id: str,
    scenario: str,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    evaluation: HypothesisEvaluationReport,
) -> None:
    if not evaluation.hard_gate_passed:
        raise ValueError(
            f"{record_case_id}: semantic hard gate failed; "
            "fresh E2 output is not eligible for gold registration"
        )

    if scenario == "candidate":
        if not portfolio.hypotheses:
            raise ValueError(
                f"{record_case_id}: candidate scenario unexpectedly abstained"
            )
        if not any(
            row.candidate_dependency != "none"
            for row in portfolio.hypotheses
        ):
            raise ValueError(
                f"{record_case_id}: candidate scenario did not preserve a "
                "candidate-dependent hypothesis"
            )
        return

    if scenario == "alignment":
        if not portfolio.hypotheses:
            raise ValueError(
                f"{record_case_id}: alignment scenario unexpectedly abstained"
            )
        if not any(row.cross_paper_synthesis for row in portfolio.hypotheses):
            raise ValueError(
                f"{record_case_id}: alignment scenario did not produce a "
                "cross-paper hypothesis"
            )
        if not any(route.uses_alignment for route in context.mechanism_routes):
            raise ValueError(
                f"{record_case_id}: alignment scenario context has no "
                "alignment-bearing route"
            )
        return

    if scenario == "partial":
        if not portfolio.hypotheses:
            raise ValueError(
                f"{record_case_id}: partial-source scenario unexpectedly abstained"
            )
        if "Kiwook_10" not in context.partial_absence_blocked_paper_ids:
            raise ValueError(
                f"{record_case_id}: partial-source safety block is missing Kiwook_10"
            )
        if not any(
            "Kiwook_10" in row.source_paper_ids
            for row in portfolio.hypotheses
        ):
            raise ValueError(
                f"{record_case_id}: partial-source output did not use positive "
                "Kiwook_10 evidence"
            )
        return

    if scenario == "abstention":
        if portfolio.hypotheses:
            raise ValueError(
                f"{record_case_id}: abstention scenario generated hypotheses"
            )
        if not (portfolio.abstention_reason or "").strip():
            raise ValueError(
                f"{record_case_id}: abstention scenario lacks abstention_reason"
            )
        if any(row.eligible_as_premise for row in context.evidence_statements):
            raise ValueError(
                f"{record_case_id}: abstention scenario unexpectedly contains "
                "an eligible positive premise"
            )
        return

    raise ValueError(f"{record_case_id}: unknown E2 scenario {scenario!r}")


def validate_required_e2_records(
    records: list[HypothesisE2OutputRecord],
) -> None:
    scenarios = [row.scenario for row in records]
    missing = sorted(REQUIRED_E2_SCENARIOS - set(scenarios))
    duplicate = sorted(
        scenario
        for scenario in REQUIRED_E2_SCENARIOS
        if scenarios.count(scenario) > 1
    )
    extra = sorted(set(scenarios) - REQUIRED_E2_SCENARIOS)
    if missing or duplicate or extra or len(records) != len(REQUIRED_E2_SCENARIOS):
        raise ValueError(
            "E2 requires exactly one fresh output for each controlled scenario; "
            f"missing={missing}, duplicate={duplicate}, extra={extra}"
        )


def worksheet_to_combined_spec(
    *,
    base_spec: HypothesisRealGoldSpec,
    worksheet: HypothesisE2HumanReviewWorksheet,
    output_suite_id: str,
) -> HypothesisRealGoldSpec:
    if len(worksheet.cases) != 4:
        raise ValueError(
            "E2 finalization requires exactly four reviewed live-output cases"
        )

    scenarios = [row.scenario for row in worksheet.cases]
    missing = sorted(REQUIRED_E2_SCENARIOS - set(scenarios))
    duplicate = sorted(
        scenario
        for scenario in REQUIRED_E2_SCENARIOS
        if scenarios.count(scenario) > 1
    )
    if missing or duplicate:
        raise ValueError(
            "E2 worksheet scenario coverage mismatch; "
            f"missing={missing}, duplicate={duplicate}"
        )

    additions: list[HypothesisRealGoldCaseSpec] = []
    for case in worksheet.cases:
        if case.approval_status != "approved":
            raise ValueError(
                f"{case.case_id}: human review is still {case.approval_status!r}; "
                "set approval_status='approved' only after reviewing all 11 dimensions"
            )

        expectations: list[SemanticGoldExpectation] = []
        for dimension in case.dimensions:
            if not dimension.human_allowed_verdicts:
                raise ValueError(
                    f"{case.case_id}/{dimension.dimension}: "
                    "human_allowed_verdicts is empty"
                )
            expectations.append(
                SemanticGoldExpectation(
                    dimension=dimension.dimension,
                    allowed_verdicts=dimension.human_allowed_verdicts,
                    critical=dimension.critical,
                    note=dimension.human_note or (
                        "Human-reviewed E2 live-output expectation. "
                        f"Critic scaffold verdict was {dimension.critic_verdict}."
                    ),
                )
            )

        additions.append(
            HypothesisRealGoldCaseSpec(
                case_id=case.case_id,
                description=case.description,
                context_path=case.context_path,
                portfolio_path=case.portfolio_path,
                expectations=expectations,
                forbid_unexpected_failures=True,
                allowed_additional_fail_dimensions=[],
                generator_version=case.generator_version,
                note=(
                    "E2 controlled-context live Hypothesis Maker output; "
                    "not additional corpus evidence. " + case.review_hint
                ),
            )
        )

    combined = [*base_spec.cases, *additions]
    ids = [row.case_id for row in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("combined real-output gold spec contains duplicate case_id")

    return HypothesisRealGoldSpec(
        suite_id=output_suite_id,
        cases=combined,
    )
