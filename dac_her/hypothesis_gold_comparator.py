from __future__ import annotations

from dac_her.hypothesis_gold_contracts import (
    SemanticGoldCaseComparison,
    SemanticGoldComparisonReport,
    SemanticGoldMismatch,
    SemanticGoldSuite,
)
from dac_her.hypothesis_semantic_contracts import HypothesisSemanticReview


class HypothesisSemanticGoldComparator:
    def compare(
        self,
        suite: SemanticGoldSuite,
        reviews: dict[str, HypothesisSemanticReview | None],
    ) -> SemanticGoldComparisonReport:
        results: list[SemanticGoldCaseComparison] = []

        for case in suite.cases:
            review = reviews.get(case.case_id)
            mismatches: list[SemanticGoldMismatch] = []
            agreements = 0

            if review is None:
                for expectation in case.expectations:
                    mismatches.append(
                        SemanticGoldMismatch(
                            case_id=case.case_id,
                            dimension=expectation.dimension,
                            allowed_verdicts=expectation.allowed_verdicts,
                            actual_verdict=None,
                            critical=True,
                            kind="missing_review",
                            note=expectation.note,
                        )
                    )
            else:
                actual = {row.dimension: row.verdict for row in review.dimensions}
                expected_dimensions = {
                    expectation.dimension
                    for expectation in case.expectations
                }
                for expectation in case.expectations:
                    verdict = actual.get(expectation.dimension)
                    if verdict is None:
                        mismatches.append(
                            SemanticGoldMismatch(
                                case_id=case.case_id,
                                dimension=expectation.dimension,
                                allowed_verdicts=expectation.allowed_verdicts,
                                actual_verdict=None,
                                critical=expectation.critical,
                                kind="missing_dimension",
                                note=expectation.note,
                            )
                        )
                    elif verdict in expectation.allowed_verdicts:
                        agreements += 1
                    else:
                        mismatches.append(
                            SemanticGoldMismatch(
                                case_id=case.case_id,
                                dimension=expectation.dimension,
                                allowed_verdicts=expectation.allowed_verdicts,
                                actual_verdict=verdict,
                                critical=expectation.critical,
                                kind=(
                                    "critical_mismatch"
                                    if expectation.critical
                                    else "noncritical_mismatch"
                                ),
                                note=expectation.note,
                            )
                        )
                        
            if review is not None and case.forbid_unexpected_failures:
                for dimension, verdict in actual.items():
                    if dimension in expected_dimensions:
                        continue

                    if verdict == "fail":
                        mismatches.append(
                            SemanticGoldMismatch(
                                case_id=case.case_id,
                                dimension=dimension,
                                allowed_verdicts=["pass", "warning", "not_applicable"],
                                actual_verdict="fail",
                                critical=True,
                                kind="unexpected_failure",
                                note=(
                                    "Critic produced an unanticipated semantic FAIL "
                                    "on a dimension not marked problematic by the gold case."
                                ),
                            )
                        )

            case_passed = not any(
                row.critical or row.kind in {"missing_review", "missing_dimension"}
                for row in mismatches
            )
            results.append(
                SemanticGoldCaseComparison(
                    case_id=case.case_id,
                    passed=case_passed,
                    exact_or_allowed_agreements=agreements,
                    mismatches=mismatches,
                )
            )

        critical = sum(
            row.critical
            for result in results
            for row in result.mismatches
            if row.kind not in {"missing_review"}
        )
        noncritical = sum(
            not row.critical
            for result in results
            for row in result.mismatches
        )
        missing_reviews = sum(
            row.kind == "missing_review"
            for result in results
            for row in result.mismatches
        )
        failed = sum(not row.passed for row in results)
        return SemanticGoldComparisonReport(
            suite_id=suite.suite_id,
            passed=failed == 0,
            case_count=len(results),
            passed_cases=len(results) - failed,
            failed_cases=failed,
            critical_mismatches=critical,
            noncritical_mismatches=noncritical,
            missing_reviews=missing_reviews,
            case_results=results,
        )
