from __future__ import annotations

from pathlib import Path

from dac_her.hypothesis_benchmark_contracts import (
    HypothesisBenchmarkCaseResult,
    HypothesisBenchmarkSuite,
    HypothesisBenchmarkSuiteResult,
)
from dac_her.hypothesis_benchmark_evaluator import HypothesisBenchmarkEvaluator
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio


def _read_context(path: Path) -> HypothesisContext:
    return HypothesisContext.model_validate_json(path.read_text(encoding="utf-8"))


def _read_portfolio(path: Path) -> HypothesisPortfolio:
    return HypothesisPortfolio.model_validate_json(path.read_text(encoding="utf-8"))


def _expectation_failures(case, report, portfolio) -> list[str]:
    failures: list[str] = []
    expectation = case.expectation
    hard_codes = {issue.code for issue in report.hard_gate_issues}
    diagnostic_codes = {issue.code for issue in report.diagnostics}

    if report.hard_gate_passed != expectation.hard_gate_pass:
        failures.append(
            f"hard_gate_pass expected={expectation.hard_gate_pass} "
            f"actual={report.hard_gate_passed}"
        )

    if expectation.expected_abstention is not None:
        abstained = not portfolio.hypotheses
        if abstained != expectation.expected_abstention:
            failures.append(
                f"abstention expected={expectation.expected_abstention} actual={abstained}"
            )

    for code in expectation.required_issue_codes:
        if code not in hard_codes:
            failures.append(f"missing required hard-gate issue: {code}")
    for code in expectation.forbidden_issue_codes:
        if code in hard_codes:
            failures.append(f"forbidden hard-gate issue present: {code}")
    for code in expectation.required_diagnostic_codes:
        if code not in diagnostic_codes:
            failures.append(f"missing required diagnostic: {code}")
    for code in expectation.forbidden_diagnostic_codes:
        if code in diagnostic_codes:
            failures.append(f"forbidden diagnostic present: {code}")

    return failures


class HypothesisBenchmarkSuiteRunner:
    def __init__(
        self,
        *,
        evaluator: HypothesisBenchmarkEvaluator | None = None,
    ) -> None:
        self.evaluator = evaluator or HypothesisBenchmarkEvaluator()

    def run_file(self, suite_path: str | Path) -> HypothesisBenchmarkSuiteResult:
        suite_path = Path(suite_path)
        suite = HypothesisBenchmarkSuite.model_validate_json(
            suite_path.read_text(encoding="utf-8")
        )
        base = suite_path.parent
        results: list[HypothesisBenchmarkCaseResult] = []

        for case in suite.cases:
            context_path = (base / case.context_path).resolve()
            portfolio_path = (base / case.portfolio_path).resolve()
            context = _read_context(context_path)
            portfolio = _read_portfolio(portfolio_path)
            report = self.evaluator.evaluate(context, portfolio)
            failures = _expectation_failures(case, report, portfolio)
            results.append(
                HypothesisBenchmarkCaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    expectation_passed=not failures,
                    expectation_failures=failures,
                    report=report,
                )
            )

        failed = sum(not result.expectation_passed for result in results)
        return HypothesisBenchmarkSuiteResult(
            suite_id=suite.suite_id,
            evaluator_version=suite.evaluator_version,
            passed=failed == 0,
            passed_cases=len(results) - failed,
            failed_cases=failed,
            case_results=results,
        )


def render_markdown(result: HypothesisBenchmarkSuiteResult) -> str:
    lines = [
        f"# Hypothesis Benchmark {result.suite_id}",
        "",
        f"- Evaluator: `{result.evaluator_version}`",
        f"- Overall: **{'PASS' if result.passed else 'FAIL'}**",
        f"- Cases: {result.passed_cases} passed / {result.failed_cases} failed",
        "",
        "| Case | Category | Expectation | Hard gate | Diagnostics |",
        "|---|---|---:|---:|---|",
    ]
    for row in result.case_results:
        diagnostics = ", ".join(issue.code for issue in row.report.diagnostics) or "-"
        lines.append(
            f"| `{row.case_id}` | {row.category} | "
            f"{'PASS' if row.expectation_passed else 'FAIL'} | "
            f"{'PASS' if row.report.hard_gate_passed else 'FAIL'} | {diagnostics} |"
        )
        if row.expectation_failures:
            for failure in row.expectation_failures:
                lines.append(f"\n  - `{row.case_id}`: {failure}")
    lines.append("")
    return "\n".join(lines)
