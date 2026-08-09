from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_benchmark_suite import (
    HypothesisBenchmarkSuiteRunner,
    render_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Hypothesis Maker v2.6.2 benchmark suite."
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite)
    result = HypothesisBenchmarkSuiteRunner().run_file(suite_path)

    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else suite_path.with_name(suite_path.stem + ".result")
    )
    json_path = Path(str(prefix) + ".json")
    md_path = Path(str(prefix) + ".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")

    print("Hypothesis benchmark complete")
    print("Suite:", result.suite_id)
    print("Evaluator:", result.evaluator_version)
    print("Overall:", "PASS" if result.passed else "FAIL")
    print("Cases:", result.passed_cases, "passed /", result.failed_cases, "failed")
    for row in result.case_results:
        print(
            f"- {row.case_id}: "
            f"{'PASS' if row.expectation_passed else 'FAIL'} "
            f"(hard={'PASS' if row.report.hard_gate_passed else 'FAIL'}, "
            f"diagnostics={len(row.report.diagnostics)})"
        )
        for failure in row.expectation_failures:
            print("  expectation:", failure)
    print("Saved:", json_path)
    print("Saved:", md_path)

    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
