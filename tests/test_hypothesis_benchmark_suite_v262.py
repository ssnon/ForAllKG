from __future__ import annotations

from pathlib import Path

from dac_her.hypothesis_benchmark_suite import HypothesisBenchmarkSuiteRunner
from scripts.build_hypothesis_v262_fixtures import build_cases


def test_generated_suite_matches_all_expectations(tmp_path: Path):
    cases = build_cases(tmp_path)
    suite = {
        "schema_version": "hypothesis-benchmark-suite-v262",
        "suite_id": "test-suite",
        "evaluator_version": "hypothesis-benchmark-evaluator-v2.6.2-a1",
        "cases": cases,
    }
    import json

    suite_path = tmp_path / "suite_v262.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    result = HypothesisBenchmarkSuiteRunner().run_file(suite_path)

    assert result.passed
    assert result.failed_cases == 0
    assert result.passed_cases == len(cases)
