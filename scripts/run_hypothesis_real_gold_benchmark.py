from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_gold_comparator import HypothesisSemanticGoldComparator
from dac_her.hypothesis_real_gold import (
    to_semantic_gold_suite,
    validate_real_gold_lineage,
)
from dac_her.hypothesis_real_gold_contracts import HypothesisRealGoldSuite
from dac_her.hypothesis_semantic_llm import (
    InstructorOpenAICompatibleSemanticCriticBackend,
)
from dac_her.hypothesis_semantic_runtime import HypothesisSemanticCriticRuntime


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_markdown(report, lineage) -> str:
    lines = [
        f"# Real-output Semantic Gold Comparison {report.suite_id}",
        "",
        f"- Lineage preflight: **{'PASS' if lineage.passed else 'FAIL'}**",
        f"- Overall: **{'PASS' if report.passed else 'FAIL'}**",
        f"- Cases: {report.passed_cases} passed / {report.failed_cases} failed",
        f"- Critical mismatches: {report.critical_mismatches}",
        f"- Noncritical mismatches: {report.noncritical_mismatches}",
        f"- Missing reviews: {report.missing_reviews}",
        "",
        "| Case | Result | Agreements | Mismatches |",
        "|---|---:|---:|---|",
    ]
    for row in report.case_results:
        mismatch = ", ".join(
            f"{m.dimension}:{m.actual_verdict or 'missing'}"
            for m in row.mismatches
        ) or "-"
        lines.append(
            f"| `{row.case_id}` | {'PASS' if row.passed else 'FAIL'} | "
            f"{row.exact_or_allowed_agreements} | {mismatch} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live semantic critic reviews against provenance-frozen "
            "real Hypothesis Maker outputs."
        )
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/hypothesis_v262/real_runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL "
            "or OPENROUTER_AGENT_MODEL is set."
        )

    gold_path = Path(args.gold).resolve()
    gold_base = gold_path.parent
    suite = HypothesisRealGoldSuite.model_validate_json(
        gold_path.read_text(encoding="utf-8")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lineage = validate_real_gold_lineage(suite, suite_path=gold_path)
    _write_json(out_dir / "lineage_preflight.json", lineage)
    if not lineage.passed:
        print("Real-output gold lineage preflight: FAIL")
        for row in lineage.case_results:
            if row.passed:
                continue
            print("-", row.case_id)
            for issue in row.issues:
                print("   ", issue.code, "-", issue.message)
        print("No semantic critic calls were made.")
        raise SystemExit(3)

    backend = InstructorOpenAICompatibleSemanticCriticBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
    )
    runtime = HypothesisSemanticCriticRuntime(backend)
    reviews = {}

    for case in suite.cases:
        context_path = Path(case.context_path)
        portfolio_path = Path(case.portfolio_path)
        if not context_path.is_absolute():
            context_path = (gold_base / context_path).resolve()
        if not portfolio_path.is_absolute():
            portfolio_path = (gold_base / portfolio_path).resolve()

        context = HypothesisContext.model_validate_json(
            context_path.read_text(encoding="utf-8")
        )
        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )
        outcome = runtime.run(context, portfolio)

        case_dir = out_dir / case.case_id
        _write_json(case_dir / "hard_evaluation.json", outcome.evaluation)
        _write_json(case_dir / "run.json", outcome.run_record)
        if outcome.generation is not None:
            _write_json(case_dir / "draft.json", outcome.generation.draft)
        if outcome.review is not None:
            _write_json(case_dir / "review.json", outcome.review)
        reviews[case.case_id] = outcome.review

        print(
            case.case_id,
            "accepted=" + str(outcome.accepted),
            "hard=" + str(outcome.evaluation.hard_gate_passed),
        )

    semantic_suite = to_semantic_gold_suite(suite)
    comparison = HypothesisSemanticGoldComparator().compare(
        semantic_suite,
        reviews,
    )
    _write_json(out_dir / "real_gold_comparison.json", comparison)
    (out_dir / "real_gold_comparison.md").write_text(
        _render_markdown(comparison, lineage),
        encoding="utf-8",
    )

    print("Real-output semantic gold benchmark complete")
    print("Lineage preflight: PASS")
    print("Overall:", "PASS" if comparison.passed else "FAIL")
    print(
        "Cases:",
        comparison.passed_cases,
        "passed /",
        comparison.failed_cases,
        "failed",
    )
    print("Critical mismatches:", comparison.critical_mismatches)
    print("Noncritical mismatches:", comparison.noncritical_mismatches)
    print("Missing reviews:", comparison.missing_reviews)
    print("Saved:", out_dir / "real_gold_comparison.json")
    print("Saved:", out_dir / "real_gold_comparison.md")
    if not comparison.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
