from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from dac_her.explorer_benchmark_contracts import (
    BenchmarkCaseResult,
    BenchmarkSuite,
    BenchmarkSuiteResult,
)
from dac_her.explorer_evaluation import evaluate_case


def _load_suite(path: Path) -> BenchmarkSuite:
    with path.open("r", encoding="utf-8") as f:
        return BenchmarkSuite.model_validate(json.load(f))


def _run_case(
    *,
    case,
    repo_root: Path,
    model: str,
    base_url: str,
    api_key_env: str,
    save_prompt: bool,
) -> int:
    packet = (repo_root / case.packet).resolve()
    prefix = (repo_root / case.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        "-m",
        "scripts.run_graph_explorer",
        "--packet",
        str(packet),
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key-env",
        api_key_env,
        "--output-prefix",
        str(prefix),
    ]
    if save_prompt:
        cmd.append("--save-prompt")
    print("[benchmark execute]", case.case_id)
    print(" ", " ".join(cmd))
    return subprocess.run(cmd, cwd=repo_root).returncode


def _write_markdown(result: BenchmarkSuiteResult, path: Path) -> None:
    lines = [
        f"# Graph Explorer benchmark: {result.suite_id}",
        "",
        f"Overall: **{'PASS' if result.passes else 'FAIL'}**",
        "",
        "| Case | Result | Repairs | Reported | Synthesis | Unresolved | Routes | Candidate violations | Partial absence | Alignment causal risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cr in result.case_results:
        m = cr.metrics
        status = "SKIP" if cr.skipped else ("PASS" if cr.passes else "FAIL")
        lines.append(
            "| {case} | {status} | {repair} | {reported} | {synth} | {unres} | {routes} | {cand} | {partial} | {align} |".format(
                case=cr.case_id,
                status=status,
                repair=m.get("repair_attempts", "-"),
                reported=m.get("reported_statement_count", "-"),
                synth=m.get("synthesis_statement_count", "-"),
                unres=m.get("unresolved_connection_count", "-"),
                routes=m.get("mechanism_route_count", "-"),
                cand=m.get("candidate_verification_violations", "-"),
                partial=m.get("partial_absence_violations", "-"),
                align=m.get("alignment_causal_risk_count", "-"),
            )
        )
        if cr.issues:
            lines.append("")
            for issue in cr.issues:
                lines.append(f"- **{cr.case_id} / {issue.severity.upper()} / {issue.code}**: {issue.message}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Run/evaluate Graph Explorer v2.5.2 benchmark suite")
    p.add_argument("--suite", default="benchmarks/graph_explorer/suite_v252.json")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--execute", action="store_true", help="run Graph Explorer before evaluating each enabled case")
    p.add_argument("--model", default=os.getenv("GRAPHAGENTS_EXPLORER_MODEL") or os.getenv("OPENROUTER_AGENT_MODEL"))
    p.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--save-prompt", action="store_true")
    p.add_argument("--allow-missing", action="store_true", help="skip cases whose artifacts are missing")
    p.add_argument("--output-json", default="data_dac/explorer_benchmark/v252.result.json")
    p.add_argument("--output-md", default="data_dac/explorer_benchmark/v252.result.md")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    suite_path = (repo_root / args.suite).resolve()
    suite = _load_suite(suite_path)

    if args.execute and not args.model:
        p.error("--execute requires --model or GRAPHAGENTS_EXPLORER_MODEL/OPENROUTER_AGENT_MODEL")

    case_results: List[BenchmarkCaseResult] = []
    for case in suite.cases:
        if not case.enabled:
            case_results.append(BenchmarkCaseResult(case_id=case.case_id, passes=True, skipped=True))
            continue

        packet_path = (repo_root / case.packet).resolve()
        if args.execute:
            if not packet_path.exists():
                if args.allow_missing:
                    case_results.append(BenchmarkCaseResult(case_id=case.case_id, passes=True, skipped=True))
                    continue
                print(f"[error] missing packet for {case.case_id}: {packet_path}", file=sys.stderr)
            else:
                rc = _run_case(
                    case=case,
                    repo_root=repo_root,
                    model=args.model,
                    base_url=args.base_url,
                    api_key_env=args.api_key_env,
                    save_prompt=args.save_prompt,
                )
                if rc not in (0, 2):
                    print(f"[warning] explorer process returned {rc} for {case.case_id}", file=sys.stderr)

        prefix = (repo_root / case.output_prefix).resolve()
        required_paths = [
            packet_path,
            Path(str(prefix) + ".report.json"),
            Path(str(prefix) + ".run.json"),
            Path(str(prefix) + ".validation.json"),
        ]
        if args.allow_missing and any(not x.exists() for x in required_paths):
            case_results.append(BenchmarkCaseResult(case_id=case.case_id, passes=True, skipped=True))
            continue

        cr = evaluate_case(case, repo_root=repo_root)
        case_results.append(cr)
        print(
            f"[{cr.case_id}] {'PASS' if cr.passes else 'FAIL'} "
            f"errors={sum(i.severity == 'error' for i in cr.issues)} "
            f"warnings={sum(i.severity == 'warning' for i in cr.issues)}"
        )

    evaluated = sum(not x.skipped for x in case_results)
    passed = sum((not x.skipped) and x.passes for x in case_results)
    failed = sum((not x.skipped) and (not x.passes) for x in case_results)
    skipped = sum(x.skipped for x in case_results)
    result = BenchmarkSuiteResult(
        suite_id=suite.suite_id,
        passes=failed == 0 and evaluated > 0,
        evaluated_cases=evaluated,
        passed_cases=passed,
        failed_cases=failed,
        skipped_cases=skipped,
        case_results=case_results,
    )

    output_json = (repo_root / args.output_json).resolve()
    output_md = (repo_root / args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    _write_markdown(result, output_md)
    print(f"Saved: {output_json}")
    print(f"Saved: {output_md}")
    print(
        f"Suite {suite.suite_id}: {'PASS' if result.passes else 'FAIL'} "
        f"(passed={passed}, failed={failed}, skipped={skipped})"
    )
    return 0 if result.passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
