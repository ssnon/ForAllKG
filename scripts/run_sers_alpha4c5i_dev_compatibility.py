from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dac_her.alpha4c5i_dev_compatibility import (
    ALPHA4C5I_DEV_COMPAT_SEMANTICS_ID,
    DEFAULT_H1_DEV_SUMMARY,
    DEFAULT_OUTPUT_ROOT,
    atomic_json,
    deterministic_downstream_probe,
    discover_exact_dev_inputs,
    load_exact_dev_paper_ids,
    make_structural_summary,
    run_h1_dev_compatibility,
    select_dev_input,
    verify_5i_component_hashes,
    verify_frozen_postmortem,
    verify_h1_dev_summary,
    write_preview_artifacts,
)


ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run alpha4c.5i DEV-only downstream compatibility. "
            "Default --run verifies the existing zero-LLM upstream DEV summary, "
            "exact DEV53 input binding, hardened exposure/prompt build, "
            "and empty-abstention compile/validate through the existing "
            "5d.1 downstream contract. --run-maker is a separate DEV-only "
            "LLM opt-in and never reads Reserve A/B."
        )
    )
    parser.add_argument("--trend-input", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--run-maker", action="store_true")

    parser.add_argument(
        "--confirm-development-only",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-llm-development-run",
        action="store_true",
    )

    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_TREND_HYPOTHESIS_MODEL")
            or os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-hypotheses", type=int, default=3)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _base_preflight(
    *,
    trend_input: Path | None,
) -> tuple[list[str], list[str], Path | None]:
    issues: list[str] = []
    issues.extend(verify_frozen_postmortem(ROOT))
    issues.extend(verify_5i_component_hashes(ROOT))
    issues.extend(verify_h1_dev_summary(ROOT))

    dev_papers: list[str] = []
    try:
        dev_papers = load_exact_dev_paper_ids(ROOT)
    except Exception as exc:
        issues.append(f"DEV split binding: {exc}")

    selected: Path | None = None
    if dev_papers:
        try:
            if trend_input is not None:
                selected, _ = select_dev_input(
                    root=ROOT,
                    dev_paper_ids=dev_papers,
                    explicit_path=trend_input,
                )
            else:
                candidates = discover_exact_dev_inputs(
                    root=ROOT,
                    dev_paper_ids=dev_papers,
                )
                if len(candidates) == 1:
                    selected = candidates[0]
                elif len(candidates) == 0:
                    issues.append(
                        "Exact DEV53 Trend input candidates: 0"
                    )
                else:
                    issues.append(
                        "Exact DEV53 Trend input candidates: "
                        + str(len(candidates))
                        + "; use --trend-input to select explicitly"
                    )
                    print("Exact DEV53 candidates:")
                    for path in candidates:
                        print(" -", path)
        except Exception as exc:
            issues.append(f"DEV input binding: {exc}")

    return issues, dev_papers, selected


def _run_maker(
    *,
    input_path: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> int:
    if not args.confirm_development_only:
        raise SystemExit(
            "--confirm-development-only is required."
        )
    if not args.confirm_llm_development_run:
        raise SystemExit(
            "--confirm-llm-development-run is required for --run-maker."
        )
    if not args.model:
        raise SystemExit(
            "--model or a configured hypothesis-model environment "
            "variable is required for --run-maker."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    prefix = output_root / "maker" / "dev_maker"
    telemetry = output_root / "maker" / "telemetry.jsonl"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_contract_hardened_trend_hypothesis_maker",
        "--input",
        str(input_path),
        "--output-prefix",
        str(prefix),
        "--model",
        str(args.model),
        "--api-key-env",
        args.api_key_env,
        "--instructor-mode",
        args.instructor_mode,
        "--temperature",
        str(args.temperature),
        "--parse-retries",
        str(args.parse_retries),
        "--max-repairs",
        "1",
        "--timeout",
        str(args.timeout),
        "--max-hypotheses",
        str(args.max_hypotheses),
        "--telemetry-path",
        str(telemetry),
        "--save-prompt",
    ]
    if args.base_url:
        cmd.extend(["--base-url", str(args.base_url)])

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    run_path = Path(str(prefix) + ".run.json")
    run_record = (
        json.loads(run_path.read_text(encoding="utf-8"))
        if run_path.is_file()
        else {}
    )
    summary = {
        "schema_version":
            "sers-alpha4c5i-dev-maker-attempt-v1",
        "semantics_id": ALPHA4C5I_DEV_COMPAT_SEMANTICS_ID,
        "development_only": True,
        "source_input_path":
            str(input_path.relative_to(ROOT)),
        "returncode": result.returncode,
        "accepted": result.returncode == 0,
        "generation_attempts":
            run_record.get("generation_attempts"),
        "repair_attempts":
            run_record.get("repair_attempts"),
        "failure_stage": run_record.get("failure_stage"),
        "validation_errors":
            run_record.get("validation_errors"),
        "validation_warnings":
            run_record.get("validation_warnings"),
        "closed_reserve_a_used": False,
        "closed_reserve_b_used": False,
        "reserve_b_rerun": False,
        "development_tuning_allowed": True,
    }
    atomic_json(
        output_root / "maker_attempt_summary.json",
        summary,
    )
    print("Maker attempt summary:",
          output_root / "maker_attempt_summary.json")
    return result.returncode


def main() -> int:
    args = parse_args()
    output_root = _resolve(args.output_root)
    explicit = (
        _resolve(args.trend_input)
        if args.trend_input is not None
        else None
    )

    issues, dev_papers, selected = _base_preflight(
        trend_input=explicit,
    )

    print("alpha4c.5i DEV downstream compatibility")
    print("Semantics:", ALPHA4C5I_DEV_COMPAT_SEMANTICS_ID)
    print("DEV papers:", len(dev_papers))
    print("Reserve A used: False")
    print("Reserve B used: False")
    print("Reserve B rerun: False")
    print("Default LLM calls: 0")
    print(
        "Selected DEV Trend input:",
        selected if selected is not None else "NONE",
    )

    if args.preflight:
        if issues:
            print("Preflight: FAIL")
            for issue in issues:
                print(" -", issue)
            return 2
        print("5h.2 postmortem: CURRENT")
        print("5i component hashes: CURRENT")
        print("5h.1 DEV summary: PASS + CURRENT")
        print("Exact DEV53 input binding: PASS")
        print("Preflight: PASS")
        print("Write performed: False")
        print("LLM calls: 0")
        return 0

    if not args.confirm_development_only:
        raise SystemExit(
            "--confirm-development-only is required for DEV execution."
        )
    if issues:
        print("Compatibility readiness: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2
    assert selected is not None

    # Reuse and re-verify the existing frozen/current 5h.1 DEV diagnostic; do not rerun or overwrite it.
    h1_counts, _ = run_h1_dev_compatibility(ROOT)
    h1_issues = verify_h1_dev_summary(ROOT)
    if h1_issues:
        print("5h.1 DEV summary verification after reuse: FAIL")
        for issue in h1_issues:
            print(" -", issue)
        return 2

    _, source = select_dev_input(
        root=ROOT,
        dev_paper_ids=dev_papers,
        explicit_path=selected,
    )

    probe = deterministic_downstream_probe(source)
    preview = write_preview_artifacts(
        root=ROOT,
        output_root=output_root / "deterministic",
        input_path=selected,
        source=source,
    )
    summary = make_structural_summary(
        root=ROOT,
        input_path=selected,
        source=source,
        h1_counts=h1_counts,
        probe=probe,
        preview=preview,
    )
    atomic_json(output_root / "summary.json", summary)

    print("DEV upstream Trend evidence:", h1_counts["trend_evidence_count"])
    print(
        "DEV upstream Precision local results:",
        h1_counts["precision_count"],
    )
    print(
        "DEV upstream CrossContext:",
        h1_counts["cross_context_count"],
    )
    print(
        "DEV upstream Grounding:",
        h1_counts["grounding_count"],
    )
    print(
        "Hardened views:",
        probe["hardened_view_count"],
    )
    print(
        "Empty-abstention compile/validate:",
        probe["empty_abstention_validation_passed"],
    )
    print("Scientific hypotheses generated by probe: 0")
    print("Count thresholds used: False")
    print("Reserve A/B used: False")
    print("LLM calls: 0")
    print(
        "Deterministic downstream compatibility: PASS"
    )
    print("Summary:", output_root / "summary.json")

    if args.run:
        return 0

    # --run-maker continues only after the deterministic gate passes.
    return _run_maker(
        input_path=selected,
        output_root=output_root,
        args=args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
