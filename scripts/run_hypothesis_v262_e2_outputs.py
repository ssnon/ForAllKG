from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from dac_her.hypothesis_benchmark_evaluator import HypothesisBenchmarkEvaluator
from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.hypothesis_e2 import (
    sha256_file,
    validate_required_e2_records,
    validate_scenario_postcondition,
)
from dac_her.hypothesis_e2_contracts import (
    HypothesisE2ContextManifest,
    HypothesisE2OutputManifest,
    HypothesisE2OutputRecord,
)
from dac_her.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from dac_her.hypothesis_prompt import HypothesisPromptAssembler
from dac_her.hypothesis_runtime import HypothesisMakerAgentRuntime


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Header must be KEY=VALUE")
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Header key may not be empty")
    return key, item


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _repo_relative(path: Path, repo: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=repo.resolve())).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate four fresh live Hypothesis Maker outputs for the "
            "v2.6.2 E2 controlled scenarios."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data_dac/hypothesis_e2/e2_context_manifest.json",
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--output-dir",
        default="data_dac/hypothesis_e2/live_outputs",
    )
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--max-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-hypotheses", type=int, default=2)
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_header,
        metavar="KEY=VALUE",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL "
            "or OPENROUTER_AGENT_MODEL is set."
        )

    repo = Path(args.repo).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest_base = manifest_path.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = HypothesisE2ContextManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    assembler = HypothesisPromptAssembler(max_hypotheses=args.max_hypotheses)
    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
    )
    runtime = HypothesisMakerAgentRuntime(
        backend,
        prompt_assembler=assembler,
        max_repairs=args.max_repairs,
    )
    evaluator = HypothesisBenchmarkEvaluator()
    records: list[HypothesisE2OutputRecord] = []

    for case in manifest.cases:
        context_path = Path(case.context_path)
        if not context_path.is_absolute():
            context_path = (manifest_base / context_path).resolve()
        context = HypothesisContext.model_validate_json(
            context_path.read_text(encoding="utf-8")
        )

        case_dir = output_dir / case.case_id
        if case_dir.exists() and any(case_dir.iterdir()):
            if not args.force:
                raise SystemExit(
                    f"{case.case_id}: output directory already exists and is non-empty: "
                    f"{case_dir}. Use --force only when a fresh rerun is intentional."
                )
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        prompt = assembler.build(context)
        prompt_path = case_dir / "prompt.txt"
        prompt_path.write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )

        outcome = runtime.run(context)
        draft_paths: list[Path] = []
        for index, draft in enumerate(outcome.draft_history):
            path = case_dir / (
                "draft.json" if index == 0 else f"repair{index}.draft.json"
            )
            _write_json(path, draft)
            draft_paths.append(path)

        run_path = case_dir / "run.json"
        validation_path = case_dir / "validation.json"
        _write_json(run_path, outcome.run_record)

        if outcome.validation is not None:
            _write_json(validation_path, outcome.validation)
        else:
            _write_json(
                validation_path,
                {
                    "passes": False,
                    "stage": "compile",
                    "issues": [
                        issue.model_dump(mode="json")
                        for issue in outcome.compile_issues
                    ],
                },
            )

        if outcome.accepted_portfolio is None:
            raise SystemExit(
                f"{case.case_id}: Hypothesis Maker did not accept a portfolio; "
                f"failure_stage={outcome.run_record.failure_stage}"
            )

        portfolio = outcome.accepted_portfolio
        portfolio_path = case_dir / "portfolio.json"
        _write_json(portfolio_path, portfolio)

        evaluation = evaluator.evaluate(context, portfolio)
        hard_path = case_dir / "hard_evaluation.json"
        _write_json(hard_path, evaluation)

        try:
            validate_scenario_postcondition(
                case.case_id,
                case.scenario,
                context,
                portfolio,
                evaluation,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        record = HypothesisE2OutputRecord(
            case_id=case.case_id,
            scenario=case.scenario,
            description=case.description,
            review_hint=case.review_hint,
            context_path=_repo_relative(context_path, repo),
            portfolio_path=_repo_relative(portfolio_path, repo),
            run_path=_repo_relative(run_path, repo),
            validation_path=_repo_relative(validation_path, repo),
            hard_evaluation_path=_repo_relative(hard_path, repo),
            prompt_path=_repo_relative(prompt_path, repo),
            draft_paths=[
                _repo_relative(path, repo)
                for path in draft_paths
            ],
            generator_version="hypothesis-maker-v2.6.1",
            prompt_version=outcome.run_record.prompt_version,
            prompt_sha256=outcome.run_record.prompt_sha256,
            context_file_sha256=sha256_file(context_path),
            portfolio_file_sha256=sha256_file(portfolio_path),
            accepted=True,
            hard_gate_passed=True,
            scenario_postcondition_passed=True,
            abstained=not bool(portfolio.hypotheses),
        )
        records.append(record)
        print(
            case.case_id,
            "scenario=" + case.scenario,
            "accepted=True",
            "hard=True",
            "abstained=" + str(record.abstained),
        )

    validate_required_e2_records(records)
    output_manifest = HypothesisE2OutputManifest(
        suite_id="hypothesis-v262-e2-fresh-live-outputs",
        cases=records,
    )
    manifest_out = output_dir / "e2_output_manifest.json"
    _write_json(manifest_out, output_manifest)

    print("Hypothesis v2.6.2 E2 fresh live outputs complete")
    print("Cases:", len(records))
    print("Saved:", manifest_out)


if __name__ == "__main__":
    main()
