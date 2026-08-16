from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.hypothesis_trend_contract_hardened_exposure import (
    build_contract_hardened_trend_maker_exposure,
)
from dac_her.hypothesis_trend_contract_hardened_llm import (
    InstructorOpenAICompatibleContractHardenedTrendBackend,
)
from dac_her.hypothesis_trend_contract_hardened_prompt import (
    ContractHardenedTrendHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_contract_hardened_runtime import (
    ContractHardenedTrendHypothesisMakerAgentRuntime,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Header must be KEY=VALUE")
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Header key may not be empty")
    return key, item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run alpha4c.5i contract-hardened Trend Hypothesis Maker. "
            "This additive runtime removes Trend direction, Trend-bound "
            "observable, falsifier-observable duplication, and new "
            "paper-absence authority from the LLM draft."
        )
    )
    parser.add_argument("--input", required=True)
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
    parser.add_argument(
        "--max-repairs",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-hypotheses", type=int, default=3)
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_header,
        metavar="KEY=VALUE",
    )
    parser.add_argument("--telemetry-path", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--save-prompt", action="store_true")
    parser.add_argument("--dry-run-prompt", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    source = TrendAwareHypothesisInput.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else input_path.with_suffix("").with_name(
            input_path.stem.replace(".input", "")
            + ".trend_hypothesis_alpha4c5i"
        )
    )

    assembler = ContractHardenedTrendHypothesisPromptAssembler(
        max_hypotheses=args.max_hypotheses
    )
    exposure = build_contract_hardened_trend_maker_exposure(source)
    prompt = assembler.build(source, exposure=exposure)
    _write_json(
        Path(str(prefix) + ".hardened_exposure.json"),
        exposure,
    )

    if args.save_prompt or args.dry_run_prompt:
        prompt_path = Path(str(prefix) + ".prompt.txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )
        _write_json(
            Path(str(prefix) + ".prompt.meta.json"),
            {
                "prompt_version": prompt.prompt_version,
                "prompt_sha256": prompt.prompt_sha256,
                "hardened_exposure_id": exposure.exposure_id,
                "hardened_exposure_sha256":
                    exposure.exposure_sha256,
                "source_directional_exposure_id":
                    exposure.source_directional_exposure_id,
                "source_directional_exposure_sha256":
                    exposure.source_directional_exposure_sha256,
                "source_input_id": source.input_id,
                "source_input_sha256": source.input_sha256,
            },
        )
        print("Prompt version:", prompt.prompt_version)
        print("Prompt SHA256:", prompt.prompt_sha256)
        print("Hardened exposure ID:", exposure.exposure_id)
        print(
            "Hardened exposure SHA256:",
            exposure.exposure_sha256,
        )
        print("Prompt saved:", prompt_path)

    if args.dry_run_prompt:
        print("LLM calls: 0")
        return

    if not args.model:
        raise SystemExit(
            "--model is required unless a Trend Hypothesis model "
            "environment variable is set."
        )

    backend = (
        InstructorOpenAICompatibleContractHardenedTrendBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=dict(args.header),
            telemetry_path=args.telemetry_path,
            telemetry_context={
                "source_trend_input_id": source.input_id,
                "source_trend_input_sha256":
                    source.input_sha256,
                "hardened_exposure_id": exposure.exposure_id,
                "hardened_exposure_sha256":
                    exposure.exposure_sha256,
            },
        )
    )
    runtime = ContractHardenedTrendHypothesisMakerAgentRuntime(
        backend,
        prompt_assembler=assembler,
        max_repairs=args.max_repairs,
    )
    outcome = runtime.run(source)

    for index, draft in enumerate(outcome.draft_history):
        suffix = (
            ".draft.json"
            if index == 0
            else f".repair{index}.draft.json"
        )
        _write_json(Path(str(prefix) + suffix), draft)

    _write_json(
        Path(str(prefix) + ".run.json"),
        outcome.run_record,
    )
    _write_json(
        Path(str(prefix) + ".hardened_exposure.json"),
        outcome.exposure,
    )

    if outcome.validation is not None:
        _write_json(
            Path(str(prefix) + ".validation.json"),
            outcome.validation,
        )
    else:
        _write_json(
            Path(str(prefix) + ".validation.json"),
            {
                "passes": False,
                "stage": "compile",
                "issues": [
                    row.model_dump(mode="json")
                    for row in outcome.compile_issues
                ],
            },
        )

    if outcome.accepted_portfolio is not None:
        _write_json(
            Path(str(prefix) + ".portfolio.json"),
            outcome.accepted_portfolio,
        )
    elif outcome.last_portfolio is not None:
        _write_json(
            Path(str(prefix) + ".rejected_portfolio.json"),
            outcome.last_portfolio,
        )

    print("Contract-hardened Trend Hypothesis Maker run complete")
    print("Run ID:", outcome.run_record.run_id)
    print("Trend input SHA256:", source.input_sha256)
    print("Hardened exposure:", outcome.exposure.exposure_id)
    print(
        "Prompt:",
        prompt.prompt_version,
        prompt.prompt_sha256,
    )
    print(
        "Backend/model:",
        outcome.run_record.backend,
        outcome.run_record.model,
    )
    print(
        "Generation attempts:",
        outcome.run_record.generation_attempts,
    )
    print("Repair attempts:", outcome.run_record.repair_attempts)
    print("Accepted:", outcome.accepted)
    print("Failure stage:", outcome.run_record.failure_stage)
    print(
        "Validation errors/warnings:",
        outcome.run_record.validation_errors,
        outcome.run_record.validation_warnings,
    )
    if outcome.accepted_portfolio is not None:
        print(
            "Portfolio ID:",
            outcome.accepted_portfolio.portfolio_id,
        )
        print(
            "Portfolio SHA256:",
            outcome.run_record.portfolio_sha256,
        )
        print(
            "Hypotheses:",
            len(outcome.accepted_portfolio.hypotheses),
        )
        print(
            "Abstained:",
            not bool(outcome.accepted_portfolio.hypotheses),
        )
        print(
            "Saved portfolio:",
            Path(str(prefix) + ".portfolio.json"),
        )
    else:
        print(
            "No contract-hardened Trend hypothesis portfolio was "
            "accepted; downstream consumers must not consume this run."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
