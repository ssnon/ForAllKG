from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.discovery_hypothesis_prompt import DiscoveryAwareHypothesisPromptAssembler
from dac_her.dual_hypothesis_context import DualHypothesisContext
from dac_her.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
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
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[attr-defined]
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing deterministic Hypothesis Maker compiler/validator while supplying "
            "a separate non-evidentiary DiscoveryBundle to the prompt."
        )
    )
    parser.add_argument("--dual-context", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL") or os.getenv("OPENROUTER_AGENT_MODEL"),
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--max-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-hypotheses", type=int, default=3)
    parser.add_argument(
        "--allow-empty-discovery",
        action="store_true",
        help=(
            "Diagnostic only: allow the discovery-aware maker to run when "
            "the DiscoveryBundle contains no inspirations."
        ),
    )
    parser.add_argument("--header", action="append", default=[], type=_header, metavar="KEY=VALUE")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--save-prompt", action="store_true")
    parser.add_argument("--dry-run-prompt", action="store_true")
    args = parser.parse_args()

    dual_path = Path(args.dual_context)
    dual = DualHypothesisContext.model_validate_json(dual_path.read_text(encoding="utf-8"))
    context = dual.grounded_context

    if not dual.discovery_bundle.inspirations and not args.allow_empty_discovery:
        raise SystemExit(
            "DiscoveryBundle contains no discovery-distinct inspirations. "
            "Refusing to run the discovery-aware Hypothesis Maker because it "
            "would collapse back to the grounded/canonical hypothesis regime. "
            "Add an exploratory traversal or broaden retrieval; use "
            "--allow-empty-discovery only for a diagnostic ablation."
        )
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else dual_path.with_suffix("").with_name(
            dual_path.stem.replace(".dual_context", "") + ".hypothesis_discovery_v280a1"
        )
    )

    assembler = DiscoveryAwareHypothesisPromptAssembler(
        dual.discovery_bundle,
        max_hypotheses=args.max_hypotheses,
    )
    prompt = assembler.build(context)
    if args.save_prompt or args.dry_run_prompt:
        prompt_path = Path(str(prefix) + ".prompt.txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            "SYSTEM\n======\n" + prompt.system_prompt + "\n\nUSER\n====\n" + prompt.user_prompt + "\n",
            encoding="utf-8",
        )
        print("Prompt version:", prompt.prompt_version)
        print("Prompt SHA256:", prompt.prompt_sha256)
        print("Prompt saved:", prompt_path)
    if args.dry_run_prompt:
        return

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or OPENROUTER_AGENT_MODEL is set."
        )

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
    outcome = runtime.run(context)

    for index, draft in enumerate(outcome.draft_history):
        suffix = ".draft.json" if index == 0 else f".repair{index}.draft.json"
        _write_json(Path(str(prefix) + suffix), draft)
    _write_json(Path(str(prefix) + ".run.json"), outcome.run_record)
    _write_json(
        Path(str(prefix) + ".discovery_lineage.json"),
        {
            "schema_version": "hypothesis-discovery-lineage-v1",
            "dual_context_id": dual.dual_context_id,
            "dual_context_sha256": dual.dual_context_sha256,
            "grounded_context_id": context.context_id,
            "grounded_context_sha256": context.context_sha256,
            "discovery_bundle_id": dual.discovery_bundle.bundle_id,
            "discovery_bundle_sha256": dual.discovery_bundle.bundle_sha256,
            "prompt_version": prompt.prompt_version,
            "prompt_sha256": prompt.prompt_sha256,
        },
    )
    if outcome.validation is not None:
        _write_json(Path(str(prefix) + ".validation.json"), outcome.validation)
    else:
        _write_json(
            Path(str(prefix) + ".validation.json"),
            {
                "passes": False,
                "stage": "compile",
                "issues": [x.model_dump(mode="json") for x in outcome.compile_issues],
            },
        )
    if outcome.accepted_portfolio is not None:
        _write_json(Path(str(prefix) + ".portfolio.json"), outcome.accepted_portfolio)
    elif outcome.last_portfolio is not None:
        _write_json(Path(str(prefix) + ".rejected_portfolio.json"), outcome.last_portfolio)

    print("Discovery-aware Hypothesis Maker run complete")
    print("Dual context ID:", dual.dual_context_id)
    print("Grounded context SHA256:", context.context_sha256)
    print("Discovery bundle:", dual.discovery_bundle.bundle_id)
    print("Prompt:", outcome.run_record.prompt_version, outcome.run_record.prompt_sha256)
    print("Accepted:", outcome.accepted)
    print("Hypotheses:", len(outcome.accepted_portfolio.hypotheses) if outcome.accepted_portfolio else 0)
    if outcome.accepted_portfolio is None:
        raise SystemExit(2)
    print("Saved portfolio:", Path(str(prefix) + ".portfolio.json"))


if __name__ == "__main__":
    main()
