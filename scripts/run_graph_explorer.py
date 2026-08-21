from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from domains.registry import get_domain_profile
from pipeline_core.discovery.explorer_contracts import GraphExplorerPacket
from pipeline_core.discovery.explorer_llm import InstructorOpenAICompatibleDraftBackend
from pipeline_core.discovery.explorer_validation import ExplorationReportValidator
from dac_her.explorer_runtime import GraphExplorerAgentRuntime


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
            "Run the evidence-bounded Graph Explorer v2.5.1: prompt -> structured "
            "ExplorationDraft -> deterministic compile -> validation -> at most one repair."
        )
    )
    parser.add_argument("--packet", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("GRAPHAGENTS_EXPLORER_MODEL"),
        help="OpenAI-compatible model name. May also be set with GRAPHAGENTS_EXPLORER_MODEL.",
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--max-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_header,
        metavar="KEY=VALUE",
        help="Optional default HTTP header for compatible providers; repeatable.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix without suffix. Defaults beside packet file.",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Save deterministic system/user prompt text for audit.",
    )
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Build and save/print prompt metadata without calling a model.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")  # type: ignore[attr-defined]
    else:
        payload = value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    packet_path = Path(args.packet)
    packet = GraphExplorerPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))

    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else packet_path.with_suffix("").with_name(packet_path.stem.replace(".packet", "") + ".explorer_v251")
    )

    from pipeline_core.discovery.explorer_prompt import ExplorerPromptAssembler

    assembler = ExplorerPromptAssembler()
    prompt = assembler.build(packet)
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
        raise SystemExit("--model is required unless GRAPHAGENTS_EXPLORER_MODEL is set.")

    backend = InstructorOpenAICompatibleDraftBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
    )
    domain_profile = get_domain_profile(
        packet.domain_profile_id
    )

    validator = ExplorationReportValidator(
        semantics=domain_profile.discovery,
    )

    runtime = GraphExplorerAgentRuntime(
        backend,
        domain_profile=domain_profile,
        prompt_assembler=assembler,
        validator=validator,
        max_repairs=args.max_repairs,
    )
    outcome = runtime.run(packet)

    # Always keep model-owned draft(s) and run metadata for reproducibility.
    for index, draft in enumerate(outcome.draft_history):
        suffix = ".draft.json" if index == 0 else f".repair{index}.draft.json"
        _write_json(Path(str(prefix) + suffix), draft)
    _write_json(Path(str(prefix) + ".run.json"), outcome.run_record)
    _write_json(
        Path(str(prefix) + ".normalization.json"),
        outcome.normalization_audit,
    )
    if outcome.normalized_draft is not None:
        _write_json(
            Path(str(prefix) + ".normalized.draft.json"),
            outcome.normalized_draft,
        )

    if outcome.validation is not None:
        _write_json(Path(str(prefix) + ".validation.json"), outcome.validation)
    else:
        _write_json(
            Path(str(prefix) + ".validation.json"),
            {
                "passes": False,
                "stage": "compile",
                "issues": [issue.model_dump(mode="json") for issue in outcome.compile_issues],
            },
        )

    if outcome.accepted_report is not None:
        _write_json(Path(str(prefix) + ".report.json"), outcome.accepted_report)
    elif outcome.last_report is not None:
        _write_json(Path(str(prefix) + ".rejected_report.json"), outcome.last_report)

    print("Graph Explorer run complete")
    print("Run ID:", outcome.run_record.run_id)
    print("Packet SHA256:", packet.packet_sha256)
    print("Prompt:", outcome.run_record.prompt_version, outcome.run_record.prompt_sha256)
    print("Backend/model:", outcome.run_record.backend, outcome.run_record.model)
    print("Generation attempts:", outcome.run_record.generation_attempts)
    print("Repair attempts:", outcome.run_record.repair_attempts)
    print(
        "Normalization:",
        outcome.run_record.normalization_applied,
        "actions=",
        outcome.run_record.normalization_action_count,
        "blocked=",
        outcome.run_record.normalization_blocked_count,
    )
    print("Accepted:", outcome.accepted)
    print("Failure stage:", outcome.run_record.failure_stage)
    print("Validation errors/warnings:", outcome.run_record.validation_errors, outcome.run_record.validation_warnings)
    if outcome.accepted_report is not None:
        print("Report ID:", outcome.accepted_report.report_id)
        print("Report SHA256:", outcome.run_record.report_sha256)
        print("Saved report:", Path(str(prefix) + ".report.json"))
    else:
        print("No report was accepted; downstream hypothesis generation must not consume this run.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
