from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompileError,
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.discovery.reframing.scientific_synthesis_specification_repair import (
    ScientificSynthesisRepairEntry,
    ScientificSynthesisRepairReport,
    build_scientific_synthesis_repair_contexts,
    build_scientific_synthesis_repair_prompt,
    lock_repair_provenance,
    _stable_id,
)


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Give first-pass CONDITIONAL scientific synthesis candidates exactly "
            "one bounded N10 specification-repair attempt. The repair may change "
            "scientific wording/predictions/falsifiers only; premise/gap/type "
            "provenance is orchestrator-locked. No authority is created."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--query-plan", required=True, type=Path)
    parser.add_argument("--intake-shadow", required=True, type=Path)
    parser.add_argument("--production-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--prompt-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--telemetry", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    plan = LiteratureQueryPlan.model_validate_json(
        args.query_plan.read_text(encoding="utf-8")
    )
    intake = _load_object(args.intake_shadow)
    gate = _load_object(args.production_gate)

    repair_contexts = build_scientific_synthesis_repair_contexts(
        portfolio=portfolio,
        query_plan=plan,
        intake_shadow=intake,
        production_gate=gate,
    )
    card_by_id = {row.hypothesis_id: row for row in portfolio.hypotheses}

    print("Scientific synthesis bounded specification repair")
    print("Eligible repair candidates:", len(repair_contexts))
    print("Continuation depth: 0 -> 1")
    print("Further repair allowed after this run: false")
    print("Fresh N10 required after repair: true")
    print("Production authority created: false")

    if args.dry_run:
        for repair_context in repair_contexts:
            original = card_by_id[repair_context.source_hypothesis_id]
            prompt = build_scientific_synthesis_repair_prompt(
                original=original,
                context=context,
                repair_context=repair_context,
            )
            print(
                "  would_repair:",
                repair_context.source_hypothesis_id,
                "missing=",
                sorted(
                    {
                        field
                        for row in repair_context.claim_diagnostics
                        for field in row.missing_fields
                    }
                ),
            )
            if args.prompt_dir is not None:
                args.prompt_dir.mkdir(parents=True, exist_ok=True)
                prompt_path = (
                    args.prompt_dir
                    / (
                        repair_context.source_hypothesis_id.replace(":", "_")
                        + ".prompt.txt"
                    )
                )
                prompt_path.write_text(
                    "SYSTEM\n======\n"
                    + prompt.system_prompt
                    + "\n\nUSER\n====\n"
                    + prompt.user_prompt
                    + "\n",
                    encoding="utf-8",
                )
        return 0

    if repair_contexts and not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=args.model or "unused",
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=0.0,
        parse_retries=1,
        telemetry_path=args.telemetry,
        telemetry_context={
            "pipeline": "scientific_synthesis_bounded_specification_repair",
            "continuation_depth": 1,
        },
    )
    compiler = HypothesisCompiler()
    validator = HypothesisValidator()

    repaired_cards = []
    entries: list[ScientificSynthesisRepairEntry] = []

    for repair_context in repair_contexts:
        original = card_by_id[repair_context.source_hypothesis_id]
        prompt = build_scientific_synthesis_repair_prompt(
            original=original,
            context=context,
            repair_context=repair_context,
        )
        if args.prompt_dir is not None:
            args.prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = (
                args.prompt_dir
                / (
                    repair_context.source_hypothesis_id.replace(":", "_")
                    + ".prompt.txt"
                )
            )
            prompt_path.write_text(
                "SYSTEM\n======\n"
                + prompt.system_prompt
                + "\n\nUSER\n====\n"
                + prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )

        generation = backend.generate(prompt)
        draft = generation.draft
        if not draft.hypotheses:
            entries.append(
                ScientificSynthesisRepairEntry(
                    source_hypothesis_id=original.hypothesis_id,
                    repair_context_id=repair_context.context_id,
                    status="abstained",
                    issues=[draft.abstention_reason or "model_abstained"],
                )
            )
            continue

        try:
            locked = lock_repair_provenance(
                original=original,
                draft=draft,
                repair_context=repair_context,
            )
            compiled = compiler.compile(context, locked)
        except (ValueError, HypothesisCompileError) as exc:
            issues = (
                [row.code for row in exc.issues]
                if isinstance(exc, HypothesisCompileError)
                else [str(exc)]
            )
            entries.append(
                ScientificSynthesisRepairEntry(
                    source_hypothesis_id=original.hypothesis_id,
                    repair_context_id=repair_context.context_id,
                    status="compile_rejected",
                    issues=issues,
                )
            )
            continue

        validation = validator.validate(context, compiled)
        if not validation.passes:
            entries.append(
                ScientificSynthesisRepairEntry(
                    source_hypothesis_id=original.hypothesis_id,
                    repair_context_id=repair_context.context_id,
                    status="validation_rejected",
                    issues=[
                        issue.code
                        for issue in validation.issues
                        if issue.severity == "error"
                    ],
                )
            )
            continue
        if len(compiled.hypotheses) != 1:
            raise RuntimeError("bounded repair compiler produced non-unit portfolio")

        repaired = compiled.hypotheses[0]
        repaired_cards.append(repaired)
        entries.append(
            ScientificSynthesisRepairEntry(
                source_hypothesis_id=original.hypothesis_id,
                repair_context_id=repair_context.context_id,
                repaired_hypothesis_id=repaired.hypothesis_id,
                status="repaired",
            )
        )
        print(
            "  repaired:",
            original.hypothesis_id,
            "->",
            repaired.hypothesis_id,
        )

    abstention = None
    if not repaired_cards:
        abstention = "no_scientific_synthesis_candidates_repaired_at_bounded_depth_one"

    output_id = _stable_id(
        "hypothesis_portfolio",
        "scientific_synthesis_bounded_repair",
        portfolio.portfolio_id,
        *[row.hypothesis_id for row in repaired_cards],
        abstention or "",
    )
    output = HypothesisPortfolio(
        portfolio_id=output_id,
        domain_profile_id=portfolio.domain_profile_id,
        source_context_id=portfolio.source_context_id,
        source_context_sha256=portfolio.source_context_sha256,
        source_report_id=portfolio.source_report_id,
        source_report_sha256=portfolio.source_report_sha256,
        hypotheses=repaired_cards,
        abstention_reason=abstention,
    )
    report = ScientificSynthesisRepairReport(
        report_id=_stable_id(
            "scientific_synthesis_bounded_repair_report",
            portfolio.portfolio_id,
            output.portfolio_id,
        ),
        source_portfolio_id=portfolio.portfolio_id,
        output_portfolio_id=output.portfolio_id,
        eligible_repair_count=len(repair_contexts),
        repaired_count=len(repaired_cards),
        entries=entries,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        output.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Repaired candidates:", report.repaired_count)
    print("Fresh N10 required: true")
    print("Production authority created: false")
    print("Output:", args.output)
    print("Report:", args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
