from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_e2 import sha256_file, validate_required_e2_records
from dac_her.hypothesis_e2_contracts import (
    HypothesisE2HumanDimension,
    HypothesisE2HumanReviewCase,
    HypothesisE2HumanReviewWorksheet,
    HypothesisE2OutputManifest,
)
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


def _repo_path(repo: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (repo / path).resolve()


def _repo_relative(repo: Path, path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=repo.resolve())).as_posix()


def _render_worksheet_markdown(
    worksheet: HypothesisE2HumanReviewWorksheet,
) -> str:
    lines = [
        f"# E2 Human Review Worksheet — {worksheet.suite_id}",
        "",
        "This worksheet is a scaffold, not gold until every case is manually reviewed.",
        "Set `approval_status` to `approved` in JSON only after reviewing all 11 dimensions.",
        "",
    ]
    for case in worksheet.cases:
        lines.extend(
            [
                f"## {case.case_id} [{case.scenario}]",
                "",
                case.description,
                "",
                f"**Review focus:** {case.review_hint}",
                "",
                "| Dimension | Critic | Human allowed | Rationale |",
                "|---|---|---|---|",
            ]
        )
        for row in case.dimensions:
            allowed = ", ".join(row.human_allowed_verdicts) or "PENDING"
            rationale = " ".join(row.critic_rationale.split()).replace("|", "\\|")
            lines.append(
                f"| `{row.dimension}` | {row.critic_verdict} | {allowed} | {rationale} |"
            )
        lines.extend(["", f"Approval: **{case.approval_status.upper()}**", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the semantic critic on E2 fresh outputs and scaffold a "
            "human-review worksheet. Critic verdicts are suggestions, not gold."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data_dac/hypothesis_e2/live_outputs/e2_output_manifest.json",
    )
    parser.add_argument("--repo", default=".")
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
        default="benchmarks/hypothesis_v262/e2_review_runs",
    )
    parser.add_argument(
        "--worksheet",
        default="benchmarks/hypothesis_v262/e2_review_worksheet.json",
    )
    parser.add_argument(
        "--prefill-suggested",
        action="store_true",
        help=(
            "Prefill human_allowed_verdicts with the critic verdict. "
            "Cases remain approval_status=pending and still require human review."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL "
            "or OPENROUTER_AGENT_MODEL is set."
        )

    repo = Path(args.repo).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = HypothesisE2OutputManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    validate_required_e2_records(manifest.cases)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worksheet_path = Path(args.worksheet).resolve()

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
    worksheet_cases: list[HypothesisE2HumanReviewCase] = []

    for record in manifest.cases:
        context_path = _repo_path(repo, record.context_path)
        portfolio_path = _repo_path(repo, record.portfolio_path)

        if sha256_file(context_path) != record.context_file_sha256:
            raise SystemExit(
                f"{record.case_id}: context bytes changed after live-output generation"
            )
        if sha256_file(portfolio_path) != record.portfolio_file_sha256:
            raise SystemExit(
                f"{record.case_id}: portfolio bytes changed after live-output generation"
            )

        context = HypothesisContext.model_validate_json(
            context_path.read_text(encoding="utf-8")
        )
        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )
        outcome = runtime.run(context, portfolio)
        if not outcome.accepted or outcome.review is None:
            raise SystemExit(
                f"{record.case_id}: semantic critic did not accept a review; "
                f"failure_stage={outcome.run_record.failure_stage}"
            )

        case_dir = output_dir / record.case_id
        _write_json(case_dir / "hard_evaluation.json", outcome.evaluation)
        _write_json(case_dir / "run.json", outcome.run_record)
        if outcome.generation is not None:
            _write_json(case_dir / "draft.json", outcome.generation.draft)
        review_path = case_dir / "review.json"
        _write_json(review_path, outcome.review)

        dimensions = [
            HypothesisE2HumanDimension(
                dimension=row.dimension,
                critic_verdict=row.verdict,
                critic_rationale=row.rationale,
                human_allowed_verdicts=(
                    [row.verdict] if args.prefill_suggested else []
                ),
                critical=True,
                human_note="",
            )
            for row in outcome.review.dimensions
        ]
        worksheet_cases.append(
            HypothesisE2HumanReviewCase(
                case_id=record.case_id,
                scenario=record.scenario,
                description=record.description,
                review_hint=record.review_hint,
                context_path=record.context_path,
                portfolio_path=record.portfolio_path,
                review_path=_repo_relative(repo, review_path),
                generator_version=record.generator_version,
                approval_status="pending",
                dimensions=dimensions,
            )
        )

        print(
            record.case_id,
            "critic=True",
            "hard=" + str(outcome.evaluation.hard_gate_passed),
        )

    worksheet = HypothesisE2HumanReviewWorksheet(
        suite_id="hypothesis-v262-e2-human-review",
        cases=worksheet_cases,
    )
    _write_json(worksheet_path, worksheet)
    worksheet_path.with_suffix(".md").write_text(
        _render_worksheet_markdown(worksheet),
        encoding="utf-8",
    )

    print("E2 human-review worksheet scaffolded")
    print("Cases:", len(worksheet.cases))
    print("JSON:", worksheet_path)
    print("Markdown:", worksheet_path.with_suffix(".md"))
    print(
        "Gold was NOT finalized. Review all dimensions, fill "
        "human_allowed_verdicts, then set each approval_status to 'approved'."
    )


if __name__ == "__main__":
    main()
