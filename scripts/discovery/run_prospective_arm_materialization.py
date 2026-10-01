from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityAssessor,
)
from pipeline_core.discovery.prospective_validation import (
    ProspectiveArmSelection,
    build_family_balanced_arm_selection,
)
from pipeline_core.discovery.scientific_portfolio_materialization import (
    ScientificPortfolioMaterializationAbstention,
    ScientificPortfolioMaterializationBatchDraft,
    compile_scientific_portfolio_materialization,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _prompt(
    *,
    context: HypothesisContext,
    pool: ScientificPortfolioCandidatePool,
    selection: ProspectiveArmSelection,
) -> tuple[str, str]:
    candidates = {
        row.candidate_id: row
        for row in pool.candidates
    }
    selected = [
        candidates[candidate_id]
        for candidate_id
        in selection.retained_candidate_ids
    ]

    eligible = [
        {
            "statement_id": row.statement_id,
            "text": row.text,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
            "requires_verification": row.requires_verification,
        }
        for row in context.evidence_statements
        if row.eligible_as_premise
    ]
    gaps = [
        {
            "statement_id": row.statement_id,
            "text": row.text,
            "claim_kind": row.claim_kind,
        }
        for row in context.evidence_statements
        if row.eligible_as_gap
    ]

    candidate_payload = []
    for row in selected:
        candidate_payload.append(
            {
                "candidate_id": row.candidate_id,
                "origin": row.origin,
                "idea_form": row.idea_form,
                "operator_id": row.operator_id,
                "scientific_intent": row.scientific_intent,
                "conceptual_change_summary":
                    row.conceptual_change_summary,
                "core_relations": row.core_relations,
                "differential_prediction":
                    row.differential_prediction,
                "falsification_condition":
                    row.falsification_condition,
                "discriminating_observation":
                    row.discriminating_observation,
                "task_relation_mode":
                    row.task_relation_mode,
                "external_literature_lineage":
                    row.external_literature_lineage,
                "candidate_or_unverified_lineage":
                    row.candidate_or_unverified_lineage,
            }
        )

    system = """
You are a grounded scientific-hypothesis materializer for a controlled
ablation study.

The supplied idea candidates are INSPIRATION ONLY. They are not evidence,
truth, novelty, or positive premises.

For each selected candidate, either:
1. materialize exactly one hypothesis using ONLY statement IDs from
   ELIGIBLE_POSITIVE_PREMISES as positive premises; or
2. explicitly abstain when that cannot be done without inventing support.

Rules:
- Do not promote external-literature inspiration, candidate/unverified
  lineage, an axis, a topology, or an evolved idea into evidence.
- premise_statement_ids may contain only IDs explicitly listed under
  ELIGIBLE_POSITIVE_PREMISES.
- gap_statement_ids may contain only IDs explicitly listed under
  ELIGIBLE_GAPS.
- Do not claim literature-wide novelty or priority.
- Do not invent unsupported numeric predictions.
- Include at least one falsifiable predicted observation and one
  falsification criterion.
- The falsifier observable MUST match a predicted-observation observable.
  Prefer copying the exact same observable string.
- State assumptions explicitly.
- Do not write a detailed experimental protocol.
- Return each candidate ID exactly once across items and abstentions.
""".strip()

    user = json.dumps(
        {
            "research_question": context.question,
            "arm": selection.arm,
            "selection_policy":
                selection.selection_policy,
            "selected_candidates":
                candidate_payload,
            "ELIGIBLE_POSITIVE_PREMISES":
                eligible,
            "ELIGIBLE_GAPS": gaps,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Materialize a deterministic family-balanced prospective "
            "ablation arm. Candidate selection uses no scientific-quality "
            "ranking; idea artifacts remain inspiration-only."
        )
    )
    p.add_argument("--context", type=Path, required=True)
    p.add_argument(
        "--candidate-pool",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--arm",
        choices=[
            "FRONTIER_BALANCED",
            "EVOLUTION_BALANCED",
        ],
        required=True,
    )
    p.add_argument(
        "--max-hypotheses",
        type=int,
        default=8,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
    )
    p.add_argument(
        "--instructor-mode",
        default="JSON",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--parse-retries",
        type=int,
        default=2,
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=240.0,
    )
    return p


def main() -> int:
    args = parser().parse_args()
    if not args.model:
        raise SystemExit(
            "--model is required unless a model environment "
            "variable is configured"
        )
    if args.max_hypotheses < 1:
        raise ValueError("--max-hypotheses must be >= 1")

    context = HypothesisContext.model_validate_json(
        args.context.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    pool = ScientificPortfolioCandidatePool.model_validate_json(
        args.candidate_pool.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    if context.context_id != pool.source_context_id:
        raise ValueError("context/candidate-pool ID mismatch")
    if context.context_sha256 != pool.source_context_sha256:
        raise ValueError("context/candidate-pool SHA mismatch")

    selection = build_family_balanced_arm_selection(
        pool=pool,
        arm=args.arm,
        max_hypotheses=args.max_hypotheses,
    )

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    system, user = _prompt(
        context=context,
        pool=pool,
        selection=selection,
    )
    (out / "materialization.prompt.txt").write_text(
        "SYSTEM\n======\n"
        + system
        + "\n\nUSER\n====\n"
        + user
        + "\n",
        encoding="utf-8",
    )

    generation_error: dict[str, Any] | None = None
    try:
        import instructor
        from openai import OpenAI

        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key available. Set {args.api_key_env}."
            )
        mode = getattr(
            instructor.Mode,
            str(args.instructor_mode).upper(),
            None,
        )
        if mode is None:
            raise ValueError(
                f"Unknown Instructor mode: {args.instructor_mode}"
            )
        kwargs: dict[str, Any] = {
            "api_key": api_key,
        }
        if args.base_url:
            kwargs["base_url"] = args.base_url
        if args.timeout is not None:
            kwargs["timeout"] = args.timeout
        client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=mode,
        )
        draft, event = run_instructor_structured_call(
            client.chat.completions,
            model=args.model,
            response_model=(
                ScientificPortfolioMaterializationBatchDraft
            ),
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            temperature=args.temperature,
            max_retries=args.parse_retries,
            telemetry_path=out / "telemetry.jsonl",
            telemetry_context={
                "pipeline":
                    "prospective_validation_ablation",
                "stage":
                    f"{args.arm.lower()}_materialization",
            },
            semantic_components={
                "authority":
                    "SHADOW_ABLATION_ONLY",
                "truth_authority": False,
                "novelty_authority": False,
            },
        )
        if not isinstance(
            draft,
            ScientificPortfolioMaterializationBatchDraft,
        ):
            draft = (
                ScientificPortfolioMaterializationBatchDraft
                .model_validate(draft)
            )
        generation = {
            "served_model":
                event.served_model,
            "provider_input_tokens":
                event.provider_input_tokens,
            "provider_output_tokens":
                event.provider_output_tokens,
            "response_id":
                event.response_id,
            "elapsed_seconds":
                event.elapsed_seconds,
        }
    except Exception as exc:
        generation_error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        draft = (
            ScientificPortfolioMaterializationBatchDraft(
                abstentions=[
                    ScientificPortfolioMaterializationAbstention(
                        candidate_id=candidate_id,
                        reason=(
                            "Prospective ablation materialization "
                            "structured call failed: "
                            f"{type(exc).__name__}"
                        ),
                    )
                    for candidate_id
                    in selection.retained_candidate_ids
                ]
            )
        )
        generation = None

    report, portfolio = (
        compile_scientific_portfolio_materialization(
            context=context,
            pool=pool,
            selection=selection,
            draft=draft,
        )
    )
    diversity = (
        HypothesisEvidenceDiversityAssessor().assess(
            context,
            portfolio,
        )
    )

    _write(out / "selection.json", selection)
    _write(out / "materialization.draft.json", draft)
    _write(out / "materialization.report.json", report)
    _write(out / "materialized.portfolio.json", portfolio)
    _write(out / "evidence_diversity.json", diversity)
    _write(
        out / "generation.json",
        {
            "schema_version":
                "prospective-arm-materialization-generation-v1",
            "arm": args.arm,
            "generation": generation,
            "generation_error": generation_error,
            "shadow_only": True,
            "production_selection_authority": False,
        },
    )

    print("Prospective arm materialization complete")
    print("arm:", args.arm)
    print(
        "selected:",
        selection.selected_candidate_count,
    )
    print(
        "unique families:",
        selection.selected_unique_family_count,
    )
    print(
        "materialized:",
        report.materialized_hypothesis_count,
    )
    print("status:", report.status_counts)
    print("generation error:", generation_error)
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
