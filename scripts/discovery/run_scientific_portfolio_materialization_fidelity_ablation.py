from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.scientific_portfolio_materialization import (
    ScientificPortfolioMaterializationBatchDraft,
    ScientificPortfolioMaterializationReport,
    compile_scientific_portfolio_materialization,
)
from pipeline_core.discovery.scientific_portfolio_prompt import (
    ScientificPortfolioPrompt,
    build_materialization_prompt,
)
from pipeline_core.discovery.scientific_portfolio_runtime import (
    InstructorOpenAICompatibleScientificPortfolioBackend,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioEvaluationReport,
    ScientificPortfolioSelectionReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FidelityStatus = Literal[
    "PRESERVED",
    "PARTIAL",
    "COLLAPSED",
    "ABSTAINED",
    "MISSING",
]


class FidelityAuditRow(StrictModel):
    candidate_id: str
    arm: Literal["A_CURRENT", "B_FIDELITY"]
    status: FidelityStatus
    rationale: str = Field(min_length=1)
    preserved_elements: list[str] = Field(default_factory=list)
    lost_or_weakened_elements: list[str] = Field(default_factory=list)


class FidelityAuditBatch(StrictModel):
    schema_version: Literal[
        "scientific-portfolio-materialization-fidelity-audit-v1"
    ] = "scientific-portfolio-materialization-fidelity-audit-v1"
    rows: list[FidelityAuditRow]

    @model_validator(mode="after")
    def _coverage(self) -> "FidelityAuditBatch":
        keys = [(row.candidate_id, row.arm) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate candidate/arm fidelity audit row")
        return self


FIDELITY_SYSTEM_ADDENDUM = r"""
FIDELITY-PRESERVING MATERIALIZATION POLICY
=========================================
The source candidate's distinctive conceptual delta is part of the scientific
object being materialized. Preserve that delta explicitly.

- Positive premise IDs still come ONLY from eligible grounded context statements.
- Grounded premises justify background/anchor facts; they do NOT need to already
  report or establish the candidate's conjectural relation.
- A candidate's distinctive mediator, moderator, interaction, regime boundary,
  temporal dependency, causal contrast, proxy challenge, or cross-source bridge
  may remain an explicit HYPOTHESIS / INFERENTIAL BRIDGE that requires verification.
- Do NOT weaken a selected idea into a generic relation merely because the generic
  relation is easier to support from the positive premises.
- Do NOT silently replace the source candidate by a safer nearby KG-grounded
  extension.
- Preserve the candidate's distinctive conceptual relation in the
  hypothesis_statement and/or inferential_bridge, and preserve a discriminating
  consequence in predicted_observations when the source candidate supplies one.
- If the distinctive conceptual delta cannot be retained without falsely asserting
  it as established evidence, state it as the proposed inference/hypothesis.
- If it still cannot be represented responsibly, ABSTAIN for that candidate.
  Abstention is preferred to conceptual flattening.
- External/open-world lineage remains inspiration only and never becomes a
  positive premise merely because its conceptual delta is preserved.
"""


FIDELITY_AUDIT_SYSTEM = r"""You are auditing conceptual fidelity in a scientific
hypothesis-generation ablation.

You are NOT judging truth, literature novelty, publication value, or whether the
grounding is sufficient. Judge only whether each materialized output preserves
the distinctive scientific idea supplied by its source candidate.

Definitions:
- PRESERVED: the source candidate's central causal/mechanistic relation or its
  distinctive moderator/mediator/interaction/boundary/temporal/contrast structure
  remains explicit and scientifically testable in the materialized hypothesis.
- PARTIAL: the central idea remains recognizable, but an important modifier,
  dependency, contrast, or discriminating consequence is weakened or lost.
- COLLAPSED: the materialized hypothesis has become a generic nearby grounded
  relation that could plausibly have been generated without the source candidate,
  or the source candidate's main conceptual delta has disappeared.
- ABSTAINED: that arm explicitly abstained for the candidate.
- MISSING: no materialization or explicit abstention is available.

Do not reward an arm because it is named CURRENT or FIDELITY. Apply the same
criterion to both arms. Preserve candidate_id and arm exactly. Return exactly two
rows per selected candidate, one for A_CURRENT and one for B_FIDELITY."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_model(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _card_payload(card: object | None) -> dict[str, Any] | None:
    if card is None:
        return None
    return {
        "hypothesis_id": getattr(card, "hypothesis_id", None),
        "title": getattr(card, "title", ""),
        "hypothesis_statement": getattr(card, "hypothesis_statement", ""),
        "inferential_bridge": getattr(card, "inferential_bridge", ""),
        "premise_statement_ids": list(getattr(card, "premise_statement_ids", [])),
        "predicted_observations": [
            {
                "observable": row.observable,
                "expected_direction": row.expected_direction,
                "rationale": row.rationale,
            }
            for row in getattr(card, "predicted_observations", [])
        ],
        "falsification_criteria": [
            {
                "observable": row.observable,
                "falsifying_outcome": row.falsifying_outcome,
            }
            for row in getattr(card, "falsification_criteria", [])
        ],
    }


def _mapping(
    *,
    report: ScientificPortfolioMaterializationReport,
    portfolio: HypothesisPortfolio,
) -> tuple[dict[str, object], dict[str, str]]:
    cards = {row.hypothesis_id: row for row in portfolio.hypotheses}
    materialized: dict[str, object] = {}
    terminal: dict[str, str] = {}
    for row in report.records:
        terminal[row.candidate_id] = row.status
        if row.hypothesis_id and row.hypothesis_id in cards:
            materialized[row.candidate_id] = cards[row.hypothesis_id]
    return materialized, terminal


def _premise_stats(portfolio: HypothesisPortfolio) -> dict[str, Any]:
    sets = [frozenset(card.premise_statement_ids) for card in portfolio.hypotheses]
    counts = Counter(tuple(sorted(x)) for x in sets)
    pairwise = []
    for a, b in combinations(sets, 2):
        union = a | b
        pairwise.append((len(a & b) / len(union)) if union else 1.0)
    shared = set(sets[0]) if sets else set()
    for row in sets[1:]:
        shared &= set(row)
    return {
        "hypothesis_count": len(sets),
        "distinct_premise_set_count": len(counts),
        "largest_premise_set_share": (
            max(counts.values()) / len(sets) if sets else 0.0
        ),
        "shared_core_premise_ids": sorted(shared),
        "mean_pairwise_premise_jaccard": (
            sum(pairwise) / len(pairwise) if pairwise else 0.0
        ),
        "max_pairwise_premise_jaccard": max(pairwise) if pairwise else 0.0,
    }


def _build_fidelity_prompt(
    *,
    context: HypothesisContext,
    pool: ScientificPortfolioCandidatePool,
    evaluation: ScientificPortfolioEvaluationReport,
    selection: ScientificPortfolioSelectionReport,
) -> ScientificPortfolioPrompt:
    base = build_materialization_prompt(
        context=context,
        pool=pool,
        evaluation=evaluation,
        selection=selection,
    )
    return ScientificPortfolioPrompt(
        kind="fidelity_preserving_materialization_ablation",
        system_prompt=base.system_prompt + "\n\n" + FIDELITY_SYSTEM_ADDENDUM,
        user_prompt=(
            base.user_prompt
            + "\n\nABLATION INSTRUCTION\n====================\n"
            + "This is the fidelity-preserving arm. Preserve the selected "
              "candidate's distinctive conceptual delta. If preservation is "
              "not possible under the unchanged evidence-authority rules, "
              "abstain rather than substitute a generic grounded extension."
        ),
    )


def _build_audit_prompt(
    *,
    pool: ScientificPortfolioCandidatePool,
    evaluation: ScientificPortfolioEvaluationReport,
    selection: ScientificPortfolioSelectionReport,
    a_report: ScientificPortfolioMaterializationReport,
    a_portfolio: HypothesisPortfolio,
    b_report: ScientificPortfolioMaterializationReport,
    b_portfolio: HypothesisPortfolio,
) -> ScientificPortfolioPrompt:
    candidates = {row.candidate_id: row for row in pool.candidates}
    evals = {row.candidate_id: row for row in evaluation.evaluations}
    a_cards, a_terminal = _mapping(report=a_report, portfolio=a_portfolio)
    b_cards, b_terminal = _mapping(report=b_report, portfolio=b_portfolio)
    rows = []
    for entry in selection.entries:
        candidate = candidates[entry.candidate_id]
        ev = evals[entry.candidate_id]
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "source_candidate": {
                    "origin": candidate.origin,
                    "source_kind": candidate.source_kind,
                    "operator_id": candidate.operator_id,
                    "idea_form": candidate.idea_form,
                    "scientific_intent": candidate.scientific_intent,
                    "conceptual_change_summary": candidate.conceptual_change_summary,
                    "core_relations": candidate.core_relations,
                    "differential_prediction": candidate.differential_prediction,
                    "falsification_condition": candidate.falsification_condition,
                    "discriminating_observation": candidate.discriminating_observation,
                    "normalized_scientific_sketch": ev.normalized_sketch.model_dump(
                        mode="json"
                    ),
                },
                "A_CURRENT": {
                    "terminal_status": a_terminal.get(candidate.candidate_id, "MISSING"),
                    "materialized_hypothesis": _card_payload(
                        a_cards.get(candidate.candidate_id)
                    ),
                },
                "B_FIDELITY": {
                    "terminal_status": b_terminal.get(candidate.candidate_id, "MISSING"),
                    "materialized_hypothesis": _card_payload(
                        b_cards.get(candidate.candidate_id)
                    ),
                },
            }
        )
    return ScientificPortfolioPrompt(
        kind="materialization_fidelity_audit",
        system_prompt=FIDELITY_AUDIT_SYSTEM,
        user_prompt=(
            "Audit both materialization arms against each source candidate.\n\n"
            + json.dumps({"candidates": rows}, ensure_ascii=False, indent=2)
        ),
    )


def _audit_summary(audit: FidelityAuditBatch) -> dict[str, Any]:
    by_arm: dict[str, Counter[str]] = {
        "A_CURRENT": Counter(),
        "B_FIDELITY": Counter(),
    }
    score = {
        "PRESERVED": 2,
        "PARTIAL": 1,
        "COLLAPSED": 0,
        "ABSTAINED": None,
        "MISSING": None,
    }
    scored: dict[str, list[int]] = {"A_CURRENT": [], "B_FIDELITY": []}
    for row in audit.rows:
        by_arm[row.arm][row.status] += 1
        s = score[row.status]
        if s is not None:
            scored[row.arm].append(s)
    result: dict[str, Any] = {}
    for arm in ("A_CURRENT", "B_FIDELITY"):
        result[arm] = {
            "status_counts": dict(sorted(by_arm[arm].items())),
            "mean_fidelity_score_nonabstained": (
                sum(scored[arm]) / len(scored[arm]) if scored[arm] else None
            ),
        }
    return result


def _external_status_counts(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    payload = _load_json(path)
    counts = payload.get("status_counts")
    return dict(counts) if isinstance(counts, dict) else None


def _run_external(
    *,
    portfolio: Path,
    output_prefix: Path,
    domain_profile: str,
    model: str,
    api_key_env: str,
    base_url: str | None,
    providers: str,
    results_per_query: int,
    provider_plan: Path | None,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_external_novelty",
        "--portfolio",
        str(portfolio),
        "--domain-profile",
        domain_profile,
        "--model",
        model,
        "--api-key-env",
        api_key_env,
        "--providers",
        providers,
        "--results-per-query",
        str(results_per_query),
        "--output-prefix",
        str(output_prefix),
        "--pre-review-coverage-shadow",
        "--downstream-gate-shadow",
        "--save-prompts",
    ]
    if base_url:
        cmd += ["--base-url", base_url]
    if provider_plan is not None and provider_plan.is_file():
        cmd += ["--provider-plan", str(provider_plan)]
    print()
    print("===== B EXTERNAL NOVELTY =====")
    print("$", " ".join(cmd))
    return subprocess.run(cmd).returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "A/B the existing Scientific Portfolio materialization against a "
            "fidelity-preserving policy using the same selected candidates and context."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--model", default=os.getenv("OPENROUTER_AGENT_MODEL") or "")
    p.add_argument(
        "--critic-model",
        default=(
            os.getenv("OPENROUTER_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--providers", default="auto")
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--skip-external-novelty", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model:
        raise SystemExit("--model or OPENROUTER_AGENT_MODEL is required")
    if not args.critic_model:
        raise SystemExit("--critic-model or OPENROUTER_CRITIC_MODEL is required")

    run = args.run_dir.expanduser().resolve()
    sp = run / "scientific_portfolio_shadow"
    ver = sp / "downstream_verification"
    out = run / "materialization_fidelity_ablation"

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"ablation output is non-empty: {out}\n"
            "rerun with --overwrite to replace it"
        )
    if args.overwrite and out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    context_path = run / "hypothesis.context.json"
    pool_path = sp / "candidate_pool.json"
    evaluation_path = sp / "evaluation.json"
    selection_path = sp / "selection.json"
    a_report_path = sp / "materialization.report.json"
    a_portfolio_path = sp / "materialized.shadow.portfolio.json"

    required = [
        context_path,
        pool_path,
        evaluation_path,
        selection_path,
        a_report_path,
        a_portfolio_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing ablation inputs:\n" + "\n".join(missing))

    context = _load_model(context_path, HypothesisContext)
    pool = _load_model(pool_path, ScientificPortfolioCandidatePool)
    evaluation = _load_model(evaluation_path, ScientificPortfolioEvaluationReport)
    selection = _load_model(selection_path, ScientificPortfolioSelectionReport)
    a_report = _load_model(a_report_path, ScientificPortfolioMaterializationReport)
    a_portfolio = _load_model(a_portfolio_path, HypothesisPortfolio)

    backend = InstructorOpenAICompatibleScientificPortfolioBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=0.0,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        telemetry_path=out / "telemetry.jsonl",
        telemetry_context={"ablation": "fidelity_preserving_materialization_v1"},
    )

    b_prompt = _build_fidelity_prompt(
        context=context,
        pool=pool,
        evaluation=evaluation,
        selection=selection,
    )
    (out / "B.materialization.prompt.txt").write_text(
        "SYSTEM\n======\n"
        + b_prompt.system_prompt
        + "\n\nUSER\n====\n"
        + b_prompt.user_prompt
        + "\n",
        encoding="utf-8",
    )

    print("===== B MATERIALIZATION =====")
    b_generation = backend.materialize(b_prompt)
    b_draft = b_generation.draft
    _write_json(out / "B.materialization.draft.json", b_draft)

    b_report, b_portfolio = compile_scientific_portfolio_materialization(
        context=context,
        pool=pool,
        selection=selection,
        draft=b_draft,
    )
    _write_json(out / "B.materialization.report.json", b_report)
    _write_json(out / "B.materialized.portfolio.json", b_portfolio)

    print("selected:", selection.retained_count)
    print("A materialized:", a_report.materialized_hypothesis_count)
    print("B materialized:", b_report.materialized_hypothesis_count)
    print("B status:", b_report.status_counts)

    audit_prompt = _build_audit_prompt(
        pool=pool,
        evaluation=evaluation,
        selection=selection,
        a_report=a_report,
        a_portfolio=a_portfolio,
        b_report=b_report,
        b_portfolio=b_portfolio,
    )
    (out / "fidelity_audit.prompt.txt").write_text(
        "SYSTEM\n======\n"
        + audit_prompt.system_prompt
        + "\n\nUSER\n====\n"
        + audit_prompt.user_prompt
        + "\n",
        encoding="utf-8",
    )
    audit, _event = backend._call(
        prompt=audit_prompt,
        response_model=FidelityAuditBatch,
        stage="materialization_fidelity_audit",
    )
    expected_keys = {
        (entry.candidate_id, arm)
        for entry in selection.entries
        for arm in ("A_CURRENT", "B_FIDELITY")
    }
    actual_keys = {(row.candidate_id, row.arm) for row in audit.rows}
    if actual_keys != expected_keys:
        raise RuntimeError(
            "fidelity audit did not cover selected candidates exactly; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    _write_json(out / "fidelity_audit.json", audit)

    provider_plan = ver / "external_novelty.provider_plan.json"
    b_external_prefix = out / "B.external_novelty"
    b_external_rc = None
    if not args.skip_external_novelty and b_portfolio.hypotheses:
        b_external_rc = _run_external(
            portfolio=out / "B.materialized.portfolio.json",
            output_prefix=b_external_prefix,
            domain_profile=context.domain_profile_id,
            model=args.critic_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            providers=args.providers,
            results_per_query=args.results_per_query,
            provider_plan=(provider_plan if provider_plan.is_file() else None),
        )

    a_ext = ver / "external_novelty.report.json"
    b_ext = Path(str(b_external_prefix) + ".report.json")

    comparison = {
        "schema_version": "scientific-portfolio-materialization-ablation-v1",
        "source_run": str(run),
        "selected_candidate_count": selection.retained_count,
        "A_CURRENT": {
            "materialized_hypothesis_count": a_report.materialized_hypothesis_count,
            "materialization_status_counts": a_report.status_counts,
            "premise_stats": _premise_stats(a_portfolio),
            "external_novelty_status_counts": _external_status_counts(a_ext),
        },
        "B_FIDELITY": {
            "materialized_hypothesis_count": b_report.materialized_hypothesis_count,
            "materialization_status_counts": b_report.status_counts,
            "premise_stats": _premise_stats(b_portfolio),
            "external_novelty_return_code": b_external_rc,
            "external_novelty_status_counts": _external_status_counts(b_ext),
        },
        "fidelity_audit": _audit_summary(audit),
        "interpretation_guard": (
            "This ablation tests whether materialization policy, holding the "
            "selected candidate set and grounded context fixed, changes conceptual "
            "fidelity and downstream novelty. It does not establish literature-wide "
            "novelty or generalization."
        ),
    }
    _write_json(out / "comparison.json", comparison)

    print()
    print("===== A/B SUMMARY =====")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print()
    print("Artifacts:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
