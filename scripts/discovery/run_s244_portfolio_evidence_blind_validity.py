from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.discovery.run_s226_validation_completion import get_client
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidencePortfolioBlindReview(StrictModel):
    disposition: Literal[
        "REDUNDANCY_REVIEW_WARRANTED",
        "NO_MATERIAL_REDUNDANCY_CONCERN",
        "INDETERMINATE",
    ]
    reasons: list[
        Literal[
            "EXACT_PREMISE_REUSE",
            "SHARED_CORE_DOMINANCE",
            "LIMITED_UNIQUE_SUPPORT",
            "DISTINCT_EVIDENCE_PATHS",
            "LEGITIMATE_SHARED_FOUNDATION",
            "INSUFFICIENT_EVIDENCE_DETAIL",
        ]
    ] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


SYSTEM = """You are a blinded scientific portfolio-evidence reviewer.

You receive one portfolio of hypotheses from the SAME scientific task.
For each hypothesis you see its title and the actual positive premise
statements used to support it.

You do NOT see any automatic diversity score, Jaccard value, duplicate count,
or prior recommendation.

Your task is NOT to judge whether the hypotheses are novel or scientifically
correct. Do not rank hypotheses.

Decide only whether the portfolio's EVIDENCE ALLOCATION warrants a redundancy
review.

Use REDUNDANCY_REVIEW_WARRANTED when the portfolio materially relies on the
same evidence sets in a way that may make superficially different hypotheses
less evidentially diverse than they appear. Relevant patterns include:
- two or more hypotheses using effectively the same full premise set;
- a shared evidence core dominating the portfolio with little hypothesis-
  specific positive support;
- distinct-looking hypotheses having very limited unique evidential support.

Use NO_MATERIAL_REDUNDANCY_CONCERN when:
- shared foundational evidence is scientifically reasonable, AND
- hypotheses also use meaningfully distinct positive evidence paths or
  sufficiently different supporting subsets;
- evidence overlap exists but does not materially collapse portfolio diversity.

Do NOT reward diversity for its own sake. Shared premises are not automatically
bad. Do NOT infer relevance of unused evidence. Judge only the supplied used
positive premises.

Use INDETERMINATE when the supplied premise text is insufficient to decide.

The anonymous evidence labels E1, E2, ... are consistent across hypotheses:
the same label means the same underlying evidence statement."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def context_statement_map(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in context.get("evidence_statements", []):
        if not isinstance(row, dict):
            continue
        sid = row.get("statement_id")
        if sid:
            result[str(sid)] = row
    return result


def build_blind_payload(
    *,
    evidence: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    statements = context_statement_map(context)

    used_ids: list[str] = []
    for card in evidence.get("cards", []):
        if not isinstance(card, dict):
            continue
        for sid in card.get("premise_statement_ids", []):
            sid = str(sid)
            if sid not in used_ids:
                used_ids.append(sid)

    labels = {
        sid: f"E{index}"
        for index, sid in enumerate(used_ids, start=1)
    }

    evidence_catalog = {}
    for sid in used_ids:
        row = statements.get(sid, {})
        evidence_catalog[labels[sid]] = {
            "text": row.get("text"),
            "epistemic_role": row.get("epistemic_role"),
            "claim_kind": row.get("claim_kind"),
            "paper_count": len(row.get("paper_ids") or []),
        }

    hypotheses = []
    for index, card in enumerate(evidence.get("cards", []), start=1):
        if not isinstance(card, dict):
            continue
        hypotheses.append(
            {
                "label": f"H{index}",
                "title": card.get("title"),
                "positive_premises": [
                    labels[str(sid)]
                    for sid in card.get("premise_statement_ids", [])
                    if str(sid) in labels
                ],
            }
        )

    return (
        {
            "evidence_catalog": evidence_catalog,
            "hypotheses": hypotheses,
        },
        labels,
    )


def expected_disposition(recommendation: str) -> str:
    if recommendation == "REDUNDANCY_REVIEW":
        return "REDUNDANCY_REVIEW_WARRANTED"
    if recommendation == "NO_ACTION":
        return "NO_MATERIAL_REDUNDANCY_CONCERN"
    return "UNMAPPED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S244 blind portfolio evidence-diversity validity audit. "
            "Reviewers see hypothesis titles and premise texts but never "
            "automatic overlap metrics or recommendations."
        )
    )
    p.add_argument("--collector", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_AGENT_MODEL") or "openai/gpt-5.6-luna",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
    )
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--max-retries", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    collector_path = args.collector.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not collector_path.is_file():
        raise RuntimeError("missing collector: " + str(collector_path))
    if output_path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: " + str(output_path)
        )

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing API key environment variable: {args.api_key_env}"
        )

    collector = load_json(collector_path)

    client = get_client(
        api_key=api_key,
        base_url=args.base_url,
        instructor_mode="tools",
        timeout=120.0,
    )

    rows: list[dict[str, Any]] = []

    print("=== S244 BLIND PORTFOLIO EVIDENCE-DIVERSITY VALIDITY ===")
    print("automatic recommendations shown to reviewer: false")
    print("overlap/Jaccard metrics shown to reviewer: false")
    print("literature retrieval: false")
    print("selection authority: false")

    available_cases = [
        case
        for case in collector.get("cases", [])
        if isinstance(case, dict)
        and (case.get("portfolio_diversity") or {}).get("measurement_status")
        == "AVAILABLE"
    ]

    for index, case in enumerate(available_cases, start=1):
        case_id = str(case.get("source_case_id") or "UNKNOWN")
        diversity = case.get("portfolio_diversity") or {}
        recommendation = str(
            diversity.get("evidence_recommendation") or "MISSING"
        )

        artifact = Path(
            str(diversity["artifact"])
        ).expanduser().resolve()
        run_dir = artifact.parent

        evidence_path = run_dir / "hypothesis_axis_a4.evidence_diversity.json"
        context_path = run_dir / "hypothesis.context.json"

        if not evidence_path.is_file():
            raise RuntimeError(
                "missing evidence-diversity report: " + str(evidence_path)
            )
        if not context_path.is_file():
            raise RuntimeError(
                "missing hypothesis context: " + str(context_path)
            )

        evidence = load_json(evidence_path)
        context = load_json(context_path)

        blind_payload, _labels = build_blind_payload(
            evidence=evidence,
            context=context,
        )

        user = json.dumps(
            blind_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        print(
            f"[{index}/{len(available_cases)}] {case_id}"
        )

        result, _event = run_instructor_structured_call(
            client.chat.completions,
            model=args.model,
            response_model=EvidencePortfolioBlindReview,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_retries=args.max_retries,
            telemetry_context={
                "pipeline": "portfolio_evidence_diversity_validation",
                "stage": "blind_portfolio_evidence_review",
                "call_kind": "structured",
                "source_case_id": case_id,
            },
        )

        if not isinstance(result, EvidencePortfolioBlindReview):
            result = EvidencePortfolioBlindReview.model_validate(result)

        expected = expected_disposition(recommendation)

        if result.disposition == "INDETERMINATE" or expected == "UNMAPPED":
            agreement = "INDETERMINATE"
        else:
            agreement = (
                "AGREES"
                if result.disposition == expected
                else "DISAGREES"
            )

        rows.append(
            {
                "source_case_id": case_id,
                "case_role": case.get("case_role"),
                "shadow_recommendation": recommendation,
                "expected_blind_disposition": expected,
                "blind_review": result.model_dump(mode="json"),
                "agreement": agreement,
            }
        )

    determinate = [
        row for row in rows
        if row["agreement"] != "INDETERMINATE"
    ]
    agreement_count = sum(
        row["agreement"] == "AGREES"
        for row in determinate
    )

    review_counts = Counter(
        row["blind_review"]["disposition"] for row in rows
    )
    reason_counts = Counter(
        reason
        for row in rows
        for reason in row["blind_review"]["reasons"]
    )

    payload = {
        "schema_version":
            "portfolio-evidence-diversity-blind-validity-s244-v1",
        "source_collector": str(collector_path),
        "case_count": len(rows),
        "determinate_case_count": len(determinate),
        "agreement_count": agreement_count,
        "agreement_fraction": (
            agreement_count / len(determinate)
            if determinate
            else None
        ),
        "blind_disposition_counts":
            dict(sorted(review_counts.items())),
        "blind_reason_counts":
            dict(sorted(reason_counts.items())),
        "rows": rows,
        "interpretation_policy": {
            "reviewer_saw_shadow_recommendation": False,
            "reviewer_saw_overlap_metrics": False,
            "reviewer_saw_unused_evidence": False,
            "novelty_signal_consumed": False,
            "literature_retrieval_performed": False,
            "portfolio_ranking_computed": False,
            "portfolio_selection_changed": False,
            "production_selection_changed": False,
        },
    }

    write_json(output_path, payload)

    print("\nS244 complete")
    print("cases:", len(rows))
    print("determinate:", len(determinate))
    print(
        "shadow/blind agreement:",
        f"{agreement_count}/{len(determinate)}"
        if determinate else "n/a",
        (
            f"({payload['agreement_fraction']:.3f})"
            if payload["agreement_fraction"] is not None
            else ""
        ),
    )
    print("blind dispositions:", dict(sorted(review_counts.items())))
    print("blind reasons:", dict(sorted(reason_counts.items())))
    print("ranking computed: false")
    print("production selection changed: false")
    print("artifact:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
